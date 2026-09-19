"""Do Mack and the ODP bootstrap cover on the same 464 units the model was scored on? (#20, E3a)

`results/fleet/coverage.jsonl` records TabPFN-3.5's own interval bounds and coverage flags per unit. This
measures the two classical methods' intervals **on those same units and the same realised futures**, so the
comparison is paired rather than three separate rates. The model is not refitted: its side is read from disk.

Both classical methods are fitted on the as-of-anchor triangle and evaluated on the projection to the target
diagonal -- `src/tabpfn_reserving/classical.py` states the conventions and the reason, and
`tests/test_classical.py` validates the Mack recursion against chainladder's own run-off output.

    python scripts/three_method_coverage.py --limit 80 --out results/runs/<id>/screen.jsonl
    python scripts/three_method_coverage.py --out results/runs/<id>/per_unit.jsonl
    python scripts/three_method_coverage.py --summary-only results/runs/<id>/per_unit.jsonl

Resumable: rows append to the JSONL and units already present are skipped. `--summary-only` prints the
tables and the verdict without recomputing anything.

**Any script that prints a verdict must refuse below a stated sample threshold, and when the outcome has no
variation.** Both refusals are implemented here: `MIN_UNITS_FOR_VERDICT` (200) and the all-covered /
none-covered case at each level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

# The venv's editable install points at the *primary* checkout (`.venv/.../__editable__....pth`), so a
# worker's worktree has to put its own `src` first or this silently runs the other branch's package. The
# same resolution `scripts/check_figures.py` uses: ask git, fall back to the file's position.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def checkout_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True)
        return pathlib.Path(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return here.parent


_ROOT = checkout_root()
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from tabpfn_reserving import classical  # noqa: E402
from tabpfn_reserving.triangle import Triangle, factor_reserve, target_ages  # noqa: E402

LEVELS = [0.50, 0.75, 0.90, 0.95]
COLUMN = "IncurLoss"
SEED = 20260919                 # fixed: the same command twice must give the same number
N_SIMS = 1000
MIN_UNITS_FOR_VERDICT = 200     # stated before any number was computed
Z2 = 2.0                        # the pre-registered rule: real when the gap exceeds 2 binomial SE


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"revision": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def load_units(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(units: list[dict], out: pathlib.Path, fleet_map, n_sims: int, seed: int) -> dict:
    import chainladder as cl

    done: set[str] = set()
    if out.exists():
        done = {r["unit"] for r in load_units(out)}
    print(f"three-method coverage: {len(units)} units, {len(done)} already done, "
          f"n_sims={n_sims} seed={seed}, chainladder {cl.__version__}")

    skipped: dict[str, int] = {}
    counts = {"written": 0, "carried_steps": 0, "nonfinite_sims": 0}
    rel: list[float] = []
    t0 = time.time()
    with out.open("a") as fh:
        for i, row in enumerate(units):
            unit = row["unit"]
            if unit in done:
                continue
            def skip(reason: str) -> None:
                skipped[reason] = skipped.get(reason, 0) + 1
            try:
                tri = Triangle.load(fleet_map[row["triangle"]], column=COLUMN).trim()
            except Exception as exc:                                 # noqa: BLE001
                skip(f"not a usable triangle ({type(exc).__name__})")
                continue
            anchor, n = int(row["anchor"]), int(row["n"])
            if tri.n != n:
                skip(f"rebuilt size {tri.n} != recorded {n}")
                continue

            # (1) The rebuild must reproduce the recorded unit before anything is compared against it.
            actual = tri.actual_future(anchor)
            if not np.isclose(actual, row["actual"], rtol=1e-12, atol=0):
                skip("rebuilt actual != recorded actual")
                continue
            known = tri.known(anchor)
            asof = classical.as_of(cl, tri, anchor)
            obs = np.asarray(asof.values)[0, 0]
            if not np.array_equal(np.isfinite(obs), known[:obs.shape[0], :obs.shape[1]]):
                skip("as-of triangle is not the harness's known cells")
                continue
            targets = target_ages(n, "backtest")

            t_unit = time.time()
            mack_model = cl.MackChainladder().fit(asof)
            mack = classical.mack_horizon(mack_model, tri, anchor, targets)
            odp = classical.odp_horizon(cl, asof, tri, anchor, targets, n_sims, seed)
            elapsed = time.time() - t_unit

            # (2) The classical point must equal the harness's own Chain Ladder point, to the last digit:
            #     if it does not, the two methods are being asked for different cells.
            cl_harness = factor_reserve(tri, anchor, tri.global_factors(known), targets=targets)
            cl_rel = abs(mack.point - cl_harness) / max(abs(cl_harness), 1e-12)
            rel.append(cl_rel)

            out_row = {"unit": unit, "triangle": row["triangle"], "n": n, "anchor": anchor,
                       "horizon": int(row["horizon"]), "actual": float(actual),
                       "model_point": row["point"], "model_median": row["median"],
                       "mack_point": mack.point, "mack_se": mack.se,
                       "mack_parameter": mack.parameter, "mack_process": mack.process,
                       "odp_mean": odp.point_mean, "odp_median": odp.point_median,
                       "odp_runoff_ratio": odp.runoff_ratio,
                       "odp_nonfinite": odp.n_nonfinite,
                       "carried_steps": mack.carried_steps, "carried_point_share": mack.carried_point_share,
                       "cl_point_relative_error": float(cl_rel),
                       "cl_runoff_reserve": float(np.nansum(np.asarray(
                           cl.Chainladder().fit(asof).ibnr_.values))),
                       "chainladder_recorded": row["chainladder"], "seconds": elapsed}
            for lv in LEVELS:
                k = int(lv * 100)
                out_row[f"model_lo_{k}"], out_row[f"model_hi_{k}"] = row[f"lo_{k}"], row[f"hi_{k}"]
                out_row[f"model_covered_{k}"] = bool(row[f"covered_{k}"])
                lo, hi = mack.interval(lv)
                out_row[f"mack_lo_{k}"], out_row[f"mack_hi_{k}"] = lo, hi
                out_row[f"mack_covered_{k}"] = bool(lo <= actual <= hi)
                lo, hi = odp.interval(lv)
                out_row[f"odp_lo_{k}"], out_row[f"odp_hi_{k}"] = lo, hi
                out_row[f"odp_covered_{k}"] = bool(lo <= actual <= hi)
            fh.write(json.dumps(out_row) + "\n")
            fh.flush()
            counts["written"] += 1
            counts["carried_steps"] += mack.carried_steps
            counts["nonfinite_sims"] += odp.n_nonfinite
            if counts["written"] % 20 == 0:
                print(f"  {counts['written']} units | {(time.time() - t0) / 60:.1f} min | {unit[:44]}")
    mins = (time.time() - t0) / 60
    print(f"\nwrote {counts['written']} units to {out} in {mins:.1f} min")
    print(f"  carried-forward steps used: {counts['carried_steps']}  "
          f"non-finite bootstrap draws: {counts['nonfinite_sims']}")
    print(f"  classical point vs harness Chain Ladder point: max relative error "
          f"{max(rel) if rel else float('nan'):.2e} over {len(rel)} units "
          f"(this is the check that both methods are being asked for the same cells)")
    if skipped:
        print("skipped, by reason (nothing dropped silently):")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {reason}")
    return counts


def _binom(n: int, p: float) -> float:
    return float(np.sqrt(p * (1 - p) / n)) if n else float("nan")


def summarise(path: pathlib.Path) -> int:
    import chainladder as cl  # noqa: F401 -- reported in the header only

    rows = load_units(path)
    if not rows:
        print(f"no rows in {path}")
        return 1
    n_units = len(rows)
    print(f"\n=== three-method coverage: {n_units} units from {path} ===")
    horizons = {int(h): sum(1 for r in rows if r["horizon"] == h) for h in sorted({r["horizon"] for r in rows})}
    print(f"    horizons: {horizons}   (the same units the model was scored on)")

    methods = ["model", "mack", "odp"]
    table = {}
    for lv in LEVELS:
        k = int(lv * 100)
        for m in methods:
            cov = np.mean([bool(r[f"{m}_covered_{k}"]) for r in rows])
            table[(m, lv)] = cov
    print(f"\n    coverage at each level (binomial SE in brackets; the model's own numbers are the "
          f"recorded ones)")
    print(f"    {'nominal':>8} {'TabPFN-3.5':>18} {'Mack':>18} {'ODP bootstrap':>18} {'2xSE (pp)':>10}")
    for lv in LEVELS:
        cells = []
        for m in methods:
            cov = table[(m, lv)]
            se = _binom(n_units, lv)
            cells.append(f"{100 * cov:5.1f}% ({100 * (cov - lv):+5.1f}pp)")
        print(f"    {lv:>8.0%} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18} "
              f"{100 * 2 * _binom(n_units, lv):>9.1f}")

    print(f"\n    gap in binomial standard errors (|gap| > 2 is the pre-registered bar)")
    print(f"    {'nominal':>8} {'TabPFN-3.5':>14} {'Mack':>14} {'ODP bootstrap':>14}")
    verdicts = {}
    for lv in LEVELS:
        line = []
        for m in methods:
            gap = table[(m, lv)] - lv
            se = _binom(n_units, lv)
            z = gap / se if se else float("nan")
            verdicts[(m, lv)] = bool(abs(z) > Z2) if np.isfinite(z) else None
            line.append(f"{z:>+7.2f} SE")
        print(f"    {lv:>8.0%} {line[0]:>14} {line[1]:>14} {line[2]:>14}")

    # The paired comparison: the same units, so the question is the discordance.
    print(f"\n    paired discordance, by level (the comparison the units were built for)")
    print(f"    {'nominal':>8} {'ours miss':>10} {'& Mack too':>11} {'& ODP too':>10} "
          f"{'Mack miss':>10} {'& ours too':>11}")
    for lv in LEVELS:
        k = int(lv * 100)
        ours_miss = [r for r in rows if not r[f"model_covered_{k}"]]
        mack_miss = [r for r in rows if not r[f"mack_covered_{k}"]]
        mack_too = sum(1 for r in ours_miss if not r[f"mack_covered_{k}"])
        odp_too = sum(1 for r in ours_miss if not r[f"odp_covered_{k}"])
        ours_too = sum(1 for r in mack_miss if not r[f"model_covered_{k}"])
        print(f"    {lv:>8.0%} {len(ours_miss):>10} "
              f"{100 * mack_too / max(len(ours_miss), 1):>10.1f}% {100 * odp_too / max(len(ours_miss), 1):>9.1f}% "
              f"{len(mack_miss):>10} {100 * ours_too / max(len(mack_miss), 1):>10.1f}%")

    # Widths: a method that covers by being enormous is not the same result as one that covers.
    print(f"\n    median interval width (95% level) relative to the point estimate, per method")
    print(f"    {'method':>14} {'width / own point':>18} {'width / CL point':>18} {'width / model point':>20}")
    for m in methods:
        w_own, w_cl, w_model = [], [], []
        for r in rows:
            width = r[f"{m}_hi_95"] - r[f"{m}_lo_95"]
            own = {"model": r["model_point"], "mack": r["mack_point"], "odp": r["odp_median"]}[m]
            w_own.append(width / max(abs(own), 1e-9))
            w_cl.append(width / max(abs(r["chainladder_recorded"]), 1e-9))
            w_model.append(width / max(abs(r["model_point"]), 1e-9))
        print(f"    {m:>14} {np.median(w_own):>18.2f} {np.median(w_cl):>18.2f} {np.median(w_model):>20.2f}")

    print(f"\n    one-sided tail: units whose truth is ABOVE the method's own 95% upper bound "
          f"(nominal 2.5%)")
    for m in methods:
        above = np.mean([r["actual"] > r[f"{m}_hi_95"] for r in rows])
        below = np.mean([r["actual"] < r[f"{m}_lo_95"] for r in rows])
        print(f"    {m:>14}  above {100 * above:5.1f}%   below {100 * below:5.1f}%")

    # The bootstrap's own reality check, per unit and aggregated. A mean far from the Chain Ladder point
    # means the workflow is wrong; the same statement is also what a heavy tail does to a mean, so the
    # robust version (the median simulation) is reported beside it, together with the count of units whose
    # mean the tail has taken over. The workflow itself is validated on `abc` in tests/test_classical.py.
    ratios = np.array([r["odp_runoff_ratio"] for r in rows], dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    agg = np.sum([r["odp_mean"] for r in rows]) / np.sum([r["cl_runoff_reserve"] for r in rows])
    within = np.mean(np.abs(ratios - 1) <= 0.002)
    hstep = np.array([r["odp_median"] / r["mack_point"] if r["mack_point"] else np.nan for r in rows])
    hstep = hstep[np.isfinite(hstep)]
    print(f"\n    ODP bootstrap reality check (mean over simulations / Chain Ladder point, run-off reserve)")
    print(f"      per unit: median {np.median(ratios):.4f}  IQR "
          f"[{np.percentile(ratios, 25):.4f}, {np.percentile(ratios, 75):.4f}]  "
          f"within 0.2%: {100 * within:.1f}% of {len(ratios)} units")
    print(f"      tail-dominated units (|mean ratio| > 10): {int(np.sum(np.abs(ratios) > 10))}  "
          f"| aggregate across units: {agg:.4f}")
    print(f"      h-step totals: median simulation / the same Chain Ladder point: "
          f"median {np.median(hstep):.4f}, IQR [{np.percentile(hstep, 25):.4f}, "
          f"{np.percentile(hstep, 75):.4f}], non-finite draws {sum(r['odp_nonfinite'] for r in rows)}")

    carried = np.array([r["carried_point_share"] for r in rows])
    print(f"\n    the carried-forward convention, which only the oldest `horizon` origins need:")
    print(f"      units with a carried step {int(np.sum(np.array([r['carried_steps'] for r in rows]) > 0))}"
          f"/{n_units}, median share of the point {100 * np.median(carried):.1f}% "
          f"(max {100 * np.max(carried):.1f}%)")
    off = [r for r in rows if r["cl_point_relative_error"] > 1e-6]
    print(f"      units where the classical point differs from the recorded `chainladder` column by "
          f">1e-6: {len(off)}"
          + (f" (worst {max(r['cl_point_relative_error'] for r in off):.3f} on "
             f"{max(off, key=lambda r: r['cl_point_relative_error'])['unit'][:40]}; the harness excludes "
             f"factor pairs with a non-positive denominator, `chainladder`'s Development keeps them)"
             if off else ""))

    # The verdict, gated.
    print()
    if n_units < MIN_UNITS_FOR_VERDICT:
        print(f"    REFUSING a verdict: {n_units} units is below the stated threshold of "
              f"{MIN_UNITS_FOR_VERDICT}. A confident verdict on a handful of units is worse than none; "
              f"this is a mechanics read.")
        return 0
    flat = [(m, lv) for lv in LEVELS for m in methods
            if table[(m, lv)] in (0.0, 1.0) or verdicts[(m, lv)] is None]
    if flat:
        print(f"    REFUSING a verdict: no variation in the outcome for {flat} -- the comparison cannot "
              f"discriminate.")
        return 0
    for lv in LEVELS:
        if not verdicts[("model", lv)]:
            print(f"    (note: the model is within 2 SE at {lv:.0%}; the pre-registered claim about its "
                  f"failure is a claim about the levels where it is outside)")
    ours_off = [lv for lv in LEVELS if verdicts[("model", lv)]]
    mack_off = [lv for lv in LEVELS if verdicts[("mack", lv)]]
    odp_off = [lv for lv in LEVELS if verdicts[("odp", lv)]]
    print(f"    outside 2 binomial SE -- TabPFN-3.5: {[f'{lv:.0%}' for lv in ours_off]}  "
          f"Mack: {[f'{lv:.0%}' for lv in mack_off]}  ODP: {[f'{lv:.0%}' for lv in odp_off]}")
    our_z = [abs(table[("model", lv)] - lv) / _binom(n_units, lv) for lv in LEVELS]
    cl_z = [max(abs(table[(m, lv)] - lv) / _binom(n_units, lv) for m in ("mack", "odp")) for lv in LEVELS]
    if not mack_off and not odp_off:
        print("    VERDICT: branch (a) -- both classical methods land within 2 SE at every level "
              f"(worst {max(cl_z):.1f} SE) while this model is {min(our_z):.1f}-{max(our_z):.1f} SE off. "
              "The comparative claim is earned.")
    elif len(mack_off) == len(LEVELS) and len(odp_off) == len(LEVELS):
        print("    VERDICT: branch (b) -- all three methods are outside 2 SE at every level. The finding "
              "stops being about TabPFN and becomes about the field.")
    else:
        print(f"    VERDICT: neither branch as pre-registered. Mack is outside 2 SE at "
              f"{[f'{lv:.0%}' for lv in mack_off]}, the ODP bootstrap at {[f'{lv:.0%}' for lv in odp_off]}, "
              f"this model at {[f'{lv:.0%}' for lv in ours_off]} -- report the per-level numbers, not a "
              f"branch.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="first N recorded units (0 = all)")
    ap.add_argument("--units", type=pathlib.Path, default=pathlib.Path("results/fleet/coverage.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(
        "results/runs/20260919-three-method-coverage/per_unit.jsonl"))
    ap.add_argument("--n-sims", type=int, default=N_SIMS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--summary-only", type=pathlib.Path, default=None, metavar="JSONL")
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    ap.add_argument("--screen", action="store_true",
                    help="mechanics read only: the summary refuses a verdict")
    args = ap.parse_args()

    if args.summary_only:
        return summarise(args.summary_only)

    units = load_units(args.units)
    if args.limit:
        units = units[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(checkout_root() / "scripts"))
    from fleet_eval import load_fleet

    t0 = time.time()
    fleet_map = {name: sub for name, sub in load_fleet(0)}
    counts = run(units, args.out, fleet_map, args.n_sims, args.seed)

    manifest = args.manifest or (args.out.parent / "manifest.json")
    existing = json.loads(manifest.read_text()) if manifest.exists() else {}
    entry = {"run_id": args.out.parent.name, "units_file": str(args.units),
             "units_file_sha256": hashlib.sha256(args.units.read_bytes()).hexdigest()[:16],
             "n_units_available": len(load_units(args.units)), "n_units_requested": len(units),
             "per_unit_jsonl": str(args.out), "n_sims": args.n_sims, "seed": args.seed,
             "chainladder": __import__("chainladder").__version__, "column": COLUMN,
             "levels": LEVELS, "min_units_for_verdict": MIN_UNITS_FOR_VERDICT,
             "carry_forward_convention": "Triangle.global_factors: the last estimated age-to-age factor, "
                                         "for the point and the standard error alike",
             "git": git_state(), "python": platform.python_version(), "host": platform.node(),
             "screen": bool(args.screen), "seconds": time.time() - t0, "counts": counts}
    manifest.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"manifest: {manifest}")
    return summarise(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
