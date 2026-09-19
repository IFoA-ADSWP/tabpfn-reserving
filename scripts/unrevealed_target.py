"""R2 (#24): the bounded-support target -- predict the share of the ultimate still to emerge.

The measured defect is one-sided: 14.7% of units have their truth above the model's own 95% upper bound
against a nominal 2.5%, while 0.9% fall below (`results/fleet/CALIBRATION.md`). The documented mechanism
(`results/remodel/FINDINGS.md` §1.1) is that TabPFN-3.5's regression distribution is a bucket grid fixed on
the pretraining prior and merely *rescaled at inference to the training target's mean and standard
deviation* -- so **what you predict sets the grid's support**. A multiplicative factor has no upper bound; a
fraction of the ultimate ends at 1 by construction.

Two cells, one factor each, on the same units:

    cell         what the model predicts
    baseline     the development factor relative to Chain Ladder  (the recorded `delta` target)
    unrevealed   the share of the ultimate still to emerge, in [0, 1)

    python scripts/unrevealed_target.py --cell baseline   --limit 150
    python scripts/unrevealed_target.py --cell unrevealed --limit 150
    python scripts/unrevealed_target.py --report

Everything else is held fixed: same units, anchors, features, target age, 300 draws, `default_rng(0)` per
unit, bar-bins route, the direct arm's training anchors and the existing `known_until` clamp, and a bare
`TabPFNRegressor(device="cpu")` -- `arm.make_model("local")` with no options, so this is not a configuration
cell (#22) and the two cannot be confused.

The reserve is rebuilt as `IBNR(origin) = predicted_share x ultimate(origin)` with **the ultimate taken from
Chain Ladder**, deliberately: holding the level on CL isolates the *support* change as the single factor
under test. (An exposure-anchored level is a separate arm, R1, and is not bundled here.)

The baseline cell is a **mandatory replication**: its rows are compared against the recorded rows in
`results/fleet/coverage.jsonl` field by field, on every field the two files share, and `--report` says
whether they agree. If it does not reproduce, the harness changed and no cell is interpretable.

**Pre-registered expectation** (`results/remodel/FINDINGS.md` §R2), written before the run: coverage at 90%
and 95% moves toward nominal (from 78.0% and 84.7%) -- because the target's support now ends where the
quantity ends -- **before any widening is applied**, by more than the binomial sampling error at the subset
size. **Falsifier:** *if 95% coverage stays within 2 binomial SE of the recorded 84.7%, the support
explanation of the one-sided tail is dead* -- reported plainly, with the paired unit-by-unit counts, and the
remaining candidates are the row-count and cross-triangle families.

Everything is on the **first N rows of `results/fleet/coverage.jsonl`**, read from that file rather than
re-derived, so both cells and every re-run score identical units and the paired test is paired. N is fixed
from a measured rate *before* any coverage number is compared, and the report states both N and the binomial
sampling error at it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import (Triangle, direct_training_rows, factor_reserve, target_ages)
from fleet_eval import load_fleet

from config_cells import (dedupe, exact_binom_two_sided, git_state, half_width, load_cell, read_units,
                          resolved_config, versions)

LEVELS = [0.50, 0.75, 0.90, 0.95]
SEED = 0

# cell -> the target the model is asked to predict. Everything else is identical between them.
CELLS: dict[str, str] = {"baseline": "delta", "unrevealed": "unrevealed"}

DEFAULT_UNITS = pathlib.Path("results/fleet/coverage.jsonl")
DEFAULT_OUT = pathlib.Path("results/runs/20260919-bounded-target")

# The recorded fleet-wide 95% coverage and 90% coverage the pre-registration quotes (`results/fleet/COVERAGE.md`).
RECORDED_95 = 84.7 / 100
RECORDED_90 = 78.0 / 100

# A verdict on a handful of units is worse than no verdict. Below this, the report prints the measurements
# and refuses the verdict; the number is stated here rather than chosen after seeing an outcome.
MIN_UNITS = 50



def run_cell(cell: str, args) -> int:
    target = CELLS[cell]
    out_path = args.out / f"{cell}.jsonl"
    manifest_path = args.out / f"manifest.{cell}.json"
    lock_path = args.out / f"{cell}.lock"
    args.out.mkdir(parents=True, exist_ok=True)

    # One writer per cell file (see `config_cells.run_cell` for the full argument): two processes appending
    # the same rows produce a file that still reads correctly -- the rows are identical, the procedure is
    # deterministic -- but no longer says how many units were run.
    if lock_path.exists():
        try:
            other = int(lock_path.read_text().split()[0])
        except Exception:  # noqa: BLE001
            other = -1
        if other > 0 and _alive(other) and other != os.getpid():
            raise SystemExit(f"{out_path} is being written by pid {other}; refusing to be a second writer")
    lock_path.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    try:
        return _run_cell(cell, target, args, out_path, manifest_path)
    finally:
        try:
            if lock_path.exists() and lock_path.read_text().split()[0] == str(os.getpid()):
                lock_path.unlink()
        except OSError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run_cell(cell: str, target: str, args, out_path: pathlib.Path, manifest_path: pathlib.Path) -> int:
    units = read_units(args.units, args.limit, args.column)
    subs = dict(load_fleet(0))
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["unit"])
            except Exception:  # noqa: BLE001
                pass
    todo = [u for u in units if u["unit"] not in done]

    print(f"cell {cell}: target={target!r} | units {len(units)} (first {args.limit} of {args.units}) "
          f"| done {len(done)} | todo {len(todo)} | draws {args.draws} | seed {SEED}", flush=True)
    if not subs:
        raise SystemExit("the fleet loaded no triangles")
    manifest = {"cell": cell, "target": target, "units_source": str(args.units), "n_units": len(units),
                "draws": args.draws, "seed": SEED, "column": args.column, "git": git_state(),
                "versions": versions(), "command": " ".join(sys.argv),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    t0 = time.time()
    skipped: dict[str, int] = {}
    n_written = 0
    resolved: dict | None = None

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with out_path.open("a") as fh:
        for i, u in enumerate(todo):
            name, anchor = u["triangle"], u["anchor"]
            sub = subs.get(name)
            if sub is None:
                skip("triangle not in the fleet")
                continue
            try:
                tri = Triangle.load(sub, column=args.column).trim()
            except Exception as exc:  # noqa: BLE001
                skip(f"not a usable triangle ({type(exc).__name__})")
                continue
            actual = tri.actual_future(anchor)
            if not np.isfinite(actual) or actual <= 0:
                skip("no scoreable future")
                continue
            train = list(range(2, max(3, anchor)))
            try:
                X, y = direct_training_rows(tri, train, target=target, known_until=anchor)
                if len(y) < 5:
                    skip("too few training rows")
                    continue
                model = arm.make_model("local")
                model.fit(X, y)
                tgt = target_ages(tri.n, "backtest")
                out = arm.reserve_direct(model, tri, anchor, args.draws,
                                         np.random.default_rng(SEED), target=target, targets=tgt)
            except Exception as exc:  # noqa: BLE001 -- one unit must not stop the cell
                skip(f"model failure ({type(exc).__name__}: {str(exc)[:60]})")
                continue
            if resolved is None:
                resolved = {"requested": {"target": target},
                            "constructor": "arm.make_model('local') with no options"}
                manifest["resolved"] = resolved
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            s = out["samples"]
            if s is None or not np.isfinite(s).all():
                skip("no usable draws")
                continue
            ga = tri.global_factors(tri.known(anchor))
            row = {"unit": u["unit"], "cell": cell, "target": target, "triangle": name,
                   "n": int(tri.n), "anchor": int(anchor), "horizon": int(tri.n - 1 - anchor),
                   "actual": float(actual), "point": float(out["reserve"]),
                   "median": float(np.median(s)), "mean": float(np.mean(s)),
                   "chainladder": float(factor_reserve(tri, anchor, ga,
                                                       targets=target_ages(tri.n, "backtest"))),
                   "draws": int(args.draws), "train_rows": int(len(y)),
                   "train_label_min": float(np.min(y)), "train_label_max": float(np.max(y)),
                   "train_labels_negative": int((y < 0).sum()),
                   "draws_method": str(out["draws_method"]),
                   "distribution_route": str(out["distribution_route"]),
                   "shares_over_one": out.get("shares_over_one")}
            for lv in LEVELS:
                lo, hi = np.quantile(s, [(1 - lv) / 2, 1 - (1 - lv) / 2])
                row[f"lo_{int(lv * 100)}"], row[f"hi_{int(lv * 100)}"] = float(lo), float(hi)
                row[f"covered_{int(lv * 100)}"] = bool(lo <= actual <= hi)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_written += 1
            if n_written % 10 == 0:
                mins = (time.time() - t0) / 60
                rate = (time.time() - t0) / max(n_written, 1)
                print(f"  {n_written}/{len(todo)} | {mins:.1f} min | {rate:.1f} s/unit | "
                      f"eta {rate * (len(todo) - n_written) / 60:.0f} min | {name[:34]} k={anchor}",
                      flush=True)

    print(f"\ncell {cell}: wrote {n_written} rows to {out_path} in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    if skipped:
        print("skipped, by reason (nothing dropped silently):")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {reason}")
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["n_rows_written"] = n_written
    manifest["skipped"] = skipped
    manifest["seconds"] = round(time.time() - t0, 1)
    manifest["sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest() if out_path.exists() else None
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------

def same(a, b) -> bool:
    """Field equality at the tolerance the other replications here use: 1e-9 relative on numbers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
    return a == b


def coverage(rows: list[dict], level: int) -> float:
    return float(np.mean([bool(r[f"covered_{level}"]) for r in rows]))


def report(args) -> int:
    cell_rows = {cell: load_cell(args.out / f"{cell}.jsonl") for cell in CELLS}
    for cell, rows in cell_rows.items():
        print(f"{cell:12s} target={CELLS[cell]:12s} {len(rows):4d} rows")
    missing = [c for c, r in cell_rows.items() if not r]
    if missing:
        raise SystemExit(f"no rows for {missing} -- the replication cell is mandatory, so nothing here is "
                         f"interpretable without it")

    subset = read_units(args.units, args.limit, args.column)
    subset_units = [u["unit"] for u in subset]
    recorded = {}
    for line in args.units.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recorded[r["unit"]] = r

    shared = [u for u in subset_units if all(u in cell_rows[c] for c in CELLS)]
    n = len(shared)
    print(f"\n=== {n} units present in both cells (subset: first {len(subset_units)} of {args.units}) ===")

    print("\n--- the replication cell, row by row against the record ---")
    compared = mismatched = 0
    flag_diffs: list[tuple[str, int]] = []
    field_diffs: list[tuple[str, str]] = []
    for u in shared:
        rec = recorded.get(u)
        if rec is None:
            continue
        got = cell_rows["baseline"][u]
        compared += 1
        common = sorted(set(rec) & set(got))
        bad = [k for k in common if not same(rec[k], got[k])]
        if bad:
            mismatched += 1
            field_diffs.extend((u, k) for k in bad)
        for lv in LEVELS:
            key = int(lv * 100)
            if bool(rec[f"covered_{key}"]) != bool(got[f"covered_{key}"]):
                flag_diffs.append((u, key))
    print(f"  rows compared against results/fleet/coverage.jsonl : {compared}")
    print(f"  fields compared (every field the two files share)  : {len(set(recorded[shared[0]]) & set(cell_rows['baseline'][shared[0]]))}")
    print(f"  rows with any field outside 1e-9 relative          : {mismatched}")
    print(f"  coverage-flag mismatches at 50/75/90/95            : {len(flag_diffs)}")
    for u, k in field_diffs[:10]:
        print(f"    field {k} differs on {u}")
    rec_rows = [recorded[u] for u in shared if u in recorded]
    for lv in LEVELS:
        key = int(lv * 100)
        print(f"  {key:>3}%: recorded {coverage(rec_rows, key):6.1%}   baseline cell "
              f"{coverage([cell_rows['baseline'][u] for u in shared], key):6.1%}")

    # A second, independent replication: the #22 run's own baseline cell, already verified against the same
    # record on the same first 75 units. Two processes, one number -- the "same command twice" check.
    prior = pathlib.Path("results/runs/20260919-config-cells/baseline.jsonl")
    if prior.exists():
        prior_rows = load_cell(prior)
        common = [u for u in shared if u in prior_rows]
        bad = 0
        for u in common:
            a, b = prior_rows[u], cell_rows["baseline"][u]
            if any(not same(a[k], b[k]) for k in sorted(set(a) & set(b))):
                bad += 1
        print(f"  against #22's recorded baseline cell ({prior}): {len(common)} units shared, "
              f"{bad} with any field outside 1e-9 relative")

    rows_base = [cell_rows["baseline"][u] for u in shared]
    rows_new = [cell_rows["unrevealed"][u] for u in shared]

    print(f"\n--- coverage (n={n}; binomial SE at the nominal level in brackets) ---")
    print(f"{'cell':12s} {'level':>6} {'covered':>8} {'empirical':>10} {'gap':>8} {'95% CI':>16} {'z':>6}")
    table: dict[str, dict[int, float]] = {}
    for cell in CELLS:
        rows = [cell_rows[cell][u] for u in shared]
        table[cell] = {}
        for lv in LEVELS:
            key = int(lv * 100)
            cov = coverage(rows, key)
            table[cell][key] = cov
            se = float(np.sqrt(max(cov * (1 - cov), 1e-12) / n))
            z = (cov - lv) / se if se > 0 else float("nan")
            print(f"{cell:12s} {key:>5}% {int(round(cov * n)):>8} {cov:>10.1%} {100 * (cov - lv):>+7.1f}pp "
                  f"[{100 * max(cov - 1.96 * se, 0):>5.1f},{100 * min(cov + 1.96 * se, 1):>5.1f}] {z:>6.2f}")

    print("\n--- paired against the replication cell, unit by unit (McNemar's exact test) ---")
    print(f"{'level':>6} {'baseline':>9} {'unrevealed':>11} {'difference':>11} {'fixed':>6} {'broke':>6} "
          f"{'p':>8} {'bar':>6}")
    paired: dict[int, tuple[int, int, float]] = {}
    for lv in LEVELS:
        key = int(lv * 100)
        n01 = sum(1 for b, o in zip(rows_base, rows_new) if not b[f"covered_{key}"] and o[f"covered_{key}"])
        n10 = sum(1 for b, o in zip(rows_base, rows_new) if b[f"covered_{key}"] and not o[f"covered_{key}"])
        p = exact_binom_two_sided(min(n01, n10), n01 + n10)
        paired[key] = (n01, n10, p)
        print(f"{key:>5}% {table['baseline'][key]:>9.1%} {table['unrevealed'][key]:>11.1%} "
              f"{100 * (n01 - n10) / n:>+10.1f}pp {n01:>6} {n10:>6} {p:>8.4f} {half_width(n, lv):>5.1f}pp")

    print("\n--- the one-sided tail, read directly (nominal 2.5% each side at 95%) ---")
    for cell, rows in (("baseline", rows_base), ("unrevealed", rows_new)):
        for key in (90, 95):
            above = sum(1 for r in rows if r["actual"] > r[f"hi_{key}"])
            below = sum(1 for r in rows if r["actual"] < r[f"lo_{key}"])
            print(f"  {cell:12s} {key}%: above the upper bound {above:4d}/{n} ({100 * above / n:5.1f}%), "
                  f"below the lower bound {below:4d}/{n} ({100 * below / n:4.1f}%)")
    ab_b = np.mean([r["actual"] > r["hi_95"] for r in rows_base])
    ab_n = np.mean([r["actual"] > r["hi_95"] for r in rows_new])
    print(f"  the upper-tail miss: baseline {100 * ab_b:.1f}% -> unrevealed {100 * ab_n:.1f}% "
          f"({100 * (ab_n - ab_b):+.1f}pp; a larger number is worse)")

    print("\n--- the centre, which the pre-registration says should not move much ---")
    for cell, rows in (("baseline", rows_base), ("unrevealed", rows_new)):
        err = np.array([(r["point"] - r["actual"]) / r["actual"] for r in rows])
        med = np.array([(r["median"] - r["actual"]) / r["actual"] for r in rows])
        neg = np.mean([r["point"] < 0 for r in rows])
        print(f"  {cell:12s} median (point-actual)/actual {np.median(err):+8.2f}   "
              f"median (median-actual)/actual {np.median(med):+8.2f}   "
              f"median |error| {np.median(np.abs(err)):6.2f}   point below zero {100 * neg:5.1f}% of units")

    over = [float(r["shares_over_one"]) for r in rows_new if r.get("shares_over_one") is not None]
    lab_max = np.array([r["train_label_max"] for r in rows_new])
    lab_neg = sum(r["train_labels_negative"] for r in rows_new)
    lab_rows = sum(r["train_rows"] for r in rows_new)
    print("\n--- the unrevealed cell's own support ---")
    print(f"  training labels: max over units, median {np.median(lab_max):.3f}, largest {lab_max.max():.3f}"
          f" (a share has to end below 1)")
    print(f"  training labels below zero: {lab_neg}/{lab_rows} ({100 * lab_neg / max(lab_rows, 1):.2f}%) from "
          f"falling cumulative cells -- left as they are, not clipped")
    print(f"  drawn shares above 1: mean {100 * np.mean(over):.2f}% of draws across {len(over)} units "
          f"(max unit {100 * max(over):.1f}%) -- the model's grid is not bounded by the labels' range")

    print("\n=== the pre-registered falsifier ===")
    if n < MIN_UNITS:
        print(f"  REFUSING A VERDICT: n={n} is below the stated threshold of {MIN_UNITS}. The tables above "
              f"are measurements; they are not a decision.")
        return 2
    key = 95
    if paired[key][0] + paired[key][1] == 0:
        print(f"  REFUSING A VERDICT: the two cells agree on every one of the {n} units at 95%, so there is "
              f"no variation for the paired test to work with.")
        return 2

    se_rec = float(np.sqrt(RECORDED_95 * (1 - RECORDED_95) / n))
    se_90 = float(np.sqrt(RECORDED_90 * (1 - RECORDED_90) / n))
    cov95, cov90 = table["unrevealed"][95], table["unrevealed"][90]
    print(f"  n = {n}. Binomial SE against the recorded rates: {100 * se_rec:.1f}pp at 95% "
          f"({100 * RECORDED_95:.1f}% recorded), {100 * se_90:.1f}pp at 90% ({100 * RECORDED_90:.1f}% recorded).")
    print(f"  expectation bar (1 SE): 95% must exceed {100 * (RECORDED_95 + se_rec):.1f}%, "
          f"90% must exceed {100 * (RECORDED_90 + se_90):.1f}%")
    print(f"  falsifier bar (2 SE): an outcome inside [{100 * (RECORDED_95 - 2 * se_rec):.1f}%, "
          f"{100 * (RECORDED_95 + 2 * se_rec):.1f}%] at 95% kills the support explanation")
    print(f"  unrevealed cell: {100 * cov95:.1f}% at 95% ({100 * (cov95 - RECORDED_95):+.1f}pp), "
          f"{100 * cov90:.1f}% at 90% ({100 * (cov90 - RECORDED_90):+.1f}pp)")
    neg_new = 100 * float(np.mean([r["point"] < 0 for r in rows_new]))
    neg_base = 100 * float(np.mean([r["point"] < 0 for r in rows_base]))
    print(f"  reading caveat: the two cells do not only differ in support. The reconstructed level is below "
          f"zero on {neg_new:.1f}% of the unrevealed units against {neg_base:.1f}% of the baseline's, so any "
          f"coverage difference here is a difference of *level and support together*, not of support alone.")

    moved95 = cov95 - RECORDED_95 > se_rec
    moved90 = cov90 - RECORDED_90 > se_90
    n01, n10, p = paired[key]
    if cov95 <= RECORDED_95 + 2 * se_rec and cov95 >= RECORDED_95 - 2 * se_rec:
        print("\n  FALSIFIER MET: 95% coverage stays within 2 binomial SE of the recorded 84.7%. The support")
        print("  explanation of the one-sided tail is dead on this subset (read with the reading caveat above).")
        print("  Paired counts at 95%:")
        print(f"  fixed {n01}, broke {n10} (McNemar p = {p:.4f}); at 90% fixed {paired[90][0]}, "
              f"broke {paired[90][1]} (p = {paired[90][2]:.4f}).")
        return 0
    if moved95:
        print("\n  FALSIFIER NOT MET, and the pre-registered expectation is met at 95%: coverage moved")
        print(f"  toward nominal by more than the {100 * se_rec:.1f}pp sampling error at this n.")
    elif moved90:
        print("\n  FALSIFIER NOT MET: 95% is outside 2 SE of the record, but the 90% movement does not clear")
        print("  its own sampling error. Partial, and reported as partial.")
    else:
        print("\n  FALSIFIER NOT MET, expectation NOT met: 95% lies outside 2 SE of the record but not on the")
        print("  side the pre-registration predicted (or not by more than the sampling error). Reported as it is.")
    print(f"  paired at 95%: fixed {n01}, broke {n10}, McNemar exact p = {p:.4f};  90%: fixed {paired[90][0]}, "
          f"broke {paired[90][1]}, p = {paired[90][2]:.4f}.")
    if args.report_out:
        manifest = {"cells": {}, "n_shared": n, "subset": {"source": str(args.units),
                                                           "n": len(subset_units), "limit": args.limit},
                    "draws": args.draws, "seed": SEED, "mins_units_for_a_verdict": MIN_UNITS,
                    "coverage": {c: {str(k): table[c][k] for k in table[c]} for c in CELLS},
                    "paired_95": {"fixed": n01, "broke": n10, "p": p},
                    "replication": {"rows_compared": compared, "field_mismatches": mismatched,
                                    "coverage_flag_mismatches": len(flag_diffs)}}
        for cell in CELLS:
            mpath = args.out / f"manifest.{cell}.json"
            cpath = args.out / f"{cell}.jsonl"
            entry: dict = {}
            if mpath.exists():
                man = json.loads(mpath.read_text())
                entry = {k: man[k] for k in ("target", "resolved", "skipped") if k in man}
                entry["last_run"] = {k: man[k] for k in ("n_rows_written", "seconds") if k in man}
                manifest.setdefault("git", man.get("git"))
                manifest.setdefault("versions", man.get("versions"))
            if cpath.exists():
                entry["rows_in_file"] = len([l for l in cpath.read_text().splitlines() if l.strip()])
                entry["sha256"] = hashlib.sha256(cpath.read_bytes()).hexdigest()
            manifest["cells"][cell] = entry
        args.report_out.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nwrote manifest to {args.report_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", choices=sorted(CELLS), help="which cell to run")
    ap.add_argument("--units", type=pathlib.Path, default=DEFAULT_UNITS,
                    help="the recorded coverage run whose first N units form the subset")
    ap.add_argument("--limit", type=int, default=150,
                    help="subset size, fixed from a measured rate before any coverage number is compared")
    ap.add_argument("--column", default="IncurLoss")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true", help="summarise the cells and stop")
    ap.add_argument("--report-out", type=pathlib.Path, default=None,
                    help="write the merged manifest here (with --report)")
    ap.add_argument("--dedupe", action="store_true",
                    help="collapse duplicate units in the cell files, refusing if any two disagree")
    args = ap.parse_args()
    if args.dedupe:
        for cell in CELLS:
            stats = dedupe(args.out / f"{cell}.jsonl")
            print(f"{cell:12s} {stats['rows']:4d} rows  {stats['units']:4d} units  "
                  f"{stats['duplicates']:3d} duplicated rows removed  {stats['conflicting']} conflicting")
        return 0
    if args.report:
        return report(args)
    if not args.cell:
        ap.error("--cell is required unless --report is given")
    return run_cell(args.cell, args)


if __name__ == "__main__":
    raise SystemExit(main())
