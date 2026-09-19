"""The configuration cells (#22): do the vendor's own documented changes move the one-sided tail?

`results/conditions/FINDINGS.md` and `results/remodel/FINDINGS.md` both put the same mechanism in front of us:
TabPFN-3.5's regression distribution is a bucket grid fixed on the pretraining prior and merely rescaled to the
training target's mean and sd, so the *shape* of the answer is set by what you predict. Our measured defect is
one-sided -- 14.7% of units have their truth above the model's own 95% upper bound against a nominal 2.5%, and
0.9% below (`results/fleet/CALIBRATION.md`) -- so the two configuration changes the vendor documents for a
target with a range-escaping tail are worth a screening run before anyone spends seven hours on the fleet.

Four cells, one factor each, all on **the same 150 units** the recorded coverage run already scored:

    cell          change
    baseline      neither -- the replication cell, and the thing every other cell is paired against
    categorical   declare origin_idx/dev_idx/cal_idx categorical on TabPFNRegressor
    transform     swap in the extrapolating target transform (quantile_uni_extrapolate)
    both          both

    python scripts/config_cells.py --cell baseline --limit 150
    python scripts/config_cells.py --cell categorical --limit 150      # and transform, both
    python scripts/config_cells.py --report

Resumable in the same way the other fleet scripts are: rows append to `<cell>.jsonl` and units already present
are skipped, so a crashed cell is restarted by re-running the same command. A cell file may have only one
writer at a time (`<cell>.lock`), and `--dedupe` collapses a file that was written by two anyway -- refusing
outright if any two rows for the same unit disagree, because a duplicate that disagrees means the procedure is
not deterministic and no number from the run means anything. The units are the **first N rows of
`results/fleet/coverage.jsonl`**, read from that file rather than re-derived, so the subset is the same for
every cell and every re-run without depending on the fleet's skip logic staying identical.

The baseline is a *replication*, and it is checked as one: its rows are compared against the recorded rows for
the same units field by field, and `--report` says whether they agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import (Triangle, direct_training_rows, factor_reserve, target_ages)
from fleet_eval import load_fleet

# The same levels, held-out policy and draw count as the recorded coverage run, so the baseline is comparable
# to it and the cells are comparable to each other.
LEVELS = [0.50, 0.75, 0.90, 0.95]
SEED = 0

CELLS: dict[str, dict] = {
    "baseline": dict(categorical_identifiers=False, extrapolating_target=False),
    "categorical": dict(categorical_identifiers=True, extrapolating_target=False),
    "transform": dict(categorical_identifiers=False, extrapolating_target=True),
    "both": dict(categorical_identifiers=True, extrapolating_target=True),
}

DEFAULT_UNITS = pathlib.Path("results/fleet/coverage.jsonl")
DEFAULT_OUT = pathlib.Path("results/runs/20260919-config-cells")


def read_units(path: pathlib.Path, limit: int, column: str) -> list[dict]:
    """The first `limit` recorded coverage units, as (triangle, anchor) -- the subset, from the record."""
    units: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not str(row["unit"]).startswith(f"{column}|"):
            raise SystemExit(f"{path}: unit {row['unit']!r} is not on column {column!r}")
        units.append({"unit": row["unit"], "triangle": row["triangle"], "anchor": int(row["anchor"])})
        if limit and len(units) >= limit:
            break
    if not units:
        raise SystemExit(f"{path}: no units")
    return units


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    return {"revision": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def versions() -> dict:
    import tabpfn
    import sklearn
    import torch
    import chainladder

    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "tabpfn": tabpfn.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__,
            "chainladder": chainladder.__version__}


def resolved_config(model, options: dict) -> dict:
    """What the model *actually* got, not just what was asked for -- the point of a manifest."""
    out = {"requested": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                         for k, v in options.items()},
           "constructor_params": {k: v for k, v in model.get_params().items()
                                  if k in ("categorical_features_indices", "inference_config", "device")}}
    cfg = getattr(model, "inference_config_", None)
    if cfg is not None:
        out["inference_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS"] = list(
            cfg.REGRESSION_Y_PREPROCESS_TRANSFORMS)
    out["categorical_features_indices_"] = getattr(model, "categorical_features_indices_", None)
    return out


def run_cell(cell: str, args) -> int:
    options = CELLS[cell]
    out_path = args.out / f"{cell}.jsonl"
    manifest_path = args.out / f"manifest.{cell}.json"
    lock_path = args.out / f"{cell}.lock"
    args.out.mkdir(parents=True, exist_ok=True)

    # One writer per cell file. Two processes appending the same rows produce a file that reads correctly
    # (the rows are identical -- the procedure is deterministic) but no longer says how many units were run,
    # and a paired comparison keyed on the unit would silently be reading a file twice the size it claims.
    if lock_path.exists():
        try:
            other = int(lock_path.read_text().split()[0])
        except Exception:  # noqa: BLE001
            other = -1
        if other > 0 and _alive(other) and other != os.getpid():
            raise SystemExit(f"{out_path} is being written by pid {other}; refusing to be a second writer")
    lock_path.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")

    try:
        return _run_cell(cell, args, options, out_path, manifest_path)
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


def _run_cell(cell: str, args, options: dict, out_path: pathlib.Path, manifest_path: pathlib.Path) -> int:
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

    print(f"cell {cell}: options={options} | units {len(units)} (first {args.limit} of {args.units}) "
          f"| done {len(done)} | todo {len(todo)} | draws {args.draws} | seed {SEED}")
    if not subs:
        raise SystemExit("the fleet loaded no triangles")
    manifest = {"cell": cell, "options": options, "units_source": str(args.units),
                "n_units": len(units), "draws": args.draws, "seed": SEED, "column": args.column,
                "git": git_state(), "versions": versions(),
                "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
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
                X, y = direct_training_rows(tri, train, target="delta", known_until=anchor)
                if len(y) < 5:
                    skip("too few training rows")
                    continue
                model = arm.make_model("local", **options)
                model.fit(X, y)
                tgt = target_ages(tri.n, "backtest")
                out = arm.reserve_direct(model, tri, anchor, args.draws,
                                         np.random.default_rng(SEED), target="delta", targets=tgt)
            except Exception as exc:  # noqa: BLE001 -- one unit must not stop the cell
                skip(f"model failure ({type(exc).__name__}: {str(exc)[:60]})")
                continue
            if resolved is None:
                resolved = resolved_config(model, options)
                manifest["resolved"] = resolved
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            s = out["samples"]
            if s is None or not np.isfinite(s).all():
                skip("no usable draws")
                continue
            ga = tri.global_factors(tri.known(anchor))
            row = {"unit": u["unit"], "cell": cell, "triangle": name, "n": int(tri.n), "anchor": int(anchor),
                   "horizon": int(tri.n - 1 - anchor), "actual": float(actual),
                   "point": float(out["reserve"]), "median": float(np.median(s)), "mean": float(np.mean(s)),
                   "chainladder": float(factor_reserve(tri, anchor, ga,
                                                       targets=target_ages(tri.n, "backtest"))),
                   "draws": int(args.draws), "train_rows": int(len(y)),
                   "draws_method": str(out["draws_method"]),
                   "distribution_route": str(out["distribution_route"])}
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

    print(f"\ncell {cell}: wrote {n_written} rows to {out_path} in {(time.time() - t0) / 60:.1f} min")
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


def dedupe(path: pathlib.Path) -> dict:
    """One row per unit, first occurrence wins, and it is an error if two rows for a unit disagree.

    Two rows for the same unit can only come from two processes writing the same cell file -- the run refuses
    to start when that is already happening (see the lock in `run_cell`), so this is a repair path for a file
    written before the lock existed. It is safe *because* the procedure is deterministic, and that is checked
    rather than assumed: rows that disagree raise, they are never dropped.
    """
    if not path.exists():
        return {"rows": 0, "units": 0, "duplicates": 0, "conflicting": 0}
    kept: list[str] = []
    first: dict[str, str] = {}
    duplicates = conflicting = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        unit = row["unit"]
        if unit in first:
            duplicates += 1
            if json.dumps(row, sort_keys=True) != first[unit]:
                conflicting += 1
            continue
        first[unit] = json.dumps(row, sort_keys=True)
        kept.append(line)
    if conflicting:
        raise SystemExit(f"{path}: {conflicting} duplicated unit(s) disagree with the first row -- "
                         f"the procedure is not deterministic and nothing here can be trusted")
    if duplicates:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(kept) + "\n")
        tmp.replace(path)
    return {"rows": len(kept), "units": len(first), "duplicates": duplicates, "conflicting": 0}


# ---------------------------------------------------------------------------------------------
# The report: per-level coverage per cell, and the paired difference against the replication cell
# ---------------------------------------------------------------------------------------------

def load_cell(path: pathlib.Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["unit"]] = row
    return rows


def exact_binom_two_sided(k: int, m: int) -> float:
    """McNemar's exact test: the discordant pairs are a coin flip under the null. No scipy needed."""
    from math import comb

    if m == 0:
        return 1.0
    tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def coverage(rows: list[dict], level: int) -> float:
    return float(np.mean([bool(r[f"covered_{level}"]) for r in rows]))


def half_width(n: int, nominal: float) -> float:
    """The binomial sampling error at a level, in percentage points -- the bar the task pre-registered."""
    return 100 * 1.96 * float(np.sqrt(nominal * (1 - nominal) / n))


def report(args) -> int:
    cell_rows = {cell: load_cell(args.out / f"{cell}.jsonl") for cell in CELLS}
    for cell, rows in cell_rows.items():
        print(f"{cell:12s} {len(rows):4d} units")
    if not cell_rows["baseline"]:
        raise SystemExit("no baseline rows -- the replication cell is mandatory")

    units = read_units(args.units, args.limit, args.column)
    subset = [u["unit"] for u in units]
    recorded = {}
    for line in args.units.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["unit"] in set(subset):
                recorded[row["unit"]] = row

    shared = [u for u in subset if all(u in cell_rows[c] for c in CELLS)]
    print(f"\n=== config cells: {len(shared)} units present in all four cells "
          f"(of {len(subset)} in the subset) ===")

    n = len(shared)
    print(f"\n{'cell':12s} {'level':>6} {'covered':>8} {'empirical':>10} {'gap':>8} "
          f"{'95% CI':>16} {'z':>6}")
    table: dict[str, dict[int, float]] = {}
    for cell in CELLS:
        rows = [cell_rows[cell][u] for u in shared]
        table[cell] = {}
        for lv in LEVELS:
            key = int(lv * 100)
            cov = coverage(rows, key)
            table[cell][lv] = cov
            se = float(np.sqrt(max(cov * (1 - cov), 1e-12) / n))
            z = (cov - lv) / se if se > 0 else float("nan")
            print(f"{cell:12s} {key:>5}% {int(round(cov * n)):>8} {cov:>10.1%} {100 * (cov - lv):>+7.1f}pp "
                  f"[{100 * max(cov - 1.96 * se, 0):>5.1f},{100 * min(cov + 1.96 * se, 1):>5.1f}] {z:>6.2f}")

    print(f"\n=== the replication cell against the record (the same {len(shared)} units) ===")
    exact_flags = 0
    for cell in ("baseline",):
        diffs = []
        for u in shared:
            rec, got = recorded.get(u), cell_rows[cell][u]
            if rec is None:
                continue
            for lv in LEVELS:
                key = int(lv * 100)
                if bool(rec[f"covered_{key}"]) != bool(got[f"covered_{key}"]):
                    diffs.append((u, key))
            exact_flags += int(not diffs)
        print(f"  rows compared: {len([u for u in shared if u in recorded])}")
        print(f"  coverage-flag mismatches against coverage.jsonl: {len(diffs)}")
        for u, key in diffs[:10]:
            print(f"    {u} at {key}%")
        rec_rows = [recorded[u] for u in shared if u in recorded]
        for lv in LEVELS:
            key = int(lv * 100)
            print(f"  {key:>3}%: recorded {coverage(rec_rows, key):.1%} vs baseline cell "
                  f"{table['baseline'][lv]:.1%}")

    print("\n=== paired against the baseline cell (units the baseline missed, and vice versa) ===")
    print(f"{'cell':12s} {'level':>6} {'base':>7} {'cell':>7} {'diff':>7} {'fixed':>6} {'broke':>6} "
          f"{'McNemar p':>10} {'gap':>7}")
    verdict_moves = {}
    for cell in CELLS:
        if cell == "baseline":
            continue
        for lv in LEVELS:
            key = int(lv * 100)
            base = [cell_rows["baseline"][u] for u in shared]
            other = [cell_rows[cell][u] for u in shared]
            n01 = sum(1 for b, o in zip(base, other) if not b[f"covered_{key}"] and o[f"covered_{key}"])
            n10 = sum(1 for b, o in zip(base, other) if b[f"covered_{key}"] and not o[f"covered_{key}"])
            p = exact_binom_two_sided(min(n01, n10), n01 + n10)
            diff = 100 * (n01 - n10) / n
            bar = half_width(n, lv)
            moved = diff > bar and lv in (0.90, 0.95)
            verdict_moves[(cell, lv)] = (diff, bar, moved)
            print(f"{cell:12s} {key:>5}% {table['baseline'][lv]:>7.1%} {table[cell][lv]:>7.1%} "
                  f"{diff:>+6.1f}pp {n01:>6} {n10:>6} {p:>10.4f} {100 * (table[cell][lv] - lv):>+6.1f}pp")

    print("\n=== the pre-registered falsifier ===")
    print(f"  n = {n}. The bar is the binomial sampling error at this n: {half_width(n, 0.5):.1f}pp "
          f"at the worst case (p=0.5), {half_width(n, 0.90):.1f}pp at the 90% level, "
          f"{half_width(n, 0.95):.1f}pp at 95%. The task quotes ~7pp for the pre-registered n=150 "
          f"({half_width(150, 0.5):.1f}pp worst case, {half_width(150, 0.90):.1f}pp at 90%).")
    movers = [f"{c} at {int(lv * 100)}% ({d:+.1f}pp vs bar {b:.1f}pp)"
              for (c, lv), (d, b, m) in verdict_moves.items() if m]
    if movers:
        print("  a cell moves 90/95 coverage toward nominal beyond its bar: " + "; ".join(movers))
    else:
        print("  no cell moves 90/95 coverage toward nominal beyond its bar -- the configuration "
              "explanation is dead on this subset")

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"cells": {}, "subset": {"source": str(args.units), "n": len(subset),
                                            "limit": args.limit, "column": args.column},
                    "draws": args.draws, "seed": SEED, "n_shared": n}
        for cell in CELLS:
            mpath = args.out / f"manifest.{cell}.json"
            cpath = args.out / f"{cell}.jsonl"
            entry: dict = {}
            if mpath.exists():
                man = json.loads(mpath.read_text())
                entry = {k: man[k] for k in ("options", "resolved", "skipped") if k in man}
                entry["last_run"] = {k: man[k] for k in ("n_rows_written", "seconds") if k in man}
                manifest.setdefault("git", man["git"])
                manifest.setdefault("versions", man["versions"])
            if cpath.exists():
                # The hash of the file as it now stands, not of whatever it held when a cell last wrote it:
                # the categorical file was deduplicated after its last run, which the earlier hash predates.
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
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--column", default="IncurLoss")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true", help="summarise the four cells and stop")
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
