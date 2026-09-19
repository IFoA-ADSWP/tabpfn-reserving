"""Row expansion: every intermediate target age, one factor per cell (#25).

Each fleet fit currently sees **6-33 training rows, median 25** (`results/conditions/rows_per_fit.log`), one row
per `(anchor, origin)` targeting the run's ceiling age. Every benchmark the vendor's claims rest on excludes
rows at that scale, and the proposed scale-up -- borrowing rows from *other* triangles -- confounds two things:
**row count** and **cross-triangle information**. This arm isolates the first, from data already on disk.

**One factor.** Both cells are built and scored inside this script, unit by unit, from the same triangle, the
same anchors, the same fitted model class and the same scoring, so the only thing that differs is the training
row set:

* `baseline` -- `direct_training_rows(..., intermediate_ages=False)`, which is what every recorded number was
  produced with.
* `expanded` -- `direct_training_rows(..., intermediate_ages=True)`: one row per `(anchor, origin, target_age)`
  for every target age inside the observed window, so the fit sees the whole run rather than only its end.

Both cells run for a unit before the next unit starts, so the unit set cannot be tuned after seeing a result.

**The replication cell is mandatory.** Every unit also present in `results/fleet/clrd-IncurLoss.jsonl` is
compared **field by field** (`actual`, `chainladder`, `direct`) against the recorded values; if any row
disagrees the script refuses to interpret either cell, because a harness that changed makes the comparison
meaningless.

**It refuses to print a verdict** below `--min-common` paired units, when the replication failed, or when the
paired outcome has no variation.

    .venv/bin/python scripts/row_expansion.py --limit 60 --out results/runs/<run_id>

Resumable: rows append to `units.jsonl` and units already present are skipped, so a crash at minute forty does
not cost forty minutes. One writer at a time is enforced with a `.lock` carrying the pid.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tabpfn_reserving import arm  # noqa: E402
from tabpfn_reserving.triangle import (Triangle, direct_training_rows, factor_reserve,  # noqa: E402
                                       target_ages)
from fleet_eval import load_fleet  # noqa: E402

RECORDED = ROOT / "results/fleet/clrd-IncurLoss.jsonl"
COLUMN = "IncurLoss"
CELLS = {"baseline": False, "expanded": True}
# Stated before the run: a paired rate below this many units is not a measurement, it is an anecdote.
MIN_COMMON = 40
# A recorded row and a reproduced row are the same number if they agree far inside any sampling error. The
# procedure is deterministic (no draws), so the tolerance is floating-point slack, not statistics.
REPLICATION_RTOL = 1e-9


def cell_reserve(tri: Triangle, anchor: int, intermediate: bool) -> tuple[float | None, int, str]:
    """The direct arm's backtest reserve under one row set. Returns (reserve, n_rows, note)."""
    train = list(range(2, max(3, anchor)))
    X, y = direct_training_rows(tri, train, target="delta", known_until=anchor,
                               intermediate_ages=intermediate)
    if len(y) < 5:
        return None, len(y), "too few training rows"
    model = arm.make_model("local")
    model.fit(X, y)
    out = arm.reserve_direct(model, tri, anchor, 0, np.random.default_rng(0), target="delta",
                             targets=target_ages(tri.n, "backtest"))
    return float(out["reserve"]), len(y), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N usable triangles (0 = all)")
    ap.add_argument("--max-units", type=int, default=0,
                    help="stop after N scored units (0 = all). The screen's unit count is fixed by this "
                         "before any comparison, so its binomial error is a stated quantity")
    ap.add_argument("--held-out", type=int, default=3, help="diagonals held out; sets the horizon")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--min-common", type=int, default=MIN_COMMON)
    ap.add_argument("--summary-only", action="store_true",
                    help="re-derive the summary from an existing units.jsonl, fitting nothing (idempotent, "
                         "so the same command twice gives the same number)")
    args = ap.parse_args()

    if args.summary_only:
        if args.out is None or not (args.out / "units.jsonl").exists():
            print(f"--summary-only needs --out pointing at a run that has units.jsonl (got {args.out})")
            return 1
        rows = [json.loads(l) for l in (args.out / "units.jsonl").read_text().splitlines() if l.strip()]
        return summarise(rows, args.min_common, print)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "_row-expansion"
    out_dir = args.out or (ROOT / "results/runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    units_path = out_dir / "units.jsonl"
    log_path = out_dir / "run.log"
    lock_path = out_dir / "units.lock"

    if lock_path.exists():
        pid = lock_path.read_text().strip()
        try:
            os.kill(int(pid), 0)
            print(f"another writer holds {lock_path} (pid {pid}) -- refusing to append to the same file")
            return 1
        except (ProcessLookupError, ValueError):
            pass
    lock_path.write_text(str(os.getpid()))

    lines: list[str] = []

    def say(line: str) -> None:
        print(line, flush=True)
        lines.append(line)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    say(f"command: scripts/row_expansion.py --limit {args.limit} --max-units {args.max_units} "
        f"--held-out {args.held_out} "
        f"--out {out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir}")
    say(f"cells: baseline (intermediate_ages=False) and expanded (intermediate_ages=True); one factor.")
    say(f"refuses a verdict below {args.min_common} paired units, on a failed replication, or on no variation.")
    say("")

    done: set[str] = set()
    if units_path.exists():
        for line in units_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["unit"])
            except Exception:  # noqa: BLE001
                pass
    say(f"units already on disk: {len(done)}")

    pairs = load_fleet(0)
    t0 = time.time()
    n_new, n_scanned = 0, 0
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    try:
        with units_path.open("a") as fh:
            for name, sub in pairs:
                if args.limit and n_scanned >= args.limit:
                    break
                if args.max_units and n_new >= args.max_units:
                    break
                try:
                    tri = Triangle.load(sub, column=COLUMN)
                except Exception as exc:  # noqa: BLE001
                    skip(f"load ({type(exc).__name__})")
                    continue
                try:
                    tri = tri.trim()
                except ValueError:
                    skip("not a proper triangle (sparse or fragmented container)")
                    continue
                n_scanned += 1
                for anchor in range(tri.n - 2, tri.n - 2 - args.held_out, -1):
                    if args.max_units and n_new >= args.max_units:
                        break
                    unit = f"{COLUMN}|k={anchor}|{name}"
                    if unit in done:
                        continue
                    actual = tri.actual_future(anchor)
                    if not np.isfinite(actual) or actual <= 0:
                        skip("no scoreable future")
                        continue
                    ga = tri.global_factors(tri.known(anchor))
                    row = {"unit": unit, "triangle": name, "column": COLUMN, "anchor": int(anchor),
                           "n": int(tri.n), "horizon": int(tri.n - 1 - anchor),
                           "actual": float(actual),
                           "chainladder": float(factor_reserve(
                               tri, anchor, ga, targets=target_ages(tri.n, "backtest")))}
                    for cell, flag in CELLS.items():
                        try:
                            reserve, n_rows, note = cell_reserve(tri, anchor, flag)
                        except Exception as exc:  # noqa: BLE001 -- one unit must not stop the fleet
                            row[f"{cell}_error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
                            continue
                        row[f"n_rows_{cell}"] = int(n_rows)
                        if reserve is None:
                            row[f"{cell}_note"] = note
                        else:
                            row[cell] = reserve
                    if not any(c in row for c in CELLS):
                        continue
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_new += 1
                    if n_new % 10 == 0:
                        say(f"  {n_new} units | {(time.time() - t0) / 60:.1f} min | {name[:36]} k={anchor}")
    finally:
        lock_path.unlink(missing_ok=True)

    say("")
    if skipped:
        say("skipped, by reason (no unit is dropped silently):")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            say(f"    {count:5d}  {reason}")
    say(f"scanned {n_scanned} triangles, wrote {n_new} units in {(time.time() - t0) / 60:.1f} min")
    say("")

    rows = [json.loads(l) for l in units_path.read_text().splitlines() if l.strip()]
    manifest = {
        "run_id": out_dir.name,
        "command": f".venv/bin/python scripts/row_expansion.py --limit {args.limit} "
                   f"--max-units {args.max_units} --held-out {args.held_out}",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_revision": git_revision(),
        "dirty": git_dirty(),
        "cells": {"baseline": {"intermediate_ages": False}, "expanded": {"intermediate_ages": True}},
        "scoring": "direct arm, delta target, known_until=anchor, backtest targets, no draws",
        "min_common": args.min_common,
        "units": len(rows),
        "model": "arm.make_model('local') = TabPFNRegressor(device='cpu')",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summarise(rows, args.min_common, say)


def git_revision() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact binomial p for the discordant pairs (no SciPy, and no normal approximation)."""
    n = fixed + broken
    if n == 0:
        return float("nan")
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def summarise(rows: list[dict], min_common: int, say) -> int:
    rec = {}
    if RECORDED.exists():
        for line in RECORDED.read_text().splitlines():
            try:
                r = json.loads(line)
                rec[r["unit"]] = r
            except Exception:  # noqa: BLE001
                pass

    say(f"=== {len(rows)} units on disk ===")
    have_both = [r for r in rows if all(c in r for c in CELLS)]
    say(f"  scored by both cells : {len(have_both)}")
    for cell in CELLS:
        say(f"  scored by {cell:8s} : {sum(1 for r in rows if cell in r)}")
    rescued = [r for r in rows if "baseline" not in r and "expanded" in r]
    say(f"  scored ONLY by expanded (the row minimum stopped the baseline): {len(rescued)}")
    lost = [r for r in rows if "baseline" in r and "expanded" not in r]
    say(f"  scored ONLY by baseline : {len(lost)}")

    say("")
    say("=== rows per fit: the recount, from the code path both cells ran ===")
    for cell in CELLS:
        c = np.array([r[f"n_rows_{cell}"] for r in rows if f"n_rows_{cell}" in r], dtype=float)
        if not c.size:
            continue
        say(f"  {cell:8s}: n={int(c.size):4d}  min {int(c.min()):4d}  p25 {int(np.percentile(c, 25)):4d}  "
            f"median {int(np.median(c)):4d}  p75 {int(np.percentile(c, 75)):4d}  max {int(c.max()):5d}")
    if have_both:
        ratio = np.array([r["n_rows_expanded"] / max(r["n_rows_baseline"], 1) for r in have_both])
        say(f"  expansion factor on the paired units: median x{np.median(ratio):.2f}, "
            f"range x{ratio.min():.2f}-x{ratio.max():.2f}")

    # ---- the replication cell, per row rather than per summary -------------------------------
    say("")
    say("=== replication: this script's baseline cell against results/fleet/clrd-IncurLoss.jsonl ===")
    shared = [r for r in have_both if r["unit"] in rec]
    say(f"  units also in the recorded file: {len(shared)}")
    worst = {k: 0.0 for k in ("actual", "chainladder", "direct")}
    mismatched = []
    for r in shared:
        old = rec[r["unit"]]
        for k in ("actual", "chainladder", "direct"):
            denom = max(abs(old[k]), 1e-12)
            d = abs(r["baseline"] - old[k]) / denom if k == "direct" else abs(r[k] - old[k]) / denom
            worst[k] = max(worst[k], d)
            if d > REPLICATION_RTOL:
                mismatched.append((r["unit"], k, old[k], r["baseline"] if k == "direct" else r[k], d))
    for k, v in worst.items():
        say(f"  worst relative difference in {k:12s}: {v:.3e}")
    if mismatched:
        say(f"  MISMATCHES: {len(mismatched)} (first 5)")
        for m in mismatched[:5]:
            say(f"    {m[0]} {m[1]}: recorded {m[2]!r} vs reproduced {m[3]!r} (rel {m[4]:.3e})")
    if not shared:
        say("  no overlap with the recorded file -- the replication cell is EMPTY, so neither cell is")
        say("  interpretable. Stop here.")
        return 3
    if mismatched:
        say("  REPLICATION FAILED: the harness is not the one that produced the recorded numbers, so no")
        say("  cell in this run is interpretable. Verdict refused.")
        return 3
    say("  REPLICATION OK: every shared unit reproduces field by field, so the only difference between")
    say("  the two cells below is the training row set.")

    # ---- the paired measurement ------------------------------------------------------------------
    say("")
    say("=== paired: same units, same anchors, same model, only the row set differs ===")
    n = len(have_both)
    if n < min_common:
        say(f"  only {n} paired units (< {min_common} stated): refusing to print a verdict.")
        return 2

    err = {c: np.array([100 * (r[c] - r["actual"]) / r["actual"] for r in have_both]) for c in CELLS}
    clerr = np.array([100 * (r["chainladder"] - r["actual"]) / r["actual"] for r in have_both])
    closer = {c: np.abs(err[c]) < np.abs(clerr) for c in CELLS}
    say(f"  {'cell':10s} {'median |err|':>13s} {'closer than CL':>16s} {'paired |err| change vs CL (median)':>36s}")
    for c in CELLS:
        paired = np.abs(err[c]) - np.abs(clerr)
        say(f"  {c:10s} {np.median(np.abs(err[c])):12.1f}% "
            f"{100 * closer[c].mean():15.1f}% {np.median(paired):35.1f}pp")
    say(f"  the null (Chain Ladder) on the same units: median |error| {np.median(np.abs(clerr)):.1f}%")
    se = 100 * math.sqrt(closer["baseline"].mean() * (1 - closer["baseline"].mean()) / n)
    say(f"  binomial sampling error of a rate at n={n}: {se:.1f}pp  (the recorded fleet rate is "
        f"38.8% +- 2.3pp at n=464)")

    say("")
    baseline_closer = closer["baseline"]
    expanded_closer = closer["expanded"]
    fixed = int(np.sum(~baseline_closer & expanded_closer))
    broken = int(np.sum(baseline_closer & ~expanded_closer))
    both = int(np.sum(baseline_closer & expanded_closer))
    neither = int(np.sum(~baseline_closer & ~expanded_closer))
    say("  the 2x2 against Chain Ladder (fixed = lost -> won, broken = won -> lost):")
    say(f"    fixed (baseline not closer, expanded closer) : {fixed}")
    say(f"    broken (baseline closer, expanded not closer): {broken}")
    say(f"    both closer: {both}      neither closer: {neither}")
    p = mcnemar_exact(fixed, broken)
    say(f"    McNemar exact, two-sided: p = {p:.3f}" if np.isfinite(p) else
        "    McNemar exact: no discordant pairs, p undefined")
    delta = 100 * (expanded_closer.mean() - baseline_closer.mean())
    say(f"    rate change: {delta:+.1f}pp  (the pre-registered bar is a rise above 38.8% + 2.3pp = 41.1%)")

    say("")
    say("  by horizon (the held-out diagonal count):")
    say(f"    {'horizon':>7s} {'units':>6s} {'baseline':>9s} {'expanded':>9s} {'delta':>8s}")
    for h in sorted({r["horizon"] for r in have_both}):
        m = np.array([r["horizon"] == h for r in have_both])
        say(f"    {h:7d} {int(m.sum()):6d} {100 * baseline_closer[m].mean():8.1f}% "
            f"{100 * expanded_closer[m].mean():8.1f}% {100 * (expanded_closer[m].mean() - baseline_closer[m].mean()):+7.1f}pp")

    say("")
    say("  by triangle size:")
    for s in sorted({r["n"] for r in have_both}):
        m = np.array([r["n"] == s for r in have_both])
        if m.sum() < 5:
            continue
        say(f"    n={s:2d} {int(m.sum()):5d} units  baseline {100 * baseline_closer[m].mean():5.1f}%  "
            f"expanded {100 * expanded_closer[m].mean():5.1f}%")

    # ---- the verdict, and the gates it is behind -------------------------------------------------
    say("")
    if fixed + broken == 0:
        say("  NO VARIATION: every paired unit moved the same way (or not at all), so the paired outcome")
        say("  carries no information about the row set. Verdict refused.")
        return 4
    if n < min_common:
        return 2
    bar = 41.1
    rate = 100 * expanded_closer.mean()
    inside = abs(rate - 38.8) < 2.0 * se_pp(n, 0.388)
    say(f"  VERDICT: expanded {rate:.1f}% vs baseline {100 * baseline_closer.mean():.1f}% on {n} paired units.")
    say(f"  pre-registered expectation was a rise above the 41.1% bar; the screen's own 2-SE band around")
    say(f"  38.8% is +-{2 * se_pp(n, 0.388):.1f}pp, so this screen {'cannot' if inside else 'can'} resolve a")
    say(f"  move of the size the bar asks for at n={n}.")
    say("  (The full-fleet confirmation, if warranted, is a separate run: this is a fixed, pre-declared")
    say("  subset and its binomial error is quoted rather than hidden.)")
    return 0


def se_pp(n: int, p: float) -> float:
    return 100 * math.sqrt(p * (1 - p) / n)


if __name__ == "__main__":
    raise SystemExit(main())
