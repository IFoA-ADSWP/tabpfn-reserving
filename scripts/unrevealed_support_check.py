"""Does the bounded target put the truth back inside the range the model was shown? (#24, free arithmetic)

`results/runs/20260919-target-support/` measured this for the **delta** target: the realised target left the
range of training labels the fit was shown on **21.1% of units upward** (5.3% of cells), against a 14.7%
upper-tail miss -- which is what gave the support explanation its mechanism (`results/remodel/FINDINGS.md`
§4). Changing what the model predicts is the intervention, so the same arithmetic is the *pre-run*
measurement, run here for **both targets on the identical units** so the two are paired rather than compared
across datasets.

Two honest qualifications, because the statistic is easy to over-read:

* **The label window is shorter than the scored window, for both targets.** The existing `known_until` clamp
  makes a training label run from each training anchor `k'` to the evaluation diagonal, while a scored cell
  runs from the evaluation diagonal to the origin's last observed age. A share therefore grows with the
  window's length for a structural reason, and so does a factor; the two targets are affected differently by
  it, which is why both are reported here rather than one against the recorded number.
* **This is descriptive.** The pre-registered bar for R2 is the coverage comparison in
  `scripts/unrevealed_target.py`. Nothing here decides anything; it says whether the mechanism moved at all,
  and it costs seconds because it is pure arithmetic on units already on disk.

    python scripts/unrevealed_support_check.py --limit 75

Writes `support.jsonl` to the same run directory as the cells, one row per unit, both targets in each row.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from tabpfn_reserving.triangle import (Triangle, chainladder_factor, direct_predict_rows,
                                       direct_training_rows, target_ages)
from fleet_eval import load_fleet

COVERAGE = pathlib.Path("results/fleet/coverage.jsonl")
OUT = pathlib.Path("results/runs/20260919-bounded-target/support.jsonl")
TARGETS = ("unrevealed", "delta")


def realised_values(tri: Triangle, anchor: int, target: str, tri_n: int) -> np.ndarray:
    """The scored cells' own target values -- the quantity the model is asked to predict, measured."""
    gf = tri.global_factors(tri.known(anchor))
    targets = target_ages(tri_n, "backtest")
    _, origins = direct_predict_rows(tri, anchor, gf, targets)
    out = []
    for a in origins:
        age, tgt = anchor - a, targets[a]
        base, end = tri.values[a, age], tri.values[a, tgt]
        if not (np.isfinite(base) and np.isfinite(end) and base > 0 and end > 0):
            continue
        if target == "unrevealed":
            out.append(1.0 - base / end)
        else:
            cl = chainladder_factor(gf, age, tgt)
            out.append((end / base) / cl if cl > 0 else np.nan)
    return np.asarray(out, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=75)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--column", default="IncurLoss")
    args = ap.parse_args()

    units = [json.loads(l) for l in COVERAGE.read_text().splitlines() if l.strip()]
    units = [u for u in units if str(u["unit"]).startswith(f"{args.column}|")]
    if args.limit:
        units = units[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_name = {name: sub for name, sub in load_fleet(0)}

    print(f"support check: {len(units)} units, both targets, {len(by_name)} triangles loaded", flush=True)
    rows, skipped = [], {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for rec in units:
        name, anchor = rec["triangle"], int(rec["anchor"])
        try:
            tri = Triangle.load(by_name[name], column=args.column).trim()
        except Exception as exc:  # noqa: BLE001
            skip(f"not a usable triangle ({type(exc).__name__})")
            continue
        row: dict = {"unit": rec["unit"], "triangle": name, "anchor": anchor,
                     "covered_95": bool(rec["covered_95"]), "covered_90": bool(rec["covered_90"])}
        ok = True
        for target in TARGETS:
            try:
                _, y = direct_training_rows(tri, list(range(2, max(3, anchor))), target=target,
                                            known_until=anchor)
                if len(y) < 5:
                    skip("too few training rows")
                    ok = False
                    break
                r = realised_values(tri, anchor, target, tri.n)
            except Exception as exc:  # noqa: BLE001 -- one unit must not stop the recount
                skip(f"arithmetic failure ({type(exc).__name__})")
                ok = False
                break
            if r.size == 0:
                skip("no scoreable cells")
                ok = False
                break
            y_max, y_min = float(np.max(y)), float(np.min(y))
            row[f"{target}_train_min"] = y_min
            row[f"{target}_train_max"] = y_max
            row[f"{target}_train_negative"] = int((y < 0).sum())
            row[f"{target}_n_train"] = int(len(y))
            row[f"{target}_n_cells"] = int(r.size)
            row[f"{target}_realised_min"] = float(r.min())
            row[f"{target}_realised_max"] = float(r.max())
            row[f"{target}_unit_above"] = bool(r.max() > y_max)
            row[f"{target}_unit_below"] = bool(r.min() < y_min)
            row[f"{target}_cells_above"] = int((r > y_max).sum())
            row[f"{target}_cells_below"] = int((r < y_min).sum())
        if ok:
            rows.append(row)

    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    if skipped:
        print("skipped, by reason (nothing dropped silently):")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {reason}")
    if not rows:
        print("nothing measured")
        return 1

    n = len(rows)
    print(f"\n=== {n} units, both targets, identical units ===")
    print(f"  {'':26s} {'unrevealed':>12s} {'delta':>12s}")
    for label, key in (("realised ABOVE training max", "unit_above"),
                       ("realised BELOW training min", "unit_below")):
        vals = [100 * np.mean([r[f"{t}_{key}"] for r in rows]) for t in TARGETS]
        print(f"  {label:26s} {vals[0]:>11.1f}% {vals[1]:>11.1f}%")
    cells_above = [100 * np.mean([r[f"{t}_cells_above"] / max(r[f"{t}_n_cells"], 1) for r in rows])
                   for t in TARGETS]
    print(f"  {'scored cells above training max':26s} {cells_above[0]:>11.1f}% {cells_above[1]:>11.1f}%")
    neg = [100 * sum(r[f"{t}_train_negative"] for r in rows)
           / max(sum(r[f"{t}_n_train"] for r in rows), 1) for t in TARGETS]
    print(f"  {'training labels below zero':26s} {neg[0]:>11.1f}% {neg[1]:>11.1f}%")
    shared_above = np.mean([r["unrevealed_unit_above"] and r["delta_unit_above"] for r in rows])
    print(f"  units outside the training range on BOTH targets: {100 * shared_above:.1f}%")
    print(f"\n  Descriptive only -- the pre-registered bar for R2 is the coverage comparison.")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
