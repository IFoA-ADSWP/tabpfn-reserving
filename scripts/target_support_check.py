"""Is the truth outside the range the model was shown? (E10 §4 -- the free measurement)

The mechanism E10 verified: TabPFN-3.5's regression distribution is a bucket grid **fixed on the pretraining
prior**, rescaled at inference to the *training target's* mean and standard deviation
(`results/remodel/FINDINGS.md` §1.1, verified in the vendor's docs and in our installed code). If that grid's
support is what limits the upper tail, then the defect should be visible without any model at all: the realised
future target should sometimes lie **outside the range of the training labels the fit was given**.

That makes this pure arithmetic on units already on disk -- no fit, no draws, no tokens.

**Pre-registered expectation, fixed before running (quoted from `results/remodel/FINDINGS.md` §4):** *if the
one-sided tail miss is a support artefact, the realised target exceeds the training-label maximum on a share of
units comparable to, or larger than, the 14.7% tail-miss rate.* **Falsifier:** *if the realised target lies
inside the training-label range on essentially every unit, then the answer is not outside the model's seen
target range at all, the support explanation dies, and R2/R3/R8 lose their stated mechanism -- leaving only the
feature/horizon kind of extrapolation (R5, R6).*

Both outcomes are informative, which is why it is worth spending.

    python scripts/target_support_check.py --limit 20     # time it
    python scripts/target_support_check.py                # all units
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tabpfn_reserving.triangle import (Triangle, direct_predict_rows, direct_training_rows, target_ages)
from fleet_eval import load_fleet

COVERAGE = pathlib.Path("results/fleet/coverage.jsonl")
OUT = pathlib.Path("results/runs/20260919-target-support")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    units = [json.loads(l) for l in COVERAGE.read_text().splitlines() if l.strip()]
    done: set[str] = set()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "support.jsonl"
    if path.exists():
        done = {json.loads(l)["unit"] for l in path.read_text().splitlines() if l.strip()}
    print(f"target-support check: {len(units)} units, {len(done)} already done")

    by_name = {name: sub for name, sub in load_fleet(0)}
    todo = [u for u in units if u["unit"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    n, t0 = 0, time.time()
    with path.open("a") as fh:
        for rec in todo:
            unit, name, anchor = rec["unit"], rec["triangle"], int(rec["anchor"])
            try:
                tri = Triangle.load(by_name[name], column="IncurLoss").trim()
                _, y = direct_training_rows(tri, list(range(2, max(3, anchor))),
                                            target="delta", known_until=anchor)
                if len(y) < 5:
                    continue
                gf = tri.global_factors(tri.known(anchor))
                targets = target_ages(tri.n, "backtest")
                Xq, origins = direct_predict_rows(tri, anchor, gf, targets)
                realised = []
                for a, row in zip(origins, Xq):
                    age, tgt = anchor - a, targets[a]
                    base, end = tri.values[a, age], tri.values[a, tgt]
                    if not (np.isfinite(base) and np.isfinite(end) and base > 0):
                        continue
                    cl = float(np.exp(row[-1])) if np.isfinite(row[-1]) else 1.0
                    if cl > 0:
                        realised.append(float(end / base) / cl)
            except Exception as exc:  # noqa: BLE001 -- one unit must not stop the recount
                print(f"    skipped {unit}: {type(exc).__name__}")
                continue
            if not realised:
                continue
            y_max, y_min = float(np.max(y)), float(np.min(y))
            r = np.asarray(realised, dtype=float)
            row = {"unit": unit, "triangle": name, "anchor": anchor, "n_train": int(len(y)),
                   "n_cells": int(r.size),
                   "train_max": y_max, "train_min": y_min,
                   "realised_max": float(r.max()), "realised_min": float(r.min()),
                   # the two questions: per unit (does ANY scored cell leave the seen range) ...
                   "unit_above": bool(r.max() > y_max), "unit_below": bool(r.min() < y_min),
                   # ... and per cell (how often does the truth leave it)
                   "cells_above": int((r > y_max).sum()), "cells_below": int((r < y_min).sum()),
                   "covered_95": bool(rec["covered_95"]), "covered_90": bool(rec["covered_90"])}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"  {n} units | {(time.time() - t0) / 60:.1f} min | {name[:34]} k={anchor}")

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print("nothing measured yet")
        return 1
    ab = np.mean([r["unit_above"] for r in rows])
    bl = np.mean([r["unit_below"] for r in rows])
    cells = np.mean([r["cells_above"] / max(r["n_cells"], 1) for r in rows])
    tail = np.mean([not r["covered_95"] for r in rows])
    print(f"\n=== {len(rows)} units ===")
    print(f"  realised target ABOVE the training-label maximum : {100*ab:5.1f}% of units  (pre-registered: >= 14.7%)")
    print(f"  realised target BELOW the training-label minimum : {100*bl:5.1f}% of units")
    print(f"  scored cells above the training maximum          : {100*cells:5.1f}% of cells")
    print(f"  recorded 95% tail miss (for comparison)          : {100*tail:5.1f}% of units")
    print("\n  If the first line is comparable to the last, the fixed-grid explanation has support. If it is near"
          "\n  zero, that explanation is dead and R2/R3/R8 lose their stated mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
