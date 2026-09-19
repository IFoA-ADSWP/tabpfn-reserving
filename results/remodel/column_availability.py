"""Read-only recount: which clrd columns are actually populated on the 464 fleet units?

E10's task asks whether the six columns the CAS database carries are untapped information. That
question has a prior question: on the units the fleet already scored, are the other columns even
present, and in the cells a model would need them?

Nothing is fitted, nothing is predicted, and no repository file is touched. This script reads the
same sample, applies the same `regular_block()` window and the same anchors as
`scripts/fleet_eval.py`, and counts finite cells per column.
Output: results/remodel/column_availability.log

    .venv/bin/python results/remodel/column_availability.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from tabpfn_reserving.triangle import Triangle  # noqa: E402

COLUMNS = ["IncurLoss", "CumPaidLoss", "BulkLoss", "EarnedPremDIR", "EarnedPremCeded", "EarnedPremNet"]
ROOT = pathlib.Path(__file__).resolve().parents[2]


def block_of(sample, idx: int):
    """The (start, size) window `trim()` would choose, plus every column on that window.

    `trim()` is computed from one column; the block it returns is the container's triangular
    window, and every other column has to be sliced to the SAME window or the comparison is
    between different cells.
    """
    tri = Triangle.load(sample.iloc[idx], column="IncurLoss")
    block = tri.regular_block()
    if block is None:
        return None
    start, size = block
    v = np.squeeze(np.asarray(sample.iloc[idx].values, dtype=float))
    if v.ndim == 2:
        v = v[None, ...]
    cols = {}
    for ci, name in enumerate(COLUMNS):
        if ci < v.shape[0]:
            cols[name] = v[ci][start:start + size, :size].copy()
    return start, size, cols


def main() -> int:
    import chainladder as cl

    sample = cl.load_sample("clrd")
    units = [json.loads(line)
             for line in (ROOT / "results/fleet/clrd-IncurLoss.jsonl").read_text().splitlines()]

    by_index: dict[int, list[dict]] = {}
    for u in units:
        by_index.setdefault(int(u["triangle"].rsplit("#", 1)[1]), []).append(u)
    print(f"E10 column availability recount -- units: {len(units)}  containers: {len(by_index)}")
    print("read-only: no model fitted, no repository file modified\n")

    # --- 0. What is BulkLoss? The task calls it "bulk/case loss". Whether Incurred = Paid + Bulk is
    #        an identity decides whether a case reserve is NEW information or a reparameterisation of
    #        two columns the fleet already had.
    v_all = np.asarray(sample.values, dtype=float)
    inc = v_all[..., COLUMNS.index("IncurLoss"), :, :]
    paid = v_all[..., COLUMNS.index("CumPaidLoss"), :, :]
    bulk = v_all[..., COLUMNS.index("BulkLoss"), :, :]
    ok = np.isfinite(inc) & np.isfinite(paid) & np.isfinite(bulk)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(inc - (paid + bulk)) / np.maximum(np.abs(inc), 1e-9)
    print("identity check over all clrd cells where all three columns are finite")
    print(f"  cells                      : {int(ok.sum())}")
    print(f"  inc == paid + bulk         : share within 1e-6 relative = {np.mean(rel[ok] < 1e-6):.4f}")
    print(f"  relative residual quantiles: p10={np.nanpercentile(rel[ok], 10):.3f} "
          f"p50={np.nanpercentile(rel[ok], 50):.3f} p90={np.nanpercentile(rel[ok], 90):.3f}")

    print("\n  first usable containers, on the latest diagonal: is bulk = inc - paid?")
    shown = 0
    for idx in range(775):
        if shown >= 5:
            break
        try:
            b = block_of(sample, idx)
        except Exception:  # noqa: BLE001
            continue
        if b is None:
            continue
        _, n, cols = b
        if "CumPaidLoss" not in cols or "BulkLoss" not in cols:
            continue
        line = []
        for a in range(n):
            d = n - 1 - a
            i_, p_, b_ = cols["IncurLoss"][a, d], cols["CumPaidLoss"][a, d], cols["BulkLoss"][a, d]
            if np.isfinite(i_) and np.isfinite(p_) and np.isfinite(b_):
                line.append(f"a{a}: inc={i_:,.0f} paid={p_:,.0f} bulk={b_:,.0f} "
                            f"(inc-paid)={i_ - p_:,.0f}")
        if line:
            print(f"   #{idx}: " + " | ".join(line[:3]))
            shown += 1

    # --- 1. Per unit: are the other columns present in the cells a fit would use, and in the scored future?
    tally = {c: {"known>=5": 0, "future_complete": 0, "future_any": 0} for c in COLUMNS}
    n_units = 0
    for idx, us in sorted(by_index.items()):
        try:
            b = block_of(sample, idx)
        except Exception:  # noqa: BLE001
            continue
        if b is None:
            continue
        _, n, cols = b
        known_tri = Triangle(name="x", values=cols["IncurLoss"])
        for u in us:
            anchor = int(u["anchor"])
            n_units += 1
            known = known_tri.known(anchor)
            for name in COLUMNS:
                if name not in cols:
                    continue
                vals = cols[name]
                fin = np.isfinite(vals)
                if int((known & fin).sum()) >= 5:
                    tally[name]["known>=5"] += 1
                need = [(a, d) for a in range(anchor + 1) for d in range(anchor - a + 1, n - a)]
                got = [bool(np.isfinite(vals[a, d])) for a, d in need]
                if need and all(got):
                    tally[name]["future_complete"] += 1
                if any(got):
                    tally[name]["future_any"] += 1

    print(f"\nover the {n_units} fleet units (same window, same anchors as scripts/fleet_eval.py):")
    print(f"  {'column':16s} {'known>=5 cells':>15s} {'future fully observed':>23s} {'future partly':>14s}")
    for name in COLUMNS:
        t = tally[name]
        print(f"  {name:16s} {t['known>=5']:>15d} {t['future_complete']:>23d} {t['future_any']:>14d}")

    # --- 2. How big is a case/bulk reserve relative to incurred, on the anchor diagonal?
    shares = []
    n_containers = 0
    for idx, us in sorted(by_index.items()):
        try:
            b = block_of(sample, idx)
        except Exception:  # noqa: BLE001
            continue
        if b is None:
            continue
        _, n, cols = b
        if "BulkLoss" not in cols:
            continue
        n_containers += 1
        il, bl = cols["IncurLoss"], cols["BulkLoss"]
        for u in us:
            anchor = int(u["anchor"])
            for a in range(anchor + 1):
                d = anchor - a
                if np.isfinite(il[a, d]) and il[a, d] != 0 and np.isfinite(bl[a, d]):
                    shares.append(bl[a, d] / il[a, d])
    shares = np.asarray(shares)
    print(f"\nBulkLoss / IncurLoss on the anchor diagonal, {len(shares)} origin-cells "
          f"across {n_containers} containers")
    if len(shares):
        q = np.percentile(shares, [10, 25, 50, 75, 90])
        print(f"  p10={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} p75={q[3]:.3f} p90={q[4]:.3f}"
              f"  share>0.5: {np.mean(shares > 0.5):.3f}")

    # --- 3. Premium as exposure: present at all, and non-zero?
    prem_true = prem_zero = prem_missing = 0
    vals: list[float] = []
    for idx, us in sorted(by_index.items()):
        try:
            b = block_of(sample, idx)
        except Exception:  # noqa: BLE001
            continue
        if b is None:
            continue
        _, _, cols = b
        p = cols.get("EarnedPremNet")
        if p is None:
            prem_missing += len(us)
            continue
        for u in us:
            anchor = int(u["anchor"])
            v = p[:anchor + 1, 0]
            fin = v[np.isfinite(v)]
            if len(fin) == 0:
                prem_missing += 1
            elif np.all(fin == 0):
                prem_zero += 1
            else:
                prem_true += 1
                vals.extend(fin[fin > 0].tolist())
    print("\nEarnedPremNet at the first development age, per unit:")
    print(f"  present and non-zero {prem_true}   all-zero {prem_zero}   missing {prem_missing}")
    if vals:
        print(f"  positive premium per origin: n={len(vals)} median {np.median(vals):,.0f} "
              f"p10 {np.percentile(vals, 10):,.0f} p90 {np.percentile(vals, 90):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
