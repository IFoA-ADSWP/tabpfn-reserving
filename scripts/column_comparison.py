"""R4 (first half): is the failure column-specific? Incurred against paid.

The pre-registered expectation, written before the run (`results/remodel/FINDINGS.md` R4): **the failure is not
column-specific.** On the units scoreable in both columns the paid arm's closer-than-Chain-Ladder rate should be
*not better than* 38.8% + 2.3pp, i.e. the column choice should not move the result.

**Falsifier, and it is a result in its own right:** *if paid and incurred differ by more than the noise floor in
either metric, then the column choice is a research result* -- and the entry's fleet finding must be reported per
column, not as one number.

**Errors are recomputed from primitives, never read from the stored fields.** `clrd-IncurLoss.jsonl` carries
`error_chainladder_pct = 0.0` on all 464 rows -- the null-defect this repository found once already, fixed in
the *summary* (fleet_eval.py:198, "everything is recomputed from primitives") but never rewritten in that file.
Reading it produced "closer than Chain Ladder: 0.0%", i.e. a fresh verdict from a known-broken column. This
script recomputes from `actual` / `direct` / `chainladder`, and refuses to run at all if those primitives are
missing.

Paired on `(triangle, anchor)` rather than compared as two independent rates, because both runs score the same
triangles at the same anchors. Units scoreable in only one column drop out of the intersection and are counted,
not silently absorbed.

    python scripts/column_comparison.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

INC = pathlib.Path("results/fleet/clrd-IncurLoss.jsonl")
PAID = pathlib.Path("results/fleet/clrd-CumPaidLoss.jsonl")
NOISE_PP = 2.3   # the paired-rate noise floor, docs/experiments.md §5.4 (38.8% ± 2.3pp at n=464)
MIN_UNITS = 100  # below this the comparison has no power and must not be adjudicated
PRIMITIVES = ("actual", "direct", "chainladder")


def key(r: dict) -> tuple[str, int]:
    return (r["triangle"], int(r["anchor"]))


def load(path: pathlib.Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            missing = [k for k in PRIMITIVES if k not in r]
            if missing:
                raise SystemExit(f"{path} lacks the primitives {missing}; nothing here may be trusted")
            out[key(r)] = r
    return out


def errs(r: dict) -> tuple[float, float]:
    """(model, chainladder) signed percentage errors, recomputed from primitives."""
    a = float(r["actual"])
    if not np.isfinite(a) or a == 0:
        return (np.nan, np.nan)
    return (100 * (float(r["direct"]) - a) / a, 100 * (float(r["chainladder"]) - a) / a)


def stale_field(path: pathlib.Path) -> bool:
    """True when the STORED chainladder error is degenerate -- the condition that produced the false 0.0%."""
    vals = [json.loads(l)["error_chainladder_pct"] for l in path.read_text().splitlines() if l.strip()]
    return bool(vals) and all(v == 0.0 for v in vals)


def main() -> int:
    inc, paid = load(INC), load(PAID)
    for path in (INC, PAID):
        if stale_field(path):
            print(f"note: {path.name} stores error_chainladder_pct = 0.0 on every row (the known null defect).")
            print("      Recomputed from primitives below; the stored field is not used anywhere here.")
    common = sorted(set(inc) & set(paid))
    print(f"\nincurred units {len(inc)} | paid units {len(paid)} | intersection {len(common)}")
    print(f"  paid-only {len(set(paid) - set(inc))}, incurred-only {len(set(inc) - set(paid))}")
    if len(common) < MIN_UNITS:
        print(f"\n  CANNOT ADJUDICATE -- {len(common)} units, need >= {MIN_UNITS}.")
        return 2

    rows = [(inc[k], paid[k]) for k in common]
    mi, ci = zip(*(errs(i) for i, _ in rows))
    mp, cp = zip(*(errs(p) for _, p in rows))
    mi, ci, mp, cp = map(lambda v: np.asarray(v, dtype=float), (mi, ci, mp, cp))
    keep = np.isfinite(mi) & np.isfinite(ci) & np.isfinite(mp) & np.isfinite(cp)
    mi, ci, mp, cp = mi[keep], ci[keep], mp[keep], cp[keep]
    n = int(keep.sum())

    rate_i = float(np.mean(np.abs(mi) < np.abs(ci)))
    rate_p = float(np.mean(np.abs(mp) < np.abs(cp)))
    print(f"\n{'metric (n=%d)' % n:<38} {'incurred':>12} {'paid':>12}")
    print(f"{'median |model error| %':<38} {np.median(np.abs(mi)):>12.1f} {np.median(np.abs(mp)):>12.1f}")
    print(f"{'median |Chain Ladder error| %':<38} {np.median(np.abs(ci)):>12.1f} {np.median(np.abs(cp)):>12.1f}")
    print(f"{'closer than Chain Ladder':<38} {rate_i:>11.1%} {rate_p:>11.1%}  (noise floor {NOISE_PP:.1f}pp)")

    dd = (np.abs(mp) < np.abs(cp)).astype(float) - (np.abs(mi) < np.abs(ci)).astype(float)
    se = dd.std(ddof=1) / np.sqrt(n)
    print(f"\npaired difference (paid - incurred) in closer-than-CL: {dd.mean():+.1%} (se {se:.1%}, n={n})")
    print(f"paired |model error| median change: {np.median(np.abs(mp)) - np.median(np.abs(mi)):+.1f}pp")

    print("\n=== pre-registered verdict ===")
    if abs(rate_p - rate_i) > NOISE_PP / 100:
        print(f"  The rates differ by {abs(rate_p - rate_i):.1%}, beyond the {NOISE_PP:.1f}pp noise floor.")
        print("  FALSIFIER FIRED: the column choice is a research result in its own right, and the entry's fleet")
        print("  finding must be reported per column rather than as one number.")
    else:
        print(f"  Paid ({rate_p:.1%}) is within the noise floor of incurred ({rate_i:.1%}).")
        print("  EXPECTATION MET: the failure is not column-specific, and the fleet result stands as one number.")
    print(f"\n  Both columns recomputed from primitives; the paid run also scores {len(paid)} units against")
    print(f"  incurred's {len(inc)}, which is a data availability fact worth reporting separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
