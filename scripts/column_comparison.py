"""R4 (first half): is the failure column-specific? Incurred against paid.

The pre-registered expectation, written before the run (`results/remodel/FINDINGS.md` R4): **the failure is not
column-specific.** On the units scoreable in both columns the paid arm's closer-than-Chain-Ladder rate should be
*not better than* 38.8% + 2.3pp, i.e. the column choice should not move the result.

**Falsifier, and it is a result in its own right:** *if paid and incurred differ by more than the noise floor in
either metric, then the column choice is a research result* — and the entry's fleet finding must be reported per
column, not as one number.

Paired on `(triangle, anchor)` rather than compared as two independent rates, because the two runs score the
same triangles at the same anchors; units whose paid column is not scoreable simply drop out of the
intersection, and the count of those is reported rather than silently absorbed.

    python scripts/column_comparison.py            # after the CumPaidLoss run completes
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

INC = pathlib.Path("results/fleet/clrd-IncurLoss.jsonl")
PAID = pathlib.Path("results/fleet/clrd-CumPaidLoss.jsonl")
NOISE_PP = 2.3  # the paired-rate noise floor, docs/experiments.md §5.4 (38.8% ± 2.3pp at n=464)
MIN_UNITS = 100  # below this the comparison has no power and must not be adjudicated


def key(r: dict) -> tuple[str, int]:
    return (r["triangle"], int(r["anchor"]))


def load(path: pathlib.Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[key(r)] = r
    return out


def closer_than_cl(r: dict) -> bool:
    e_model = abs(r["error_direct_pct"])
    e_cl = abs(r["error_chainladder_pct"])
    return e_model < e_cl


def main() -> int:
    inc, paid = load(INC), load(PAID)
    common = sorted(set(inc) & set(paid))
    print(f"incurred units {len(inc)} | paid units {len(paid)} | intersection {len(common)}")
    print(f"  paid-only {len(set(paid) - set(inc))}, incurred-only {len(set(inc) - set(paid))} "
          "(-- scored futures are observed on 460/464 for paid per the recount)")
    if not common:
        print("no overlap yet -- has the paid run finished?")
        return 1

    rows = [(inc[k], paid[k]) for k in common]
    a = np.array([abs(i["error_direct_pct"]) for i, _ in rows])
    b = np.array([abs(p["error_direct_pct"]) for _, p in rows])
    ca = np.mean([closer_than_cl(i) for i, _ in rows])
    cb = np.mean([closer_than_cl(p) for _, p in rows])
    diff = np.array([closer_than_cl(p) - closer_than_cl(i) for i, p in rows])
    se = diff.std(ddof=1) / np.sqrt(diff.size)
    se_rate = np.sqrt(ca * (1 - ca) / len(rows))

    print(f"\n{'metric':<34} {'incurred':>12} {'paid':>12}")
    print(f"{'median |error| %':<34} {np.median(a):>12.1f} {np.median(b):>12.1f}")
    print(f"{'closer than Chain Ladder':<34} {ca:>11.1%} {cb:>11.1%}")
    print(f"{'  noise floor (+/-)':<34} {NOISE_PP:>10.1f}pp {NOISE_PP:>10.1f}pp")
    print(f"\npaired difference (paid - incurred), closer-than-CL: {diff.mean():+.1%} "
          f"(se {se:.1%}, n={diff.size})")
    print(f"  paired |error| median change: {np.median(b) - np.median(a):+.1f}pp")

    print("\n=== pre-registered verdict ===")
    # Guard: a verdict is only issued where there is power to issue it. Caught live -- on an
    # intersection of 3 units with both rates at 0.0% this printed "EXPECTATION MET", which is not a
    # weaker version of a result, it is the absence of one. Third occurrence of this shape in one
    # session; hence an explicit guard rather than care.
    if len(common) < MIN_UNITS or (ca == 0.0 and cb == 0.0):
        print(f"  CANNOT ADJUDICATE -- {len(common)} units in the intersection (need >= {MIN_UNITS}), "
              f"rates {ca:.1%} vs {cb:.1%}.")
        print("  A verdict here would be a statement about the sample size, not about the columns.")
        print("  Wait for the run to finish; do not read this line as the expectation being met.")
        return 2
    moved = abs(cb - ca) > NOISE_PP / 100
    if not moved:
        print(f"  The paid arm's rate ({cb:.1%}) is within the noise floor of the incurred arm's ({ca:.1%}).")
        print("  EXPECTATION MET: the failure is not column-specific, and the fleet result stands as one number.")
    else:
        print(f"  The rates differ by {abs(cb - ca):.1%}, beyond the {NOISE_PP:.1f}pp noise floor.")
        print("  FALSIFIER FIRED: the column choice is a research result in its own right. The entry's fleet")
        print("  finding must be reported per column, not as one number, and the paid/incurred gap becomes a")
        print("  candidate explanation for the incurred arm's case-reserve noise.")
    print(f"\n  (rates: incurred {ca:.1%}, paid {cb:.1%}; incurred's own se {se_rate:.1%} on this intersection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
