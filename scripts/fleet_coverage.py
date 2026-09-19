"""Does the distribution cover the truth at the rate it claims? (#8, the second half)

The fleet already answered the question about the *centre*: the direct arm's point estimate is closer than
Chain Ladder on 38.8% of evaluations. This measures the *spread*, which that result says nothing about, and
it is the claim the entry actually rests on.

Coverage is a per-unit binary outcome -- was the actual future inside the interval -- so unlike a percentage
error it has no denominator and cannot be distorted by a triangle whose reserve is a few hundred units. That
makes it the right fleet statistic, and it is the one the pre-registered E3 asked for.

    python scripts/fleet_coverage.py --draws 300          # all units, resumable
    python scripts/fleet_coverage.py --limit 40           # a quick read

Resumable: rows append to the JSONL and completed units are skipped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import Triangle, direct_training_rows, factor_reserve, target_ages
from fleet_eval import load_fleet

# The same column and held-out policy as the point-estimate run, so the two are directly comparable.
LEVELS = [0.50, 0.75, 0.90, 0.95]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--column", default="IncurLoss")
    ap.add_argument("--held-out", type=int, default=3)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/fleet/coverage.jsonl"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                done.add(json.loads(line)["unit"])
            except Exception:  # noqa: BLE001
                pass
    print(f"coverage: column={args.column} held out={args.held_out} draws={args.draws} "
          f"| already done: {len(done)}")

    pairs = load_fleet(args.limit)
    n_rows, skipped = 0, {}
    t0 = time.time()

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with args.out.open("a") as fh:
        for name, sub in pairs:
            try:
                tri = Triangle.load(sub, column=args.column).trim()
            except Exception as exc:  # noqa: BLE001
                skip(f"not a usable triangle ({type(exc).__name__})")
                continue
            for anchor in range(tri.n - 2, tri.n - 2 - args.held_out, -1):
                unit = f"{args.column}|k={anchor}|{name}"
                if unit in done:
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
                    model = arm.make_model("local")
                    model.fit(X, y)
                    tgt = target_ages(tri.n, "backtest")
                    out = arm.reserve_direct(model, tri, anchor, args.draws,
                                             np.random.default_rng(0), target="delta", targets=tgt)
                except Exception as exc:  # noqa: BLE001 -- one unit must not stop the fleet
                    skip(f"model failure ({type(exc).__name__})")
                    continue
                s = out["samples"]
                if s is None or not np.isfinite(s).all():
                    skip("no usable draws")
                    continue
                ga = tri.global_factors(tri.known(anchor))
                row = {"unit": unit, "triangle": name, "n": int(tri.n), "anchor": int(anchor),
                       "horizon": int(tri.n - 1 - anchor), "actual": float(actual),
                       "point": float(out["reserve"]),
                       "median": float(np.median(s)), "mean": float(np.mean(s)),
                       "chainladder": float(factor_reserve(tri, anchor, ga,
                                                           targets=target_ages(tri.n, "backtest"))),
                       "draws": int(args.draws)}
                for lv in LEVELS:
                    lo, hi = np.quantile(s, [(1 - lv) / 2, 1 - (1 - lv) / 2])
                    row[f"lo_{int(lv * 100)}"], row[f"hi_{int(lv * 100)}"] = float(lo), float(hi)
                    row[f"covered_{int(lv * 100)}"] = bool(lo <= actual <= hi)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_rows += 1
                if n_rows % 20 == 0:
                    print(f"  {n_rows} units | {(time.time() - t0) / 60:.1f} min | {name[:36]} k={anchor}")
    print(f"\nwrote {n_rows} coverage units to {args.out} in {(time.time() - t0) / 60:.1f} min")
    if skipped:
        print("skipped, by reason (nothing dropped silently):")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {reason}")
    return summarise_coverage(args.out)


def summarise_coverage(path: pathlib.Path) -> int:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    if not rows:
        print("no coverage units")
        return 1
    print(f"\n=== coverage: {len(rows)} units, horizons "
          f"{sorted(set(r['horizon'] for r in rows))} ===")
    print(f"{'nominal':>8} {'units':>6} {'covered':>9} {'empirical':>10} {'gap':>8} "
          f"{'median width / point':>22}")
    for lv in LEVELS:
        key = f"covered_{int(lv * 100)}"
        sub = [r for r in rows if key in r]
        if not sub:
            continue
        cov = np.mean([bool(r[key]) for r in sub])
        widths = np.array([(r[f"hi_{int(lv * 100)}"] - r[f"lo_{int(lv * 100)}"])
                           / max(abs(r["point"]), 1e-9) for r in sub])
        print(f"{lv:>8.0%} {len(sub):>6} {int(cov * len(sub)):>9} {cov:>10.1%} "
              f"{100 * (cov - lv):>+7.1f}pp {np.median(widths):>21.2f}")
    print("\n  by horizon:")
    for h in sorted({r["horizon"] for r in rows}):
        hh = [r for r in rows if r["horizon"] == h]
        line = "  ".join(f"{lv:.0%}: {100 * np.mean([bool(r[f'covered_{int(lv * 100)}']) for r in hh]):.0f}%"
                         for lv in LEVELS if f"covered_{int(lv * 100)}" in hh[0])
        print(f"    horizon {h} (n={len(hh):4d}):  {line}")
    med = np.array([r["median"] for r in rows])
    act = np.array([r["actual"] for r in rows])
    print(f"\n  centre check: the sampled median is inside the truth's own scale in "
          f"{100 * np.mean((med > 0) == (act > 0)):.1f}% of units (sign agreement)")
    print(f"  a perfectly calibrated 50% interval would cover 50%; see the gap column for the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
