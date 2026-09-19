"""The fleet: does the arm deviate usefully from Chain Ladder where Chain Ladder is wrong? (#8)

The question is no longer "how close is it to Chain Ladder" -- with a Chain Ladder-relative target, an arm
predicting a correction of 1.0 *is* Chain Ladder, so agreement is partly by construction. The question is
whether the model's correction earns its place, and that needs triangles the single-triangle work could not
provide.

**The null is explicit**: the pure Chain Ladder arm, whose error is zero by definition on the delta target.
Beating it means the learned correction improved on doing nothing.

**The horizon ceiling is a design property, not a limitation to hide.** A 10x10 triangle has one diagonal of
unknown cells, so any held-out evaluation projects a fixed number of steps: hold out the last diagonal and
every projection is 1 step; hold out three and every projection is 3 steps. Depth 9 exists only where the
whole tail is missing, which cannot be scored. So this evaluates horizons 1-3, and the production claim's
horizon is an extrapolation from here.

    python scripts/fleet_eval.py --limit 20 --out results/fleet/pilot.jsonl
    python scripts/fleet_eval.py                 # all 775, resumable

Resumable by design: rows append to the JSONL, and triangles already present are skipped, because a 775-triangle
run is hours of unattended CPU and a crash at hour three must not cost three hours.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import (Triangle, direct_training_rows, factor_reserve, target_ages,
                                       training_rows)

COLUMN = "IncurLoss"       # the reserving column; CumPaidLoss is the robustness check


def load_fleet(limit: int = 0) -> list[tuple[str, object]]:
    """Every (company, line of business) triangle in the CAS Loss Reserve Database."""
    import chainladder as cl

    sample = cl.load_sample("clrd")
    keys = list(sample.key_labels)
    pairs: list[tuple[str, object]] = []
    for i in range(np.asarray(sample.values).shape[0]):
        sub = sample.iloc[i]
        name = " / ".join(str(k) for k in keys) + f"#{i}"
        pairs.append((name, sub))
        if limit and len(pairs) >= limit:
            break
    return pairs


def reserve_direct_at(model, tri: Triangle, anchor: int) -> float:
    """The direct arm's reserve at an anchor, backtest targets -- what the data can score."""
    out = arm.reserve_direct(model, tri, anchor, 0, np.random.default_rng(0), target="delta",
                             targets=target_ages(tri.n, "backtest"))
    return out["reserve"]


def reserve_recursive_at(model, tri: Triangle, anchor: int) -> float:
    out = arm.reserve(model, tri, anchor, 0, np.random.default_rng(0), target="delta",
                      targets=target_ages(tri.n, "backtest"))
    return out["reserve"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N triangles (0 = all 775)")
    ap.add_argument("--column", default=COLUMN)
    ap.add_argument("--held-out", type=int, default=3,
                    help="how many diagonals to hold out; sets the projection horizon (see the docstring)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/fleet/clrd.jsonl"))
    ap.add_argument("--arms", nargs="*", default=["direct"],
                    choices=["direct", "recursive"], help="which arms to evaluate")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            try:
                done.add(json.loads(line)["unit"])
            except Exception:  # noqa: BLE001
                pass
    print(f"fleet: column={args.column} held out={args.held_out} diagonals "
          f"| already done: {len(done)}")

    pairs = load_fleet(args.limit)
    print(f"       loaded {len(pairs)} triangles from clrd")

    n_rows = 0
    skipped: dict[str, int] = {}
    t_start = time.time()

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with args.out.open("a") as fh:
        for name, sub in pairs:
            try:
                tri = Triangle.load(sub, column=args.column)
            except Exception as exc:  # noqa: BLE001
                skip(f"load ({type(exc).__name__})")
                continue
            try:
                tri = tri.trim()
            except ValueError:
                skip("not a proper triangle (sparse or fragmented container)")
                continue
            for anchor in range(tri.n - 2, tri.n - 2 - args.held_out, -1):
                unit = f"{args.column}|k={anchor}|{name}"
                if unit in done:
                    continue
                ga = tri.global_factors(tri.known(anchor))
                actual = tri.actual_future(anchor)
                if not np.isfinite(actual) or actual <= 0:
                    skip("no scoreable future")
                    continue
                row = {"unit": unit, "triangle": name, "column": args.column, "anchor": int(anchor),
                       "n": int(tri.n), "horizon": int(tri.n - 1 - anchor), "actual": float(actual),
                       "chainladder": float(factor_reserve(tri, anchor, ga,
                                                           targets=target_ages(tri.n, "backtest")))}
                for which in args.arms:
                    train = list(range(2, max(3, anchor)))
                    try:
                        if which == "direct":
                            X, y = direct_training_rows(tri, train, target="delta",
                                                        known_until=anchor)
                            if len(y) < 5:
                                skip("too few training rows (direct)")
                                continue
                            model = arm.make_model("local")
                            model.fit(X, y)
                            row["direct"] = float(reserve_direct_at(model, tri, anchor))
                        else:
                            X, y = training_rows(tri, anchor, ga, target="delta")
                            if len(y) < 5:
                                skip("too few training rows (direct)")
                                continue
                            model = arm.make_model("local")
                            model.fit(X, y)
                            row["recursive"] = float(reserve_recursive_at(model, tri, anchor))
                    except Exception as exc:  # noqa: BLE001 -- one triangle must not stop the fleet
                        row[f"{which}_error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
                if not any(w in row for w in args.arms):
                    continue          # no arm produced a number -- not a result, and not a row
                # Chain Ladder's own error against the actual future. NOT zero: the null is a *method* with
                # the same task, not a perfect answer. (Setting this to 0.0 by construction -- as the first
                # version did -- makes every paired comparison a tautology against a fabricated ideal.)
                row["error_chainladder_pct"] = 100 * (row["chainladder"] - actual) / actual
                for which in args.arms:
                    if which in row:
                        row[f"error_{which}_pct"] = 100 * (row[which] - actual) / actual
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_rows += 1
                if n_rows % 25 == 0:
                    mins = (time.time() - t_start) / 60
                    print(f"  {n_rows} rows | {mins:.1f} min | last {name[:40]} k={anchor}")
    print(f"\nwrote {n_rows} rows to {args.out} in {(time.time() - t_start) / 60:.1f} min")
    if skipped:
        print("skipped, by reason (no unit is dropped silently):")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {reason}")
    return summarise(args.out)


def summarise(path: pathlib.Path) -> int:
    """The paired comparison: the null is Chain Ladder, so the question is the difference.

    Levels are reported but not led with. The fleet runs from three-origin triangles with a reserve of a few
    hundred currency units to full 10x10 lines in the millions, so a percentage error has a near-zero
    denominator on the small ones and explodes regardless of method -- the first 96 rows had a median error
    of 178% and a worst case of 28,679%. Chain Ladder suffers the same, which is exactly why the paired test
    and the currency-weighted aggregate are the answers and the levels are context.
    """
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    rows = [r for r in rows if np.isfinite(r.get("actual", np.nan))]
    if not rows:
        print("no rows")
        return 1

    print(f"\n=== fleet summary: {len(rows)} evaluations ===")
    sizes = {int(k): int(v) for k, v in zip(*np.unique([r["n"] for r in rows], return_counts=True))}
    horizons = {int(k): int(v) for k, v in zip(*np.unique([r["horizon"] for r in rows],
                                                          return_counts=True))}
    print(f"    effective triangle sizes: {sizes}")
    print(f"    horizons: {horizons}")

    for which in sorted({w for r in rows for w in ("direct", "recursive") if w in r}):
        # Everything is recomputed from primitives (`actual`, `chainladder`, the arm's reserve). The stored
        # percentage fields are convenient but derived, and the first version of this harness stored a
        # Chain-Ladder error of 0.0 for every row -- against which no arm can ever look better.
        for r in rows:
            if np.isfinite(r.get("actual", np.nan)) and r["actual"] != 0:
                r["error_chainladder_pct"] = 100 * (r["chainladder"] - r["actual"]) / r["actual"]
                if which in r:
                    r[f"error_{which}_pct"] = 100 * (r[which] - r["actual"]) / r["actual"]
        sub = [r for r in rows if which in r
               and np.isfinite(r[f"error_{which}_pct"]) and np.isfinite(r.get("error_chainladder_pct", np.nan))]
        if not sub:
            continue
        err = np.array([r[f"error_{which}_pct"] for r in sub])
        cl = np.array([r["error_chainladder_pct"] for r in sub])
        actual = np.array([abs(r["actual"]) for r in sub])
        paired = np.abs(err) - np.abs(cl)
        moved = np.abs(np.array([r[which] for r in sub]) - np.array([r["chainladder"] for r in sub]))
        denom = np.maximum(np.abs(np.array([r["chainladder"] for r in sub])), 1e-9)

        print(f"\n  --- {which} arm, {len(sub)} paired evaluations against the null ---")
        print(f"    the null is the Chain Ladder arm on the same cells and the same actual future; it is")
        print(f"    wrong on most of these triangles, which is what makes it a comparison rather than a")
        print(f"    tautology. (The first version of this harness stored its error as 0.0 by construction.)")
        print(f"    |error| median   arm {np.median(np.abs(err)):9.1f}%   Chain Ladder "
              f"{np.median(np.abs(cl)):9.1f}%")
        print(f"    closer than Chain Ladder on {100 * np.mean(paired < 0):5.1f}% of triangles, "
              f"worse on {100 * np.mean(paired > 0):5.1f}%")
        print(f"    paired |error| change: median {np.median(paired):+9.1f} percentage points, "
              f"mean {np.mean(paired):+9.1f}")
        print(f"    the model moves off Chain Ladder by a median of {100 * np.median(moved / denom):.1f}% "
              f"of the reserve (so how much correction it is actually applying)")

        # The currency-weighted aggregate: what an actuary would care about, and immune to tiny denominators.
        big = actual >= np.quantile(actual, 0.5)
        sub_big = [r for r, m in zip(sub, big) if m]
        err_big = np.array([abs(r[f"error_{which}_pct"]) for r in sub_big])
        cl_big = np.array([abs(r["error_chainladder_pct"]) for r in sub_big])
        print(f"    restricted to the larger half of triangles (|actual| >= "
              f"{np.quantile(actual, 0.5):,.0f}): median |error| arm {np.median(err_big):8.1f}%  "
              f"Chain Ladder {np.median(cl_big):8.1f}%")
        print(f"    currency-weighted aggregate: arm "
              f"{100 * np.sum(np.abs(err) * actual) / np.sum(actual):8.1f}%  Chain Ladder "
              f"{100 * np.sum(np.abs(cl) * actual) / np.sum(actual):8.1f}%")

        print(f"    by horizon:")
        for h in sorted({r["horizon"] for r in sub}):
            hh = [r for r in sub if r["horizon"] == h]
            eh = np.array([abs(r[f"error_{which}_pct"]) for r in hh])
            ch = np.array([abs(r["error_chainladder_pct"]) for r in hh])
            ph = np.abs(eh) - np.abs(ch)
            print(f"      horizon {h}: n={len(hh):4d}  median |error| arm {np.median(eh):8.1f}%  "
                  f"CL {np.median(ch):8.1f}%  closer on {100 * np.mean(ph < 0):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
