#!/usr/bin/env python3
"""PROTOTYPE — throwaway. Answers one question: does the 19th arm's output fit the metric we already trust?

Not production. No tests, no error handling beyond runnable. Never merged; captured on
`prototype/nineteenth-depth` as a primary source.

The question. The recorded depth-bias finding is a regression of `log(predicted ratio / actual ratio)` on
depth, slope +0.0317. Before spending six fits on 19th (#7), check that predictions coming back from 19th can
be pushed through that same transform and produce a slope — including detecting one we inject on purpose.

THE THING THE FIRST CUT GOT WRONG, kept here because it is the whole point of prototyping:

    Scoring every unobserved cell against the truth lets depth run to 12, because it silently uses
    *level* values the model never had. Depth only exists because the recursion multiplies its own
    predictions forward. Score the walk, not the cells.

So `walk()` reproduces the recursion: predict the next diagonal, write the predicted ratio into the running
level, then predict the diagonal after that using that level. Depth is the step index.

Why the ratio and not the delta. The arm trains on `target_delta = (C[a,d]/C[a,d-1]) / gf[d-1]`, but `gf`
cancels inside the log error: `log(delta_pred/delta_actual) == log(ratio_pred/ratio_actual)`. This works on raw
ratios and needs none of the global-factor machinery — a real simplification for the real arm too.

    python3 scripts/prototype_nineteenth_depth.py --stand-in perfect   # expect: no variation, no slope
    python3 scripts/prototype_nineteenth_depth.py --stand-in bias      # expect: slope ~= the injected bias
    python3 scripts/prototype_nineteenth_depth.py --stand-in bias --noise 0.05

Real predictions plug in at `predict_ratio()`, which is the only function that changes: it takes the cell and
the running level and returns the model's ratio for that step.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
import statistics

DEFAULT_DATA = pathlib.Path(
    "/Users/Scott/projects/19th-private-alpha/data/synthetic_population_full.csv"
)


def load_cells(path: pathlib.Path) -> dict[int, dict[tuple[int, int], float]]:
    cells: dict[int, dict[tuple[int, int], float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            triangle = int(row["triangle_id"])
            cells.setdefault(triangle, {})[(int(row["origin"]), int(row["development"]))] = float(
                row["cumulative_paid"]
            )
    return cells


def predict_ratio(cell, running_level, actual_ratio, bias, rng, noise):
    """THE SEAM. Stand-in here; real 19th predictions would be looked up by (triangle, origin, development)."""
    ratio = actual_ratio * math.exp(bias * cell["depth"])
    if noise and rng is not None:
        ratio *= math.exp(rng.gauss(0.0, noise))
    return ratio


def walk(cells, anchor: int, size: int, bias: float, noise: float, seed: int):
    """Run the recursion forward from one anchor, recording per-step ratio error and its depth."""
    rng = random.Random(seed)
    steps = []
    for triangle, truth in sorted(cells.items()):
        # The model sees only what is known at the anchor.
        running = {key: value for key, value in truth.items() if sum(key) <= anchor}
        for offset in range(1, size - anchor):
            diagonal = anchor + offset
            for origin in range(0, min(anchor, size - 1) + 1):
                development = diagonal - origin
                if development < 1 or development >= size:
                    continue
                level_before = running.get((origin, development - 1))
                true_now = truth.get((origin, development))
                true_before = truth.get((origin, development - 1))
                if level_before is None or true_now is None or not true_before:
                    continue

                depth = (development - 1) - (anchor - origin)
                actual_ratio = true_now / true_before
                cell = {
                    "triangle": triangle,
                    "origin": origin,
                    "development": development,
                    "depth": depth,
                    "actual_ratio": actual_ratio,
                    "level_before": level_before,
                }
                predicted = predict_ratio(cell, level_before, actual_ratio, bias, rng, noise)
                # The projected level is what the next step consumes — this is what makes depth exist.
                running[(origin, development)] = level_before * predicted
                steps.append({**cell, "predicted_ratio": predicted})
    return steps


def regress(steps):
    errors = [math.log(s["predicted_ratio"] / s["actual_ratio"]) for s in steps]
    xs = [s["depth"] for s in steps]
    n = len(errors)
    mean_x, mean_y = statistics.mean(xs), statistics.mean(errors)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, errors)) / sxx if sxx else float("nan")
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, errors)]
    se = (
        math.sqrt(sum(r * r for r in residuals) / (n - 2) / sxx)
        if n > 2 and sxx > 0
        else float("nan")
    )
    return errors, slope, se, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--anchor", type=int, default=3, help="k; 3 gives depth 0..5 on a 10-origin triangle")
    parser.add_argument("--stand-in", choices=["perfect", "bias"], default="bias")
    parser.add_argument("--bias", type=float, default=0.03, help="injected per-step bias")
    parser.add_argument("--noise", type=float, default=0.0, help="lognormal sd on each predicted ratio")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("PROTOTYPE — not production\n")
    cells = load_cells(args.data)
    steps = walk(cells, args.anchor, 10, args.bias if args.stand_in == "bias" else 0.0, args.noise, args.seed)
    if not steps:
        raise SystemExit("no scored steps — check the anchor")

    errors, slope, se, n = regress(steps)
    depths = [s["depth"] for s in steps]

    print(f"anchor k={args.anchor}  stand-in={args.stand_in}  noise={args.noise}  scored steps={n}")
    print(f"depth range: {min(depths)}..{max(depths)}   triangles: {len({s['triangle'] for s in steps})}\n")

    print(f"{'depth':>6} {'steps':>6} {'mean log error':>15} {'x per step':>11}")
    for depth in sorted(set(depths)):
        at_depth = [e for e, s in zip(errors, steps) if s["depth"] == depth]
        mean = statistics.mean(at_depth)
        print(f"{depth:>6} {len(at_depth):>6} {mean:>15.4f} {math.exp(mean):>11.3f}")

    print()
    if len({round(e, 12) for e in errors}) < 2:
        print("verdict: no variation in the errors — no slope to estimate")
    elif n < 50:
        print(f"verdict: only {n} scored steps — the real script refuses a slope below 50")
    else:
        t = slope / se if se and not math.isnan(se) else float("nan")
        print(f"slope {slope:+.4f} per step (se {se:.4f}, t ≈ {t:.1f})")
        print(f"compounds to x{math.exp(slope * max(depths)):.2f} over {max(depths)} steps")
        if args.stand_in == "bias":
            print(f"\nPROTOTYPE CHECK: injected {args.bias:+.4f}, recovered {slope:+.4f}")


if __name__ == "__main__":
    main()
