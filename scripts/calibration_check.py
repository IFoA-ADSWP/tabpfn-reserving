"""Can the miscalibration be corrected? (#21)

The coverage run measured the failure: under-coverage at every level, intervals too wide, the actual above the
sampled median in 67.7% of units. That is a *location-and-scale* error, and location-and-scale errors are the
kind of thing that can be learned from other triangles and applied to unseen ones -- a different claim from
"the model is calibrated", and a more useful one.

Method. Work in ratio space (each unit divided by its own point estimate, so a triangle of 7s and one of
millions weigh the same). Fit a single affine correction q' = d + s*q by minimising pinball loss over the eight
recorded quantiles on a training half of the units, then measure coverage on the held-out half. Two controls,
because a correction fitted and evaluated on the same data proves nothing:

  * the test half is never used to fit;
  * a placebo fits the same procedure to *shuffled* targets -- if that also restores coverage, the result is
    an artefact of the fitting, not a property of the model.

SEED is fixed, so the whole thing reproduces exactly.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

SEED = 0
LEVELS = [0.50, 0.75, 0.90, 0.95]
# The recorded bounds are the (1-tau)/2 and (1+tau)/2 quantiles of the predictive distribution.
TAU = {0.50: (0.25, 0.75), 0.75: (0.125, 0.875), 0.90: (0.05, 0.95), 0.95: (0.025, 0.975)}


def load(path="results/fleet/coverage.jsonl"):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    units, dropped = [], 0
    for r in rows:
        p = abs(r["point"])
        if p < 1.0:  # a unit whose reserve is ~0 cannot be normalised by it
            dropped += 1
            continue
        lo = {lv: r[f"lo_{int(lv * 100)}"] / p for lv in LEVELS}
        hi = {lv: r[f"hi_{int(lv * 100)}"] / p for lv in LEVELS}
        q, tau = [], []
        for lv in LEVELS:
            for side, t in ((lo[lv], TAU[lv][0]), (hi[lv], TAU[lv][1])):
                q.append(side)
                tau.append(t)
        units.append({"a": r["actual"] / p, "lo": lo, "hi": hi, "q": q, "tau": tau,
                      "horizon": r["horizon"], "unit": r["unit"]})
    return units, dropped, len(rows)


def _pinball(y, q, tau):
    """Total pinball loss. y: (n,), q: (n_models, n_units, n_q), tau: (n_q,)."""
    diff = y[None, :, None] - q
    return np.maximum(tau[None, None, :] * diff, (tau[None, None, :] - 1.0) * diff).sum(axis=(1, 2))


def fit(train):
    """Grid-search the affine correction (d, s) minimising pinball loss over all quantiles."""
    y = np.array([u["a"] for u in train])
    Q = np.array([u["q"] for u in train])
    tau = np.array(train[0]["tau"])
    best_ds, best_loss = (0.0, 1.0), np.inf
    for d in np.arange(-1.0, 1.001, 0.02):
        losses = _pinball(y, d + np.arange(0.05, 3.001, 0.02)[:, None, None] * Q[None, :, :], tau)
        i = int(np.argmin(losses))
        if losses[i] < best_loss:
            best_loss = float(losses[i])
            best_ds = (float(d), float(0.05 + 0.02 * i))
    return best_ds


def coverage(units, d: float = 0.0, s: float = 1.0) -> dict:
    return {lv: float(np.mean([(d + s * u["lo"][lv]) <= u["a"] <= (d + s * u["hi"][lv]) for u in units]))
            for lv in LEVELS}


def main():
    units, dropped, total = load()
    print(f"units {total}, usable {len(units)}, dropped (point estimate ~0) {dropped}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(units))
    half = len(units) // 2
    train = [units[i] for i in idx[:half]]
    test = [units[i] for i in idx[half:]]
    hz = [sum(1 for u in test if u["horizon"] == h) for h in (1, 2, 3)]
    print(f"train {len(train)}, test {len(test)}  (test horizons 1/2/3: {hz[0]}/{hz[1]}/{hz[2]})")

    d, s = fit(train)
    shuffled = [dict(u, a=x["a"]) for u, x in zip(train, rng.permutation(train))]
    dp, sp = fit(shuffled)
    print(f"\ncorrection fitted on the training half :  q' = {d:+.3f} + {s:.3f} * q")
    print(f"same procedure on SHUFFLED targets     :  q' = {dp:+.3f} + {sp:.3f} * q")

    before_tr, before_te = coverage(train), coverage(test)
    after_te, placebo_te = coverage(test, d, s), coverage(test, dp, sp)

    print(f"\n{'nominal':>8} | {'train before':>12} | {'test before':>11} | {'test corrected':>14} | {'test placebo':>12}")
    print("-" * 72)
    for lv in LEVELS:
        print(f"{lv:>7.0%} | {before_tr[lv]:>11.1%} | {before_te[lv]:>10.1%} | {after_te[lv]:>13.1%} | {placebo_te[lv]:>11.1%}")

    gap = lambda c: float(np.mean([abs(c[lv] - lv) for lv in LEVELS]))
    print(f"\nmean |gap from nominal| -- test before {gap(before_te):.1%} | corrected {gap(after_te):.1%} | placebo {gap(placebo_te):.1%}")


if __name__ == "__main__":
    main()
