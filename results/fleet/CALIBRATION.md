# Can the miscalibration be corrected? (#21) — three attempts, one clean finding

`results/fleet/COVERAGE.md` measured the failure. This asks the obvious next question: is it *correctable*?
Everything below runs on the 464 recorded units, split 231/231 — fitted on one half, evaluated on the other —
so no number here is fitted and scored on the same units. `scripts/calibration_check.py`, `SEED=0`.

## Attempt 1 — affine correction, fitted by pinball loss. **Fails.**

Work in ratio space (each unit divided by its own point estimate, so a triangle of 7s and one of millions weigh
the same). Fit `q' = d + s·q` over the eight recorded quantiles by minimising pinball loss on the training half:

```
fitted:            q' = +0.560 + 1.030 q        (almost pure upward shift, no rescaling)
same on shuffled:  q' = +0.760 + 0.230 q

nominal | test before | test corrected | test placebo
   50%  |    39.0%    |     32.9%      |    15.2%
   75%  |    61.5%    |     58.0%      |    25.5%
   90%  |    77.9%    |     79.2%      |    36.8%
   95%  |    84.4%    |     86.1%      |    45.0%
mean |gap|            |  11.8% → 13.4%  (slightly worse)
```

A per-horizon version is worse still (50% level: 39.0% → 31.2%). The diagnosis said "the centre is too low",
but shifting the whole distribution up does not fix unseen triangles — so the error is not a common offset.

## Why: the error is heavy-tailed and unit-specific

```
actual/point − 1 over all 464 units:  median −0.42   mean +2.36   p10 −0.95   p90 +4.16
```

The typical unit is **over**-predicted by 42%; the mean is dragged to +236% by a small number of extreme units.
No global location-and-scale transform can fix a distribution whose errors are like that.

## The clean finding: the failure is entirely one-sided

```
on the held-out half, where does the truth sit?
  above the model's 95% upper bound:  34 of 231 = 14.7%   (nominal 2.5% — 5.9x too often)
  below the model's 95% lower bound:   2 of 231 =  0.9%   (nominal 2.5% — if anything too wide)
  excess over the upper bound, as a fraction of the centre: median +0.36, p90 +1.92, max 5.2x
```

**The upper tail is far too short and the lower tail is not.** This sharpens `COVERAGE.md`, which said
"centring, not width": the more precise statement is that the model's distribution misses *upward*, badly and
often, and never downward — 1 unit in 7 is beyond its own 95% bound on the high side. That is the same
one-sidedness the "actual exceeds the sampled median in 67.7% of units" line captured, measured at the tail
where a risk margin is actually read.

## Attempt 2 — coverage-matched widening. **Works, and is not a result.**

Fit one scaling factor per bound per level so that empirical coverage on the training half hits nominal:

```
nominal | s_lo  s_hi | train after | test before | test after | placebo
   50%  | 0.52  2.32 |    50.2%    |    39.0%    |   53.2%    |  62.3%
   75%  | 0.71  1.97 |    75.3%    |    61.5%    |   77.5%    |  80.1%
   90%  | 0.92  2.38 |    90.0%    |    77.9%    |   97.4%    |  92.6%
   95%  | 0.94  2.68 |    94.8%    |    84.4%    |   99.1%    |  93.1%
                                              4.3% mean gap   5.5%
```

The fit does what it claims on train, and it transfers: 11.8% → 4.3% mean gap on unseen triangles, with the
upper bound needing to be **2.3–2.7×** further out and the lower bound slightly *tighter* — consistent with the
one-sided finding above.

**But the placebo gets 5.5%, nearly the same, so this is not evidence of anything the model taught us.** Two
honest reasons to distrust the apparent win, both worth stating rather than burying:

1. The correction is a *single global widening* — no unit-level information is used — so the gain is "make the
   intervals much wider", which any method's intervals can claim. Without #20 (Mack and the ODP bootstrap
   measured on these same units) there is no way to know this is anything other than arithmetic on widths.
2. The placebo is a **weak control for a marginal-coverage fit**: shuffling reassigns the same actuals across
   the same intervals, so the aggregate fraction the bisection targets is barely disturbed. It rules out
   "fitted on noise", not "fitted on nothing in particular". A control that *would* discriminate needs the
   correction to be conditional on the unit (horizon, triangle size, the model's disagreement with Chain
   Ladder) — which is untested and is what #21 asks for.

## What this leaves

- **Correctable by widening, not by calibration.** The honest claim the entry can make is that the miscalibration
  is one-sided and repairable at a stated cost in width — and that no *learnable* correction was found.
- **Two bugs found in my own analysis before the numbers meant anything**: a bisection with inverted brackets
  (converged to a 30× interval and 100% coverage — which reads as a perfect fix), and an inverted target on the
  lower tail (collapsed the lower bound, producing a "placebo" that matched a broken fix). Both were caught by
  printing the *train* coverage as a self-check. A correction that achieves 100% coverage is not a correction.
- **The tail asymmetry is the transferable result**, and it is what a practitioner should take: a risk margin
  read off this distribution's upper percentiles would be materially too small — not on a few triangles, but on
  one in seven.
