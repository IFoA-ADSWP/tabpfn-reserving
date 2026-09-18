# Depth bias: what causes the over-reserve (#17, hypothesis 1) — SUPPORTED

**The hypothesis.** Every training row sits at depth ≤ 0: the cell's level is observed. Production asks for
steps 1 through 9. If the model is unbiased where it was trained and biased upward where it is extrapolating,
the per-step error compounds and a small per-step bias becomes a large reserve error.

**Measured.** 287 scored steps of the delta arm across `abc` and `genins`, at every anchor from k=3 up to the
last but one. For each step: `log(predicted ratio / actual ratio)`, against the depth of that step.

| depth | steps | mean log error | × per step | exposure-weighted |
|---|---|---|---|---|
| 0 | 88 | +0.0102 | 1.010 | +0.0157 |
| 1 | 69 | +0.0209 | 1.021 | +0.0365 |
| 2 | 52 | +0.0501 | 1.051 | +0.0793 |
| 3 | 37 | +0.0979 | 1.103 | +0.1538 |
| 4 | 24 | +0.1374 | 1.147 | +0.2082 |
| 5 | 13 | +0.1881 | 1.207 | +0.2977 |
| 6 | 4 | +0.1446 | 1.156 | +0.1439 |

```
regressed on depth:  slope = +0.0317 per step  (se 0.0036, t ≈ 8.7)
                     intercept = +0.0001        <- unbiased at depth 0, by construction
compounds to ×1.33 over nine steps; the measured production over-reserve on abc delta is ×1.67
per triangle:  abc +0.0190 (se 0.0016)   genins +0.0617 (se 0.0079)
```

**Verdict: supported, and the mechanism is now specific.** The model is unbiased exactly where it has training
examples and increasingly biased upward as it extrapolates — about **+3.2% per step**, reaching +19% by the
fifth step, in £-terms worse still. Compounded over the nine steps production actually asks for, that is
**×1.33** against a measured over-reserve of **×1.67**: most of the error, from one mechanism.

That is the signature of predicting outside the training distribution, and it makes the fix concrete rather
than speculative: **make deep steps in-distribution by training across anchors**, so the model sees cells whose
level is its own prediction instead of only cells whose level is known. See #18.

## The methodological trap this run first fell into, because it is easy to repeat

The first version used anchors `n-4 … n-2` — the last three — because that is the natural thing to do. Those
anchors have projections at most **2 steps** long, so the deepest measurable step was depth 2. On that window
the per-step slope for `abc` was **−0.0018 (se 0.0030): indistinguishable from zero**, and the honest-looking
conclusion was "no depth bias; hypothesis dead". With anchors from k=3 the same triangle gives **+0.0190
(se 0.0016)**.

**A backtest anchored near the ultimate cannot see the depths production uses, and it reports a flat line
while doing so.** Measure bias against depth, and check that the depth range measured covers the depth range
claimed.

## Where the money was

The largest single residuals are all `genins` long-horizon origins, and they are enormous:

```
genins k=3 origin=2  6 steps  base 1,292,306  predicted 37,562,473  actual 3,617,009  +938%
genins k=3 origin=3  6 steps  base   310,608  predicted 24,344,004  actual 4,277,660  +469%
genins k=3 origin=1  6 steps  base 2,170,033  predicted 12,848,231  actual 3,169,052  +305%
```

Six steps of a +24%-per-step bias gives a factor of about four; these are factors of ten and thirty, because
the error also compounds *through the base* — each step's prediction becomes the next step's level, so the bias
enters both the ratio and the thing the ratio is applied to. The per-step diagnostic measures the ratio only;
the by-origin reconstruction, which replays both paths from the anchor base, is what shows the full effect.

## Provenance

`results/runs/20260918-023600_depth-bias/` — `run.log` (the console record, including the earlier
misleading window) and `depth_bias.json` (every scored step: triangle, anchor, origin, development, depth,
predicted and actual ratio, currency exposure, log error).
