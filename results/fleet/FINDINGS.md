# The fleet: does the correction earn its place? (#8) — **No**

> **Caution for anyone reading `clrd-IncurLoss.jsonl`.** Its stored **`error_chainladder_pct` is `0.0` on all
> 464 rows** — the null defect found and fixed once already *in the summary* (`scripts/fleet_eval.py:198`,
> "everything is recomputed from primitives") but never rewritten in that file. Reading the field gives
> "closer than Chain Ladder on 0.0%", which is impossible: the recorded rate below is **38.8%**. Recompute from
> the primitives each row carries (`actual`, `direct`, `chainladder`); `scripts/column_comparison.py` does
> exactly that and reports the condition when it meets it. The numbers in this document are the canonical ones
> and were produced by the recomputation.


The single-triangle work could not answer this. With a Chain Ladder-relative target, an arm predicting a
correction of 1.0 *is* Chain Ladder, so "close to Chain Ladder" is partly by construction — and on `abc` the
direct arm was −1.2% against it, which is not a win. The question that survives is whether the model's
correction earns its place **where Chain Ladder is wrong**, and that needed a fleet.

## What was measured

The CAS Loss Reserve Database, `IncurLoss`, evaluated by holding out the last three diagonals so the future
being predicted is observed and scoreable:

```
containers scanned        775
  refused, not a triangle 136   (sparse or fragmented: a company with three accident years sits in the
                                 last three origins of a 10x10 container; some have gaps mid-triangle)
  usable triangles        639   (sizes: 383 of 10, and 256 between 6 and 9)
evaluations              464   (horizons: 177 at 1 step, 148 at 2, 139 at 3)
skipped, counted         1288 no scoreable future (the future is negative or absent)
                          165 too few training rows (direct arm)
```

Nothing was dropped silently, and no model failure is hidden: zero exceptions were recorded across the run.

**The null is the Chain Ladder arm on the same cells and the same actual future.** Note the first version of
the harness stored that null's error as **0.0 by construction** — against which no arm can ever look better,
and "closer on 0.0%" is what it printed. Chain Ladder has its own error; conflating "the model's correction
is zero" with "the method is perfect" is the kind of instrumentation lie that makes a result meaningless while
looking complete.

## The result

| statistic | direct arm | Chain Ladder (the null) |
|---|---|---|
| \|error\| median | **162.6%** | **121.3%** |
| \|error\| median, larger half of triangles | 122.4% | 108.9% |
| closer than the null | **38.8%** of evaluations | 61.2% |
| paired \|error\| change | median **+30.9** percentage points (worse) | — |

By horizon, the arm is closer than Chain Ladder on **35.6%** (1 step), **41.9%** (2 steps) and **39.6%**
(3 steps) — it loses at every horizon, so this is not a horizon artefact.

**And it is not being cautious.** The model moves the reserve by a **median of 141.8% of Chain Ladder's
reserve** on these triangles. It is applying an enormous correction, and the correction is wrong more often
than not. That is the worst possible combination: not a model that hedges toward the incumbent, but one that
overrides it and loses.

**The failure is variance, not a consistent bias.** Deciles of the paired difference (arm's |error| minus
Chain Ladder's, in percentage points):

```
p10      p25      p50      p75      p90
-174     -41      +31     +208     +849
```

The arm is better by 25 percentage points or more on **30.6%** of evaluations, and worse by that much on
**51.3%**. It wins big sometimes and loses bigger more often — an unstable estimator, not a mildly biased one.
A model like that cannot be shipped with a "usually about right" claim, which is precisely why the paired test
was worth the fifty minutes.

All four headline numbers above were recomputed independently from the raw fields (`actual`, `chainladder`,
`direct`) rather than read from the harness's own summary, and they reproduce exactly.

The mean (+4,136 percentage points) and the currency-weighted aggregate (3.5 million per cent against Chain
Ladder's 13,908) are unusable — a handful of near-zero-denominator cases dominate both — so the medians and
the paired win rate are the reportable statistics, and they are all consistent.

## What this settles, and what it does not

**Settled.** The point estimate is not competitive, at fleet scale, at any horizon measured, in the
configuration that survived the single-triangle work. That was a pre-registered possible outcome, and it is
now an answer rather than a hope. Anyone reading this project should not expect a reserving tool that beats
Chain Ladder on the point estimate.

**Not settled.** The *distribution* claim — the reason the entry exists — is untouched by this: nothing here
measures coverage, interval width against Mack or the ODP bootstrap, or whether the model's uncertainty is
better calibrated than theirs. The fleet says the centre of the distribution is not where the incumbent's is
and not where the truth is. It says nothing about the spread.

**And one honest caveat about the test itself.** In production the arm trains on the full run to the ultimate;
here every training label is clamped to the evaluation anchor (`known_until`) so the model cannot be trained
on the tail it is about to predict. That clamp makes this a *harder* test than production — but the
unclamped alternative is not a test at all, and a number produced that way would have been wrong in the
flattering direction.

## Where the entry's substance now sits

Three things, in order of how much weight they can carry:

1. **The distribution** — a full predictive distribution from one forward pass, drawn from the model's own bar
   distribution, whose arithmetic reproduces the CAS package to the pound, and whose intervals *contain* the
   incumbent's answer on every triangle tested. That is the deliverable.
2. **The depth-bias diagnostic** (`results/runs/20260918-023600_depth-bias/FINDINGS.md`) — a measured,
   reproducible account of *when this class of model should not be trusted*, with a clean zero intercept and a
   false-negative trap documented.
3. **This negative result**, at fleet scale, with the null explicit and the funnel visible. An entry that
   reports "it does not beat the incumbent, here is 464 evaluations and here is why" is a better technical
   artefact than one that reports a single flattering triangle.
