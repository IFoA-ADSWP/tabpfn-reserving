# #14 tried, measured, and not adopted — a negative result

**The hypothesis.** In `features_for` (`mode="frozen"`), the origin's level and last ratio are read at the
*anchor*, so they are identical for every step of that origin's projection: the model is asked to predict a
ratio for a cell whose current level it was never shown, and the depth of the projection is invisible to it.
Feeding it the running level should reduce the over-reserve.

**The pre-registered bar** (written into #14 before the code existed): beat the delta arm's **+67% on `abc`**,
with point-to-median agreement inside 5%, or it does not ship.

## The measurement

Three configurations, `abc`, production anchor, delta target, 300 draws:

| features | seed | point reserve | error vs Chain Ladder | p50 |
|---|---|---|---|---|
| frozen | 0 | 8,801,618 | **+66.8%** | 8,481,123 |
| frozen | 0 (repeat) | 8,801,618 | +66.8% | 8,481,123 |
| running | 0 | 8,616,245 | **+63.3%** | 8,307,185 |
| running | 1 | 8,616,245 | +63.3% | 8,439,168 |

Chain Ladder, like-for-like on the same cells: **5,277,760**.

## Verdict: measured, marginal, not adopted

**The information helps by 2.1%** — 8,616,245 against 8,801,618, a deterministic difference, six times larger
than nothing and thirty times smaller than the error it was meant to explain. Against a 67% over-reserve that
is not a fix; calling it one because it clears the bar's letter would be exactly the kind of rounding the
pre-registration exists to prevent. **So the default stays `frozen`, and #14 closes as tried and not
adopted** — not as unfinished.

**And that is a result, not a shrug.** The hypothesis was that the compounding comes from the model not
knowing where the projection has got to. The model now knows, and the compounding is essentially unchanged.
**Feature staleness is therefore not the explanation** — the error is not an information deficit the model
could have closed had we described the cell better. That redirects the next experiment rather than repeating
this one, and #17 carries it with the bar restated.

## Two things the repeats bought us, which are worth more than the 2.1%

**The pipeline is deterministic.** The same command twice produced the same reserve to the pound, and the
point estimate does not depend on the seed at all (seeds 0 and 1 agree on 8,616,245) — only the draws do.
This is E0's run-to-run control, unmeasured until now: it means any future difference larger than the draw
noise is attributable, and it is what the reproducibility criterion asks for. The old spike's numbers could
not claim this, because its code was never run twice under the same conditions.

**The draw noise floor at 300 draws is about 1.6%.** The running arm's median moved from 8,307,185 to
8,439,168 between seeds while its point estimate did not move at all. So percentiles from 300 draws are
quoted to a precision they do not have — which is #16, now with a number attached.
