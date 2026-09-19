# The direct arm: delete the recursion (#18)

**What was built.** `--direct` predicts each origin's *whole remaining development* in one row — the factor
from the anchor's last observed level straight to the target age — with **horizon** as a feature and Chain
Ladder's implied factor as a prior to correct. No recursion, nothing predicted is fed back.

The recursive arm's failure is architectural, not incidental: every intermediate level it feeds the model *is
the model's own guess*, so nine of them multiply. Here they add.

## The measurement

Production anchor (the last diagonal), 300 bar-bins draws, `abc`, `genins`, `ukmotor`:

| triangle | target | direct arm | Chain Ladder | error | fit |
|---|---|---|---|---|---|
| `abc` | delta | 5,211,802 | 5,277,760 | **−1.2%** | 4.8 s |
| `abc` | ratio | 2,371,208 | 5,277,760 | **−55.1%** | 4.2 s |
| `genins` | delta | 14,298,765 | 18,680,856 | **−23.5%** | 5.2 s |
| `ukmotor` | delta | 19,253 | 28,656 | **−32.8%** | 4.3 s |

For comparison, the *recursive* delta arm on `abc` was **+66.8%** (8,801,618).

## Verdict against the pre-registered bar: not met

The bar was *production error on `abc` below +20% **and** reproduced on `genins` and `ukmotor`*. The first
half passes emphatically — +66.8% became −1.2%. The second half fails: −23.5% and −32.8% are outside a
sensible tolerance, and they are **under**-reserves where the recursive arm overshot.

So it is not adopted as a solution, and it is not a failure either, because it settles the question that
mattered:

**The compounding was architectural.** Deleting the recursion removed a ×1.67 error down to a near-tie on
`abc`, with no change to the model, the features or the hyperparameters. The error now adds across origins
rather than multiplying across steps, which is why the correction no longer snowballs with the horizon.

## The caveat that governs how to read the whole table

With `target="delta"` the label is *the factor relative to Chain Ladder's own projection*, so **an arm that
predicts a correction of 1.0 everywhere reproduces Chain Ladder exactly.** Agreement with Chain Ladder is
therefore partly by construction, and the delta column is best read as "how far the model moved off the
incumbent", not as accuracy.

The ratio column is the control for that, and it is the uncomfortable one: with **no** Chain Ladder prior the
direct arm halves the reserve (−55.1%). So the model's own multi-horizon development factors are not
sufficient; what produces the near-tie is the Chain Ladder prior plus a small learned correction. The honest
description of the delta arm is **Chain Ladder with a learned correction**, not a foundation model doing
reserving.

That reframes the only question worth asking next: not "how close is it to Chain Ladder" (the target answers
that), but **"does it deviate usefully where Chain Ladder is wrong?"** Which needs a fleet, and a control:
the pure Chain Ladder arm is the null, and its error is 0 by definition.

## The caveat the CLI itself prints, and what it costs

The run prints its own coverage check, and it does not pass:

```
horizons in training 1-8  |  horizons required 0-10
```

Training rows come from anchors 2…9, so the deepest horizon the model ever sees is 8. Production asks for
**0–10**: the two youngest accident years are extrapolated beyond *every* example in the training set — the
same class of defect as the anchoring trap in the depth-bias work (a measurement whose range does not cover
the claim), caught this time before it became a conclusion rather than after.

It is small in currency terms because the two youngest origins carry the smallest bases, and horizon 0 is
degenerate (nothing left to develop, so the reserve contribution is zero by construction). But it is not
nothing, and it means the −1.2% on `abc` is achieved with the deepest two origins guessing beyond the range
the model has examples for. Widening the training anchors to k=1 would push the deepest training horizon to
9 and cost one more row per anchor; going further is impossible on a single triangle, which is the argument
for the fleet rather than more work here.

## The distribution, for what it is worth alongside the point estimate

| triangle (delta) | point | mean | median | p5 | p95 | Chain Ladder |
|---|---|---|---|---|---|---|
| `abc` | 5,211,802 | 5,329,680 | 5,581,983 | 2,351,587 | 7,413,958 | 5,277,760 |
| `genins` | 14,298,765 | 13,631,064 | 15,065,690 | 3,045,318 | 23,625,730 | 18,680,856 |
| `ukmotor` | 19,253 | 18,726 | 21,787 | 4,643 | 29,112 | 28,656 |

**The interval contains the incumbent's answer on all three triangles**, where the recursive arm's ruled it
out: on `abc` its p5 was 7,452,235 against Chain Ladder's 5,277,760, so the whole interval sat above the
standard method — a distribution asserting the incumbent was impossible. The direct arm's p5 is 2,351,587
with Chain Ladder comfortably inside; `genins` places it between p75 and p90, and `ukmotor` between p90 and
p95.

That is the distributional consequence of deleting the compounding, and it is the more defensible statement
regardless of the point estimate: an honest interval should *cover* the incumbent where the incumbent is
plausible, not contradict it by construction.

The shapes differ between the two triangles — `abc`'s point sits below its median with a downside reaching
less than half the point, `genins` spreads across a factor of nine from p5 to p99 — so on one triangle the arm
is confidently low and on the other it is simply unsure. Against Chain Ladder-relative targets, both are the
model saying "Chain Ladder may be wrong here", which is a claim for the fleet, not for one triangle.


## A process note worth keeping

The first run of this appeared to produce `8,801,618` — the *recursive* arm's number, to the pound, on
`abc` — because the patch that was supposed to add the `--direct` branch silently did not apply, while other
edits in the same script did. The script's guard asserted that *a* replacement had happened, not that *this*
replacement had. `--direct` was accepted on the command line and ignored.

What caught it was the *shape* of the number, not the number: a new arm reproducing the old arm's output to
the pound is a copy, not a result. The lesson is the one this project keeps relearning — assert the specific
change, and treat a suspiciously exact agreement as a bug in the measurement before treating it as a finding.
