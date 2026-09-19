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

## A process note worth keeping

The first run of this appeared to produce `8,801,618` — the *recursive* arm's number, to the pound, on
`abc` — because the patch that was supposed to add the `--direct` branch silently did not apply, while other
edits in the same script did. The script's guard asserted that *a* replacement had happened, not that *this*
replacement had. `--direct` was accepted on the command line and ignored.

What caught it was the *shape* of the number, not the number: a new arm reproducing the old arm's output to
the pound is a copy, not a result. The lesson is the one this project keeps relearning — assert the specific
change, and treat a suspiciously exact agreement as a bug in the measurement before treating it as a finding.
