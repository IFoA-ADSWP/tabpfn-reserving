# Loss reserving

The vocabulary this project uses to describe loss triangles, the classical methods it compares against, and the
model arms it runs.

> **Drafted by the engineering side. Actuarial review pending.** Several of these terms have loose usage in
> practice — *link ratio* and *age-to-age factor* in particular are often used interchangeably. The distinction
> drawn here is the one this codebase depends on, not a claim about industry convention.

## Language

**Reserve**:
Money held against claims that have occurred but are not yet settled. UK practice often says *claims provision*;
this project uses "reserve" throughout.
_Avoid_: liability, estimate

**Loss triangle**:
A table of cumulative claim payments by accident year and development period. The unknown lower-right corner is
what a reserving method must estimate.
_Avoid_: development triangle, run-off triangle

**Accident year**:
The year in which the claims arose. The row of the triangle.
_Avoid_: origin year, origin period

**Development period**:
How far a claim has developed, counted in years from the start of its accident year. The column of the triangle.
_Avoid_: age, lag, delay

**Cumulative paid**:
Total amount paid to date for one accident year at one development period.
_Avoid_: losses, incurred

**Ultimate**:
The estimated total cost of an accident year once all development is finished.
_Avoid_: final loss, total incurred

**Link ratio**:
The growth of one accident year from one development period to the next — `C[a,d] / C[a,d-1]`.
_Avoid_: development factor, age-to-age factor

**Age-to-age factor**:
The volume-weighted average of link ratios across accident years at one development step. A property of the
column, not of a single cell.
_Avoid_: link ratio, average link

**Chain Ladder**:
The classical method: estimate age-to-age factors, multiply them forward into the empty cells, and sum those
cells to give the reserve.
_Avoid_: standard method, basic method

**Mack**:
Chain Ladder plus an analytical estimate of the uncertainty around it.
_Avoid_: Mack's method, Mack chain ladder

**ODP bootstrap**:
A simulation-based alternative to Mack for the range around a reserve.
_Avoid_: bootstrap, over-dispersed Poisson

**Anchor**:
The diagonal of the triangle a model is allowed to see, written `k = a + d` for the last observed diagonal.
_Avoid_: valuation date, cut-off

**Depth**:
How many steps past its last observed cell an accident year has been projected, `depth = (d - 1) - (k - a)` for
anchor `k`. Training rows sit at depth ≤ −1; the first predicted cell is depth 0.
_Avoid_: horizon, step, lead time

**Arm**:
One prediction method in a comparison — for example the TabPFN arm, the Mack arm, or the Chain Ladder arm.
_Avoid_: model, approach, variant

**Delta arm**:
The arm that predicts a link ratio divided by Chain Ladder's own factor, so that `1.0` reproduces Chain Ladder
exactly and the model learns only a correction.
_Avoid_: residual arm, ratio arm

**Reserve distribution**:
The range of reserve outcomes, not a single number. Capital requirements are read from a percentile of it.
_Avoid_: reserve range, uncertainty band

**Coverage**:
Whether a stated interval actually contains the truth at its stated rate. A 90 % interval should contain the
truth 90 % of the time across many triangles.
_Avoid_: calibration, confidence

**IBNR**:
Claims incurred but not yet reported. Part of what a reserve covers.
_Avoid_: outstanding claims
