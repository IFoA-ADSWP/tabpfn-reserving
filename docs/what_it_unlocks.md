# What it unlocks — for a reserving actuary

**One sentence.** A loss triangle can be handed to TabPFN-3.5 as a prediction problem, and the whole
predictive distribution of the reserve comes back in a single forward pass on a laptop CPU — while the
classical method needs a thousand simulated refits for the same object.

**And one warning, up front, because this page is for someone who might use it.** The *point estimate* from
this method is not competitive, and the repository measures that rather than hiding it: over 464 paired
evaluations on the CAS Loss Reserve Database it was closer than Chain Ladder on 38.8% of triangles
(`results/fleet/FINDINGS.md`). **Do not use it to produce a reserve.** Use it for the distribution, for a
second opinion, and for the diagnostic in the last section — which is the part that is genuinely new.

## What the day-to-day actually needs is a quantile

The reserve is not the number that gets used. What gets used is a *quantile* of it:

- **A risk margin**, or an **IFRS 17 risk adjustment** — the cost of holding the uncertainty, which is a
  percentile of the reserve distribution above the central estimate.
- **A Solvency II capital figure** — a 99.5th percentile. Reporting a reserve with an analytical standard
  error and then implying a distribution through a normal approximation is a choice that the standard error
  does not make for you, and it is wrong for a skewed reserve distribution.

Mack's method gives you a standard error built on a distributional assumption. The ODP bootstrap gives you an
empirical distribution from *n* refits — and it is the right idea. This is the same object, from one pass.

## What works today

- **The distribution.** `output_type="full"` returns the model's own bar distribution: bin edges and weights.
  The draws in this repository are sampled from it directly (inverse-CDF over the bins), not from an
  interpolated grid of quantiles — a distinction that mattered: the grid version understated the `abc` 99th
  percentile by 62%.
- **It runs on CPU, with no API calls and no tuning.** A production run on a 10×10 triangle is about 1–2
  minutes end to end. There is no hyperparameter to choose, and nothing was tuned for any triangle in the
  repository — that is the claim being tested rather than a convenience.
- **It is honest about the incumbent.** On `abc`, `genins` and `ukmotor` the model's interval *contains* Chain
  Ladder's answer. An interval that excluded the standard method would be making a claim it cannot support.
- **It knows the triangle as a domain object.** The reframing — each unknown cell a supervised target, every
  feature restricted to what the valuation date could have known — is written down in `docs/method.md`, so
  the approach can be attacked where it is weak rather than admired.

## What does not work today, stated plainly

- **The point estimate.** See the warning above. At fleet scale the model applies a large correction (a median
  of 142% of Chain Ladder's reserve) and is wrong more often than right. It is an unstable estimator, not a
  mildly biased one.
- **No tail factor.** The reserve stops at the longest development age observed anywhere in the triangle. A
  genuine ultimate has a tail beyond that, which this does not model.
- **No correlation between cells.** Per-cell draws are independent in the recursive arm, so the joint
  distribution is missing the correlation a triangle actually has (`#15`).
- **The horizon limit.** A 10×10 triangle has one unknown diagonal, so single-triangle validation cannot
  reach the depth production asks for. The fleet measures 1–3 steps; production is up to nine, and that gap
  is stated rather than papered over.
- **The uncertainty is wide.** On `abc` the 5th percentile was 2.35m against a point of 5.21m. A wide
  interval is honest, but it is not decision-ready for a capital calculation without knowing whether the
  width is *right* — which is what the coverage measurement in `results/fleet/coverage.jsonl` is for.

## The part that is genuinely new: knowing when not to trust it

The most useful thing here for a practising actuary is not the reserve. It is a measured answer to *when a
model of this class stops being reliable*.

Recursive forecasting asks the model for the next step, feeds its own answer back, and asks again — nine times
for the youngest accident year. Measuring every step against depth:

```
depth        0       1       2       3       4       5
bias     +0.010  +0.021  +0.050  +0.098  +0.137  +0.188
```

**Unbiased where the model has training examples; +3.2% per step (se 0.36%) where it extrapolates**, with a
clean zero intercept and 287 scored steps behind it. That is the shape of a model being asked to predict
outside what it has seen, and it is measurable on your own triangle in one command (§ below).

Two practical consequences, and the second is the reason this section exists:

1. Deleting the recursion removed the compounding: the same model, features and settings went from +66.8% to
   −1.2% on `abc`. The architecture, not the model's knowledge, was the problem.
2. **A backtest that ends near the ultimate cannot see this.** Anchored at the last three valuation dates, the
   deepest measurable step is depth 2 and the bias reads **−0.0018 (se 0.0030)** — flat, and the effect looks
   absent. Anchored from nine years back the same triangle gives **+0.0190 (se 0.0016)**. If your validation
   window is short, you will conclude a recursive model is fine when it is not.

## Running it on your own triangle

```bash
pip install -e .
python -m tabpfn_reserving abc --distribution     # bundled CAS sample: reserve + distribution + figure
python -m tabpfn_reserving abc --direct           # direct arm: no recursion, no compounding
python -m tabpfn_reserving abc --backtest         # per-anchor, scored
```

**The CLI takes a bundled sample name**, not an arbitrary triangle: your own data needs four lines of the
library (`#19` covers making this a flag):

```python
import chainladder as cl, numpy as np
from tabpfn_reserving import arm
from tabpfn_reserving.triangle import Triangle, direct_training_rows, target_ages

tri = Triangle.load(cl.load_sample("ukmotor"))       # or build one from a numpy array
X, y = direct_training_rows(tri, list(range(2, tri.n - 2)), target="delta")
model = arm.make_model("local").fit(X, y)
out = arm.reserve_direct(model, tri, tri.n - 1, 300, np.random.default_rng(0), target="delta")
print(out["reserve"], out["reserve_mean"], out["reserve_median"])
```

`Triangle.load` refuses collections (775 triangles keyed by company *and* line of business) and multi-column
samples (`mcl` carries incurred *and* paid) rather than silently picking one — that refusal is deliberate,
because the choice changes the numbers while looking equally legitimate either way.

Every run writes a record: the command, the git revision, package versions, a fingerprint of the matrix it
used, the fit and predict wall-clock, the distribution route (`bar-bins` versus a quantile grid), each step's
error, and the draws themselves. Nothing in this repository quotes a number that cannot be regenerated from
the command beside it.
