# E10 — reformulating the reserving problem for a tabular foundation model

> **Status:** written 2026-09-19, reading and analysis only. **No model was fitted, no repository code was
> changed, and no result is claimed here.** Every number of *ours* is quoted from a file in `results/`, except
> one new **read-only recount** of the data columns (`results/remodel/column_availability.py` →
> `column_availability.log`), which loads triangles, applies the same window and the same anchors as
> `scripts/fleet_eval.py`, and counts finite cells — it predicts nothing.
>
> **The one-sentence answer.** The vendor's documentation does not contain a formulation change for our
> problem — what it contains is a **target-transform, an exposure-as-feature convention, a decomposition, and a
> fixed bucket grid**, and the sharpest of those is a mechanism: TabPFN-3.5's regression distribution is a fixed
> quantile grid on the pretraining prior, **rescaled to the training target's mean and standard deviation**, so
> the shape of the answer is set by *what you choose to predict* — which is exactly the lever the mechanism in
> `results/runs/20260918-023600_depth-bias/FINDINGS.md` says we need. The cheapest high-value family moves the
> **level** off the model and onto exposure (premium), which we have never used and which is present for all 464
> units; one arm is free (the paid column, already nominated as a robustness check in our own harness docstring)
> and one is free of compute entirely (a pure-arithmetic measurement of how often the *truth* lies outside the
> range the model was shown).

---

## 0. What was read, and how

Fetched 2026-09-19. Docs pages were read as the vendor's own markdown source (`docs.priorlabs.ai/<page>.md`,
discovered via `llms.txt`), the report as its own PDF (text extracted locally), and every claim about *our*
environment was checked against the installed code, not against the docs.

| # | Source | What it is | Tag |
|---|---|---|---|
| S1 | `https://docs.priorlabs.ai/improving-performance.md` | the escalation path, ordered | verified (quoted) |
| S2 | `.../improving-performance/preprocessing.md` | column typing, target transforms, row subsampling, tuning order | verified (quoted) |
| S3 | `.../improving-performance/feature-engineering.md` | what the vendor means by a domain feature | verified (quoted) |
| S4 | `.../improving-performance/model-parameters.md` | temperature, metric tuning, n_estimators | verified (quoted) |
| S5 | `.../capabilities/predictive-distribution.md` | **the bar distribution mechanism** | verified (quoted) + verified in code |
| S6 | `.../cookbook/insurance_claim_modeling.md` | the vendor's own insurance worked example | verified (quoted) — results **not** re-run |
| S7 | `TabPFN-3.5: Technical Report` (33 pp.) — `https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf` | joint classification/regression training; ScoringBench | verified (quoted) |
| S8 | installed `tabpfn` **9.0.0** (`/Users/Scott/projects/tabpfn-reserving/.venv`) | what our pinned package actually accepts | verified (code) |
| S9 | `Beyond IID: How General Are Tabular Foundation Models, Really?` — arXiv `2606.30410` | the non-IID benchmark's own scope | verified (abstract) |
| S10 | `TabArena: A Living Benchmark…` — arXiv `2506.16791` (v4, 66 pp.) | the vendor's headline benchmark's own scope | verified (PDF text) |
| S11 | CAS Open Source Software, *Chain Ladder Package* — `opensourcesoftware.casact.org/chain-ladder` | what `BulkLoss` is | verified (quoted) |
| S12 | installed `chainladder` **0.10.1** | the exposure-based estimands that already exist next door | verified (code/docstrings) |
| S13 | `results/conditions/FINDINGS.md` | our own prior reading pass (E7) | our file |

Tags: **verified** (read at source, quote reproduced) · **unverified** (the source says it, we cannot check it
from here) · **corrected** (a statement that needed narrowing to stay inside its source).

---

## 1. The reformulation levers the vendor actually documents

### 1.1 Where the principle lands: the parameterisation, not the model

The property that organises everything below is documented in S5 and confirmed in our own installed code:

> **S5 [verified]:** "**The bucket borders are not fitted to your dataset.** They are quantile-based bins fixed
> on TabPFN's synthetic prior before pretraining, in a standardized (z-normalized) space, and never change
> afterwards. At inference time they are simply **rescaled to your target's mean and standard deviation**,
> which is why `preds["criterion"].borders` looks dataset-specific."

> **S8 [verified, code]:** `tabpfn/finetuning/data_util.py:457-460` builds the raw-space distribution as
> `FullSupportBarDistribution(znorm_space_bardist_.borders * train_std + train_mean)`; the class docstring is
> "Bar distribution with half-normal tails, giving support over all of R."

> **S5 [verified]:** "TabPFN treats regression as classification over a grid of buckets on the target axis."
> "For inputs well outside the training range, treat the distribution with caution: bucket boundaries are fixed
> at training time and the model does not flag OOD inputs automatically."

**What this licenses, stated no more strongly than the source.** The *shape* of the predictive distribution is a
rescaled copy of one fixed grid: the target's own scale and spread set where its resolution sits. That is a
mechanism-level reason to expect a **one-sided, too-short upper tail** when the quantity being predicted has a
tail the training target does not — which is our measurement, and the correspondence is **ours, not the
source's** (S5 names the clipping and the absence of an OOD flag; it does not name reserving). It also means a
formulation change is not cosmetic: **changing what is predicted changes the grid the model must fit.**

### 1.2 The target transform — the one lever the docs name for our failure mode

> **S2 [verified]**, `REGRESSION_Y_PREPROCESS_TRANSFORMS` table: `"quantile_uni_extrapolate"` — "**The target has
> ordered tail behavior that must extend beyond the training range.** Validate inverse-transformed predictions
> carefully." And: "Quantile transforms usually clip values outside the training range to the output boundary.
> The extrapolating version keeps some information about how far a new value is outside that range."

> **S2 [verified]**, practical tuning order step 6: "For distribution shift, try `"quantile_uni_extrapolate"`
> alongside the default representation." Step 5: "For skewed regression targets, test one extra target
> transform." Step 8: "Keep a change only when it improves validation results across several seeds or splits."

> **S2 [verified]**, "Add one option at a time because every extra target transform changes the estimator mix."

The same page lists `1_plus_log` ("the target is non-negative and spans a large range"), `safepower`,
`quantile_norm`/`quantile_uni` ("the target is strongly skewed or has heavy tails. **Quantile transforms can
lose useful information in the tails.**"), and notes the base default is `(None, "safepower")`.

> **S8 [verified, code]:** the name is registered in the installed 9.0.0 (`PreprocessorConfig.name` is a
> `Literal` containing `'quantile_uni_extrapolate'`), and
> `TabPFNRegressor(inference_config={"REGRESSION_Y_PREPROCESS_TRANSFORMS": ("quantile_uni_extrapolate",)})`
> constructs. So the arm §5.1b pre-registered is runnable in the pinned environment, not just documented.

### 1.3 Decomposition and exposure — documented in the vendor's own insurance cookbook

S6 is the closest thing Prior Labs publishes to our problem (claim pricing, not reserving). Four statements
matter, and each is a *formulation* choice with a measured effect **in their task**:

> **S6 [verified]:** on a 96%-zero, heavy-tailed monetary target, "TabPFN's default target handling is not built
> for a target that is 96% zeros with a heavy tail on the rest… The fix is the same one the GLMs bake in through
> their log link: model the target on a **log scale**." Measured: Tweedie deviance **74,748 → 39.7**, D²
> **−2175 → −0.156**, ranking unchanged.

> **S6 [verified]:** "The two-stage split helps TabPFN too. Occurrence × conditional aggregate cost is better
> calibrated than a single regressor on the raw zero-inflated amount — the actuarial intuition still pays off."
> Measured, like-for-like (both with the log target transform): Gini **0.3063 → 0.3210**, D²
> **−0.156 → −0.023**.

> **S6 [verified]:** "GLMs fold exposure in through `sample_weight` and an offset. **TabPFN has neither**, so we
> hand it the information directly instead: Exposure becomes an ordinary input feature." And, for the target:
> "We model the *bounded* `ClaimAmount` rather than the exploding per-exposure `PurePremium` (dividing a claim by
> a tiny exposure blows up the tail)."

> **S6 [verified]:** "Out of the box TabPFN **under-prices the portfolio**: summed over the test set it charges
> only ~60% of the actual claim cost. A single factor estimated on held-out training data improves the balance
> from ~0.6 to ~0.96" — with the caveat that makes the calibration slice a design decision, not a formality:
> "TabPFN predicts by conditioning on its training rows *in-context*, so it effectively memorizes them — a policy
> it saw at fit time gets an unrealistically good prediction. Estimating the factor on such rows would
> under-correct, so we re-fit both stages on a subset and score the held-out slice they never saw."

**Note what `sample_weight` means here [verified, S6]:** exposure-weighting is *unavailable* to the local
regressor and must become a feature. This is not an aside — it removes an entire family (weighted fits) from
the menu and redirects it into features.

### 1.4 The escalation path, in the vendor's own order

> **S1 [verified]:** (1) "**Check column types** … **Column typing is the cheapest change and often the largest
> gain.**"; (2) "**Feature engineering** — Add domain features TabPFN cannot derive from raw columns: **ratios,
> interactions, group aggregations, and external signals.**"; (3) row subsampling; (4) metric tuning, "**check
> calibration on skewed targets**"; (5) feature selection; (6) "Experiment with different `PREPROCESS_TRANSFORMS`
> **and target transforms**"; (7) "**Thinking mode** spends more compute at fit time and is **the strongest
> option for grouped and time-ordered data**… Locally, fine-tune…"

> **S2 [verified]**, categorical detection: "an undeclared integer column with more than four distinct values is
> numeric. Store numbers, product codes, and zip codes stored as integers must be declared." Our
> `origin_idx/dev_idx/cal_idx` are exactly that (6–10 distinct integers, passed as floats in a NumPy array).

### 1.5 What is *not* available, or does not bite — verified, so it can be dismissed rather than argued

| Lever the docs describe | Verified state for us | Consequence |
|---|---|---|
| `sample_weight` / offset in a regression fit | **S8:** `TabPFNRegressor.fit(self, X: XType, y: YType)` — **no weight argument**; **S6** confirms ("TabPFN has neither") | weight-based designs are out; exposure must be a **feature** |
| Row subsampling / `"majority_downsample"` | **S2:** it exists to fit tables above the row limit, to trade runtime, or to control the class/target mix per estimator. Our fits are **6–33 rows** (`results/conditions/rows_per_fit.log`) | cannot bite; a zero-heavy *target* is not our case either |
| `POLYNOMIAL_FEATURES`, feature selection, estimator count, SVD | **S2:** aimed at wide tables ("beyond ~6,000 columns" for the feature budget) and at interactions you do not know in advance | our table is 12 columns × ≤33 rows; low expected value, low cost if ever wanted |
| Thinking mode (`group_col`/`time_col`) | **S1/S2 + E7 [verified]:** "the strongest option for grouped **and time-ordered** data", **API/client only** | the vendor's own answer for our split type is *unavailable locally* and costs tokens |
| Fine-tuning | **S1/S2 [verified]:** for "a specialized domain or distribution shift" | parked deliberately (`docs/experiments.md` §2/§5.5: GPU hours, contradicts the zero-shot claim) |

### 1.6 What the published literature adds — and what it takes away

- **The benchmarks underneath the vendor's ranks exclude our regime.** BeyondArena's own scope statement, which
  E7 already quoted from source: non-IID data "ranging from tiny to large (100-1M) … **Out of scope are
  few-shot predictions (<100)**" (`results/conditions/FINDINGS.md` §2.2, B4). **New here and verified at
  source (S10):** TabArena — the benchmark whose Elo the vendor leads — says the same thing at a *higher* floor:
  "We explicitly leave for future work use cases such as non-IID data (e.g., temporal dependencies, subject
  groups, or distribution shifts); **few-shot predictions, or very small data (e.g., less than 500 training
  samples)**". Our fits see 6–33. Two independent scope statements, both above us.
- **BeyondArena's headline finding is our finding's shape [verified, S9 abstract]:** "existing tabular
  foundation models excel on tiny- to medium-sized **IID** data, while traditional tree-based and deep learning
  models still dominate on **non-IID**, large, and high-dimensional datasets."
- **The "classification is the strong axis" premise needs correcting, not repeating.** `docs/experiments.md`
  rests A4 on "this project's own house finding that a count target reframed as classification moves TabPFN from
  its weakest axis to its strongest". No source we have supports it, and for 3.5 two sources point the other
  way: **S7 [verified]** "We modified our model to be trained **jointly on classification and regression**, with
  the task type supplied as an input"; and ScoringBench scores "the full predictive distribution of regression
  models with proper scoring rules", where 3.5 ranks **first of 52** (mean rank 2.85, `§C.7`). **[corrected]**
  A4 survives, but on a different mechanism (§3, R3) — the ladder's value is *where you put the thresholds*,
  not a claim about the model's weaker axis.
- **`bulk/case loss` in this task's own body is wrong, and it matters for what we test.**
  **S11 [verified]:** "the incurred loss data in the Excel file is the total loss reserve, and **the bulk loss
  data is the IBNR data**." Our own recount agrees it is *not* the case reserve: over the 19,988 clrd cells
  where `IncurLoss`, `CumPaidLoss` and `BulkLoss` are all finite, `IncurLoss = CumPaidLoss + BulkLoss` holds
  within 1e-6 on only **6.3%**, with a relative residual median of **17.3%** (p90 48.5%). **`BulkLoss` is the
  IBNR/bulk component; the case reserve is `IncurLoss − CumPaidLoss`, a derived quantity.** **[corrected]**
- **The classical side already ships the reformulations we would otherwise invent.** **S12 [verified,
  docstrings]:** `BornhuetterFerguson` "Blends development with an apriori ultimate"; `ExpectedLoss` "ignores all
  data in the triangle, and only uses the `sample_weight` modified by the apriori"; `CapeCod` "Estimates apriori
  loss ratios from exposure"; `Benktander` "iterative generalization", where "When `n_iters`=1, the result is
  equivalent to the BornhuetterFerguson method. When `n_iters`>>1, the result converges to the traditional
  Chainladder model"; `IncrementalAdditive` (Schmidt 2006) "expected incremental losses satisfy
  `E[Z_{i,k}] = eta_i * gamma_k`, where `eta_i` is exposure…"; `CaseOutstanding` "Estimates incremental paid
  amounts and case-reserve runoff as fractions of the prior lag's carried case reserve… useful when case
  reserves should inform paid ultimates. A triangle with both paid and incurred columns is required"; and
  `MunichAdjustment` — "The Munich method heavily relies on the ratio of paid/incurred and its inverse."
  (`MunichChainladder` itself is **not** in 0.10.1; `MunichAdjustment` is.)
- **A documentation fact worth having, for any ladder arm.** **S7 [verified]:** the first-place ranks in the
  report's summary table belong to `TabPFN-3.5-Thinking` "on the first four benchmarks" — the OSS base
  checkpoint at defaults, which is what we ran, is not the configuration the headline describes.

---

## 2. Untapped information in data we already have

`chainladder`'s `clrd` carries six columns per container (**verified, S12/code**):
`IncurLoss`, `CumPaidLoss`, `BulkLoss`, `EarnedPremDIR`, `EarnedPremCeded`, `EarnedPremNet`. The fleet ran
`IncurLoss` alone, and the harness says so in one line: `COLUMN = "IncurLoss"  # the reserving column;
CumPaidLoss is the robustness check` (`scripts/fleet_eval.py:36`) — a robustness check that was never run.

**Recount, read-only, on exactly the 464 units already on disk** (257 containers; same `regular_block()` window
and the same anchors as `scripts/fleet_eval.py`; `results/remodel/column_availability.log`):

| column | units with ≥5 anchor-known cells | units with the **scored future** fully observed | verdict |
|---|---|---|---|
| `IncurLoss` | 464 | 464 | what we ran |
| **`CumPaidLoss`** | **464** | **460** | runnable as a **second target column** *and* as features |
| `BulkLoss` (IBNR/bulk component) | 454 | **88** | usable as a **feature**, not as a scored target |
| `EarnedPremDIR` | 464 | 464 | exposure |
| `EarnedPremCeded` | 441 | 379 | exposure (ceded) |
| **`EarnedPremNet`** | **464** | **464** | **present and non-zero on all 464 units** (median 1,515 per origin, p10 130, p90 16,078) |

`BulkLoss / IncurLoss` on the anchor diagonal (2,352 origin-cells): median **0.073**, p75 0.245, p90 0.491 —
i.e. the IBNR part is material on roughly the top quarter of origins, which is exactly where our long horizons
sit.

**Are these the kind of features the vendor says to add?** Yes, and specifically — S3's definition is "ratios,
interactions, **group aggregations**, and **external signals**" that "TabPFN cannot derive from raw columns
alone". Each of the three candidates qualifies for a stated reason:

| the information | what it is, mechanically | what it targets in our results |
|---|---|---|
| **paid vs incurred** — the paid/incurred ratio at an anchor, and the paid run-off pattern | a second *view* of the same origins; the model sees one column per fit, so P/I is genuinely underived | the *level* the model starts from. CL's factor on a thin diagonal is what the model is correcting, and the P/I ratio is the standard actuarial signal for whether the incurred figure is inflated by case strengthening (`MunichAdjustment`, `CaseOutstanding`, S12). Targets the **+3.2%/step level drift** (`depth-bias/FINDINGS.md`) and the **median 142% move off CL that is wrong more often than right** (`fleet/FINDINGS.md`) |
| **case reserve** = `IncurLoss − CumPaidLoss` (not `BulkLoss`) | a ratio/level, computable on 464/464 units | the model's own uncertainty about age-to-age movement: case reserves are what convert into paid over the next steps. Targets the **upper tail** if the model is failing to represent the possibility of large upward development (CALIBRATION.md: 14.7% above the 95% bound) |
| **premium as exposure** | an external signal per origin, complete for all 464 units, and the anchor for a Bornhuetter-Ferguson/CapeCod prior | the same failure the vendor's own cookbook measures: "Out of the box TabPFN **under-prices**… a single factor… improves the balance from ~0.6 to ~0.96" (S6) is the pricing analogue of our **under-centring** (truth above the sampled median in 67.7% of units, COVERAGE.md) |

**One trap, documented, that we should not walk into.** The obvious use of exposure is to normalise the
*reserve* by it (a pure-premium-style target). S6 explicitly did the opposite and says why: "we model the
*bounded* `ClaimAmount` rather than the exploding per-exposure `PurePremium` (dividing a claim by a tiny
exposure blows up the tail)". So exposure belongs in the **features** and in the **prior for the level** — not
as a divisor on a target whose tail is the thing under test.

---

## 3. Ranked reformulations

Ordered by (chance it moves *our measured* failure) × (fidelity to the principle: move work away from
extrapolation) ÷ cost. Every entry names the file its target number comes from.

### R1 — Hand the level to exposure: predict the correction to a Bornhuetter-Ferguson/CapeCod prior instead of to Chain Ladder

**Idea.** Keep everything else in the direct arm, but re-anchor the delta target: the prior is
`premium × a-priori loss ratio × %developed` (BF/CapeCod, S12) rather than CL's own projected diagonal, so the
model is asked only for a bounded correction to a level that comes from exposure.
**Evidence.** S12 `BornhuetterFerguson` ("Blends development with an apriori ultimate"), `CapeCod` ("Estimates
apriori loss ratios from exposure"), `Benktander` (interpolates BF→CL as `n_iters` grows) — **verified
(docstrings, pinned version)**; S6 "Exposure becomes an ordinary input feature" — **verified**; S2 `1_plus_log`
for a non-negative target spanning a large range — **verified**.
**Targets.** `results/runs/20260918-023600_depth-bias/FINDINGS.md`: per-step bias is **zero at depth 0 and
+3.2%/step (se 0.36%) where it extrapolates**, compounding to ×1.33 over nine steps against a measured ×1.67
over-reserve — i.e. *the level the recursion multiplies into* is where the money leaks. Secondary target:
`results/fleet/FINDINGS.md` median move off CL of **141.8% of the CL reserve**.
**Cost.** Feature build (premium per origin; an a-priori LR from the anchor's own mature origins; the
%developed from the anchor's factors) + one paired fleet run. ~**1.7 h CPU** with draws, ~**0.85 h** point-only
(`results/fleet/coverage.log`: 104.2 min for 464 units; `results/runs/20260919-pricing/pricing.md`).
**Pre-registered expectation and falsifier.** Expectation: the *median move off CL falls* (the model stops
overriding a stable prior by a factor of ~1.4) and the paired closer-than-CL rate rises above **38.8% ± 2.3pp**
(`docs/experiments.md` §5.4 noise floor). **Falsifier: if the paired rate stays inside 38.8% ± 2.3pp, the
level-anchoring family is dead for this model at this n and should be dropped, not re-tuned.**
**Design warning, pre-registered too.** Re-anchoring the target changes the *null*: the paired baseline must
become BF's own reserve on the same cells, not CL's. Comparing a BF-anchored arm against CL would flatter it
for a reason that has nothing to do with the model.

### R2 — Change the target's support, not just its unit: predict a bounded fraction (%unreported / remaining development)

**Idea.** Make the predicted quantity one that lives in `[0, 1]` by construction — the share of the ultimate
still to emerge — instead of a multiplicative factor whose reference point (1.0) *is* Chain Ladder.
**Evidence.** S5 [**verified**]: the bucket grid is fixed on the prior and merely rescaled to the training
target's mean and sd, so "what you predict" sets the grid's shape and resolution; S6 [**verified**]: the fix for
a heavy-tailed monetary target was a target reparameterisation, measured (Tweedie deviance 74,748 → 39.7),
ranking unchanged. Our own structure: with the delta target, "an arm that predicts a correction of 1.0
everywhere reproduces Chain Ladder exactly" (`src/tabpfn_reserving/arm.py`, `reserve_direct` docstring) — a
bounded fraction has no such degenerate point.
**Targets.** `results/fleet/CALIBRATION.md` — the miss is **one-sided**: 14.7% of units above the model's own
95% upper bound against 0.9% below, with the excess a median of +36% of the centre. A target whose support
*ends* at the ultimate cannot produce that drift the same way a multiplicative factor can.
**Cost.** Same as R1 (~0.85–1.7 h) plus the same feature build; targets are pure arithmetic from the anchor.
**Pre-registered expectation and falsifier.** Expectation: **coverage at 90/95 moves toward nominal before any
widening is applied** (from 78.0% / 84.7%), while the point estimate is broadly unchanged — because this is
supposed to move *support*, not the centre. **Falsifier: if coverage at the 95% level stays within 2 binomial
SE of 84.7%, the support/parameterisation explanation of the one-sided tail is dead** and R3/R6 are the
remaining candidates.

### R3 — A4, run at last: a threshold ladder, distribution rebuilt from the survival curve

**Idea.** Predict `P(development exceeds threshold)` over a ladder of thresholds placed where the risk margin is
read, and rebuild the distribution from the survival function.
**Evidence.** The arm is ours and pre-registered (`docs/experiments.md` §2, E2/A4) but **never run**; its
stated rationale is **corrected here** (§1.6): 3.5 is trained jointly for classification and regression (S7),
so "classification is its strong axis and regression is not" is not a supported premise. What *is* supported:
regression *is* classification over a bucket grid (S5, verified), and S5's `icdf` "inverts the piecewise-uniform
CDF exactly **within the model's bucket grid**… Empirical calibration on held-out data is still recommended" —
so a ladder we place ourselves gives resolution where the grid may not, and lets the tail be checked level by
level rather than read off one fixed rescaling.
**Targets.** `results/fleet/CALIBRATION.md` (the one-sided 14.7%/0.9% tail miss) and `results/fleet/COVERAGE.md`
(under-coverage at every level, z −4.8…−6.6). It does **not** target the 38.8% point-estimate rate.
**Cost.** Highest of the target-transform family: a different label vector per threshold means the KV cache
cannot be shared across thresholds, so ≈ 464 units × (number of thresholds) fits. At "the fit is ~5s"
(`docs/readiness.md` §0) a 9-rung ladder is **≈ 6 h CPU**; unattended, no tokens.
**Pre-registered expectation and falsifier.** Expectation: with a ladder whose top rungs sit at the model's own
95th-percentile scale, coverage at 90/95 lands **within 2 binomial SE (≈2.2pp at n=464) of nominal** *before*
widening. **Falsifier: the rebuilt distribution's 95% coverage stays ≤ 86%, or its pinball/CRPS loss is no
better than the recorded bar-bins route on the same units** — in which case the ladder buys nothing and A4 is
closed for good.

### R4 — Use the other columns: paid as a second scored target, paid/incurred and case-reserve ratios as features

**Idea.** Run the fleet harness on `CumPaidLoss` as a second column (paired against the incurred result), and
add the P/I ratio and the case-reserve share as features to the direct arm.
**Evidence.** S3 [**verified**]: add "ratios, interactions, group aggregations" the model "cannot derive from raw
columns alone"; S12 [**verified**]: `MunichAdjustment` "relies on the ratio of paid/incurred and its inverse",
`CaseOutstanding` is "useful when case reserves should inform paid ultimates"; our recount (§2): paid is fully
observed in the scored future on **460 of 464** units; `scripts/fleet_eval.py:36` already nominates
`CumPaidLoss` as "the robustness check".
**Targets.** `results/fleet/FINDINGS.md` (38.8%; the **variance** result — "it wins big sometimes and loses
bigger more often") and `results/fleet/COVERAGE.md` (the centring failure): if the failure is partly an artefact
of the incurred column's case-reserve noise, the paid column should show it — and the P/I ratio is the
documented signal for that noise.
**Cost.** **Free of design, data and tokens** — the harness already takes `--column`. ~**0.85 h** point-only
per column, ~1.7 h with draws.
**Pre-registered expectation and falsifier.** Expectation: on the intersection of units scoreable in both
columns, the paid arm's closer-than-CL rate is **not better than 38.8% + 2.3pp** and its one-sided tail miss is
**not below 14.7% − ~2pp**, i.e. **the failure is not column-specific** — *and* that the feature variant shrinks
the model's median move off CL below 141.8% without losing the paired rate. **Falsifier: if paid and incurred
differ by more than the noise floor in either metric, then the column choice is a research result in its own
right (and the entry's fleet result must be reported per column, not as one number).**

### R5 — Cross-triangle context (#10 / pre-registered E2b)

**Idea.** Give each fit in-context rows drawn from other triangles (guarded by R1–R5 in `docs/experiments.md`
§2), so the deviation from the prior is learned from tens of thousands of transitions rather than 6–33.
**Evidence.** Our own E2b pre-registration, with its leakage rules and a restated bar (§5.4). Vendor side:
nothing more than the mechanism — E7's check found **no documented row floor** and no documented remedy for our
scale, so this remains a test of **our** hypothesis, not of the vendor's fix (`results/conditions/FINDINGS.md`
§3).
**Targets.** `results/conditions/rows_per_fit.log` — **6–33 training rows per fit, median 25, none above 33** —
and through it the 38.8% rate.
**Cost.** Highest: build + the five guard tests + ≈2 h compute.
**Pre-registered expectation and falsifier.** Already fixed in §5.4: paired against the direct arm on the same
units, required to clear the noise floor; **the mandatory replication cell is `abc` → 5,211,802**, and if
**R3's cross-triangle null scores better than the self-fit, the guard is broken — stop, do not interpret.**

### R6 — Two-stage decomposition: predicted level × predicted shape

**Idea.** Split the target into an ultimate-level quantity (a loss ratio on premium) and a development-shape
quantity (%emerged), predict each, and multiply — the vendor's own hurdle analogue applied to a triangle.
**Evidence.** S6 [**verified** quote, **unverified** as a transfer]: the two-stage split measurably improved
both ranking and calibration in their pricing task (Gini 0.3063 → 0.3210; D² −0.156 → −0.023) at **678k
policies**;
our regime is 6–33 rows. S12 `IncrementalAdditive` (Schmidt 2006) is the actuarial form of the same factorisation.
**Targets.** `results/fleet/FINDINGS.md` (the variance of the correction) and `depth-bias/FINDINGS.md` (the
level, again). Strictly weaker evidence than R1 — which is why it ranks below it. The vendor's own measurement
is on a **claim-pricing task at 678k policies**, so this arm is a *transfer*, not a replication.
**Cost.** Build + ~1.7 h.
**Pre-registered expectation and falsifier.** Expectation: the product arm beats the single-stage on the paired
median |error| by more than the 1.6% draw-noise floor on the median. **Falsifier: no paired improvement → the
pricing-task decomposition does not transfer, and the transfer claim is recorded as falsified.** (Note: the
cookbook's own *measured* effect is on a different task, so this arm is a transfer, not a replication.)

### R7 — Thinking mode on the same 464 units

**Idea.** Run the vendor's documented strongest option for "grouped and time-ordered data" on the identical
units.
**Evidence.** S1/S2 [**verified**]: "Thinking mode spends more compute at fit time and is **the strongest option
for grouped and time-ordered data**"; S7 [**verified**]: the headline ranks belong to Thinking, and S2 `models`
[**verified**, availability column, quoted in E7] lists it as "Hosted API & `tabpfn-client`, VPC" only; E7
[**verified**]: our split is temporal, so this is the documented answer for our split type that we have not
tried.
**Targets.** `results/conditions/FINDINGS.md` §3-B2 — the documentation predicts **parity with tuned
conventional models on temporal/grouped splits**, and our 38.8% is the shape parity-at-best predicts. This arm
decides whether the negative belongs to *the OSS base checkpoint at defaults* or to the method.
**Cost.** Token metering on the hosted API, plus wire-up; no new design.
**Pre-registered expectation and falsifier.** Expectation: if the documented configuration for our split type
closes the gap, the closer-than-CL rate rises above **38.8% + 2.3pp** and coverage at 90/95 moves toward nominal.
**Falsifier: no movement beyond the noise floor → the negative is the method's in this regime, not the
checkpoint's, which is a materially stronger sentence for the entry.**

### R8 — The configuration cells already pre-registered as §5.1b / #22

**Idea.** Declare `origin_idx`/`dev_idx`/`cal_idx` categorical, and give the delta target
`quantile_uni_extrapolate` (one factor per cell; baseline cell must reproduce the recorded numbers).
**Evidence.** S1 "Column typing is the cheapest change and often the largest gain"; S2 "an undeclared integer
column with more than four distinct values is numeric" and the target-transform table; S8 [**verified, code**]:
both are accepted by the installed 9.0.0.
**Targets.** `results/fleet/CALIBRATION.md` (the one-sided tail) and `results/conditions/FINDINGS.md` §2.3-C2/C5
(the guidance we did not follow).
**Cost.** ~**1.7 h per cell**; 3 cells (baseline + each change) ≈ 5 h.
**Pre-registered expectation and falsifier.** Already written in §5.1b: "coverage at 90/95 should move **toward
nominal before any widening is applied**… If it does **not** move, the out-of-distribution-support explanation
is dead". Plus the replication cell: the baseline must reproduce the recorded direct-arm numbers or no cell in
the run is interpretable.

---

## 4. Free to test

**Free of compute entirely (seconds, no model call).**

1. **The paid column as a second scored target (R4, first half).** Same harness, `--column CumPaidLoss`,
   same units, same anchors — the existing paired design applies and `scripts/fleet_eval.py:36` already calls it
   the robustness check. What it would show if it worked: a closer-than-CL rate materially above 38.8% or a
   two-sided rather than one-sided tail miss would relocate the finding to the *column*, which changes what the
   entry's fleet result means. (Compute: ~50 min point-only; still no new design, no new sampling, no tokens.)
2. **The conditional-recalibration diagnostic (#21) on the recorded units**, without refitting anything: fit the
   correction on half the 464 units and test it on the other half, *conditioned on recorded fields the model
   already emitted* — horizon, triangle size, its own disagreement with CL, its interval width — and compare
   against the feature-permutation control that §5.3 specifies. `results/fleet/CALIBRATION.md` already showed the
   **un**conditional version is not a result (the placebo matched at 5.5% vs 4.3%); this is the discriminating
   version, and it costs minutes. What it would show if it worked: that the upper-tail miss is *unit-predictable*
   — the precondition for any formulation that lets the model learn the dependence itself.

**Free of design and tokens (paired, but real CPU).** R1, R2, R4, R8 all run on the 464 units already on disk
with no new sampling and the existing noise floor (draw noise ≈1.6% on the median at 300 draws; ±2.3pp on the
paired rate). R4's column change needs no new feature at all; R8 needs no new feature either — only a
declaration and a config key.

**The one measurement that should precede the family, and is free.** *How often is the correct answer outside
the range the model was shown?* For each of the 464 units: recompute the direct arm's **training labels** at the
anchor (`direct_training_rows(..., target="delta", known_until=anchor)`) and the **realised** delta target for
each scored cell, and count the units whose realised target exceeds the training-label maximum. This is
arithmetic on disk — no model, no fit, seconds. **Pre-registered expectation: if the one-sided tail miss is a
support artefact, the realised target exceeds the training-label maximum on a share of units comparable to, or
larger than, the 14.7% tail-miss rate. Falsifier: if the realised target lies inside the training-label range on
essentially every unit, then the answer is not outside the model's seen *target* range at all, the support
explanation dies, and R2/R3/R8 lose their stated mechanism — leaving only the feature/horizon kind of
extrapolation (R5, R6).** It is deliberately **not run here**: running it would spend the pre-registration it
exists to provide.

**Outcome — run 2026-09-19 after this report was written, 464 units in 15 seconds**
(`scripts/target_support_check.py` → `results/runs/20260919-target-support/support.jsonl`):

```
realised target ABOVE the training-label maximum :  21.1% of units   (pre-registered: >= 14.7%)
realised target BELOW the training-label minimum :   7.8% of units
scored cells above the training maximum          :   5.3% of cells
recorded 95% tail miss (for comparison)          :  15.3% of units
```

**The expectation is met, and the asymmetry points the same way as the failure**: the truth leaves the range of
training labels the fit was shown on **one unit in five**, and about **2.7× more often above than below** — the
same direction as the coverage defect (14.7% above the 95% bound against 0.9% below). So the fixed-grid
explanation **has support** and R2/R3/R8 keep their stated mechanism.

**Two honest qualifications, because the headline number is a maximum over cells.** First, the per-unit
statistic counts a unit as "outside" if **any** of its scored cells leaves the range, so it is inflated relative
to the per-cell rate — and the per-cell rate is **5.3%**, not 21%. The pre-registered per-unit framing is
reported as agreed, but the cell-level number is the honest measure of how often the truth is outside, and both
are stored. Second, support is not proof: the grid mechanism is verified in the source and in our installed
code, but the *correspondence between that grid and our tail miss remains our inference* — this raises its
plausibility and does not establish it. The discriminating test is R2: change the target's support and see
whether coverage at 90/95 moves toward nominal **before** any widening.

---

## 5. Not worth running, and why (the negative space)

| Idea | Why not |
|---|---|
| **Recursive diagonal-stepping (A3)** | **Already run and deleted** — it compounded the bias (`docs/experiments.md` §2), and the mechanism now explains why: each step multiplies a per-step bias of +3.2% into a level the model has never seen (`depth-bias/FINDINGS.md`). Re-running it with a better target is R1/R2; re-running it as-is is repetition. |
| **Sample-weighting / exposure-weighted fits** | **Unavailable**, verified: `TabPFNRegressor.fit(self, X, y)` takes no weights, and S6 says the same ("TabPFN has neither"). Any design that relies on weighting must be reformulated as *features* (R1/R4) — and the vendor's own workaround is exactly that. |
| **Row subsampling / `"majority_downsample"`** | Documented for tables above the row limit and for zero-heavy targets (S2); ours are 6–33 rows with no zero mass in cumulative figures. It cannot bite. |
| **Exposure-normalised *targets* (pure-premium-style)** | **Documented as a trap** by the vendor's own cookbook: they modelled the bounded amount rather than the per-exposure one because "dividing a claim by a tiny exposure blows up the tail" (S6). Exposure belongs in the features and the prior, never as a divisor on a target whose tail is under test. |
| **Widening the intervals to fix coverage** | Runs, works, and **is not a result** — already measured (`results/fleet/CALIBRATION.md`: 11.8% → 4.3% mean gap with a 2.3–2.7× upper widening, and a placebo at 5.5%). It is arithmetic on widths, not evidence about the model, and it changes no claim. |
| **A5 as a feature study (raw cells vs identifiers vs link ratios vs calendar effects)** | Largely subsumed: our features are *already* the domain ratios the vendor's step 2 asks for, and the identifier question is exactly one cell of #22 (declare them categorical) rather than a family. Calendar effects cannot be reached by adding a number — S2 states an undeclared integer with >4 values is numeric — so the only documented routes are the datetime path (we have no date column, only indices) and Thinking mode (R7). |
| **Fine-tuning** | Parked with a reason (`docs/experiments.md` §5.5): GPU hours, and it contradicts the entry's zero-shot claim. S1 lists it for "a specialized domain or distribution shift", which is our situation — but it changes what the entry *is*. |
| **Feature selection, polynomial features, estimator-count tuning, SVD** | S2 scopes these to wide tables (thousands of columns) and to interactions not known in advance; our table is 12 columns × ≤33 rows. Low expected value; not free in time (S2's own step 8: a change is kept only if it holds across several seeds/splits). |
| **Blending the model with CL/Mack/bootstrap (#12)** | `docs/experiments.md` §5.5: it makes the model a meta-learner, which the rubric punishes at 50% weight. |
| **Re-tuning A4's rationale as "classification is the model's strong axis"** | **Corrected** (§1.6): 3.5 is trained jointly on both tasks (S7) and leads a distributional benchmark. Use the ladder for grid placement, not for a claim the sources do not support. |

---

## 6. Corrections this pass forces on our own writing

1. **`docs/experiments.md` §2/E2-A4's rationale** — "this project's own house finding that a count target
   reframed as classification moves TabPFN from its weakest axis to its strongest" — has no source and is
   contradicted for 3.5 by S7 (joint training) and ScoringBench (first of 52 on distributional scoring).
   Replace with the mechanism that *is* documented (fixed bucket grid; thresholds we place).
2. **This task's own phrase "bulk/case loss"** — `BulkLoss` is the IBNR/bulk component (S11 + our recount:
   the identity holds on 6.3% of cells), not the case reserve. The case reserve is `IncurLoss − CumPaidLoss`.
3. **"No feature engineering"** was already corrected by E7; this pass adds the *specific* missing features the
   vendor's own cookbook uses — exposure as an input feature — and the fact that **weighting is not available**,
   which is why it must be a feature.
4. **The untapped data is real and quantified**: `CumPaidLoss` scoreable on 460/464 units, premium complete and
   non-zero on 464/464, `BulkLoss` a usable *feature* on 454/464 but a scored target on only 88/464.
5. **The regime exclusion now has a second, higher floor**: TabArena leaves "few-shot predictions, or very small
   data (e.g., less than 500 training samples)" for future work (S10), alongside BeyondArena's <100 (S9/E7). Our
   6–33 rows sit below both, in the benchmarks the vendor's ranks are computed on.

---

## 7. What could not be verified

- **The cookbook's numbers** (S6: Gini 0.32, the ~0.6 → 0.96 balance, the Tweedie deviances). Quoted verbatim,
  never re-run here; and it is a *different task* at 678k rows, so it is evidence about the mechanism, not
  about reserving. Marked **unverified as a transfer** wherever used (R6).
- **Whether our prediction inputs are "well outside the training range"** in the sense S5 means. Not measurable
  without fitting (this task forbids it); the free measurement in §4 is the closest honest substitute and is
  deliberately left unrun.
- **Whether 3.5 carries TabPFN-3's o.o.d.-compatible preprocessing behaviour** — E7's open item, unchanged.
- **The hosted variants** (Thinking/Plus/Fast): no cell in this report is measured on them; their ranks are
  quoted from S7 as reported, not reproduced.
- **Any cause-and-effect between the fixed bucket grid and our 14.7% tail miss.** S5 documents the grid and the
  missing OOD flag; our measurement documents the tail. The link is **ours and inferred**, and R2/R3/R8 are the
  tests that would convert it into evidence — which is why each carries a falsifier that kills it.

---

## 8. Provenance

- This report: `results/remodel/FINDINGS.md`. Reading and analysis only; **no model was fitted and no existing
  code was changed**.
- The one new number set: `results/remodel/column_availability.log`, produced by
  `.venv/bin/python results/remodel/column_availability.py` — a read-only recount (same `regular_block()`
  window, same anchors, same 464 units as `scripts/fleet_eval.py`; counts finite cells per column, predicts
  nothing).
- Every other number of ours is quoted from `results/fleet/FINDINGS.md`, `results/fleet/COVERAGE.md`,
  `results/fleet/CALIBRATION.md`, `results/fleet/coverage.log`, `results/runs/20260918-023600_depth-bias/FINDINGS.md`,
  `results/runs/20260919-pricing/pricing.md`, `results/conditions/FINDINGS.md`, `results/conditions/rows_per_fit.log`,
  `docs/experiments.md`, `docs/readiness.md` and `scripts/fleet_eval.py:36`.
- Environment checks (`tabpfn` 9.0.0 and `chainladder` 0.10.1 signatures, registries and docstrings) were made
  against the pinned `.venv` at `/Users/Scott/projects/tabpfn-reserving/.venv`.
- External sources were fetched 2026-09-19; the report PDFs were read as text extracted from the vendors'
  and authors' own files, and the docs pages as their own markdown source.
