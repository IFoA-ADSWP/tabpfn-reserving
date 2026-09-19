# The goal, and the experiments that get us there

> **Status:** design note, 2026-09-17. Written before any run. It fixes what we are trying to establish, which
> experiments establish it, and which creative uses of the model we have decided to test rather than admire.
> Bars and falsifiers live in [`method.md`](method.md); schedule and roles in [`brief.md`](brief.md).

## 1. The goal, at three altitudes

**As a product.** Point the repository at a loss triangle and get a reserve *with a usable distribution*,
from a single forward pass — against the method the industry uses, which reaches the same distribution from a
thousand simulated refits. One command, public data, reproducible.

**As a scientific question.** Not "can a foundation model reserve?" — that invites a single cherry-picked
triangle. The question is: **is a tabular foundation model's predictive distribution calibrated for the
reserving problem?** The point estimate is the easy half and the industry already has a good answer for it.
The uncertainty is the half that matters, because a risk margin, an IFRS 17 risk adjustment and a Solvency II
capital figure are all *quantiles*, and the incumbent quantiles are known to be optimistic.

**As a judged entry.** A judge should conclude three things in five minutes: this uses TabPFN-3.5
prominently and natively; the reframing is a genuine idea rather than a wrapper; and the numbers could be
reproduced by a stranger. The distributions, the fleet and the honest failure case are how those three
conclusions get earned.

## 2. The experiments

Staged so each one is worth running on its own, and gated so a surprise costs days rather than weeks. Every
arm is compared on the **same folds** — the triangle analogy of this project's house rule that a comparison
is only as good as the folds being shared.

### E0 — Calibrate the instrument, before claiming anything

Cheap, and it is the difference between a measurement and a story.

| Arm | What it establishes |
|---|---|
| Same config, twice | Run-to-run variation. Without it, a small edge is indistinguishable from noise |
| **Placebo: shuffled targets** | The reserve should collapse toward a naive level. If a shuffled target still produces a plausible reserve, the harness is leaking and every later number is void |
| Anchor-time assertion | Every feature row is checked to use only cells with `origin + development ≤ as-of`. Printed with each result, not assumed |
| Naive baselines | Reserve = last observed cumulative (no development); reserve = volume-weighted factor applied flat. Any model that cannot beat these has nothing to say |

### E1 — Head-to-head on the classics (the MVP experiment)

| | |
|---|---|
| Objects | `abc`, `genins`, `mcl`, `ukmotor` — small, well-understood, published examples |
| Arms | Chain Ladder · Mack · ODP bootstrap · TabPFN-default |
| Design | **Every diagonal is a valuation date.** Predict the remaining reserve as of each one and score against what actually happened — so one triangle yields many observations instead of one |
| Metrics | Reserve error (per accident year and total), rank, and the sign of the error |
| Settles | Parity on the point estimate, and whether any edge is bigger than E0's noise |

### E2 — Mechanism: which formulation predicts best (the intellectual core)

This is the experiment that turns a demo into research. Same data, five ways of asking the question:

| Arm | The idea | Plays to |
|---|---|---|
| **A1 — amounts** | Predict each future cell's cumulative amount directly | The obvious baseline |
| **A2 — link ratios** | Predict the *development factor* `C_{a,d+1}/C_{a,d}` instead, then apply it | The estimand actuaries actually trust; bounded target; comparison to Chain Ladder becomes direct |
| **A3 — recursive** | Predict one diagonal ahead, append it to the context, predict the next | How an actuary projects; the far cells become a sequence of near cells |
| **A4 — distributional classification** | Predict `P(development exceeds threshold)` for a ladder of thresholds; rebuild a distribution from the survival function | This project's own house finding that a count target reframed as classification moves TabPFN from its weakest axis to its strongest |
| **A5 — features** | Raw cells vs origin/development only vs explicit link ratios vs calendar-year effects | Tests our own claim that no feature engineering is needed — and recovers *what the model is using* |

A5's interpretability arm matters beyond performance: Chain Ladder **imposes** no accident-year effect by
construction. If the model finds one, that is a result an actuary can use, and it is the kind of finding that
reads as originality rather than as a leaderboard.

### E2b — the fleet as context (promoted from parked after run 2)

**Why this exists.** The Δ arm showed that no deviation from Chain Ladder is learnable from 6–36 rows: each
fit sees one triangle's cells. The deviation has to be learned from *many* triangles, which is in-context
meta-learning's whole mechanism. Idea #7 in §3 was parked on leakage grounds; run 2 makes it the leading
candidate for the point estimate.

**Design.** A fit for triangle T receives its own observed cell-transitions **plus a context sample of
transitions drawn from other triangles in the database** — features and targets, as in-context examples.
The model predicts T's unknown cells as before. Everything else (anchors, scoring target, arms) is unchanged,
so a result is comparable to runs 1 and 2.

**The leakage rules, which are the whole design.** Without them this produces beautiful, meaningless
numbers, and they are why it was parked:

| Rule | Statement |
|---|---|
| R1 | No context row may come from T itself. |
| R2 | **Every context transition must be observable as of T's anchor** — its calendar valuation date must be at or before the anchor being predicted. This is what makes the arm legitimate rather than clever: information from other insurers up to the same calendar date is exactly what a reserving actuary actually has (industry statistics, market data). Anything later is the future, whoever it belongs to. |
| R3 | **A cross-triangle null**: rerun with T's own rows removed from the context entirely. This is pure cross-triangle prediction — if it scores well, R1 and R2 are holding; if it scores *better* than the self-fit, something is wrong with the guard and the whole arm is void. |
| R4 | The shuffled-target placebo, applied to the context rows. |
| R5 | The anchor-time assertion already in the harness, extended to assert every context row's date ≤ anchor. |

**Pre-registered bar, fixed before the run.** The arm is worth keeping only if it (a) beats the raw-ratio
arm's median absolute error of **14.9%**, and (b) does not lose to the Chain Ladder arm on the mean — i.e.
it inherits no blowups. Failing either, it is reported and dropped, not re-tuned until it passes.

**Data at hand — checked, not assumed.** `chainladder` bundles the CAS Loss Reserve Database as `clrd`:
**775 triangles**, indexed by (GRNAME, LOB) — insurer-groups across six lines of business — each 10 origins ×
10 developments, carrying incurred loss, cumulative paid, bulk loss and earned premium. `clrd2025` is the
2026 refresh with ten more accident years, and there are ~40 further single-triangle sets (`raa`, `usauto`,
`mack_1997`, the `friedland_*` family) for extra test cases. So the context available to one fit goes from
**6–36 rows to tens of thousands of transitions** — precisely the sample size run 2 identified as missing,
and enough to power E4's regime map at the same time.

**What it would mean if it works:** the first evidence that cross-triangle context — the fleet — is what a
tabular foundation model needs to beat the actuarial standard on its own home ground, and a genuine
"showcase a harness / formalize a new problem" result rather than a domain demo.

### E3 — The distribution, which is the point

| Arm | What it establishes |
|---|---|
| Coverage at 50/75/90/95% over the fleet, TabPFN vs Mack vs ODP | Whether the intervals are honest, and whether they are *more* honest than the incumbent's |
| CRPS / pinball loss per cell | A proper distributional score — coverage alone rewards being wide |
| **Joint vs independent sampling of cells** | A subtlety with teeth: summing independently-sampled cell quantiles ignores the correlation between cells that Mack models explicitly. Predicting the whole future diagonal in one fit lets the shared context induce dependence. Testing this is the difference between a distribution and an interval arithmetic trick |
| Interval width vs Mack's | Narrower and better-covering is the win; wider and better-covering is just conservative |

### E4 — The fleet: where it wins, and the decision rule

The CAS Loss Reserve Database through `chainladder`: hundreds of real insurer triangles across six lines of
business. Sliced by triangle size, tail length, line of business and valuation age — producing the same kind
of adoption rule the working party's insurance benchmark produced ("use it when …; prefer the incumbent
when …"). This is what gives E1–E3 statistical weight rather than anecdote.

### E5 — Cost, and the workbench claim

Per-triangle wall-clock against the ODP bootstrap on the same machine; fleet throughput; the **Fast
checkpoint** arm; and **KV-cache reuse**, which is the interesting one — fitting once on the observed cells
and evaluating many query cells (every future cell, every valuation date, every triangle) is exactly the
pattern `fit_with_cache` exists for.

### E6 — Where it fails, and whether compute helps

The worst triangles and cells, stated in the open. Plus one targeted arm: **Thinking mode on the late and
thin cells**, where signal is weakest and the tail is decided. If inference-time compute buys calibration
where data is scarce, that is a genuinely new result; if it does not, that is worth one line and no more.

### Not running (parked, with reasons)

Fine-tuning (GPU, hours, and it contradicts the entry's claim) · individual-claim reserving (different public
data problem) · a Kaggle-style accuracy chase (crowded, and the model is not the subject).

## 3. Creative uses of TabPFN — ranked, with verdicts

The model is not a drop-in regressor here. What follows is every genuinely distinct way its capabilities
could meet this task, scored on whether it shows the model off *and* is defensible actuarially.

| # | The idea | Why it is creative | Verdict |
|---|---|---|---|
| 1 | **The distribution as the product** — sample the cell-level predictive distribution and sum | The industry retires a *point estimate* and buys the distribution separately with simulation; here it falls out of the same call | **Headline** (E3) |
| 2 | **Joint rather than marginal sampling** — one fit over the whole future diagonal so the shared context induces dependence | Nothing in the naive version gets correlation right, and Mack explicitly models it. This is where a foundation model could beat a formula rather than imitate it | **In** (E3) — highest scientific value per hour |
| 3 | **Distributional classification** — a ladder of "exceeds threshold" binaries, distribution rebuilt from the survival curve | Uses this project's own discovered lever (classification is the model's strong axis; regression is not) on a target that is stubbornly continuous | **In** (E2/A4) |
| 4 | **Predict link ratios, not amounts** | Puts the model in development-factor space — the estimand actuaries trust — so the comparison to Chain Ladder is apples-to-apples and the output is bounded | **In** (E2/A2) |
| 5 | **Recursive diagonal-stepping** — each predicted diagonal becomes context for the next | Mirrors how a real projection unfolds, and converts a long extrapolation into a series of short ones | **In** (E2/A3) |
| 6 | **KV-cache as the fleet engine** — fit once per triangle, evaluate every query cell, diagonal and re-valuation cheaply | The workbench claim ("re-reserve the book every quarter") stops being rhetorical | **In** (E5) |
| 7 | **The fleet as context** — give the model other triangles' summaries, so cross-insurer information informs this insurer's development | In-context meta-learning is the model's defining mechanism, and reserving has never had a way to borrow strength across a book of triangles | **Park** — highest ceiling, highest leakage risk. Revisit as an explicitly controlled experiment if E1–E3 land early |
| 8 | **Triangle embeddings** — embed and cluster the fleet by development pattern | A map of the fleet is both a demo asset and a defensible basis for E4's regime slices, better than slicing on size alone | **Stretch** — cheap to try on the fleet, and visually strong |
| 9 | **Anomaly detection as an audit tool** — unsupervised flagging of odd cells and diagonals | Shows a capability nobody brings to a hackathon; a real actuarial workflow (someone eyeballs every triangle today) | **Stretch** |
| 10 | **Calendar-year trend as a time series** — forecast the diagonal (inflation, court awards) and feed it back as a feature | Uses a *different* model in the TabPFN family to attack the gap every classical method has: calendar-year effects | **Park** — one arm, easy to bolt on if E2 shows calendar effects matter |
| 11 | **Synthetic tail generation** — use the generative capability to stress-test methods where real data is thinnest | Attractive, but risks reading as method-for-method's-sake | **Park** |
| 12 | **Model blending** — let TabPFN learn to weight Chain Ladder, Mack and the bootstrap | Practical, and a likely accuracy win — but it makes the model a meta-learner rather than the subject, which the rubric punishes at 50% weight | **Park** |

**The three that earn their place on the science as well as the demo:** joint sampling (#2), distributional
classification (#3), and link-ratio prediction (#4). Each is a different answer to "what is being predicted",
each is cheap, and E2 measures them against each other rather than asserting one.

## 4. What this buys, stated as the judge's five minutes

1. A reserve, a distribution, and a wall-clock — against Chain Ladder, Mack and the ODP bootstrap (E1, E3, E5).
2. The reframing is not packaging: four formulations of the same question, measured (E2).
3. It is a measurement, not a pitch: a placebo arm, shared folds, pre-registered bars, and the failures
   published next to the wins (E0, E6).

---

# 5. Stage 2 — the programme after the first results

> **Status: written 2026-09-19, after E1–E4 and the fleet ran.** §2 above is unchanged: it is the
> pre-registration, dated before any run, and it stays legible as such. This section is what the results
> changed. Findings it builds on: `results/fleet/FINDINGS.md` (the point estimate loses),
> `results/fleet/COVERAGE.md` (the intervals under-cover at every level, one-sided at the tail),
> `results/fleet/CALIBRATION.md` (widening repairs coverage; nothing *learnable* was found),
> `results/runs/20260918-023600_depth-bias/FINDINGS.md` (the mechanism).

**The question, in one sentence.** Which of these three methods — TabPFN-3.5, Mack's method, the ODP
bootstrap — produces intervals that actually cover on real loss triangles, and what does each one cost in
width to do it?

That sentence is now the whole programme. Everything below bears on it, and anything that does not is out of
scope and named in §5.5.

## 5.1 E7 — the conditions check: is our negative inside the vendor's own envelope?

**Cheapest step, first, because it can change what the entry claims rather than merely adding to it.** Our
negative so far has been explained by our own hypothesis (40–60 training rows as then quoted — recounted to
6–33 since, `results/conditions/rows_per_fit.log`). But a negative is
only informative once it is checked against the conditions the *source* documents: if Prior Labs' own guidance
says the model does not extrapolate, or states a row count it needs, then our failure is **predicted by the
documentation** — and the entry's claim changes from "we tried a model and it did not work" to "we reproduced
the documented envelope limit on a new domain", which is a stronger and more useful sentence.

| | |
|---|---|
| Method | Read the primary sources — the TabPFN-3.5 technical report and the current docs, at source, not via summaries. Tag every claim **verified / unverified / corrected**, and keep the external-evidence record separate from our own results |
| Conditions list | Both directions: where the source says the method succeeds, and where it says it fails or is unlikely to help — including operational guidance and the vendor's own defaults |
| Verdict per condition | **Fits** (predicts our negative) · **predicts the opposite** (we meet a success condition — the most valuable line in the report) · **ruled out** (our setup is outside it) · **outside the evidence** (our regime is not covered at all) |
| Cost | ~1 hour, no compute |
| Claim if we stop here | "Our negative on this domain is / is not explained by the method's own documented conditions, at a scale those conditions do / do not cover" |
| Stop rule | If the source documents a row-count floor above ours, or excludes extrapolation, then E2b is **promoted**: it becomes the test of the vendor's stated fix rather than a speculative arm |

**Outcome — ran 2026-09-19, `results/conditions/FINDINGS.md`, 23 minutes, reading only.** **Neither trigger
fired.** No source states a training-row floor above ours and none excludes extrapolation, so the stop rule did
**not** promote E2b: it remains a test of *our* hypothesis, not of a documented remedy. Where our profile lands,
3 / 3 / 2 / 3 across the four buckets:

- **fits (3)** — the **split type**: on temporal and grouped splits tuned conventional models retain the highest
  performance and TabPFN-3.5 only matches them at non-large scale; on small datasets a different foundation
  model leads; and the docs warn that for inputs outside the training range the *distribution itself* is not to
  be trusted and the model will not flag it — which is the one condition that names our one-sided tail failure.
- **predicts the opposite (3)** — the vendor names *claim amounts* as a target it improved distributions for;
  it documents **extrapolation** as a capability; and it publishes a cluster of guidance for exactly our regime
  that we did not follow (column typing, the extrapolating quantile transform, Thinking mode for time-ordered
  rows, calibration checking on skewed targets).
- **ruled out (2)** — the size ceilings (~1M rows) cannot bear on us; the headline ranks belong to a
  configuration we did not run.
- **outside the evidence (3)** — the decisive one: the vendor's non-i.i.d. evidence base spans **100–1M rows**
  and declares sub-100-row prediction *out of scope*. Our 6–33-row fits are **unreached** by it.

**Two corrections it forced on the entry**, both applied: the entry had been quoting **"40–60 training rows per
fit"** (a single-triangle figure — the fleet ran **6–33, median 25**), and **"no feature engineering"** stops
being a free claim, since the docs call column typing *"the cheapest change and often the largest gain"* and
warn specifically against passing an identifier as a number.

## 5.1b E8 — the configuration arms (added by the conditions check)

**The sources name two changes for our regime that we did not make, and both run on units already on disk** —
so each is **paired** against the result we already have: no new design, no new sampling, the existing noise
floor applies, and neither costs tokens. This is the cheapest substantive arm in the programme, which is why it
now sits here.

| | |
|---|---|
| Arms | **(a)** declare `origin_idx` / `dev_idx` / `cal_idx` **categorical** rather than leaving them as floats in a NumPy array — the docs' cheapest-and-largest-gain lever, and they warn specifically against replacing an identifier with a number; **(b)** give the delta target the **`quantile_uni_extrapolate`** transform, since the default *clips* values outside the training range and our entire failure mode is extrapolation beyond it |
| Isolation | One factor per cell: baseline (current features) · +categorical · +extrapolating transform · both. Same 464 units, same anchors, same scoring as `results/fleet/FINDINGS.md` |
| Pre-registered expectation, **written from the sources before the run** | If the one-sided upper-tail miss (14.7% above the 95% bound against a 0.9% miss below) is a fixed-support artefact of clipped targets, coverage at 90/95 should move **toward nominal before any widening is applied**. If it does **not** move, the out-of-distribution-support explanation is dead and the miscalibration is something else — which is a usable result either way |
| Replication cell | The baseline cell must reproduce the recorded direct-arm numbers on the units it covers. If it does not, the harness changed and **no** cell in the run is interpretable |
| Cost | Minutes, local, no tokens |
| Claim if it moves | The failure becomes *configuration-specific*: the OSS base checkpoint at defaults loses, and a documented two-line change recovers part of the gap |
| Claim if it does not move | The failure survives the vendor's own recommended settings for our regime — a materially stronger negative, and one that closes the most obvious review question ("did you try what the docs say?") |

## 5.2 E3a — three-method coverage on identical units (#20)

E3 already names this arm. It is now the hinge: every claim the entry currently makes is *about our model*,
and this is the only step that can make any claim *comparative*.

| | |
|---|---|
| Objects | The same 464 coverage units, same column (`IncurLoss`), same held-out diagonals, same realised futures — already on disk, so the comparison is paired rather than three separate rates |
| Arms | TabPFN-3.5 (measured) · Mack (`clchainladder.MackChainladder`) · ODP bootstrap (`BootstrapODPSample`) — both present in the pinned environment, v0.10.1 |
| Metrics | Empirical coverage at 50/75/90/95 with binomial intervals · **paired** per-unit: does the truth fall outside Mack's interval on the same units where it falls outside ours · interval width relative to the point, per method |
| Price first | **Measure one unit before designing the run.** Mack is analytical and instant; the bootstrap is a simulation and could cost seconds or minutes per triangle. Price 1 unit, multiply by 464, and only then commit |
| Pre-registered decision rule | Judge a gap real when it exceeds **2 binomial SE** (≈2.2pp at n=464). Two outcomes, both publishable: **(a)** Mack/ODP within 2 SE of nominal while ours is 5–7 SE off → the comparative claim is earned: *this method's intervals are wrong where the standard one's are right, and here is how*. **(b)** All three outside 2 SE → the finding stops being about TabPFN and becomes about the field: *reserving intervals do not cover as advertised on real triangles, including the classical ones* |
| Cost | Pricing + ~2–4 hours compute, unattended |
| Claim if we stop here | One of (a) or (b) above, plus the one-sided tail finding: the failure is a too-short upper tail, not general sloppiness |
| Stop rule | None — it is a measurement, and both branches are usable. It **gates** E9 and E10 |

## 5.3 E9 — can the model flag where the classical intervals fail? (new claim, cheap)

Only reachable once 5.2 has per-unit Mack intervals. **This is the one arm that could give the model something
it wins at**, and it plays to the entry's existing strongest section (knowing when not to trust it).

| | |
|---|---|
| Question | Does anything the model emits — its disagreement with Chain Ladder, or its interval width — predict the units where **Mack's own interval misses**? |
| Design | Rank correlation between the model's signal and Mack's miss indicator on the same units, with a **feature-permutation control** (preserves each signal's marginal, destroys its pairing with the outcome) — the control that discriminates, unlike the shuffled-target placebo in `CALIBRATION.md` |
| Pre-registered rule | Claim it only if the correlation's 95% interval excludes zero **and** the permutation control is flat. Report the paired difference, not two rates |
| Cost | Minutes on data already on disk |
| Claim if it passes | "Not a better reserving method, but a detector for where the standard method's uncertainty is unreliable" — a positive result, and novel |
| Stop rule | If the correlation interval includes zero, record one line and drop it. No re-slicing until it passes |

## 5.4 E2b — the fleet as context (#10), gated last

Already pre-registered in §2 above, including the five leakage rules and the bar. **Two amendments, both
forced by the fleet results:**

1. **The bar is restated in fleet terms.** E2b's original bar (beat 14.9% median |error|; do not lose to Chain
   Ladder on the mean) was set on the n=11 spike. Against the fleet it must instead be paired against the
   direct arm **on the same units**, with the effect required to clear the measured noise floor (draw noise
   ≈1.6% on the median at 300 draws; the "closer than Chain Ladder" rate is 38.8% ± 2.3pp).
2. **One replication cell is mandatory.** Re-run the direct arm on `abc` inside the same run and require
   **5,211,802** — a known prior result. If it does not land, the harness changed and no other cell in the run
   is interpretable. One arm, free.

| | |
|---|---|
| Isolation | **One factor varies**: the context contents (none · same-line-of-business neighbours · random triangles). Anchors, targets, arms and scoring are held fixed at the fleet evaluation's settings, so every cell is directly comparable to a number we already have |
| Cost | Build, then 464 units × ~13s ≈ 2 hours compute, unattended; plus the R1–R5 guard tests |
| Claim if it passes | A competitive point estimate, and the entry's headline changes: the fleet as context is what the model needed |
| Claim if it fails | **The negative becomes definitive** — the diagnosis was tested with the fix it implies, and it did not hold. Worth more than the arm that would have been run instead |
| Stop rule | If the R3 cross-triangle null scores *better* than the self-fit, the guard is broken: stop, fix, do not interpret |

## 5.5 Not doing (named, so they can be declined explicitly)

- **Conditional recalibration (#21)** until 5.2 exists — a widening cannot be judged without Mack's width on
  the same units. It is minutes of work, gated, not forgotten.
- **Joint sampling / correlation structure (#15)** — a real gap, and it changes no current claim.
- **The regime map (#11)** — slicing a failing method by regime is premature; 5.2 and 5.3 set the regime
  question properly.
- **Fine-tuning** (GPU hours, and it contradicts the entry's claim) · **blending with the classical methods**
  (idea #12: it makes the model a meta-learner, which the rubric punishes at 50% weight) · **any further
  single-triangle tuning** (`abc`, `genins` and `ukmotor` are characterised; more of it is repetition).

## 5.6 Claims, and what stopping buys

| Stop after | The claim, stated as a sentence a stranger can check |
|---|---|
| 5.1 | Our negative is (or is not) explained by the method's own documented conditions, at a scale those conditions do (or do not) cover — **answered: it is not, and our regime is unreached by their evidence base** |
| 5.1b | The failure survives — or does not survive — the vendor's own recommended configuration for our regime (categorical identifiers, the extrapolating transform), which closes the "did you try what the docs say?" question |
| 5.2 | This method's intervals fail to cover at every level; the standard method's do / do not, measured on the same units — and the failure is a too-short upper tail |
| 5.3 | The model's disagreement with Chain Ladder flags where Mack's intervals are unreliable (only if the control is flat) |
| 5.4 | Cross-triangle context does / does not make the deviation learnable — the last arm the diagnosis implies |

**The recommendation, and the ask.** Run **5.1 then 5.2**. Together they are about a day, they need no new
model code, and between them they decide which entry we are submitting — a comparative result, or a finding
about the field. 5.3 is minutes after 5.2 and is the only arm that could hand the model a win. 5.4 is the
biggest build and the only path to a competitive point estimate, and it is better run *after* 5.1 has said
whether the vendor's own documentation predicts our negative.

**Not part of this:** any spend. Every step runs on the local CPU, on data already in the repository.
