# E7 — the conditions check: is our negative predicted by the vendor's own documentation?

> **Status:** written 2026-09-19, reading only. No model was fitted for this report, no existing code was
> changed, and every number of *ours* below is quoted from a file in `results/` (the one new number — the
> training rows per fit — is recomputed from `results/fleet/clrd-IncurLoss.jsonl` by a read-only recount,
> `results/conditions/rows_per_fit.py` → `rows_per_fit.log`).
>
> **The one-sentence answer.** No source we read documents a training-row **floor** above ours, and no source
> excludes extrapolation — the two things that would have made our negative a *predicted* outcome. What the
> sources do document is that on **temporal and grouped splits** conventional tuned models retain the best
> performance and TabPFN-3.5 only *matches* them, and that on **small datasets** it is not the leader. Those
> are conditions we meet, so our negative is **partly predicted by the documentation — by the split, not by the
> sample size** — while the regime we actually ran (6–33 training rows per fit) sits **below the floor of the
> vendor's own non-IID evidence base**, which declares sub-100-row prediction out of scope. The entry's claim
> therefore changes in a narrower way than the pre-registration hoped: not "we reproduced the documented
> envelope limit", but "the documented envelope does not cover us, and the parts of it that do point the same
> way as our result".

---

## 0. What was read, and how

Every quote below was taken from the primary pages, fetched 2026-09-19, and is reproduced verbatim (including
the vendor's own typography). Nothing here is taken from a summary, a blog post, a benchmark aggregator, or
recollection. The PDF's text was read as a text-converted rendering of the vendor's own file; the docs pages
were read as the vendor's own markdown source (the `docs.priorlabs.ai/<page>.md` route their docs host serves).

| # | Source | What it is | Why it is in scope |
|---|---|---|---|
| S1 | `TabPFN-3.5: Technical Report`, Prior Labs Team, 14 Sep 2026 — `https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf` (linked from `https://priorlabs.ai/technical-reports/tabpfn-3-5`) | 33 pages, the model report | The vendor's own account of what 3.5 does and where it was evaluated |
| S2 | Prior Labs documentation — `https://docs.priorlabs.ai/` — pages `models`, `faq`, `improving-performance`, `improving-performance/preprocessing`, `improving-performance/feature-engineering`, `capabilities/predictive-distribution`, `capabilities/thinking-mode`, `capabilities/regression`, `capabilities/fine-tuning`, `changelog/tabpfn-3.5` | The vendor's current operational documentation | Operational guidance, stated defaults, and limits — the task asks for these, not only paper findings |
| S3 | Prior Labs OSS repository — `https://github.com/PriorLabs/TabPFN` (raw `README.md`, `main`) | The package we actually ran | "How to get the best results" is stated here |
| S4 | `TabPFN-3: Technical Report`, arXiv `2605.13986v2` (May 2026) | The predecessor report, cited as `[1]` by S1 | The extrapolation claim lives here, not in S1 |
| S5 | `Beyond IID: How General Are Tabular Foundation Models, Really?` (BeyondArena), arXiv `2606.30410` | The benchmark S1 cites as `[3]` for its non-IID claims | Its **scope statement** is what decides whether our regime is covered at all |

Tags used throughout: **verified** (read at source; the quote is reproduced), **unverified** (the source says
it, but it is a claim we cannot check from here — a customer quote, a slice definition, a benchmark result we
did not re-run), **corrected** (a statement that needed narrowing to stay inside its source).

Two errors the tagging is meant to catch, both checked for explicitly and **not** found in the sources:
a limit quoted in *rows* when the source says *cells* (every size limit in S1–S3 is stated in **rows × columns**;
the word "cells" appears in S1's architecture section and in one figure label, and never as a limit), and a numeric
threshold attributed to Prior Labs that Prior Labs never published (**no** minimum-sample-size number exists in
any source we read; see §3, C1).

---

## 1. Our profile — the thing being checked

External evidence is worthless against an unstated profile, so it is stated here first, entirely from files.

| | Our setup | Where it comes from |
|---|---|---|
| **Task** | Per-cell development factors on a cumulative loss triangle; the reserve is the sum over accident years | `src/tabpfn_reserving/triangle.py` (`direct_features`, `direct_predict_rows`); `docs/method.md` |
| **Scale** | **6–33 training rows per fit, median 25**, over 464 scored evaluations; 0 units above 40 rows | `results/conditions/rows_per_fit.log` (recount of `results/fleet/clrd-IncurLoss.jsonl`) |
| **Target** | The development factor *relative to Chain Ladder's own implied factor* (`target="delta"`) — a positive, skewed, bounded ratio | `scripts/fleet_eval.py`; `results/fleet/FINDINGS.md` |
| **Features** | 10 numeric columns: `origin_idx, dev_idx, cal_idx, latest_cum, log_latest_cum, own_last_ratio, global_factor_prev, global_factor_available, horizon, log_chainladder_factor` — all passed as a NumPy float array, none declared categorical | `src/tabpfn_reserving/triangle.py` (`FEATURES`, `DIRECT_FEATURES`) |
| **Split** | **Temporal**: the last three diagonals are held out, and every training label is clamped to the evaluation anchor so no future cell is ever a training label | `scripts/fleet_eval.py` (`known_until=anchor`); `results/fleet/FINDINGS.md` §"one honest caveat" |
| **Horizon** | 1–3 steps in the fleet (production asks for 9) | `results/fleet/FINDINGS.md`; `results/runs/20260918-023600_depth-bias/FINDINGS.md` |
| **Model** | OSS `tabpfn` **9.0.0** (the TabPFN-3.5 family), `TabPFNRegressor(device="cpu")`, **default inference configuration**, no tuning, no Thinking mode, no fine-tuning | `requirements.txt`; `src/tabpfn_reserving/arm.py` (`make_model`); `docs/method.md` "no hyperparameter to set" |
| **Distribution** | 300 inverse-CDF draws from the model's own bar distribution per unit | `results/fleet/COVERAGE.md` |
| **Baseline** | Chain Ladder arithmetic on the same cells, via the CAS package (`chainladder` 0.10.1) | `results/fleet/FINDINGS.md` |
| **Measured result** | Closer than Chain Ladder on **38.8%** of 464; median \|error\| **162.6%** vs **121.3%**; coverage **39.2 / 59.9 / 78.0 / 84.7%** against 50 / 75 / 90 / 95; **13.8%** of units above the model's own 95% upper bound against a nominal 2.5% (14.7% on the 231-unit held-out half; both cited from `results/fleet/COVERAGE.md`, which reconciles them) | `results/fleet/FINDINGS.md`, `results/fleet/COVERAGE.md`, `results/fleet/CALIBRATION.md` |

**Corrected on our side while building this table.** The entry had been describing the model as seeing
**"40–60 training rows per fit"** (`README.md` "Did it work?", `docs/experiments.md` §5). That number came from
single-triangle runs; the fleet's direct arms saw **6–33 rows, median 25**, and no unit saw more than 40. The
recount is `results/conditions/rows_per_fit.log`. The correction moves our own explanation *towards* its
conclusion (the fits were smaller than we said) but it also means the entry has been quoting a scale it never
ran, which is worth fixing before the number is used again.

---

## 2. The conditions, both directions

### 2.1 Where the source says the method succeeds

| # | Condition, as the source states it | Tag | Source (verbatim) |
|---|---|---|---|
| **A1** | Ranks first on the standard i.i.d. benchmarks | verified | S1: "TabPFN-3.5 sets a new state of the art on standard tabular prediction in TabArena… TabArena 1 of 89 … TALENT 1 of 37" |
| **A2** | Handles non-i.i.d. data — grouped **and temporal** splits — including tiny datasets | verified | S1: "On BeyondArena, which spans grouped and temporal splits, tiny to million-row tables, and text and high-cardinality features, TabPFN-3.5 ranks first overall, and for every subset of non-large (<100K rows) datasets" |
| **A3** | Its **predictive distributions** are of high quality — first on the benchmark that scores distributions properly | verified | S1 §C.7: "ScoringBench … evaluates the full predictive distribution of regression models with proper scoring rules rather than point error. It has 101 OpenML regression datasets, each subsampled to 3,000 rows and scored with 5-fold cross-validation… TabPFN-3.5 places first with a mean rank of 2.85" |
| **A4** | Distributional quality **specifically for skewed, zero-heavy targets like claim amounts** | verified | S2 `changelog/tabpfn-3.5`: "TabPFN-3.5 also improves predictive distributions for skewed and zero-heavy regression targets such as claim amounts and customer spend. It ranks first on ScoringBench" |
| **A5** | The model can **extrapolate** — capability, not just a hope | verified for TabPFN-3 | S4 §2: "**Out-of-distribution prior.** We add out-of-distribution prediction tasks, allowing models trained on our prior data to remain performant under distribution shifts, as well as moving from pure interpolation to extrapolation." Figure 26 caption: "Example demonstrating the extrapolation capabilities of TabPFN-3 (using our out-of-distribution compatible preprocessing), comparing to CatBoost. As can be seen, TabPFN-3 is able to extrapolate successfu…" |
| **A6** | Small data is this model family's founding strength | verified | S1 Figure 1(a) table: the largest recommended row count for **TabPFN-v1 is 1,000**; S2 `faq`: CPU inference is supported but "only suitable for small datasets"; S5 abstract: "existing tabular foundation models **excel on tiny- to medium-sized IID data**" |
| **A7** | One vendor-published testimonial reports exactly our domain | verified quote / **unverified as evidence** | S1: "TabPFN outperforms gradient boosting on real-life insurance datasets without any tuning. And you get prediction confidence intervals, which are super useful for risk applications." — Kacper Wieczorek, Marshmallow. A quote, not a measurement: no units, no split, no baseline protocol. It is recorded because it is the closest thing in the sources to our problem, and it is **not** counted as a condition in §3 |

### 2.2 Where the source says it fails, or is less likely to help

| # | Condition, as the source states it | Tag | Source (verbatim) |
|---|---|---|---|
| **B1** | Size limits are **upper** bounds with a row↔column trade-off; none is a floor | verified | S2 `models`: TabPFN-3.5 "Max training rows 1,000,000 / Max columns 20,000"; "For TabPFN-3.5, we recommend datasets with **up to 6,000 columns**; the model ceiling is **20,000 columns**. Row and column limits trade off"; S3: "**Mind the dataset size**: TabPFN works best on datasets within its recommended size limits. TabPFN-3.5 … accept up to 1,000,000 rows and 20,000 features" |
| **B2** | On **temporal and grouped** splits, conventional models retain the best performance; TabPFN only **matches** them at non-large scale | verified | S1 §2.2: "Tuned and ensembled MLPs **retain the highest performance on grouped, temporal, and large datasets**, but TabPFN-3.5 substantially narrows these gaps… When restricted to BeyondArena non-large (datasets with up to 100K rows), TabPFN-3.5 **matches** the best baselines for grouped and temporal datasets" |
| **B3** | On **small** datasets, another tabular foundation model leads | verified quote / **unverified** slice definition | S1 §C.2.2: "TabPFN-3.5 leads both models overall and across every data slice **except small datasets, where TabFM leads**." The report does not define its size slices; that definition lives in the benchmark papers, which we did not read for it |
| **B4** | The documented non-IID evidence base **excludes** our scale | verified | S5 §1: "BeyondArena focuses on evaluating predictive machine learning models … on non-IID data, ranging from **tiny to large (100-1M)** … **Out of scope are few-shot predictions (<100)**" |
| **B5** | For inputs outside the training range, the distribution itself is not to be trusted, and the model will not warn you | verified | S2 `capabilities/predictive-distribution` (Summary): "For inputs well outside the training range, treat the distribution with caution: bucket boundaries are fixed at training time and the model does not flag OOD inputs automatically." |
| **B6** | Quantile preprocessing **clips** values outside the training range by default | verified | S2 `improving-performance/preprocessing`: "Quantile transforms usually clip values outside the training range to the output boundary. The extrapolating version keeps some information about how far a new value is outside that range." — `"quantile_uni_extrapolate"` "Acts like `quantile_uni` inside the training range and **extends linearly outside it**" |
| **B7** | The headline ranks belong to a configuration that is **not** the one we ran | verified | S1 Table 1 caption: "Rank is the position of the best TabPFN-3.5 member (**TabPFN-3.5-Thinking on the first four benchmarks**…". S2 `models`, availability column: TabPFN-3.5-Thinking — "Hosted API & `tabpfn-client`, VPC" (the base model we ran is the one shipped in the OSS package) |
| **B8** | Calibration must be **checked**, on skewed targets, rather than assumed | verified | S2 `improving-performance`: "Use `eval_metric` and `tuning_config` to optimize for your specific evaluation metric, **and check calibration on skewed targets**" |

### 2.3 Operational guidance and stated defaults (what the source tells a practitioner to do)

Compliance is recorded against the arm we actually ran — `TabPFNRegressor(device="cpu")`, default inference
configuration, NumPy features, no tuning.

| # | Guidance | Our compliance |
|---|---|---|
| **C1** | "Feed in data as raw as possible… additional processing often hurts" — but **add domain features** the model cannot derive (ratios, interactions, group aggregations) | **Followed, in the vendor's step-2 sense**: our features are domain ratios (the link ratio, the last observed ratio, Chain Ladder's own implied factor handed in as a prior). The entry's framing of this as "no feature engineering" is a claim about *cost*, and the docs describe feature engineering as "one of the most impactful ways to improve TabPFN's performance" |
| **C2** | "Check column types" first: "Declare categorical columns, especially integer-coded ones… For grouped data, include the group identifier as a categorical column. **Column typing is the cheapest change and often the largest gain**" | **Not followed.** `origin_idx`, `dev_idx`, `cal_idx` are identifiers passed as floats in a NumPy array; nothing is declared categorical (S2 `improving-performance` step 1; `feature-engineering`: "Do not replace the identifier with the fingerprint feature or hash it into a number"; `faq`: "Categorical strings/categories … are handled automatically" does **not** cover integers passed as floats) |
| **C3** | For "grouped and time-ordered rows", Thinking mode "is the strongest option"; `group_col` / `time_col` / `group_time_col` exist for exactly that, and are **client/API only** | **Not followed, and not available to us locally.** Our report is a temporal split and the docs' answer to temporal structure is an API-only feature (`capabilities/thinking-mode`, "Grouped and time-ordered rows") |
| **C4** | Fine-tune "when you have a specialized domain or distribution shift" | **Not followed** (parked deliberately: `docs/experiments.md` §2, GPU hours and it contradicts the entry's "zero-shot" claim) |
| **C5** | For a target with "ordered tail behavior that must extend beyond the training range", use the extrapolating transform, and "validate inverse-transformed predictions carefully" | **Not followed** — the default representation was used, and no target transform was tried (B6) |
| **C6** | CPU is supported but "only suitable for small datasets"; "TabPFN is slow to execute on a CPU" | **Consistent with our usage** (S2 `faq`, S3) |
| **C7** | Determinism: "With a fixed seed and in the same environment TabPFN inference is deterministic" | **Independently reproduced**: "the same command twice gives the same reserve to the pound" (`README.md`, `docs/readiness.md`) |

---

## 3. The four-bucket check

Definitions used, exactly as the task specifies. **Fits** — the condition is one we meet and it predicts a
negative like ours. **Predicts the opposite** — we meet a *success* condition, so the source points away from
our result. **Ruled out** — our setup is outside the condition (the condition cannot bear on our result either
way). **Outside the evidence** — the condition's evidence base does not cover our regime.

| # | Condition | Why our profile lands there | Verdict |
|---|---|---|---|
| A1 | i.i.d. benchmark leadership | Our split is a held-out **diagonal** of one table — temporal, not i.i.d. — and no benchmark in S1 evaluates per-cell ratios or a deterministically-computed actuarial baseline | **ruled out** |
| A2 | Non-i.i.d. coverage on "tiny to million-row" grouped/temporal data | Our split type is covered (temporal), our scale is **not**: B4 puts the evidence base at 100–1M rows and declares sub-100-row prediction out of scope; our fits see 6–33. Also, the benchmark's comparison partners are tuned MLPs/GBDTs, not Chain Ladder | **outside the evidence** |
| B2 | On temporal/grouped splits, conventional models retain the best performance; TabPFN only matches them at non-large scale | We are a temporal split. The source does **not** claim a lead for the model in our split type; the documented expectation is parity with strong conventional methods, and a deterministic domain method is exactly the conventional thing here. Our measured result — closer than the baseline on 38.8%, worse on 61.2% — is the shape that parity-at-best predicts | **fits** (their baseline class ≠ ours; the substitution of Chain Ladder for "tuned and ensembled MLPs" is ours, stated in §4) |
| B3 | "small datasets, where TabFM leads" | We are at the small end of every scale the sources state. The slice is undefined in the report, so this is a weaker signal than B2 — and the comparison is between two foundation models, so it predicts "not the leader on small data", not "loses to Chain Ladder" | **fits** |
| B5 | OOD inputs: the distribution's bucket boundaries are fixed at training time and OOD is not flagged | Our measured failure is **one-sided and at the tail**: 14.7% of units have their truth above the model's own 95% upper bound, 0.9% below the 95% lower bound (`results/fleet/CALIBRATION.md`). A distribution whose support is fixed by the training targets fails in exactly that direction when the truth lies beyond that support. **We have not verified that our prediction inputs are "well outside the training range" in the sense this sentence means** — the correspondence is our inference, not the source's statement | **fits** (mechanism named by the source; correspondence asserted by us) |
| B6, C5 | Quantile transforms clip outside the training range; an extrapolating variant exists and was not used | We ran the default representation on a target whose tail is the thing under test, having been told the default clips at the boundary | **predicts the opposite** |
| C2, C3, C4, B8 | The vendor's escalation path for exactly our regime — declare identifiers categorical; Thinking mode for time-ordered rows; fine-tune for distribution shift; check calibration on skewed targets | We meet the regime the guidance addresses and followed none of it. Cheapest of all is C2 ("often the largest gain"), which is a column-typing change | **predicts the opposite** |
| A4 | Distributional quality "for skewed and zero-heavy regression targets such as claim amounts" | Our target is a positive, skewed ratio on incurred claim/loss triangles — the regime the vendor names by name — and our measured intervals under-cover at every level | **predicts the opposite** |
| A5 | Extrapolation capability (o.o.d. prior) | Our own explanation for the negative is "it drifts where it extrapolates" (`README.md`; `results/runs/20260918-023600_depth-bias/FINDINGS.md`), and the vendor documents the opposite capability. **The mapping is ours**: S4's demonstration is a synthetic o.o.d. example for **TabPFN-3** under o.o.d.-compatible preprocessing, whereas our failure arises in recursion and in per-cell horizon extrapolation with default preprocessing | **predicts the opposite** |
| A6 | Small data is the family's strength (v1's ceiling was 1,000 rows; "TFMs excel on tiny- to medium-sized **IID** data") | Our own explanation is "40–60 rows is too few to learn development patterns" (the figure was recounted to 6–33 during this check — §1). The source neither states a floor nor agrees: its tiny-data claim is explicitly qualified to **IID**, and our split is not. So it does not predict our negative, and it does not support our explanation either | **outside the evidence** |
| B1, B7 | Upper size limits; the headline configurations are API-only | The ceiling is ~1,000,000 rows above us, so it cannot bear on our result at all; and the OSS base checkpoint we ran is not the variant the leaderboard ranks | **ruled out** |
| B4 | The evidence base excludes sub-100-row prediction | This is the benchmark's own scope statement, not a result — it decides coverage, not direction | **outside the evidence** (it is the reason A2 lands there) |
| A7 | The insurance testimonial | A quote with no units, split, or baseline protocol | not scored (see §2.1) |

**Tally of the four buckets.** *Fits*: 3 (B2, B3, B5). *Predicts the opposite*: 3 (A4, A5, and the guidance
cluster B6/C2/C3/C4/B8). *Ruled out*: 2 (A1, B1+B7). *Outside the evidence*: 3 (A2, A6, B4).

The pre-registered branch in `docs/experiments.md` §5.1 was: "if Prior Labs' own guidance says the model does
not extrapolate, or states a row count it needs, then our failure is predicted by the documentation". **Neither
holds.** The source claims extrapolation (A5), states no row floor (B1), and its coverage of our regime ends at
100 rows (B4). The buckets that do predict a negative are about the **split** and about the **unfollowed
guidance**, not about the sample size — so the stop rule that would have promoted E2b as "the test of the
vendor's stated fix" is **not** triggered by a documented row floor. What E2b would test is our own hypothesis,
unchanged in status: plausible, untested, and not the vendor's stated fix.

---

## 4. The scoped verdict

> **At 6–33 training rows per fit (median 25) on incurred-loss reserving triangles, under a temporal
> held-out-diagonal split, the OSS TabPFN-3.5 base checkpoint in its default inference configuration did not
> beat Chain Ladder** — closer on only 38.8% of 464 evaluations, median |error| 162.6% against the incumbent's
> 121.3%, and intervals covering 39.2 / 59.9 / 78.0 / 84.7% against 50 / 75 / 90 / 95 with the miss one-sided
> at the upper tail — **consistent with published findings that on temporal and grouped splits conventional
> tuned and ensembled models retain the highest performance and TabPFN-3.5 only matches them at non-large
> scale, at a scale those findings do not cover** (the vendor's non-IID evidence base spans 100–1M rows and
> declares few-shot prediction below 100 rows out of scope; our fits see 6–33), **with the vendor's documented
> guidance for exactly this regime not followed** (Thinking mode for grouped and time-ordered data; the
> extrapolating quantile transform for ordered tails; categorical declaration of the identifier columns;
> calibration checking on skewed targets).

Two things this sentence deliberately does **not** say. It does not claim the vendor's documentation explains
our negative — the only conditions that predict a negative are the split-type condition (B2/B3) and the
unfollowed guidance, and neither is a statement about sample size. And it does not claim we have been
*excluded* by the documentation — we have been *unreached* by it.

**What changes in the entry.** Not the headline claim ("we measured a failure and characterised it"), and not
a promotion to "we reproduced the documented envelope limit". What changes:

1. `README.md` / `docs/experiments.md` must stop saying "40–60 training rows per fit": the fleet ran 6–33
   (`results/conditions/rows_per_fit.log`).
2. The entry's explanation ("too few rows, so it falls back on its prior") keeps its status as **our**
   hypothesis, but it can no longer be presented as aligned with the vendor's framing — the vendor's
   small-data claim is about i.i.d. data, and its non-IID evidence starts at 100 rows (B4, A6).
3. The entry gains a sentence it did not have: *the split type is where the documentation expects parity with
   strong conventional methods rather than a win (B2)* — which is a stronger, checkable statement than
   "it did not work".
4. "No feature engineering" stops being a free claim. The docs call feature engineering one of the most
   impactful levers, and column typing "the cheapest change and often the largest gain" (C1, C2).

---

## 5. Outside the evidence — stated explicitly

Everything in this section is a regime our measurements sit in that **no source we read covers**. It is the
part of the report that constrains how far the §4 verdict may travel.

1. **Scale.** 6–33 training rows per fit. The documented non-IID evidence base starts at 100 rows and calls
   below that "few-shot … out of scope" (B4). The documented distributional evidence (ScoringBench) is
   3,000-row datasets (A3). No source states or tests behaviour at our n.
2. **Task.** No source evaluation we found predicts **per-cell development factors**, holds out a
   **diagonal**, or is scored against a **deterministically computed actuarial method**. The vendor's
   benchmarks score tables, not triangles, and their baselines are ML models, so "beats RealMLP on TabArena"
   and "beats Chain Ladder on a triangle" are not the same measurement.
3. **Split within a single table.** BeyondArena's temporal slice holds out *rows later in time* across a
   dataset; ours holds out cells of one triangular table and additionally clamps every training label to the
   evaluation anchor. Its 100–1M row scope statement is about datasets, and the analogy to a 55-cell triangle
   is ours.
4. **The extrapolation evidence.** A5 is a synthetic figure for **TabPFN-3** under o.o.d.-compatible
   preprocessing. Our extrapolation is of a different kind — self-conditioned recursion in the recursive arm,
   and per-cell horizons in the direct arm — and no source describes either.
5. **Configuration.** The first-place ranks in S1 Table 1 are TabPFN-3.5-**Thinking** on the four benchmarks
   where we would want the comparison, and Thinking is API/enterprise only (B7). Our result is about the OSS
   base checkpoint at defaults, and the sources contain no claim that this configuration is their strongest.
6. **The one-sided tail.** Our sharpest measurement — 14.7% of units above the model's own 95% bound versus a
   0.9% miss below (`results/fleet/CALIBRATION.md`) — has no counterpart in any source. It is not predicted,
   and it is not contradicted; it is simply not covered.

---

## 6. The cheapest disambiguating test the sources suggest

The sources themselves name the configurations that follow from this check, ordered by cost. All of them run
on units already on disk (`results/fleet/clrd-IncurLoss.jsonl`), so each is **paired** against the result we
already have — no new design, no new sampling, and the existing noise floor applies.

1. **Column typing + extrapolating target transform, on the same 464 units** (local, minutes, no tokens).
   Declare `origin_idx`/`dev_idx`/`cal_idx` as categorical and give the delta target the
   `quantile_uni_extrapolate` transform (C2, C5, B6). Pre-register the expectation from the sources: if the
   one-sided upper-tail miss is a fixed-support artefact, coverage at 90/95 should move toward nominal *before*
   any widening is applied; if coverage does not move, the OOD-support explanation of §3-B5 is dead and the
   miscalibration is something else.
2. **Thinking mode on the same 464 units** (API, token cost). The vendor's own answer to "temporal, grouped,
   time-ordered" data (C3), and the decisive test of the §4 verdict's second clause: if the documented
   configuration for our split type closes the gap to Chain Ladder, then the negative belongs to the OSS base
   checkpoint at defaults, not to the method — which changes what the entry claims.
3. **The fleet as context (#10 / pre-registered E2b).** Now the test of **our** hypothesis rather than the
   vendor's fix, and it should be described that way: the vendor documents no row floor, so this arm tests
   "does more in-context data fix it" — an open question, not a documented remedy.

What none of these tests is *the* answer to: whether Mack's method or the ODP bootstrap cover on these same
units. That is **#20 / E3a**, and it is unchanged by this report — this check moves no comparative claim.

---

## 7. What could not be verified

- **The size-slice definitions** behind B3 ("small datasets") and A2's "tiny" — not stated in S1; they live in
  the TabArena and BeyondArena papers, which we did not read for their slice definitions.
- **The benchmark results themselves.** Every TabArena, BeyondArena, STRABLE, MulTaBench, TALENT and
  ScoringBench number above is reproduced from what the sources publish; none was re-run here. S1 is explicit
  that baselines are published results, not re-runs — which is the source's own limitation, carried forward.
- **The insurance testimonial** (A7) as evidence of anything. It has no units, split or protocol.
- **Whether our prediction inputs are "well outside the training range"** in the sense of B5. Not measured,
  and not measurable without fitting the model, which this task forbids.
- **Whether TabPFN-3.5 carries the o.o.d. prior of A5.** S1 does not restate the extrapolation claim; it cites
  S4 as `[1]`. The claim is verified for TabPFN-3 and **unverified for 3.5**.
- **Any variant of the model other than the OSS base checkpoint at default configuration** — the entire
  Thinking/Plus/Fast column of the sources is untested here.

## 8. Provenance

- This report: `results/conditions/FINDINGS.md`. Reading only: no model fit, no code change.
- The one new number: `results/conditions/rows_per_fit.log`, produced by
  `.venv/bin/python results/conditions/rows_per_fit.py` — a read-only recount that loads the same triangles,
  column, trims and anchors as `scripts/fleet_eval.py` and calls the package's own
  `direct_training_rows(..., target="delta", known_until=anchor)`, counting rows and predicting nothing.
- Every other number of ours is quoted from `results/fleet/FINDINGS.md`, `results/fleet/COVERAGE.md`,
  `results/fleet/CALIBRATION.md`, `results/runs/20260918-023600_depth-bias/FINDINGS.md` and `README.md`.
- External sources were fetched 2026-09-19 and are listed with URLs in §0; quotes are verbatim from those
  fetches, and the S1 PDF text was read as a converted rendering of the vendor's own file.
