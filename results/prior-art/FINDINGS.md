# E11 — prior art, practice, and the novelty check: what to borrow and what to attribute

> **Status:** written 2026-09-19, reading and analysis only. **No model was fitted, no repository code was changed,
> and no result is claimed here.** Every number of *ours* is quoted from a file under `results/`, with the file
> named next to it. External claims carry a tag: **verified** (read at source, quote reproduced) ·
> **unverified** (the source says it, we cannot check it from here) · **corrected** (a statement that needed
> narrowing to stay inside its source). The search itself is part of the result: §0.3 lists every query made and
> §7 lists everything that could not be reached.
>
> **The one-sentence answer.** The prior art does **not** contain a reserving study that measures the
> calibration or coverage of a learning method's intervals — the neural-reserving line measures *point
> accuracy*, and where it does reach the distribution it scores it with a pinball/log score and never asks
> whether the stated 90% interval covers 90% — and it contains **no evaluation of a foundation model on loss
> triangles at all**, so both entry claims survive in a narrowed form; but the *general* result that a tabular
> foundation model's predictive distribution is well calibrated at n≈50 and drifts at larger n **has now been
> measured elsewhere** (`2603.26611`, at n = 50–20,000), and the mechanism behind our one-sided upper tail is
> the **vendor's own documented** bucket-grid behaviour, so the entry must attribute the method, the mechanism
> and the small-n calibration finding, and claim only what is left: **the first measurement of interval
> coverage for a foundation model on reserving triangles, the first to locate the failure in the depth of a
> recursive projection, and the first to compare against the classical intervals on identical units.**

**What this changes in the entry, in three lines.**

1. **"First to measure this" survives only as a scoped sentence.** Replace any claim that no one has looked at
   uncertainty in ML reserving with: *the neural-reserving literature has measured distributions with proper
   scores but not interval coverage; the one foundation-model density benchmark that does measure coverage
   starts at 50 rows and is not a reserving task; and no source we found evaluates a foundation model on a
   loss triangle.* §1.4 and §2.2 carry the exact sentences.
2. **Two things must now be attributed rather than claimed as ours**: the **fixed bucket grid rescaled to the
   training target's mean and sd** (the vendor's documentation, read by E7/E10 and re-read here) as the
   mechanism behind the short upper tail, and the **general small-n behaviour of a tabular foundation model's
   distribution** (`2603.26611`).
3. **The depth-bias measurement is the part that is cleanest as ours** — multi-step error accumulation is
   classical (P12), but the per-depth measurement *on a reserving recursion* and the **anchoring trap** are not
   stated in any source we could reach (§2.1). That is the claim to lead with, and the one to pre-register
   against.

---

## 0. What was read, and how

### 0.1 Sources

| # | Source | What it is | Tag |
|---|---|---|---|
| **P1** | Kuo (2019), *DeepTriangle: A Deep Learning Approach to Loss Reserving*, Risks 7(3):97 — `arXiv:1804.09253v4` | the canonical deep-learning reserving paper | **verified** (PDF text read; §4.1–§4.3 and §5 read in full) |
| **P2** | Al-Mudafer, Avanzi, Taylor & Wong, *Stochastic loss reserving with mixture density neural networks* — `arXiv:2108.07924`; published *Insurance: Mathematics and Economics* 2022 (`10.1016/j.insmatheco.2022.03.010`) | the neural-reserving paper whose subject *is* the distribution | **verified** (preprint PDF read; abstract, §1.2, §3.3, §4.4–§5.2.4) |
| **P3** | Balona & Richman (2020), *The Actuary and IBNR Techniques: A Machine Learning Approach* — SSRN 3697256 / IFoA document library | ML as a *selector* over classical reserving techniques | **verified** (PDF text read: abstract + §1–§2.2) |
| **P4** | Deprez, Verbeke & Verdonck (2026), *Is TabPFN the Silver Bullet for Insurance Pricing?* — `arXiv:2605.22892v2` | the nearest published foundation-model application in insurance | **verified** (PDF text read in full) |
| **P5** | *Benchmarking Tabular Foundation Models for Conditional Density Estimation in Regression* (2026) — `arXiv:2603.26611v1` | the benchmark that measures TFM density calibration and 90% coverage, n = 50–20,000 | **verified** (HTML text read; abstract, metrics, calibration sections) |
| **P6** | Prior Labs, `docs.priorlabs.ai/models.md` (fetched 2026-09-19) | the vendor's own limit table, per model version | **verified** (quote reproduced in §1.6) |
| **P7** | Prior Labs Team, *TabPFN-3: Technical Report* — `arXiv:2605.13986v2` | the "cell-budget frontier" wording | **verified** (HTML text read) |
| **P8** | Prior Labs Team, *TabPFN-2.5* — `arXiv:2511.08667v2` | the release the pricing paper's v2.6 sits above | **verified** (HTML text read) |
| **P9** | IFoA *Machine Learning in Reserving* working party — blog `institute-and-faculty-of-actuaries.github.io/mlr-blog` and book `mlrwp.github.io/mlrwp-book` | the working-party literature: GLM in R/Python, LASSO, an mlr3 worked example with GLM-CL/random forest/XGBoost/LASSO, NN diagnostics, uncertainty, surveys | **verified** (posts and chapters fetched; the pages quoted are reproduced) |
| **P10** | Taylor & McGuire (2023), *Model Error (or Ambiguity) and Its Estimation, with Particular Application to Loss Reserving*, Risks 11(11):185 (`10.3390/risks11110185`) | the working party's own uncertainty paper (Bayesian LASSO model averaging over admissible models) | **abstract verified via the publisher's DOI record; full text NOT reached** (MDPI returns 403 to this box). Any statement about whether it reports *coverage* is therefore **unverified** and must not be made |
| **P11** | Taylor (2019), *Loss Reserving Models: Granular and Machine Learning Forms*, Risks 7(3):82 | the survey that compares granular and ML reserving | **abstract verified** (DOI record); full text not reached |
| **P12** | Ben Taieb & Atiya (2016), *A Bias and Variance Analysis for Multistep-Ahead Time Series Forecasting*, IEEE TNNLS (`10.1109/TNNLS.2015.2411629`) | the primary source for multi-step error accumulation | **bibliographic record verified** (title/venue/year/DOI); the abstract was not retrievable from the indexes reachable here — **wording unverified** |
| **P13** | Xu & Xie (2021), *Conformal prediction for time series* (`arXiv:2010.09107v15`, **EnbPI**) | distribution-free intervals for ordered data without exchangeability | **verified** (abstract) |
| **P14** | Romano, Patterson & Candès (2019), *Conformalized Quantile Regression* (`arXiv:1905.03222`) | conformal + quantile regression, adaptive to heteroscedasticity | **verified** (abstract) |
| **P15** | *Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking* (2026) — `arXiv:2601.18509v2` | the method inventory for the ordered-data defect | **verified** (abstract) |
| **P16** | *Copula Conformal Prediction for Multi-step Time Series Forecasting* (`arXiv:2212.03281`, ICLR 2024) | multi-step conformal that keeps step dependence | **verified** (abstract) |
| **P17** | *Relational Conformal Prediction for Correlated Time Series* (`arXiv:2502.09443`) | conformal + quantile regression across correlated sequences | **verified** (abstract) |
| **P18** | Lhaut & Lopez (2026), *Gradient boosting for extremes: sampling theory and application to insurance* (`arXiv:2606.14268`) | covariate-dependent Generalised Pareto tails by boosting (peaks-over-threshold) | **verified** (abstract) |
| **P19** | Cai, Abdallah & Jeganathan (2025), *Penalized Copula Mixed Models for Intercompany Loss Reserving and Risk Capital* (`arXiv:2509.05426`) | borrowing strength across insurers' triangles | **verified** (abstract) |
| **P20** | Padayachy, Richman, Scognamiglio & Wüthrich (2025), *In-Context Learning Enhanced Credibility Transformer* (`arXiv:2509.08122`); Richman, Scognamiglio & Wüthrich, *The credibility transformer*, EAJ 15(2):345–379 (2025) | credibility as an architectural prior; in-context batches of *similar instances* | **verified** (abstract of the 2025 paper; the EAJ paper via the reference list of P4 — bibliographic only) |
| **P21** | Ding et al. (2025), *L2C-TabPFN: Longitudinal Progression Prediction of Alzheimer's Disease with Tabular Foundation Model* (`arXiv:2508.17649`) | a published tabular-FM application on **ordered/panel** data | **verified** (abstract) |
| **P22** | Helli, Schnurr, Hollmann, Müller & Hutter (2024), *Drift-Resilient TabPFN* (`arXiv:2411.10634`) | in-context learning of temporal distribution shift | **verified** (abstract) |
| **P23** | Abdullah (2026), *Understanding Context Sampling in TabPFN on Small Tabular Datasets* (`arXiv:2607.26628`) | what the in-context *context* does at small n | **verified** (abstract) |
| **P24** | Ye et al. (2026), *Evaluating TabPFN for MCI→AD Conversion in Data-Limited Settings* (`arXiv:2604.27195`) | TabPFN vs XGBoost/RF across training sizes N = 50–1,000 | **verified** (abstract); **result outcome unverified** (full text not read) |
| **P25** | Avanzi, Li, Wong & Xian, *Ensemble distributional forecasting for insurance loss reserving* (`arXiv:2206.08541`) | combining *stochastic* reserving models on distributional criteria, not just the centre | **verified** (abstract) |
| **P26** | Our own files (not prior art, quoted as such): `results/fleet/FINDINGS.md`, `results/fleet/COVERAGE.md`, `results/fleet/CALIBRATION.md`, `results/fleet/coverage.log`, `results/runs/20260918-023600_depth-bias/FINDINGS.md`, `results/runs/20260919-config-cells/FINDINGS.md`, `results/runs/20260919-column-comparison/FINDINGS.md`, `results/runs/20260919-pricing/pricing.md`, `results/runs/20260919-target-support/`, `results/conditions/FINDINGS.md`, `results/conditions/rows_per_fit.log`, `results/remodel/FINDINGS.md` | the measured position this pass is checking | our files |

Two more sources were read for context and are cited only bibliographically, never for a claim:
England & Verrall (2002), *Stochastic claims reserving in general insurance*, British Actuarial Journal 8(3),
443–518 (via P1's reference list); and the vendor's `docs.priorlabs.ai/improving-performance.md` route, which
E7 and E10 already read and which this pass did not re-read.

### 0.2 How each source was read

Primary PDFs were downloaded from the authors' own hosts (`arxiv.org/pdf/...`, `actuaries.org.uk/...`) and the
text extracted locally; HTML was read from `arxiv.org/html/...`; docs were read as the vendor's own markdown
(`docs.priorlabs.ai/<page>.md`). Nothing in this report comes from a search snippet, a blog paraphrase of a
paper, or recollection — the snippets returned by the search engines were used only to *find* documents, and
every claim above is from the document itself. Where a search result was as far as I got (the items tagged
**unverified**), it says so at the claim.

### 0.3 The search, recorded

Engines: the arXiv API (`export.arxiv.org/api/query`) — the main instrument, because most of this literature is
there; Crossref and OpenAlex bibliographic APIs; Semantic Scholar's paper endpoint for abstracts of
publisher-hosted work; DuckDuckGo's HTML endpoint for the actuarial/professional literature that is not on
arXiv. `web_search` was unavailable this session (all four configured backends failed: timeouts on
Keenable/Exa/Parallel, HTTP 403 from Firecrawl), and a headless browser could not be started (no Chrome on this
box) — so the search ran through direct HTTP against the sources above.

Queries, verbatim, in the order they were run (arXiv `all:` fields unless noted):

```
loss reserving AND neural network            claims reserving AND machine learning
loss reserving AND uncertainty               reserving AND conformal prediction
foundation model AND loss reserving          reserving AND quantile
reserving AND distributional                 gradient boosting AND loss reserving
attention AND claims reserving               neural network AND IBNR
tabular foundation model                     TabPFN AND insurance
TabPFN                                       conformal AND insurance reserving
extreme value AND claims reserving           credibility AND loss reserving
backtesting AND reserving                    conformal AND time series AND coverage
TabPFN AND small                             generalized Pareto AND reserving
TabPFN AND calibration                       bar distribution AND TabPFN
multi-step AND error accumulation AND forecasting
reserving AND empirical coverage             reserving intervals AND calibration
Crossref: neural network loss reserving insurance · gradient boosting claims reserving
  non-life insurance · Mack chain ladder prediction error model validation coverage ·
  McGuire Taylor measuring uncertainty loss reserving machine learning bootstrap ·
  Taieb Atiya bias variance multistep ahead time series forecasting ·
  empirical coverage of prediction intervals claims reserving backtesting simulation ·
  machine learning claims reserving point estimate uncertainty assessment review
OpenAlex: machine learning loss reserving uncertainty intervals ·
  Loss Reserving Models: Granular and Machine Learning Forms ·
  neural network claims reserving quantile coverage calibration
DuckDuckGo: "loss reserving" machine learning working party CAS ·
  McGuire Taylor ... open access paper · Wüthrich neural networks applied to chain-ladder reserving ·
  conformal prediction loss reserving insurance triangle · TabPFN loss reserving triangle IBNR ·
  Richman Balona actuary IBNR machine learning approach published journal ·
  xgboost gradient boosting loss reserving triangle study results
```

The two searches that decide this task returned what the report says they return: `conformal AND insurance
reserving` → **0 results**; `foundation model AND loss reserving` → **0 results**; and the one hit for
`TabPFN AND insurance` is pricing, not reserving.

---

## 1. Prior art in ML / neural reserving — the novelty check that matters most

### 1.1 The table

| Source | What was modelled | On what data | Headline result | **What it says about uncertainty** |
|---|---|---|---|---|
| **P1** Kuo (2019), DeepTriangle | paid losses and claims outstanding jointly, as sequences, by a GRU encoder-decoder with a company-code embedding | NAIC Schedule P, 1988–1997, 4 lines of business × 50 companies; accident years × 10 development lags | MAPE/RMSPE better than Mack, ODP, Bayesian MCMC (CIT/LIT) and AutoML on all four lines | **Explicitly none.** "For the stochastic models, we use the means of the predictive distributions as the point estimates to which we compare the actual outcomes." §5: "**While this study focuses on prediction of point estimates, future extensions may include outputting distributions in order to address reserve variability.**" The 95% bands in Figures 4–5 come from the 100-model ensemble spread, not from a validated predictive distribution — **[corrected]** the figures can be mistaken for intervals; the text is clear they are not |
| **P2** Al-Mudafer et al. (2021/2022), MDN | a Mixture Density Network over incremental claims, outputs the parameters of a mixture Gaussian / mixture Log-Gaussian | 200 simulated 40×40 triangles (4 SynthETIC scenarios) + 10 real aggregate triangles split from AUSI (Auto Bodily Injury, 36×36, quarterly 2005–2014) | MDN beats the cross-classified ODP on central estimates *and* on quantiles in most triangles; RMSE 30.8% of ccODP in environment 2, 61.9% in environment 4, 84.2% on AUSI (their Table 1) | **The closest thing in the literature to our question, and it stops one step short.** Metrics are Negative-Log-Likelihood "log score", RMSE, and **quantile scores (pinball loss) at 75% and 95%** — plus qualitative plots of 25/75/95 risk margins against empirical margins from hundreds of simulations. **No empirical interval coverage is reported, and no calibration test.** A quantile score answers "is the quantile close", not "does the 95% bound cover 95% of the time" |
| **P3** Balona & Richman (2020) | supervised selection over classical IBNR techniques (CL, BF, CC, ILR) and their variants, on rolling sub-triangles | real triangles; the "dynamic" re-reserving view | a systematic way to pick the technique that minimises an actuarial scoring objective out-of-sample; "more accurate reserves" | **Point-estimate selection, explicitly.** The framework's own framing is the best estimate and its error; interval coverage is not the object. It is the paper that made "ML in reserving" a supervised-learning question — and it left the distribution to the classical stochastic methods |
| **P4** Deprez et al. (2026) | **TabPFN** (v2.6 and v3) on claim frequency and severity | two public MTPL datasets (French `freMTPL2`, Belgian `beMTPL97`), 5-fold CV | **negative**: "TabPFN does not achieve the best deviance on any dataset–task combination. GLM is best three of the four"; fold-to-fold variance substantially larger; inference time exceeds GLM+XGBoost combined train+predict even at small contexts; deviance **non-monotonic in context size** | RMSE and Poisson/gamma deviance only — **no interval or calibration metric**. But it is the only published insurance evaluation of the model family we use, and it is a negative one; it names reserving explicitly as the untested case: "The forecasting capabilities can be applied for loss reserving with run-off triangles (in P&C and Health insurance)" |
| **P5** CDE benchmark (2026) | **TabPFN and TabICL variants** as conditional density estimators vs parametric, quantile, tree and neural baselines | 39 real datasets, **training sizes n = 50 to 20,000** (+ an SDSS case study at 50,000) | FMs achieve the best CDE loss, log-likelihood and CRPS on the large majority of datasets; **calibration "competitive at small sample sizes but ... lags behind task-specific neural baselines at larger sample sizes"**; TabPFN calibration ranks 4.4–4.6 at n = 50, mid-table (7–9) at n ≥ 5,000; one failure case is explicitly a **right-tail** one, where "a simple LogNormal-Homo-Ridge model beats RealTabPFN-2.5 because its strong parametric bias yields better estimates with little data, **especially in the right tail**" | **This is the paper that already measures what §1.4 asks about — for tabular regression, not reserving.** Metrics: CDE loss, log-likelihood, CRPS, **PIT-based Kolmogorov–Smirnov calibration, and empirical coverage of 90% predictive intervals**. Conclusion includes: "strong performance on proper scoring rules does not automatically guarantee the best calibration" |
| **P9** IFoA MLRWP (blog + book, 2020–2026) | the practitioner literature: GLM reserving in R and Python; LASSO "self-assembling" reserving models; an mlr3 worked example with GLM-CL, decision tree, random forest, XGBoost and LASSO on a triangle; NN diagnostics; surveys of UK/Canada/Italy use | a single simulated triangle in the worked example; survey data for adoption | in the worked example the LASSO model performed best, **with the authors' own caveat**: "you should not make any conclusions about the relative performance of the different ML methods based on this work" | The working party's **own** uncertainty work is the exception (P10): it splits forecast error into internal model error, external model error, parameter error, process error and model-distribution error and estimates the first, third and fourth (Bayesian LASSO posterior + bootstrap). Its honest limit is stated in the blog: the estimate "is still incomplete"; and the barrier it names is adoption, not methodology |
| **P25** Avanzi et al., ensemble distributional forecasting | ensembling multiple **stochastic** reserving models | a complex synthetic dataset | the optimised ensemble beats model selection and equal weights — "not only with central estimates but also relevant quantiles, such as the 75th percentile of reserves" | Takes the distribution seriously — the combination criterion "considers the full distributional properties of the ensemble and not just the central estimate" — and is the nearest work to *validating* a reserving distribution across models. Still scored on quantiles, not on coverage |
| **P11** Taylor (2019), survey | granular vs machine-learning reserving, in the context of their predecessors | — | a taxonomy and a comparison of the two families' development potential | Abstract-level: a survey of *methods*; the abstract makes no claim about interval validation |

**The classical side, for contrast (not prior art in ML).** Mack's distribution-free chain ladder and the ODP
bootstrap (England & Verrall 2002, quoted via P1's reference list) are where reserving uncertainty actually
comes from in practice, and the literature *does* scrutinise them — the Crossref sweep for this pass returned a
live line of work computing and correcting Mack's ultimate prediction error, e.g. *Unbiased estimator for the
ultimate claim prediction error in the chain-ladder model of Mack*, Annals of Actuarial Science (2022,
`10.1017/s1748499522000082`), and *Estimation error and bootstrapping in the chain-ladder model of Mack*, EAJ
(2020, `10.1007/s13385-020-00241-2`) **[both bibliographic only — title/venue/year/DOI verified, contents not
read]**. The point for us: **the classical intervals are studied hard, and ours is not the first interval in
reserving to be interrogated — it is the first *learning method's* interval to be.** That sentence has to be
earned by the measurement in `results/fleet/COVERAGE.md`, not asserted.

### 1.2 Answer to the first decisive question

> **Has anyone measured the calibration or coverage of a learning method's reserving intervals?**

**Not found — with one adjacent exception that has to be attributed.** Concretely:

- **In reserving: no.** (i) P1 states in its own conclusion that uncertainty is out of scope. (ii) P2, the one
  neural-reserving paper whose subject is the *distribution*, scores it with quantile scores (pinball) and a
  log score and never reports empirical coverage. (iii) P3 selects techniques on point accuracy. (iv) P25
  ensembles distributions and reports quantiles. (v) The working party's own uncertainty paper (P10) estimates
  error *components*, including internal model error — it is the nearest thing to a calibration study in
  reserving, and **its full text was not reachable from this box**, so no statement about whether it validates
  coverage may be made here (it is on the list to read before submission).
- **For tabular foundation models: yes, and it must be attributed.** P5 measures **PIT-based calibration and
  empirical 90% coverage** of TabPFN's (and TabICL's) predictive distributions across **n = 50 to 20,000** on 39
  real datasets, and finds calibration *best at small n* and comparatively worse at larger n; and it reports a
  right-tail failure case at n = 50. It is not a reserving task, its smallest n is 50 rows against our 6–33, and
  its units are i.i.d. rows rather than ordered units of a projection — but the *question we thought we were
  first to ask* has been asked of this model family, in this decade, and answered roughly the way our own
  results point.

**The attribution the entry must carry.** Write the claim as: *calibration of a learning method's reserving
intervals has not been measured — the neural-reserving literature scores distributions with proper scores but
not coverage — while for tabular foundation models generally, calibration and 90% coverage have been measured
from n = 50 upward, in an i.i.d. regression setting (arXiv:2603.26611). What is new here is the reserving
object, the order (our units are steps of a recursive projection, not exchangeable rows), the scale (6–33
training rows, median 25), and the comparison against the incumbent's intervals on identical units.* The last
of those four is the one that is not yet done — it is #20/E3a in `docs/experiments.md` §5.2 and it is still open.

### 1.3 Answer to the second decisive question

> **Has anyone evaluated a foundation model on loss triangles?**

**Not found.** The searches above returned **zero** hits for `foundation model AND loss reserving` and for any
TabPFN/reserving combination; the single TabPFN-and-insurance paper is **pricing** (P4), it is **negative**, and
it names reserving as future work. What is named and what differs, in order of distance from us:

| Named | What it is | What differs from us |
|---|---|---|
| **P4** Deprez et al. (2026), TabPFN on MTPL pricing | the nearest published TFM-in-insurance evaluation | **Pricing**, not reserving: frequency/severity on policy-level rows with hundreds of thousands of training rows. No triangle, no recursion, no reserve, no intervals; scored on RMSE and deviance |
| **P21/P22/P23/P24**, tabular FMs on ordered/individual-level data | TFM applications where the data are ordered (longitudinal, temporal-shift, context-sampled) | Clinical/observational panels, not reserving; classification in the clinical cases; no actuarial baseline, no reserve |
| **P2** MDN (2021) | a *learned distribution* on aggregate triangles — not a foundation model | Per-triangle training with a hyper-parameter search; 40×40 simulated triangles and 36×36 real ones; no in-context learning, no pretrained prior |
| **P1** DeepTriangle (2019) | deep learning on triangles across companies | Trained per line of business on a GPU; not zero-shot; point estimates only |
| `IFoA-ADSWP/tabpfn-reserving` (this repository) | TabPFN-3.5 on triangles | **ours** — and it is the only such evaluation the search can find, which is itself a result worth stating in the entry, carefully: *we found no prior evaluation of a foundation model on loss triangles*, not *none exists* |

### 1.4 The two error checks this task asked for, explicitly

**Check 1 — a size limit quoted in *rows* when the source says *cells*.** Looked for in the vendor's own
documentation and reports, and in every third-party source read above (P1–P5).

- The vendor's **docs** state limits as **rows × columns** with the trade-off explicit: "TabPFN-3.5 | Max
  training rows 1,000,000 | Max columns 20,000", then "For TabPFN-3.5, we recommend datasets with **up to 6,000
  columns**; the model ceiling is **20,000 columns**. **Row and column limits trade off, so their maximum
  values cannot necessarily be used together**" (P6 **[verified]**), plus the headline "TabPFN-3 and later
  versions support up to 1,000,000 rows, **subject to feature count** and checkpoint/API limits" (P6).
- The vendor's **report** states the envelope as a **cell budget**: "TabPFN-3 is benchmarked along a
  **cell-budget frontier**: up to 1M rows at 200 features, 100k rows at 2,000 features, or 1k rows at 20,000
  features" (P7 **[verified]**), and the 2.5 report frames its own gain as "a **20× increase in data cells**
  compared to TabPFNv2" (P8 **[verified]**).
- **What I found, stated precisely.** No source I read quotes a *cell* figure as a *row* figure. What P4 does is
  the neighbouring error: it uses the **row count alone as the context-size ladder** ("Context sizes of 2,000,
  5,000, 10,000, 50,000, 100,000 (the maximum context size of TabPFN-v2.6)") for a table whose feature count
  never varies — so the row number is presented as the binding limit, while the vendor's own framing is a joint
  row–feature budget and the report for that family states the *designed* size as 50,000 with 100,000 reported
  "outside our validated range" (P8 **[verified]**). Tag: **[corrected]** — a row-only reading of a joint
  limit, and it is in *their* paper, not ours.
- **In our own documents: clean.** A sweep of `README.md`, `docs/*.md` and `results/**/*.md` for size-limit
  numbers returns only vendor-quoted limits carrying the trade-off (E7's B1 row), and the two benchmark *scope*
  statements quoted verbatim (BeyondArena's "100-1M" rows and its sub-100 exclusion; TabArena's "less than 500
  training samples"), both of which E7/E10 already carried. No row-for-cell substitution, and no unmarked
  superseded limit.

**Check 2 — a numeric threshold attributed to a vendor or author who never published it.** Found, once, and it
is not ours:

> **P4 [corrected]:** "We benchmark two versions of TabPFN: **TabPFN-v2.6, which scales to 100,000 samples and
> 2,000 features (Hollmann et al. 2025)**, and TabPFN-v3 (Grinsztajn et al. 2026), whose updated architecture
> scales to 1,000,000 samples provided the feature count remains below 200."

Why this is an attribution error and not a harmless shorthand: the cited work is Hollmann et al. (2025),
*Accurate predictions on small data with a tabular foundation model*, **Nature 637**, 319–326 — the
**TabPFNv2** paper, whose model the vendor's own current table lists at **10,000 training rows and 500
columns** (P6 **[verified]**). The 100,000 figure belongs to **v2.6**, a later release, and to Prior Labs'
*current* documentation (P6: "TabPFN-2.6 | Max training rows 100,000 | Max columns 2,000"), while the
release report immediately below it says the model "was designed for up to 50,000 rows" and that "while
TabPFN-2.5 was designed for up to 50,000 rows, we note that this limit is not strict and report strong results
on benchmarks with up to 100,000 training samples" (P8 **[verified]**). So **the number is the vendor's, the
version is not the cited paper's, and the designed limit is half of what is quoted.** The correct sentence
would cite the vendor's model table for v2.6 and keep the two-release distinction.

Nothing in our own writing commits this error: E7 §3-C1 already established that no source states a
minimum-sample-size threshold and that we quote no such number, and this pass found no new one (the vendor
still publishes **no row floor**; P6 is a ceiling table).

---

## 2. The novelty check on our own two findings

### 2.1 The depth-bias result, and the anchoring trap

**The claim being checked.** Per-step bias is **zero where the model has training rows and +3.2% per step (se
0.36%) where it extrapolates**, measured against depth into the projection, compounding to ×1.33 over nine
steps against a measured ×1.67 over-reserve, with an intercept of +0.0001 — and the **anchoring trap**: on
anchors `n-4 … n-2` the same triangle reads **−0.0018 (se 0.0030)**, flat, because the depths production uses
are not reachable from there (`results/runs/20260918-023600_depth-bias/FINDINGS.md`).

**Verdict: partially.**

- **Stated already (the general phenomenon).** Multi-step-ahead forecasting error accumulation is classical:
  Ben Taieb & Atiya (2016), *A Bias and Variance Analysis for Multistep-Ahead Time Series Forecasting*, IEEE
  TNNLS **[bibliographic verified; contents not read]** is the standard reference for a bias/variance
  decomposition of multi-step forecasts, and the direct-vs-recursive distinction it formalises is exactly the
  distinction between our two arms (deleting the recursion took the same model, features and settings from
  +66.8% to −1.2% on `abc` — `results/fleet/FINDINGS.md` quotes the direct arm's limit; `5,211,802` is its
  reserve on `abc`, `results/runs/direct-arm.md`). Also, the same literature says what our trap says: an
  evaluation window that does not cover the horizon claimed cannot see the effect. **So neither "errors
  accumulate in multi-step forecasts" nor "you must evaluate over the horizon you claim" is novel, and the
  entry must not present either as a discovery.**
- **Partially (close, but not the same statement).** The forecasting literature measures multi-step error
  against *horizon* on a time series with a fixed origin; our measurement is **per-step bias against depth into
  a self-conditioned recursion where the model's own output becomes the next step's input**, on a
  *deterministically-computed* actuarial baseline, with a **clean zero at depth 0 by construction** showing the
  measurement is not itself biased. The recursive/self-conditioned case is discussed qualitatively in the
  neural-reserving literature — P2 puts it plainly: "the NN's black box modelling can cause long range forecasts
  to become unstable … it may not be adequate to project the behaviour of the highly flexible function fitted
  to the upper triangle to the lower triangle" — and P2's *answer* to it is the opposite of ours: **explicit
  projection constraints** imposed by the actuary during training, plus an MSE term added to the NLL to stop
  volatility from inflating when the central estimate drifts. That is a borrowable method (see §5-B5), and it is
  evidence that the failure we measure is one practitioners already knew to guard against.
- **Not found (ours).** No source we could reach states a **per-depth bias for a reserving projection**, and
  none states the **anchoring trap** as a measurement — that a backtest anchored near the ultimate *reports a
  flat line while the effect is present*. The trap is a design-of-evaluation point; the rolling-origin vs
  fixed-origin literature (P2 cites Tashman 2000 and Bergmeir & Benítez 2012; Al-Mudafer's own contribution is to
  apply rolling-origin **inside one triangle**, and P2 reports that it "was visibly unable to capture the
  inflation shock accurately for environment 3, due to a lack of training data in the later calendar periods")
  is the nearest thing, and it is about *which data are used for validation*, not about *which depths a backtest
  can see*.

**The claim the entry can make, as a sentence:** *per-step bias in a recursive reserving projection, measured
against depth, is zero where the model has training rows and grows by +3.2% per step where it extrapolates; and
a backtest whose anchors leave only short projections readable reports a flat line while that effect is
present. Neither statement is in the prior art we could reach; the phenomenon of multi-step error accumulation
is classical, and the corresponding guard — constraining the projection during training — is documented in the
neural-reserving literature.*

### 2.2 The one-sided short upper tail, and the fixed bucket grid

**The claim being checked.** Coverage under-covers at every level — **39.2 / 59.9 / 78.0 / 84.7%** against
50 / 75 / 90 / 95 (`results/fleet/COVERAGE.md`) — and the miss is **one-sided**: **14.7%** of units have their
truth above the model's own 95% upper bound against a nominal 2.5%, while **0.9%** fall below
(`results/fleet/CALIBRATION.md`); the mechanism is claimed to be a bucket grid fixed on the pretraining prior
and rescaled to the training target's mean and sd, with a free arithmetic check showing the truth leaves the
training-label range on **21.1% of units (5.3% of cells)**, one-sided upward
(`results/remodel/FINDINGS.md` §4).

**Verdict: partially — the mechanism is the vendor's, the direction and magnitude on a real reserving task are
ours, and the general small-n behaviour is now measured elsewhere.**

- **Stated already (the mechanism).** The vendor's documentation names the grid and its rescaling, warns that
  "for inputs well outside the training range, treat the distribution with caution: bucket boundaries are fixed
  at training time and the model does not flag OOD inputs automatically", and the extrapolating transform's own
  description says quantile transforms "usually clip values outside the training range to the output boundary"
  (quoted in `results/conditions/FINDINGS.md` §2.2 B5/B6 and `results/remodel/FINDINGS.md` §1.1–§1.2). **The
  mechanism is not a discovery of ours and must be attributed when used.** What is ours is the *correspondence*:
  the source names the clipping and the absent OOD flag; it does not name reserving, does not say the failure is
  one-sided, and does not give a magnitude.
- **Stated already (the general TFM behaviour).** P5 measures PIT calibration and 90% coverage for TabPFN from
  **n = 50** upward, finds calibration strongest at small n and deteriorating at n ≥ 5,000, and documents a
  case where a simple parametric model's "strong parametric bias yields better estimates with little data,
  **especially in the right tail**". That is not our claim (their n ≥ 50, i.i.d. rows, no recursion), but it is
  the same *kind* of finding in the same family, and a reviewer who knows it will read our claim in its light.
  **Attribute it.**
- **Partially (close, and it changes the framing).** P2's MDN is trained end-to-end with an NLL loss on the
  *mixture* parameters, so the tail is fitted, not inherited; and P2's own weakness list is the mirror image of
  ours: "the MDN often over-estimates volatility in later DQs, also due to a lack of data in that region", and
  "using the NLL loss function alone can encourage the MDN to over-estimate the volatility when its central
  estimate is inaccurate". A too-wide tail from a fitted mixture and a too-short tail from a fixed rescaled grid
  are different defects with the same root cause (thin data at depth), so P2 is *not* prior art for our claim —
  but it is prior art for the *diagnosis* and it supplies two remedies (a mixture Log-Gaussian target "which
  provides a positive, heavier-tailed option", and projection constraints) that transfer directly.
- **Not found (ours).** No source we could reach links a **fixed, non-fitted bucket grid** to a **one-sided**
  interval failure, and none measures the **asymmetry** (14.7% above the 95% bound against 0.9% below) on any
  task. The direction is the measurement that is ours.

**The claim the entry can make, as a sentence:** *on 464 fleet evaluations this model's intervals under-cover
at every level, and the miss is one-sided — 14.7% of units above its own 95% upper bound against a nominal
2.5%, 0.9% below — which is the direction a predictive distribution inherits when its shape is a bucket grid
fixed on the pretraining prior and merely rescaled to the training targets' mean and sd (the vendor's own
documented mechanism), and which is consistent with the direction, though not the setting, of the calibration
results now published for tabular foundation models generally.*

---

## 3. Case studies of foundation models under scarcity

Everything here is a *published application*, with the tag for what I could verify. The column that matters for
us is the last one.

| Source | n (rows) and structure | What the practitioners **did** | Tag | What transfers to the triangle |
|---|---|---|---|---|
| **P5** CDE benchmark | **n = 50 → 20,000**, i.i.d. rows; 39 datasets | benchmarked TFM density outputs against five families of purpose-built density estimators; used **PIT-KS and 90% coverage as first-class metrics**, not only CRPS; concluded that **post-hoc recalibration is a valuable complement** where calibration lags | **verified** (abstract + §4 findings) | (i) The metric set to copy: **PIT + coverage at a stated nominal**, alongside a proper score. (ii) Their own recommendation — post-hoc recalibration — is exactly what `results/fleet/CALIBRATION.md` tested and found *not* to be a result (placebo 5.5% against 4.3% mean gap); their evidence is i.i.d. and ours is ordered, which is the difference that makes a **conditional/ordered** recalibration the live version (§5-B1) |
| **P21** L2C-TabPFN (Alzheimer's, TADPOLE) | **tens of rows per patient-record, panel**; longitudinal clinical sequences | the central move is a **representation transform**: a "longitudinal-to-cross-sectional (L2C) transformation" that "converts sequential patient records into fixed-length feature vectors" before the TFM sees them | **verified** (abstract) | **This is the closest published analogue of what a triangle harness does** — flatten an ordered panel into rows the model can ingest in context. Theirs is classification; ours is regression on a ratio. The lesson is double-edged: it works, and it **discards the order**, which is precisely what the depth-bias finding says we pay for (`+3.2%` per step where the model extrapolates). Any entry sentence about "the triangle as a table" should cite this as precedent, and pair it with our measurement of what the flattening costs |
| **P22** Drift-Resilient TabPFN | tabular, **temporal distribution shifts** | rather than transform the data, they change **what the model learns in context** — an ICL procedure that approximates the Bayesian predictive under temporal drift, evaluated against classical supervised learning | **verified** (abstract) | The one published line that treats *ordered* structure as a modelling problem inside the PFN rather than a feature-engineering problem. It is not runnable on our pinned OSS checkpoint as published; worth naming as the state of the art we are *not* using, and it strengthens the E2b/R5 rationale (the vendor's own answer to ordered data is API-only, per E7 §2.3 C3) |
| **P23** Context sampling on small TabPFN | small tabular classification, 15 OpenML datasets | systematic study of **context size and context composition**: whether larger contexts reduce prediction variability across random draws; whether accuracy depends on preserving the training distribution or on feature-space coverage; whether expensive selection pays | **verified** (abstract) | Directly relevant to **R5/#10 (the fleet as context)**: it says the *choice of context* is a first-class design decision with measurable stability effects — which is exactly the design E2b leaves open (same-line-of-business neighbours · random triangles). It also warns that at small n the context *is* the whole fit (our 6–33 rows, median 25, `results/conditions/rows_per_fit.log`), so there is no sampling choice to make *within* a fit — only across fits |
| **P24** TabPFN for MCI→AD conversion | training sizes **N = 50 → 1,000**, longitudinal-adjacent clinical features | swept training-set size and compared against XGBoost and random forest | **verified** abstract; **result unverified** (full text not read — do not quote an outcome) | A published "TFM under data scarcity" protocol with an explicit **size sweep**. If we ever want the claim "our n is below every published TFM application", this is the paper that bounds it from below at 50 |
| **P4** TabPFN for insurance pricing | large (public MTPL datasets) | raw inputs for TabPFN, preprocessing for GLM/XGBoost; **context-size ladder** (2k → 100k); 5-fold CV; RMSE + deviance + wall-clock | **verified** (full text) | Two transferable warnings: performance can be **non-monotonic in context size**, and fold-to-fold variance is larger than the baselines'. Ours is a different regime (6–33 rows), but the *instability* claim is now published for this model family in insurance |
| **P9** IFoA MLRWP worked example + surveys | a single simulated triangle; survey respondents | mlr3 pipeline with GLM-CL, decision tree, random forest, XGBoost, LASSO; train/validation split inside the triangle; RMSE selection; the authors' caveat that the relative ranking should not be read as a result | **verified** (chapters fetched) | The *practice* baseline: what an actuary with R actually does with ML on a triangle today. It is point-estimate-first, and it makes no uncertainty claim — which is the gap our entry addresses |

**The synthesis for §3, in two sentences.** Published TFM work on scarce data converges on three moves —
**transform the ordered panel into cross-sectional rows** (P21), **change what is learned in context rather
than the data** (P22, P20), or **treat the context itself as a design variable with a stability cost** (P23) —
and the one published insurance evaluation of this model family is negative and stability-focused (P4). Of the
three, our harness already does the first (that is what `direct_features`/`direct_training_rows` do), the
second is API-only in the pinned environment, and the third is the design question E2b owns.

---

## 4. Adjacent methods for our two specific defects

### 4.1 A too-short upper tail

- **Extreme-value / peaks-over-threshold.** Lhaut & Lopez (2026), *Gradient boosting for extremes: sampling
  theory and application to insurance* (`arXiv:2606.14268`) **[verified abstract]**: covariate-dependent
  **Generalised Pareto** estimation in a **peaks-over-threshold** setting by gradient boosting, with an
  orthogonal reparametrisation, cast in an ERM framework with **non-asymptotic error bounds** that separate
  statistical fluctuations, the approximation bias from the GP asymptotics, and the boosting approximation. The
  relevance is architectural: our model cannot grow a tail it has no rows for, and a GP tail is *fitted to the
  exceedances* — a small, well-understood number of parameters — rather than inherited from a rescaled grid. The
  classical actuarial EVT-in-reserving literature exists but was **not reached at source** in this pass (see
  §7); it should be read before any EVT arm is pre-registered.
- **Tail-specific conformal.** Conformalized quantile regression (Romano, Patterson & Candès 2019,
  `arXiv:1905.03222`) **[verified abstract]** is the standard "conformal + quantile regression" construction:
  "fully adaptive to heteroscedasticity", a finite-sample coverage guarantee, and it *corrects a quantile
  regression's own miscoverage* rather than widening globally — which is the difference between it and the
  global widening already measured as "works, and is not a result" in `results/fleet/CALIBRATION.md`.
- **Where conformal is not free here.** All of it assumes exchangeability of the calibration units, or an
  explicit treatment of the dependence (§4.2). Our 464 units are ordered and clustered inside triangles, so a
  plain split-conformal on them would inherit exactly the defect our marginal recalibration already showed
  (placebo 5.5% against 4.3% mean gap).

### 4.2 Ordered and panel data

The i.i.d. versions are what we tried; the ordered versions are a live literature:

| Source | What it does about the ordering | Tag |
|---|---|---|
| **P13** Xu & Xie, *Conformal prediction for time series* (EnbPI) | wraps ensemble predictors, **does not require exchangeability**, avoids data splitting, keeps residuals as a sequential process | verified (abstract) |
| **P15** *Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking* (2026) | the inventory: which CP families exist for sequential data, and how each repairs the violated exchangeability | verified (abstract) |
| **P16** *Copula Conformal Prediction for Multi-step Time Series Forecasting* (ICLR 2024) | keeps **step dependence** in the multi-step case — the multi-step analogue of our multi-step reserve | verified (abstract) |
| **P17** *Relational Conformal Prediction for Correlated Time Series* | exploits observations at **correlated sequences** (graph representations) with conformal + quantile regression — the nearest thing to "units inside the same triangle are correlated" | verified (abstract) |

**Why this is the right family for us and not a detour.** Our units are *steps at a depth* inside a projection,
and their errors are correlated within a triangle and trend with depth; the marginal recalibration failed
because it ignored both. Every one of the four sources above exists precisely because the i.i.d. conformal
guarantee is void under that structure.

### 4.3 Pooling strength across triangles

- **Credibility as a principle for exactly our defect** — P20: the *Credibility Transformer* (EAJ 2025)
  **[bibliographic only]** and its *In-Context Learning Enhanced* successor (2025, `arXiv:2509.08122`)
  **[verified abstract]**, which "increase[s] the information set by a context batch consisting of **similar
  instances**" so that a model "enhance[s] the CLS token representations of the instances by additional
  in-context information". That is the R5/E2b idea — context drawn from *similar* units — published and
  implemented, but inside a transformer trained for insurance rather than a pretrained TFM.
- **Borrowing across insurers' triangles with dependence and heterogeneity** — P19 Cai, Abdallah & Jeganathan
  (2025), *Penalized Copula Mixed Models for Intercompany Loss Reserving and Risk Capital* **[verified
  abstract]**: mixed-effects marginals + company-specific copula dependence, with **an L1 penalty on the fixed
  effects to stabilise estimation in the tail of the loss triangles, where observations are limited**. The
  last clause is our defect named by another literature, and the mechanism (shrinkage where data are thin) is
  the actuarial form of what the depth-bias fix needs.
- **Classical credibility and exposure-anchored priors are already in our pinned environment** — Bühlmann-Straub
  / Cape Cod / Bornhuetter-Ferguson / Benktander, via `chainladder` 0.10.1's own docstrings (E10 §1.6 S12
  **[verified, docstrings]**). R1 in `results/remodel/FINDINGS.md` already proposes the exposure-anchored prior;
  what this pass adds is that **the pooling literature also supplies the *shrinkage* target** for the R5 context
  design (similar units), not only the level.

---

## 5. Ranked, borrowable, and implementable

Ordered by (chance of moving a *measured* defect) × (fidelity to a source we read) ÷ cost. Every entry names the
file its target number comes from, a cost in hours, and a pre-registered expectation **with its falsifier,
written before any run**. Costs are grounded in recorded timings: a fleet coverage pass is **104 min** for 464
units at ~**13.4 s/unit** (`results/fleet/coverage.log`; `results/runs/20260919-pricing/pricing.md`), while
Mack is **0.13 s** and the ODP bootstrap **0.38 s** per unit (≈1.4 and ≈3.0 min for the same 464). **Anything
computed from the recorded draws is minutes of arithmetic and no model call** — that is why the cheap end of
this list is not a compromise.

### B1 — Conformal calibration over *ordered* units, not i.i.d. ones
**Idea, in one sentence.** Recalibrate the model's own intervals with a conformal procedure built for ordered
data (EnbPI-style sequential residuals, or the multi-step copula variant), using the recorded units in
chronology rather than as an exchangeable pool.
**Source and tag.** Xu & Xie, `arXiv:2010.09107` (EnbPI) **[verified abstract]**; `arXiv:2601.18509`
(method inventory) **[verified abstract]**; `arXiv:2212.03281` (multi-step, ICLR 2024) **[verified abstract]**;
Romano et al. `arXiv:1905.03222` (CQR) **[verified abstract]**.
**Targets.** `results/fleet/CALIBRATION.md` — the one-sided tail miss (**14.7% above the 95% bound against 0.9%
below**) and the global widening that "works, and is not a result" (11.8% → 4.3% mean gap, placebo 5.5%).
**Cost.** **2–3 h** (implementation + arithmetic on `results/fleet/coverage.jsonl`; no model calls).
**Pre-registered expectation and falsifier.** Expectation: **sequential/weighted conformal calibration closes
more of the mean coverage gap than the global widening at the same interval width**, and it does so
*conditionally* — the residual gap is not concentrated in the deep units. **Falsifier: if its mean gap is not
better than the 4.3% that the placebo also achieves (5.5%), or if its width inflation exceeds the 2.3–2.7× the
naive widening needed, the ordered-conformal repair is arithmetic in disguise and is dropped.** The
discriminating control is the one `CALIBRATION.md` already identifies as missing: permute each signal while
preserving its marginal, rather than shuffling the actuals.

### B2 — Fit the tail, do not inherit it: a peaks-over-threshold tail on the upper bound
**Idea.** Replace the model's upper bound above a high nominal level with a **Generalised-Pareto tail fitted to
the recorded exceedances**, per horizon, so the tail is estimated from our own residuals rather than inherited
from the rescaled bucket grid.
**Source and tag.** Lhaut & Lopez (2026), `arXiv:2606.14268` **[verified abstract]** (covariate-dependent GP by
boosting, POT, non-asymptotic bounds). *The classical actuarial EVT-in-reserving sources were not reached at
source in this pass and must be read before this arm is pre-registered (§7).*
**Targets.** `results/fleet/CALIBRATION.md` (14.7% above the 95% bound; the excess over the bound is a median of
+36% of the centre, p90 +1.92, max 5.2×) and `results/remodel/FINDINGS.md` §4 (**21.1% of units / 5.3% of
cells** outside the training-label range, one-sided upward).
**Cost.** **3–4 h**: fit a GP above a threshold per horizon on the recorded draws, recompute coverage at
90/95/99 on the held-out half, with the same 231/231 split `CALIBRATION.md` uses.
**Pre-registered expectation and falsifier.** Expectation: the tail-miss rate at 95% falls from **14.7% to
≤5%** at a width increase **smaller than the 2.3–2.7× global widening**, because a fitted GP tail is
tail-shaped where a rescaled grid is not. **Falsifier: if the tail miss does not fall below 10% at a width
increase under 2×, or if the fitted GP's own return level is unstable across the two halves, the grid is not
the binding constraint on the bound and R2/R3 in `results/remodel/FINDINGS.md` remain the only live tests.**

### B3 — A conditional recalibration that uses what the model already emits (the discriminating version of #21)
**Idea.** Fit the coverage correction on half the units *conditioned on fields the model already emits* —
horizon, triangle size, its own disagreement with Chain Ladder, its interval width — and test on the other
half, with a feature-permutation control.
**Source and tag.** P5 (`arXiv:2603.26611`) **[verified]** — its own conclusion that "post-hoc recalibration
may be a valuable complement", and its measurement that TFM calibration is not uniformly good; the *design* is
ours (`results/remodel/FINDINGS.md` §4, item 2).
**Targets.** `results/fleet/CALIBRATION.md` (the placebo's 5.5% against the fix's 4.3% — the reported reason no
learnable correction was found) and `results/fleet/COVERAGE.md` (the under-coverage at every level, z −4.8 to
−6.6).
**Cost.** **1–2 h**, arithmetic on the recorded units; the permutation control is the only care needed.
**Pre-registered expectation and falsifier.** Expectation: **the upper-tail miss is unit-predictable** — a
conditional correction beats the placebo by more than the 1.6% draw-noise floor on the median. **Falsifier: a
permutation control matching it within that floor → the miss is not unit-predictable from what the model emits,
the conditional family closes, and the entry says so.**

### B4 — Constrain the projection during the fit (the neural-reserving answer to our depth bias)
**Idea.** Take P2's remedy literally: add a **penalty on the projected central estimates** beyond a stated
depth (non-negativity, monotone convergence toward the observed development pattern) and an MSE term alongside
the NLL, so the model is discouraged from the drift the depth-bias diagnostic measures.
**Source and tag.** P2 §4.2 and §6.1 **[verified, full text]**: "The practitioner can place upper and lower
bounds on the central estimates of any desired set of cells … and penalise the MDN if its central estimates
fall outside those boundaries", applied to ~10% of lower-triangle cells, with the measured result that the
upper-triangle fit was "virtually unchanged"; plus the NLL+MSE finding (environment 2).
**Targets.** `results/runs/20260918-023600_depth-bias/FINDINGS.md` — **+3.2% per step (se 0.36%)**, ×1.33
compounded against a measured ×1.67 over-reserve — and, through it, `results/fleet/FINDINGS.md`'s median move
off Chain Ladder of **141.8%**.
**Cost.** **4–6 h**: an opt-in constraint in the direct arm (the pattern `results/runs/20260919-config-cells`
already establishes for opt-in model changes: one factor per cell, default path unchanged, pinned by tests),
then a re-run of the fleet. **A re-run is 104 min of CPU per configuration** — a single cell, not a grid.
**Pre-registered expectation and falsifier.** Expectation: the per-step slope for the constrained arm is
**below the unconstrained +3.2%** by more than its standard error, with the depth-0 intercept staying at zero.
**Falsifier: if the slope is unchanged within 2 se (≈0.72pp), the drift is not a projection-geometry problem
the model can be penalised out of, and the remaining routes are the target parameterisations (R1/R2).**

### B5 — A heavier-tailed target, chosen for the tail rather than for the skew
**Idea.** Follow P2's own remedy list and E10's R2: predict a quantity whose *support* matches the risk being
read — a mixture **Log-Gaussian**-style target on a log scale, or the bounded share-of-ultimate (support in
[0,1]) — so the tail is generated by the target's family rather than by a rescaled grid.
**Source and tag.** P2 **[verified, full text]**: the mixture Log-Gaussian "helps to address the practical
limitations to the flexibility of the mixture Gaussian by providing a **positive, heavier-tailed option**", and
it "linearises the data, which can make training simpler"; same finding, different task, in the vendor's own
insurance cookbook (quoted in `results/remodel/FINDINGS.md` §1.3, **[verified quote, unverified as a
transfer]**).
**Targets.** `results/fleet/CALIBRATION.md` (the one-sided tail) and `results/fleet/COVERAGE.md` (the centring
failure — the truth exceeds the sampled median in **67.7%** of units).
**Cost.** **3–5 h**: a target transform + a fleet pass at 104 min per configuration, one cell.
**Pre-registered expectation and falsifier.** Expectation: **coverage at 90/95 moves toward nominal before any
widening is applied**, from 78.0% / 84.7%. **Falsifier: if 95% coverage stays within 2 binomial SE (≈3.4pp at
n = 464) of 84.7%, the target-family explanation is dead and the defect is in the model's scale estimate, not
its shape.** (This is R2's pre-registration in `results/remodel/FINDINGS.md`; this pass adds P2 as its
published precedent and the log-mixture as its second form.)

### B6 — The fleet as context, designed as a *similar-units* batch
**Idea.** Run E2b/#10 with the context composed of **similar** units rather than any others — same line of
business, similar size, same maturity — which is the design P20's in-context credibility transformer publishes
as "a context batch consisting of similar instances".
**Source and tag.** P20 (`arXiv:2509.08122`) **[verified abstract]**; **P23** (`arXiv:2607.26628`)
**[verified abstract]** on context composition having measurable stability effects; P19 **[verified abstract]**
on shrinkage where triangles are thin. The leakage guard stays ours (`docs/experiments.md` §2, R1–R5).
**Targets.** `results/conditions/rows_per_fit.log` (**6–33 training rows per fit, median 25**) and through it
the 38.8% paired rate in `results/fleet/FINDINGS.md`.
**Cost.** **> 1 day** — the context builder is new code, and the guards are load-bearing. **Listed here because
it is the highest-ceiling idea, not because it is a candidate for the next slot** (§5b).
**Pre-registered expectation and falsifier.** Unchanged from §5.4: paired on the same units against the direct
arm, must clear **38.8% ± 2.3pp** and 1.6% on the median; the mandatory replication cell is `abc` →
**5,211,802** (`results/runs/direct-arm.md`); **if the cross-triangle null scores better than the self-fit, the
guard is broken — stop and do not interpret.**

**Two pre-registered dependencies, to stop the list from being read as six independent bets.** B1 and B3 both
consume the recorded draws and share the permutation control, so they should be run as one sitting (B3 first —
it is cheaper and it decides whether *any* unit-conditional correction exists). B2 and B5 are the two live
answers to the tail and should be compared on **the same held-out half** as `CALIBRATION.md` uses, in ratio
space, so the width cost is comparable. B4 changes the model and therefore invalidates comparison with the
recorded numbers unless its baseline cell reproduces them.

### 5b. Out of scope — more than a day, marked rather than ranked

| Idea | Why it is out of scope |
|---|---|
| **Thinking mode on the 464 units** (E7 §6, R7) | API-only in the pinned environment and metered in tokens — a spend decision, not an experiment slot; it also changes what the entry is (zero-shot, local, CPU) |
| **Fine-tuning on actuarial triangles** | GPU hours, and it contradicts the entry's zero-shot claim (`docs/experiments.md` §5.5) |
| **Cross-triangle context (B6) as a first move** | new build + five guard tests + ≈2 h compute; the highest ceiling and the highest risk of a meaningless number |
| **A threshold ladder with a rebuild from the survival curve (R3)** | ≈ 464 × rungs fits: at ~13.4 s/unit a 9-rung ladder is ≈ 6 h of CPU and needs a new label vector per rung, so it is a day of work even unattended |
| **Training a credibility transformer or an MDN-equivalent on our data** | P20/P2 are architectures to *train*; our entry's claim is zero-shot, and the training data are 775 small triangles |

### 5c. The negative space — attractive, and not worth running here

| Idea | Why not |
|---|---|
| **More draws per unit to fix coverage** | draw noise is ≈1.6% on the median at 300 draws (`docs/experiments.md` §5.4); the gaps being fixed are 10–15 percentage points |
| **Widening the intervals** | already measured: 11.8% → 4.3% mean gap with a 2.3–2.7× upper widening and a placebo at 5.5% (`results/fleet/CALIBRATION.md`) — arithmetic on widths, and it changes no claim |
| **Re-tuning hyper-parameters** | the model has none to set per triangle, and the vendor's two documented configuration changes were tested and moved the centre, never the tail (`results/runs/20260919-config-cells/FINDINGS.md`) |
| **A different loss column** | measured: the failure is not column-specific (paid 36.2% against incurred 38.2% on the 414-unit intersection, paired −1.9% with se 3.3%, inside the 2.3pp floor) — `results/runs/20260919-column-comparison/FINDINGS.md` |
| **Blending the model with Chain Ladder / Mack / the bootstrap** | makes the model a meta-learner, which the rubric punishes at 50% weight (`docs/experiments.md` §5.5); P25 is the actuarial way to do it and it needs trained stochastic components we do not have |
| **Row subsampling / `"majority_downsample"`** | documented for tables above the row limit and for zero-heavy targets; our fits are 6–33 rows with no zero mass in cumulative figures |
| **Exposure-normalised targets (pure-premium style)** | the vendor's own cookbook documents it as the trap that "blows up the tail" (E10 §1.3) |
| **Copying P21's L2C transform as if it were new** | our harness already does the flattening; the publishable version is to **measure what the flattening costs** (the depth-bias diagnostic) rather than to re-derive it |
| **Adopting P22 (Drift-Resilient TabPFN) off the shelf** | not the pinned OSS checkpoint; it is a different trained model, so it would replace the entry's subject rather than improve it |
| **Sampling-based tail stress tests (synthetic triangles)** | reads as method-for-method's-sake, and our defect is measured on real triangles already |

---

## 6. Corrections this pass forces on our own writing

1. **Attribute the mechanism.** Any sentence that presents the fixed bucket grid rescaled to the training
   target's mean and sd as *our* explanation of the one-sided tail must name the vendor's documentation as its
   source (`results/remodel/FINDINGS.md` §1.1 already does; `docs/submission.md` says "documented *and* verified
   in the installed code", which is right).
2. **Attribute the general small-n calibration result.** `docs/submission.md`'s calibration bullet is correct as
   written ("we know of no prior measurement of this kind for reserving") but must add the attribution to
   `arXiv:2603.26611`, which measures PIT calibration and 90% coverage for tabular foundation models from n = 50
   upward, and to the vendor's own OOD warning, so the claim cannot be read as "nobody has looked at this
   family's calibration".
3. **Narrow the "first" claims to what survives**: *first coverage measurement for a foundation model on loss
   triangles*; *first per-depth bias measurement on a reserving recursion, with the anchoring trap*; *first
   comparison against the classical intervals on identical units* (**still unrun** — `docs/experiments.md`
   §5.2/#20). Anything broader than those three is not supported by this search.
4. **The neural-reserving literature is not a point-estimate literature, and must not be described as one.**
   P2 (MDN) and P25 (distributional ensembles) both take the distribution seriously; they evaluate it with
   quantile/log scores rather than coverage. The entry's framing should be *the literature scores distributions
   but does not audit their coverage*, not *nobody has modelled uncertainty*.
5. **The one published insurance evaluation of this model family is negative (P4) and should be cited as such.**
   It strengthens the entry's negative and pre-empts "did you check whether anyone else found this?" — and it
   carries the two error classes this task asked about (§1.4), which is a reason to cite it carefully rather
   than to adopt its wording.
6. **No correction is needed to our own size-limit statements** (Check 1) — but the sweep is worth repeating
   whenever a new doc quotes a ceiling: the vendor's number is a **joint** row × column budget with an explicit
   trade-off, and a row number alone understates how the limit binds.

---

## 7. What could not be reached, and what that costs

- **P10, Taylor & McGuire (2023), Risks 11(11):185 — full text.** MDPI returns 403 to this box (and the browser
  fallback is unavailable here: no Chrome is installed). Only the published abstract was read. **Consequence:
  the strongest candidate for "someone already measured an ML method's reserving uncertainty" is unexamined in
  its detail**, and it is the one item that could still kill the entry's "first coverage measurement" framing.
  It must be read before submission.
- **P11, Taylor (2019) Risks 7(3):82 — full text**, same cause; abstract only.
- **P12, Ben Taieb & Atiya (2016) — the abstract.** The bibliographic record is verified; the paper's own
  wording of the bias/variance decomposition is not. Any sentence that attributes a *statement* to it needs the
  paper first.
- **The classical EVT-in-reserving literature, and Mack's-coverage studies, at source.** Titles/venues/DOIs were
  verified through Crossref; contents were not read. §4.1's EVT arm (B2) therefore carries a note, not a
  citation.
- **The CAS E-Forum / Annals of Actuarial Science GLM-and-GBM reserving studies.** The searches returned the
  textbooks and the survey (P11) rather than a primary GBM reserving study; the practice-level source actually
  read here is the working party's own mlr3 chapter (P9). **So the "gradient boosting / GLM reserving study"
  row of §1.1 is incomplete**: it rests on the working-party material and on P4's GBM baselines, not on a
  primary GBM-on-triangles paper.
- **The Nature TabPFN paper itself** (Hollmann et al. 2025, *Nature* 637, 319–326). Not fetched (paywalled); its
  documented limit is quoted here from the vendor's own current table (P6) and from the 2.5 report's restatement
  of the v2 scaling (P8) — both the vendor's. The §1.4 Check 2 conclusion rests on *the vendor's* record of what
  that paper's model supports, which is the right frame for an attribution check but is not the paper.
- **`web_search` itself** (all four backends failed) and the headless browser (no Chrome). Recorded because the
  search is part of the result: the queries in §0.3 were run against the sources directly, and a reviewer with
  working search backends could still find something this pass did not.
- **The vendor's `improving-performance` pages** were not re-read here; E7 and E10 read them at source, and this
  report relies on their quotations rather than re-verifying them.

---

## 8. Provenance

- This report: `results/prior-art/FINDINGS.md`. **Reading and analysis only — no model was fitted, no repository
  code was changed, and no result is claimed.** The only numbers of ours in it are quoted, with their file:
  `results/fleet/FINDINGS.md` (38.8%; |error| medians 162.6% and 121.3%; median move off CL 141.8%; direct arm
  on `abc` −1.2% against the recursive arm and the `5,211,802` replication value from `results/runs/direct-arm.md`),
  `results/fleet/COVERAGE.md` (39.2 / 59.9 / 78.0 / 84.7%; z; the 67.7% centring line; 4.55× interval width),
  `results/fleet/CALIBRATION.md` (14.7% / 0.9%; the +36% median excess; 11.8% → 4.3% with a 5.5% placebo;
  2.3–2.7× widening), `results/runs/20260918-023600_depth-bias/FINDINGS.md` (+3.2%/step, se 0.36%, ×1.33 vs
  ×1.67, the −0.0018/+0.0190 trap), `results/conditions/rows_per_fit.log` (6–33 rows, median 25),
  `results/conditions/FINDINGS.md` (the vendor conditions and the benchmark scope statements),
  `results/remodel/FINDINGS.md` (21.1% of units / 5.3% of cells; R1–R8), `results/runs/20260919-config-cells/FINDINGS.md`
  (the configuration cells), `results/runs/20260919-column-comparison/FINDINGS.md` (paid 36.2% vs incurred 38.2%,
  paired −1.9% se 3.3%, the 2.3pp floor, and the stale-field defect), `results/runs/20260919-pricing/pricing.md`
  (0.13 s / 0.38 s / ~13.4 s per unit; 104 min), `results/runs/20260919-target-support/`.
- External sources were fetched 2026-09-19 as described in §0.2, and are listed with identifiers in §0.1. Quotes
  are reproduced from those fetches; where a claim is quoted from a *search result* rather than the document, it
  is tagged **unverified** at the claim and named in §7.
- No new measurement was produced by this pass; there is nothing here to reproduce beyond the quotes, and every
  quote's URL is in §0.1.
