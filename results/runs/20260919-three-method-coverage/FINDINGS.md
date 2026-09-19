# E3a — do Mack and the ODP bootstrap cover on the same 464 units? (#20)

The entry's every claim was *about our model*: it is closer than Chain Ladder on 38.8% of 464 evaluations and
its intervals under-cover at every level — 39.2 / 59.9 / 78.0 / 84.7% against nominal 50 / 75 / 90 / 95
(`results/fleet/COVERAGE.md`). Nothing measured whether the **standard methods** cover on those same units, so
no comparative claim was possible. This measures them, paired, on the identical units and realised futures.
The model was not refitted: its side is read from `results/fleet/coverage.jsonl` as recorded.

**The pre-registered rule** (`docs/experiments.md` §5.2): a gap is real when it exceeds **2 binomial SE**
(≈2.2pp at n=464). **(a)** Mack/ODP within 2 SE while ours is 5–7 SE off → the comparative claim is earned.
**(b)** All three outside 2 SE → the finding stops being about TabPFN and becomes about the field.

**The verdict, stated explicitly: neither branch as pre-registered, and the honest reading is nearer (b) than
(a).** All three methods under-cover on these units, Mack at every level by −6.2 to −15.7pp (−2.7 to −15.5 SE),
and the ODP bootstrap at three of four levels. The model is not uniquely miscalibrated — it is the *worst*
calibrated of the three at every level, which is a weaker and different claim from (a). Branch (a) is
excluded: no standard method lands within 2 SE throughout. Branch (b) is *not* satisfied as written, because
one method at one level does: the ODP bootstrap at 75% (−0.2pp, −0.11 SE). If (b) is restated as "no method
covers as advertised at the levels a risk margin is read (90/95)", then (b) holds and all three methods fail
(ODP at −2.1 and −4.0 SE; Mack at −11.1 and −15.5 SE). The rule was pre-registered on the levels, not on a
reading of them, so this is reported as a third outcome rather than dressed up as either branch.

## What was measured, and on which cells

`results/runs/20260919-three-method-coverage/per_unit.jsonl` — **464/464 units**, the same units, column
(`IncurLoss`), held-out diagonals and realised futures as the model's run. Both classical methods are fitted
on the **as-of-anchor triangle** — the same truncation the harness uses, `Triangle.known(anchor)` — and
evaluated on the projection to the **target diagonal** (every origin exactly `h` steps past its last observed
age), which is the future a coverage unit scores. `src/tabpfn_reserving/classical.py` states the conventions;
three of them are worth naming because they are choices:

1. **The classical methods had to be asked for the same cells, not their own headline number.** A classical
   interval published for a triangle is the *run-off* reserve, and on a unit that is a different quantity: on
   the first unit of the run the as-of triangle's run-off reserve is 422.39 where the unit's realised future
   is 7.00. Running the classical methods against that would not have been a comparison. Both are therefore
   evaluated at the unit's target diagonal.
2. **Ages the as-of triangle has no observation for** — needed by the oldest `h` origins of every unit, 1455
   of whose steps are past the last diagonal — are projected with the last estimated factor **carried
   forward**, for the point and the standard error alike, exactly as `Triangle.global_factors` does. On 464/464
   units a carried step was used; its share of the classical point is a median 2.1% (max 92.0% on the smallest
   triangles, where the carried origin *is* the reserve).
3. **Mack's recursion is re-implemented so it can stop at each origin's target**, and validated against
   chainladder's own output: stopped at the triangle's ultimate it reproduces `ibnr_`, `mack_std_err_`,
   `process_risk_` and `total_process_risk_` to 1e-9 (`tests/test_classical.py`). Mack's intervals are
   normal-based (`point ± z·se`) — the method's own assumption, stated rather than adjusted.
   `chainladder.total_parameter_risk_` is deliberately **not** used: its operand ("every cell from the latest
   valuation date") is not Mack's derivative sum, and on the first unit it gives a run-off total SE of 350.13
   against the consistent recursion's 315.54. The aggregate used here is the one that coincides with
   chainladder's per-origin recursion in the run-off case.

**The bootstrap is the pre-registered one**: `BootstrapODPSample(n_sims=1000, random_state=20260919)` on the
as-of triangle, then a `Chainladder()` refit on every resample, `nansum` on the ensemble (`.sum()` is all-NaN
on the unobserved cells — an earlier attempt at this number was withdrawn for that reason). 1000 finite draws
on every unit, 0 non-finite across the run.

## The result

Coverage at each level; the model's columns are its own recorded numbers, so the whole table is one
comparison and not three runs.

| nominal | TabPFN-3.5 | Mack | ODP bootstrap | 2 SE (pp) |
|---|---|---|---|---|
| 50% | **39.2%** (−10.8pp, −4.6 SE) | **43.8%** (−6.2pp, −2.7 SE) | **55.2%** (+5.2pp, +2.2 SE) | 4.6 |
| 75% | **59.9%** (−15.1pp, −7.5 SE) | **63.4%** (−11.6pp, −5.8 SE) | **74.8%** (−0.2pp, −0.1 SE) | 4.0 |
| 90% | **78.0%** (−12.0pp, −8.6 SE) | **74.6%** (−15.4pp, −11.1 SE) | **87.1%** (−2.9pp, −2.1 SE) | 2.8 |
| 95% | **84.7%** (−10.3pp, −10.2 SE) | **79.3%** (−15.7pp, −15.5 SE) | **90.9%** (−4.1pp, −4.0 SE) | 2.0 |

Mack is the **worst** of the three at 90% and 95% — worse than the model whose miscalibration motivated the
experiment — and the bootstrap is the best at every level while still outside 2 SE at three of them.

## The paired comparison, which is what the units exist for

| nominal | ours miss | Mack misses too | ODP misses too | Mack's misses | ours miss too |
|---|---|---|---|---|---|
| 50% | 282 | 67.0% | 56.0% | 261 | 72.4% |
| 75% | 186 | 53.8% | 37.6% | 170 | 58.8% |
| 90% | 102 | 50.0% | 29.4% | 118 | 43.2% |
| 95% | 71 | 52.1% | 26.8% | 96 | 38.5% |

The failures are **shared, not distinctive**. At 95%, of the 71 units the model misses Mack misses 37, and
Mack misses 96 units of which the model misses 37: the two miss overlapping but different halves, and each
covers many units the other does not (59 units the model covers and Mack misses; 34 the reverse). A model
whose interval failures were its own would miss where the standard method is right; this one misses in the
same weather.

## Width: covering by being enormous is not covering

Median interval width at the 95% level, relative to the point estimate.

| method | width / own point | width / Chain Ladder point | width / model point |
|---|---|---|---|
| TabPFN-3.5 | **6.38** | 11.47 | 6.38 |
| Mack | **7.37** | 7.37 | 4.10 |
| ODP bootstrap | **28.12** | 24.57 | 15.12 |

The bootstrap buys its better coverage with intervals ~4× wider than Mack's and ~4.4× wider than the model's;
at 28× the point estimate it is close to uninformative as a risk margin. The **interval score** (a proper
score: width plus a penalty for a miss) makes the trade explicit, and the ranking inverts:

| nominal | TabPFN-3.5 | Mack | ODP bootstrap |
|---|---|---|---|
| 50% | 107,231 | **64,626** | **42,879** |
| 75% | 82,720 | 51,358 | **22,875** |
| 90% | 74,493 | **42,020** | 44,687 |
| 95% | 850,661 | **37,866** | 115,134 |

The model's 95% score is dominated by units where its interval is both very wide and wrong — the same failure
`COVERAGE.md` describes, now visible as a scoring loss rather than a coverage gap. The bootstrap wins at 50%
and 75% on width; at 90% and 95% Mack wins outright.

## Why the classical intervals miss: the same mechanism, and not the normal assumption

**It is not the normal assumption.** Mack's intervals are normal-based, so its under-coverage could have been
a distributional artefact. It is not: rescaling Mack's standard error to whatever multiple would reproduce
nominal two-sided coverage gives an **implied z of 0.640 / 1.225 / 1.875 / 2.620 at 50/75/90/95** — within
−5%/+7% of nominal at 50% and 75%, +14% at 90% and **+34% at 95%**, rising with horizon (2.04 SE at one step,
2.63 at two, 2.96 at three). Mack's standard error is roughly the right size near the centre and too small in
the tail — the same signature our model shows, and the same one `results/fleet/CALIBRATION.md` found for it.

**It is centring.** The truth sits above the classical point estimate in **77.6% of Mack's units** (median
`(actual − point)/|point| = +145%`) and 77.8% of the bootstrap's (+159%) — against the model's 67.7% at
`/docs/submission.md`'s recorded +122%. Chain Ladder-based reserves are too low on these triangles, and both
classical methods inherit it. Both are *less* centred on the truth than the model, and both still cover
better, because their spread is wider.

**And the tail is one-sided for all three**, truth above the method's own 95% upper bound against a nominal
2.5%:

| method | above the 95% bound | below the 95% bound |
|---|---|---|
| TabPFN-3.5 | **13.8%** | 1.5% |
| Mack | **18.8%** | 1.9% |
| ODP bootstrap | **7.3%** | 1.7% |

The model's 13.8% is `results/fleet/CALIBRATION.md`'s **14.7%** "on the held-out half of the fleet" — a
different slice of the same run, not a contradiction; this table is over all 464 units. Mack's upper tail is
shorter still. **On the units where a risk margin is actually read, the standard method is worse than ours,
not better.**

## The bootstrap's own correctness check

The pre-registered check is that the bootstrap's mean lands within ~0.2% of the Chain Ladder point (1.002 on
`abc`). Where it can be measured cleanly, it does — **on `abc` the ratio is reproduced in
`tests/test_classical.py`**, 1000 finite draws. On a fleet unit the mean is not a usable check and the honest
number is the robust one:

- the **median simulation** against the same Chain Ladder point at the h-step target: median **0.9351**, IQR
  [0.671, 1.017] — near 1, as a working bootstrap of Chain Ladder should be;
- the **run-off** mean/CL ratio is tail-dominated on **106 of 464 units** (|ratio| > 10; per-unit median
  1.0251, IQR [−0.20, 2.41]). A 3×3 triangle with a near-zero denominator, resampled 1000 times, has a
  sample mean with no useful precision — the tail, not the workflow. **No finite-draw failure exists in the
  run** (0 non-finite draws of 464,000), and a construction failure would show as infinities or NaNs, not as a
  finite mean. This is reported as a limit of the diagnostic, not as a pass or a failure.

## What this does and does not change

**It kills the comparative claim the entry might have wanted.** "This method's intervals are wrong where the
standard one's are right" is not available: at 90% and 95% the standard method is worse. Any sentence
contrasting our interval failure with a calibrated incumbent would be false, and `docs/submission.md`'s
existing caution against a comparative claim was correct.

**It replaces it with a sharper one, about the field.** On these 464 real evaluations — CAS Loss Reserve
Database triangles, incurred losses, one to three diagonals held out — none of the three methods covers as
advertised at the levels a risk margin is read, and the failure has the same shape in all three: a short
**upper** tail (13.8 / 18.8 / 7.3% above the 95% bound against a nominal 2.5%), the truth above the point
estimate nearly four times out of five, and a required multiplier on the standard error of ~1.3–1.5× at 90%
and ~1.6–3× at 95%. Our model's failure is *evidence about the difficulty*, not an aberration — which is
branch (b)'s substance, reached without (b)'s letter.

**What it does not say.** Coverage is one property. It does not say Mack or Chain Ladder is a bad method —
they are the industry's baseline, and their point estimates beat the model's on these triangles
(`results/fleet/FINDINGS.md`). It does not test calibration conditional on unit size or reserve magnitude; the
units range from a realised future of a few currency units to millions, and coverage is a binary so that
range is invisible in the rate. And the classical evaluation is at the harness's truncation, so these are not
the numbers a practitioner would publish for a full triangle — they are the numbers a coverage comparison
requires.

## Provenance

`results/runs/20260919-three-method-coverage/`: `per_unit.jsonl` (464 rows: unit, both classical methods'
points, standard errors, all four levels' bounds per method, coverage flags, widths, the model's recorded
bounds and flags, interval-score inputs), `run.log` (the run's own output, warnings included),
`screen.jsonl` and `screen-manifest.json` (the first **80** units — the subset size was fixed at 80 before any
coverage number was compared, and the screen's only purpose was mechanics; it ran and passed before the full
run started), `manifest.json` (git revision `c9c8bc8`, branch, `dirty: true` — the run used the working tree
that became this commit —, `chainladder` 0.10.1, 1000 simulations, seed, levels, threshold).

Reproduced: `python scripts/three_method_coverage.py` (464 units, 8.0 min; Mack 0.09 s and the full ODP
workflow 0.91 s per unit at n≈10, consistent with the pricing run's 0.13/0.38 s — the ODP figure is higher
because this workflow also refits Chain Ladder on the as-of triangle per unit); the same command over the
first 60 units reproduces every recorded field of those 60 units **exactly** (0 differing of 60 × 46 fields),
and `--summary-only` reproduces the tables. `pytest -q tests/` green, including the new `tests/test_classical.py`
(9 tests: the Mack recursion against chainladder's run-off output, the horizon point against the harness's own
`factor_reserve`, interval ordering, the bootstrap's `abc` check, and bootstrap reproducibility).

**The script refuses a verdict below a stated threshold** (`MIN_UNITS_FOR_VERDICT = 200`, fixed before any
number was computed) **and when the outcome has no variation** — the 80-unit screen prints its tables and
refuses; this report's verdict stands on 464 units with variation at every level.
