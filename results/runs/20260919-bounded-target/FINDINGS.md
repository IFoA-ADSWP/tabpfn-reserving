# R2 (#24) — the bounded-support target: the one-sided tail gets **worse**, not better

> **Status:** a screening run, 2026-09-19, **75 units × 2 cells**, one factor between them, on the recorded
> coverage units. The change was made, it ran, and the pre-registered expectation is **not met**: 95% coverage
> moved **14.0pp away from nominal** (88.0% → 70.7% on this subset) with a paired McNemar p of **0.0023**,
> and the one-sided upper-tail miss **doubled** (12.0% → 29.3% of units above the model's own 95% upper bound,
> nominal 2.5%).
>
> **The pre-registered falsifier does not fire, and its expectation is not met either.** The falsifier's
> condition was *"if 95% coverage stays within 2 binomial SE of the recorded 84.7%"* — it does not (70.7% is
> outside [76.4%, 93.0%]) — and the expectation was that coverage would move **toward** nominal. Neither
> branch is the outcome, so the support explanation of the tail is **neither confirmed nor killed by this
> run**; what is closed is **R2 as pre-registered**, because the instrument failed to isolate the factor it
> was built to isolate: the reconstructed **level** broke at the same time as its support changed.
>
> The replication cell did its job: **all 75 of its rows are identical to the recorded rows in
> `results/fleet/coverage.jsonl`** on every one of the 23 fields the two files share (1e-9 relative), with 0
> coverage-flag mismatches — and identical again to #22's own baseline cell, a second process on the same
> units. The harness did not change, so both cells below are differences, not two implementations.

## What was run

One change, made opt-in so no recorded number moves: `direct_training_rows(..., target="unrevealed")` labels
each training row with **the share of the target level still to emerge**, `1 - C[a, age] / C[a, target_age]`
(both cells observed, under the existing `known_until` clamp), and `arm.reserve_direct(..., target=
"unrevealed")` rebuilds `IBNR(origin) = predicted_share × ultimate(origin)` **with the ultimate taken from
Chain Ladder** — deliberately, so the level stays where it was and only the support of the predicted quantity
changes. The labels are **not clipped** into `[0, 1)`: a falling cumulative gives a negative share and is
reported as one. `arm.make_model("local")` is built with no options.

| cell | what the model predicts | reserve rebuilt as |
|---|---|---|
| `baseline` | the development factor relative to Chain Ladder (the recorded `delta` target) | `Σ base·(factor·cl − 1)` |
| `unrevealed` | the share of the ultimate still to emerge, in `[0, 1)` where the cumulative rises | `Σ share · (base·cl)` |

Everything else is held fixed and identical to `scripts/fleet_coverage.py`: `IncurLoss`, three diagonals held
out, the same anchors and training anchors (`range(2, max(3, anchor))`), the same `known_until` clamp, the
same features, 300 draws per unit from the bar distribution, `default_rng(0)` re-seeded per unit. The two
cells differ **only** in the target and the reconstruction — this is not a #22-style configuration cell.

`scripts/unrevealed_target.py` runs one cell per invocation, is resumable, locks its cell file against a
second writer, and `--report` pairs the cells unit by unit.

## The subset, fixed before any coverage number was read

**The first 75 rows of `results/fleet/coverage.jsonl`**, taken from that file rather than re-derived, so the
two cells and every re-run score identical units. 75 and not 150 because the rate was measured first: fits on
this box ran at **33–65 s per unit** (load average 12–40 on 8 cores, sibling kanban work alongside), against
~13.5 s on a calm one, so 464 units is ~5 h for a single cell. 75 is also the size and the units #22 used, so
this run's baseline can be checked against that run's as well as against the record. The subset was fixed,
and `PREREGISTRATION.md` written into this directory, before either cell was started; no coverage number
existed when that was done.

**Binomial sampling error at n=75, on the recorded rates: 4.2pp at 95% and 4.8pp at 90%** (11.3pp at the
worst case, p=0.5). The pre-registered bars follow from it: the expectation needed 95% coverage above
**88.9%** (or 90% above 82.8%), and the falsifier's death condition was an outcome inside **[76.4%, 93.0%]**
at 95%.

## The replication cell

`baseline` against the record, on the same 75 units:

```
rows compared against results/fleet/coverage.jsonl : 75
fields compared (every field the two files share)  : 23
rows with any field outside 1e-9 relative          :  0
coverage-flag mismatches at 50/75/90/95            :  0
50%: recorded 40.0%  baseline 40.0%     75%: 62.7% / 62.7%
90%: recorded 78.7%  baseline 78.7%     95%: 88.0% / 88.0%
against #22's recorded baseline cell: 75 units shared, 0 with any field outside 1e-9 relative
```

The recorded full-fleet coverage is 39.2% / 59.9% / 78.0% / 84.7%; **the subset's own record is
40.0% / 62.7% / 78.7% / 88.0%**, and the falsifier is applied to the recorded 84.7% as pre-registered while
the paired comparison is against the subset's own baseline. Two independent processes — this run's and #22's
— produced byte-identical rows for the same 75 units, which is the "same command twice = the same number"
check done where it costs nothing.

## The coverage table (n=75, the same units in both cells)

| cell | 50% | 75% | 90% | 95% |
|---|---|---|---|---|
| nominal | 50% | 75% | 90% | 95% |
| `baseline` | 40.0% (−10.0pp) | 62.7% (−12.3pp) | **78.7% (−11.3pp)** | **88.0% (−7.0pp)** |
| `unrevealed` | 25.3% (−24.7pp) | 44.0% (−31.0pp) | **60.0% (−30.0pp)** | **70.7% (−24.3pp)** |

Every level moves **further** from nominal, and the 95% interval no longer contains the nominal value at any
level on this subset.

## Paired against the replication cell, unit by unit

`fixed` = units the baseline missed and this cell covered; `broke` = the reverse; `p` is McNemar's exact test
on the discordant pairs. Two separate rates would hide that this is the same 75 units.

| level | baseline | unrevealed | difference | fixed | broke | McNemar p |
|---|---|---|---|---|---|---|
| 50% | 40.0% | 25.3% | −14.7pp | 6 | 17 | 0.0347 |
| 75% | 62.7% | 44.0% | −18.7pp | 3 | 17 | 0.0026 |
| 90% | 78.7% | 60.0% | **−18.7pp** | **4** | **18** | **0.0043** |
| 95% | 88.0% | 70.7% | **−17.3pp** | **2** | **15** | **0.0023** |

The movement is one-sided at the unit level as well as in rate: at 95% **15 units broke and 2 were fixed**,
at 90% **18 broke and 4 were fixed**. This is not sampling noise, and it is not the sign the pre-registration
predicted.

## The one-sided tail, read directly

| cell | level | above the upper bound | below the lower bound | nominal |
|---|---|---|---|---|
| `baseline` | 90% | 15/75 = 20.0% | 1/75 = 1.3% | 5% / 5% |
| `baseline` | 95% | 9/75 = **12.0%** | 0/75 = 0.0% | 2.5% / 2.5% |
| `unrevealed` | 90% | 29/75 = **38.7%** | 1/75 = 1.3% | 5% / 5% |
| `unrevealed` | 95% | 22/75 = **29.3%** | 0/75 = 0.0% | 2.5% / 2.5% |

The upper-tail miss **doubles** (12.0% → 29.3% at 95%) and the lower tail stays empty. The failure the change
was aimed at is the failure it **amplified**, in the same one-sided direction.

## Why: the level broke, and it broke downward

The centre, which the pre-registration expected to be broadly unchanged:

| cell | median (point − actual)/actual | median |error| | units with a **negative** point estimate |
|---|---|---|
| `baseline` | −0.34 | 1.82 | 33.3% |
| `unrevealed` | **−2.07** | 2.40 | **72.0%** |

The reconstruction is systematically too low, and on **72% of the units it is below zero** — a reserve cannot
be negative. Two facts in the cell's own record explain it, and both are properties of the specified
instrument on this column rather than of the model's fit:

* **The share is a signed quantity here, not a fraction in `[0, 1)`.** Of the **1,809 training labels** the 75
  fits were given, **869 (48.0%) are below zero**, because cumulative incurred falls between the training
  diagonal and the evaluation diagonal on those rows (case reserves released). The label is a difference of
  two levels, so a fall gives a negative share; the target is bounded **above** at 1 by construction and
  unbounded below, and the models are fitted on a label set whose centre is near zero (per-unit maximum
  label: median 0.291, largest 0.860).
* **The reconstruction hangs that signed share on a level that is not signed.** `share × (base·cl)`, with
  `base·cl` the Chain Ladder ultimate, turns a small negative share into a negative reserve of the same
  magnitude as the reserve itself. A model whose predicted shares are negative where the realised shares are
  slightly positive puts its whole interval below the truth — which is exactly the amplified upper-tail miss
  above.

**The support idea itself did what it says on the label side, and still did not bound the distribution.** The
largest per-unit training-label maximum across the 75 fits is **0.860**: the labels end below 1 as
constructed, and no label was clipped to make that true. But **0.36% of the 300-draw samples per unit still
ran above 1** (4.9% in the worst unit), because the bar distribution is *rescaled to the labels' mean and
standard deviation* — it does not inherit their range. A bounded target therefore gives bounded **labels**, not
a bounded **predictive support**, and this run is the measurement of that gap rather than a refutation of the
grid mechanism itself (`results/remodel/FINDINGS.md` §1.1).

## The free check, run first, on the same 75 units, for both targets

`scripts/unrevealed_support_check.py` (arithmetic only, seconds) measures what
`results/runs/20260919-target-support/` measured for the delta target: does the realised value leave the range
of training labels the fit was shown? Run for **both targets on identical units**, because the existing
`known_until` clamp makes a training label's window shorter than a scored cell's — a structural asymmetry that
affects a share and a factor differently, and is why the two are paired here rather than compared against the
recorded delta number:

| | unrevealed | delta |
|---|---|---|
| realised value **above** the training maximum | 42.7% of units (10.4% of cells) | 22.7% of units (3.6% of cells) |
| realised value **below** the training minimum | 8.0% of units | 10.7% of units |
| training labels below zero | 48.0% | 0.1% |
| outside the range on **both** targets | 20.0% of units | — |

The bounded target leaves the training range **more often**, not less — and the reason is the same one as
above: it is a one-sided bound, and it is the *lower* side of the label distribution that the falling
cumulatives dominate. This is descriptive, not a second pre-registered test; what it does is say in advance
that this arm was unlikely to be a repair.

## The falsifier's verdict, stated explicitly

* **The literal condition does not fire.** 95% coverage is **70.7%**, outside the 2 SE band
  **[76.4%, 93.0%]** around the recorded 84.7%. The pre-registered branch *"the support explanation of the
  one-sided tail is dead"* is therefore **not** what this run produced, and it is not reported as if it were.
* **The pre-registered expectation is not met either.** Coverage did not move toward nominal by more than the
  sampling error; it moved **away** by 14.0pp at 95% and 18.0pp at 90%, with paired counts of 2 fixed against
  15 broken at 95% (p = 0.0023) and 4 against 18 at 90% (p = 0.0043).
* **So the honest verdict is: the route as pre-registered is closed, and it does not decide the support
  question.** The two cells do not differ only in support — the reconstructed level differs too, by the
  mechanism in the section above (72% negative point estimates against the baseline's 33%) — so a coverage
  difference here cannot be attributed to the support change alone. The instrument cannot isolate the factor
  it was designed to isolate, on this column, with this label definition.
* **What remains after this.** R2 in the form pre-registered is spent: it was the last pre-registered test of
  the *tail*, and it produced a decisive result in the wrong direction. The candidates that were waiting
  behind it are unchanged — **R1** (an exposure-anchored level, which this run's failure is *consistent with*
  but does not establish, since the level here is Chain Ladder's and the break is in the label, not in the
  anchor), **R3** (a threshold ladder with the distribution rebuilt from the survival curve), and the
  row-count and cross-triangle families. **The reading that a bounded target is worthless is not supported by
  this run either**: what it shows is that bounding the labels does not bound the rescaled grid, and that a
  signed share cannot carry a level. A bounded *correction* on an anchored level (R1) is a different
  instrument and is untested here.

## Files

* `unrevealed.jsonl`, `baseline.jsonl` — 75 rows each, one row per unit: the realised future, the point
  estimate, the sampled median/mean, the eight interval bounds and the four coverage flags, plus (for the
  unrevealed cell) the training-label range, the count of negative labels and the measured fraction of drawn
  shares above 1.
* `support.jsonl` — the free arithmetic check, both targets, same units.
* `manifest.json` (merged: subset, per-cell `sha256`, resolved configuration, git revision, package versions,
  the paired result) and `manifest.<cell>.json` (each cell's own run record, including what was skipped and
  why). `sha256`: baseline `b3b2e54d…`, unrevealed `bca98c42…`.
* `run.log` — the driver: load at start and end, both cells' own logs, timings (baseline 43.4 min,
  unrevealed 38.3 min at 33–65 s/unit).
* `report.txt` — the tables above as printed.
* `PREREGISTRATION.md` — the subset, the bars and the falsifier, written before either cell ran.

The manifests record `dirty: true`, which is accurate provenance rather than an oversight: the run used this
branch's working tree, because `target="unrevealed"` and the reconstruction were not committed when it
started. The commit that carries them is **`624d897`**, on top of the integration line `d36f85e`, and the two
cell files' `sha256` are recorded above. Re-running one unit of the `unrevealed` cell from that committed
revision reproduced the recorded row **byte for byte**, so the code under test is the code in the
repository and not only the working tree that happened to be on disk.

## Reproduce

```
.venv/bin/python scripts/unrevealed_target.py --cell baseline   --limit 75   # 43.4 min
.venv/bin/python scripts/unrevealed_target.py --cell unrevealed --limit 75   # 38.3 min
.venv/bin/python scripts/unrevealed_support_check.py --limit 75              # 26 s
.venv/bin/python scripts/unrevealed_target.py --report --limit 75
```

`python -m pytest -q tests/` is green (47 passed), including seven new tests for this target: the label
recomputed from the triangle by hand, the bound on a monotone triangle, a falling cumulative giving a
negative share rather than a clipped one, the reconstruction against `Σ share·base·cl` computed by hand, the
degenerate point (a model predicting Chain Ladder's own share reproduces Chain Ladder's reserve exactly), the
point and the distribution being the same quantity, and an unknown target being refused rather than silently
treated as `ratio`.
