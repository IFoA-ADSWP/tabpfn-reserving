# The configuration cells (#22) — the vendor's two documented changes do not move the tail

> **Status:** a screening run, 2026-09-19, 75 units × 4 cells, one factor per cell, on the recorded coverage
> units. **The pre-registered falsifier is met: neither documented change moves 90% or 95% coverage toward
> nominal by more than the sampling error, and on this subset that explanation is dead.** The one movement
> there is sits at the *centre* of the distribution, not the tail.
>
> The replication cell did its job: **all 75 of its rows are identical to the recorded rows** in
> `results/fleet/coverage.jsonl`, field by field, which is what makes the other three cells readable as
> differences rather than as a second implementation.

## What was run

`src/tabpfn_reserving/arm.py:make_model` gained two **opt-in** options; with no options it builds exactly the
recorded `TabPFNRegressor(device="cpu")`, and four tests in `tests/test_reserving.py` pin that (the default
configuration, one moved parameter per option, the declared positions really are the identifier columns, and
the client backend refusing the options rather than dropping them).

| cell | change |
|---|---|
| `baseline` | neither — the replication cell |
| `categorical` | `categorical_features_indices=[0, 1, 2]` (`origin_idx`, `dev_idx`, `cal_idx`) |
| `transform` | `inference_config={"REGRESSION_Y_PREPROCESS_TRANSFORMS": ("quantile_uni_extrapolate",)}` |
| `both` | both |

Same units, anchors, features, target and procedure as the recorded run: `IncurLoss`, three diagonals held
out, the direct arm's delta target, 300 draws per unit from the bar distribution (`draws_method=bar-bins`,
`distribution_route=exact` for all 300 rows), `default_rng(0)` re-seeded per unit, training labels clamped to
the evaluation anchor. `scripts/config_cells.py` runs one cell per invocation and is resumable.

**The subset is the first 75 rows of `results/fleet/coverage.jsonl`, read from that file** rather than
re-derived, so every cell and every re-run scores the identical units: 45 triangles, horizons 30/21/24 at one,
two and three steps, triangle sizes 6–10, 7–33 training rows per fit (median 25).

**Why 75 and not the pre-registered 150.** The machine was 2–5× oversubscribed for most of the window — load
20–40 on 8 cores, swap 13.4/14.3 GB used, a sibling `fleet_eval` job and several agent sessions alongside.
Measured: 4 cells at once = 0.75–2 units/min in total (each process ~12% of a core, stuck in uninterruptible
I/O, 600 MB of weights per process not fitting in what is left of the RAM); **1 cell at a time = 2.2–5.6
units/min**, which is what the run settled into. The limit was set at 60, then raised to 75, both times
before any coverage number was compared. At n=75 the binomial sampling error is **6.8pp at the 90% level**
(4.9pp at 95%, 11.3pp worst case p=0.5), against 4.8pp / 3.5pp at n=150. The run remains resumable: raise
`--limit` and re-run the four commands; the report uses the units common to all four cells, so the cells
extend together.

## The replication cell

`baseline` against the record, on the same 75 units:

```
coverage-flag mismatches against coverage.jsonl:  0
point, median, mean, chainladder, lo_/hi_ at 50/75/90/95 : identical to 1e-9 relative, on all 75 rows
50%: recorded 40.0% vs baseline cell 40.0%
75%: recorded 62.7% vs baseline cell 62.7%
90%: recorded 78.7% vs baseline cell 78.7%
95%: recorded 88.0% vs baseline cell 88.0%
```

The recorded full-fleet coverage is 39.2% / 59.9% / 78.0% / 84.7%; the subset's own record is
40.0% / 62.7% / 78.7% / 88.0%, and every number below is compared against **the subset's**, not the fleet's.

An incidental confirmation of "same command twice = same number": a second process re-ran the categorical
cell for 30 units that were already in its file. All 30 duplicated rows were **byte-identical** to the first
(`--dedupe`, which refuses outright if any two rows for a unit disagree). Those 105 rows are now 75.

## The coverage table (n=75, the same units in every cell)

| cell | 50% | 75% | 90% | 95% |
|---|---|---|---|---|
| nominal | 50% | 75% | 90% | 95% |
| `baseline` | 40.0% (−10.0pp) | 62.7% (−12.3pp) | **78.7% (−11.3pp)** | **88.0% (−7.0pp)** |
| `categorical` | 45.3% (−4.7pp) | 62.7% (−12.3pp) | **78.7% (−11.3pp)** | **90.7% (−4.3pp)** |
| `transform` | 46.7% (−3.3pp) | 62.7% (−12.3pp) | **78.7% (−11.3pp)** | **81.3% (−13.7pp)** |
| `both` | 49.3% (−0.7pp) | 62.7% (−12.3pp) | **80.0% (−10.0pp)** | **84.0% (−11.0pp)** |

## Paired against the baseline, unit by unit

`fixed` = units the baseline missed and this cell covered; `broke` = the reverse; `p` is McNemar's exact test
on the discordant pairs.

| cell | level | baseline | cell | difference | fixed | broke | p |
|---|---|---|---|---|---|---|---|
| `categorical` | 50% | 40.0% | 45.3% | **+5.3pp** | 8 | 4 | 0.39 |
| `categorical` | 75% | 62.7% | 62.7% | 0.0pp | 5 | 5 | 1.00 |
| `categorical` | 90% | 78.7% | 78.7% | **0.0pp** | 3 | 3 | 1.00 |
| `categorical` | 95% | 88.0% | 90.7% | +2.7pp | 3 | 1 | 0.63 |
| `transform` | 50% | 40.0% | 46.7% | **+6.7pp** | 7 | 2 | 0.18 |
| `transform` | 75% | 62.7% | 62.7% | 0.0pp | 6 | 6 | 1.00 |
| `transform` | 90% | 78.7% | 78.7% | **0.0pp** | 3 | 3 | 1.00 |
| `transform` | 95% | 88.0% | 81.3% | **−6.7pp** | 1 | 6 | 0.13 |
| `both` | 50% | 40.0% | 49.3% | **+9.3pp** | 9 | 2 | 0.07 |
| `both` | 75% | 62.7% | 62.7% | 0.0pp | 6 | 6 | 1.00 |
| `both` | 90% | 78.7% | 80.0% | +1.3pp | 5 | 4 | 1.00 |
| `both` | 95% | 88.0% | 84.0% | −4.0pp | 2 | 5 | 0.45 |

## The falsifier's verdict

**Dead, on this subset.** The pre-registered bar at n=75 is 6.8pp at the 90% level (4.9pp at 95%; 11.3pp
under the worst-case p=0.5 formula). No cell moves 90% or 95% coverage toward nominal by more than its bar:

* at **90%**, the categorical and transform cells move the coverage by **exactly 0.0pp** (−3 units and +3
  units, i.e. exactly as many fixed as broken); the `both` cell moves it +1.3pp;
* at **95%**, the categorical cell's +2.7pp is 3 units against 1 and not distinguishable from noise
  (p = 0.63), and the transform and `both` cells move **away** from nominal (−6.7pp and −4.0pp).

The one consistent movement is at the **bottom** of the distribution, not the top: 50% coverage rises from
40.0% to 45.3% / 46.7% / 49.3%, the `both` cell reaching nominal at the median (p = 0.07). That is the
direction the miscalibration's *centre* would move if the model were being nudged, and it is not the tail the
entry rests on.

By horizon, the tail miss is where it always was and the cells do not help it — at three steps (n=24) 90%
coverage is 75% for the baseline, 67% for `categorical`, 62% for `transform` and 62% for `both`. Reading the
tail directly: the truth is **above** the model's own 95% upper bound on 12.0% of baseline units (9/75,
nominal 2.5%), 9.3% for `categorical`, 16.0% for `transform` and 13.3% for `both` — and **below** the 95%
lower bound on 0.0% / 0.0% / 2.7% / 2.7%. The one-sidedness survives every cell.

## What this does and does not say

* **It does say:** two specific configuration changes the vendor documents for a target whose tail must extend
  beyond the training range — declaring the identifier columns categorical, and the extrapolating target
  transform — do not repair the one-sided upper-tail miss at n=75. The transform, which is the one the docs
  name for this failure mode, is if anything the worse of the two.
* **It does not say** that the documented *mechanism* (a bucket grid fixed on the prior and rescaled to the
  training target's mean and sd) is wrong. What it says is that this pair of knobs on top of it is not the
  lever. The bucket grid is still rescaled by what you predict, which is the argument for changing the
  *target's scale* rather than the model's configuration — the exposure/level reformulation in
  `results/remodel/FINDINGS.md` — and the arithmetic fact that the truth lies above the range of training
  labels the fit was shown (`results/runs/20260919-target-support/`) is untouched by anything here.
* **The power is stated, not assumed.** At n=75 the bar is 6.8pp at 90% and 4.9pp at 95% on the marginal
  proportion, 11.3pp under the worst-case formula. At 90% the observed movement is exactly zero, so this is
  not a bar too high to clear; at 95% the categorical cell's +2.7pp would need roughly three times as much
  movement to matter. A full repair — the tail miss is 11.3pp at 90% — would have shown.

**Files:** `baseline.jsonl`, `categorical.jsonl`, `transform.jsonl`, `both.jsonl` (75 rows each),
`manifest.json` (git revision, package versions, and the *resolved* configuration of each cell's model),
`run.log` (driver lines plus every cell's own log), `report.txt` (the tables above, as printed).

The manifest's `dirty: true` is accurate provenance rather than an oversight: the run used this branch's
working tree on top of revision `3c34318`, because the new `make_model` options were not committed at the
time. The `resolved` block is the record that matters for reading the cells — it shows the four models really
were configured differently (`categorical_features_indices` `null` vs `[0, 1, 2]`;
`REGRESSION_Y_PREPROCESS_TRANSFORMS` `(None, "safepower")` vs `("quantile_uni_extrapolate",)`).

**Reproduce:** `.venv/bin/python scripts/config_cells.py --cell <cell> --limit 75` for each of the four cells
from the repo root, then `--dedupe` and `--report`. `python -m pytest -q tests/` is green (40 passed).
