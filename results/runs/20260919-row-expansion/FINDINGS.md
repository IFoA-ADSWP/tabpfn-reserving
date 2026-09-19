# Row expansion: every intermediate target age (#25) — the falsifier is met

**The question.** Each fleet fit currently sees **6–33 training rows, median 25** (`results/conditions/rows_per_fit.log`),
and every benchmark the vendor's claims rest on excludes rows at that scale. The proposed scale-up borrows rows
from *other* triangles, which confounds two things: **row count** and **cross-triangle information**. This arm
isolates the first, from data already on disk.

**One factor.** `direct_training_rows(..., intermediate_ages=True)`: for anchor `k`, origin `a`, ages
`age = k - a` and every `target_age` with `age < target_age <= min(n-1-a, known_until-a)`. Opt-in, **default
unchanged**, so no recorded number moves; the expanded set **contains** the unexpanded one (the ceiling row is
still emitted), and every factor is measured from the same anchor cell, so the added rows are shorter versions
of the same run rather than a different estimand.

## The counts, before the results

Recount from the code path both cells ran, over the 90 screened units (`units.jsonl`, fields
`n_rows_baseline` / `n_rows_expanded`):

```
baseline (intermediate_ages=False)  min  7  p25 18  median 25  p75 33  max  33
expanded (intermediate_ages=True)   min 10  p25 40  median 65  p75 98  max  98
expansion factor on the paired units: median x2.60, range x1.43-x2.97
```

The baseline recount matches the recorded fleet figure (6–33, median 25) exactly on the units it covers, which
is the first check on any number here.

## The replication cell, and it is exact

Every unit also present in `results/fleet/clrd-IncurLoss.jsonl` is compared **field by field** (`actual`,
`chainladder`, `direct`):

```
units also in the recorded file: 90
worst relative difference in actual      : 0.000e+00
worst relative difference in chainladder : 0.000e+00
worst relative difference in direct      : 0.000e+00
REPLICATION OK
```

So the harness is the one that produced the recorded numbers, and the only difference between the two cells
below is the training row set.

## Measurement 1 — the fleet rate

Fixed screen, declared before the run: the first 90 scored units in clrd order. The script refuses a verdict
below 60 paired units, on a failed replication, or when the paired outcome has no variation. Errors are
recomputed from primitives (`actual`, `direct`, `chainladder`); `error_chainladder_pct` is `0.0` on every row
of the recorded file and is never read.

| cell | median \|error\| | closer than Chain Ladder | paired \|error\| change vs CL (median) |
|---|---|---|---|
| baseline (6–33 rows) | 181.2% | **47.8%** | +18.0pp |
| expanded (10–98 rows) | 176.8% | **50.0%** | +4.9pp |
| the null (Chain Ladder) | 131.0% | — | — |

```
fixed  (baseline not closer, expanded closer) :   7
broken (baseline closer, expanded not closer) :   5
both closer: 38      neither closer: 40
McNemar exact, two-sided: p = 0.774
rate change: +2.2pp   (the pre-registered bar is a rise above 38.8% + 2.3pp = 41.1%)
binomial sampling error at n=90: 5.3pp   (the recorded rate is 38.8% +- 2.3pp at n=464)
```

By horizon (`held-out` diagonals): 1 step 42.9% → 42.9% (+0.0pp, n=35), 2 steps 66.7% → 59.3% (−7.4pp,
n=27), 3 steps 35.7% → 50.0% (+14.3pp, n=28). By triangle size: n=9 40.0% → 60.0% (10 units), n=10
48.0% → 48.0% (75 units). **The +2.2pp move is inside this screen's own binomial error, and the screen can
resolve the 2.3pp the bar asks for (5.3pp SE); even the optimistic horizon-3 read (+14.3pp on n=28) is a
subgroup of a fixed subset and is reported as one, not as the result.**

## Measurement 2 — the depth bias

**The recorded +0.0317 per step belongs to the recursive arm**, which trains on `training_rows` — not on
`direct_training_rows`. The row expansion cannot reach it by construction, so the recorded number is
**reproduced, not replaced**: `results/runs/20260919-depth-bias-replication/` re-runs the recorded window
(every anchor from k=3 to the last but one, `--triangles abc genins --targets delta`) and returns

```
recorded: n=287  slope=+0.03168 (se 0.00363)  intercept=+0.00011
this run: n=287  slope=+0.03168 (se 0.00363)  intercept=+0.00011
per triangle: abc +0.01903 (se 0.00161)   genins +0.06170 (se 0.00785)
REPLICATION OK
```

The arm whose row set *did* change is the direct arm, and the same per-prediction-log-error-vs-depth
regression is run there, before and after the expansion, on the same triangles and the same anchor window.
`depth` is the number of development steps one prediction spans — a **different regression** from the
recursive table's, so the two slopes are not comparable; only the direct arm's own before/after is like for
like. The direct arm predicts each run whole, so the recursive runs' currency decomposition and by-origin
replay do not exist for it and are reported as absent rather than approximated.

| run | n | slope | se | intercept | log err / step |
|---|---|---|---|---|---|
| direct, baseline rows | 287 | **+0.06421** | 0.00694 | −0.13224 | −0.00030 |
| direct, expanded rows | 287 | **+0.06371** | 0.00630 | −0.11118 | +0.00991 |

```
per triangle      baseline            expanded
        abc  +0.04329 (se 0.00620)  +0.04678 (se 0.00536)
     genins  +0.11423 (se 0.01295)  +0.10557 (se 0.01214)

by depth (steps spanned)   baseline    expanded
    1 (88 steps)            -0.0205     -0.0047
    2 (69)                  -0.0288     -0.0067
    3 (52)                  -0.0040     +0.0211
    4 (37)                  +0.0753     +0.1045
    5 (24)                  +0.2166     +0.2277
    6 (13)                  +0.3374     +0.3446
    7 (4)                   +0.5567     +0.5567
```

**Slope change: 0.06421 → 0.06371, i.e. −0.0005 — an order of magnitude inside either fit's own standard
error (0.0069 / 0.0063).** Both arms keep the same shape: flat or slightly negative through short runs, then
a per-prediction bias that grows with the span of the prediction, unchanged at every depth above 4. The
per-step normalisation of the pooled rows moves from ≈0 (×0.9997) to +1.0% per step, which is not a slope
change in the wrong direction so much as the short-run points moving toward zero while the long-run points
stay put.

## The falsifier's verdict, stated explicitly

Pre-registered: *if the slope is unchanged within its own standard error **and** the fleet rate stays inside
the noise floor, then row count is not the binding constraint for this model on this data — which kills the
"more rows of its own" family outright and puts all the weight on cross-triangle context (E2b).*

**Both legs of the falsifier hold.** The depth-bias slope moved by −0.0005 against a standard error of 0.0063,
and the fleet rate moved +2.2pp against a binomial standard error of 5.3pp — the bar the task set (a rise
above 41.1%) was not reached. **This is a result, not a failure: the in-triangle row expansion is not the
binding constraint.** A fit that sees 2.6× the rows of its own triangle, every one of them leakage-clamped and
inside the same run, gains nothing measurable. What remains is the family of rows that carry *different*
information rather than merely more of the same — the fleet-as-context build (E2b), which is now the only
route left to a competitive point estimate and the one the measurement points at.

## Provenance and reproduction

- `results/runs/20260919-row-expansion/` — `units.jsonl` (90 units, both cells, rows-per-fit per cell),
  `run.log` (the console record), `manifest.json` (git revision, `dirty` flag, cells, scoring, model, screen
  size). Re-derive the summary without fitting anything: `.venv/bin/python scripts/row_expansion.py
  --summary-only --min-common 60 --out results/runs/20260919-row-expansion` — the same command twice gives
  the same numbers.
- **`results/runs/20260919-depth-bias-replication/`** — the recorded recursive run **reproduced exactly**
  (n=287, slope +0.03168, se 0.00363, intercept +0.00011; abc +0.01903, genins +0.06170), so the harness is the
  one the recorded slope came from.
- **`results/runs/20260919-203432_depth-bias-direct/`** — the direct arm's own per-prediction-error-vs-depth
  regression under **baseline** rows (n=287, slope +0.06421, se 0.00694, intercept −0.13224).
- **`results/runs/20260919-203642_depth-bias-direct/`** — the same regression under **expanded** rows
  (n=287, slope +0.06371, se 0.00630, intercept −0.11118). The direct arm predicts each run whole, so the
  recursive runs' currency decomposition and by-origin replay are deliberately absent from both: there is no
  per-step factor to replay and reporting one would be fiction.
- Screen cost: 80.1 minutes for 90 units × 2 cells on a box shared with a sibling TabPFN task (~1.9 min/unit
  of contended CPU, against ~13.5 s/unit on a calm box). Nothing is claimed beyond the screen; the full fleet
  would take ~5 h at this contention.

## Two harness defects found and fixed on the way, both of which would have reported the wrong thing

1. **`pytest` inside a worktree was testing the primary checkout's code.** The venv lives only in the primary
   checkout and its editable install points at *that* `src/`, so a worktree run imported the other tree's
   package — a new API raised `TypeError` and a behaviour change would have been silently certified by code
   that was never edited. Fixed with `[tool.pytest.ini_options] pythonpath = ["src"]` in `pyproject.toml`.
   `scripts/depth_bias.py` had the same hole for the same reason and now carries the same bootstrap the other
   fleet scripts do.
2. **`results/runs/<id>/FINDINGS.md` was outside the figure checker's corpus.** `results/*.md` and
   `results/**/*.md` did not reach it, so every run's deliverable was unchecked while the checker reported a
   pass — that gap was live in this branch (`results/runs/20260919-config-cells/FINDINGS.md` quotes the fleet
   figures and was never scanned). Closed with `RUN_CORPUS = ["results/*/*/*.md"]`, with the self-test
   extended to assert a run directory is in the corpus.

Both are marked **corrected** here for the superseded phrase "pytest from the repo root is the same gate
everywhere": it was not, and a script run from a worktree without its own `sys.path` bootstrap was reporting
another checkout's numbers.
