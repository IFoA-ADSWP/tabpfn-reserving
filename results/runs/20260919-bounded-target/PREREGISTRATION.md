# R2 (#24) — pre-registration, written before the cells were run

Written 2026-09-19, before `unrevealed_target.py --cell baseline` and `--cell unrevealed` were started on this
branch's working tree. It fixes the subset, the expectation, the falsifier and the minimum sample *before* any
coverage number from this run existed. The expectation and falsifier are quoted from
`results/remodel/FINDINGS.md` §R2, which was written before this arm was run at all.

## The fixed subset

**The first 75 rows of `results/fleet/coverage.jsonl`**, read from that file rather than re-derived, so both
cells and every re-run score the identical units. Reasons, in order:

1. **The rate was measured first, and it is the binding constraint.** Three fits on this box while other
   kanban work was running measured 20–67 s per unit (load average 26 on 8 cores). At that rate 464 units is
   ~5 h for one cell and the pre-registered screening protocol says to screen first; 75 units is ~25–80 min
   per cell.
2. **75 is the size #22 used** (`results/runs/20260919-config-cells/FINDINGS.md`), on the same 75 units and
   the same harness, so this run's replication cell can be checked against that run's as well as against the
   record — two independent processes on one number.
3. The subset size was **not** chosen from any coverage number: no coverage number from either cell existed
   when this was written.

**Sampling error at n=75**, on the recorded rates: 4.4pp at 95% (p=0.847), 4.8pp at 90% (p=0.780), worst case
(11.0pp) at p=0.5. Registered in advance as the bar.

## The pre-registered expectation

Coverage at **90% and 95% moves toward nominal** (from the recorded 78.0% and 84.7%) **before any widening is
applied**, by more than the binomial sampling error at n=75 (i.e. 95% must exceed 89.1%, 90% must exceed
82.8%), because the target's support now ends where the quantity ends.

## The falsifier

**If 95% coverage stays within 2 binomial SE of the recorded 84.7% — that is, inside [76.6%, 92.8%] at
n=75 — the support explanation of the one-sided tail is dead.** Report that plainly, with the paired
unit-by-unit counts (fixed/broken and McNemar's exact test), and the remaining candidates are the row-count
and cross-triangle families.

## What can and cannot be concluded from this run

* The two cells are **paired**: same units, anchors, features, target age, 300 draws, `default_rng(0)` per
  unit, bar-bins route, training anchors and the existing `known_until` clamp, a bare
  `TabPFNRegressor(device="cpu")`. The only difference is what the model is asked to predict and how the
  reserve is rebuilt from it.
* The replication cell (`baseline`, the recorded `delta` target) is **mandatory** and is checked row by row
  against `results/fleet/coverage.jsonl` on every field the two files share. If it does not reproduce, no
  cell here is interpretable and the run is reported as that instead.
* At n=75 the *falsifier* is a crude instrument for a partial repair — a move from 84.7% to 90% sits inside
  the 2 SE band. The **paired** test is the finer reading and is reported alongside it; a run that is
  inconclusive by rate but significant by paired count will be reported as exactly that, and not as a death.
