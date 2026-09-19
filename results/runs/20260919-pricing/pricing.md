# E3a pricing — what the classical methods cost, and one finding that outranks the number

**The question this answered** (`docs/experiments.md` §5.2, issue #20): the three-method coverage comparison
needs Mack's and the ODP bootstrap's intervals on the same 464 units where TabPFN-3.5's are measured. Mack is
analytical; the bootstrap is a simulation. Before designing a 464-unit run, price one unit.

Measured on `abc`, one triangle, one anchor, in the pinned environment (`chainladder` 0.10.1), on a laptop:

| method | wall-clock per triangle | → 464 units | verified how |
|---|---|---|---|
| **Mack** (`MackChainladder`) | **0.13 s** | **~1.4 min** | `mack_std_err_` (1,1,11,12) and `total_mack_std_err_` = **152,712.93**, matching the value this repository has stored since the first single-triangle run |
| **ODP bootstrap** (`BootstrapODPSample`, 1000 sims, **plus a Chain Ladder refit on every resample**) | **0.38 s** | **~3.0 min** | 1000 draws, **0 finite losses**; mean **5,286,238** against Chain Ladder's **5,277,760** — ratio **1.002**, which is what a working ODP bootstrap must show, since it targets Chain Ladder |
| **TabPFN-3.5's own fleet coverage run** (the thing being compared against) | ~13.4 s | **104 min** | `results/fleet/coverage.log` |

**The gate is cleared: the comparison costs minutes, not hours.** No experiment design needed beyond what §5.2
already specifies.

## The finding that outranks the number

**The classical method is not the slow one.** The entry has been framing the contrast as *"the whole
distribution in a single forward pass, where the incumbent needs a thousand simulated refits"* — which reads as
a speed argument, and the measurement says it is the wrong way round. The bootstrap's thousand refits cost
**0.38 s**, because each refit is Chain Ladder arithmetic on a 10×10 table. The model's forward pass costs
**~13.4 s per unit**. On the evidence, the foundation model is roughly **25–35× slower** for the same object,
on CPU.

What survives, and should be the entry's claim instead: the model produces the distribution **without a
simulation step you have to design** — no resampling scheme, no assumed process distribution, no adjustment for
bias — and it runs on CPU with no API calls. Convenience and modelling assumptions, not speed. Any sentence in
the entry that implies otherwise needs fixing before submission.

## Two API facts worth recording, because one cost a worker 41 minutes

1. **`full_std_err_` is not the per-cell future standard error.** Its projection column is **all zeros**
   (`max 0.118` over the observed cells, `0` on the column that would matter), so a per-cell interval built from
   it would silently be zero-width. The usable attributes are `mack_std_err_` (1,1,11,12 — by development age,
   projection column included, last value 107,943.59), `total_mack_std_err_` (152,712.93),
   `total_parameter_risk_` (95,912.86) and `total_process_risk_` (118,835.87).
2. **This repository already reads them.** `src/tabpfn_reserving/triangle.py:449-453` fits `MackChainladder`
   and pulls exactly those attributes behind `hasattr` guards with `nansum`/`pick_column`. The dispatched worker
   reverse-engineered the same thing from scratch over nine revisions and never produced a deliverable; the
   correct move was to read our own module first. Recorded on the task board as the reason it was stopped.

A third, smaller one: summing a bootstrap ensemble's `ibnr_` with `.sum()` returns all-NaN, because the
unobserved cells are NaN. `nansum` over the ensemble axis is required, and the first attempt at this number was
**withdrawn** rather than reported: it priced a workflow whose output was 1000 NaNs.

## Provenance

`.venv/bin/python`, `chainladder` 0.10.1, one triangle (`abc`), timings as single runs on a machine under load
from another agent, so they are **upper bounds**; the ratios are large enough that contention does not change
the conclusion. Records: `results/runs/20260919-pricing/pricing.json`.
