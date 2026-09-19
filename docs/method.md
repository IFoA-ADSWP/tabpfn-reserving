# What this project shows, and how it will be judged

This page is the demonstration plan. It exists so that the claims are fixed before the numbers are, and so
that a reader can tell what is being asserted and what would disprove it. The schedule, roles and definition
of done are in [`brief.md`](brief.md).

## The demonstration, in one sentence

Point the repository at a loss triangle and get a reserve **with its distribution**, from a single forward
pass — beside the method the industry uses, which reaches the same distribution through a thousand simulated
refits.

## The reframing

A triangle is the actuarial object nobody treats as tabular. Accident periods down the side, development
periods across the top, and a lower half that is unknown — that unknown region *is* the reserve. Current
practice is arithmetic on selected development factors, with uncertainty added afterwards by an analytical
formula (Mack) or a simulation (the ODP bootstrap).

Reframed as prediction, each unobserved cell is a supervised target:

```python
cells = triangle_to_cells(triangle)                    # one row per cell, with features
train, future = anchor_split(cells, at=valuation_date)  # the lower triangle is the test set
model = TabPFNRegressor()                              # tabpfn 9.0.0 -> TabPFN-3.5; no tuning
model.fit(train[FEATURES], train[TARGET])
point = model.predict(future[FEATURES])
full  = model.predict(future[FEATURES], output_type="full")   # the whole distribution, same pass
```

Two properties are visible in that code rather than argued in prose: there is **no hyperparameter to set**,
and the distribution arrives from the same call that would have produced a point estimate. Features are
restricted to what was known at the valuation date, so the future cannot leak into the features.

## The six beats

| # | Beat | What the viewer sees | What it demonstrates |
|---|---|---|---|
| 1 | The reframing | The same numbers twice: a shaded triangle, then a table of cells | A domain object turned into a prediction task, in about twenty lines |
| 2 | The reserve, zero-shot | TabPFN-3.5 and Chain Ladder on one triangle, by accident year and in total | The model is pointed at an object it has never seen and returns a defensible number |
| 3 | The distribution | Percentiles (50/75/90/99.5) with Mack's and the bootstrap's overlaid, and the wall-clock beside them | The capability the project rests on: a distribution, not a standard error, from one pass |
| 4 | Coverage | Empirical against nominal coverage for all three methods, over the fleet | Interval honesty — the actual regulatory question, since Mack's standard error is known to be optimistic in practice |
| 5 | Speed | Per-triangle wall-clock over hundreds of triangles, log scale, plus the Fast checkpoint | Reserving a whole book rather than one triangle |
| 6 | Where it does not win | The triangles and cells where Chain Ladder is closer | That the measurement is a measurement |

## What is claimed, and what would falsify it

| Claim | Backed by | Falsified by |
|---|---|---|
| A full reserve distribution from one forward pass | The distribution figure and the wall-clock bar beside it | Quantiles that miss badly, or a cost that is not lower |
| The point estimate is competitive with Chain Ladder | Per-triangle error across the fleet, and the direction of any bias | Systematic bias across the fleet, not noise on one triangle |
| Intervals more honest than Mack's | The coverage curve for all three methods on the same triangles | Coverage no closer to nominal than Mack's |
| Fleet-scale in minutes | Per-triangle timing, same machine, both methods | **FALSIFIED 2026-09-19** — the bootstrap finishes faster: Mack 0.13 s and the ODP bootstrap 0.38 s per triangle against the model's ~13.4 s, i.e. ~25–35× the other way (`results/runs/20260919-pricing/pricing.md`). The bar asked for a speed win and there is none; what the model offers instead is the distribution without a simulation to design. Recorded as a missed bar, not re-scoped |
| Zero-shot: no tuning, nothing fitted per triangle | The code above | A per-triangle hyperparameter |

## The bars, fixed before the runs

| Beat | Bar |
|---|---|
| Coverage | The 90% interval covers within ±3 points of nominal across the fleet, and is at least as close to nominal as Mack's on the same triangles |
| Point estimate | No systematic bias across the fleet; per-triangle errors reported in full, losses included |
| Speed | At least an order of magnitude faster per triangle than the ODP bootstrap, measured on the same machine |

If one of these fails, this page and the README say so, and the result stands as the result.

## What is not claimed

- **Not** a replacement for actuarial judgement, and not a regulatory-grade reserving model.
- **Not** a tail-extrapolation method: the far end of a triangle is where every method is weakest, and this
  project reports that rather than smoothing it over.
- **Not** individual-claim reserving, which is a separate and active research literature.
- **Not** a stress test of the model's scale claims. A triangle is tens to hundreds of rows; this uses
  TabPFN-3.5 in its small-data strength, and says so.

## Open method questions the first runs settle

1. Does the reframed cell-level prediction produce a *sane* reserve — within a sensible distance of Chain
   Ladder on a triangle where Chain Ladder is trusted?
2. Does `output_type="full"` return per-cell quantiles that can be summed into a reserve distribution?
3. Does it run on CPU in seconds for triangles of this size?
4. Does the anchor-time split hold — i.e. is the apparent performance real, rather than leakage?

The answers to all four are recorded here with the code that produced them.
