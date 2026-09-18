# The first production run: `abc`

**What this is.** Not a backtest. The anchor is the last diagonal, so there is no future in the data to
score against — the reserve is the projection every origin has not yet developed to, and this is what the
tool would actually report a reserving actuary. Run via the CLI, so the record below is reproducible from
the repository alone:

```bash
python -m tabpfn_reserving abc --distribution \
    --figure results/figures/abc_reserve_distribution.png \
    --json results/runs/abc-reserve.json
```

The distribution figure is `results/figures/abc_reserve_distribution.png`; the machine-readable record is
`results/runs/abc-reserve.json`; the console log is `results/runs/abc-reserve.log`.

## The numbers

| quantity | value |
|---|---|
| reserve, TabPFN-3.5 (point) | **12,844,223** |
| reserve, Chain Ladder (same cells) | **5,277,760** |
| reserve, median of the model's distribution | 9,758,413 |
| p5 / p95 of the distribution | 7,180,819 / 17,226,566 |
| fit / predict | 4.4 s / 42.9 s, CPU, no API |
| distribution route | `exact` — read off the bar distribution's bin weights, not a quantile grid |

## Three findings, none of them comfortable

**1. The model over-reserves by 2.4×.** 12.84M against Chain Ladder's 5.28M on the same cells and the same
development ages. This is not the modelling choice showing through; it is the recursive ratio arm
compounding. `abc` needs up to nine development steps for its youngest origin, and a per-step
over-prediction of ~10% becomes 1.1⁹ ≈ 2.4× by the ultimate. It is the same pathology recorded at
`k=5` in `FINDINGS.md`, seen at full strength because production asks for the whole tail at once.

**2. The distribution excludes the incumbent's answer.** Chain Ladder's 5.28M sits below the model's p5 of
7.18M. Whatever the model's distribution is capturing, it is not the uncertainty that would let it agree
with the standard method — the two do not overlap at all. A wide interval is not the same as an honest one.

**3. The point reserve is 32% above the median of its own distribution.** The point figure compounds
per-cell *predictions*; the draws compound per-cell *samples*. Product-of-means and median-of-products are
different numbers for a right-skewed ratio distribution, and the gap is the compounding artefact surfacing
again, now as an internal inconsistency a reserving actuary would notice immediately.

## What is nonetheless working

The arithmetic underneath is verified, not asserted: `triangle.factor_reserve` reproduces the CAS
`chainladder` package's own `ibnr` **to the pound** when given the same targets — ratio 1.000 on `abc`,
`genins` and `ukmotor` — and returns 0 at the last anchor under the backtest definition, which is what it
must do when there is no future left to project. So the classical comparison is a fair one, and the
discrepancy above belongs to the model arm.

The distribution route is `` exact ``: the bar distribution's `logits` are bin weights over its `borders`, so
the quantiles are read off the cumulative weights with interpolation inside the bin, retiring the caveat in
`docs/method.md` about a 15-level grid.

## What it means for the entry

The distribution mechanism works and is reproducible. The **point estimate does not**, in production, on a
short-tailed triangle. That is not a reason to hide the arm — it is the honest state of it, and it is why the
next experiment is the fleet-as-context arm rather than more presentation.

Pre-registered in `docs/method.md`: this arm was expected to carry signal, and it does in the *shape* of the
uncertainty. It was never registered as competitive on the point estimate, and it is not.

## A consequence for E3, which is worth more than the result itself

The spike's coverage figures (55/82/91/91 at 50/75/90/95) came from **backtests**, where each origin is
predicted at most a few development steps ahead. This production run asks for the **whole tail at once** —
up to nine compounding steps — and at that horizon the point estimate is 2.4× high and the interval misses
Chain Ladder completely.

Those two facts are not in tension; they are the same fact at two horizons. **Error compounds with the
horizon, and so does the interval's inability to cover.** So E3 as pre-registered — coverage over the fleet
— is not sufficient on its own: run over backtests it would measure the easy end of the problem and report a
flattering number. E3 must measure coverage **at the horizon the tool is actually used at**, or it measures
nothing that matters. That is a change to the experiment design, and it is recorded here rather than applied
quietly.
