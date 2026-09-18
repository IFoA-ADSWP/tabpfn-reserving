# The delta arm in production: `abc`

The same triangle and the same anchor as [`abc-reserve.md`](abc-reserve.md), with `--target delta`: the model
learns the link ratio *relative to* the volume-weighted factor for that age, so it only has to correct a
strong stable prior rather than rediscover it.

```bash
python -m tabpfn_reserving abc --target delta --distribution \
    --figure results/figures/abc_reserve_distribution_delta.png \
    --json   results/runs/abc-reserve-delta.json
```

**This is the first production run of the delta arm with correct scaling.** Until the fix in `1cda94c`, the
delta arm's *draws* were never rescaled by the global factor while its point predictions were, so the
distribution it reported described a different quantity than the number printed above it. The backtest
figures in `results/spike/FINDINGS.md` are unaffected — the spike had the scaling right — but no production
number for this arm existed before now.

## The comparison

| | ratio arm | delta arm | Chain Ladder |
|---|---|---|---|
| point reserve | 12,844,223 | **8,801,618** | 5,277,760 |
| error vs Chain Ladder | +143% | **+67%** | — |
| p5 … p99 | 7.18m … 21.53m | **7.26m … 10.97m** | — |
| median | 9,758,413 | **8,481,123** | — |
| point vs its own median | +32% | **+4%** | — |
| fit / predict | 4.9 s / 48.1 s | 4.6 s / 34.7 s | — |

## What it means

**Dividing out the stable factors removes about half the compounding.** The raw-ratio arm has to rediscover
that losses develop by roughly the volume-weighted factor at each age; when it misses, it misses
multiplicatively, nine times over for the youngest origin. Handing it that factor as a prior turns the task
into a correction, and the error falls from +143% to +67%. That is the same mechanism that made the delta arm
degenerate to Chain Ladder in the backtests (`k=5`: +20.2% against +19.3%) — a strong prior plus a small
correction is close to the classical method, which cuts both ways.

**The distribution is now coherent with its own point estimate.** The point sits 4% above the median instead
of 32%, so the internal inconsistency in the ratio arm was largely the compounding artefact rather than a
separate defect.

**And it still over-reserves by two thirds, with an interval that still excludes Chain Ladder.** The whole
delta distribution — p5 of 7.26m — sits above the classical answer of 5.28m. So the arm is *better behaved*
and still not *right*, and the honest read is that rescaling is a partial fix: the remaining error is in the
model's own corrections, which accumulate over a horizon at which it has never seen an example.

This does not change the plan. It strengthens the case that the next experiment is structural — the
fleet-as-context arm (`#10`) — rather than another rescaling, and it gives that experiment a clean baseline
to beat: **+67% on `abc`, point-to-median agreement inside 5%**.
