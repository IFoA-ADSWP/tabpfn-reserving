# tabpfn-reserving

**Loss reserving as a prediction problem: a reserve with its full predictive distribution, from a single
forward pass.**

Built with **[TabPFN-3.5](https://priorlabs.ai/technical-reports/tabpfn-3-5)** for the Prior Labs
[TabPFN-3.5 Hackathon](https://platform.priorlabs.ai/hackathon-3.5) — *Formalize a new problem* / *Showcase a
harness*.

> **Status: working, results provisional.** The method is settled, the package runs end to end on CPU, and
> the first production run is stored in `results/runs/`. Its verdict is not flattering and is written up
> rather than buried: the **distribution** works, the **point estimate over-reserves on `abc` by 2.4×**
> against Chain Ladder ([`results/runs/abc-reserve.md`](results/runs/abc-reserve.md)). The bars this project
> is judged against were pre-registered in [`docs/method.md`](docs/method.md) before any of these numbers
> existed. Live gap list: [`docs/readiness.md`](docs/readiness.md).

## The idea

A loss triangle is the actuarial object that nobody treats as tabular. Accident periods down the side,
development periods across the top, and a lower half that is unknown: that unknown region *is* the reserve.
In current practice it is not a modelling problem at all — it is arithmetic on selected development factors,
with uncertainty bolted on afterwards by an analytical formula (Mack) or by simulation (the ODP bootstrap).

Reframed as a prediction problem, each cell of the lower triangle is a supervised target, with features
restricted to what a valuation date would actually have known. The reserve is the sum of the predictions,
and TabPFN-3.5 returns **the whole predictive distribution in the same forward pass** — `output_type="full"`,
no extra inference cost. The standard practice it is compared against obtains the same distribution from a
thousand simulated refits.

That is the claim this repository sets out to measure rather than assert.

## What it shows

| # | Beat | What it demonstrates |
|---|---|---|
| 1 | **The reframing** | The same numbers twice: as a shaded triangle, then as a table of cells. A domain object, turned into a prediction task, in twenty lines |
| 2 | **The reserve, zero-shot** | TabPFN-3.5 against Chain Ladder on the same triangle — no tuning, no feature engineering, raw values straight in |
| 3 | **The distribution** | The reserve as a distribution: percentiles beside Mack's and the bootstrap's, with the wall-clock for each. A risk margin, an IFRS 17 risk adjustment and a Solvency II capital figure are quantiles, not standard errors |
| 4 | **Coverage** | Do the 90% intervals contain the truth 90% of the time — for all three methods, over hundreds of real triangles |
| 5 | **Speed** | Reserve a whole book, not one triangle: per-triangle wall-clock, including the Fast checkpoint |
| 6 | **Where it does not win** | The triangles and cells where Chain Ladder is closer, stated in the open |

## Why a triangle is a different problem from claims modelling

Prior work on tabular foundation models in insurance has mostly tested flat, claim-level regression at scale
— hundreds of thousands of rows, heavily zero-inflated outcomes — where models of this class do not win
(see the IFoA ADSWP evidence base linked below). A triangle is not that object: a 10×10 triangle has ~55
observed cells and a 20×20 has ~210, with no zero mass in the cumulative figures. This project is a
deliberate test of the small, structured, sequential end of the problem space, and it reports the result
either way.

## Data and comparison

Both sides are public and maintained, so the comparison has authority:

- **Triangles** — the [CAS Loss Reserve Database](https://www.casact.org/publications-research/research/research-resources)
  (Meyers/Shi, from NAIC Schedule P) and the classic sample triangles. These ship **inside the
  `chainladder` package** (the `clrd` set alone is 775 triangles), so nothing is downloaded and nothing is
  vendored: the package version is pinned, and every run records a **fingerprint of the matrices it
  actually used** in its manifest, so a dataset change cannot silently invalidate a stored result.
- **Baseline** — [`chainladder`](https://github.com/casact/chainladder-python), the Casualty Actuarial
  Society's own Python package: Chain Ladder, Mack's method and the ODP bootstrap.

## Quickstart

The package installs from this repository and runs on CPU. The first run downloads the TabPFN-3.5 weights
once and needs a Prior Labs token in `TABPFN_TOKEN` (`results/spike/FINDINGS.md` has the two-minute version
and the hosted alternative).

```bash
git clone https://github.com/IFoA-ADSWP/tabpfn-reserving
cd tabpfn-reserving
python -m venv .venv && . .venv/bin/activate
pip install -e .
export TABPFN_TOKEN=...               # your Prior Labs token

# The reserve as of the latest diagonal, with its distribution, from one triangle
python -m tabpfn_reserving abc --distribution

# The same, writing the figure and the machine-readable run record
python -m tabpfn_reserving abc --distribution \
    --figure results/figures/abc_reserve_distribution.png \
    --json   results/runs/abc-reserve.json

# Scored against what actually developed afterwards, one row per anchor
python -m tabpfn_reserving abc --backtest
```

Real output from the first of those, unedited — including the part that does not flatter the method:

```
abc: 11x11  anchor = last diagonal (age 132)  55 observed transitions
  reserve (compounded point)     12,844,223   fit 4.4s  predict 39.2s
  reserve (chain-ladder)          5,277,760   like-for-like, on the same cells
  reserve (CAS package)           5,277,760   reference: includes a tail nobody can score
  reserve (sampled mean)         11,909,189   -7.3% vs the point
  reserve (sampled median)        9,671,798   -24.7% vs the point
  distribution             p5=7,452,235  p50=9,671,798  p95=20,049,586  p99=34,882,828
  distribution route          exact quantiles, bar-bins draws, 300 of them
```

Three summaries of "the reserve" are printed because they are not equal, and the gap between them is the
compounding rather than a rounding error — the point figure compounds per-cell predictions, the others
accumulate sampled paths. Which is which is stated rather than implied.

The figure it writes is the one below; `results/runs/abc-reserve.md` says what the numbers mean and
`docs/readiness.md` says what is still missing.

![The reserve as a distribution: abc, TabPFN-3.5, 300 draws](results/figures/abc_reserve_distribution.png)

Current state, honestly: [`docs/readiness.md`](docs/readiness.md).

## Reproducibility

Every figure quoted in this README is produced by a command in it and stored in `results/`. No number is
quoted that the repository cannot regenerate from public data, which is what the hackathon terms require.

## Licence

Apache-2.0 — see [LICENSE](LICENSE).

## Acknowledgements

Developed in the [IFoA Actuarial Data Science Working Party](https://www.actuaries.org.uk/) (ADSWP), whose
open evidence base on tabular foundation models in insurance —
[`IFoA-ADSWP/tabular-foundation-model`](https://github.com/IFoA-ADSWP/tabular-foundation-model) — supplies
the decision rules and the prior-negative results this project tests against. This repository is a separate,
self-contained artefact and does not redistribute that work.

TabPFN is developed by [Prior Labs](https://priorlabs.ai). This is an independent entry to their hackathon.
