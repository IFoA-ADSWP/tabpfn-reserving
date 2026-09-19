# Submission description

*Written to satisfy clause 3.5: "a description of what your project does sufficiently detailed to allow a
third-party developer to comprehend the project." Everything here is verifiable from the repository.*

## What it is

**Loss reserving as a prediction problem.** A loss triangle is the actuarial object nobody treats as tabular:
accident periods down the side, development periods across the top, and a lower half that is unknown — and
that unknown region *is* the reserve. Current practice does not model it: it is arithmetic on selected
development factors, with uncertainty added afterwards by an analytical formula (Mack) or by simulation (the
ODP bootstrap).

This project reframes every cell of the lower triangle as a **supervised target**, with features restricted to
what a valuation date could actually have known, and hands the whole triangle to **TabPFN-3.5** — which returns
the complete predictive distribution of the reserve in a single forward pass, on a CPU, with no tuning and no
API calls. The comparison is against the Casualty Actuarial Society's own `chainladder` package: Chain Ladder,
Mack's method, and the ODP bootstrap.

**Track:** *Formalize a new problem* / *Showcase a harness*. Built with TabPFN-3.5 (`tabpfn` 9.0.0).

## What a third-party developer can run

```bash
git clone https://github.com/IFoA-ADSWP/tabpfn-reserving && cd tabpfn-reserving
python -m venv .venv && . .venv/bin/activate && pip install -e .
export TABPFN_TOKEN=...                        # a Prior Labs token; the first run downloads the 3.5 weights

python -m tabpfn_reserving abc --distribution   # reserve + its distribution + a figure + a run record
python -m tabpfn_reserving abc --direct         # the same, without the recursion (see below)
python -m tabpfn_reserving abc --backtest       # scored against what actually developed
pytest -q tests/                                # 32 tests, ~8 seconds, no token, no model fit
```

Data ships inside the pinned `chainladder` package — the CAS Loss Reserve Database is 775 real triangles, so
nothing is downloaded and nothing is vendored. Every run writes a record: the command, the git revision,
package versions, a fingerprint of the matrix it used, wall-clock, the distribution route, any errors, and the
draws themselves.

## What it found

The repository is organised around one rule: a number is not a result until it reproduces, and the baseline
must reproduce the incumbent before any comparison means anything.

- **The arithmetic is verified, not asserted.** The hand-written Chain Ladder reproduces the CAS package's own
  `ibnr` **to the pound** — ratio 1.000 on three triangles, and per column on a two-column sample.
- **The distribution is real.** `output_type="full"` returns the model's own bar distribution (bin edges and
  weights); draws are taken from it by inverse-CDF sampling. An earlier version interpolated a 15-level
  quantile grid, which clamped the tail — the 99th percentile was understated by **62%**.
- **The point estimate is not competitive, and the repository says so first.** Over **464 paired evaluations**
  across 639 real triangles, the model was closer than Chain Ladder on **38.8%** of them, with a median error
  of 162.6% against the incumbent's 121.3%, while moving the reserve by a median of 142% of Chain Ladder's.
  It loses at every horizon measured.
- **Why it fails is measured, and that is the most useful thing here.** Predicting a triangle is recursive —
  predict a step, feed your own answer back, repeat. Scoring every step against *how deep into the projection
  it sits* shows the model is **unbiased where it has training examples and biased +3.2% per step (se 0.36%)
  where it extrapolates**, compounding to ×1.33 over nine steps against a measured ×1.67. Deleting the
  recursion removed the compounding entirely: the same model, features and settings went from +66.8% to −1.2%
  on `abc`.
- **A trap worth more than the number.** Anchored at the last three valuation dates, the deepest measurable
  step is two development periods and that bias reads **−0.0018 (se 0.0030)** — flat, and the effect looks
  absent. Anchored nine years back, the same triangle reads **+0.0190 (se 0.0016)**. A backtest that ends near
  the ultimate cannot see what production does, and it reports a flat line while doing so.
- **Calibration is poor, and it is measured rather than assumed.** Over 464 fleet evaluations the intervals
  under-cover at **every** nominal level — 39.2% at 50, 59.9% at 75, 78.0% at 90, 84.7% at 95, at z between
  −4.8 and −6.6 — and not because they are narrow (the median 90% interval is 4.55 times the point estimate).
  The actual future exceeds the sampled median in **67.7%** of cases, so this is a systematic low centring, the
  same failure the point estimate shows from the other side. `results/fleet/COVERAGE.md`. **Not** established:
  whether Mack's or the ODP bootstrap's intervals do better on the same units — that comparison has not been
  run, so the honest claim is that this model is miscalibrated, not that it is worse than the standard method.
- **Determinism and noise are stated.** The same command twice gives the same reserve to the pound; the point
  estimate does not depend on the seed; the draw noise floor is ~1.6% on the median at 300 draws.

## What it does not do

Stated because a judge should not have to find out:

- No tail factor: the reserve stops at the longest development age observed in the triangle.
- No correlation between cells, so the joint distribution is missing the correlation a triangle actually has.
- The fleet measures horizons 1–3. A 10×10 triangle has exactly one unknown diagonal, so single-triangle
  validation cannot reach the nine steps production asks for; that gap is documented rather than papered over.
- 136 of the 775 fleet containers are not proper triangles (companies with short histories, or gaps) and are
  refused visibly rather than silently reduced.

## Why it is worth submitting

Not because the model wins — it does not. Because the project answers *when a tabular foundation model should
not be trusted on a sequential actuarial problem*, with a diagnostic anyone can re-run on their own triangle
in one command, a clean zero intercept that shows the measurement works, and the false-negative trap that
nearly hid it. The distribution mechanism is the deliverable; the honest accounting of where it fails is the
contribution.

## Repository map

| path | what |
|---|---|
| `src/tabpfn_reserving/` | the package: `triangle.py` (cells, factors, CAS baselines), `arm.py` (the recursive and direct arms, distribution sampling), `cli.py` |
| `docs/method.md` | the method and the **pre-registered** success criteria, written before any result existed |
| `docs/what_it_unlocks.md` | the practitioner's page: what a reserving actuary can and cannot use this for |
| `docs/readiness.md` | the live gap ledger, with a resumption protocol for a cold reader |
| `results/fleet/FINDINGS.md` | 464 paired fleet evaluations and the negative verdict |
| `results/runs/20260918-023600_depth-bias/FINDINGS.md` | the depth-bias measurement |
| `results/runs/direct-arm.md` | the arm that removed the compounding, and its own limits |
| `tests/` | 32 tests: the CAS contract, the reserve conventions, leakage, the sampler, the fleet guards |
