# Spike run 1 — the reframing works, the distribution looks right, the point estimate does not yet compete

**Command:** `.venv/bin/python scripts/spike_e0_e1.py --triangles abc genins mcl ukmotor`
**Artifacts:** `results/spike/spike_e0_e1.csv`, `spike_e0_e1.json`, `probe.json`
**Environment:** tabpfn 9.0.0 (TabPFN-3.5) · chainladder 0.10.1 · torch 2.14.0 · numpy 2.5.3 · pandas 2.3.3 · scikit-learn 1.9.1 · CPU only, no API calls · seed 0, 300 draws.

## What was actually run

Every diagonal is a valuation date. At anchor *k* the books hold origins 0..*k*; cells with
origin+development ≤ *k* are known and everything beyond is predicted. Because the full triangle is
observed to its last diagonal, the **observed part** of that future is the scoring target — the tail
beyond the last diagonal cannot be scored and is not scored.

Two arms of the same family are compared on identical cells:

- **`tabpfn`** — the recursive link-ratio arm: TabPFN-3.5 predicts `C[a,d]/C[a,d-1]`, one diagonal at a
  time, each prediction becoming context for the next.
- **`cl_factor`** — Chain Ladder arithmetic over exactly the same cells, from volume-weighted factors
  computed at the anchor with a flat tail beyond it. This is the like-for-like comparison.
- **`cl_ibnr` / `mack`** — the CAS package's own reserve on the as-of triangle. **Reference only:** its
  `ibnr` includes the tail beyond the last diagonal, so it is not scored like-for-like.
- **`placebo`** — identical pipeline with the training targets shuffled. **The control that decides
  whether any of the rest means anything.**

## Results

| triangle | k | train rows | actual future | tabpfn | err % | cl_factor | err % | placebo | err % |
|---|---|---|---|---|---|---|---|---|---|
| abc | 5 | 15 | 2,092,802 | 2,152,788 | **+2.9** | 2,497,504 | +19.3 | 60,289,027 | +2,781 |
| abc | 6 | 21 | 2,049,742 | 2,114,874 | **+3.2** | 2,093,769 | +2.1 | 25,172,973 | +1,128 |
| abc | 7 | 28 | 2,089,266 | 1,777,489 | −14.9 | 1,838,219 | −12.0 | 13,359,732 | +539 |
| abc | 8 | 36 | 2,055,248 | 1,907,989 | **−7.2** | 1,723,768 | −16.1 | 7,962,908 | +287 |
| genins | 5 | 15 | 12,221,385 | 19,811,736 | +62.1 | 20,970,810 | +71.6 | 431,478,486 | +3,431 |
| genins | 6 | 21 | 10,554,061 | 15,646,682 | +48.3 | 12,025,073 | **+13.9** | 210,116,241 | +1,891 |
| genins | 7 | 28 | 9,507,303 | 8,856,074 | **−6.8** | 7,985,314 | −16.0 | 83,262,658 | +776 |
| mcl | 3 | 6 | 2,948 | 20,451 | +593.7 | 2,021 | **−31.4** | 22,263 | +655 |
| mcl | 4 | 10 | 2,144 | 8,418 | +292.6 | 2,339 | **+9.1** | 26,461 | +1,134 |
| ukmotor | 3 | 6 | 16,255 | 36,662 | +125.5 | 29,935 | **+84.2** | 41,038 | +153 |
| ukmotor | 4 | 10 | 14,682 | 13,595 | **−7.4** | 17,128 | +16.7 | 51,114 | +248 |

**Summary:** median absolute error — tabpfn **14.9%**, cl_factor **16.1%**, placebo **775.8%**.
Mean error — tabpfn **+99.3%**, cl_factor **+12.9%**. TabPFN is the closer of the two on **5 of 11**
evaluations; Chain Ladder on 6.

**Predictive-interval coverage (TabPFN, n = 11):**

| nominal | 50% | 75% | 90% | 95% |
|---|---|---|---|---|
| empirical | **55%** | **82%** | **91%** | **91%** |

## What this establishes

1. **The reframing works end to end, on CPU, with no API.** A triangle becomes a prediction problem, the
   model returns a reserve and its distribution, and the whole thing runs on a laptop. Spike gate 1 passes.
2. **`output_type="full"` gives a real distribution** — mean, median, mode, a quantile grid, and the bar
   distribution's own criterion and 5000-bin logits behind it. Spike gate 2 passes. The default quantile
   grid is [0.1 .. 0.9], too narrow for a 95% interval, so the harness requests a denser grid.
3. **The distribution looks right.** Empirical coverage tracks nominal closely — 55/82/91/91 against
   50/75/90/95 — on eleven evaluations. This is the project's central claim, and the first evidence
   supports it. It needs E3's full treatment before it can be called a result.
4. **The control works, so none of the above is an artefact.** The shuffled-target placebo is 52× worse
   at the median (775.8% vs 14.9%). A leaking harness would have scored well with shuffled targets.
5. **The point estimate is not competitive yet, and the failure is specific.** The median is a tie
   (14.9% vs 16.1%), but the mean is eight times worse (+99.3% vs +12.9%) because the recursive arm
   occasionally blows up: +594% and +293% on `mcl`, +126% on `ukmotor`, +62% on `genins`. Multiplying
   predicted ratios along a diagonal compounds any upward bias, and the smallest triangles are where it
   compounds worst.

## What it rules out

- **Recursive link-ratio prediction as the headline arm.** It is not competitive as it stands, and its
  failures are the same shape as its design: compounding. The other formulations in
  `docs/experiments.md` §E2 are now the priority, not a variant to try later.
- **Very small triangles as evidence.** At `mcl` k=3 there are **six** training rows and the model scores
  +593.7% while the placebo scores +655.2% — indistinguishable. That is not a model failure so much as an
  absence of any data to learn from, and it belongs in the regime boundary, stated in the open.

## What to run next, in order

1. **The Δ-to-Chain-Ladder arm** — predict the ratio *relative to* the volume-weighted factor rather than
   the raw ratio. The model then only has to learn the deviation from a strong, stable prior, which is the
   most direct fix for compounding bias, and it should be a one-line change of target.
2. **Direct cumulative prediction (A1)** — one batching change that removes the recursion entirely and,
   as a side effect, cuts the runtime: 20–68s per anchor today because each diagonal is a separate
   inference pass.
3. **A minimum-evidence rule** — require enough observed cells (≥ ~20 training rows) before scoring, and
   report the boundary where the model stops beating its own placebo.
4. Then distributional classification (A4) and the calibration work in E3, which is where the strongest
   evidence already sits.

## Caveats, stated so they cannot be discovered later

- **n = 11 evaluations.** The coverage figures are indicative, not a result; E3 runs them over the fleet.
- **Only the observed future is scored.** A method's projection beyond the last diagonal is unmeasurable
  from data and is excluded, which is why `cl_ibnr` is a reference and not the head-to-head.
- **The factor arm carries a flat tail** beyond the anchor, so both arms are handicapped on immature
  anchors — visible in `genins` k=5, where *both* over-predict (71.6% and 62.1%).
- **The reserve intervals come from marginal (independent) cell sampling.** Summing independent cell draws
  usually understates the variance of the total, yet coverage is adequate-to-conservative here — which
  makes the joint-sampling arm more interesting, not less.
- **Timings are single-machine and unoptimised,** and were taken while other work ran on the same machine.
  They are not E5's measurement.
