# Are we ready? A gap ledger

> **Status:** updated 2026-09-18, end of day 4 of 19. Checked against the hackathon terms, the brief's
> definition of done, and the rubric weights — not against how the work feels.
>
> **Verdict then: not ready, and the gap was not the science. Verdict now: the science got real, and the
> gap is the artefact a judge opens.**
>
> Every gap is filed as an issue in this repository (**#1–#18**) — this page is the ledger, the issues are the
> worklist. Labels: `submission-blocker`, `human-only`, `judge-facing`, `experiment`, `packaging`, `redesign`.

## 0. Resuming this project

Written so that a reader with **none** of the previous session's context can pick it up. Do these in order.

**Read, in this order:** the top of [`../README.md`](../README.md) (what the project is and what it currently
claims) → this page (where it stands and what is next) → [`../results/runs/20260918-023600_depth-bias/FINDINGS.md`](../results/runs/20260918-023600_depth-bias/FINDINGS.md)
(the finding the entry leans on) → `gh issue list --repo IFoA-ADSWP/tabpfn-reserving` (the worklist).

**Confirm the environment before believing anything else.** The rules are four: nothing is true until
reproduced, the same command twice must give the same number, the baseline must reproduce the incumbent, and
**no script issues a verdict it has no power to support** — anything that prints a conclusion must refuse below
a stated sample threshold and when the outcome has no variation. The fourth is not theoretical: three times in
one session a script reported a confident verdict on three units, zero misses, and a rate of 0.0%.

```bash
cd ~/projects/tabpfn-reserving            # canonical clone; .venv is Python 3.12.7 with tabpfn 9.0.0
.venv/bin/python -m pytest -q tests/      # 36 tests, ~15-30s, NO token and NO model fit needed
.venv/bin/python scripts/check_figures.py # no superseded figure appears unmarked in the documents
```

Then one real run and one reproduction of the finding. The first run of the day downloads the 3.5 weights
(~30s) and needs `TABPFN_TOKEN`; the CLI reads it from `~/.config/tfm/keys.env` itself, so a stale shell
export cannot poison a run.

```bash
.venv/bin/python -m tabpfn_reserving abc --distribution
#   expect: compounded point 12,844,223 | chain-ladder 5,277,760 | route "bar-bins"

.venv/bin/python scripts/depth_bias.py --triangles abc genins --targets delta --anchors-from 3
#   expect: slope +0.0317 per step (se 0.0036), intercept ~+0.0001, 287 scored steps
```

If those three agree, the state described below is real and you can start. If they don't, the numbers have
drifted and this page is what needs fixing first.

**Pitfalls that cost time here, all of them paid for already:** a production run is ~1–2 minutes of CPU (the
fit is ~5s, the recursion's predictions dominate); `clrd` is 775 triangles and `mcl` carries incurred *and*
paid, so **both must be named explicitly** or the loader refuses them — that refusal is deliberate, it is not
a bug to work around; and a backtest anchored near the ultimate cannot see the long-horizon steps, so it will
report a flat line while telling you nothing (see the trap in the depth-bias FINDINGS).

**Then start at #18** — delete the recursion and predict the horizon directly, capped at one implementation,
one measurement, one decision. **#1 (joining and submitting) is not the machine's to do** and is the only item
with a deadline that no further work can move.

## 1. Compliance — the terms, clause by clause

| Clause | Requirement | State |
|---|---|---|
| 2.2 | A valid Prior Labs account; **entries are tied to the account** | ⚠️ Account exists (token works, 3.5 licence accepted), but **the entry is not joined and nothing is submitted**. Only the entrant can do this (**#1**) |
| 2.4 | Accept the terms | ⚠️ Same step as joining |
| 3.1 | Built with TabPFN-3.5, as a core part | ✅ `tabpfn` 9.0.0 = the 3.5 family, running locally on CPU |
| 3.2 | The repository must contain **the code and instructions to run it** | ✅ **Closed (#3, 6ae586b).** `src/tabpfn_reserving` + `pip install -e .`; the README's quickstart is copied from real output, and 25 tests run in 7 seconds without a token |
| 3.2 | Input data included **or available at a public URL** | ✅ Ships inside the pinned `chainladder` package; CAS publishes the source. **Now explicit**: `mcl` carries incurred *and* paid, so the column must be named or the run is refused |
| 3.3 | Rights to everything published | ✅ No third-party restricted data |
| 3.5 | Public repository **under Apache-2.0** | ✅ `IFoA-ADSWP/tabpfn-reserving` |
| 3.5 | **A description** a third-party developer can comprehend | ❌ **Still does not exist (#2)** |
| 3.7 | Submit before 6 Oct, 23:59 CEST | ⚠️ Nothing submitted; target 2 Oct (**#1**) |

## 2. Judge-facing — where the rubric weights land

| Weight | Needs | State |
|---|---|---|
| **50%** Showcase of TabPFN-3.5 | A working prototype demonstrating the capabilities convincingly | ✅/⚠️ The distribution is now shown, not described: a log-scaled figure read off the model's own bar distribution, with the exact sampling route recorded per run (**#16**). Still missing: the reframing view and the coverage curve (**#4**) |
| **30%** Creativity and originality | The reframing, plus practical value | ⚠️ The reframing is real, and the **depth-bias finding** is now the strongest originality evidence in the repository (§3.1). `docs/what_it_unlocks.md` still does not exist (**#6**) |
| **20%** Technical quality and reproducibility | A stranger can run it | ✅/⚠️ Package, CLI, 25 tests, CI on 3.11/3.12, per-run records with fingerprints and the draws themselves. Missing: the notebook (**#5**), pinned requirements (**#12**) |

## 3. Scientific — the honest ones

1. **The point estimate is not competitive, and this is now an answer rather than a hope.** 464 paired
   evaluations over 639 real triangles: closer than Chain Ladder on **38.8%**, median |error| 162.6% against
   the incumbent's 121.3%, moving the reserve by a median of 142% of Chain Ladder's, and losing at every
   horizon. The *mechanism* was worth finding — per-step bias is zero where the model has training rows and
   grows **+3.2% per step** as it extrapolates (287 scored steps, intercept +0.0001), and deleting the
   recursion took `abc` from +66.8% to −1.2% — but the fix did not make the estimate usable.
   → `results/fleet/FINDINGS.md`, `results/runs/20260918-023600_depth-bias/FINDINGS.md`.
2. **Calibration is poor, and it is the finding that matters most — because the distribution is the claim.**
   Over 464 fleet evaluations the intervals under-cover at **every** nominal level: 39.2% at 50, 59.9% at 75,
   78.0% at 90, 84.7% at 95 (z −4.8 to −6.6). They are *wide* — the median 90% interval is 4.55× the point
   estimate — and the truth sits above the sampled median in **67.7%** of units, so the misfit is **centring,
   not width**, which is the same failure the point-estimate result shows from the other side.
   **Settled since — this was the entry's largest open question, and the answer is the reverse of what it feared:**
   Mack and the ODP bootstrap were measured on the **identical** 464 units, and all three under-cover. At the
   levels a risk margin is actually read, **Mack's is the worst of the three** — 74.6% at nominal 90% and 79.3%
   at nominal 95%, against this model's 78.0% and 84.7% — with the bootstrap best (87.1% / 90.9%) while still
   outside 2 SE at three of four levels. The failures are shared rather than distinctive. So the honest
   comparative claim is that this model is miscalibrated *like the standard method*, and less so where it matters.
   → `results/fleet/COVERAGE.md`, `results/runs/20260919-three-method-coverage/FINDINGS.md`.
3. **The independent-draws limitation is stated but not fixed** (**#15**): per-cell draws are independent, so
   the p99 — the number a risk margin actually uses — has no correlation structure. Cheap to fix, not yet done.
4. **Timings are contaminated** (**#9**); a clean E5 run is needed before any speed claim.
5. **E4's regime map is unrun** (**#11**), so there is no "use it when…" rule.
6. ~~`requirements.txt` is unpinned.~~ **Closed (#12).** Pinned to the environment that produced `results/`,
   with `pyproject.toml` pinning the two packages that determine the science and leaving lower bounds on the
   rest so an install still works across platforms.

### 3.1 The finding, in one paragraph

A tabular foundation model used recursively is **unbiased exactly where it has training examples and biased
upward where it extrapolates**, by about 3.2% per development step, with a clean zero intercept. That is a
measurable, reproducible statement about *when a model of this class should not be trusted*, it comes with the
diagnostic that establishes it (`scripts/depth_bias.py`, 287 scored steps in one command), and the trap that
nearly hid it is documented too: backtests anchored near the ultimate cannot see the depths production uses and
report a flat line while doing so. This is the intellectual centrepiece of the entry.

## 4. What is genuinely done

- **Feasibility, end to end**, and now packaged: a triangle becomes a prediction problem, a reserve and its
  full distribution come back from one forward pass, on CPU, with no API calls.
- **The distribution claim is verified rather than asserted** — drawn from the model's own bar distribution,
  and the arithmetic underneath reproduces the CAS package's Chain Ladder **to the pound** (pinned by tests on
  three triangles, and per column on a two-column sample).
- **The over-reserve is diagnosed, not excused** (§3.1).
- **The measurement discipline is real**: the same command twice gives the same reserve to the pound, the point
  estimate does not depend on the seed, and the draw noise floor (~1.6% at 300 draws) is stated beside every
  percentile.
- **25 tests and CI**, catching three defects of one class that had already shipped once.
- **Provenance**: per-run manifests, fingerprints, logs, the draws themselves, and every figure regenerable
  from a committed command.
- **The pre-registration**, which is why a mixed result reads as an answer.

## 5. The plan, with estimates

Measured throughput, not optimism: this session completed three substantial increments (a feature mode and its
controlled A/B; the sampling overhaul; the depth-bias diagnostic). Today is **Friday 18 September**.

| item | state |
|---|---|
| **#18** direct horizon prediction — delete the recursion | **done** — the compounding went, the accuracy did not |
| **#8** the fleet: does the correction earn its place, and is the distribution calibrated | **done** — the point estimate loses (38.8% closer) and the intervals are miscalibrated at every level |
| **#6** unlocks page, **#7** README table, **#12** pins, **#5** notebook, **#4** figures | **done** |
| **#2** submission description | **done** — final numbers in, ready to submit |
| **#20** calibrate against Mack / the ODP bootstrap on the same units | the one comparison that would turn "miscalibrated" into "better or worse than standard practice" |
| **#1** join and submit | yours, ~30 min |
| **to a submittable entry** | **nothing left** — the repository satisfies every clause of the terms |

**The estimate is spent.** The 4–6 working days it projected went into #18, the fleet, the coverage run and the
packaging in two days of sessions, and the repository now satisfies every clause of the terms. What remains is
**#1** (joining and submitting, not the machine's to do), the parked research items below, and **#20** — the one
comparison that would turn "this model is miscalibrated" into "better or worse than standard practice".

**Parked deliberately:** **#10** the fleet-as-context arm (biggest build, most uncertain payoff — and now the
direct test of the leading explanation for the failure, so it is worth more than it was), **#11** the regime map,
**#15** the correlation structure (worth doing properly or stating as a limitation, not half-doing), **#9** clean
timings (parked in kanban as `t_22989d51`, waiting for an idle machine), **#19** the CLI for a user's own triangle.

**Stage 2 is planned and pre-registered** in `docs/experiments.md` §5: the conditions check against the vendor's
own documented envelope, three-method coverage on identical units (**#20**), the diagnostic arm (can the model
flag where Mack's intervals fail), and the fleet-as-context arm with its bar restated in fleet terms. Ordered
cheapest-that-can-invalidate first, each with a decision rule, a stop rule and a claim it supports.

**Stage 2, step 5.1 is done** — `results/conditions/FINDINGS.md`. Neither pre-registered trigger fired: the
vendor states no training-row floor and does not exclude extrapolation. Our regime is **unreached** by their
evidence base (theirs spans 100–1M rows and declares sub-100-row prediction out of scope; ours runs 6–33).
Their documentation predicts our negative on the **split** — on temporal and grouped data it expects parity
with tuned conventional models, not a win — and three conditions point the other way, which became step
**5.1b** (**#22**): declare the identifier columns categorical and use the extrapolating quantile transform,
both on units already on disk. The check corrected two claims of ours: the row count (a **6–33** recount, not
the single-triangle "40–60") and **"no feature engineering"**, which was not true.

**Landed since:** the E5 timing harness (`scripts/timing_bench.py`, merged) — which refuses to measure while
the machine is busy, on battery, or when the test suite has drifted from its baseline, and names what is in the
way. The measurement itself still waits (`t_22989d51`, blocked with a kind, commented with why).

**Stage 2 outcomes so far.** Three explanations were tested and **closed**: the **column** (R4 — paid 36.2% vs
incurred 38.2%, paired −1.9% inside the 2.3pp floor; `results/runs/20260919-column-comparison/`), the
**configuration** (#22 — the vendor's documented changes move the centre and never the tail; the 95% level gets
*−6.7pp* on the extrapolating transform; `results/runs/20260919-config-cells/`), and widening (not a result).
A free arithmetic measurement supports the mechanism: the truth leaves the range of training labels the fit was
shown on **21.1% of units** (5.3% of cells), one-sided upward. The **consolidated strategy is `docs/experiments.md`
§6**, with a stopping rule; **R2 (bounded target, #24)** and the **row expansion (#25)** are dispatched against
it. Cost: the classical methods are 25–35× *faster* than the model, so a pre-registered speed bar is recorded as
falsified.

**The comparative run landed (#20), and it is the entry's strongest result.** Mack and the ODP bootstrap were
measured on the identical 464 units: **all three methods under-cover**, and at 90/95 **Mack's is the worst**
(74.6% / 79.3% against this model's 78.0% / 84.7%), with the bootstrap best but still outside 2 SE at three of
four levels. The failures are *shared* — the model "misses in the same weather" as the standard method, not in
its own. That converts the entry from a negative about our method into a finding about practice.

**The novelty check landed (E11, `results/prior-art/FINDINGS.md`)** and narrows the claims honestly: no reserving
study measures interval coverage of a learning method's intervals, and no source evaluates a foundation model on
a loss triangle — so three firsts survive (first interval-coverage measurement on reserving triangles, first to
locate the failure in projection depth, first to compare against the classical intervals on identical units) —
while the bucket-grid mechanism and the general small-n calibration behaviour must be **attributed**, not claimed.

**Two arms were still running at the freeze**: R2 (the bounded target, #24) and the row expansion (#25). Both
carry pre-registered falsifiers and are decisive either way; §6's stopping rule governs what happens if they
land after the freeze — record the outcome, do not re-open the programme.

## 6. The one thing that is not mine to do

**Joining the hackathon and submitting (#1)** — about thirty minutes of the entrant's time. Until it happens
there is no entry however good the repository is, it cannot be compressed later, and it is the only item on this
page with a hard deadline attached that no further work can move.
