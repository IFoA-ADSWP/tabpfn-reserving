# Are we ready? A gap ledger

> **Status:** updated 2026-09-18, end of day 4 of 19. Checked against the hackathon terms, the brief's
> definition of done, and the rubric weights — not against how the work feels.
>
> **Verdict then: not ready, and the gap was not the science. Verdict now: the science got real, and the
> gap is the artefact a judge opens.**
>
> Every gap is filed as an issue in this repository (**#1–#18**) — this page is the ledger, the issues are the
> worklist. Labels: `submission-blocker`, `human-only`, `judge-facing`, `experiment`, `packaging`, `redesign`.

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

1. **The point estimate explains itself, and the fix is queued.** It over-reserves (abc delta, +63.3% at best,
   against Chain Ladder) and the cause is now measured rather than guessed: **per-step bias is zero where the
   model has training rows and grows +3.2% per step as it extrapolates** (287 scored steps, t≈8.7,
   intercept +0.0001), compounding to ×1.33 over the nine steps production asks for, against a measured ×1.67.
   → `results/runs/20260918-023600_depth-bias/FINDINGS.md`, **#17 answered, #18 to build**.
2. **n = 11.** The coverage result (55/82/91/91) is *indicative, not a result*, and E3 must be run at the
   horizon the tool is used at — a backtest-based version would flatter us. **The largest gap (#8).**
3. **The independent-draws limitation is stated but not fixed** (**#15**): per-cell draws are independent, so
   the p99 — the number a risk margin actually uses — has no correlation structure. Cheap to fix, not yet done.
4. **Timings are contaminated** (**#9**); a clean E5 run is needed before any speed claim.
5. **E4's regime map is unrun** (**#11**), so there is no "use it when…" rule.
6. `requirements.txt` is unpinned (**#12**) — a five-minute job for the packaging pass.

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

| item | estimate |
|---|---|
| **#18** direct horizon prediction — delete the recursion, remove the compounding | 1.0–1.5 d |
| **#8** E3 coverage over one `clrd` line of business, at the production horizon | 1.5–2.0 d |
| packaging: **#5** notebook, **#6** unlocks page, **#7** README table, **#12** pins | 1.0–1.5 d |
| **#2** submission description + final read-through | 0.5–1.0 d |
| **to a submittable entry** | **4.0–6.0 d** |

That lands **Thursday 24 – Tuesday 29 September**: on the brief's own 25 September target, with 5 working days
of slack to 2 October and 12 to the hard close on 6 October. The variance is not build time — compute is cheap
and unattended — it is how many iterations **#18** needs. So it is capped: **one implementation, one
measurement, one decision.** If it does not clear the bar first time, the honest-negative write-up costs half a
day and the plan falls back to packaging, with the distribution as the deliverable.

**Parked deliberately:** **#10** the fleet-as-context arm (biggest build, most uncertain payoff, and #18 is a
better-motivated attack on the same problem), **#11** the regime map, **#15** the correlation structure until
there is time to do it properly, **#9** clean timings.

## 6. The one thing that is not mine to do

**Joining the hackathon and submitting (#1)** — about thirty minutes of the entrant's time. Until it happens
there is no entry however good the repository is, it cannot be compressed later, and it is the only item on this
page with a hard deadline attached that no further work can move.
