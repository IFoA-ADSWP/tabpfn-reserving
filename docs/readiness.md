# Are we ready? A gap ledger

> **Status:** readiness assessment, 2026-09-18 (day 4 of 19). Checked against the hackathon terms, the
> brief's definition of done, and the rubric weights — not against how the work feels. **Verdict: not yet,
> and the gap is not the science.** The hard parts are proven; what is missing is almost entirely the
> artefact a judge opens.
>
> **Every gap below is filed as an issue in this repository (#1–#12)** — this page is the ledger, the issues
> are the worklist. Labels: `submission-blocker`, `human-only`, `judge-facing`, `experiment`, `packaging`.

## 1. Compliance gaps — the terms, clause by clause

| Clause | Requirement | State |
|---|---|---|
| 2.2 | A valid Prior Labs account; **entries are tied to the account** | ⚠️ An account exists (the token works, the 3.5 licence is accepted) but **the entry is not joined and nothing is submitted**. Only the entrant can do this |
| 2.4 | Accept the terms | ⚠️ Same step as joining |
| 3.1 | Built with TabPFN-3.5, as a core part | ✅ `tabpfn` 9.0.0 = the 3.5 family, running locally on CPU |
| 3.2 | The repository must contain **the code and instructions needed to run your project** | ❌ **The instructions are broken.** The README's quickstart names `python -m tabpfn_reserving …`; no such package exists. A reader can only run `scripts/spike_e0_e1.py` |
| 3.2 | Input data included **or available at a public URL** | ✅ Data ships inside the pinned `chainladder` package; CAS publishes the source at a URL. The README should name that URL explicitly |
| 3.3 | Rights to everything published | ✅ No third-party restricted data; CAS/package data only |
| 3.5 | Public repository **under Apache-2.0** | ✅ `IFoA-ADSWP/tabpfn-reserving`, public, GitHub detects Apache-2.0 |
| 3.5 | **A description** a third-party developer can comprehend | ❌ **Does not exist** |
| 3.7 | Submit before 6 Oct, 23:59 CEST | ⚠️ Nothing submitted; target 2 Oct |

## 2. Judge-facing gaps — where the rubric weights actually land

| Weight | Needs | State |
|---|---|---|
| **50%** Showcase of TabPFN-3.5 | A working prototype demonstrating the capabilities convincingly | ⚠️ The capabilities are demonstrated *in a CSV*. **Not one figure exists.** No notebook, no demo, no picture of the distribution — which is the whole claim |
| **30%** Creativity and originality | The reframing, plus practical value | ⚠️ The reframing is real and written down; the **practical-value page (`docs/what_it_unlocks.md`) does not exist**, so the reserving actuary's reason to care is currently one sentence in a README |
| **20%** Technical quality and reproducibility | A stranger can run it | ⚠️ Provenance is now excellent (manifests, fingerprints, logs, errors). But there is no package, no CLI, no notebook, no test, no CI — and the README's quickstart does not run |

## 3. Scientific gaps — the honest ones

1. **n = 11.** The coverage result (55/82/91/91 against nominal 50/75/90/95) is *indicative, not a result*.
   E3 over the fleet is what turns it into one, and it has not run. This is the largest scientific gap.
2. **The intervals are interpolated** from a 15-level quantile grid. We now know the exact route — the bar
   distribution's `borders` and `logits` — and have not taken it.
3. **Timings are contaminated.** They were measured while other work ran on the same machine; E5 needs a
   clean run before any speed claim is made.
4. **The point estimate loses to Chain Ladder** in both formulations tried. The fleet-as-context arm that
   might fix it is designed, controlled and barred — and unbuilt.
5. **E4's regime map is unrun**, so there is no "use it when…" rule — which is the house style and the most
   useful thing we could hand an actuary.
6. `requirements.txt` is unpinned, though the brief said pins land with the first results. Results exist.

## 4. What is genuinely done

Not everything is a gap, and the expensive parts are the ones that are closed:

- **Feasibility, proven end to end**: 3.5 running locally on CPU with no API calls, a triangle becoming a
  prediction problem, a reserve and its distribution coming back.
- **The distribution claim has first evidence**, and it is the claim the project rests on.
- **The controls work.** A shuffled-target placebo at 775% median error against the model's 14.9% means the
  harness is not leaking — which is what makes every other number readable.
- **Provenance discipline** — per-run manifests, data fingerprints, committed logs, recorded errors.
- **A public Apache-2.0 repository** in the working party's own org.
- **The pre-registration**, which is why a mixed result reads as an answer rather than a failure.

## 5. The judgement call

Two paths compete for the same fourteen days, and they are not the same bet:

**Path A — make what exists presentable.** Package, CLI, figures, notebook, `what_it_unlocks`, the
description. Roughly four to five days. It guarantees a *complete, honest, runnable* entry whose claim is the
distribution, with the honest finding that the point estimate does not beat Chain Ladder.

**Path B — chase the fleet arm.** Higher ceiling: it is the one route to a point estimate that competes, and
the data (775 triangles) is already on disk. But it is research, its duration is unknown, and it competes for
the same days.

**Recommendation: A first, B after.** Get to a submittable entry by ~25–26 September, submit it, then spend
whatever remains on the fleet arm and re-submit — the latest version counts. That is exactly the sequencing
the brief set out, and the reason it set aside four days of slack.

Filed as: **#1** join and submit (yours), **#2** the description, **#3** the package and CLI, **#4** figures,
**#5** the notebook, **#6** the unlocks page, **#7** the README table — then **#8–#11** the experiments, and
**#12** the pins.

**The one thing that is not mine to do:** joining the hackathon and submitting. Until that happens there is
no entry, however good the repository is.
