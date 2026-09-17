# Project brief — `tabpfn-reserving`

> **Status:** initiating document, 2026-09-17 (day 3 of 19). `README.md` says what the project *is*,
> `docs/method.md` fixes the claims and the bars, and this brief says who does what, by when, and what counts
> as done. Where this brief and `method.md` disagree about a number, `method.md` wins.

## 1. Objective

Deliver a **public, reproducible project** that demonstrates one TabPFN-3.5 capability convincingly — the
full predictive distribution — in a domain where the incumbent method is not machine learning at all: **loss
reserving**. Concretely: point the repository at a loss triangle and get a reserve *with its distribution*,
measured against Chain Ladder, Mack and the ODP bootstrap on public data.

## 2. Why this project

| The rubric | Where this project answers it |
|---|---|
| **50%** — how convincingly the model's capabilities are demonstrated in a working prototype | Five capabilities on one spine, each with a named job: default prediction, **predictive distribution**, Thinking mode, Fast, calibration/coverage |
| **30%** — novelty and practical value | A mathematical object the model has never seen, reframed as prediction. Practical value a reserving actuary can name: a risk margin, an IFRS 17 risk adjustment and a Solvency II capital figure are *quantiles*, not standard errors |
| **20%** — technical quality and reproducibility | One command, public data by URL and hash, results committed with the code that produced them |

Secondary value: the workbench and the fleet backtest are usable material for the working party's paper stage
after the competition, whatever the placement.

## 3. The event, as constraints

Self-contained, so nobody has to reconstruct it from the Terms:

| | |
|---|---|
| Window | 15 Sept 2026 00:00 CEST → **6 Oct 2026 23:59 CEST** (Tuesday). Today is day 3 of 19 |
| Who enters | Individuals, 18+, **one Prior Labs account**; entries are tied to the account (clause 2.2) |
| The entry | A link to a public repository **under Apache License 2.0**, plus a description a third-party developer can follow (3.5) |
| Data | Input data included, **or available at a public URL** (3.2) |
| Rights | The submitter certifies they hold the rights to everything published (3.3) |
| Multiplicity | More than one project may be submitted; each judged separately; the **latest** version of an entry counts (3.7) |
| Credits | API free with a limited allowance; **extra credits released on joining** (3.4) |
| Prizes | 1st NVIDIA DGX Spark · 2nd NVIDIA Jetson AGX Orin 64GB · 3rd RTX 4090 · honourable mentions are recognition (6.1) |

## 4. Scope

**In:** the triangle-to-cells reframing · TabPFN-3.5 arms (Base, Thinking, Fast) · the reserve distribution ·
anchor-time discipline against leakage · coverage measurement · the fleet backtest over public triangles ·
a CLI and a Colab notebook · committed results · the two public docs.

**Out, deliberately:** individual-claim reserving (a live research literature, harder public data) ·
fine-tuning (needs a GPU, hours and a sponsor, and the claim here is that none of that is necessary) ·
tail-extrapolation methodology · claims *pricing* benchmarks · a scale stress test (a triangle is tens to
hundreds of rows — this uses the model in its small-data strength) · anything requiring private or
working-party portfolio data.

## 5. Deliverables

| Artifact | State | Done when |
|---|---|---|
| Public Apache-2.0 repository | **Exists** — `IFoA-ADSWP/tabpfn-reserving`, commit `4dc9e72` | — |
| `docs/brief.md` (this page) | This commit | — |
| `docs/method.md` — claims, falsifiers, pre-registered bars | **Exists** | Revisited once, after the spike, if a gate fails |
| `src/` package + CLI | Not started | One triangle end to end from the command line |
| `scripts/fetch_data.py` | Not started | Downloads public triangles, records URLs and hashes |
| `notebooks/quickstart.ipynb` | Not started | Runs end to end in Colab, distribution plotted |
| `results/` | Not started | Every number the README quotes is in here, with the command that made it |
| `README.md` with real output | Draft exists, no results | Quickstart copied from actual runs |
| Submission description | Not started | A third-party developer can follow it without asking us anything |
| Two-minute video *(optional field)* | Not started | Beats 1–4 on screen |

## 6. Plan and dates

Critical path is the spike, then one triangle, then the fleet. Dates are targets, not aspirations: the
submission goes in **2 October**, four days early, because updates are allowed until close and the latest
version counts.

| When | What | Gate |
|---|---|---|
| Thu 17 Sept (today) | Brief; spike environment; the four spike gates | — |
| Fri 18 – Sat 19 | **Spike**: one triangle end to end in a throwaway venv | Gates 1–4 (§9). If gate 2 fails, the headline changes and we replan *before* building |
| Sun 20 – Mon 21 | **MVP**: reframing + reserve + distribution + Chain Ladder comparison, committed | One command reproduces the first figure |
| Tue 22 – Thu 24 | **v1**: fleet backtest — coverage, per-triangle error, wall-clock; Thinking arm on late cells; Fast arm | The pre-registered bars are measurable |
| Fri 25 – Sat 26 | Figures and `results/`; README rewritten from real output | No README number without a command |
| Sun 27 – Mon 28 | Colab notebook; `docs/what_it_unlocks.md`; video script | A stranger can run it |
| Tue 29 – Wed 30 | Submission description drafted; internal read-through | Readable from the PR alone |
| Thu 1 – Fri 2 Oct | **Submit** (buffer). Video recorded | Entry is in, clause 3.5 satisfied |
| Sat 3 – Mon 5 | Improvements only; re-submit as the latest version | Each change is small and verified |
| Tue 6 Oct 23:59 CEST | **Close** — nothing after this counts | — |

## 7. Success criteria

1. **The bars in `method.md`**, pre-registered: coverage within ±3 points of nominal and at least as close
   as Mack's on the same triangles; no systematic bias across the fleet; at least an order of magnitude
   faster per triangle than the ODP bootstrap.
2. **A complete, reproducible submission** — public repo under Apache-2.0, README that runs, data public by
   URL and hash, description written, entry submitted by 2 October.
3. **Honest reporting** — if a bar fails, the README says so and the result stands as the result. This is not
   a soft criterion: it is the reason the project is credible at all.
4. Stretch, and explicitly not the measure of success: an honourable mention.

## 8. Roles and decision rights

| Role | Who | Notes |
|---|---|---|
| **Entrant** | **Open** — must be one individual (clause 2.2), because entries are tied to one account and prizes ship to one address | The only genuinely blocking role. The working party is acknowledged in the README, not presented as the entrant |
| Repository owner | `IFoA-ADSWP` (public, Apache-2.0) | Created 2026-09-17 |
| Engineering | This session, on branches; commits reviewed before they reach `main` | Evidence discipline in §10 |
| Spend authority | Any paid run needs an explicit yes. **None is anticipated** — triangle-scale inference runs on CPU with the open-source package, so the core project needs no API credits at all | The API path is an arm (Thinking), not the backbone |
| Working-party sign-off | Using workstream time, and the acknowledgement of the ADSWP evidence base | Recorded in the ADSWP workstream docs, PR #194 |

## 9. Risks and mitigations

| Risk | Early signal | Response |
|---|---|---|
| `output_type="full"` does not return per-cell quantiles that can be summed into a reserve distribution | Spike gate 2, on day 4 | The headline moves to beat 4 (coverage from repeated sampling) or beat 2; replan before building |
| The point estimate is systematically worse than Chain Ladder | Fleet backtest, mid-plan | The entry is a *measurement*: the bars are pre-registered, so this is a result, not a failure |
| Coverage is no better than Mack's | Fleet backtest | Say so; the entry keeps the distribution-shape and speed beats |
| Leakage makes the first result look excellent | Spike gate 4 | Anchor-time split enforced in code; the split is printed with every result |
| The clock | Any milestone slipping a day | MVP first; submit 2 October; improve afterwards, since the latest version counts |
| Credits or rate limits | Running the Thinking arm | The core path is CPU-local and needs no API; the subscription-free allowance covers the rest |

## 10. Working agreements

- **No number in the README that a command in the repository cannot regenerate.**
- Claims and falsifiers are written **before** the runs (`method.md`), not after.
- Results are committed, and the environment that produced them is pinned next to them.
- Data is downloaded at run time, never vendored; no secrets, tokens or portfolio data in the repository.
- Every figure carries the command that made it.
- Failures are reported in the same place as successes.

## 11. Definition of done — the submission checklist

- [ ] Repository public, on `main`, **Apache-2.0** (clause 3.5)
- [ ] `README.md` runs from a clean clone, with output copied from real runs
- [ ] Data available by public URL with hashes recorded (clause 3.2)
- [ ] `results/` committed, with the environment pins
- [ ] Notebook runs in Colab
- [ ] Submission description written for a third-party developer
- [ ] Entry submitted, **before 2 October**, with four days of slack
- [ ] Nothing in the entry that we cannot certify the rights to (clause 3.3)

## 12. Open decisions

| # | Decision | Owner |
|---|---|---|
| 1 | **Who enters** (clause 2.2 — one individual, one account) | Scott |
| 2 | Joining the hackathon, which releases the extra API credits | Scott (needs a Prior Labs account sign-in) |
| 3 | Whether to submit a second, cheaper entry (the ADSWP 3.5 version-drift re-test) once this one is safe | Scott |

Working-party-side documents for this project — the event assessment, the idea log and the demonstration plan
— live in `IFoA-ADSWP/tabular-foundation-model` (PR #194) rather than here.
