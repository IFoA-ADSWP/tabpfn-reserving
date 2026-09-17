# The goal, and the experiments that get us there

> **Status:** design note, 2026-09-17. Written before any run. It fixes what we are trying to establish, which
> experiments establish it, and which creative uses of the model we have decided to test rather than admire.
> Bars and falsifiers live in [`method.md`](method.md); schedule and roles in [`brief.md`](brief.md).

## 1. The goal, at three altitudes

**As a product.** Point the repository at a loss triangle and get a reserve *with a usable distribution*,
from a single forward pass — against the method the industry uses, which reaches the same distribution from a
thousand simulated refits. One command, public data, reproducible.

**As a scientific question.** Not "can a foundation model reserve?" — that invites a single cherry-picked
triangle. The question is: **is a tabular foundation model's predictive distribution calibrated for the
reserving problem?** The point estimate is the easy half and the industry already has a good answer for it.
The uncertainty is the half that matters, because a risk margin, an IFRS 17 risk adjustment and a Solvency II
capital figure are all *quantiles*, and the incumbent quantiles are known to be optimistic.

**As a judged entry.** A judge should conclude three things in five minutes: this uses TabPFN-3.5
prominently and natively; the reframing is a genuine idea rather than a wrapper; and the numbers could be
reproduced by a stranger. The distributions, the fleet and the honest failure case are how those three
conclusions get earned.

## 2. The experiments

Staged so each one is worth running on its own, and gated so a surprise costs days rather than weeks. Every
arm is compared on the **same folds** — the triangle analogy of this project's house rule that a comparison
is only as good as the folds being shared.

### E0 — Calibrate the instrument, before claiming anything

Cheap, and it is the difference between a measurement and a story.

| Arm | What it establishes |
|---|---|
| Same config, twice | Run-to-run variation. Without it, a small edge is indistinguishable from noise |
| **Placebo: shuffled targets** | The reserve should collapse toward a naive level. If a shuffled target still produces a plausible reserve, the harness is leaking and every later number is void |
| Anchor-time assertion | Every feature row is checked to use only cells with `origin + development ≤ as-of`. Printed with each result, not assumed |
| Naive baselines | Reserve = last observed cumulative (no development); reserve = volume-weighted factor applied flat. Any model that cannot beat these has nothing to say |

### E1 — Head-to-head on the classics (the MVP experiment)

| | |
|---|---|
| Objects | `abc`, `genins`, `mcl`, `ukmotor` — small, well-understood, published examples |
| Arms | Chain Ladder · Mack · ODP bootstrap · TabPFN-default |
| Design | **Every diagonal is a valuation date.** Predict the remaining reserve as of each one and score against what actually happened — so one triangle yields many observations instead of one |
| Metrics | Reserve error (per accident year and total), rank, and the sign of the error |
| Settles | Parity on the point estimate, and whether any edge is bigger than E0's noise |

### E2 — Mechanism: which formulation predicts best (the intellectual core)

This is the experiment that turns a demo into research. Same data, five ways of asking the question:

| Arm | The idea | Plays to |
|---|---|---|
| **A1 — amounts** | Predict each future cell's cumulative amount directly | The obvious baseline |
| **A2 — link ratios** | Predict the *development factor* `C_{a,d+1}/C_{a,d}` instead, then apply it | The estimand actuaries actually trust; bounded target; comparison to Chain Ladder becomes direct |
| **A3 — recursive** | Predict one diagonal ahead, append it to the context, predict the next | How an actuary projects; the far cells become a sequence of near cells |
| **A4 — distributional classification** | Predict `P(development exceeds threshold)` for a ladder of thresholds; rebuild a distribution from the survival function | This project's own house finding that a count target reframed as classification moves TabPFN from its weakest axis to its strongest |
| **A5 — features** | Raw cells vs origin/development only vs explicit link ratios vs calendar-year effects | Tests our own claim that no feature engineering is needed — and recovers *what the model is using* |

A5's interpretability arm matters beyond performance: Chain Ladder **imposes** no accident-year effect by
construction. If the model finds one, that is a result an actuary can use, and it is the kind of finding that
reads as originality rather than as a leaderboard.

### E3 — The distribution, which is the point

| Arm | What it establishes |
|---|---|
| Coverage at 50/75/90/95% over the fleet, TabPFN vs Mack vs ODP | Whether the intervals are honest, and whether they are *more* honest than the incumbent's |
| CRPS / pinball loss per cell | A proper distributional score — coverage alone rewards being wide |
| **Joint vs independent sampling of cells** | A subtlety with teeth: summing independently-sampled cell quantiles ignores the correlation between cells that Mack models explicitly. Predicting the whole future diagonal in one fit lets the shared context induce dependence. Testing this is the difference between a distribution and an interval arithmetic trick |
| Interval width vs Mack's | Narrower and better-covering is the win; wider and better-covering is just conservative |

### E4 — The fleet: where it wins, and the decision rule

The CAS Loss Reserve Database through `chainladder`: hundreds of real insurer triangles across six lines of
business. Sliced by triangle size, tail length, line of business and valuation age — producing the same kind
of adoption rule the working party's insurance benchmark produced ("use it when …; prefer the incumbent
when …"). This is what gives E1–E3 statistical weight rather than anecdote.

### E5 — Cost, and the workbench claim

Per-triangle wall-clock against the ODP bootstrap on the same machine; fleet throughput; the **Fast
checkpoint** arm; and **KV-cache reuse**, which is the interesting one — fitting once on the observed cells
and evaluating many query cells (every future cell, every valuation date, every triangle) is exactly the
pattern `fit_with_cache` exists for.

### E6 — Where it fails, and whether compute helps

The worst triangles and cells, stated in the open. Plus one targeted arm: **Thinking mode on the late and
thin cells**, where signal is weakest and the tail is decided. If inference-time compute buys calibration
where data is scarce, that is a genuinely new result; if it does not, that is worth one line and no more.

### Not running (parked, with reasons)

Fine-tuning (GPU, hours, and it contradicts the entry's claim) · individual-claim reserving (different public
data problem) · a Kaggle-style accuracy chase (crowded, and the model is not the subject).

## 3. Creative uses of TabPFN — ranked, with verdicts

The model is not a drop-in regressor here. What follows is every genuinely distinct way its capabilities
could meet this task, scored on whether it shows the model off *and* is defensible actuarially.

| # | The idea | Why it is creative | Verdict |
|---|---|---|---|
| 1 | **The distribution as the product** — sample the cell-level predictive distribution and sum | The industry retires a *point estimate* and buys the distribution separately with simulation; here it falls out of the same call | **Headline** (E3) |
| 2 | **Joint rather than marginal sampling** — one fit over the whole future diagonal so the shared context induces dependence | Nothing in the naive version gets correlation right, and Mack explicitly models it. This is where a foundation model could beat a formula rather than imitate it | **In** (E3) — highest scientific value per hour |
| 3 | **Distributional classification** — a ladder of "exceeds threshold" binaries, distribution rebuilt from the survival curve | Uses this project's own discovered lever (classification is the model's strong axis; regression is not) on a target that is stubbornly continuous | **In** (E2/A4) |
| 4 | **Predict link ratios, not amounts** | Puts the model in development-factor space — the estimand actuaries trust — so the comparison to Chain Ladder is apples-to-apples and the output is bounded | **In** (E2/A2) |
| 5 | **Recursive diagonal-stepping** — each predicted diagonal becomes context for the next | Mirrors how a real projection unfolds, and converts a long extrapolation into a series of short ones | **In** (E2/A3) |
| 6 | **KV-cache as the fleet engine** — fit once per triangle, evaluate every query cell, diagonal and re-valuation cheaply | The workbench claim ("re-reserve the book every quarter") stops being rhetorical | **In** (E5) |
| 7 | **The fleet as context** — give the model other triangles' summaries, so cross-insurer information informs this insurer's development | In-context meta-learning is the model's defining mechanism, and reserving has never had a way to borrow strength across a book of triangles | **Park** — highest ceiling, highest leakage risk. Revisit as an explicitly controlled experiment if E1–E3 land early |
| 8 | **Triangle embeddings** — embed and cluster the fleet by development pattern | A map of the fleet is both a demo asset and a defensible basis for E4's regime slices, better than slicing on size alone | **Stretch** — cheap to try on the fleet, and visually strong |
| 9 | **Anomaly detection as an audit tool** — unsupervised flagging of odd cells and diagonals | Shows a capability nobody brings to a hackathon; a real actuarial workflow (someone eyeballs every triangle today) | **Stretch** |
| 10 | **Calendar-year trend as a time series** — forecast the diagonal (inflation, court awards) and feed it back as a feature | Uses a *different* model in the TabPFN family to attack the gap every classical method has: calendar-year effects | **Park** — one arm, easy to bolt on if E2 shows calendar effects matter |
| 11 | **Synthetic tail generation** — use the generative capability to stress-test methods where real data is thinnest | Attractive, but risks reading as method-for-method's-sake | **Park** |
| 12 | **Model blending** — let TabPFN learn to weight Chain Ladder, Mack and the bootstrap | Practical, and a likely accuracy win — but it makes the model a meta-learner rather than the subject, which the rubric punishes at 50% weight | **Park** |

**The three that earn their place on the science as well as the demo:** joint sampling (#2), distributional
classification (#3), and link-ratio prediction (#4). Each is a different answer to "what is being predicted",
each is cheap, and E2 measures them against each other rather than asserting one.

## 4. What this buys, stated as the judge's five minutes

1. A reserve, a distribution, and a wall-clock — against Chain Ladder, Mack and the ODP bootstrap (E1, E3, E5).
2. The reframing is not packaging: four formulations of the same question, measured (E2).
3. It is a measurement, not a pitch: a placebo arm, shared folds, pre-registered bars, and the failures
   published next to the wins (E0, E6).
