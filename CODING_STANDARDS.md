# Coding Standards

Judgement calls only. There is no linter or formatter configured in this repo, so nothing here restates
mechanical style. What follows is the discipline the repository actually lives by: a claim is pre-registered,
a number is regenerable, a difference clears its noise floor, and a check proves it can fail.

## Verification

- `pytest -q tests/` — the arithmetic and leakage tests, in seconds, with no token and no model fit (CI job
  `pytest`, Python 3.11 and 3.12)
- `python scripts/check_figures.py` — fails when a superseded figure appears in a document unmarked; exit 2
  means the checker itself was blind. `--self-test` proves it can fail and can pass

No lint, format or type-check command exists in the repo's config or CI; say so rather than inventing one.

## Claims and evidence

### Pre-register the bar before the run

Bars and falsifiers live in `docs/method.md` and are fixed before the numbers exist, so an outcome cannot
quietly become the thing that was being tested. A missed bar is recorded as falsified — the speed bar is the
standing example — not re-scoped into a different claim.

### Every quoted number is regenerable by a committed command

A number in a document comes from `results/`, and from a script that can rebuild it. Figures are produced by
`scripts/make_figures.py` from the recorded units rather than typed in, "so it cannot drift from the record".
If you cannot name the command, the number does not go in the prose.

### A difference has to clear the measured noise floor

Comparisons report a standard error or the noise floor beside the difference: 2.3pp on the paired
comparisons, ~1.6% draw noise on the median at 300 draws. An effect inside the floor is reported as no
effect, even when its sign flatters the method.

### The null is measured, not assumed

A comparison's baseline is computed on the same units and the same held-out diagonals as the arm. The one
time a stored null was zero by construction, "closer on 0.0%" looked like a result; `results/fleet/FINDINGS.md`
records the defect and the correction. If the arithmetic makes the null zero, the harness is wrong.

### Shared folds, or no comparison

Every arm is scored on the same folds and the same realised future — "a comparison is only as good as the
folds being shared". An arm evaluated on its own units is an anecdote, not a cell in the table.

### A check must be able to fail

`scripts/check_figures.py` exits 2 when it finds nothing to check, and ships `--self-test` with a corpus that
must fail and one that must pass. A check that cannot fail is indistinguishable from no check, so new gates
arrive with a control arm.

## Reproducibility and records

### Every run writes its own record

A run records the command, the git revision, package versions, a fingerprint of the matrix it used,
wall-clock, the distribution route, and any errors. Run directories live under `results/runs/<run-id>/` with a
`FINDINGS.md`; the draws themselves are kept where the record can point at them.

### Do not change the defaults that produced recorded results

`arm.make_model` is the single construction point and every option defaults to the recorded configuration;
new behaviour is opt-in. Changing a default would invalidate stored runs silently instead of adding a cell to
them, which is the reason the two documented options are flags.

### Pin the environment that produced the results

`requirements.txt` pins exactly the versions behind `results/`; `pyproject.toml` keeps lower bounds only so
the package installs across platforms. A result quoted as reproducible names the pinned environment, and the
pins move when the environment that produced the results moves.

## Leakage and correctness

### Features see nothing past the anchor

Every feature row is restricted to cells with `origin + development <= anchor` for the valuation date being
predicted, and the assertion is printed with each result rather than assumed. The anchor-time check exists
because the failure is silent: leaked cells produce a plausible reserve.

### A re-implemented classical method is pinned to the reference package

`classical.mack_horizon` re-implements Mack's recursion to stop at each origin's target age, and is
legitimate only because it reproduces `chainladder`'s `ibnr_`, `mack_std_err_` and process risk exactly at
run-off. That is the first test, and everything else in the module rests on it. Any re-derivation of a
baseline gets the same pin before it is used.

### Refuse loudly rather than return a number

The CLI refuses a triangle that would return `NaN` instead of emitting a plausible-looking reserve
(#19). When an input cannot support the computation, the correct output is a refusal that names the reason.

## Working in a worktree

### Tests and scripts must import the tree they are in

`pyproject.toml` sets `pythonpath = ["src"]` because the venv lives only in the primary checkout and its
editable install points there; without it, a pytest run in a worktree silently certifies code from the other
tree. Scripts carry their own `sys.path` bootstrap for the same reason. Do not remove either, and do not
trust a pass that imported the wrong tree.

## Documents

### The negative is reported where the claim was made

The README leads with the failure, the coverage table states that all three methods miss, and a falsified bar
stays visible beside the claim it falsified. A later document that changes a number has to mark the superseded
one, or `check_figures.py` will fail the run.
