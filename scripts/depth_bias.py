"""Is the compounding per-step bias, and does that bias grow with depth? (#17, hypothesis 1)

In both arms the over-reserve is multiplicative: the point estimate compounds a per-step ratio once per
development step, up to nine times for the youngest origin. So a per-step bias of a few percent becomes a
large error by the ultimate, and the only question worth asking is whether such a bias is there.

Every training row sits at depth <= 0 -- the cell's level is observed -- while production asks for steps 1
through 9. So this measures, for every step of every projection:

    log(predicted ratio / actual ratio), against depth

If the mean log error is flat in depth, the compounding is not per-step bias and hypothesis 1 is dead. If it
rises, the slope * depth is the whole story, and the slope is the per-step bias to attack.

Two arms, and which one is being measured is named rather than assumed (#25):

* ``--arm recursive`` (the default, and what the recorded run used) measures the per-cell link-ratio error of
  a step at depth `d`, i.e. how far that step sits beyond the training distribution.
* ``--arm direct`` measures the same *question* for the arm whose training rows the row expansion changes:
  one prediction per `(origin, target age)`, `depth` is the number of development steps that single
  prediction spans, and the error is the error of that whole run. It is a **different regression** from the
  recursive table's, so its slope is not the recorded slope; the like-for-like comparison is the direct arm
  before and after the expansion, on the same triangles and anchors.

Run:  python scripts/depth_bias.py [--triangles abc genins mcl ukmotor] [--targets delta ratio]
                                   [--arm recursive|direct] [--intermediate-ages]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# The venv lives only in the primary checkout and its editable install points at *that* checkout's `src/`,
# so without this bootstrap a script run from a kanban worktree imports the other tree's package -- which
# looks like "my change had no effect" and reports another branch's numbers as this run's. Same bootstrap as
# `scripts/fleet_eval.py`, and the reason the fleet scripts have always been run through it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import Triangle, direct_features, direct_training_rows, training_rows

# A slope is a verdict, so it is behind a stated threshold and a variation check: below this many scored
# steps the run reports its table and refuses the regression, and a run whose scored errors are all identical
# has no slope to estimate.
MIN_STEPS_FOR_A_SLOPE = 50


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Slope, intercept and slope standard error. Small enough to write out rather than import."""
    n = len(x)
    if n < 3:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "slope_se": float("nan")}
    xbar, ybar = x.mean(), y.mean()
    sxx = ((x - xbar) ** 2).sum()
    if sxx == 0:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "slope_se": float("nan")}
    slope = ((x - xbar) * (y - ybar)).sum() / sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    dof = n - 2
    se = float(np.sqrt((resid ** 2).sum() / dof / sxx)) if dof > 0 else float("nan")
    return {"n": n, "slope": float(slope), "intercept": float(intercept), "slope_se": se}


def refusal(n: int, values: np.ndarray) -> str | None:
    """Why a slope must not be published, or None. Threshold first, then variation."""
    if n < MIN_STEPS_FOR_A_SLOPE:
        return (f"REFUSING A SLOPE: {n} scored steps is below the stated minimum of {MIN_STEPS_FOR_A_SLOPE}. "
                f"The table above is a description of this run, not a per-step bias.")
    if np.nanstd(values) == 0:
        return ("REFUSING A SLOPE: every scored step has the same log error -- the outcome has no variation, "
                "so there is nothing to regress.")
    return None


def direct_steps(model, tri: Triangle, anchor: int, gf: np.ndarray, target: str) -> list[dict]:
    """Scored predictions for the *direct* arm, one per (origin, target age) the anchor can score.

    The direct arm predicts an origin's whole remaining factor in one row, so there is no recursion to trace:
    ``depth`` is the number of development steps that one prediction spans, and ``log_error`` is the error of
    that whole run. **This is not the recursive table's quantity** -- there, depth is how far a single step
    sits beyond the training distribution and the error is one cell's link ratio -- so the two slopes are not
    comparable and only the direct arm's own before/after comparison is like for like.

    Targets stop at each origin's last *observed* age (`n-1-a`), which is the only future the data can score;
    the clamp that keeps the labels honest lives in the row builder, not here.
    """
    queries = []
    for a in range(anchor + 1):
        age = anchor - a
        base = tri.values[a, age]
        if not (np.isfinite(base) and base > 0):
            continue
        for t in range(age + 1, tri.n - a):
            end = tri.values[a, t]
            if not (np.isfinite(end) and end > 0):
                continue
            queries.append((a, int(t), int(age), float(base), float(end),
                            direct_features(tri, a, anchor, gf, t)))
    if not queries:
        return []
    X = np.asarray([q[5] for q in queries], dtype=float)
    point = np.atleast_1d(np.asarray(model.predict(X), dtype=float))
    rows = []
    for (a, t, age, base, end, row), p in zip(queries, point):
        # The same rescaling `arm.reserve_direct` applies, so this diagnostic measures the arm the fleet runs.
        cl = np.exp(row[-1]) if np.isfinite(row[-1]) else 1.0
        pred = float(p) * (cl if target == "delta" else 1.0)
        actual = end / base
        rows.append({
            "origin": int(a), "dev": int(t), "depth": int(t - age),
            "predicted_ratio": float(pred), "actual_ratio": float(actual),
            "prev_observed": float(base), "base_observed": float(base),
            "log_error": (float(np.log(pred / actual)) if pred > 0 and actual > 0 else float("nan")),
        })
    return rows


def decompose(rows: list[dict], target: str) -> dict:
    """Currency decomposition of the reserve error, per origin and per depth.

    A reserve is a sum of currency increments, so the honest statistic is not the unweighted mean of log
    ratios -- a 1% error on a large origin moves the reserve more than 5% on a small one, and averaging
    logs weights them equally. Each step is weighted by the exposure it acts on, and the per-origin
    reconstruction is checked against the reserve the run reported, so the decomposition cannot quietly be
    describing a different number from the one it claims to explain.
    """
    out: dict = {"by_depth": {}, "by_origin": [], "check": {}}
    sub = [r for r in rows if r["target"] == target]
    if not sub:
        return out

    # Per step, weighted by the currency level the step acts on (using the observed history as exposure).
    by_depth: dict[int, dict] = {}
    for r in sub:
        if not (np.isfinite(r["actual_ratio"]) and np.isfinite(r["prev_observed"])):
            continue
        d = r["depth"]
        rec = by_depth.setdefault(d, {"n": 0, "log_sum": 0.0, "exposure": 0.0, "weighted_err": 0.0,
                                      "log_weighted": 0.0})
        rec["n"] += 1
        rec["log_sum"] += r["log_error"]
        exposure = float(r["prev_observed"])
        rec["exposure"] += exposure
        # currency residual of this step's ratio, as it acts on the observed level
        rec["weighted_err"] += exposure * (r["predicted_ratio"] - r["actual_ratio"])
        rec["log_weighted"] += exposure * r["log_error"]
    for d, rec in by_depth.items():
        out["by_depth"][d] = {
            "n": rec["n"],
            "mean_log_error": rec["log_sum"] / rec["n"] if rec["n"] else float("nan"),
            "exposure": rec["exposure"],
            "currency_bias_per_unit_exposure": (rec["log_weighted"] / rec["exposure"]
                                                if rec["exposure"] else float("nan")),
            "currency_residual": rec["weighted_err"],
        }

    # Per origin: rebuild both paths from the anchor base and compare the increments. This reconstructs
    # the reserve the run reported, which is the check that the attribution is of the actual number.
    for (tri_name, anchor), group in _group_by(sub).items():
        for a, steps in _group_by_origin(group).items():
            steps = sorted(steps, key=lambda r: r["dev"])
            base = steps[0]["prev_observed"]
            if not np.isfinite(base) or base <= 0:
                continue
            pred, actual = base, base
            for r in steps:
                pred *= r["predicted_ratio"]
                if np.isfinite(r["actual_ratio"]):
                    actual *= r["actual_ratio"]
            out["by_origin"].append({
                "triangle": tri_name, "anchor": anchor, "origin": a, "steps": len(steps),
                "depth_max": steps[-1]["depth"], "base_observed": base,
                "predicted_increment": pred - base, "actual_increment": actual - base,
                "residual": (pred - base) - (actual - base),
            })
    return out


def _group_by(rows: list[dict]) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["triangle"], r["anchor"]), []).append(r)
    return groups


def _group_by_origin(rows: list[dict]) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["origin"], []).append(r)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triangles", nargs="*", default=["abc", "genins", "mcl", "ukmotor"])
    ap.add_argument("--targets", nargs="*", default=["delta", "ratio"])
    ap.add_argument("--anchors-behind", type=int, default=3, help="how many anchors below the last to use")
    ap.add_argument("--anchors-from", type=int, default=None,
                    help="use every anchor from this one up to the last but one. Earlier anchors project "
                         "further, so this is what reaches production-like depths: with the last few "
                         "anchors the deepest step measurable is n-2-k, i.e. 0-2 steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="recursive", choices=["recursive", "direct"],
                    help="which arm's steps to score; 'direct' is the arm the row expansion changes (#25)")
    ap.add_argument("--intermediate-ages", action="store_true",
                    help="direct arm only: train on every intermediate target age, not just the ceiling")
    args = ap.parse_args()
    # mcl carries incurred and paid; the column changes the numbers, so it is named here (the spike script
    # silently took the first, which is incurred -- so these are comparable with the spike's mcl figures).
    COLUMNS = {"mcl": "incurred"}

    run_id = time.strftime("%Y%m%d-%H%M%S") + ("_depth-bias" if args.arm == "recursive"
                                              else "_depth-bias-direct")
    out_dir = pathlib.Path("results/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    rows: list[dict] = []

    def say(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    # The recorded run's log names only two of its flags, so its own command line cannot be recovered from
    # it. This one records what was actually invoked.
    say(f"command: {sys.executable} scripts/depth_bias.py " + " ".join(sys.argv[1:]))
    say(f"anchor policy: the last {args.anchors_behind} anchors below the ultimate, per triangle"
        + (f"; every anchor from {args.anchors_from} up to the last but one"
           if args.anchors_from is not None else ""))
    say(f"arm: {args.arm}" + (f" | intermediate_ages={args.intermediate_ages}" if args.arm == "direct" else ""))
    say("")

    for name in args.triangles:
        tri = Triangle.load(name, column=COLUMNS.get(name))
        anchors = (list(range(args.anchors_from, tri.n - 1)) if args.anchors_from is not None
                   else list(range(tri.n - 1 - args.anchors_behind, tri.n - 1)))
        max_depth = max((tri.n - 2 - k) for k in anchors) if anchors else 0
        say(f"{name}: anchors {anchors} -- deepest step reachable is depth {max_depth}")
        for target in args.targets:
            for k in anchors:
                gf = tri.global_factors(tri.known(k))
                if args.arm == "recursive":
                    X, y = training_rows(tri, k, gf, target=target)
                    model = arm.make_model("local")
                    model.fit(X, y)
                    trace: list = []
                    arm.reserve(model, tri, k, 0, np.random.default_rng(args.seed),
                                target=target, trace=trace)
                    scored = [t for t in trace if np.isfinite(t["log_error"])]
                else:
                    # The fleet's own row policy, so this diagnostic measures the arm the fleet cell does:
                    # training anchors 2..k-1, every label clamped to the evaluation anchor (#18, #25).
                    X, y = direct_training_rows(tri, list(range(2, max(3, k))), target=target,
                                                known_until=k, intermediate_ages=args.intermediate_ages)
                    model = arm.make_model("local")
                    model.fit(X, y)
                    scored = [t for t in direct_steps(model, tri, k, gf, target)
                              if np.isfinite(t["log_error"])]
                for t in scored:
                    rows.append({"triangle": name, "target": target, "anchor": int(k), **t})
                if scored:
                    errs = np.array([t["log_error"] for t in scored])
                    say(f"{name:>8} {target:>5} k={k:<2} rows={len(y):<4} steps scored={len(scored):<3} "
                        f"mean log error={errs.mean():+.4f}"
                        + (f"  (x{np.exp(errs.mean()):.4f} per step)" if args.arm == "recursive"
                           else "  (per prediction, across steps)"))

    if not rows:
        say("no scored steps -- nothing to conclude")
        return 1

    # The direct arm's depth is the span of a whole prediction, so its table carries the per-step
    # normalisation as an extra column; the recursive arm's rows are single steps already, and there is
    # nothing to normalise. The two are named differently so a reader cannot quote one as the other.
    per_step = args.arm == "direct"
    header = "  <-- depth = steps spanned by ONE prediction" if per_step else ""
    say("")
    say(f"=== mean log error by depth (all triangles and anchors pooled) ==={header}")
    say(f"{'depth':>5} {'steps':>6} {'mean log err':>13} {'x error':>9} {'se':>8}"
        + (f" {'log err/step':>13}" if per_step else ""))
    depths = sorted({r["depth"] for r in rows})
    by_depth = {}
    for d in depths:
        e = np.array([r["log_error"] for r in rows if r["depth"] == d])
        by_depth[d] = {"n": int(len(e)), "mean": float(e.mean()), "se": float(e.std(ddof=1) / np.sqrt(len(e)))
                       if len(e) > 1 else float("nan"),
                       "mean_per_step": float(np.mean(e / d)) if d else float("nan")}
        say(f"{d:>5} {len(e):>6} {e.mean():>+13.4f} {np.exp(e.mean()):>9.4f} "
            f"{by_depth[d]['se']:>8.4f}" + (f" {by_depth[d]['mean_per_step']:>+18.4f}" if per_step else ""))

    x = np.array([r["depth"] for r in rows], dtype=float)
    yv = np.array([r["log_error"] for r in rows], dtype=float)
    fit = ols(x, yv)
    say("")
    say("=== regressed on depth ===")
    blocked = refusal(fit["n"], yv)
    if blocked:
        say(f"  {blocked}")
        fit = {**fit, "refused": blocked}
    else:
        say(f"  n={fit['n']}  slope={fit['slope']:+.5f} per step (se {fit['slope_se']:.5f})  "
            f"intercept={fit['intercept']:+.5f}")
        if args.arm == "recursive":
            say(f"  a slope of {fit['slope']:+.5f} compounds to x{np.exp(fit['slope'] * 9):.4f} over nine steps")
            say(f"  the measured over-reserve on abc delta production is x1.67 (see results/runs/abc-reserve-delta.md)")
        else:
            step = float(np.mean(yv / x))
            say(f"  the same rows normalised per step: mean log error per step = {step:+.5f} "
                f"(x{np.exp(step):.4f} per step)")
            say(f"  this is the *direct* arm's regression, whose depth is the length of the whole prediction;")
            say(f"  the recursive table's depth is a single cell's distance beyond the training range. Two")
            say(f"  different regressions, so the recorded +0.0317 is reproduced separately, not compared here.")

    per_triangle = {}
    say("")
    say("=== the same slope, per triangle (delta arm) ===")
    for name in args.triangles:
        sub = [r for r in rows if r["triangle"] == name and r["target"] == "delta"]
        if len(sub) < 3:
            continue
        f = ols(np.array([r["depth"] for r in sub], float), np.array([r["log_error"] for r in sub], float))
        per_triangle[name] = f
        say(f"  {name:>8}  n={f['n']:<3} slope={f['slope']:+.5f} (se {f['slope_se']:.5f})  "
            f"mean log error={np.mean([r['log_error'] for r in sub]):+.4f}")

    if args.arm == "direct":
        # `decompose` replays one path by multiplying its per-step ratios. The direct arm predicts each run
        # whole, so there is no per-step factor to multiply: the currency column would be right and the
        # by-origin reconstruction would be fiction. Both sections are skipped rather than half-reported.
        decompositions: dict = {}
        say("")
        say("=== currency decomposition and by-origin residuals: recursive arm only ===")
        say("  the direct arm predicts each run whole, so the per-step replay those sections are built on does")
        say("  not exist for it. The recursive ones are in")
        say("  results/runs/20260918-023600_depth-bias/FINDINGS.md.")
    else:
        decompositions = {t: decompose(rows, t) for t in args.targets}
    for target, dec in decompositions.items():
        if not dec["by_depth"]:
            continue
        say("")
        say(f"=== currency decomposition, {target} arm (a reserve is a sum, so weight the steps) ===")
        say(f"{'depth':>5} {'steps':>6} {'exposure':>15} {'mean log err':>13} {'exposure-weighted log err':>27}")
        for d in sorted(dec["by_depth"], key=lambda x: int(x)):
            rec = dec["by_depth"][d]
            say(f"{d:>5} {rec['n']:>6} {rec['exposure']:>15,.0f} {rec['mean_log_error']:>+13.4f} "
                f"{rec['currency_bias_per_unit_exposure']:>+27.4f}")

    if decompositions.get("delta", {}).get("by_origin"):
        say("")
        say("=== the largest residuals by origin, delta arm (where the money actually is) ===")
        for r in sorted(decompositions["delta"]["by_origin"], key=lambda r: -abs(r["residual"]))[:8]:
            pct = (100 * r["residual"] / r["actual_increment"]) if r["actual_increment"] else float("nan")
            say(f"  {r['triangle']:>8} k={r['anchor']:<2} origin={r['origin']:<2} steps={r['steps']:<2} "
                f"base={r['base_observed']:>10,.0f} predicted={r['predicted_increment']:>12,.0f} "
                f"actual={r['actual_increment']:>12,.0f} residual={r['residual']:>+12,.0f} ({pct:+.1f}%)")

    (out_dir / "depth_bias.json").write_text(json.dumps(
        {"command": " ".join([sys.executable, "scripts/depth_bias.py"] + sys.argv[1:]),
         "arm": args.arm, "intermediate_ages": bool(args.intermediate_ages),
         "fingerprints": {n: Triangle.load(n, column=COLUMNS.get(n)).fingerprint()
                          for n in args.triangles},
         "rows": rows, "by_depth": {str(k): v for k, v in by_depth.items()},
         "fit": fit, "per_triangle_delta": per_triangle,
         "decompositions": decompositions}, indent=2, default=str))
    say("")
    say(f"wrote {out_dir}/depth_bias.json and run.log ({len(rows)} scored steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
