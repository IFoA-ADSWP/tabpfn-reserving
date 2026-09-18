"""Is the compounding per-step bias, and does that bias grow with depth? (#17, hypothesis 1)

In both arms the over-reserve is multiplicative: the point estimate compounds a per-step ratio once per
development step, up to nine times for the youngest origin. So a per-step bias of a few percent becomes a
large error by the ultimate, and the only question worth asking is whether such a bias is there.

Every training row sits at depth <= 0 -- the cell's level is observed -- while production asks for steps 1
through 9. So this measures, for every step of every projection:

    log(predicted ratio / actual ratio), against depth

If the mean log error is flat in depth, the compounding is not per-step bias and hypothesis 1 is dead. If it
rises, the slope * depth is the whole story, and the slope is the per-step bias to attack.

Run:  python scripts/depth_bias.py [--triangles abc genins mcl ukmotor] [--targets delta ratio]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import Triangle, training_rows


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
    args = ap.parse_args()
    # mcl carries incurred and paid; the column changes the numbers, so it is named here (the spike script
    # silently took the first, which is incurred -- so these are comparable with the spike's mcl figures).
    COLUMNS = {"mcl": "incurred"}

    run_id = time.strftime("%Y%m%d-%H%M%S") + "_depth-bias"
    out_dir = pathlib.Path("results/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    rows: list[dict] = []

    def say(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    say(f"command: depth_bias.py --triangles {' '.join(args.triangles)} --targets {' '.join(args.targets)}")
    say(f"anchor policy: the last {args.anchors_behind} anchors below the ultimate, per triangle")
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
                X, y = training_rows(tri, k, gf, target=target)
                model = arm.make_model("local")
                model.fit(X, y)
                trace: list = []
                out = arm.reserve(model, tri, k, 0, np.random.default_rng(args.seed),
                                  target=target, trace=trace)
                scored = [t for t in trace if np.isfinite(t["log_error"])]
                for t in scored:
                    rows.append({"triangle": name, "target": target, "anchor": int(k), **t})
                if scored:
                    errs = np.array([t["log_error"] for t in scored])
                    say(f"{name:>8} {target:>5} k={k:<2} steps scored={len(scored):<3} "
                        f"mean log error={errs.mean():+.4f}  (x{np.exp(errs.mean()):.4f} per step)")

    if not rows:
        say("no scored steps -- nothing to conclude")
        return 1

    say("")
    say("=== mean log error by depth (all triangles and anchors pooled) ===")
    say(f"{'depth':>5} {'steps':>6} {'mean log err':>13} {'x error':>9} {'se':>8}")
    depths = sorted({r["depth"] for r in rows})
    by_depth = {}
    for d in depths:
        e = np.array([r["log_error"] for r in rows if r["depth"] == d])
        by_depth[d] = {"n": int(len(e)), "mean": float(e.mean()), "se": float(e.std(ddof=1) / np.sqrt(len(e)))
                       if len(e) > 1 else float("nan")}
        say(f"{d:>5} {len(e):>6} {e.mean():>+13.4f} {np.exp(e.mean()):>9.4f} "
            f"{by_depth[d]['se']:>8.4f}")

    x = np.array([r["depth"] for r in rows], dtype=float)
    yv = np.array([r["log_error"] for r in rows], dtype=float)
    fit = ols(x, yv)
    say("")
    say("=== regressed on depth ===")
    say(f"  n={fit['n']}  slope={fit['slope']:+.5f} per step (se {fit['slope_se']:.5f})  "
        f"intercept={fit['intercept']:+.5f}")
    if np.isfinite(fit["slope"]):
        say(f"  a slope of {fit['slope']:+.5f} compounds to x{np.exp(fit['slope'] * 9):.4f} over nine steps")
        say(f"  the measured over-reserve on abc delta production is x1.67 (see results/runs/abc-reserve-delta.md)")

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
        {"command": " ".join(["python", "scripts/depth_bias.py"] + args.triangles),
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
