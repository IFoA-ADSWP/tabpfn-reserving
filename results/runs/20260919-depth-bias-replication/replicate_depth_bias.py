"""Reproduce the recorded depth-bias run exactly, in this worktree, on this session's interpreter.

The recorded run is `results/runs/20260918-023600_depth-bias/` with 287 scored steps, anchors [3..9] for abc
and [3..8] for genins, slope +0.0317 (se 0.0036), intercept +0.0001, per-triangle abc +0.0190 and genins
+0.0617. Its own log does not record which flags produced that anchor policy, so the policy is recovered
from the log's first lines ("the last 3 anchors below the ultimate") and from the anchors it printed.

This is the mandatory replication cell for the depth-bias measurement: if the numbers do not come back, the
harness changed and the re-measured slope is not comparable to the recorded one.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from tabpfn_reserving import arm  # noqa: E402
from tabpfn_reserving.triangle import Triangle, training_rows  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path(".")


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "slope_se": float("nan")}
    xbar, ybar = x.mean(), y.mean()
    sxx = ((x - xbar) ** 2).sum()
    slope = ((x - xbar) * (y - ybar)).sum() / sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx))
    return {"n": n, "slope": float(slope), "intercept": float(intercept), "slope_se": se}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors-from", type=int, default=3)
    ap.add_argument("--triangles", nargs="*", default=["abc", "genins"])
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/runs/20260919-depth-bias-replication"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import tabpfn_reserving
    from tabpfn_reserving.triangle import training_rows as tr_check  # noqa: F401
    print("package:", tabpfn_reserving.__file__, flush=True)

    rows: list[dict] = []
    t0 = time.time()
    log_lines: list[str] = []

    def say(line: str) -> None:
        print(line, flush=True)
        log_lines.append(line)

    say("command: " + " ".join([sys.executable, "scripts/replicate_depth_bias.py", *sys.argv[1:]]))
    say("anchor policy: every anchor from --anchors-from up to the last but one (the recorded run's window)")
    say("")
    for name in args.triangles:
        tri = Triangle.load(name, column=None)
        anchors = list(range(args.anchors_from, tri.n - 1))
        say(f"{name}: anchors {anchors} -- deepest step reachable is depth {tri.n - 2 - anchors[0]}")
        for k in anchors:
            gf = tri.global_factors(tri.known(k))
            X, y = training_rows(tri, k, gf, target="delta")
            model = arm.make_model("local")
            model.fit(X, y)
            trace: list = []
            arm.reserve(model, tri, k, 0, np.random.default_rng(0), target="delta", trace=trace)
            scored = [t for t in trace if np.isfinite(t["log_error"])]
            for t in scored:
                rows.append({"triangle": name, "target": "delta", "anchor": int(k), **t})
            if scored:
                errs = np.array([t["log_error"] for t in scored])
                say(f"{name:>8} delta k={k:<2} steps scored={len(scored):<3} "
                    f"mean log error={errs.mean():+.4f}  (x{np.exp(errs.mean()):.4f} per step)")

    say("")
    say("=== mean log error by depth (all triangles and anchors pooled) ===")
    say(f"{'depth':>5} {'steps':>6} {'mean log err':>13} {'x error':>9} {'se':>8}")
    depths = sorted({r["depth"] for r in rows})
    for d in depths:
        e = np.array([r["log_error"] for r in rows if r["depth"] == d])
        se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float("nan")
        say(f"{d:>5} {len(e):>6} {e.mean():>+13.4f} {np.exp(e.mean()):>9.4f} {se:>8.4f}")

    x = np.array([r["depth"] for r in rows], dtype=float)
    yv = np.array([r["log_error"] for r in rows], dtype=float)
    fit = ols(x, yv)
    say("")
    say("=== regressed on depth ===")
    say(f"  n={fit['n']}  slope={fit['slope']:+.5f} per step (se {fit['slope_se']:.5f})  "
        f"intercept={fit['intercept']:+.5f}")
    per_triangle = {}
    say("")
    say("=== the same slope, per triangle (delta arm) ===")
    for name in args.triangles:
        sub = [r for r in rows if r["triangle"] == name]
        if len(sub) < 3:
            continue
        f = ols(np.array([r["depth"] for r in sub], float), np.array([r["log_error"] for r in sub], float))
        per_triangle[name] = f
        say(f"  {name:>8}  n={f['n']:<3} slope={f['slope']:+.5f} (se {f['slope_se']:.5f})  "
            f"mean log error={np.mean([r['log_error'] for r in sub]):+.4f}")

    say("")
    recorded = {"n": 287, "slope": 0.03168, "slope_se": 0.00363, "intercept": 0.00011}
    say("=== against the recorded run (results/runs/20260918-023600_depth-bias) ===")
    say(f"  recorded: n={recorded['n']} slope={recorded['slope']:+.5f} (se {recorded['slope_se']:.5f}) "
        f"intercept={recorded['intercept']:+.5f}")
    say(f"  this run: n={fit['n']} slope={fit['slope']:+.5f} (se {fit['slope_se']:.5f}) "
        f"intercept={fit['intercept']:+.5f}")
    ok = fit["n"] == recorded["n"] and abs(fit["slope"] - recorded["slope"]) < 1e-4
    say(f"  REPLICATION {'OK -- same harness' if ok else 'FAILED -- the harness changed, do not compare'}")

    (args.out / "run.log").write_text("\n".join(log_lines) + "\n")
    (args.out / "depth_bias.json").write_text(json.dumps(
        {"command": " ".join([sys.executable, "scripts/replicate_depth_bias.py", *sys.argv[1:]]),
         "arm": "recursive", "rows": rows, "fit": fit, "per_triangle_delta": per_triangle,
         "recorded": recorded, "replication_ok": bool(ok)}, indent=2, default=str))
    say(f"wrote {args.out}/depth_bias.json and run.log ({len(rows)} scored steps)")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
