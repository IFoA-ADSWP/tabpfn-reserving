"""Command line: one triangle in, a reserve and its distribution out.

    python -m tabpfn_reserving abc                      # reserve as of the latest diagonal
    python -m tabpfn_reserving abc --distribution       # with percentiles and a figure
    python -m tabpfn_reserving abc --backtest           # scored against what actually happened

Nothing here tunes anything. That is the claim being tested, not a convenience.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import warnings

import numpy as np

from .arm import QUANTILE_LEVELS, make_model, reserve
from .triangle import Triangle, chainladder_baseline, factor_reserve, training_rows

warnings.filterwarnings("ignore")


def load_token() -> str | None:
    """TABPFN_TOKEN from the key file, so a run never depends on a stale shell export."""
    tok = os.environ.get("TABPFN_TOKEN")
    key = pathlib.Path.home() / ".config" / "tfm" / "keys.env"
    if key.is_file():
        for line in key.read_text().splitlines():
            m = re.match(r'\s*(?:export\s+)?TABPFN_TOKEN\s*=\s*(.+)', line)
            if m:
                tok = m.group(1).strip().strip('"').strip("'")
    return tok


def percentile_table(samples: np.ndarray) -> dict:
    return {f"p{int(q * 100)}": float(np.quantile(samples, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}


def figure(samples: np.ndarray, point: float, path: pathlib.Path, title: str) -> None:
    """The picture the claim rests on: the reserve as a distribution, not a number.

    Log x-axis when the draws are positive, which they are for a reserve: the distribution is strongly
    right-skewed and the point estimate is many times the median, so a linear axis crushes every draw but
    the handful in the tail into one bar and hides the shape being claimed. The axis is labelled, because a
    log axis that does not say so is a different kind of misrepresentation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = samples[np.isfinite(samples)]
    positive = finite[finite > 0]
    use_log = len(positive) == len(finite) and len(positive) > 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if use_log:
        bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 45)
        ax.set_xscale("log")
        ax.set_xlabel("reserve (log scale)")
    else:
        bins = 50
        ax.set_xlabel("reserve")
    ax.hist(finite, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.4)
    for q, style in ((0.5, "-"), (0.9, "--"), (0.99, ":")):
        v = float(np.quantile(finite, q))
        ax.axvline(v, color="#333333", linestyle=style, linewidth=1.2,
                   label=f"p{int(q * 100)} = {v:,.0f}")
    ax.axvline(point, color="#C44E52", linewidth=1.6, label=f"point reserve = {point:,.0f}")
    ax.set_ylabel("draws")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_reserve(args, tri: Triangle) -> dict:
    """The production case: everything observed, project to the longest development age seen."""
    anchor = tri.n - 1
    X, y = training_rows(tri, anchor, tri.global_factors(tri.known(anchor)), target=args.target,
                         mode=args.features)
    print(f"{tri.name}: {tri.n}x{tri.n}  anchor = last diagonal (age {tri.ages[-1]})  "
          f"{len(y)} observed transitions")
    print(f"  projecting every origin to age {tri.ages[-1]} -- the longest development in the\n  triangle. The tail beyond it needs a tail factor and is out of scope.")
    model = make_model(args.backend)
    t0 = time.time()
    model.fit(X, y)
    fit_s = time.time() - t0
    t0 = time.time()
    out = reserve(model, tri, anchor, args.draws, np.random.default_rng(args.seed),
                  target=args.target, targets=[tri.n - 1] * tri.n, mode=args.features)
    pred_s = time.time() - t0

    cl = chainladder_baseline(tri, anchor)
    factor = factor_reserve(tri, anchor, tri.global_factors(tri.known(anchor)),
                            targets=[tri.n - 1] * tri.n)
    print(f"  reserve (compounded point)  {out['reserve']:>13,.0f}   fit {fit_s:.1f}s  predict {pred_s:.1f}s")
    print(f"  reserve (chain-ladder)      {factor:>13,.0f}   like-for-like, on the same cells")
    if "chainladder_ibnr" in cl:
        print(f"  reserve (CAS package)       {cl['chainladder_ibnr']:>13,.0f}   reference: includes a tail "
              f"nobody can score")
    if "mack_total_mack_std_err" in cl:
        print(f"  Mack standard error         {cl['mack_total_mack_std_err']:>13,.0f}")
    result = {"mode": "reserve", "triangle": tri.name, "fingerprint": tri.fingerprint(),
              "anchor": anchor, "target": args.target, "features": args.features,
              "reserve": out["reserve"], "reserve_mean": out["reserve_mean"],
              "reserve_median": out["reserve_median"], "draws": args.draws,
              "factor_reserve": factor, "chainladder": cl,
              "distribution_route": out["distribution_route"],
              "draws_method": out["draws_method"],
              "fit_seconds": fit_s, "predict_seconds": pred_s}
    if out["samples"] is not None:
        # Say which summary is which. The three differ by the compounding, not by rounding.
        print(f"  reserve (sampled mean)      {out['reserve_mean']:>13,.0f}   "
              f"{100 * (out['reserve_mean'] - out['reserve']) / out['reserve']:+.1f}% vs the point")
        print(f"  reserve (sampled median)    {out['reserve_median']:>13,.0f}   "
              f"{100 * (out['reserve_median'] - out['reserve']) / out['reserve']:+.1f}% vs the point")
        result["percentiles"] = percentile_table(out["samples"])
        print("  distribution             " + "  ".join(f"{k}={v:,.0f}"
                                                        for k, v in result["percentiles"].items()))
        print(f"  distribution route          {out['distribution_route']} quantiles, "
              f"{out['draws_method']} draws, {args.draws} of them")
        print(f"  note: {args.draws} draws means p99 is carried by roughly the top "
              f"{max(1, args.draws // 100)} draws, and the median moves about 1.6% between seeds "
              f"(measured, results/runs/abc-reserve-delta-running.md)")
    return result, out["samples"]


def run_backtest(args, tri: Triangle) -> dict:
    """Scored: every anchor below the last, against what actually developed afterwards."""
    rng = np.random.default_rng(args.seed)
    rows = []
    anchors = list(range(tri.n // 2, tri.n - 2))
    if args.anchors:
        anchors = anchors[: args.anchors]
    print(f"{tri.name}: backtest over anchors {anchors}  (target={args.target}, draws={args.draws})")
    for k in anchors:
        known = tri.known(k)
        gf = tri.global_factors(known)
        actual = tri.actual_future(k)
        X, y = training_rows(tri, k, gf, target=args.target, mode=args.features)
        model = make_model(args.backend)
        model.fit(X, y)
        out = reserve(model, tri, k, args.draws, rng, target=args.target,
                      targets=[tri.n - 1 - a for a in range(tri.n)], mode=args.features)
        factor = factor_reserve(tri, k, gf, targets=[tri.n - 1 - a for a in range(tri.n)])
        row = {"anchor": int(k), "n_train_rows": int(len(y)), "actual": actual,
               "tabpfn": out["reserve"], "factor": factor,
               "err_pct_tabpfn": 100 * (out["reserve"] - actual) / actual if actual else None,
               "err_pct_factor": 100 * (factor - actual) / actual if actual else None,
               "distribution_route": out["distribution_route"]}
        if out["samples"] is not None:
            for q in (0.5, 0.75, 0.9, 0.95):
                lo, hi = np.quantile(out["samples"], [(1 - q) / 2, 1 - (1 - q) / 2])
                row[f"covered_{q}"] = bool(lo <= actual <= hi)
        rows.append(row)
        print(f"  k={k:<3} actual={actual:>13,.0f}  tabpfn={out['reserve']:>13,.0f} "
              f"({row['err_pct_tabpfn']:+7.1f}%)  chain-ladder={factor:>13,.0f} "
              f"({row['err_pct_factor']:+7.1f}%)")
    return {"mode": "backtest", "triangle": tri.name, "fingerprint": tri.fingerprint(),
            "target": args.target, "rows": rows}, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tabpfn_reserving", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("triangle", help="a bundled chainladder sample, e.g. abc, genins, mcl, clrd")
    ap.add_argument("--features", choices=["frozen", "running"], default="frozen",
                    help="frozen: the cell is described by the origin's position at the anchor, the same "
                         "for every step of the recursion. running: also describe where the projection has "
                         "already pushed that origin (issue #14)")
    ap.add_argument("--target", choices=["ratio", "delta"], default="ratio",
                    help="ratio: learn raw link ratios. delta: learn the ratio relative to the "
                         "volume-weighted factor (measured: reproduces Chain Ladder)")
    ap.add_argument("--draws", type=int, default=300, help="draws per cell for the reserve distribution")
    ap.add_argument("--backtest", action="store_true", help="score every anchor below the last")
    ap.add_argument("--anchors", type=int, default=0, help="cap the anchors in backtest mode")
    ap.add_argument("--distribution", action="store_true", help="print the reserve percentiles")
    ap.add_argument("--figure", type=pathlib.Path, help="write the reserve distribution to a PNG")
    ap.add_argument("--json", type=pathlib.Path, dest="json_out", help="write the run record here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["local", "client"], default="local")
    args = ap.parse_args(argv)

    tok = load_token()
    if tok:
        os.environ["TABPFN_TOKEN"] = tok
    tri = Triangle.load(args.triangle)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    result, samples = (run_backtest(args, tri) if args.backtest else run_reserve(args, tri))
    result |= {"command": " ".join(["tabpfn_reserving"] + (argv or sys.argv[1:])),
               "started_at": started, "seed": args.seed, "draws": args.draws,
               "features": args.features,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.figure and samples is not None:
        title = (f"{tri.name}: reserve distribution, TabPFN-3.5 ({args.target} arm, "
                 f"{args.draws} draws)")
        figure(samples, result["reserve"], args.figure, title)
        print(f"  wrote {args.figure}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # Keep the draws themselves, not just their percentiles: the joint distribution is the artefact
        # this project is about, and a figure that can only be redrawn by re-fitting the model is a figure
        # nobody will check.
        if samples is not None:
            samples_path = args.json_out.with_name(args.json_out.stem + "-samples.npy")
            np.save(samples_path, samples)
            result["samples_file"] = samples_path.name
        args.json_out.write_text(json.dumps(result, indent=2, default=str))
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
