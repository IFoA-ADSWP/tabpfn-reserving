"""The figures, regenerable from committed commands.

Every figure in the README is produced here or by a CLI run, so no picture in this repository is a
hand-made illustration of something that might have changed since.

    python scripts/make_figures.py            # writes results/figures/reframing.png and .pdf
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tabpfn_reserving.triangle import (Triangle, direct_features, features_for, gf_used,
                                       chainladder_factor)

OUT = pathlib.Path("results/figures")

# The nominal levels the coverage run records, kept in step with scripts/fleet_coverage.py.
LEVELS = [0.50, 0.75, 0.90, 0.95]


def reframing(triangle: str = "abc") -> pathlib.Path:
    """Left: the domain object. Right: the same numbers as a supervised problem.

    This is the whole reframing on one page, which is why it is the first figure rather than a table of
    results: a triangle is not a sequence and not a regression on flat rows, and the picture is the argument.
    """
    tri = Triangle.load(triangle)
    n = tri.n
    anchor = n - 1
    gf = tri.global_factors(tri.known(anchor))
    observed = tri.known(anchor)

    fig, (ax_tri, ax_tbl) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                         gridspec_kw={"width_ratios": [1.05, 1.25]})

    # ---- left: the triangle ------------------------------------------------------------------
    shown = np.where(observed, np.log10(np.where(tri.values > 0, tri.values, np.nan)), np.nan)
    ax_tri.imshow(np.ma.masked_invalid(shown), cmap="Blues", aspect="auto")
    ax_tri.imshow(np.ma.masked_where(observed, np.ones_like(tri.values) * 0.0),
                  cmap="Greys", vmin=0, vmax=1.6, aspect="auto")
    for a in range(n):
        for d in range(n):
            v = tri.values[a, d]
            txt = f"{v / 1000:,.0f}k" if np.isfinite(v) and v >= 1000 else (
                f"{v:,.0f}" if np.isfinite(v) else "·")
            ax_tri.text(d, a, txt, ha="center", va="center", fontsize=5.6,
                        color="white" if observed[a, d] else "#555555")
    ax_tri.plot([n - 1], [0], "o", color="#C44E52", markersize=0)   # keep the axis honest
    ax_tri.set_xticks(range(0, n, 2))
    ax_tri.set_xticklabels([str(tri.ages[i]) for i in range(0, n, 2)], fontsize=7)
    ax_tri.set_yticks(range(0, n, 2))
    ax_tri.set_yticklabels([tri.origin[i][:7] for i in range(0, n, 2)], fontsize=7)
    ax_tri.set_xlabel("development age (months)", fontsize=8)
    ax_tri.set_ylabel("accident year", fontsize=8)
    ax_tri.set_title(f"a loss triangle, {triangle}: {int(observed.sum())} observed cells,\n"
                     f"and a lower-right half that is the reserve", fontsize=9)

    # the valuation-date diagonal, drawn along the boundary of the observed region
    xs, ys = [], []
    for a in range(n):
        last = anchor - a
        if 0 <= last < n:
            xs.append(last + 0.5)
            ys.append(a - 0.5)
    ax_tri.plot(xs, ys, color="#C44E52", linewidth=1.4, label="valuation date")
    ax_tri.legend(frameon=False, fontsize=7, loc="lower left")

    # ---- right: the same numbers as a supervised problem -------------------------------------
    # Two valuation dates, so the *horizon* column varies: at a single anchor every origin shares the same
    # horizon, which is the property that makes training across anchors necessary. The target is each
    # origin's realised factor from that anchor to its own last observed age -- observed, because these are
    # training rows. (A production run's target is unobserved by definition; that is why there is a model.)
    rows, labels = [], []
    for k in (n - 4, n - 3):
        gf_k = tri.global_factors(tri.known(k))
        for a in range(1, min(k, 6) + 1):
            age, target_age = k - a, n - 1 - a
            if target_age <= age:
                continue
            base, end = tri.values[a, age], tri.values[a, target_age]
            if not (np.isfinite(base) and np.isfinite(end) and base > 0):
                continue
            row = features_for(tri, a, age + 1, k, gf_k, tri.values)
            horizon = target_age - age
            rows.append([f"k={k}", f"{tri.origin[a][:7]}", f"{age}", f"{row[3]:,.0f}",
                         f"{row[5]:.4f}", f"{row[6]:.4f}", f"{horizon}",
                         f"{end / base:.4f}"])
            labels.append("")
        rows.append(["", "", "", "", "", "", "", ""])   # a blank line between the two valuation dates
        labels.append("")

    cols = ["valuation\ndate", "accident\nyear", "dev age\n(months)", "latest\ncumulative",
            "own last\nratio", "CL factor\n(prev age)", "horizon\n(steps)", "target\n(realised factor)"]
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.8)
    tbl.scale(1.0, 1.18)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor("#EEF3FA")
            cell.set_height(0.095)
        if c == len(cols) - 1 and r > 0:
            cell.set_facecolor("#FBEEEE")          # the column being predicted
        if c == len(cols) - 2 and r > 0:
            cell.set_facecolor("#F5F5F5")
    ax_tbl.set_title("the same numbers as a prediction task: features restricted to what that\n"
                     "valuation date knew, the realised factor as the target (red)",
                     fontsize=9)

    fig.suptitle("A triangle is not a sequence and not a table of independent claims — "
                 "it is a small, structured prediction problem", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"reframing.{ext}", dpi=150)
    plt.close(fig)
    return OUT / "reframing.png"


def coverage_curve(path: pathlib.Path | str = "results/fleet/coverage.jsonl") -> pathlib.Path:
    """Empirical coverage against nominal, with binomial bars, over the fleet.

    The distribution claim is the deliverable, so this is the figure that tests it: if the model's intervals
    are calibrated, the points sit on the diagonal. They do not, and the gap is drawn with its sampling error
    rather than asserted, because a calibration plot without error bars invites reading noise as structure.
    """
    import json

    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if all(f"covered_{int(lv * 100)}" in r for lv in LEVELS):
            rows.append(r)
    if not rows:
        raise SystemExit(f"no coverage rows in {path}")

    n = len(rows)
    nominal = np.array(LEVELS)
    empirical = np.array([np.mean([bool(r[f"covered_{int(lv * 100)}"]) for r in rows]) for lv in LEVELS])
    se = np.sqrt(empirical * (1 - empirical) / n)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1.0, linestyle="--",
            label="perfect calibration")
    ax.errorbar(nominal, empirical, yerr=1.96 * se, fmt="o", color="#C44E52", markersize=7,
                capsize=4, linewidth=1.4, label=f"TabPFN-3.5, {n} fleet evaluations")
    for lv, emp in zip(nominal, empirical):
        ax.annotate(f"{100 * (emp - lv):+.0f}pp", (lv, emp), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="#7a2f32")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Do the intervals contain the truth as often as they claim?\n"
                 f"{n} fleet evaluations, held-out diagonals, 95% binomial bars", fontsize=10)
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"coverage_curve.{ext}", dpi=150)
    plt.close(fig)
    return OUT / "coverage_curve.png"


def tail_asymmetry(path: pathlib.Path | str = "results/fleet/coverage.jsonl") -> pathlib.Path:
    """Where the truth actually lands relative to the model's own 95% bounds.

    The sharpest single number in this repository, and until now it lived only in prose: the intervals
    under-cover *one-sidedly*. Every figure on this plot is computed from the recorded units here -- nothing is
    typed in -- so the picture cannot drift away from `results/fleet/coverage.jsonl`.
    """
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "hi_95" in r and "median" in r]
    if not rows:
        raise SystemExit(f"no recorded units in {path}")
    n = len(rows)
    above = sum(1 for r in rows if r["actual"] > r["hi_95"])
    below = sum(1 for r in rows if r["actual"] < r["lo_95"])
    above_med = sum(1 for r in rows if r["actual"] > r["median"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    x = np.arange(2)
    w = 0.36
    ax1.bar(x - w / 2, [0.025, 0.025], w, label="claimed (nominal 2.5%)", color="#999999")
    ax1.bar(x + w / 2, [above / n, below / n], w, label="observed", color="#C44E52")
    for xi, (cnt, val) in zip(x, ((above, above / n), (below, below / n))):
        ax1.text(xi + w / 2, val + 0.006, f"{cnt} of {n}\n{100*val:.1f}%", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["truth ABOVE the\n95% upper bound", "truth BELOW the\n95% lower bound"], fontsize=9)
    ax1.set_ylabel("share of units")
    ax1.set_ylim(0, max(above, below) / n * 1.35)
    ax1.set_title("The miss is one-sided", fontsize=10)
    ax1.grid(alpha=0.25, linewidth=0.5, axis="y")
    ax1.legend(frameon=False, fontsize=8)

    ax2.bar([0, 1], [0.5, above_med / n], 0.5, color=["#999999", "#C44E52"])
    ax2.text(0, 0.51, "claimed 50%", ha="center", fontsize=9)
    ax2.text(1, above_med / n + 0.01, f"{above_med} of {n}\n{100*above_med/n:.1f}%", ha="center", fontsize=9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["nominal", "observed"], fontsize=9)
    ax2.set_ylim(0, 0.95)
    ax2.set_title("…and the centre sits low too", fontsize=10)
    ax2.grid(alpha=0.25, linewidth=0.5, axis="y")

    fig.suptitle(f"Where the truth lands inside the model's own predictive distribution  (n = {n} fleet units)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tail_asymmetry.{ext}", dpi=150)
    plt.close(fig)
    print(f"  tail asymmetry: above 95% bound {100*above/n:.1f}%, below {100*below/n:.1f}%, "
          f"above median {100*above_med/n:.1f}%  (n={n})")
    return OUT / "tail_asymmetry.png"


if __name__ == "__main__":
    p = reframing()
    print(f"wrote {p} and {p.with_suffix('.pdf')}")
    try:
        c = coverage_curve()
        print(f"wrote {c} and {c.with_suffix('.pdf')}")
    except SystemExit as exc:
        print(f"coverage curve skipped: {exc}")
    try:
        t = tail_asymmetry()
        print(f"wrote {t} and {t.with_suffix('.pdf')}")
    except SystemExit as exc:
        print(f"tail asymmetry skipped: {exc}")
