"""The figures, regenerable from committed commands.

Every figure in the README is produced here or by a CLI run, so no picture in this repository is a
hand-made illustration of something that might have changed since.

    python scripts/make_figures.py            # writes results/figures/reframing.png and .pdf
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tabpfn_reserving.triangle import (Triangle, direct_features, features_for, gf_used,
                                       chainladder_factor)

OUT = pathlib.Path("results/figures")


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


if __name__ == "__main__":
    p = reframing()
    print(f"wrote {p} and {p.with_suffix('.pdf')}")
