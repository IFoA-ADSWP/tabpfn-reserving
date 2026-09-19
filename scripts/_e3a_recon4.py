"""Scratch recon #4: tail-extended Mack on an as-of triangle, and the target columns."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

import chainladder as cl
from fleet_eval import load_fleet
from tabpfn_reserving.triangle import Triangle, factor_reserve, target_ages


def asof(name, sub, column, k):
    tri = Triangle.load(sub, column=column).trim()
    n = tri.n
    rows = []
    for a in range(n):
        for d in range(n):
            if a + d <= k and np.isfinite(tri.values[a, d]):
                rows.append({"origin": tri.origin[a], "development": tri.ages[d],
                             "value": float(tri.values[a, d])})
    df = pd.DataFrame(rows)
    clt = cl.Triangle(df, origin="origin", development="development", columns=["value"],
                      cumulative=True)
    return tri, clt


def main() -> None:
    pairs = load_fleet(0)
    i = 7
    name, sub = pairs[i]
    tri = Triangle.load(sub, column="IncurLoss").trim()
    n = tri.n
    print("triangle:", name, "n:", n, "fingerprint:", tri.fingerprint())
    for k in (n - 2, n - 4):
        h = n - 1 - k
        tri_k, clt = asof(name, sub, "IncurLoss", k)
        print(f"\n=== anchor {k} horizon {h}  clt={clt.shape} ddims={list(clt.ddims)} ===")
        ga = tri_k.global_factors(tri_k.known(k))
        print("  repo gf (estimable 0..k-1):", np.round(ga[: k + 1], 6))
        dev = cl.Development().fit_transform(clt)
        ldf = np.asarray(dev.ldf_.values)[0, 0, 0]
        print("  cl   ldf:", np.round(ldf, 6), "ddims", list(dev.ldf_.ddims))
        # carry the last estimable factor forward to build the tail chain
        tail_f = float(ldf[-1])
        print("  tail factor (carried forward):", tail_f)
        ext = clt
        for _ in range(h):
            ext = cl.TailConstant(tail=tail_f).fit_transform(ext)
        print("  extended ddims:", list(ext.ddims), "shape", ext.shape)
        m = cl.MackChainladder().fit(ext)
        print("  mack_std_err_ ddims:", list(m.mack_std_err_.ddims))
        ft = np.asarray(m.full_triangle_.values)[0, 0]
        se = np.asarray(m.mack_std_err_.values)[0, 0]
        print("  full_triangle_ shape", ft.shape)
        print("  std err per origin / dev col:")
        for a in range(k + 1):
            print(f"    origin {a}: target idx {n-1-a} (=age {tri.ages[n-1-a]}) "
                  f"full={ft[a, n-1-a]:.4f} latest={tri.values[a, k-a]:.4f} "
                  f"se={se[a, n-1-a]:.4f}  | full-to-ultimate se={se[a, -1]:.4f}")
        # point estimate from the restricted read
        pt = sum(float(ft[a, n - 1 - a]) - tri.values[a, k - a] for a in range(k + 1))
        repo = factor_reserve(tri_k, k, ga, targets=target_ages(n, "backtest"))
        print("  restricted point from chainladder full_triangle_:", pt)
        print("  repo factor_reserve(backtest):                    ", repo)
        print("  actual_future(k):", tri.actual_future(k))
        print("  cl ibnr total (to ultimate):", float(np.nansum(m.ibnr_.values)),
              " total_mack_std_err_:", float(m.total_mack_std_err_.values[0, 0]))
        print("  quadrature of restricted per-origin se:",
              float(np.sqrt(sum(float(se[a, n - 1 - a]) ** 2 for a in range(k + 1)))))


main()
