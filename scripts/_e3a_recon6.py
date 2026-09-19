"""Scratch recon #6: tail-extended Mack on an as-of triangle -- do the target columns exist?"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

import chainladder as cl
from fleet_eval import load_fleet
from tabpfn_reserving.triangle import Triangle, factor_reserve, target_ages


def asof(tri, k):
    n = tri.n
    rows = []
    for a in range(n):
        for d in range(n):
            if a + d <= k and np.isfinite(tri.values[a, d]):
                rows.append({"origin": tri.origin[a], "development": tri.ages[d],
                             "value": float(tri.values[a, d])})
    df = pd.DataFrame(rows)
    return cl.Triangle(df, origin="origin", development="development", columns=["value"],
                       cumulative=True)


pairs = load_fleet(0)
tri = Triangle.load(pairs[7][1], column="IncurLoss").trim()
n = tri.n
print("n:", n, tri.fingerprint())

for k in (n - 2, n - 4):
    h = n - 1 - k
    clt = asof(tri, k)
    base = cl.Development().fit_transform(clt)
    f = np.asarray(base.ldf_.values)[0, 0, 0]
    t_f = float(f[-1])
    print(f"\n=== k={k} h={h}: base ddims={list(clt.ddims)} ldf={np.round(f, 6)} tail={t_f} ===")
    ext = base
    for step in range(h):
        try:
            ext = cl.TailConstant(tail=t_f).fit_transform(ext)
        except Exception as exc:  # noqa: BLE001
            print("   TailConstant failed:", type(exc).__name__, exc)
            break
        ext.w_ = base.w_
        ext.sigma_ = base.sigma_
        ext.average_ = base.average_
        print(f"   after {step + 1} tails: ddims={list(ext.ddims)} "
              f"cdf={np.round(np.asarray(ext.cdf_.values)[0, 0, 0], 4)}")
    try:
        m = cl.MackChainladder().fit(ext)
    except Exception as exc:  # noqa: BLE001
        print("   Mack fit failed:", type(exc).__name__, exc)
        continue
    ft = np.asarray(m.full_triangle_.values)[0, 0]
    se = np.asarray(m.mack_std_err_.values)[0, 0]
    print("   full_triangle_", ft.shape, "mack_std_err_", se.shape,
          "ddims", list(m.mack_std_err_.ddims))
    print("   full_triangle_:\n", np.round(ft, 3))
    print("   mack_std_err_:\n", np.round(se, 3))
    print("   total_mack_std_err_:", float(m.total_mack_std_err_.values[0, 0]))
    ga = tri.global_factors(tri.known(k))
    repo = factor_reserve(tri, k, ga, targets=target_ages(n, "backtest"))
    pt = 0.0
    for a in range(k + 1):
        col = n - 1 - a
        if col < ft.shape[1]:
            pt += ft[a, col] - tri.values[a, k - a]
            print(f"   origin {a}: target col {col} (age idx {col}) full={ft[a, col]:.4f} "
                  f"latest={tri.values[a, k - a]:.4f} se={se[a, col]:.4f}")
        else:
            print(f"   origin {a}: target col {col} MISSING (array has {ft.shape[1]})")
    print("   restricted point (chainladder):", pt, " repo:", repo,
          " diff:", pt - repo, " actual:", tri.actual_future(k))
