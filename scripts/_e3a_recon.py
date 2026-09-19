"""Scratch recon for the E3a pricing task -- not a deliverable."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

import chainladder as cl
from fleet_eval import load_fleet
from tabpfn_reserving.triangle import Triangle, target_ages  # noqa

t0 = time.time()
pairs = load_fleet(0)
print(f"load_fleet: {len(pairs)} triangles in {time.time() - t0:.2f}s")

i = 7
name, sub = pairs[i]
print("picked:", name)
tri = Triangle.load(sub, column="IncurLoss").trim()
print("trimmed tri:", tri.name, tri.values.shape, tri.column)
print("origins:", tri.origin, "ages:", tri.ages)

n = tri.n
k = n - 2
print(f"n={n} anchor={k}")
for kk in range(n - 2, n - 5, -1):
    print(f"  anchor {kk}: actual_future={tri.actual_future(kk)}")

mask = np.zeros((n, n), dtype=bool)
for a in range(n):
    for d in range(n):
        if a + d <= k:
            mask[a, d] = True

rows = []
for a in range(n):
    for d in range(n):
        if mask[a, d] and np.isfinite(tri.values[a, d]):
            rows.append({"origin": tri.origin[a], "development": tri.ages[d],
                         "value": float(tri.values[a, d])})
df = pd.DataFrame(rows)
print("long-form rows:", len(df), "observed cells:", int(mask.sum()))
clt = cl.Triangle(df, origin="origin", development="development", columns=["value"], cumulative=True)
print("clt shape:", clt.shape, "odims:", [str(o) for o in clt.odims], "ddims:", list(clt.ddims))
print("clt nan_triangle:\n", clt.nan_triangle.astype(int))

# --- the arithmetic check: chainladder's projection restricted to the observed square must equal
# --- this repo's own factor_reserve in backtest mode.
ga = tri.global_factors(tri.known(k))
cl_ibnr = np.asarray(cl.Chainladder().fit(clt).ibnr_.values)[0, 0]
print("chainladder ibnr per origin:\n", np.round(cl_ibnr, 4))
tgt = target_ages(n, "backtest")
print("backtest targets:", tgt)
# per-origin comparison on the origin's own last observed age
full0 = np.asarray(cl.Chainladder().fit(clt).full_triangle_.values)[0, 0]
repo_per_origin = []
cl_per_origin = []
for a in range(k + 1):
    latest = tri.values[a, k - a]
    repo = latest * np.prod([ga[d - 1] for d in range(k - a + 1, tgt[a] + 1)]) - latest
    repo_per_origin.append(repo)
    cl_per_origin.append(full0[a, tgt[a]] - latest)
print("repo per-origin backtest:", np.round(repo_per_origin, 3))
print("cl   per-origin backtest:", np.round(cl_per_origin, 3))
print("max abs diff:", np.max(np.abs(np.array(repo_per_origin) - np.array(cl_per_origin))))

t0 = time.time()
mack = cl.MackChainladder().fit(clt)
print(f"\nmack fit: {time.time() - t0:.4f}s")
print("ibnr_ total:", float(np.nansum(mack.ibnr_.values)))
print("total_mack_std_err_:", mack.total_mack_std_err_)
print("mack_std_err_ shape:", np.asarray(mack.mack_std_err_.values).shape,
      "ddims:", list(mack.mack_std_err_.ddims))
print("mack_std_err_:\n", np.round(np.asarray(mack.mack_std_err_.values)[0, 0], 2))
print("ldf_ shape:", np.asarray(mack.ldf_.values).shape, "ddims:", list(mack.ldf_.ddims))
print("ldf_ :", np.round(np.asarray(mack.ldf_.values)[0, 0, 0], 6))
print("cdf_ :", np.round(np.asarray(mack.cdf_.values)[0, 0, 0], 6), list(mack.cdf_.ddims))
print("sigma_ :", np.round(np.asarray(mack.sigma_.values)[0, 0, 0], 6), list(mack.sigma_.ddims))
print("std_err_ :", np.round(np.asarray(mack.std_err_.values)[0, 0], 6))
print("full_triangle_:\n", np.round(np.asarray(mack.full_triangle_.values)[0, 0], 1))
print("total_process_risk_ tail:", np.asarray(mack.total_process_risk_.values)[0, 0, 0, -3:])
print("total_parameter_risk_ tail:", np.asarray(mack.total_parameter_risk_.values)[0, 0, 0, -3:])
print("process_risk_:\n", np.round(np.asarray(mack.process_risk_.values)[0, 0], 2))

full = np.asarray(mack.full_triangle_.values)[0, 0, :, :n]
pred = 0.0
for a in range(k + 1):
    pred += full[a, tgt[a]] - tri.values[a, k - a]
print("restricted (backtest) Mack point:", pred, " actual:", tri.actual_future(k))

for n_sims in (20, 100, 300):
    t0 = time.time()
    bs = cl.BootstrapODPSample(n_sims=n_sims, random_state=42).fit_transform(clt)
    t1 = time.time()
    out = cl.Chainladder().fit(bs)
    t2 = time.time()
    ft = np.asarray(out.full_triangle_.values)
    print(f"n_sims={n_sims}: sample {t1 - t0:.3f}s  fit {t2 - t1:.3f}s "
          f"({(t2 - t1) / n_sims * 1000:.3f} ms/sim)  full shape {ft.shape}")
    tot = np.zeros(n_sims)
    for a in range(k + 1):
        tot += ft[:, 0, a, tgt[a]] - tri.values[a, k - a]
    print("   restricted reserves: mean %.2f  sd %.2f  p50 %.2f  q95 %.2f" %
          (tot.mean(), tot.std(), np.quantile(tot, 0.5), np.quantile(tot, 0.95)))
    print("   scale_:", float(bs.scale_))
