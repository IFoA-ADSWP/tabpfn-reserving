"""Scratch recon #8: replicate chainladder's per-origin Mack risk in numpy (uncapped), then it can be capped."""
from __future__ import annotations

import numpy as np

import chainladder as cl


def replicate_per_origin(m):
    X = m.X_
    full = np.asarray(m.full_triangle_.values)[0, 0][:, : len(X.ddims)]   # (I, J)
    nt = np.asarray(X.nan_triangle).astype(float)                        # (I, J) 1 = observed
    I, J = full.shape
    ldf_all = np.asarray(m.ldf_.values)[0, 0, 0]
    cdf = np.asarray(m.cdf_.values)[0, 0, 0]
    extend = ldf_all.shape[-1] - J + 1
    f = np.concatenate((ldf_all[: J - 1], cdf[-extend: -extend + 1]))
    sigma = np.asarray(X.sigma_.values)[0, 0, 0][:J]
    se_par = np.asarray(X.std_err_.values)[0, 0]                         # (1, J) or (I, J)
    w = np.asarray(X.w_)[0, 0]
    w = np.concatenate((w, np.ones((I, 1))), axis=1)
    w[np.isnan(w)] = 1.0
    avg = {"regression": 0, "volume": 1, "simple": 2}
    alpha = np.array([avg.get(a, a) for a in X.average_])
    alpha = np.concatenate((alpha, [alpha[-1]]))
    se_proc = (sigma[None, :] / np.where(full > 0, full, np.nan) ** ((2 - alpha) / 2)) * w
    se_proc = np.nan_to_num(se_proc)

    nans = np.concatenate((nt, np.zeros((I, 1))), axis=1)
    nans = 1 - nans

    def rec(t1):
        risk = np.zeros((I, 1))
        for j in range(t1.shape[1]):
            t = np.sqrt(np.nan_to_num(t1[:, j: j + 1]) ** 2 + (f[j] * risk[:, -1:]) ** 2)
            t = t * nans[:, j + 1: j + 2]
            risk = np.concatenate((risk, np.nan_to_num(t)), axis=1)
        return risk

    par = rec(np.nan_to_num(full) * se_par)
    proc = rec(np.nan_to_num(full) * se_proc)
    return par, proc, f, sigma


for name in ("ukmotor", "raa", "abc", "genins"):
    tr = cl.load_sample(name)
    m = cl.MackChainladder().fit(tr)
    par, proc, f, sigma = replicate_per_origin(m)
    mine = np.sqrt(par[:, -1] ** 2 + proc[:, -1] ** 2)
    theirs = np.asarray(m.mack_std_err_.values)[0, 0, :, -1]
    print(f"=== {name}")
    print("  mine  ", np.round(mine, 4))
    print("  theirs", np.round(theirs, 4))
    print("  max rel err:", np.nanmax(np.abs((mine - theirs) / np.where(theirs == 0, np.nan, theirs))))
    print("  process risk mine", np.round(proc[:, -1], 4), " theirs",
          np.round(np.asarray(m.process_risk_.values)[0, 0, :, -1], 4))
    print("  parameter risk mine", np.round(par[:, -1], 4), " theirs",
          np.round(np.asarray(m.parameter_risk_.values)[0, 0, :, -1], 4))
