"""Scratch recon #5: replicate chainladder's Mack recursion exactly, in numpy, so it can be truncated."""
from __future__ import annotations

import numpy as np

import chainladder as cl


def replicate(m, targets=None):
    """chainladder/methods/mack.py _mack_recursion, in numpy.

    targets: optional per-origin development *index* (into the ddims axis) to stop each origin's
    reserve at.  None means "every origin to the last development age" (the library's own behaviour).
    """
    X = m.X_
    full = np.asarray(m.full_triangle_.values)[0, 0]            # (I, J) projected, observed kept
    vals = np.asarray(X.values)[0, 0]                            # (I, J)
    nt = np.asarray(X.nan_triangle)                              # (I, J) 1 where observed
    I, J = vals.shape
    f = np.asarray(m.ldf_.values)[0, 0, 0][:J]                   # factor age j -> j+1
    sigma = np.asarray(X.sigma_.values)[0, 0, 0][:J]             # sigma^2 by age
    w = np.asarray(getattr(X.w_, "values", X.w_))               # weights
    w = w[0, 0] if w.ndim == 4 else w
    se = np.asarray(X.std_err_.values)[0, 0]                     # sigma / sqrt(C^(2-a)) * w
    print("    replicate: shapes w", w.shape, "se", se.shape, "sigma", sigma.shape, "full", full.shape)
    latest = np.array([vals[i, int(np.flatnonzero(nt[i]).max())] for i in range(I)])
    F = np.where(nt > 0, 0.0, full - latest[:, None])            # accumulated future by age index
    print("    replicate: F row0", np.round(F[0], 3), "f", np.round(f, 5))

    # --- the aggregate recursion (their total_parameter_risk_ + total_process_risk_)
    def rec(t1_row, nans=None):
        risk = np.zeros((t1_row.shape[0], 1))
        for j in range(t1_row.shape[1]):
            t = np.sqrt(t1_row[:, j: j + 1] ** 2 + (f[j] * risk[:, -1:]) ** 2)
            if nans is not None:
                t = t * nans[:, j + 1: j + 2]
            risk = np.concatenate((risk, t), axis=1)
        return risk

    par_t1 = (F * se).sum(0, keepdims=True)
    proc_t1 = (F * (sigma[None, :] / np.sqrt(np.where(full > 0, full, np.nan)) * w)).reshape(1, -1)
    par = rec(par_t1)
    proc_agg = np.sqrt((rec(proc_t1) ** 2)).ravel()
    per_origin_nans = 1 - np.concatenate((nt, np.zeros((I, 1))), axis=1)
    proc_per = rec(np.where(nt > 0, 0.0, full * se), nans=per_origin_nans)
    par_per = rec(np.where(nt > 0, 0.0, full * se), nans=per_origin_nans)
    return dict(f=f, sigma=sigma, se=se, F=F, par=par.ravel(), proc_agg=proc_agg,
                proc_per=proc_per, par_per=par_per, full=full, nt=nt, latest=latest)


for name in ("ukmotor", "raa", "abc", "genins"):
    tr = cl.load_sample(name)
    m = cl.MackChainladder().fit(tr)
    r = replicate(m)
    tot = float(np.sqrt(r["par"][-1] ** 2 + r["proc_agg"][-1] ** 2))
    print(f"=== {name}: reported {float(m.total_mack_std_err_.values[0, 0]):.4f} "
          f"mine(param+proc) {tot:.4f}  par {r['par'][-1]:.3f} (cl "
          f"{float(np.asarray(m.total_parameter_risk_.values)[0, 0, 0, -1]):.3f})  "
          f"proc {r['proc_agg'][-1]:.3f} (cl "
          f"{float(np.asarray(m.total_process_risk_.values)[0, 0, 0, -1]):.3f})")
    mine_pp = np.sqrt(r["par_per"][:, -1] ** 2 + r["proc_per"][:, -1] ** 2)
    cl_pp = np.asarray(m.mack_std_err_.values)[0, 0, :, -1]
    print(f"    per-origin mine {np.round(mine_pp, 2)}")
    print(f"    per-origin  cl  {np.round(cl_pp, 2)}")
