"""Scratch recon #3: decompose chainladder's Mack total into parameter/process and compare."""
from __future__ import annotations

import numpy as np

import chainladder as cl


def parts(name: str) -> None:
    tr = cl.load_sample(name)
    m = cl.MackChainladder().fit(tr)
    vals = np.asarray(m.X_.values)[0, 0]
    n = vals.shape[0]
    ldf = np.asarray(m.ldf_.values)[0, 0, 0][:n]
    sig = np.asarray(m.X_.sigma_.values)[0, 0, 0][:n]
    print(f"\n=== {name} n={n} ===")
    print("  total_mack_std_err_: %.4f" % float(m.total_mack_std_err_.values[0, 0]))
    print("  total_parameter_risk_ final:",
          np.asarray(m.total_parameter_risk_.values)[0, 0, 0, -2:])
    print("  total_process_risk_ final:",
          np.asarray(m.total_process_risk_.values)[0, 0, 0, -2:])
    print("  std_err_[0,0]:", np.round(np.asarray(m.X_.std_err_.values)[0, 0], 6))
    print("  sigma/sqrt(C):", np.round(np.array([sig[j] / np.sqrt(vals[0, j]) if np.isfinite(vals[0, j]) and vals[0, j] > 0 else np.nan for j in range(n)]), 6))

    starts = [n - 1 - i for i in range(n)]
    chat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            chat[i, j] = vals[i, j] if j <= starts[i] else chat[i, j - 1] * ldf[j - 1]
    S = np.array([np.nansum(vals[: n - j, j]) for j in range(n)])
    C_ult = chat[:, n - 1]
    par = 0.0
    for j in range(n - 1):
        par += (sig[j] / ldf[j] ** 2) * (np.nansum(C_ult) ** 2) / S[j]
    proc = 0.0
    for i in range(n):
        for j in range(starts[i], n - 1):
            proc += C_ult[i] ** 2 * (sig[j] / ldf[j] ** 2) / chat[i, j]
    print("  mine: parameter %.2f  process %.2f  total %.2f"
          % (np.sqrt(par), np.sqrt(proc), np.sqrt(par + proc)))
    # cross-check: per-origin SEs
    mine_pi, cl_pi = [], []
    for i in range(n):
        s = sum((sig[j] / ldf[j] ** 2) * (1 / chat[i, j] + 1 / S[j]) for j in range(starts[i], n - 1))
        mine_pi.append(np.sqrt(C_ult[i] ** 2 * s))
        cl_pi.append(np.asarray(m.mack_std_err_.values)[0, 0, i, -1])
    print("  per-origin mine:", np.round(mine_pi, 2))
    print("  per-origin  cl :", np.round(cl_pi, 2))


for nm in ("ukmotor", "raa", "abc"):
    parts(nm)
