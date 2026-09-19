"""Scratch recon #7: shapes of the arrays chainladder's Mack recursion uses, then replicate it."""
from __future__ import annotations

import numpy as np

import chainladder as cl

for name in ("ukmotor", "raa"):
    tr = cl.load_sample(name)
    m = cl.MackChainladder().fit(tr)
    X = m.X_
    Xn = np.asarray(X.values)
    print(f"\n=== {name} X.values {Xn.shape} nan_triangle {np.asarray(X.nan_triangle).shape} "
          f"ddims {list(X.ddims)} ldf_ {np.asarray(m.ldf_.values).shape} "
          f"cdf_ {np.asarray(m.cdf_.values).shape} sigma_ {np.asarray(X.sigma_.values).shape} "
          f"std_err_ {np.asarray(X.std_err_.values).shape} w_ {np.asarray(X.w_).shape} "
          f"full_triangle_ {np.asarray(m.full_triangle_.values).shape} "
          f"ultimate_ {np.asarray(m.ultimate_.values).shape}")
    print("  ldf_:", np.round(np.asarray(m.ldf_.values)[0, 0, 0], 5))
    print("  cdf_:", np.round(np.asarray(m.cdf_.values)[0, 0, 0], 5))
    print("  sigma_:", np.round(np.asarray(X.sigma_.values)[0, 0, 0], 5))
    print("  std_err_ [0,0]:\n", np.round(np.asarray(X.std_err_.values)[0, 0], 5))
    print("  w_ [0,0]:\n", np.round(np.asarray(X.w_)[0, 0], 5))
    print("  full_triangle_ [0,0]:\n", np.round(np.asarray(m.full_triangle_.values)[0, 0], 3))
    print("  X.values [0,0]:\n", np.round(Xn[0, 0], 3))
    print("  total_par:", np.asarray(m.total_parameter_risk_.values)[0, 0, 0, -3:])
    print("  total_proc:", np.asarray(m.total_process_risk_.values)[0, 0, 0, -3:])
    print("  proc_per (per origin, last col):\n",
          np.round(np.asarray(m.process_risk_.values)[0, 0, :, -2:], 3))
