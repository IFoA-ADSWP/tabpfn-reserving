"""Scratch recon #9: why is the replication zero?"""
from __future__ import annotations

import numpy as np

import chainladder as cl

tr = cl.load_sample("ukmotor")
m = cl.MackChainladder().fit(tr)
X = m.X_
full = np.asarray(m.full_triangle_.values)[0, 0]
print("full shape", full.shape)
nt = np.asarray(X.nan_triangle)
print("nan_triangle (first rows):\n", nt[:3])
print("full[:, :7]:\n", np.round(full[:, :7], 3))
se_par = np.asarray(X.std_err_.values)[0, 0]
print("se_par shape", se_par.shape, se_par)
print("t1 = full*se:\n", np.round(np.nan_to_num(full[:, :7]) * se_par, 4))
nans = 1 - np.concatenate((np.nan_to_num(nt), np.zeros((nt.shape[0], 1))), axis=1)
print("nans:\n", nans)
print("f:", np.round(np.asarray(m.ldf_.values)[0, 0, 0], 5))
print("cdf:", np.round(np.asarray(m.cdf_.values)[0, 0, 0], 5))
