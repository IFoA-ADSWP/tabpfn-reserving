"""Scratch recon #2: validate a closed-form Mack aggregate MSE against chainladder's recursion."""
from __future__ import annotations

import numpy as np

import chainladder as cl


def mack_total_mse(chat,  # (I, K+1) projected cumulatives, index j of the age axis
                   f,     # f[j] factor from age j -> j+1
                   s2,    # sigma2[j]
                   S,     # S[j] volume behind f[j]
                   starts, targets):
    """Mack (1993) aggregate prediction error for sum_i (Chat[i, t_i] - Chat[i, s_i]).

    Var_i  = Chat_it^2 * sum_j (s2_j/f_j^2) (1/Chat_ij + 1/S_j)
    Cov_il = Chat_it Chat_lt * sum_j (s2_j/f_j^2) / S_j     (shared factor-estimation error)
    """
    I = len(starts)
    total = 0.0
    for i in range(I):
        sub = 0.0
        for j in range(starts[i], targets[i]):
            sub += (s2[j] / f[j] ** 2) * (1.0 / chat[i, j] + 1.0 / S[j])
        total += chat[i, targets[i]] ** 2 * sub
        for l in range(i + 1, I):
            lo, hi = max(starts[i], starts[l]), min(targets[i], targets[l])
            cov = 0.0
            for j in range(lo, hi):
                cov += (s2[j] / f[j] ** 2) / S[j]
            total += 2 * chat[i, targets[i]] * chat[l, targets[l]] * cov
    return total


def one(name: str) -> None:
    tr = cl.load_sample(name)
    m = cl.MackChainladder().fit(tr)
    vals = np.asarray(m.X_.values)[0, 0]
    n = vals.shape[0]
    full = np.asarray(m.full_triangle_.values)[0, 0][:, :n]
    ldf = np.asarray(m.ldf_.values)[0, 0, 0]
    cdf = np.asarray(m.cdf_.values)[0, 0, 0]
    sig = np.asarray(m.X_.sigma_.values)[0, 0, 0]
    print(f"\n=== {name}: n={n} ===")
    print("  ldf:", np.round(ldf, 6))
    print("  sigma2:", np.round(sig, 6))
    print("  ddims ldf:", list(m.ldf_.ddims), "sigma:", list(m.X_.sigma_.ddims))
    print("  reported total_mack_std_err_:", float(m.total_mack_std_err_.values[0, 0]))

    # candidate index mapping: ldf[j] = factor age index j -> j+1
    starts = [n - 1 - i for i in range(n)]
    targets = [(n - 1) for i in range(n)]
    chat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if j <= starts[i]:
                chat[i, j] = vals[i, j]
            else:
                chat[i, j] = chat[i, j - 1] * ldf[j - 1]
    print("  my projection vs full_triangle_: max abs rel diff =",
          np.nanmax(np.abs((chat - full) / np.where(full == 0, np.nan, full))))

    S = np.array([np.nansum(vals[: n - j, j]) for j in range(n)])
    mse = mack_total_mse(chat, ldf, sig, S, starts, targets)
    print("  my closed form total se:", np.sqrt(mse))
    print("  ratio (mine/cl):", np.sqrt(mse) / float(m.total_mack_std_err_.values[0, 0]))
    per_origin = []
    for i in range(n):
        sub = sum((sig[j] / ldf[j] ** 2) * (1 / chat[i, j] + 1 / S[j]) for j in range(starts[i], n - 1))
        per_origin.append(chat[i, n - 1] ** 2 * sub)
    print("  quadrature of per-origin se:", np.sqrt(np.sum(per_origin)))
    print("  cl per-origin se:", np.round(np.asarray(m.mack_std_err_.values)[0, 0, :, -1], 3))


for name in ("abc", "genins", "ukmotor", "raa", "mcl"):
    try:
        one(name)
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: FAILED {type(exc).__name__}: {exc}")
