#!/usr/bin/env python
"""Spike: E0 (calibrate the instrument) and E1 (head-to-head), on the classic triangles.

Design, in one place:

* **Every diagonal is a valuation date.** At anchor k, the books contain origins 0..k; cells with
  origin+development <= k are known and everything beyond is unknown. Because the full triangle is
  observed to its last diagonal, the *observed* part of that future is the scoring target. The tail
  beyond the last diagonal is not scoreable and is not scored.
* **The TabPFN arm predicts link ratios, recursively** (arms A2/A3 in docs/experiments.md). It is the
  estimand Chain Ladder uses, the target is bounded and stationary, and it stays leak-free: the global
  factor for an unobserved age does not exist at the anchor, so the model is handed the last available
  one plus a flag saying so.
* **E0 controls run for every cell of the grid**: a no-development floor, a shuffled-target placebo, a
  run-to-run repeat, and an anchor-time assertion printed with each result.

Usage:
    .venv/bin/python scripts/spike_e0_e1.py                    # local weights (TabPFN-3.5)
    .venv/bin/python scripts/spike_e0_e1.py --triangles abc genins
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "spike"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
N_DRAWS = 300
QUANTILE_LEVELS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]


def load_token() -> str | None:
    """TABPFN_TOKEN from the key file, so the run never depends on a stale shell export."""
    tok = os.environ.get("TABPFN_TOKEN")
    key = Path.home() / ".config" / "tfm" / "keys.env"
    if key.is_file():
        for line in key.read_text().splitlines():
            m = re.match(r'\s*(?:export\s+)?TABPFN_TOKEN\s*=\s*(.+)', line)
            if m:
                tok = m.group(1).strip().strip('"').strip("'")
    return tok


def make_model(backend: str):
    if backend == "client":
        from tabpfn_client import TabPFNRegressor
        return TabPFNRegressor()
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(device="cpu")


# ------------------------------------------------------------------------------------------------ data
def load_triangle(name: str) -> dict:
    import chainladder as cl

    tri = cl.load_sample(name)
    vals = np.asarray(tri.values, dtype=float)
    while vals.ndim > 2:
        vals = vals[0]
    return {
        "name": name,
        "origin": [str(o)[:10] for o in tri.origin],
        "ages": [int(a) for a in tri.development],
        "C": vals,
        "tri": tri,
    }


def known_mask(C: np.ndarray, k: int) -> np.ndarray:
    """Cells on the books at anchor k: origins 0..k, development up to the anchor diagonal."""
    N = C.shape[0]
    m = np.zeros_like(C, dtype=bool)
    for a in range(N):
        for d in range(N):
            if a <= k and a + d <= k and np.isfinite(C[a, d]):
                m[a, d] = True
    return m


def global_factors(C: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Volume-weighted age-to-age factors from cells known at the anchor. gf[j] maps age j -> j+1.

    Ages beyond the anchor have no observations, so the last available factor is carried forward and
    `gf_available` tells the model that this happened.
    """
    N = C.shape[0]
    gf = np.full(N, np.nan)
    for j in range(N - 1):
        num = den = 0.0
        for a in range(N):
            if known[a, j] and known[a, j + 1] and C[a, j] > 0:
                num += C[a, j + 1]
                den += C[a, j]
        if den > 0:
            gf[j] = num / den
    last = np.nan
    for j in range(N):
        if np.isfinite(gf[j]):
            last = gf[j]
        elif np.isfinite(last):
            gf[j] = last
    return gf


def gf_used(d: int, k: int, gf: np.ndarray) -> tuple:
    """The global factor this cell is entitled to see, and whether it exists at the anchor.

    Shared by the feature builder and the recursion so the two can never disagree about what was known.
    """
    j = d - 1
    if j <= k and j < len(gf) and np.isfinite(gf[j]):
        return float(gf[j]), 1.0
    return (float(gf[k]) if k < len(gf) and np.isfinite(gf[k]) else 1.0), 0.0


def features_for(C, known, a, d, k, gf) -> list:
    """Features for the link ratio C[a,d]/C[a,d-1], using only cells known at anchor k."""
    N = C.shape[0]
    last_age = min(k - a, N - 1)          # youngest observed age for this origin
    latest_cum = C[a, last_age] if last_age >= 0 else np.nan
    prev_cum = C[a, last_age - 1] if last_age >= 1 else np.nan
    own_last_ratio = (latest_cum / prev_cum) if (last_age >= 1 and prev_cum and prev_cum > 0) else np.nan
    gf_val, gf_available = gf_used(d, k, gf)
    return [a, d, a + d, latest_cum, np.log1p(max(latest_cum, 0.0)), own_last_ratio, gf_val, gf_available]


FEATURES = ["origin_idx", "dev_idx", "cal_idx", "latest_cum", "log_latest_cum",
            "own_last_ratio", "global_factor_prev", "global_factor_available"]


def training_rows(C, known, k, gf, shuffle=False, rng=None, target="ratio"):
    """Observed cell -> next cell transitions at anchor k.

    `target="ratio"` learns the raw link ratio. `target="delta"` learns the ratio *relative to* the
    volume-weighted factor for that age, so the model only has to correct a strong stable prior rather
    than rediscover it -- the direct attack on the compounding that blew up the raw-ratio arm.
    """
    N = C.shape[0]
    X, y = [], []
    for a in range(N):
        for d in range(1, N):
            if not (known[a, d] and known[a, d - 1]):
                continue
            prev, cur = C[a, d - 1], C[a, d]
            if prev <= 0 or not np.isfinite(cur):
                continue
            ratio = cur / prev
            if target == "delta":
                base, avail = gf_used(d, k, gf)
                if base <= 0:
                    continue
                ratio = ratio / base
            X.append(features_for(C, known, a, d, k, gf))
            y.append(ratio)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if shuffle and len(y):
        y = y[rng.permutation(len(y))]
    return X, y


def draws_from(model, Xq, n, rng):
    """Per-row draws from the model's predictive distribution, or None if it will not give one.

    The default grid is [0.1 .. 0.9], which cannot speak to a 95% interval, so a denser grid is
    requested explicitly and inverted by interpolation. The exact route (the bar distribution's own
    inverse CDF, exposed as `criterion` + 5000-bin `logits` under output_type="full") is the upgrade
    path for the final version.
    """
    try:
        q = np.asarray(model.predict(Xq, output_type="quantiles", quantiles=QUANTILE_LEVELS), dtype=float)
    except Exception as exc:
        draws_from.error = f"{type(exc).__name__}: {exc}"
        return None
    try:
        levels = np.asarray(QUANTILE_LEVELS, dtype=float)
        if q.ndim != 2:
            draws_from.error = f"unexpected quantile ndim {q.ndim} shape {q.shape}"
            return None
        if q.shape[0] == len(levels) and q.shape[1] == Xq.shape[0]:
            pass
        elif q.shape[1] == len(levels) and q.shape[0] == Xq.shape[0]:
            q = q.T
        else:
            draws_from.error = f"quantile shape {q.shape} vs levels {len(levels)} rows {Xq.shape[0]}"
            return None
        out = np.empty((n, Xq.shape[0]))
        u = rng.uniform(levels[0], levels[-1], size=n)
        for r in range(Xq.shape[0]):
            out[:, r] = np.interp(u, levels, q[:, r])
        return out
    except Exception as exc:
        draws_from.error = f"extraction: {type(exc).__name__}: {exc}"
        return None


def predict_recursive(model, C, known, k, gf, n_draws, rng, target="ratio"):
    """Fill every unknown cell of the book, one diagonal at a time. Returns point and sample reserves."""
    N = C.shape[0]
    Chat = C.copy()
    Chats = np.repeat(C[None, :, :], n_draws, axis=0).astype(float) if n_draws else None
    for diag in range(k + 1, 2 * N - 1):
        coords = [(a, diag - a) for a in range(N) if 1 <= diag - a < N and a <= k]
        if not coords:
            continue
        Xq = np.asarray([features_for(Chat, known, a, d, k, gf) for a, d in coords], dtype=float)
        point = np.atleast_1d(np.asarray(model.predict(Xq), dtype=float))
        draws = draws_from(model, Xq, n_draws, rng) if n_draws else None
        for i, (a, d) in enumerate(coords):
            prev = Chat[a, d - 1]
            if not (np.isfinite(prev) and prev > 0):
                continue
            base, _ = gf_used(d, k, gf)          # the same factor the feature was given
            r_hat = float(point[i])
            if target == "delta":
                r_hat = r_hat * base
            if not (np.isfinite(r_hat) and r_hat > 0):
                r_hat = 1.0
            Chat[a, d] = prev * r_hat
            if Chats is not None:
                col = draws[:, i] if draws is not None else np.full(n_draws, r_hat)
                if target == "delta":
                    col = col * base
                col = np.where(np.isfinite(col) & (col > 0), col, r_hat)
                Chats[:, a, d] = Chats[:, a, d - 1] * col
    reserve_point = 0.0
    for a in range(k + 1):
        last_age = N - 1 - a
        reserve_point += Chat[a, last_age] - C[a, k - a]
    reserve_samples = None
    if Chats is not None:
        reserve_samples = np.zeros(n_draws)
        for a in range(k + 1):
            last_age = N - 1 - a
            reserve_samples += Chats[:, a, last_age] - C[a, k - a]
    return reserve_point, reserve_samples


def factor_reserve(C, known, k, gf) -> float:
    """Chain-ladder arithmetic over the SAME cells the model is scored on.

    This is the fair comparison: both arms predict the observed part of the future. Volume-weighted
    factors come from cells known at the anchor; ages past the anchor carry the last factor forward
    (a flat tail), which is standard practice and is stated in the results.
    """
    N = C.shape[0]
    total = 0.0
    for a in range(k + 1):
        base = C[a, k - a]
        proj = base
        for d in range(k - a + 1, N - a):
            proj *= gf[d - 1]
        total += proj - base
    return float(total)


def chainladder_baseline(tri, k):
    """The CAS package on the as-of triangle — the authoritative reference.

    Its `ibnr` includes the tail beyond the last observed diagonal, which cannot be scored, so it is
    reported as a reference alongside the like-for-like factor arm, not as the head-to-head.
    """
    import chainladder as cl

    out: dict = {"cl_note": "ibnr includes the tail beyond the last diagonal; reference only"}
    try:
        sub = tri[tri.valuation <= tri.valuation[k]]
        out["cl_shape"] = list(np.asarray(sub.values).shape[-2:])
        out["cl_ibnr_reference"] = float(np.nansum(np.asarray(cl.Chainladder().fit(sub).ibnr_)))
    except Exception as exc:
        out["cl_error"] = f"{type(exc).__name__}: {exc}"
    try:
        mack = cl.MackChainladder().fit(sub)
        out["mack_ibnr_reference"] = float(np.nansum(np.asarray(mack.ibnr_)))
        for attr in ("total_mack_std_err_", "total_process_std_err_", "total_parameter_std_err_"):
            if hasattr(mack, attr):
                out[f"mack_{attr.rstrip('_')}"] = float(np.nansum(np.asarray(getattr(mack, attr))))
    except Exception as exc:
        out["mack_error"] = f"{type(exc).__name__}: {exc}"
    return out


def actual_future(C, k):
    N = C.shape[0]
    return float(sum(C[a, N - 1 - a] - C[a, k - a] for a in range(k + 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triangles", nargs="*", default=["abc", "genins", "mcl", "ukmotor"])
    ap.add_argument("--backend", choices=["local", "client"], default="local")
    ap.add_argument("--target", choices=["ratio", "delta"], default="ratio",
                    help="ratio: learn raw link ratios. delta: learn the ratio relative to the "
                         "volume-weighted factor, correcting a strong prior instead of rediscovering it.")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    args = ap.parse_args()

    tok = load_token()
    if tok:
        os.environ["TABPFN_TOKEN"] = tok
    rng = np.random.default_rng(SEED)
    report = {
        "seed": SEED, "n_draws": args.draws, "backend": args.backend,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "env": {}, "triangles": {}, "notes": [],
    }
    import importlib.metadata as md
    for pkg in ("tabpfn", "tabpfn-client", "chainladder", "torch", "numpy", "pandas", "scikit-learn"):
        try:
            report["env"][pkg] = md.version(pkg)
        except Exception:
            pass
    print("env:", report["env"])

    rows = []
    for name in args.triangles:
        d = load_triangle(name)
        C, N = d["C"], d["C"].shape[0]
        print(f"\n=== {name}: {N}x{N} (cumulative) ===")
        for k in range(N // 2, N - 2):
            known = known_mask(C, k)
            assert not np.any(known & (np.add.outer(np.arange(N), np.arange(N)) > k)), "anchor leak"
            gf = global_factors(C, known)
            actual = actual_future(C, k)
            train_X, train_y = training_rows(C, known, k, gf, target=args.target)
            shuf_X, shuf_y = training_rows(C, known, k, gf, shuffle=True, rng=rng, target=args.target)

            row = {"triangle": name, "anchor_k": int(k), "target": args.target,
                   "n_origins_in_scope": int(k + 1),
                   "n_train_rows": int(len(train_y)), "actual_future": actual,
                   "naive_no_development": 0.0}

            t0 = time.time()
            model = make_model(args.backend)
            model.fit(train_X, train_y)
            row["fit_seconds"] = time.time() - t0
            t0 = time.time()
            res, samples = predict_recursive(model, C, known, k, gf, args.draws, rng, target=args.target)
            row["predict_seconds"] = time.time() - t0
            row["tabpfn_reserve"] = res

            t0 = time.time()
            rep_model = make_model(args.backend)
            rep_model.fit(train_X, train_y)
            rep_res, _ = predict_recursive(rep_model, C, known, k, gf, 0, rng, target=args.target)
            row["tabpfn_repeat_reserve"] = rep_res
            row["repeat_seconds"] = time.time() - t0

            t0 = time.time()
            plc_model = make_model(args.backend)
            plc_model.fit(shuf_X, shuf_y)
            plc_res, _ = predict_recursive(plc_model, C, known, k, gf, 0, rng, target=args.target)
            row["placebo_reserve"] = plc_res
            row["placebo_seconds"] = time.time() - t0

            row["cl_factor_reserve"] = factor_reserve(C, known, k, gf)
            row.update(chainladder_baseline(d["tri"], k))
            for key in ("tabpfn_reserve", "placebo_reserve", "cl_factor_reserve",
                        "cl_ibnr_reference", "mack_ibnr_reference", "naive_no_development"):
                if key in row and actual:
                    row[f"err_pct_{key}"] = 100 * (row[key] - actual) / actual
            if samples is not None and np.any(samples > 0):
                row["tabpfn_interval"] = [
                    float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))]
                for lvl in (0.5, 0.75, 0.9, 0.95):
                    lo, hi = np.quantile(samples, [(1 - lvl) / 2, 1 - (1 - lvl) / 2])
                    row[f"covered_{lvl}"] = bool(lo <= actual <= hi)
            rows.append(row)
            print(f"  k={k:2d} actual={actual:13,.0f} | tabpfn={res:13,.0f} ({row.get('err_pct_tabpfn_reserve', float('nan')):+6.1f}%) "
                  f"| placebo={plc_res:13,.0f} ({row.get('err_pct_placebo_reserve', float('nan')):+6.1f}%) "
                  f"| CLf={row.get('cl_factor_reserve', float('nan')):13,.0f} ({row.get('err_pct_cl_factor_reserve', float('nan')):+6.1f}%) "
                  f"| mack={row.get('mack_ibnr_reference', float('nan')):13,.0f} "
                  f"| fit={row['fit_seconds']:.1f}s pred={row['predict_seconds']:.1f}s")

    df = pd.DataFrame(rows)
    stem = f"spike_e0_e1_{args.target}"
    df.to_csv(OUT / f"{stem}.csv", index=False)
    report["triangles"] = df.to_dict(orient="records")
    report["draws_error"] = getattr(draws_from, "error", None)
    (OUT / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT}/{stem}.csv")
    if df.empty:
        return 1
    print("\n--- summary ---")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        cols = [c for c in ["triangle", "anchor_k", "actual_future", "tabpfn_reserve", "placebo_reserve",
                            "cl_reserve", "mack_reserve"] if c in df.columns]
        print(df[cols].to_string(index=False))
    for key in ("tabpfn_reserve", "placebo_reserve", "cl_factor_reserve", "cl_ibnr_reference", "mack_ibnr_reference"):
        col = f"err_pct_{key}"
        if col in df:
            print(f"  mean err% {key:22s} {df[col].mean():+8.2f}   median abs {df[col].abs().median():7.2f}")
    for lvl in (0.5, 0.75, 0.9, 0.95):
        c = f"covered_{lvl}"
        if c in df:
            print(f"  coverage {lvl:.0%} (tabpfn): {df[c].mean():.0%} of {df[c].notna().sum()} scored")
    if report["draws_error"]:
        print("\ndistribution probe:", report["draws_error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
