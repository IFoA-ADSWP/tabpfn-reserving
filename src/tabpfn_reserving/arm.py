"""The TabPFN arm: link ratios predicted recursively, and the reserve distribution that falls out.

Two routes to the distribution, and the function says which it used rather than leaving it to be guessed:

* **exact** -- the bar distribution behind `output_type="full"`: `logits` are bin weights over the `borders`
  grid, so quantiles are read off the cumulative weights with linear interpolation inside the bin. No
  approximation beyond the bin width.
* **grid** -- the quantile levels the API returns, interpolated. Used when the borders are not exposed for
  the active backend.

Both are reported in the run record, because "we drew from the model's distribution" means different things
on the two routes.
"""
from __future__ import annotations

import numpy as np
from tabpfn import TabPFNRegressor

from .triangle import Triangle, features_for, gf_used, target_ages

QUANTILE_LEVELS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]


def make_model(backend: str = "local"):
    if backend == "client":
        from tabpfn_client import TabPFNRegressor as ClientRegressor
        return ClientRegressor()
    return TabPFNRegressor(device="cpu")


def _borders_from(full: dict) -> np.ndarray | None:
    """The bin edges of the bar distribution, wherever this backend puts them."""
    if isinstance(full, dict):
        if "borders" in full:
            return np.asarray(full["borders"], dtype=float)
        crit = full.get("criterion")
        if crit is not None and hasattr(crit, "borders"):
            return np.asarray(crit.borders, dtype=float)
    return None


def quantiles(model, X: np.ndarray, levels=QUANTILE_LEVELS) -> tuple[np.ndarray, str]:
    """Quantiles of the predictive distribution per row. Returns ((n_rows, n_levels), route)."""
    try:
        full = model.predict(X, output_type="full")
        borders = _borders_from(full)
        logits = full.get("logits") if isinstance(full, dict) else None
        if borders is not None and logits is not None:
            w = np.asarray(logits, dtype=float)
            w = np.exp(w - w.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            cw = np.cumsum(w, axis=1)
            out = np.empty((w.shape[0], len(levels)))
            for i, lv in enumerate(levels):
                idx = np.argmax(cw >= lv, axis=1)
                rows = np.arange(w.shape[0])
                lo_edge = borders[idx]
                hi_edge = borders[idx + 1]
                prev_cw = np.where(idx > 0, cw[rows, np.maximum(idx - 1, 0)], 0.0)
                width = cw[rows, idx] - prev_cw
                frac = np.where(width > 0, (lv - prev_cw) / np.where(width > 0, width, 1.0), 0.0)
                out[:, i] = lo_edge + frac * (hi_edge - lo_edge)
            return out, "exact"
    except Exception:  # noqa: BLE001 -- fall through to the grid route
        pass
    q = np.asarray(model.predict(X, output_type="quantiles", quantiles=list(levels)), dtype=float)
    if q.ndim == 2 and q.shape[0] == len(levels) and q.shape[1] == X.shape[0]:
        q = q.T
    return q, "grid"


def _bar_weights(full: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Bin edges and normalised bin weights of the bar distribution, if this backend exposes them.

    A `FullSupportBarDistribution` is piecewise-uniform over `borders` with weights given by `logits`, so
    it can be sampled directly -- no grid, no interpolation, no clamped tails.
    """
    borders = _borders_from(full)
    logits = full.get("logits") if isinstance(full, dict) else None
    if borders is None or logits is None:
        return None
    w = np.asarray(logits, dtype=float)
    if w.ndim != 2 or len(borders) != w.shape[1] + 1:
        return None
    w = np.exp(w - w.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return np.asarray(borders, dtype=float), w


def sample_from_bars(borders: np.ndarray, weights: np.ndarray, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw n samples per row by inverse-CDF sampling straight from the bar distribution.

    Bin index by the cumulative weights, position uniform inside the bin -- which is the distribution the
    model actually returns. Unlike inverting a 15-level grid this does not clamp the tails, and it uses
    every bin the model has, so a p99 is a p99 rather than an interpolation between two reported levels.
    """
    cum = np.cumsum(weights, axis=1)
    out = np.empty((n, weights.shape[0]))
    for r in range(weights.shape[0]):
        u = rng.random(n)
        idx = np.clip(np.searchsorted(cum[r], u), 0, len(borders) - 2)
        lo, hi = borders[idx], borders[idx + 1]
        out[:, r] = lo + rng.random(n) * (hi - lo)
    return out


def draws(model, X: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray | None, dict]:
    """n draws per row, and a record of how they were obtained.

    Two methods, and which one was used is reported rather than assumed:

    * **bar-bins** -- inverse-CDF sampling from the bar distribution. Exact given the model's own output.
    * **quantile-inversion** -- interpolate a fixed quantile grid. Used when the borders are not exposed
      for the active backend; the grid is 15 levels wide, so tails are clamped and the mixture is coarse.
    """
    try:
        bars = _bar_weights(model.predict(X, output_type="full"))
        if bars is not None:
            borders, weights = bars
            return sample_from_bars(borders, weights, n, rng), {
                "method": "bar-bins", "quantile_route": "exact"}
    except Exception as exc:  # noqa: BLE001 -- fall through to the grid, but say so
        failure = f"{type(exc).__name__}: {exc}"
    else:
        failure = "no borders in the model's full output"

    q, route = quantiles(model, X)
    if q.shape[1] != len(QUANTILE_LEVELS) or q.shape[0] != X.shape[0]:
        return None, {"method": "none", "quantile_route": route,
                      "why": f"unusable quantile shape {q.shape}"}
    levels = np.asarray(QUANTILE_LEVELS, dtype=float)
    u = rng.uniform(levels[0], levels[-1], size=n)
    out = np.empty((n, X.shape[0]))
    for r in range(X.shape[0]):
        out[:, r] = np.interp(u, levels, q[r])   # invert the quantile function
    return out, {"method": "quantile-inversion", "quantile_route": route, "why": failure}


def reserve(
    model,
    tri: Triangle,
    anchor: int,
    n_draws: int,
    rng: np.random.Generator,
    target: str = "ratio",
    targets: list[int] | None = None,
    mode: str = "frozen",
    trace: list | None = None,
) -> dict:
    """Project every unknown cell one diagonal at a time; return the reserve and its distribution.

    Recursion is not decoration: the global factor for an unobserved age does not exist at the anchor, so
    a direct prediction of the far cells would need a tail assumption the model is not entitled to make.

    `targets[a]` is the development age to project origin `a` to, and it is the difference between the two
    uses of this function:

    * **backtest** -- `targets[a] = n-1-a`, each origin's last *observed* age, because that is the only
      future the data can score.
    * **production** -- `targets[a] = n-1`, the longest development seen anywhere in the triangle, which
      for most origins lies beyond anything observed and is exactly the reserve being asked for.

    Getting these the same way round is not cosmetic: using the backtest definition in production returns
    a reserve of zero, because for every origin the difference is taken against a cell that was already
    observed.
    """
    n = tri.n
    tgt = targets if targets is not None else target_ages(n, "production")
    gf = tri.global_factors(tri.known(anchor))
    running = tri.values.copy()
    draws_by_cell: list[tuple[list, np.ndarray | None]] = []
    point_ratios: dict[tuple[int, int], float] = {}
    routes: set[str] = set()
    draw_methods: set[str] = set()

    for diag in range(anchor + 1, 2 * n - 1):
        coords = [(a, diag - a) for a in range(n) if 1 <= diag - a < n and a <= anchor]
        if not coords:
            continue
        X = np.asarray([features_for(tri, a, d, anchor, gf, running, mode=mode) for a, d in coords],
                       dtype=float)
        point = np.atleast_1d(np.asarray(model.predict(X), dtype=float))
        if n_draws:
            d_draws, info = draws(model, X, n_draws, rng)
            draw_methods.add(str(info.get("method", "unknown")))
            routes.add(str(info.get("quantile_route", "unknown")))
        else:
            d_draws = None
        for i, (a, d) in enumerate(coords):
            prev = running[a, d - 1]
            if not (np.isfinite(prev) and prev > 0):
                continue
            base, _ = gf_used(d, anchor, gf)
            r_hat = float(point[i]) * (base if target == "delta" else 1.0)
            if not (np.isfinite(r_hat) and r_hat > 0):
                r_hat = 1.0
            running[a, d] = prev * r_hat
            point_ratios[(a, d)] = r_hat
            if trace is not None:
                # The per-step record, so a diagnosis can be run on the steps rather than on the total.
                # `depth` is how far into the projection this step sits: every training row is at depth
                # <= 0, so any bias that grows with depth is bias in the extrapolation, not in the fit.
                actual = tri.values[a, d]
                actual_ratio = (actual / tri.values[a, d - 1]
                                if np.isfinite(actual) and tri.values[a, d - 1] > 0 else np.nan)
                trace.append({
                    "origin": int(a), "dev": int(d), "depth": int((d - 1) - (anchor - a)),
                    "predicted_ratio": float(r_hat), "actual_ratio": float(actual_ratio),
                    "prev_observed": float(tri.values[a, d - 1]),
                    "log_error": (float(np.log(r_hat / actual_ratio))
                                  if np.isfinite(actual_ratio) and actual_ratio > 0 and r_hat > 0
                                  else np.nan),
                    "base_observed": float(tri.values[a, anchor - a]),
                })
        draws_by_cell.append((coords, d_draws))

    point_reserve = sum(running[a, tgt[a]] - tri.values[a, anchor - a]
                        for a in range(anchor + 1))

    samples = None
    if n_draws:
        paths = np.repeat(tri.values[None, :, :], n_draws, axis=0).astype(float)
        for coords, d_draws in draws_by_cell:
            if d_draws is None:
                continue
            for i, (a, d) in enumerate(coords):
                base, _ = gf_used(d, anchor, gf)
                # The delta arm's draws live on the same relative scale as its point predictions, so
                # they need the same rescaling. Without this the sampled reserve is a product of
                # near-1.0 ratios while the point reserve is a product of factor-sized ones, and the
                # distribution silently describes a different quantity than the headline number.
                col = d_draws[:, i] * (base if target == "delta" else 1.0)
                col = np.where(np.isfinite(col) & (col > 0), col, point_ratios.get((a, d), 1.0))
                paths[:, a, d] = paths[:, a, d - 1] * col
        samples = np.array([paths[:, a, tgt[a]] - tri.values[a, anchor - a]
                            for a in range(anchor + 1)]).sum(axis=0)

    return {
        "triangle": tri.name,
        "anchor": int(anchor),
        "target": target,
        "features": mode,
        # Three summaries of the same object. They are reported separately because they are not equal and
        # the difference is the compounding, not a rounding error: `reserve` multiplies per-cell
        # predictions, `mean`/`median` are the sampled paths. Calling any of them "the reserve" without
        # saying which is what let a point estimate sit 32% above its own distribution unnoticed.
        "reserve": float(point_reserve),
        "reserve_mean": float(np.mean(samples)) if samples is not None else None,
        "reserve_median": float(np.median(samples)) if samples is not None else None,
        "samples": samples,
        "distribution_route": ",".join(sorted(routes)) or "none",
        "draws_method": ",".join(sorted(draw_methods)) or "none",
        "targets": [int(t) for t in tgt[: anchor + 1]],
    }
