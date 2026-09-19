"""The classical methods over the cells the model is scored on (#20, `docs/experiments.md` 5.2).

Mack's method and the ODP bootstrap both publish an interval for the **run-off** reserve: the projection from
the latest observed diagonal to each origin's ultimate. A fleet coverage unit is **not** that quantity. A
unit's realised future is the last `h` diagonals of the triangle -- `h` steps of development for every origin,
held out so that it is observed -- so a classical interval is only comparable to it if it is computed for the
same cells. Measured on one real unit, the difference is not cosmetic: the as-of triangle's own run-off
reserve is 422.39 where the unit's realised future is 7.00, because the run-off projects the newest origin to
the oldest origin's maturity instead of one diagonal.

So both methods are fitted on the as-of-anchor triangle -- the same truncation the harness uses -- and
evaluated on the projection to the **target diagonal**, the cells `targets[a]` for each origin `a`, which all
lie `h` steps beyond that origin's last observed age. Two conventions are taken from the harness and applied
identically to both methods, because a comparison run with two different truncations measures the truncations
rather than the methods:

* the as-of view is `Triangle.known(anchor)` -- the cells with `origin + development <= anchor`;
* ages the as-of triangle has no observation for (needed only by the oldest `h` origins, whose target lies
  past the last diagonal) are projected with the **last estimated age-to-age factor carried forward**, exactly
  as `Triangle.global_factors` does -- for the point estimate and for the standard error alike. The harness's
  own `factor_reserve` already does this for the point, and the point computed here agrees with the
  `chainladder` column the coverage run recorded, to the last digit, on every unit of the run.

Mack's recursion is re-implemented rather than read off `mack_std_err_` for one reason: chainladder fits to
the as-of triangle's ultimate, and this needs the same recursion **stopped at each origin's target**. The
re-implementation is validated in `tests/test_classical.py` against chainladder's own output -- it reproduces
`mack_std_err_`, `process_risk_` and `total_process_risk_` to 1e-9 on a run-off fit -- so the only difference
in use is where the accumulation stops. Mack's intervals are normal-based (`point +/- z * se`), which is the
method's own assumption and is stated as such in the findings.

`chainladder.total_parameter_risk_` is deliberately **not** used. Its operand is "every cell from the latest
valuation date onward", which is not Mack's derivative-based sum: on the first unit of the run it gives a
run-off total standard error of 350.13 where the consistent recursion gives 315.54. The aggregate used here
is the one consistent with the per-origin recursion that chainladder itself computes (and that
`tests/test_classical.py` reproduces exactly).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .triangle import Triangle

# The normal quantiles Mack's method implies, stated rather than looked up per call.
Z = {0.50: 0.6744897501960817, 0.75: 1.1503493803760079,
     0.90: 1.6448536269514722, 0.95: 1.959963984540054}
LEVELS = [0.50, 0.75, 0.90, 0.95]


@dataclass
class HorizonReserve:
    """A reserve and its prediction error, over one unit's target cells only."""

    point: float
    parameter: float
    process: float
    per_origin_point: list = field(default_factory=list)
    per_origin_parameter: list = field(default_factory=list)
    per_origin_process: list = field(default_factory=list)
    carried_steps: int = 0
    carried_point_share: float = 0.0

    @property
    def se(self) -> float:
        return float(np.sqrt(self.parameter ** 2 + self.process ** 2))

    def interval(self, level: float) -> tuple[float, float]:
        half = Z[level] * self.se
        return self.point - half, self.point + half


def as_of(chainladder, tri: Triangle, anchor: int):
    """The triangle as of `anchor`: exactly the harness's known cells, and nothing else.

    Built from the trimmed array rather than by slicing `tri.raw[valuation <= ...]`, because `raw` is the
    whole CAS container -- which can be larger than the trimmed block, and can carry data outside it. The
    constructor drops the all-NaN trailing origin and development columns, which is what makes the result a
    proper `(anchor + 1) x (anchor + 1)` triangle.
    """
    import pandas as pd

    vals = np.where(tri.known(anchor), tri.values, np.nan)
    frame = pd.DataFrame(vals, index=pd.Index(tri.origin, name="origin"),
                         columns=pd.Index(tri.ages, name="development"))
    return chainladder.Triangle(frame.stack().rename("values").reset_index(), origin="origin",
                                development="development", columns=["values"], cumulative=True)


class _Fitted:
    """The arrays a fitted Mack model contributes, with the harness's carry-forward conventions applied."""

    def __init__(self, model, values: np.ndarray, anchor: int):
        self.anchor = anchor
        # The as-of triangle's own axes -- `anchor + 1` origins and `anchor + 1` development columns --
        # NOT the harness's `n x n`: the mask removes the last origin and age, chainladder drops the
        # resulting all-NaN column, and the projected values are indexed on what is left. Using the
        # harness's width here reads chainladder's post-triangle column (a factor-1.0 extension) as if it
        # were the target, which is exactly the carried-forward step this module exists to get right.
        self.n_orig, self.n_dev = int(model.X_.shape[2]), int(model.X_.shape[-1])
        self.values = values
        ldf = np.asarray(model.ldf_.values).ravel()
        self.std_err = np.asarray(model.X_.std_err_.values).ravel()
        self.full = np.asarray(model.full_triangle_.values)[0, 0][:, :self.n_dev]
        self.full_std_err = np.asarray(model.full_std_err_.values)[0, 0][:, :self.n_dev]
        # The last factor estimated from the as-of data is at index anchor - 1: the transition
        # anchor-1 -> anchor is observed for the oldest origin and nothing beyond it is observed at all.
        self.last_estimable = float(ldf[anchor - 1])
        self.ldf = ldf

    def factor(self, j: int) -> float:
        """Age-to-age factor `j -> j+1`, carrying the last estimated one forward past the as-of edge."""
        return float(self.ldf[j]) if j <= self.anchor - 1 else self.last_estimable

    def scale(self, j: int) -> float:
        """`std_err_` at age `j`: the standard error of the age-to-age factor, carried forward with it."""
        return float(self.std_err[j]) if j <= self.anchor - 1 else float(self.std_err[self.anchor - 1])

    def process_scale(self, a: int, j: int) -> float:
        return float(self.full_std_err[a, j]) if j <= self.anchor - 1 else float(self.full_std_err[a, self.anchor - 1])

    def value(self, a: int, j: int) -> float:
        """Projected cumulative value for origin `a` at development index `j`, in the as-of geometry."""
        if j < self.n_dev:
            return float(self.full[a, j])
        out = float(self.full[a, self.n_dev - 1])
        for k in range(self.n_dev - 1, j):
            out *= self.factor(k)
        return out


def mack_horizon(model, tri: Triangle, anchor: int, targets) -> HorizonReserve:
    """Mack's point and prediction error for the reserve over the target cells.

    `model` is a `MackChainladder` fitted on `as_of(chainladder, tri, anchor)`. The per-origin recursion is
    chainladder's own, accumulated from each origin's last observed age up to its target and then stopped.
    """
    f = _Fitted(model, tri.values, anchor)
    out = HorizonReserve(point=0.0, parameter=0.0, process=0.0)
    latest = {}
    for a in range(anchor + 1):
        if a >= f.n_orig:
            continue
        last = anchor - a
        latest[a] = last
        rp = rr = 0.0
        for k in range(last, int(targets[a])):
            v = f.value(a, k)
            rp = float(np.sqrt((v * f.scale(k)) ** 2 + (f.factor(k) * rp) ** 2))
            rr = float(np.sqrt((v * f.process_scale(a, k)) ** 2 + (f.factor(k) * rr) ** 2))
        point = f.value(a, int(targets[a])) - float(tri.values[a, last])
        out.per_origin_point.append(point)
        out.per_origin_parameter.append(rp)
        out.per_origin_process.append(rr)
        out.point += point
        if int(targets[a]) > anchor:
            out.carried_steps += int(targets[a]) - anchor

    # The aggregate parameter risk. Mack's total is the *derivative sum* over origins -- for each estimated
    # factor f_k, the sensitivities of every origin's target value add before being squared -- so the
    # explicit form is a sum of squares with no recursion:
    #
    #     Var(total) = sum_k ( std_err_k * sum_a d C[a, T_a] / d f_k )^2,
    #     d C[a, T_a] / d f_k = C[a, k] * prod_{j=k+1}^{T_a-1} f_j   for k in [latest_a, T_a - 1].
    #
    # (This is the same formula the per-origin recursion above implements; see the module docstring for why
    # chainladder's `total_parameter_risk_` is not this number.)
    total = 0.0
    for k in range(0, int(max(targets[:anchor + 1]))):
        derivative = 0.0
        for a in range(anchor + 1):
            if a >= f.n_orig or not (latest[a] <= k <= int(targets[a]) - 1):
                continue
            carry = 1.0
            for j in range(k + 1, int(targets[a])):
                carry *= f.factor(j)
            derivative += f.value(a, k) * carry
        total += (derivative * f.scale(k)) ** 2
    out.parameter = float(np.sqrt(total))
    out.process = float(np.sqrt(sum(x ** 2 for x in out.per_origin_process)))
    denom = sum(abs(x) for x in out.per_origin_point) or 1.0
    carried = sum(abs(out.per_origin_point[a]) for a in range(len(out.per_origin_point))
                  if int(targets[a]) > anchor)
    out.carried_point_share = float(carried / denom)
    return out


@dataclass
class OdpResult:
    """The ODP bootstrap's h-step totals, one per simulation."""

    totals: np.ndarray
    n_nonfinite: int
    point_mean: float
    point_median: float
    runoff_mean: float
    runoff_cl: float

    @property
    def runoff_ratio(self) -> float:
        """mean over simulations / the Chain Ladder point, on the run-off reserve -- the reality check."""
        return float(self.runoff_mean / self.runoff_cl) if self.runoff_cl else float("nan")

    def interval(self, level: float) -> tuple[float, float]:
        lo, hi = np.quantile(self.totals, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)


def odp_horizon(chainladder, asof, tri: Triangle, anchor: int, targets, n_sims: int,
                seed: int) -> OdpResult:
    """The ODP bootstrap's totals over the unit's target cells, with a Chain Ladder refit per resample.

    The recipe is the pre-registered one: `BootstrapODPSample(n_sims, random_state=seed)` on the as-of
    triangle, then `Chainladder()` on every resample, then the total over the target cells of each resample's
    own projection. `nansum` is required on any ensemble sum -- the unobserved cells are NaN.
    """
    boot = chainladder.BootstrapODPSample(n_sims=n_sims, random_state=seed).fit(asof)
    fit = chainladder.Chainladder().fit(boot.resampled_triangles_)
    ldf = np.asarray(fit.ldf_.values).reshape(n_sims, -1)
    latest = np.asarray(fit.X_.latest_diagonal.values).reshape(n_sims, -1)
    last = ldf[:, anchor - 1]
    n_orig = latest.shape[1]
    totals = np.zeros(n_sims)
    for s in range(n_sims):
        for a in range(min(anchor + 1, n_orig)):
            base = latest[s, a]
            v = base
            for k in range(anchor - a, int(targets[a])):
                v *= ldf[s, k] if k <= anchor - 1 else last[s]
            totals[s] += v - base
    good = np.isfinite(totals)
    # The run-off reserve of the same resamples, which is what the pre-registered reality check is about.
    runoff = np.nan_to_num(np.asarray(fit.ibnr_.values).reshape(n_sims, -1)).sum(axis=1)
    runoff_cl = float(np.nansum(np.asarray(chainladder.Chainladder().fit(asof).ibnr_.values)))
    return OdpResult(totals=totals[good], n_nonfinite=int((~good).sum()),
                     point_mean=float(np.mean(totals[good])) if good.any() else float("nan"),
                     point_median=float(np.median(totals[good])) if good.any() else float("nan"),
                     runoff_mean=float(np.mean(runoff)), runoff_cl=runoff_cl)
