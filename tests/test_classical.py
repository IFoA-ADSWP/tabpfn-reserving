"""The horizon-restricted classical methods, held to chainladder's own output.

`classical.mack_horizon` re-implements Mack's recursion so it can be stopped at each origin's target age
instead of at the triangle's ultimate. The only reason that is legitimate is that the re-implementation is
the same recursion: with the target set to the triangle's last age -- the run-off -- it must reproduce
chainladder's `ibnr_`, `mack_std_err_`, `process_risk_` and `total_process_risk_` exactly. That is the first
test, and everything else in the module rests on it.
"""
from __future__ import annotations

import numpy as np
import pytest

import chainladder as cl

from tabpfn_reserving.classical import LEVELS, as_of, mack_horizon, odp_horizon
from tabpfn_reserving.triangle import Triangle, factor_reserve, target_ages

SAMPLES = ["raa", "ukmotor", "genins"]


@pytest.mark.parametrize("sample", SAMPLES)
def test_mack_recursion_reproduces_chainladder_on_the_run_off(sample):
    """Stopped at the triangle's ultimate, the recursion IS chainladder's -- per origin and in total."""
    tri = Triangle.load(sample).trim()
    n = tri.n
    anchor = n - 1
    model = cl.MackChainladder().fit(as_of(cl, tri, anchor))
    out = mack_horizon(model, tri, anchor, target_ages(n, "production"))

    ibnr = np.nan_to_num(np.asarray(model.ibnr_.values)).ravel()
    assert out.point == pytest.approx(float(ibnr.sum()), rel=1e-9, abs=1e-9)
    assert out.per_origin_point == pytest.approx(list(ibnr), rel=1e-9, abs=1e-9)

    se = np.nan_to_num(np.asarray(model.mack_std_err_.values)[0, 0][:, -1])
    mine = np.sqrt(np.array(out.per_origin_parameter) ** 2 + np.array(out.per_origin_process) ** 2)
    assert np.allclose(mine, se, rtol=1e-9)

    assert out.process == pytest.approx(
        float(np.nan_to_num(np.asarray(model.total_process_risk_.values)).ravel()[-1]), rel=1e-9)

    # The aggregate parameter risk is Mack's derivative sum, which for the run-off collapses to
    # sqrt(sum_k (std_err_k / f_k)^2 * (sum of the ultimates of the origins that use f_k)^2) -- the
    # derivative of an origin's ultimate with respect to f_k is zero until that origin has reached age k,
    # so the origins summed at each k are the older ones. Computed here from primitives.
    ult = np.nan_to_num(np.asarray(model.ultimate_.values)).ravel()
    ldf = np.asarray(model.ldf_.values).ravel()
    std = np.asarray(model.X_.std_err_.values).ravel()
    closed = np.sqrt(sum((std[k] / ldf[k]) ** 2 * np.sum(ult[n - 1 - k:]) ** 2 for k in range(n - 1)))
    assert out.parameter == pytest.approx(closed, rel=1e-9)


@pytest.mark.parametrize("sample", SAMPLES)
def test_horizon_point_is_the_harness_chainladder_point(sample):
    """Truncated at an anchor, the classical point must be the harness's own `factor_reserve`.

    This is the convention check: the two methods have to be asked for the same cells, and `factor_reserve`
    is what the coverage run recorded in its `chainladder` column.
    """
    full = Triangle.load(sample).trim()
    for anchor in range(full.n - 2, max(full.n - 5, 2), -1):
        n = full.n
        tri = full
        targets = target_ages(n, "backtest")
        model = cl.MackChainladder().fit(as_of(cl, tri, anchor))
        out = mack_horizon(model, tri, anchor, targets)
        harness = factor_reserve(tri, anchor, tri.global_factors(tri.known(anchor)), targets=targets)
        assert out.point == pytest.approx(harness, rel=1e-9, abs=1e-9), f"{sample} anchor {anchor}"
        # Every target is `horizon` steps beyond the origin's last observed age, for every origin.
        assert all(int(targets[a]) - (anchor - a) == n - 1 - anchor for a in range(anchor + 1))


def test_mack_intervals_are_ordered_and_centred_on_the_point():
    tri = Triangle.load("raa").trim()
    anchor = tri.n - 2
    model = cl.MackChainladder().fit(as_of(cl, tri, anchor))
    out = mack_horizon(model, tri, anchor, target_ages(tri.n, "backtest"))
    bounds = [out.interval(lv) for lv in LEVELS]
    assert all(lo < out.point < hi for lo, hi in bounds)
    assert all(bounds[i][0] >= bounds[i + 1][0] and bounds[i][1] <= bounds[i + 1][1]
               for i in range(len(bounds) - 1))
    assert out.se > 0


def test_odp_bootstrap_mean_lands_on_the_chain_ladder_point():
    """The pre-registered reality check, on the triangle it was measured on (`abc`, ratio 1.002).

    A bootstrap whose mean is not the Chain Ladder point is not a bootstrap of this method, and no coverage
    number from it would mean anything.
    """
    tri = Triangle.load("abc").trim()
    anchor = tri.n - 1
    out = odp_horizon(cl, as_of(cl, tri, anchor), tri, anchor, target_ages(tri.n, "production"),
                      n_sims=1000, seed=20260919)
    assert out.n_nonfinite == 0
    assert abs(out.runoff_ratio - 1) < 0.002, out.runoff_ratio


def test_odp_bootstrap_is_reproducible():
    tri = Triangle.load("raa").trim()
    anchor = tri.n - 2
    targets = target_ages(tri.n, "backtest")
    a = odp_horizon(cl, as_of(cl, tri, anchor), tri, anchor, targets, n_sims=200, seed=7)
    b = odp_horizon(cl, as_of(cl, tri, anchor), tri, anchor, targets, n_sims=200, seed=7)
    assert np.array_equal(a.totals, b.totals)
    assert a.interval(0.90)[0] == b.interval(0.90)[0]
