"""Regression tests for the two bugs that shipped in the first CLI run, and the contracts around them.

Both bugs were the same mistake -- the reserve's *target ages* -- and both were invisible until a number
came out as zero. So the tests here are about the arithmetic contracts, not about the model: they run in
seconds, with no TabPFN fit and no token, by standing a stub regressor in for TabPFN-3.5.
"""
from __future__ import annotations

import numpy as np
import pytest

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import (
    Triangle, chainladder_baseline, factor_reserve, target_ages, training_rows,
)


class StubModel:
    """Predicts a constant ratio. Tests the projection, the sampling and the scaling -- not the skill."""

    def __init__(self, value: float = 1.05) -> None:
        self.value = value

    def fit(self, X, y):  # noqa: ANN001, ANN201
        return self

    def predict(self, X, output_type: str = "mean", quantiles=None):  # noqa: ANN001, ANN201
        n = len(X)
        if output_type == "quantiles":
            levels = np.asarray(quantiles, dtype=float)
            # a symmetric spread around the constant, so the mean of the draws equals the point
            return np.tile(self.value * (0.9 + 0.2 * levels), (n, 1)).T
        return np.full(n, self.value)


TRIANGLES = ["abc", "genins", "ukmotor"]


# ---------------------------------------------------------------------------------------------
# The arithmetic contract: our Chain Ladder must be the CAS package's Chain Ladder.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", TRIANGLES)
def test_factor_reserve_reproduces_the_cas_package(name: str) -> None:
    """If these ever drift apart the comparison is no longer a comparison, it is two methods talking."""
    tri = Triangle.load(name)
    anchor = tri.n - 1
    ours = factor_reserve(tri, anchor, tri.global_factors(tri.known(anchor)))
    theirs = chainladder_baseline(tri, anchor)["chainladder_ibnr"]
    assert ours == pytest.approx(theirs, rel=1e-9), f"{name}: {ours} vs {theirs}"


@pytest.mark.parametrize("name", TRIANGLES)
def test_backtest_convention_is_zero_at_the_last_anchor(name: str) -> None:
    """The backtest definition projects to each origin's last observed age; at the last anchor, there is
    no unobserved development left, so it must return exactly zero. It returning zero in *production* was
    the bug -- here it is the correct answer, and pinning it is what makes the distinction testable."""
    tri = Triangle.load(name)
    anchor = tri.n - 1
    gf = tri.global_factors(tri.known(anchor))
    assert factor_reserve(tri, anchor, gf, targets=target_ages(tri.n, "backtest")) == pytest.approx(0.0)


def test_the_two_functions_agree_on_their_default() -> None:
    """The bug in one line: the model arm defaulted to production targets and the baseline to backtest
    targets, so calling both without arguments silently compared 12.8m against 0."""
    tri = Triangle.load("abc")
    anchor = tri.n - 1
    gf = tri.global_factors(tri.known(anchor))
    assert factor_reserve(tri, anchor, gf) == pytest.approx(
        factor_reserve(tri, anchor, gf, targets=target_ages(tri.n, "production")))


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="production"):
        target_ages(10, "whatever-this-is")


# ---------------------------------------------------------------------------------------------
# The production reserve: non-zero, and consistent with its own distribution.
# ---------------------------------------------------------------------------------------------

def test_production_reserve_is_not_zero() -> None:
    """The regression guard for the bug that produced the first CLI run's zeros."""
    tri = Triangle.load("abc")
    out = arm.reserve(StubModel(), tri, tri.n - 1, 0, np.random.default_rng(0), target="ratio")
    assert out["reserve"] > 0, "production reserve collapsed to zero -- target ages wrong again"
    assert out["targets"] == [tri.n - 1] * (tri.n - 1 + 1)


@pytest.mark.parametrize("target", ["ratio", "delta"])
def test_draws_and_point_describe_the_same_quantity(target: str) -> None:
    """The delta arm's draws must be rescaled by the same global factor as its point predictions.

    Without that rescaling the point reserve is a product of factor-sized ratios while the sampled reserve
    is a product of near-1.0 ratios, and the distribution quietly describes a different quantity than the
    number printed above it. With a constant stub the two must agree closely.
    """
    tri = Triangle.load("abc")
    out = arm.reserve(StubModel(), tri, tri.n - 1, 500, np.random.default_rng(0), target=target)
    assert out["samples"] is not None
    assert np.mean(out["samples"]) == pytest.approx(out["reserve"], rel=0.25), (
        f"{target}: point reserve {out['reserve']:,.0f} but mean of the distribution "
        f"{np.mean(out['samples']):,.0f} -- the two are on different scales"
    )


def test_both_arms_are_scored_on_the_same_cells() -> None:
    """Model and baseline must agree on the target ages, at every anchor, or the comparison is void.

    The anchor here is deliberately *not* the last one: at `n-1` the backtest convention is zero for every
    origin, which is correct and is pinned by its own test above. A mid-triangle anchor is where the two
    arms have something to disagree about.
    """
    tri = Triangle.load("ukmotor")
    anchor = 4
    gf = tri.global_factors(tri.known(anchor))
    bt = target_ages(tri.n, "backtest")
    model = arm.reserve(StubModel(), tri, anchor, 0, np.random.default_rng(0), targets=bt)
    assert model["targets"] == bt[: anchor + 1]
    assert factor_reserve(tri, anchor, gf, targets=bt) > 0


# ---------------------------------------------------------------------------------------------
# No leakage: features may not see anything that was not observable at the anchor.
# ---------------------------------------------------------------------------------------------

def test_features_ignore_the_unknown_region() -> None:
    """Corrupt every cell after the anchor and the training features must be bit-identical. Anything that
    moves is a look-ahead, and a look-ahead is the difference between a forecast and a fit."""
    tri = Triangle.load("abc")
    anchor = 6
    known = tri.known(anchor)
    gf = tri.global_factors(known)
    X_clean, y_clean = training_rows(tri, anchor, gf)

    corrupted = tri.values.copy()
    corrupted[~known] = 9.9e9
    doctored = Triangle(name=tri.name, values=corrupted, columns=tri.columns,
                        origin=tri.origin, ages=tri.ages)
    X_dirty, y_dirty = training_rows(doctored, anchor, gf)

    assert np.array_equal(X_clean, X_dirty)
    assert np.array_equal(y_clean, y_dirty)


# ---------------------------------------------------------------------------------------------
# A collection is not a triangle.
# ---------------------------------------------------------------------------------------------

def test_a_single_triangle_still_loads() -> None:
    """The guard against collections must not reject the actual triangles -- it very nearly did."""
    assert Triangle.load("abc").values.shape == (11, 11)


def test_a_collection_of_triangles_is_refused() -> None:
    """`clrd` is 775 triangles. Slicing it to the first member would have reported a reserve for an
    arbitrary line of business with nothing downstream able to notice."""
    with pytest.raises(ValueError, match="collection"):
        Triangle.load("clrd")
