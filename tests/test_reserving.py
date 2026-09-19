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
    DIRECT_FEATURES, FEATURES, IDENTIFIER_FEATURES, IDENTIFIER_INDICES, RUNNING_FEATURES, Triangle,
    chainladder_baseline, direct_training_rows, factor_reserve, features_for, target_ages, training_rows,
)


class StubModel:
    """Predicts a constant ratio. Tests the projection, the sampling and the scaling -- not the skill.

    `spread` controls how wide the predictive distribution is, which matters when comparing a point
    estimate against the mean of its own draws: with a wide spread the Monte Carlo error is large enough to
    swamp the structural property being tested, so the tests that care about structure use a tight one.
    """

    def __init__(self, value: float = 1.05, spread: float = 0.2) -> None:
        self.value = value
        self.spread = spread

    def fit(self, X, y):  # noqa: ANN001, ANN201
        return self

    def predict(self, X, output_type: str = "mean", quantiles=None):  # noqa: ANN001, ANN201
        n = len(X)
        if output_type == "quantiles":
            levels = np.asarray(quantiles, dtype=float)
            # symmetric around the constant, so the mean of the draws equals the point
            half = self.spread / 2
            return np.tile(self.value * (1 - half + self.spread * levels), (n, 1)).T
        return np.full(n, self.value)


TRIANGLES = ["abc", "genins", "ukmotor"]


# ---------------------------------------------------------------------------------------------
# The construction contract: the recorded configuration is the default, and the options are opt-in (#22)
# ---------------------------------------------------------------------------------------------

def test_the_default_construction_is_the_recorded_configuration() -> None:
    """Every number in `results/` came from `make_model()` with no arguments.

    The two configuration options are meant to be an *addition* to the recorded runs -- four cells, one factor
    each -- so if the defaults drift the baseline cell stops being a replication and every recorded result
    silently starts describing code that no longer exists. Nothing here fits a model, so the check is free.
    """
    from tabpfn import TabPFNRegressor

    model = arm.make_model("local")
    params = model.get_params()
    assert isinstance(model, TabPFNRegressor)
    assert params["device"] == "cpu"
    assert params["categorical_features_indices"] is None, "identifiers are being declared categorical"
    assert params["inference_config"] is None, "the target transform stack is no longer the default"


def test_each_configuration_option_moves_exactly_one_parameter() -> None:
    """One factor per cell. An option that moved a second parameter would break the 2x2 and leave the
    paired differences with two explanations."""
    baseline = arm.make_model("local").get_params()
    categorical = arm.make_model("local", categorical_identifiers=True).get_params()
    transform = arm.make_model("local", extrapolating_target=True).get_params()
    both = arm.make_model("local", categorical_identifiers=True, extrapolating_target=True).get_params()

    moved = {key for key in baseline if baseline[key] != categorical[key]}
    assert moved == {"categorical_features_indices"}, moved
    assert categorical["categorical_features_indices"] == IDENTIFIER_INDICES

    moved = {key for key in baseline if baseline[key] != transform[key]}
    assert moved == {"inference_config"}, moved
    assert transform["inference_config"] == {
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": arm.EXTRAPOLATING_TRANSFORMS}

    assert both["categorical_features_indices"] == IDENTIFIER_INDICES
    assert both["inference_config"] == transform["inference_config"]


def test_the_declared_categorical_columns_really_are_the_identifiers() -> None:
    """Positions, not names, reach TabPFN -- so the positions have to be checked against the columns they are
    supposed to name, and against the matrix the direct arm is actually fitted on (it appends to FEATURES)."""
    tri = Triangle.load("abc")
    X, _ = direct_training_rows(tri, [6], target="delta")
    identifiers = X[:, IDENTIFIER_INDICES]

    assert IDENTIFIER_INDICES == [FEATURES.index(name) for name in IDENTIFIER_FEATURES]
    assert X.shape[1] > IDENTIFIER_INDICES[-1], "the direct arm does not carry the identifier columns"
    assert np.array_equal(identifiers, np.round(identifiers)), "an identifier column is not a whole number"
    assert identifiers[:, 0].min() >= 0 and identifiers[:, 1].min() >= 1
    assert identifiers[:, 2].max() <= 2 * (tri.n - 1), "calendar index outside the triangle's own range"


def test_the_client_backend_refuses_the_configuration_options() -> None:
    """The options are `TabPFNRegressor` parameters. Dropping them quietly on the client would report an
    unconfigured cell as a configured one -- which is the failure mode this whole run is built to avoid."""
    with pytest.raises(ValueError, match="client"):
        arm.make_model("client", categorical_identifiers=True)
    with pytest.raises(ValueError, match="client"):
        arm.make_model("client", extrapolating_target=True)


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
# Sampling: the draws must come from the distribution the model actually returned.
# ---------------------------------------------------------------------------------------------

class StubBarModel:
    """Returns a bar distribution built by hand, so the sampler can be checked against arithmetic."""

    def __init__(self, borders: np.ndarray, weights: np.ndarray) -> None:
        self.borders = np.asarray(borders, dtype=float)
        self.weights = np.asarray(weights, dtype=float)

    def fit(self, X, y):  # noqa: ANN001, ANN201
        return self

    def predict(self, X, output_type: str = "mean", quantiles=None):  # noqa: ANN001, ANN201
        n = len(X)
        if output_type == "full":
            logits = np.log(np.tile(self.weights, (n, 1)))
            return {"borders": self.borders, "logits": logits}
        return np.full(n, float(np.average(self.borders[:-1], weights=self.weights)))


def test_draws_from_a_uniform_bar_distribution_are_uniform() -> None:
    """Twenty equal bins over [0, 20] must sample as a uniform on [0, 20]: mean 10, nothing outside."""
    borders = np.linspace(0, 20, 21)
    model = StubBarModel(borders, np.ones(20))
    X = np.zeros((3, 8))
    samples, info = arm.draws(model, X, 20_000, np.random.default_rng(0))

    assert info["method"] == "bar-bins"
    assert info["quantile_route"] == "exact"
    assert samples.shape == (20_000, 3)
    assert samples.min() >= 0 and samples.max() <= 20
    assert np.mean(samples) == pytest.approx(10.0, abs=0.1)


def test_draws_from_a_point_mass_stay_in_that_bin() -> None:
    """All the weight in one bin means every draw lands in that bin -- the sampler's basic contract, and
    the thing an interpolated 15-level grid cannot do at any resolution."""
    borders = np.linspace(0, 20, 21)
    weights = np.zeros(20)
    weights[5] = 1.0
    model = StubBarModel(borders, weights)
    samples, _ = arm.draws(model, np.zeros((2, 8)), 1_000, np.random.default_rng(0))

    assert samples.min() >= borders[5] and samples.max() <= borders[6]


def test_the_sampler_reports_when_it_has_to_fall_back_to_the_grid() -> None:
    """A backend that exposes no borders must say so rather than silently sampling a coarser object."""
    samples, info = arm.draws(StubModel(), np.zeros((2, 8)), 200, np.random.default_rng(0))
    assert info["method"] == "quantile-inversion"
    assert "why" in info



@pytest.mark.parametrize("mode", ["frozen", "running"])
def test_features_ignore_the_unknown_region(mode: str) -> None:
    """Corrupt every cell after the anchor and the training features must be bit-identical. Anything that
    moves is a look-ahead, and a look-ahead is the difference between a forecast and a fit.

    Checked for both feature sets: `running` is allowed to see the model's *own* projections, never the
    truth, and the training rows are all observed cells, so corrupting the future must change nothing.
    """
    tri = Triangle.load("abc")
    anchor = 6
    known = tri.known(anchor)
    gf = tri.global_factors(known)
    X_clean, y_clean = training_rows(tri, anchor, gf, mode=mode)

    corrupted = tri.values.copy()
    corrupted[~known] = 9.9e9
    doctored = Triangle(name=tri.name, values=corrupted, columns=tri.columns,
                        origin=tri.origin, ages=tri.ages)
    X_dirty, y_dirty = training_rows(doctored, anchor, gf, mode=mode)

    assert np.array_equal(X_clean, X_dirty, equal_nan=True)
    assert np.array_equal(y_clean, y_dirty, equal_nan=True)


def test_running_features_extend_the_frozen_ones() -> None:
    """The arms must be nested, so an improvement is attributable to the information and not to a change
    of representation."""
    tri = Triangle.load("abc")
    anchor = tri.n - 1
    gf = tri.global_factors(tri.known(anchor))
    frozen = features_for(tri, 3, 8, anchor, gf, tri.values, mode="frozen")
    running = features_for(tri, 3, 8, anchor, gf, tri.values, mode="running")
    assert len(frozen) == len(FEATURES)
    assert len(running) == len(FEATURES) + len(RUNNING_FEATURES)
    assert running[: len(frozen)] == frozen


def test_frozen_features_are_blind_to_the_projection_and_running_ones_are_not() -> None:
    """The claim in #14, tested directly rather than argued.

    The frozen vector still varies across steps -- it carries the development and calendar indices, which
    are the only thing telling the steps apart. What it cannot do is say *how far the projection has pushed
    the origin*: its anchor-state block (the origin's level and last ratio at the anchor) is identical at
    every step, so "first step" and "ninth step" are described identically apart from the index. In running
    mode the level already reached enters that block.
    """
    tri = Triangle.load("abc")
    anchor = tri.n - 1
    gf = tri.global_factors(tri.known(anchor))
    a = 5
    projected = tri.values.copy()
    for d in range(anchor - a + 1, tri.n):
        projected[a, d] = projected[a, d - 1] * 1.05

    steps = list(range(anchor - a + 1, tri.n))
    anchor_state = slice(3, 6)      # latest_cum, log_latest_cum, own_last_ratio -- origin state
    projection_state = slice(len(FEATURES), len(FEATURES) + len(RUNNING_FEATURES))

    frozen_anchor = {tuple(features_for(tri, a, d, anchor, gf, projected, mode="frozen")[anchor_state])
                     for d in steps}
    running_projection = {tuple(features_for(tri, a, d, anchor, gf, projected, mode="running")
                                [projection_state]) for d in steps}
    frozen_has_projection_block = len(features_for(tri, a, steps[0], anchor, gf, projected,
                                                   mode="frozen")) > len(FEATURES)

    assert len(frozen_anchor) == 1, "the frozen anchor-state block should not vary across the projection"
    assert not frozen_has_projection_block, "frozen mode must not carry a projection block at all"
    assert len(running_projection) == len(steps), \
        "running mode should describe every step of the projection distinctly"


# ---------------------------------------------------------------------------------------------
# Fleet containers: sparse, shifted, and not triangles until trimmed
# ---------------------------------------------------------------------------------------------

def _shifted_container() -> Triangle:
    """A proper 5x5 triangle sitting inside a 10x10 container, as clrd actually ships them."""
    vals = np.full((10, 10), np.nan)
    for a in range(5):
        for d in range(5 - a):
            vals[3 + a, d] = 100.0 * (a + 1) * (1 + 0.1 * d)
    return Triangle(name="shifted", values=vals, columns=["IncurLoss"],
                    origin=[str(2000 + i) for i in range(10)], ages=list(range(10)))


def test_trim_finds_the_effective_triangle_inside_a_container() -> None:
    tri = _shifted_container().trim()
    assert tri.n == 5
    assert tri.origin[0] == "2003"          # the block starts at container origin 3
    # A proper triangle is filled exactly on and above the diagonal -- `isfinite().all()` would be the
    # assertion that the data is *not* a triangle.
    expected = np.zeros((5, 5), dtype=bool)
    for a in range(5):
        for d in range(5 - a):
            expected[a, d] = True
    assert np.array_equal(np.isfinite(tri.values), expected)
    assert np.isfinite(tri.values).sum() == tri.n * (tri.n + 1) // 2


def test_trim_refuses_a_fragmented_container() -> None:
    """Gaps in the middle are not a triangle, and pretending otherwise turns every downstream number into
    NaN without an error."""
    vals = np.full((10, 10), np.nan)
    for a in range(5):
        for d in range(5 - a):
            vals[2 + a, d] = 1.0
    vals[4, 0] = np.nan                      # punch a hole in the middle
    with pytest.raises(ValueError, match="gapless"):
        Triangle(name="fragmented", values=vals, columns=["x"],
                 origin=[str(i) for i in range(10)], ages=list(range(10))).trim()


def test_training_labels_cannot_come_from_beyond_the_evaluation_anchor() -> None:
    """Without the clamp the direct arm is trained on the very tail it is asked to predict -- which would
    flatter every fleet number. The check is made non-vacuous: turning the clamp off must *change* the
    labels when the future is corrupted, or the test proves nothing."""
    tri = Triangle.load("abc")
    anchor = 6
    known = tri.known(anchor)
    train = list(range(2, anchor))

    corrupted = tri.values.copy()
    corrupted[~known] = 9.9e9
    doctored = Triangle(name=tri.name, values=corrupted, columns=tri.columns,
                        origin=tri.origin, ages=tri.ages)

    _, y_clamped_clean = direct_training_rows(tri, train, target="ratio", known_until=anchor)
    _, y_clamped_dirty = direct_training_rows(doctored, train, target="ratio", known_until=anchor)
    _, y_open_dirty = direct_training_rows(doctored, train, target="ratio")

    assert np.array_equal(y_clamped_clean, y_clamped_dirty), "labels read past the evaluation anchor"
    assert not np.allclose(y_clamped_clean, y_open_dirty), "the clamp never mattered -- vacuous test"



def test_direct_arm_covers_a_range_of_horizons_and_the_recursive_one_cannot() -> None:
    """The horizon only varies *across* anchors -- at one anchor every origin develops to its own last
    observed age. Training across anchors is therefore what gives the model horizons to learn from."""
    tri = Triangle.load("abc")
    X, y = direct_training_rows(tri, list(range(2, tri.n - 1)), target="delta")
    assert len(X) == len(y) > 20
    assert X.shape[1] == len(DIRECT_FEATURES)
    horizons = sorted(set(X[:, -2]))
    assert len(horizons) > 3, "only one horizon seen -- the arm has nothing to learn the horizon from"


def test_direct_features_do_not_see_the_future_it_is_predicting() -> None:
    """The *labels* legitimately come from the future -- that is what supervised learning is. The
    *features* must not, and the check is worth making non-vacuous: the corruption has to actually reach
    the labels, or the test proves nothing about the features."""
    tri = Triangle.load("abc")
    anchor = 6
    known = tri.known(anchor)
    X_clean, y_clean = direct_training_rows(tri, [anchor], target="ratio")

    corrupted = tri.values.copy()
    corrupted[~known] = 9.9e9
    doctored = Triangle(name=tri.name, values=corrupted, columns=tri.columns,
                        origin=tri.origin, ages=tri.ages)
    X_dirty, y_dirty = direct_training_rows(doctored, [anchor], target="ratio")

    assert np.array_equal(X_clean, X_dirty, equal_nan=True), "features saw the future"
    assert not np.allclose(y_clean, y_dirty), "the corruption never reached the labels -- vacuous test"


def test_direct_point_and_distribution_cannot_disagree() -> None:
    """Both are the same sum of the same per-origin factors, so the gap the recursive arm shows between its
    point estimate and its own distribution is gone by construction, not by tuning.

    A tight predictive spread is used deliberately: this is a claim about structure, and a wide spread's
    Monte Carlo error would swamp it at 400 draws (it very nearly did -- see the first version of this
    test, which failed at a 12% gap that was entirely sampling noise on a reserve that is 5% of the base).
    """
    tri = Triangle.load("abc")
    out = arm.reserve_direct(StubModel(1.05, spread=0.001), tri, tri.n - 1, 2000,
                             np.random.default_rng(0), target="ratio")
    # At the production anchor every origin but the oldest has something left to develop, and each one's
    # horizon is simply its accident-year index.
    assert out["origins"] == list(range(1, tri.n))
    assert out["horizons"] == list(range(1, tri.n))
    assert out["samples"] is not None
    assert np.mean(out["samples"]) == pytest.approx(out["reserve"], rel=0.005), (
        f"point {out['reserve']:,.0f} vs distribution mean {np.mean(out['samples']):,.0f}")


def test_direct_arm_with_a_constant_model_is_arithmetic() -> None:
    """With a model that predicts a constant factor, the reserve is a sum that can be checked by hand --
    which is how the plumbing gets tested without TabPFN."""
    tri = Triangle.load("abc")
    out = arm.reserve_direct(StubModel(1.10), tri, tri.n - 1, 0, np.random.default_rng(0),
                             target="ratio")
    base = np.array([tri.values[a, tri.n - 1 - a] for a in out["origins"]])
    assert out["reserve"] == pytest.approx(float(np.sum(base * 0.10)))



def test_a_single_triangle_still_loads() -> None:
    """The guard against collections must not reject the actual triangles -- it very nearly did."""
    assert Triangle.load("abc").values.shape == (11, 11)


def test_a_collection_of_triangles_is_refused() -> None:
    """`clrd` is 775 triangles. Slicing it to the first member would have reported a reserve for an
    arbitrary line of business with nothing downstream able to notice."""
    with pytest.raises(ValueError, match="collection"):
        Triangle.load("clrd")


def test_a_multi_column_sample_must_name_its_column() -> None:
    """`mcl` carries both incurred and paid. The spike script silently took the first; the choice changes
    the numbers while looking equally legitimate either way, so it has to be said out loud."""
    with pytest.raises(ValueError, match="columns"):
        Triangle.load("mcl")


def test_the_named_column_is_the_one_actually_used() -> None:
    incurred = Triangle.load("mcl", column="incurred")
    paid = Triangle.load("mcl", column="paid")
    assert incurred.column == "incurred" and paid.column == "paid"
    assert not np.allclose(incurred.values, paid.values, equal_nan=True)
    assert incurred.values[0, 0] == 978 and paid.values[0, 0] == 576


@pytest.mark.parametrize("column", ["incurred", "paid"])
def test_the_baseline_is_sliced_to_the_modelled_column(column: str) -> None:
    """The CAS package fits one Chain Ladder per column. If the baseline summed both while the model saw
    one, the comparison would silently be between different triangles."""
    tri = Triangle.load("mcl", column=column)
    anchor = tri.n - 1
    ours = factor_reserve(tri, anchor, tri.global_factors(tri.known(anchor)))
    theirs = chainladder_baseline(tri, anchor)["chainladder_ibnr"]
    assert ours == pytest.approx(theirs, rel=1e-9), f"{column}: {ours} vs {theirs}"
