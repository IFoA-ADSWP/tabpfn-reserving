"""Triangles as prediction problems: loading, cells, factors, and the classical baselines.

The reframing lives here. A loss triangle is accident periods by development periods with a lower half
that is unknown -- and that unknown region *is* the reserve. Nothing below treats it as arithmetic on
selected factors, which is what makes it a prediction problem rather than a projection.

Every feature is restricted to what was observable at the valuation date being predicted. `anchor` is
that date; cells with `origin + development > anchor` do not exist as far as anything here is concerned.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Triangle:
    """A cumulative loss triangle, with its provenance."""

    name: str
    values: np.ndarray                      # (N, N) cumulative, NaN where not yet observed
    columns: list[str] = field(default_factory=list)
    origin: list[str] = field(default_factory=list)
    ages: list[int] = field(default_factory=list)
    raw: object = None                      # the chainladder object, for the CAS baselines
    column_index: int = 0                   # which column of a multi-column sample this is
    column: str | None = None

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @classmethod
    def load(cls, name, column=None) -> "Triangle":
        """Load a bundled sample by name, or wrap a chainladder triangle object directly.

        Two ways a sample can be more than one triangle, and they are treated differently:

        * **keys** (e.g. `clrd`, 775 triangles across lines of business) -- always refused. There is no
          principled way to pick one, and silently reporting the first would be a reserve for an arbitrary
          line of business with nothing downstream able to notice.
        * **columns** (e.g. `mcl`, which carries `incurred` and `paid`) -- must be *named*, because the
          choice changes the numbers while looking legitimate either way. `mcl` was silently reduced to its
          first column (`incurred`) by the spike script for exactly this reason.
        """
        import chainladder as cl

        if not isinstance(name, str):
            tri = name
            name = str(getattr(tri, "columns", [getattr(tri, "key_labels", ["unnamed"])])[0])
        else:
            tri = cl.load_sample(name)
        vals = np.asarray(tri.values, dtype=float)
        shape = vals.shape
        n_keys = int(np.prod(shape[:-3])) if vals.ndim > 3 else 1
        n_columns = int(shape[-3]) if vals.ndim >= 3 else 1
        columns = [str(c) for c in getattr(tri, "columns", [])] or [f"column{i}" for i in range(n_columns)]

        if n_keys > 1:
            key = getattr(tri, "key_labels", None)
            raise ValueError(
                f"{name!r} is a collection of {n_keys} triangles, not one: its values are {shape}"
                + (f", keyed by {list(key)}" if key is not None else "")
                + ". Taking the first slice silently would report a reserve for an arbitrary member of the "
                  "set, and nothing downstream could tell. Pass one triangle in instead -- index the sample "
                  "down to a single triangle and hand that object to Triangle.load()."
            )
        if n_columns > 1 and column is None:
            raise ValueError(
                f"{name!r} carries {n_columns} columns ({', '.join(columns)}), and the choice changes the "
                f"numbers. Name one: Triangle.load({name!r}, column={columns[0]!r})."
            )
        col_idx = 0
        if n_columns > 1:
            col_idx = columns.index(column) if isinstance(column, str) else int(column)
        if vals.ndim >= 3:
            vals = vals[..., col_idx, :, :]
        while vals.ndim > 2:
            vals = vals[0]
        return cls(
            name=name,
            values=vals,
            columns=[columns[col_idx]],
            origin=[str(o)[:10] for o in tri.origin],
            ages=[int(a) for a in tri.development],
            raw=tri,
            column_index=col_idx,
            column=columns[col_idx],
        )

    def fingerprint(self) -> dict:
        """A hash of the numbers a result was computed from -- not of 'the database'."""
        a = np.nan_to_num(self.values, nan=-1.0, posinf=-1.0, neginf=-1.0)
        return {
            "name": self.name,
            "shape": list(a.shape),
            "columns": self.columns,
            "sha256_16": hashlib.sha256(np.round(a, 6).tobytes()).hexdigest()[:16],
        }

    def regular_block(self) -> tuple[int, int] | None:
        """The largest contiguous block of origins that forms a *proper* triangle, or None.

        A proper triangle needs each origin's observations to be a gapless prefix of ages, and each
        successive origin to be exactly one age shorter. Returns `(first_origin, size)`.

        This exists because the CAS database ships its triangles in fixed-size containers: a company with
        three accident years occupies the *last* three origins of a 10x10 array and leaves the rest NaN, and
        a company that changed line of business can leave gaps in the middle. Summing over all ten origins
        then returns NaN, and every number downstream becomes silently nothing.
        """
        obs = np.isfinite(self.values)
        n = self.n
        last_age = np.full(n, -1)
        gapless = np.zeros(n, dtype=bool)
        for a in range(n):
            filled = np.flatnonzero(obs[a])
            if len(filled) == 0:
                continue
            last_age[a] = int(filled[-1])
            gapless[a] = len(filled) == last_age[a] + 1 and filled[0] == 0

        best: tuple[int, int] | None = None
        for start in range(n):
            if not gapless[start]:
                continue
            size = 1
            while (start + size < n and gapless[start + size]
                   and last_age[start + size] == last_age[start] - size):
                size += 1
            if size < 3:
                continue
            # A proper triangle of size `size` needs the first origin to reach age `size - 1`.
            if last_age[start] + 1 < size:
                continue
            if best is None or size > best[1]:
                best = (start, size)
        return best

    def trim(self) -> "Triangle":
        """The effective triangle, or raise if there is not one.

        Everything else in this package assumes a proper triangle, and the fleet containers are not one.
        """
        block = self.regular_block()
        if block is None:
            raise ValueError(f"{self.name!r}: no gapless triangular block of at least 3 origins")
        start, size = block
        return Triangle(
            name=self.name,
            values=self.values[start:start + size, :size].copy(),
            columns=list(self.columns),
            origin=self.origin[start:start + size],
            ages=self.ages[:size],
            raw=self.raw,
            column_index=self.column_index,
            column=self.column,
        )

    def known(self, anchor: int) -> np.ndarray:
        """Cells observable at `anchor`: origins 0..anchor, development up to the anchor diagonal."""
        mask = np.zeros_like(self.values, dtype=bool)
        for a in range(self.n):
            for d in range(self.n):
                if a <= anchor and a + d <= anchor and np.isfinite(self.values[a, d]):
                    mask[a, d] = True
        return mask

    def global_factors(self, known: np.ndarray) -> np.ndarray:
        """Volume-weighted age-to-age factors from cells known at the anchor; `gf[j]` maps age j -> j+1.

        Ages beyond the anchor have no observations, so the last available factor is carried forward.
        `gf_available` marks that for the model rather than hiding it.
        """
        n = self.n
        gf = np.full(n, np.nan)
        for j in range(n - 1):
            num = den = 0.0
            for a in range(n):
                if known[a, j] and known[a, j + 1] and self.values[a, j] > 0:
                    num += self.values[a, j + 1]
                    den += self.values[a, j]
            if den > 0:
                gf[j] = num / den
        last = np.nan
        for j in range(n):
            if np.isfinite(gf[j]):
                last = gf[j]
            elif np.isfinite(last):
                gf[j] = last
        return gf

    def actual_future(self, anchor: int) -> float:
        """The observed part of the future: what actually developed after the anchor.

        Only finite contributions are summed, and a NaN is reported as NaN rather than skipped silently --
        an origin with no data after the anchor is not a zero, it is an absence, and averaging the two is how
        a fleet measurement becomes fiction.

        The tail beyond the last diagonal is unmeasurable from data and is deliberately excluded, which is
        why the classical reserve's own `ibnr` (which includes it) is not used as the head-to-head.
        """
        total = 0.0
        for a in range(anchor + 1):
            end, base = self.values[a, self.n - 1 - a], self.values[a, anchor - a]
            if not (np.isfinite(end) and np.isfinite(base)):
                return float("nan")
            total += end - base
        return float(total)


FEATURES = ["origin_idx", "dev_idx", "cal_idx", "latest_cum", "log_latest_cum",
            "own_last_ratio", "global_factor_prev", "global_factor_available"]

# The three columns of FEATURES that are *identifiers* rather than measurements: accident period, development
# period and calendar period. Positions are derived from the names rather than written down, so reordering
# FEATURES cannot silently point the model's column typing at the wrong column. `arm.make_model` declares
# these categorical when asked (#22); `direct_features` appends to FEATURES, so the positions carry over.
IDENTIFIER_FEATURES = ["origin_idx", "dev_idx", "cal_idx"]
IDENTIFIER_INDICES = [FEATURES.index(name) for name in IDENTIFIER_FEATURES]

# Appended only when the arm is allowed to see where it has already pushed the origin. Kept as *extra*
# columns rather than replacing the frozen ones, so the two arms are nested and an improvement (or its
# absence) can be attributed to the information rather than to a change of representation.
RUNNING_FEATURES = ["steps_into_projection", "level_before", "log_level_before",
                    "factor_since_anchor", "step_ratio"]


def gf_used(d: int, anchor: int, gf: np.ndarray) -> tuple[float, float]:
    """The factor this cell is entitled to see, and whether it existed at the anchor.

    Shared by the feature builder and the recursion so the two cannot disagree about what was known.
    """
    j = d - 1
    if j <= anchor and j < len(gf) and np.isfinite(gf[j]):
        return float(gf[j]), 1.0
    return (float(gf[anchor]) if anchor < len(gf) and np.isfinite(gf[anchor]) else 1.0), 0.0


def features_for(tri: Triangle, a: int, d: int, anchor: int, gf: np.ndarray, running: np.ndarray,
                 mode: str = "frozen") -> list:
    """Features for the link ratio C[a,d]/C[a,d-1], using only cells known at the anchor *or already
    projected by this model* -- never the truth.

    `running` is the triangle as projected so far: observed cells plus anything predicted.

    `mode="frozen"` describes the cell with the origin's position at the *anchor*, which is the same for
    every cell of that origin and does not change as the projection proceeds. `mode="running"` adds where
    the origin has actually got to. The difference matters: in frozen mode the model is asked to predict a
    ratio for a cell whose current level it was never shown, so the depth of the projection is invisible to
    it -- suspected cause of the residual over-reserve. See issue #14.
    """
    n = tri.n
    last_age = min(anchor - a, n - 1)
    latest = running[a, last_age] if last_age >= 0 else np.nan
    prev = running[a, last_age - 1] if last_age >= 1 else np.nan
    own_last_ratio = (latest / prev) if (last_age >= 1 and prev and prev > 0) else np.nan
    gf_val, gf_available = gf_used(d, anchor, gf)
    row = [a, d, a + d, latest, np.log1p(max(latest, 0.0)), own_last_ratio, gf_val, gf_available]
    if mode == "running":
        level = running[a, d - 1] if d >= 1 else np.nan
        prev_level = running[a, d - 2] if d >= 2 else np.nan
        anchor_level = running[a, anchor - a] if 0 <= anchor - a < n else np.nan
        steps = d - 1 - (anchor - a)
        factor_since = (level / anchor_level) if (np.isfinite(level) and np.isfinite(anchor_level)
                                                 and anchor_level > 0) else np.nan
        step_ratio = (level / prev_level) if (np.isfinite(prev_level) and prev_level > 0) else np.nan
        row += [steps, level,
                np.log1p(max(level, 0.0)) if np.isfinite(level) else np.nan,
                factor_since, step_ratio]
    return row


def training_rows(tri: Triangle, anchor: int, gf: np.ndarray, target: str = "ratio",
                  shuffle: bool = False, rng: np.random.Generator | None = None,
                  mode: str = "frozen"):
    """Observed cell -> next cell transitions at the anchor: the link-ratio training set.

    `target="ratio"` learns the raw link ratio. `target="delta"` learns the ratio relative to the
    volume-weighted factor, which turned out to reproduce Chain Ladder rather than improve on it.

    Training rows are always observed cells, so in `mode="running"` they are built at depth <= 0 -- the
    model learns from positions where the level is known, and is then asked to predict at positions where
    its own predictions form the level. That train/predict mismatch is the point of the experiment, not an
    oversight, and it is why the arm has to beat a stated bar to ship.
    """
    known = tri.known(anchor)
    X, y = [], []
    for a in range(tri.n):
        for d in range(1, tri.n):
            if not (known[a, d] and known[a, d - 1]):
                continue
            prev_val, cur = tri.values[a, d - 1], tri.values[a, d]
            if prev_val <= 0 or not np.isfinite(cur):
                continue
            ratio = cur / prev_val
            if target == "delta":
                base, _ = gf_used(d, anchor, gf)
                if base <= 0:
                    continue
                ratio = ratio / base
            X.append(features_for(tri, a, d, anchor, gf, tri.values, mode=mode))
            y.append(ratio)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if shuffle and len(y):
        y = y[rng.permutation(len(y))]
    return X, y


# ---------------------------------------------------------------------------------------------
# The direct arm: one prediction per origin, no recursion (#18)
# ---------------------------------------------------------------------------------------------

DIRECT_FEATURES = FEATURES + ["horizon", "log_chainladder_factor"]


def chainladder_factor(gf: np.ndarray, from_age: int, to_age: int) -> float:
    """Chain Ladder's implied development factor between two ages, from the anchor's own factors.

    Mechanical, uses nothing but the anchor's triangle, and is handed to the model as a prior to correct --
    the same role it plays in the delta arm.
    """
    factor = 1.0
    for j in range(from_age, to_age):
        factor *= float(gf[j]) if j < len(gf) and np.isfinite(gf[j]) else 1.0
    return factor


def direct_features(tri: Triangle, a: int, anchor: int, gf: np.ndarray, target_age: int) -> list:
    """Features for the *whole remaining development* of one origin, from the anchor to `target_age`.

    The origin's state is described exactly as the recursive arm describes the next cell it is about to
    predict -- the features are computed at `anchor - a + 1`, the cell that would come next -- and the only
    thing that distinguishes this prediction from that one is `horizon`: how many development steps away the
    target sits. One row, one prediction, one error.
    """
    age = anchor - a
    row = features_for(tri, a, age + 1, anchor, gf, tri.values)
    horizon = max(target_age - age, 0)
    cl = chainladder_factor(gf, age, target_age)
    return row + [horizon, float(np.log(cl)) if cl > 0 else np.nan]


def direct_training_rows(tri: Triangle, anchors, target: str = "delta", known_until: int | None = None):
    """Rows for the direct arm: one per (anchor, origin) pair across every training anchor.

    The horizon is what varies here, and it only varies *across* anchors -- at a fixed anchor every origin
    develops to its own last observed age, so every row from that anchor shares a horizon. Training across
    anchors is therefore the only way the model sees a range of horizons at all, which is what makes it able
    to answer for a horizon production asks about.

    `known_until` is the *evaluation* anchor's diagonal, and it exists to stop a subtle leak: a training row
    from anchor `k'` normally takes its label from each origin's last observed age, which in a backtest lies
    beyond the anchor being evaluated -- so the model would be trained on the tail it is about to be asked to
    predict. Passing `known_until=anchor` clamps every training label to a cell the evaluation anchor could
    already see, and the training horizon becomes the interval between the two valuation dates rather than
    the full run to the ultimate.
    """
    ceiling = (tri.n - 1) if known_until is None else int(known_until)
    X, y = [], []
    for k in anchors:
        gf = tri.global_factors(tri.known(k))
        for a in range(k + 1):
            age = k - a
            target_age = min(tri.n - 1 - a, ceiling - a)
            if target_age <= age:
                continue
            base, end = tri.values[a, age], tri.values[a, target_age]
            if not (np.isfinite(base) and np.isfinite(end) and base > 0):
                continue
            factor = end / base
            row = direct_features(tri, a, k, gf, target_age)
            cl = np.exp(row[-1]) if np.isfinite(row[-1]) else 1.0
            if target == "delta":
                if not (cl > 0):
                    continue
                factor = factor / cl
            X.append(row)
            y.append(factor)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def direct_predict_rows(tri: Triangle, anchor: int, gf: np.ndarray, targets) -> tuple[np.ndarray, list]:
    """The origins to predict at the anchor, and their feature rows."""
    rows, origins = [], []
    for a in range(anchor + 1):
        age = anchor - a
        if targets[a] <= age:
            continue
        rows.append(direct_features(tri, a, anchor, gf, targets[a]))
        origins.append(a)
    return np.asarray(rows, dtype=float), origins


def target_ages(n: int, mode: str) -> list[int]:
    """The development age each origin is projected to, in ONE place, because two copies can disagree.

    `mode="production"` is the reserve being asked for: every origin out to the longest development seen
    anywhere in the triangle. `mode="backtest"` is the only future the data can score: each origin out to
    its own last observed age. A model arm and a baseline arm run in different modes produce two numbers
    that look comparable and are not, which is exactly the bug that returned a reserve of zero.
    """
    if mode == "production":
        return [n - 1] * n
    if mode == "backtest":
        return [n - 1 - a for a in range(n)]
    raise ValueError(f"unknown mode {mode!r}: expected 'production' or 'backtest'")


def factor_reserve(tri: Triangle, anchor: int, gf: np.ndarray, targets: list[int] | None = None) -> float:
    """Chain Ladder arithmetic over exactly the cells the model is scored on -- the fair comparison.

    Defaults to the production convention, the same default as the model arm in `arm.reserve`, so calling
    the two without arguments compares like with like.
    """
    n = tri.n
    tgt = targets if targets is not None else target_ages(n, "production")
    total = 0.0
    for a in range(anchor + 1):
        base = tri.values[a, anchor - a]
        proj = base
        for d in range(anchor - a + 1, tgt[a] + 1):
            proj *= gf[d - 1] if d - 1 < len(gf) else 1.0
        total += proj - base
    return float(total)


def chainladder_baseline(tri: Triangle, anchor: int) -> dict:
    """The CAS package on the as-of triangle.

    Its `ibnr` projects past the last observed diagonal, so it includes a tail nobody can score. It is
    reported as a reference beside the like-for-like factor arm, never as the head-to-head.
    """
    import chainladder as cl

    out: dict = {"note": "ibnr includes the tail beyond the last diagonal; reference only"}

    def pick_column(arr: np.ndarray) -> np.ndarray:
        """Slice the column axis only where there is one. The Mack arrays are not all the same rank as
        `ibnr_`, and indexing a 2-d array with three indices is an IndexError, not a silent no-op."""
        arr = np.asarray(arr).astype(float)
        if arr.ndim >= 3 and arr.shape[-3] > 1:
            return arr[..., tri.column_index, :, :]
        return arr

    try:
        sub = tri.raw[tri.raw.valuation <= tri.raw.valuation[anchor]]
        out["shape"] = list(np.asarray(sub.values).shape[-2:])
        fitted = np.asarray(cl.Chainladder().fit(sub).ibnr_).astype(float)
        if fitted.ndim >= 3 and fitted.shape[-3] > 1:
            out["column"] = tri.column
        out["chainladder_ibnr"] = float(np.nansum(pick_column(fitted)))
        mack = cl.MackChainladder().fit(sub)
        out["mack_ibnr"] = float(np.nansum(pick_column(mack.ibnr_)))
        for attr in ("total_mack_std_err_", "total_process_std_err_", "total_parameter_std_err_"):
            if hasattr(mack, attr):
                out[attr.rstrip("_")] = float(np.nansum(pick_column(getattr(mack, attr))))
    except Exception as exc:  # noqa: BLE001 -- a baseline failure must not take the arm down
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
