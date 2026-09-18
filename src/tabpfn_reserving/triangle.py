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

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @classmethod
    def load(cls, name) -> "Triangle":
        """Load a bundled sample by name, or wrap a chainladder triangle object directly.

        The CAS Loss Reserve Database (`clrd`) ships with chainladder, but it is a *collection* -- 775
        triangles across lines of business -- so `load("clrd")` is refused rather than silently reduced to
        its first member. Pass one of its triangles in explicitly.
        """
        import chainladder as cl

        if not isinstance(name, str):
            tri = name
            name = str(getattr(tri, "columns", [getattr(tri, "key_labels", ["unnamed"])])[0])
        else:
            tri = cl.load_sample(name)
        vals = np.asarray(tri.values, dtype=float)
        # chainladder values are (keys, columns, origin, development), so a single triangle keeps two
        # leading dimensions of length 1 and a collection does not. Counting dims alone is not enough:
        # abc is 4-d too, at (1, 1, 11, 11).
        n_slices = int(np.prod(vals.shape[:-2])) if vals.ndim > 2 else 1
        if n_slices > 1:
            key = getattr(tri, "key_labels", None)
            raise ValueError(
                f"{name!r} is a collection of {n_slices} triangles, not one: its values are {vals.shape}"
                + (f", keyed by {list(key)}" if key is not None else "")
                + ". Taking the first slice silently would report a reserve for an arbitrary member of the "
                  "set, and nothing downstream could tell. Pass one triangle in instead -- index the sample "
                  "down to a single triangle and hand that object to Triangle.load()."
            )
        while vals.ndim > 2:
            vals = vals[0]
        return cls(
            name=name,
            values=vals,
            columns=[str(c) for c in tri.columns],
            origin=[str(o)[:10] for o in tri.origin],
            ages=[int(a) for a in tri.development],
            raw=tri,
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

        The tail beyond the last diagonal is unmeasurable from data and is deliberately excluded, which
        is why the classical reserve's own `ibnr` (which includes it) is not used as the head-to-head.
        """
        return float(sum(self.values[a, self.n - 1 - a] - self.values[a, anchor - a]
                         for a in range(anchor + 1)))


FEATURES = ["origin_idx", "dev_idx", "cal_idx", "latest_cum", "log_latest_cum",
            "own_last_ratio", "global_factor_prev", "global_factor_available"]


def gf_used(d: int, anchor: int, gf: np.ndarray) -> tuple[float, float]:
    """The factor this cell is entitled to see, and whether it existed at the anchor.

    Shared by the feature builder and the recursion so the two cannot disagree about what was known.
    """
    j = d - 1
    if j <= anchor and j < len(gf) and np.isfinite(gf[j]):
        return float(gf[j]), 1.0
    return (float(gf[anchor]) if anchor < len(gf) and np.isfinite(gf[anchor]) else 1.0), 0.0


def features_for(tri: Triangle, a: int, d: int, anchor: int, gf: np.ndarray, running: np.ndarray) -> list:
    """Features for the link ratio C[a,d]/C[a,d-1], using only cells known at the anchor.

    `running` is the triangle as projected so far -- observed cells plus anything already predicted.
    """
    n = tri.n
    last_age = min(anchor - a, n - 1)
    latest = running[a, last_age] if last_age >= 0 else np.nan
    prev = running[a, last_age - 1] if last_age >= 1 else np.nan
    own_last_ratio = (latest / prev) if (last_age >= 1 and prev and prev > 0) else np.nan
    gf_val, gf_available = gf_used(d, anchor, gf)
    return [a, d, a + d, latest, np.log1p(max(latest, 0.0)), own_last_ratio, gf_val, gf_available]


def training_rows(tri: Triangle, anchor: int, gf: np.ndarray, target: str = "ratio",
                  shuffle: bool = False, rng: np.random.Generator | None = None):
    """Observed cell -> next cell transitions at the anchor: the link-ratio training set.

    `target="ratio"` learns the raw link ratio. `target="delta"` learns the ratio relative to the
    volume-weighted factor, which turned out to reproduce Chain Ladder rather than improve on it.
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
            X.append(features_for(tri, a, d, anchor, gf, tri.values))
            y.append(ratio)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if shuffle and len(y):
        y = y[rng.permutation(len(y))]
    return X, y


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
    try:
        sub = tri.raw[tri.raw.valuation <= tri.raw.valuation[anchor]]
        out["shape"] = list(np.asarray(sub.values).shape[-2:])
        out["chainladder_ibnr"] = float(np.nansum(np.asarray(cl.Chainladder().fit(sub).ibnr_)))
        mack = cl.MackChainladder().fit(sub)
        out["mack_ibnr"] = float(np.nansum(np.asarray(mack.ibnr_)))
        for attr in ("total_mack_std_err_", "total_process_std_err_", "total_parameter_std_err_"):
            if hasattr(mack, attr):
                out[attr.rstrip("_")] = float(np.nansum(np.asarray(getattr(mack, attr))))
    except Exception as exc:  # noqa: BLE001 -- a baseline failure must not take the arm down
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
