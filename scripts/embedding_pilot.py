"""Can the model's own embeddings flag where its distribution is untrustworthy? (#23, pilot)

The vendor's documentation says out-of-distribution inputs are not flagged: "bucket boundaries are fixed at
training time and the model does not flag OOD inputs automatically". Our sharpest measured defect is exactly
that, and it is one-sided -- 14.7% of units have their truth above the model's own 95% upper bound against a
nominal 2.5%, while 0.9% fall below (`results/fleet/CALIBRATION.md`).

So: does an embedding-space score for a unit's *query rows* predict whether that unit's interval misses?

**Paired by construction.** The units, anchors, features, target and fit are rebuilt exactly as
`scripts/fleet_coverage.py` builds them, and the outcome is *joined from* `results/fleet/coverage.jsonl` rather
than recomputed. So the scores are attached to the same 464 units whose coverage is already published.

Scores per unit, from one fit that already happens:
  knn       mean distance from each query row's embedding to the nearest context-row embedding
  centroid  distance between the query centroid and the context centroid
  member    spread across the 8 ensemble members -- an epistemic-uncertainty proxy, free to collect

Reads only; nothing about the model, the data or the earlier results is changed.

    python scripts/embedding_pilot.py --limit 100        # pilot: screen before estimating
    python scripts/embedding_pilot.py --analyse          # test the scores against the recorded misses

Two house rules this file is written to respect: the scores use **features only**, never the realised target
(the outcome is joined afterwards, from a file), and every interval reported carries its permutation control --
`CALIBRATION.md` records a placebo that provably could not discriminate, and this does not repeat that mistake.
Resumable: completed units are skipped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tabpfn_reserving import arm
from tabpfn_reserving.triangle import (Triangle, direct_predict_rows, direct_training_rows, target_ages)
from fleet_eval import load_fleet

COVERAGE = pathlib.Path("results/fleet/coverage.jsonl")
OUT = pathlib.Path("results/runs/20260919-embedding-pilot")
SCORES = OUT / "scores.jsonl"
VECTORS = OUT / "pooled_vectors.npz"


def unit_scores(Etr: np.ndarray, Eq: np.ndarray) -> dict:
    """Etr, Eq: (members, rows, dim). Standardise on the context, then measure distance."""
    tr, q = Etr.mean(axis=0), Eq.mean(axis=0)
    mu, sd = tr.mean(axis=0), tr.std(axis=0) + 1e-6
    tr_s, q_s = (tr - mu) / sd, (q - mu) / sd
    dist = np.linalg.norm(q_s[:, None, :] - tr_s[None, :, :], axis=-1)      # (n_q, n_tr)
    return {
        "knn": float(dist.min(axis=1).mean()),
        "centroid": float(np.linalg.norm(q_s.mean(axis=0) - tr_s.mean(axis=0))),
        "member": float(Eq.std(axis=0).mean()),
        "n_train_rows": int(Etr.shape[1]), "n_query_rows": int(Eq.shape[1]),
    }


def collect(limit: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for line in COVERAGE.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            outcomes[r["unit"]] = r
    done = set()
    if SCORES.exists():
        for line in SCORES.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["unit"])
    print(f"embedding pilot: {len(outcomes)} recorded coverage units, {len(done)} already scored")

    by_name = {name: sub for name, sub in load_fleet(0)}
    vectors: dict[str, np.ndarray] = {}
    if VECTORS.exists():
        vectors = dict(np.load(VECTORS, allow_pickle=False))

    todo = [u for u in outcomes if u not in done]
    if limit:
        todo = todo[:limit]
    n, t0 = 0, time.time()
    with SCORES.open("a") as fh:
        for unit in todo:
            try:
                column, k, name = unit.split("|")
                anchor = int(k.split("=")[1])
                tri = Triangle.load(by_name[name], column=column).trim()
                train = list(range(2, max(3, anchor)))
                X, y = direct_training_rows(tri, train, target="delta", known_until=anchor)
                if len(y) < 5:
                    continue
                model = arm.make_model("local")
                model.fit(X, y)
                gf = tri.global_factors(tri.known(anchor))
                Xq, _ = direct_predict_rows(tri, anchor, gf, target_ages(tri.n, "backtest"))
                Etr = model.get_embeddings(X, data_source="train")
                Eq = model.get_embeddings(Xq, data_source="test")
            except Exception as exc:  # noqa: BLE001 -- one unit must not stop the pilot
                print(f"    skipped {unit}: {type(exc).__name__}: {exc}")
                continue
            rec = {"unit": unit, "triangle": name, "anchor": anchor,
                   "horizon": int(tri.n - 1 - anchor), **unit_scores(Etr, Eq)}
            # the outcomes, joined from the published coverage run -- never recomputed here
            for key in ("actual", "point", "median", "chainladder"):
                rec[key] = float(outcomes[unit][key])
            for key in ("covered_50", "covered_75", "covered_90", "covered_95"):
                rec[key] = bool(outcomes[unit][key])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            vectors[unit] = np.concatenate([Etr.mean(axis=0).ravel().astype(np.float32),
                                            Eq.mean(axis=0).ravel().astype(np.float32)])
            n += 1
            if n % 10 == 0:
                print(f"  {n} scored | {(time.time() - t0) / 60:.1f} min | {name[:34]} k={anchor}")
    if vectors:
        np.savez_compressed(VECTORS, **vectors)
    print(f"\nscored {n} units in {(time.time() - t0) / 60:.1f} min -> {SCORES}")
    return 0


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(score of a missed unit > score of a covered unit) -- rank-based, no shape assumed."""
    pos, neg = x[y == 1], x[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def analyse(level: int = 95, n_perm: int = 2000, seed: int = 0) -> int:
    rows = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    if not rows:
        print(f"nothing scored yet ({SCORES}); run the collect step first")
        return 1
    miss = np.array([not r[f"covered_{level}"] for r in rows], dtype=int)
    print(f"=== {len(rows)} units, {miss.sum()} miss the {level}% interval "
          f"({100 * miss.mean():.1f}%, nominal {100 - level}%) ===")
    if miss.sum() == 0 or miss.sum() == len(miss):
        print("\n  CANNOT TEST -- no variation in the outcome: "
              f"{miss.sum()} of {len(miss)} units miss, so there is nothing to rank against.")
        print("  This is emphatically NOT evidence of calibration. It is a sample too small or too lucky to")
        print("  answer the question. Score more units; do not read this line as a result.")
        return 2
    rng = np.random.default_rng(seed)
    for score in ("knn", "centroid", "member"):
        x = np.array([r[score] for r in rows])
        a = auc(x, miss)
        null = np.array([auc(rng.permutation(x), miss) for _ in range(n_perm)])
        p = float((np.abs(null - 0.5) >= abs(a - 0.5)).mean())
        print(f"  {score:>9}: AUC {a:.3f}   permutation null 2.5-97.5% "
              f"[{np.percentile(null, 2.5):.3f}, {np.percentile(null, 97.5):.3f}]   p={p:.3f}")
    print("\n  AUC 0.5 = no signal. The permutation control is the test: if the interval excludes the"
          "\n  observed AUC, the score is doing something; if it contains it, this pilot has found nothing.")
    print("  A pilot screens; it does not estimate. Scale only if a score clears the control here.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--level", type=int, default=95)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    return analyse(args.level, args.n_perm) if args.analyse else collect(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
