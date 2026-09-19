"""How many training rows did each fleet fit actually see?

E7 (the conditions check) hinges on scale, and the number the entry had been quoting -- 40-60 training
rows per fit -- came from a single-triangle run, not from the fleet. This recomputes the count from the
fleet's own recorded units: the same triangles, the same column, the same trim, the same anchors, and the
same call into the package's own training-row builder that `scripts/fleet_eval.py` makes.

**Reading and counting only.** No model is constructed, nothing is fitted, nothing is predicted, no file
is written. Run it with the pinned environment:

    .venv/bin/python results/conditions/rows_per_fit.py | tee results/conditions/rows_per_fit.log
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tabpfn_reserving.triangle import Triangle, direct_training_rows  # noqa: E402


def main() -> int:
    import chainladder as cl

    units: list[tuple[str, int, int]] = []
    for line in (ROOT / "results/fleet/clrd-IncurLoss.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        units.append((r["triangle"], int(r["anchor"]), int(r["n"])))

    sample = cl.load_sample("clrd")
    keys = list(sample.key_labels)
    cache = {}
    for i in range(np.asarray(sample.values).shape[0]):
        cache[" / ".join(str(k) for k in keys) + f"#{i}"] = sample.iloc[i]

    counts = []
    unusable = 0
    for name, anchor, n in units:
        sub = cache.get(name)
        if sub is None:
            unusable += 1
            continue
        tri = Triangle.load(sub, column="IncurLoss").trim()
        if tri.n != n:
            unusable += 1
            continue
        # Exactly what fleet_eval.py does for the direct arm, including the label clamp at the anchor.
        X, y = direct_training_rows(tri, list(range(2, max(3, anchor))), target="delta",
                                    known_until=anchor)
        counts.append(len(y))

    c = np.array(counts)
    print("recount of the training rows each fleet fit saw (direct arm, IncurLoss, 464 units)")
    print(f"  units recorded in results/fleet/clrd-IncurLoss.jsonl : {len(units)}")
    print(f"  units recounted                                      : {len(c)}")
    print(f"  unusable                                             : {unusable}")
    print(f"  rows per fit: min {int(c.min())}  p25 {int(np.percentile(c, 25))}  "
          f"median {int(np.median(c))}  p75 {int(np.percentile(c, 75))}  max {int(c.max())}")
    print(f"  units below 20 rows  : {int((c < 20).sum())}")
    print(f"  units above 40 rows  : {int((c > 40).sum())}")
    print(f"  units above 60 rows  : {int((c > 60).sum())}")
    print(f"  histogram (rows: units) : {dict(sorted(Counter(c.tolist()).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
