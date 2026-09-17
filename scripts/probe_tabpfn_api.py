#!/usr/bin/env python
"""Probe: does TabPFN-3.5 load locally, and what shape is `output_type="full"`?

This is the day-one question in docs/experiments.md (spike gate 2). It is deliberately tiny: fit on a
handful of rows, predict, and print exactly what the distribution object is rather than assuming.
Run: .venv/bin/python scripts/probe_tabpfn_api.py
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "spike"
OUT.mkdir(parents=True, exist_ok=True)
report: dict = {}
rng = np.random.default_rng(0)

try:
    import tabpfn
    report["tabpfn_version"] = getattr(tabpfn, "__version__", "unknown")
    report["tabpfn_path"] = str(Path(tabpfn.__file__).parent)
except Exception as exc:
    print("FATAL import:", traceback.format_exc())
    (OUT / "probe.json").write_text(json.dumps({"import_error": traceback.format_exc()}, indent=2))
    raise SystemExit(2)

from tabpfn import TabPFNRegressor  # noqa: E402

# Tiny regression problem: y = 2*x0 + noise, so the model has something to learn and we can check the
# distribution is not degenerate.
n, p = 60, 4
X = rng.normal(size=(n, p))
y = 2.0 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.2, size=n)
Xq = rng.normal(size=(5, p))

model = TabPFNRegressor(device="cpu")
t0 = time.time()
try:
    model.fit(X, y)
    report["fit_seconds"] = time.time() - t0
    report["fit_ok"] = True
except Exception:
    report["fit_ok"] = False
    report["fit_error"] = traceback.format_exc()
    print(json.dumps(report, indent=2))
    (OUT / "probe.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(3)

t0 = time.time()
point = model.predict(Xq)
report["predict_seconds"] = time.time() - t0
report["point"] = np.asarray(point).ravel().tolist()
print("point:", np.round(report["point"], 4).tolist())

# The distribution. Print the structure rather than guessing it.
try:
    full = model.predict(Xq, output_type="full")
    report["full_type"] = type(full).__name__
    if isinstance(full, dict):
        report["full_keys"] = list(full.keys())
        for k, v in full.items():
            entry = {"type": type(v).__name__}
            try:
                a = np.asarray(v)
                entry["shape"] = list(a.shape)
                entry["dtype"] = str(a.dtype)
                if a.dtype.kind in "fiu" and a.size <= 40:
                    entry["values"] = np.round(a, 6).tolist()
                elif a.dtype.kind in "fiu":
                    entry["head"] = np.round(a.ravel()[:12], 6).tolist()
                    entry["min"] = float(np.nanmin(a))
                    entry["max"] = float(np.nanmax(a))
            except Exception as exc:
                entry["array_error"] = f"{type(exc).__name__}: {exc}"
            report.setdefault("full_contents", {})[k] = entry
    else:
        report["full_repr"] = repr(full)[:2000]
        report["full_dir"] = [d for d in dir(full) if not d.startswith("_")][:40]
except Exception:
    report["full_error"] = traceback.format_exc()
print("full:", report.get("full_type"), report.get("full_keys") or report.get("full_error", "")[:200])

# Also record the constructor signature/target so the arms can be built correctly.
try:
    from tabpfn.model_loading import ModelVersion  # may not exist in every release
    report["model_version_enum"] = [m.name for m in ModelVersion]
except Exception as exc:
    report["model_version_enum"] = f"{type(exc).__name__}: {exc}"

(OUT / "probe.json").write_text(json.dumps(report, indent=2, default=str))
print("\nwrote", OUT / "probe.json")
