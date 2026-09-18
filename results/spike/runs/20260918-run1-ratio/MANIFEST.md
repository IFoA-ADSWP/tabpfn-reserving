# Run manifest — run 1, the raw-ratio arm

> **Retrofit, written 2026-09-18.** Runs 1 and 2 predate the manifest discipline. This file records what is
> known about the run and **flags what is not**, rather than presenting a reconstruction as a record. From
> run 3 onward the harness writes this file itself.

| | |
|---|---|
| run id | `20260918-run1-ratio` |
| command | `.venv/bin/python scripts/spike_e0_e1.py --triangles abc genins mcl ukmotor` |
| arm | recursive link-ratio prediction (target = raw ratio) |
| seed / draws | 0 / 300 |
| ran | 2026-09-18, ~00:13–00:30 BST |
| **code revision** | **Uncommitted at run time.** The harness was committed afterwards as `3f6e1f3`. The working tree was dirty. This is the one thing the retrofit cannot close, and the reason the manifest is now written automatically. |
| environment | tabpfn 9.0.0 · tabpfn-client 0.6.0 · chainladder 0.10.1 · torch 2.14.0 · numpy 2.5.3 · pandas 2.3.3 · scikit-learn 1.9.0 · CPU only, no API calls |
| data | `chainladder` 0.10.1 bundled samples — the CAS Loss Reserve Database lineage. **Not downloaded**; fingerprints below, computed at retrofit time from the same package version |
| evaluations | 11 (4 triangles × 2–4 anchors) |
| outputs | `spike_e0_e1_ratio.csv`, `spike_e0_e1_ratio.json`, `run.log` |

## Data fingerprints

`sha256_16` is taken over the rounded cumulative matrix that actually entered the run — not over "the CAS
database" — so a change of dataset version cannot silently invalidate this result. Where a sample carries
more than one column, the matrix hashed is the **first** column, which is what the harness loads.

| dataset | shape | first column | sha256_16 |
|---|---|---|---|
| abc | 11 × 11 | values | `bc97a934645b8bb0` |
| genins | 10 × 10 | values | `f5a238fd5403e5a6` |
| mcl | 7 × 7 | incurred | `8d05d44fe4541817` |
| ukmotor | 7 × 7 | values | `f7bfffdaaaf49664` |

## Errors

**None recorded — because nothing recorded them.** The script at the time captured only three internal
conditions (`cl_error`, `mack_error`, `draws_error`) and stored them inside the JSON; there was no error list,
and a failure at any point killed the whole run and wrote nothing.

That is not hypothetical. Two earlier attempts at this run produced **no artifacts at all**:
`attempt-superseded.log` is what survives of one of them — it shows two completed evaluations and then stops,
because the process was superseded rather than because it finished. The other died outright on a
`chainladder` `Triangle` constructor error, and its traceback exists only in this conversation, nowhere in the
repository.

The harness now: records every failure as `{where, type, message}` in the manifest, keeps the run alive
around a failed arm, fingerprints the data it used, and stamps the git revision. Run 3 is the first run
covered by that discipline.
