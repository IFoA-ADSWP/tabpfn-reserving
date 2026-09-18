# Run manifest — run 2, the Δ (deviation-from-Chain-Ladder) arm

> **Retrofit, written 2026-09-18.** Same caveat as run 1: the discipline did not exist when this ran, so this
> records what is known and flags what is not.

| | |
|---|---|
| run id | `20260918-run2-delta` |
| command | `.venv/bin/python scripts/spike_e0_e1.py --triangles abc genins mcl ukmotor --target delta` |
| arm | recursive prediction of the ratio **relative to** the volume-weighted factor at that age |
| seed / draws | 0 / 300 |
| ran | 2026-09-18, ~00:35–00:52 BST |
| **code revision** | **Uncommitted at run time.** The `--target delta` arm was committed afterwards as `e7ff017`. Working tree dirty. |
| environment | tabpfn 9.0.0 · tabpfn-client 0.6.0 · chainladder 0.10.1 · torch 2.14.0 · numpy 2.5.3 · pandas 2.3.3 · scikit-learn 1.9.0 · CPU only, no API calls |
| data | identical to run 1 — same four triangles, same fingerprints (see run 1's manifest). Same folds, same anchors, same seed: **only the target changed**, which is what makes the two runs comparable |
| evaluations | 11 |
| outputs | `spike_e0_e1_delta.csv`, `spike_e0_e1_delta.json`, `run.log` |

## Errors

None raised during the run; as with run 1, nothing would have been stored had one occurred. The results
themselves contain the notable *negative* finding, which is stored as data rather than as an error: the arm
reproduces the Chain Ladder arm to 1.86% at the median (see `../FINDINGS.md`).
