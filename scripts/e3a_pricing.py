"""Price ONE unit of E3a before anyone designs the 464-unit run (#20, docs/experiments.md 5.2).

Mack is analytical; the ODP bootstrap is a simulation. Both are refitted on the *anchor's* triangle
-- the cells the TabPFN coverage unit was scored on are the cells a+b > anchor -- and both are asked
for the same scalar the coverage file already stores:

    R = sum_i [ C_hat(i, n-1-i) - C(i, anchor - i) ]        (i = 0..anchor)

which is `Triangle.actual_future(anchor)` in distribution. That is a *staircase* target: every origin
projected to its own last observed age, i.e. to the same diagonal. Chain Ladder's own projection is
"every origin to the same development age", so the staircase target is not something a stock
`MackChainladder().fit(asof).ibnr_` can express, and the two are not interchangeable (the second one
projects past the observed square for every origin but the oldest). This script therefore:

  * estimates the anchor's age-to-age factors with chainladder's own `Development` (volume weighted),
    and carries the last estimable factor forward into the transitions past the anchor diagonal --
    the same convention `triangle.global_factors` uses for the TabPFN arm, so all three arms see the
    same information;
  * uses chainladder's `MackChainladder` for the Mack *machinery*, by replicating its per-origin
    recursion (mack.py:_mack_recursion) in numpy and checking the replication reproduces the library's
    own `mack_std_err_` / `total_mack_std_err_` exactly before truncating it at the staircase target;
  * uses chainladder's `BootstrapODPSample` for the bootstrap's parameter resampling and its
    documented gamma process noise (`scale_`), with the per-sim projection done in numpy under the
    same carry-forward convention.

Nothing here launches the 464-unit run. It prices one unit and stops.

    .venv/bin/python scripts/e3a_pricing.py --sims 1000 --out results/runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import chainladder as cl

from fleet_eval import load_fleet
from tabpfn_reserving.triangle import Triangle, factor_reserve, target_ages

# --- APPEND HERE ---
