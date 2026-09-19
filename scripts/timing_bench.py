"""E5: clean timing measurement (#9).

Measures the three figures the project quotes — **fit seconds**, **predict seconds**, and the **total
wall-clock of a production run on `abc`** — for the recursive arm and the direct arm, repeated so that a
*spread* is reported instead of a single figure.

    python scripts/timing_bench.py                 # refuses unless the machine is idle
    python scripts/timing_bench.py --repeats 3

It **refuses** rather than warns. Every wall-clock in this repository before #9 was taken with other runs in
flight on the same machine, and the whole point of the exercise is that a loaded machine cannot support a
speed claim. The guard is the repository's own idleness criterion rather than a judgement call:

* `pgrep -fl 'tabpfn_reserving|fleet_'` must be empty — no other run of *this* project;
* the 1-minute load average must be below `--max-load` — no saturation from anything else, and the top CPU
  consumers are printed so a refusal says what is in the way;
* with `--check-tests`, the test suite is timed against the ~8 s baseline `docs/readiness.md` quotes.

Nothing here touches the model: every number comes from the CLI, so the code measured is the code shipped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

# The arms the project quotes, and the production command for each. The `abc` anchors are deliberate: they
# are the numbers in the README and in results/runs/direct-arm.md.
ARMS: dict[str, list[str]] = {
    "recursive": ["abc", "--distribution"],
    "direct": ["abc", "--direct", "--target", "delta", "--distribution"],
}

FIT_PREDICT = re.compile(r"fit (\d+\.\d+)s\s+predict (\d+\.\d+)s")


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def machine() -> dict:
    """Everything a reader needs to know what the numbers are numbers *of*."""
    batt = sh("pmset -g batt").replace("\n", " | ")
    low = sh("pmset -g | grep lowpowermode").split()[-1] if sh("pmset -g | grep lowpowermode") else "?"
    load = sh("uptime").split("load averages:")[-1].strip()
    return {
        "cpu": sh("sysctl -n machdep.cpu.brand_string"),
        "ncpu": sh("sysctl -n hw.ncpu"),
        "physical_cpu": sh("sysctl -n hw.physicalcpu"),
        "ram_gib": round(int(sh("sysctl -n hw.memsize")) / 2**30, 1),
        "platform": sh("sw_vers -productVersion"),
        "power": batt,
        "on_mains": "AC Power" in batt,
        "low_power_mode": low,
        "load_average": load,
        "top_cpu": sh("ps -Ao %cpu,pid,comm -r | head -6"),
    }


def other_runs() -> str:
    return sh("pgrep -fl 'tabpfn_reserving|fleet_'")


def check_idle(max_load: float, check_tests: bool, allow_busy: bool) -> dict:
    """The gate. Returns the machine record; exits when the machine is not quiet, unless overridden."""
    m = machine()
    busy = other_runs()
    load1 = float(m["load_average"].split()[0])

    problems = []
    if busy:
        problems.append(f"another run of this project is in flight:\n{busy}")
    if load1 > max_load:
        problems.append(f"1-minute load average {load1} exceeds --max-load {max_load} "
                        f"({m['ncpu']} cores); top consumers:\n{m['top_cpu']}")
    if not m["on_mains"]:
        # Not fatal by itself — the number is recorded either way — but a timing claim taken on a
        # discharging battery in low power mode is a claim about the worst case, not about the machine.
        problems.append(f"on battery, not mains, low_power_mode={m['low_power_mode']}: {m['power']}")
    if check_tests:
        t0 = time.perf_counter()
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], cwd=REPO,
                             capture_output=True, text=True)
        dt = time.perf_counter() - t0
        m["test_suite_seconds"] = round(dt, 1)
        m["test_suite_line"] = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if dt > 12.0:
            problems.append(f"the test suite took {dt:.1f}s against the ~8s baseline "
                            f"(results/runs/**: 32 tests, no token, no model fit)")

    if problems and not allow_busy:
        print("REFUSING to measure: the machine is not idle.\n")
        for p in problems:
            print(f"- {p}\n")
        print("A timing taken now would replace one contaminated number with another. Free the machine "
              "(quit other builds and agents, plug into mains) and re-run; `--allow-busy` records a loaded "
              "measurement deliberately, and labels it as one.")
        sys.exit(3)
    if problems:
        print("WARNING: measuring on a busy machine because --allow-busy was given.\n")
        for p in problems:
            print(f"- {p}\n")
    return m


def one_run(arm: str, args: list[str], tag: str, scratch: pathlib.Path) -> dict:
    cmd = [sys.executable, "-m", "tabpfn_reserving", *args, "--json", str(scratch / f"{arm}-{tag}.json")]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    total = time.perf_counter() - t0
    match = FIT_PREDICT.search(proc.stdout)
    record = json.loads((scratch / f"{arm}-{tag}.json").read_text())
    row = {
        "arm": arm,
        "run": tag,
        "command": "python -m tabpfn_reserving " + " ".join(shlex.quote(a) for a in args),
        "fit_seconds": record["fit_seconds"],
        "predict_seconds": record["predict_seconds"],
        "total_seconds": round(total, 1),
        "reserve": record["reserve"],
        "returncode": proc.returncode,
        "printed_fit_predict": match.group(0) if match else None,
    }
    print(f"  {arm:<9} {tag}: fit {row['fit_seconds']:6.2f}s  predict {row['predict_seconds']:6.2f}s  "
          f"total {row['total_seconds']:6.1f}s  reserve {row['reserve']:>13,.0f}  rc={proc.returncode}")
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
    return row


def spread(values: list[float]) -> dict:
    lo, hi = min(values), max(values)
    mid = sorted(values)[len(values) // 2]
    return {"min": round(lo, 2), "median": round(mid, 2), "max": round(hi, 2),
            "spread_pct": round(100 * (hi - lo) / mid, 1) if mid else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3, help="runs per arm, reported as a spread")
    ap.add_argument("--arms", nargs="*", choices=sorted(ARMS), default=sorted(ARMS))
    ap.add_argument("--max-load", type=float, default=1.0,
                    help="refuse above this 1-minute load average (default: 1.0)")
    ap.add_argument("--warmup", action="store_true", default=True,
                    help="one discarded run per arm first, so weights and threads are warm")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--check-tests", action="store_true", help="also time the test suite as an idle probe")
    ap.add_argument("--allow-busy", action="store_true", help="measure anyway, labelled as loaded")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results" / "runs" / "e5-timings.json")
    ap.add_argument("--scratch", type=pathlib.Path, default=pathlib.Path("/tmp/e5-timing-scratch"))
    args = ap.parse_args(argv)

    before = check_idle(args.max_load, args.check_tests, args.allow_busy)
    args.scratch.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm in args.arms:
        if args.warmup:
            print(f"{arm}: warm-up run (discarded -- weights cached, thread pools sized)")
            one_run(arm, ARMS[arm], "warmup", args.scratch)
        for i in range(1, args.repeats + 1):
            rows.append(one_run(arm, ARMS[arm], str(i), args.scratch))

    after = machine()
    summary = {arm: {k: spread([r[k] for r in rows if r["arm"] == arm])
                     for k in ("fit_seconds", "predict_seconds", "total_seconds")}
               for arm in args.arms}
    reserves = {arm: sorted({r["reserve"] for r in rows if r["arm"] == arm}) for arm in args.arms}

    print("\n== spread over {} repeats ==".format(args.repeats))
    for arm, s in summary.items():
        print(f"{arm:<9} fit {s['fit_seconds']['min']}-{s['fit_seconds']['max']}s "
              f"({s['fit_seconds']['spread_pct']}%)   "
              f"predict {s['predict_seconds']['min']}-{s['predict_seconds']['max']}s "
              f"({s['predict_seconds']['spread_pct']}%)   "
              f"total {s['total_seconds']['min']}-{s['total_seconds']['max']}s "
              f"({s['total_seconds']['spread_pct']}%)")
        if len(reserves[arm]) != 1:
            print(f"  WARNING: {arm} is not deterministic -- {len(reserves[arm])} distinct reserves "
                  f"{reserves[arm]}; the fit/predict times must be reported as unstable, not averaged")
        else:
            print(f"  deterministic: every repeat returned {reserves[arm][0]:,.0f}")

    result = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "repeats": args.repeats,
              "arms": {k: ["python", "-m", "tabpfn_reserving", *v] for k, v in ARMS.items()},
              "machine_before": before, "machine_after": after, "runs": rows,
              "spread": summary, "reserves": reserves,
              "venv_python": sys.executable, "tabpfn": _version("tabpfn"),
              "loaded_run": bool(args.allow_busy)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


def _version(pkg: str) -> str:
    from importlib.metadata import version
    try:
        return version(pkg)
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    raise SystemExit(main())
