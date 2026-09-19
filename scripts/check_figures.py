#!/usr/bin/env python3
"""Fail when a superseded figure appears in the documents without being marked as superseded.

Why this exists. Twice in one working day the entry quoted a number that was true of *one* run, then repeated
it across documents until nobody re-read it: the training rows per fit ("40-60", from single-triangle runs;
the fleet recount gives 6-33) and "no feature engineering" (the features include hand-chosen domain ratios).
Vigilance caught both, late. This makes the sweep mechanical.

Two failure classes, with different exit codes, because they need different responses:

  1  a superseded figure appears on a line with no marker  -> the documents regressed
  2  a canonical figure was not found at all, or no files were read -> the CHECKER is blind

Exit 2 exists because a check that cannot fail is indistinguishable from no check: if the patterns stop
matching because the documents were renamed, or the corpus came back empty, that must read as an error and
never as a pass. Same reason `--self-test` exists: it builds a corpus in which the checker MUST fail, and one
in which it MUST pass, and reports both, so "it passed" is never taken on trust.

Markers that license a superseded figure to appear (case-insensitive): correct, supersede, was wrong,
not true, stops being, historical, no longer, earlier claim.

What the corpus is. The checkout this copy of the script belongs to — resolved through git rather than
through the file's position, so it is right in a kanban worktree too (see `checkout_root`). A worktree run
scans the worktree's own documents, which are the ones that run is editing; the exclusion in
`is_other_checkout` keeps a *sibling* worktree's copy of the same documents out of the corpus.

Usage:
    python scripts/check_figures.py              # scan this checkout
    python scripts/check_figures.py --self-test  # control arm: prove it can fail and can pass
    python scripts/check_figures.py --list       # show what it holds, and where each figure came from
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
CHECKOUT = HERE.parent
WORKTREES = ".worktrees"


def checkout_root() -> pathlib.Path:
    """The root of the git checkout this copy of the script belongs to.

    Not `__file__.parents[1]` alone: under the kanban dispatcher the script runs from a linked worktree at
    `<repo>/.worktrees/<task-id>/scripts/`, where a root derived from the file's position is whatever depth
    the file happens to sit at. Asking git names the checkout explicitly, and inside a worktree the answer is
    the worktree — which is the corpus that run is about, the branch the worker is editing. (Resolving
    instead to the clone that *hosts* the worktree would read another branch's documents and pass the very
    document edit the check exists to catch.) Falls back to the file's own position when git is missing or
    this is not a checkout.
    """
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return CHECKOUT
    top = out.stdout.strip()
    return pathlib.Path(top) if top and pathlib.Path(top).is_dir() else CHECKOUT


ROOT = checkout_root()

# (pattern, minimum occurrences, what it is and where the canonical value lives)
CANONICAL = [
    (r"6[–-]33", 1, "training rows per fit over the fleet — results/conditions/rows_per_fit.log"),
    (r"median 25", 1, "the median rows per fit — same record"),
    (r"38\.8%", 1, "fraction of fleet units where the model beat Chain Ladder — results/fleet/FINDINGS.md"),
    (r"162\.6%", 1, "the model's median |error| — same"),
    (r"121\.3%", 1, "Chain Ladder's median |error| — same"),
    (r"39\.2%", 1, "coverage at nominal 50% — results/fleet/COVERAGE.md"),
    (r"84\.7%", 1, "coverage at nominal 95% — same"),
    (r"14\.7%", 1, "units above the model's own 95% upper bound, on the 231-unit analysis half — results/fleet/CALIBRATION.md"),
    (r"13\.8%", 1, "the same rate over all 464 units, the canonical sample — results/fleet/COVERAGE.md"),

    (r"5,211,802", 1, "the direct arm's reserve on abc — results/runs/direct-arm.md"),
]

# (pattern, why it is superseded, what replaced it)
SUPERSEDED = [
    (r"40[–-]60 (training )?rows", "a single-triangle figure", "6-33 rows, median 25"),
    (r"median 90% interval is 4\.7", "an earlier width ratio", "4.55x the point estimate"),
    (r"60\.9% at 75", "the interim 111-unit coverage read", "59.9% over 464 units"),
    # The CLAIM, not the phrase: quoting "no feature engineering" while describing it, or in the pre-registration
    # that set out to test it, is correct. Asserting it is what was wrong.
    (r"no tuning, no feature engineering|no feature engineering, raw values",
     "our features are hand-chosen domain ratios", "no tuning, nothing fitted per triangle"),
    (r"closer than Chain Ladder on 0\.0%", "the null whose error was 0.0 by construction", "recomputed from the raw JSONL"),
]

# A line carrying one of these words is allowed to quote a superseded figure. A correction is often announced
# in a heading one line above the quote, so the window is the line plus its neighbours rather than the line alone.
MARKER = re.compile(r"(?i)correct|supersede|was wrong|not true|stops being|stop saying|historical|no longer|"
                    r"earlier claim|recount|replaced")
WINDOW = 2

CORPUS = ["README.md", "docs/*.md", "results/*.md", "results/**/*.md"]


def _rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:  # a corpus built by the self-test lives outside the checkout
        return str(p)


def is_other_checkout(p: pathlib.Path) -> bool:
    """True when p is the documents inside *another* checkout of this repository (a sibling worktree).

    Measured relative to ROOT, which is the whole point. From the main clone the corpus patterns are
    root-relative and never reach into `.worktrees/`, but from inside a worktree *every* matched path has
    `.worktrees` in its absolute parts, so the exclusion emptied the corpus and the checker reported exit 2
    — blindness — which the test read as a document regression. Relative to ROOT a worktree's own documents
    are just `README.md` and `results/…`, so they are read; a nested `.worktrees/…` path, reachable only
    from the clone that hosts a worktree, is still excluded so the same document is not counted twice.
    """
    try:
        rel = p.relative_to(ROOT)
    except ValueError:  # a corpus built by the self-test lives outside the checkout
        return False
    return WORKTREES in rel.parts


def files() -> list[pathlib.Path]:
    seen, out = set(), []
    for pat in CORPUS:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file() and not is_other_checkout(p) and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def scan(paths: list[pathlib.Path]) -> tuple[int, list[str]]:
    """Returns (exit_code, report lines)."""
    out: list[str] = []
    if not paths:
        return 2, [f"ERROR: no files matched the corpus under {ROOT} — the checker is reading nothing, "
                   "which is not a pass."]

    out.append(f"scanned {len(paths)} file(s) under {ROOT}: " + ", ".join(_rel(p) for p in paths))


    # Class 2 first: blindness.
    blind = []
    for pattern, minimum, what in CANONICAL:
        rx = re.compile(pattern)
        found = sum(len(rx.findall(p.read_text(errors="replace"))) for p in paths)
        if found < minimum:
            blind.append(f"  ERROR: canonical figure not found ({found} < {minimum}): /{pattern}/ — {what}")
    if blind:
        out += ["", "CANONICAL FIGURES MISSING — the documents may have been reorganised, or the figure was", "deleted. Either way this is an ERROR, not a pass:"] + blind
        return 2, out

    # Class 1: regressions.
    out.append(f"canonical figures: all {len(CANONICAL)} present")
    offenders = []
    for p in paths:
        lines = p.read_text(errors="replace").splitlines()
        for pattern, why, replaced_by in SUPERSEDED:
            rx = re.compile(pattern)
            for n, line in enumerate(lines, 1):
                if not rx.search(line):
                    continue
                context = "\n".join(lines[max(0, n - 1 - WINDOW): n + WINDOW])
                if MARKER.search(context):
                    continue
                offenders.append((p, n, line.strip(), why, replaced_by))

    if offenders:
        out += ["", f"SUPERSEDED FIGURES NOT MARKED ({len(offenders)}) — add a marker word (corrected / superseded /", "was wrong / not true / historical) to the line, or remove the figure:"]
        for p, n, line, why, replaced_by in offenders:
            out.append(f"  {_rel(p)}:{n}  [{why} -> {replaced_by}]")
            out.append(f"      {line[:160]}")
        return 1, out

    out.append(f"superseded figures: all {len(SUPERSEDED)} patterns checked, none unmarked")
    return 0, out


def self_test() -> int:
    """Control arm. Builds corpora the checker MUST fail on and MUST pass on, and reports both."""
    ok = True
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)

        # A corpus that would pass if the checker were awake: canonical figures present, superseded marked.
        good = d / "good.md"
        good.write_text(
            "6–33 rows, median 25\n38.8% closer\n162.6% and 121.3%\n39.2% and 84.7%\n14.7% above\n"
            "13.8% over all units\n5,211,802\n"
            "This corrects an earlier claim of 40–60 training rows.\n"
        )
        # A corpus that MUST fail: otherwise healthy -- every canonical figure present -- with ONE unmarked
        # superseded figure. It has to be otherwise healthy, or its exit code means "the checker is blind"
        # rather than "the checker caught the regression", and the control tests nothing.
        bad = d / "bad.md"
        bad.write_text("6–33 rows, median 25\n38.8%\n162.6%\n121.3%\n39.2%\n84.7%\n14.7%\n13.8%\n5,211,802\n"
                       "The model gets 40–60 training rows per fit and needs no feature engineering.\n")
        for label, corpus in (("healthy", good), ("must-fail", bad)):
            missing = [pat for pat, _m, _w in CANONICAL if not re.search(pat, corpus.read_text())]
            if missing:
                print(f"  SELF-TEST CORPUS IS STALE -- the {label} corpus does not exercise every canonical figure:")
                for pat in missing:
                    print(f"    /{pat}/")
                print("  Add each missing figure to that corpus above; a control arm that cannot pass is as"
                      "\n  useless as one that cannot fail, and the two look identical from outside.")
                return 2
        # A corpus that MUST read as the checker being blind: canonical figures absent.
        empty = d / "empty.md"
        empty.write_text("nothing that matches any pattern\n")

        cases = [("healthy corpus", [good], 0), ("unmarked superseded figure", [bad], 1),
                 ("blind checker (no canonical figures)", [empty], 2), ("no files at all", [], 2)]
        for label, paths, expected in cases:
            code, _ = scan(paths)
            verdict = "ok" if code == expected else f"WRONG (expected {expected})"
            if code != expected:
                ok = False
            print(f"  {label:<38} exit {code}  {verdict}")
    print(f"\nself-test: {'PASS — the checker can fail and can pass' if ok else 'FAIL — the checker is not trustworthy'}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="control arm: prove the checker can fail and pass")
    ap.add_argument("--list", action="store_true", help="show the figures held, and their sources")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.list:
        print(f"checkout: {ROOT}")
        print("canonical (must be present):")
        for pattern, minimum, what in CANONICAL:
            print(f"  /{pattern}/  >= {minimum}  {what}")
        print("\nsuperseded (only on a marked line):")
        for pattern, why, replaced_by in SUPERSEDED:
            print(f"  /{pattern}/  {why}  ->  {replaced_by}")
        return 0

    code, report = scan(files())
    print("\n".join(report))
    return code


if __name__ == "__main__":
    sys.exit(main())
