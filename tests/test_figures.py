"""The figure checker: it must be able to fail, and it must pass on these documents.

Two tests, deliberately. The first runs the checker's own control arm (it builds a corpus containing an
unmarked superseded figure and asserts the checker fails on it, and a healthy corpus the checker passes on,
and an empty corpus it reports as blindness rather than success). Without that, a green second test would mean
nothing — a checker that reads nothing passes everything.

Both run in CI via .github/workflows/tests.yml, so a document edit that reintroduces a superseded figure
fails the build rather than waiting to be noticed.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_figures.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECKER), *args], capture_output=True, text=True, cwd=ROOT)


def test_the_checker_can_fail_and_can_pass() -> None:
    """The control arm. If this fails, the checker is not trustworthy and the test below proves nothing."""
    r = run("--self-test")
    assert r.returncode == 0, f"the checker's own control arm failed:\n{r.stdout}\n{r.stderr}"


def test_no_unmarked_superseded_figures_in_the_documents() -> None:
    r = run()
    assert r.returncode == 0, (
        "a superseded figure appears in the documents without being marked as superseded "
        f"(or the checker has gone blind):\n{r.stdout}\n{r.stderr}"
    )
