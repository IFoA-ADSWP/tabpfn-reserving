"""The figure checker: it must be able to fail, and it must pass on these documents.

Two of these tests run the checker as a process, deliberately. The first runs its own control arm (it builds a
corpus containing an unmarked superseded figure and asserts the checker fails on it, and a healthy corpus the
checker passes on, and an empty corpus it reports as blindness rather than success). Without that, a green
document test would mean nothing — a checker that reads nothing passes everything.

The second test distinguishes the checker's two failure classes, because they need opposite responses. Exit
1 is a document regression: a superseded figure is back and the documents need an edit. Exit 2 is blindness:
the corpus came back empty or a canonical figure is missing, and the checker — not the documents — is what
needs fixing. The first version collapsed the two into one assertion message, so a worker running inside a
kanban worktree (where the corpus was empty and the checker blind) was told its document had regressed.

Both run in CI via .github/workflows/tests.yml, so a document edit that reintroduces a superseded figure
fails the build rather than waiting to be noticed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_figures.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECKER), *args], capture_output=True, text=True, cwd=ROOT)


def checker_module():  # noqa: ANN201 - a dynamically loaded module has no importable name to annotate
    """The checker loaded as a module, so its corpus can be inspected rather than only its exit code."""
    spec = importlib.util.spec_from_file_location("check_figures", CHECKER)
    assert spec is not None and spec.loader is not None, f"cannot load {CHECKER} as a module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_checker_can_fail_and_can_pass() -> None:
    """The control arm. If this fails, the checker is not trustworthy and the test below proves nothing."""
    r = run("--self-test")
    assert r.returncode == 0, f"the checker's own control arm failed:\n{r.stdout}\n{r.stderr}"


def test_the_corpus_is_not_empty_wherever_this_suite_runs() -> None:
    """The regression guard for the worktree blindness.

    Inside a kanban worktree every path the corpus patterns match has `.worktrees` in its absolute parts, so
    the exclusion emptied the corpus, the checker returned exit 2, and the test below reported that as a
    superseded figure in the documents. An empty corpus here is a bug in the checker, never a clean bill of
    health — and it is worth its own test, because the failure is otherwise only visible as the absence of a
    test that would have caught something.
    """
    module = checker_module()
    corpus = module.files()
    assert corpus, f"the checker's corpus is empty at {module.ROOT} — it is reading nothing"
    assert all(p.is_file() for p in corpus)
    assert all(module.ROOT in p.parents for p in corpus)
    assert "README.md" in {p.name for p in corpus}


def test_a_sibling_worktree_stays_out_of_the_corpus() -> None:
    """The exclusion must survive the fix, but it must be decided on the path *relative to the corpus root*.

    Deciding it on the absolute parts is what broke the checker: from the main clone no corpus pattern ever
    reaches a `.worktrees/` copy, and from inside a worktree every path looks like one. Relative to the root,
    a sibling worktree's copy of a document is still excluded — so it is not counted twice — while the
    checkout's own documents are read.
    """
    module = checker_module()
    assert module.is_other_checkout(module.ROOT / ".worktrees" / "t_example" / "README.md")
    assert module.is_other_checkout(module.ROOT / "results" / ".worktrees" / "x" / "FINDINGS.md")
    assert not module.is_other_checkout(module.ROOT / "README.md")
    assert not module.is_other_checkout(module.ROOT / "results" / "fleet" / "FINDINGS.md")


def test_no_unmarked_superseded_figures_in_the_documents() -> None:
    r = run()
    assert r.returncode != 2, (
        "the figure checker reports BLINDNESS (exit 2), which is not a document regression: it read no "
        "files, or a canonical figure was not found in the ones it read. The corpus, not the documents, is "
        f"what needs fixing:\n{r.stdout}\n{r.stderr}"
    )
    assert r.returncode == 0, (
        "a superseded figure appears in the documents without being marked as superseded (exit 1, a real "
        f"document regression):\n{r.stdout}\n{r.stderr}"
    )
