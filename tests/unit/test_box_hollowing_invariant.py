"""The box gate must catch *partial* hollowing, not just a total loss.

The two table-level guards added earlier catch a box with no knowledge at all,
and a box whose coverage cells are mostly absent. Neither catches the shape the
external audit flagged as B-P2-3: a box that keeps 1 of 9 assertions still has a
non-empty assertions table and a low absent ratio, so it ships green while the
knowledge layer is 8/9 gone.

The per-predicate reconciliation (emitted vs accepted, written by the box writer
into ``qc/coverage.json``) gives the gate an exact number to compare.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
from cbb_stage1.schemas import find_script

BOX_LINT = find_script("box_lint.py", ROOT)

def example_dir() -> Path:
    """The example box fixture, wherever the checkout keeps it.

    Published layout has it at ``<project>/example``; the workspace layout has it under
    ``设计/抽取段-v6/example``. Tests that pinned the second one failed on a clone.

    When it is absent the caller **skips**. The fixture is built from a copyrighted
    novel, so the public repo does not ship it — these assertions are a workspace-only
    check, and saying so out loud is better than a red suite a reader cannot act on.
    """
    for candidate in (
        ROOT / "example",
        ROOT.parent / "设计" / "抽取段-v6" / "example",
    ):
        if candidate.is_dir():
            return candidate
    pytest.skip(
        "example box fixture not shipped in this checkout (built from copyrighted "
        "source); runs in the authoring workspace only"
    )

def _load_box_lint():
    spec = importlib.util.spec_from_file_location("_cbb_box_lint_hollow", BOX_LINT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_lint(box: Path) -> tuple[int, str]:
    module = _load_box_lint()
    import contextlib
    import io
    import sys

    buffer = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = [str(BOX_LINT), str(box)]
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = module.main()
    finally:
        sys.argv = old_argv
    return code, buffer.getvalue()


@pytest.fixture()
def box_with_coverage(tmp_path: Path) -> Path:
    """A copy of the example box plus a qc/coverage.json we can manipulate."""
    target = tmp_path / "box"
    shutil.copytree(example_dir(), target)
    coverage = {
        "book_id": "example",
        "spec_version": "2.2.0",
        "aspects": [],
        "libraries": [],
        # box_lint requires the full 6 libraries x 10 aspects grid.
        "cells": [
            {"library": library, "aspect": aspect, "status": "present", "count": 1}
            for library in ("实体", "关系", "事件", "埋线", "设定", "时序")
            for aspect in (
                "presence", "identity", "attributes", "relations", "events",
                "time", "causal", "motivation", "theme", "style",
            )
        ],
        "counts": {},
        "generated_at": "2026-01-01T00:00:00",
        "predicate_reconciliation": {},
    }
    (target / "qc").mkdir(exist_ok=True)
    (target / "qc" / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def test_full_reconciliation_passes(box_with_coverage: Path) -> None:
    """Every predicate that emitted something also reached the box."""
    coverage_path = box_with_coverage / "qc" / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["predicate_reconciliation"] = {
        "亲属": {"emitted": 1, "accepted": 1},
        "性质": {"emitted": 3, "accepted": 3},
    }
    coverage_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    code, output = _run_lint(box_with_coverage)
    assert code == 0, output


def test_a_predicate_that_emitted_but_never_landed_fails(box_with_coverage: Path) -> None:
    """The partial-hollowing shape: one predicate entirely lost.

    A table-level check cannot see this — the assertions table is still non-empty
    because other predicates landed.
    """
    coverage_path = box_with_coverage / "qc" / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["predicate_reconciliation"] = {
        "亲属": {"emitted": 1, "accepted": 1},
        "性质": {"emitted": 3, "accepted": 0},
    }
    coverage_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    code, output = _run_lint(box_with_coverage)
    assert code != 0, "the gate did not catch a fully-hollowed predicate"
    assert "谓词整类被隔离" in output
    assert "性质" in output


def test_reconciliation_absent_does_not_false_kill(box_with_coverage: Path) -> None:
    """A box built before this field existed must still lint."""
    coverage_path = box_with_coverage / "qc" / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage.pop("predicate_reconciliation", None)
    coverage_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    code, output = _run_lint(box_with_coverage)
    assert code == 0, output
