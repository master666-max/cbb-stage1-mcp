import pytest
from pathlib import Path

from cbb_stage1.box.lint import lint_box


def _example_dir() -> Path:
    """The example box fixture, wherever the checkout keeps it.

    Not shipped in the public repo: the fixture is built from copyrighted source. There
    the test skips with a named reason instead of going red on something a reader cannot
    act on.
    """
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "example", root.parent / "设计" / "抽取段-v6" / "example"):
        if candidate.is_dir():
            return candidate
    pytest.skip("example box fixture not shipped in this checkout (copyrighted source)")


def test_example_box_passes_lint():
    assert lint_box(_example_dir()) == []
