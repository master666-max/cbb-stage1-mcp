"""The canonical box_lint script must be findable in a published checkout.

Measured on the cleaned publish tree: **50 tests failed** with

    BuildError: canonical box_lint gate failed / box_lint rc=1

because ``box/lint.py`` went straight to the sibling design tree
(``<parent>/设计/抽取段-v6/tools/box_lint.py``) and a clone has no sibling. The box
writer treats the canonical lint as one of its hard gates, so every build failed.

Same shape of defect as the schemas, same fix: resolve by chain, ship the script with the
code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cbb_stage1 import schemas
from cbb_stage1.box import lint


def _fake_checkout(root: Path, *, script_where: str) -> Path:
    """Build a minimal checkout with box_lint.py in one of the shipped locations."""
    package = root / "src" / "cbb_stage1"
    package.mkdir(parents=True)
    if script_where == "tools":
        target = root / "tools"
    elif script_where == "legacy":
        target = root.parent / "设计" / "抽取段-v6" / "tools"
    else:
        target = package / "box"
    target.mkdir(parents=True, exist_ok=True)
    (target / "box_lint.py").write_text(
        '"""stub"""\n'
        "def main(argv=None):\n"
        "    print('OK：0 个错误')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    schemas_dir = root / "schemas"
    schemas_dir.mkdir()
    (schemas_dir / "assertion.schema.json").write_text("{}", encoding="utf-8")
    return root


def test_the_script_is_found_in_tools(tmp_path: Path) -> None:
    """The published layout: ``<project>/tools/box_lint.py``."""
    project = _fake_checkout(tmp_path / "repo", script_where="tools")
    found = schemas.find_script("box_lint.py", project)
    assert found is not None
    assert found == project / "tools" / "box_lint.py"


def test_the_script_is_found_beside_the_package(tmp_path: Path) -> None:
    """A fallback: shipped inside the package."""
    project = _fake_checkout(tmp_path / "repo", script_where="package")
    found = schemas.find_script("box_lint.py", project)
    assert found is not None
    assert found.parent.name == "box"


def test_the_legacy_sibling_layout_still_resolves(tmp_path: Path) -> None:
    """The in-workspace layout must keep working while the move is happening."""
    project = _fake_checkout(tmp_path / "V5" / "cbb_stage1_v6", script_where="legacy")
    found = schemas.find_script("box_lint.py", project)
    assert found is not None
    assert "设计" in str(found)


def test_absent_returns_none_so_the_caller_can_report_it(tmp_path: Path) -> None:
    """No guessing: the caller decides what to say."""
    project = tmp_path / "empty"
    (project / "src" / "cbb_stage1").mkdir(parents=True)
    assert schemas.find_script("box_lint.py", project) is None


def test_the_box_lint_wrapper_uses_the_resolver(tmp_path: Path) -> None:
    """The wrapper must not go straight to the sibling path any more."""
    project = _fake_checkout(tmp_path / "repo", script_where="tools")
    module, script = lint._load_box_lint(project)
    assert module is not None, f"box_lint not found; looked at {script}"
    assert script == project / "tools" / "box_lint.py"
