"""The schema resolver must find the schemas in a published checkout, not only in the
workspace this project grew up in.

The published failure this prevents: clone the repo, run ``build-box``, and get

    BuildError: box schemas not found: <clone>/设计/抽取段-v6/schemas

because the schemas lived in a sibling design tree that a clone does not have.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cbb_stage1 import schemas


def _fake_project(root: Path, *, at_root: bool = False) -> Path:
    """A minimal checkout: ``src/cbb_stage1`` plus schemas somewhere."""
    package = root / "src" / "cbb_stage1"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = root / "schemas" if at_root else package / "schemas"
    target.mkdir(parents=True)
    (target / "assertion.schema.json").write_text("{}", encoding="utf-8")
    return root


def test_the_published_layout_is_found(tmp_path: Path) -> None:
    """``<project>/schemas`` — the layout the release uses."""
    project = _fake_project(tmp_path / "repo", at_root=True)
    found = schemas.schemas_dir(project)
    assert found == project / "schemas"


def test_schemas_beside_the_package_are_found(tmp_path: Path) -> None:
    """``src/cbb_stage1/schemas`` — the other shipped layout."""
    project = _fake_project(tmp_path / "repo", at_root=False)
    found = schemas.schemas_dir(project)
    assert found.name == "schemas"
    assert (found / "assertion.schema.json").is_file()


def test_the_legacy_sibling_layout_still_works(tmp_path: Path) -> None:
    """The in-workspace layout: ``<parent>/设计/抽取段-v6/schemas``.

    This is where the schemas live today; dropping support for it would break the
    working tree while the published layout is being introduced.
    """
    workspace = tmp_path / "V5"
    project = workspace / "cbb_stage1_v6"
    (project / "src" / "cbb_stage1").mkdir(parents=True)
    legacy = workspace.joinpath(*schemas.LEGACY)
    legacy.mkdir(parents=True)
    (legacy / "assertion.schema.json").write_text("{}", encoding="utf-8")

    found = schemas.schemas_dir(project)
    assert found == legacy


def test_a_missing_schema_tree_names_every_path_it_tried(tmp_path: Path) -> None:
    """A broken install must say where it looked, not just that something is missing."""
    project = tmp_path / "empty-repo"
    (project / "src" / "cbb_stage1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        schemas.schemas_dir(project)
    message = str(excinfo.value)
    assert "Looked in" in message
    assert "schemas" in message
