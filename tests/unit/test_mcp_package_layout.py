"""The packaged layout must resolve its own root, and must ship the box schemas.

Running the installed MCP package surfaced two defects that no test had covered:

  1. **The root is computed one level high.**
     `_TOOLS_DIR = Path(__file__).resolve().parent` then `PROJECT_ROOT = _TOOLS_DIR.parent`
     is correct while the file lives at `<project>/tools/<file>`. The package flattens it to
     `<package>/server.py`, so the same arithmetic yields the *parent of the package*.
     Consequences, all measured from `cbb_doctor` on the real install:
       `tools_dir`   -> <install-dir>\\tools        (absent) -> gate_tools BLOCKED
       `schema_dir`  -> <install-dir>\\设计\\...      (absent) -> schema_dir BLOCKED
       `root`        -> <install-dir>\\src\\cbb_stage1 (absent) -> call_wiring/field_coverage
     Everything keyed off a relative path was silently pointing outside the package.

  2. **The package ships no `schemas/`.**
     `build_publish.py` copies it (`copytree(DESIGN / "schemas", OUT / "schemas")`), the
     package builder never mentions it, so `cbb_lint_box` and `cbb_field_coverage` have no
     schemas to work from however the root resolves.

Both are packaging defects, so both are fixed at the source rather than worked around.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SERVER = PROJECT / "tools" / "mcp_pipeline_server.py"
BUILDER = PROJECT / "tools" / "build_mcp_package.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def server():
    return _load(SERVER, "cbb_mcp_server_layout")


@pytest.fixture(scope="module")
def builder():
    return _load(BUILDER, "cbb_mcp_builder_layout")


# ── 1. root resolution must fit both layouts ─────────────────────────────
def test_the_source_layout_resolves_to_the_project(server, tmp_path: Path) -> None:
    """`<project>/tools/<file>` -> `<project>`."""
    root = tmp_path / "proj"
    (root / "src" / "cbb_stage1").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools" / "mcp_pipeline_server.py").write_text("# stand-in", encoding="utf-8")
    assert server._find_project_root(root / "tools" / "mcp_pipeline_server.py") == root


def test_the_packaged_layout_resolves_to_the_package(server, tmp_path: Path) -> None:
    """`<package>/server.py` -> `<package>`. This is the case that shipped broken."""
    pkg = tmp_path / "cbb-mcp"
    (pkg / "cbb_stage1").mkdir(parents=True)
    (pkg / "server.py").write_text("# stand-in", encoding="utf-8")
    assert server._find_project_root(pkg / "server.py") == pkg


def test_the_packaged_root_makes_the_tools_dir_findable(server, tmp_path: Path) -> None:
    """The blocker, restated as its own assertion: tools/ must land inside the package."""
    pkg = tmp_path / "cbb-mcp"
    (pkg / "cbb_stage1").mkdir(parents=True)
    (pkg / "server.py").write_text("", encoding="utf-8")
    (pkg / "field_coverage.py").write_text("", encoding="utf-8")
    root = server._find_project_root(pkg / "server.py")
    assert (root / "field_coverage.py").is_file()
    assert root != pkg.parent, "must not resolve to the package's parent"


def test_the_packaged_root_puts_schemas_inside_the_package(server, tmp_path: Path) -> None:
    pkg = tmp_path / "cbb-mcp"
    (pkg / "cbb_stage1").mkdir(parents=True)
    (pkg / "schemas").mkdir()
    (pkg / "server.py").write_text("", encoding="utf-8")
    root = server._find_project_root(pkg / "server.py")
    assert (root / "schemas").is_dir()


def test_resolution_never_escapes_to_a_parent_that_lacks_the_package(server, tmp_path: Path) -> None:
    """With no marker anywhere, the answer must be the start directory, not its parent."""
    lonely = tmp_path / "nothing" / "here"
    lonely.mkdir(parents=True)
    (lonely / "server.py").write_text("", encoding="utf-8")
    assert server._find_project_root(lonely / "server.py") == lonely


def test_the_real_install_resolves_to_the_real_package(server) -> None:
    """Guards the actual installed copy when one is registered, skipped otherwise.

    The location comes from the DSH profile patch rather than a literal: this suite ships
    publicly, and a hardcoded machine path here failed the portability gate (it did).
    """
    import re

    patch = Path.home() / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"
    if not patch.is_file():
        pytest.skip("no DSH desktop profile on this machine")
    text = patch.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"([A-Za-z]:[\\/][^'\"\n]*server\.py)", text)
    if not match:
        pytest.skip("the profile patch does not register an MCP server script")
    install = Path(match.group(1).replace("\\\\", "\\"))
    if not install.is_file():
        pytest.skip(f"registered server is absent: {install}")
    root = server._find_project_root(install)
    assert root == install.parent, f"resolved {root}, expected {install.parent}"


def test_the_shipped_server_uses_the_robust_resolver() -> None:
    """A leftover `_TOOLS_DIR.parent` would silently reinstate the bug.

    Only *code* counts. The first version of this test matched the substring anywhere in
    the file and so failed on the comment explaining the bug — a test that cannot tell an
    explanation from a defect is testing the documentation, not the behaviour.
    """
    import ast as _ast

    text = SERVER.read_text(encoding="utf-8")
    tree = _ast.parse(text)
    offenders: list[int] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Attribute) and node.attr == "parent":
            value = node.value
            if isinstance(value, _ast.Name) and value.id == "_TOOLS_DIR":
                offenders.append(node.lineno)
    assert offenders == [], (
        f"_TOOLS_DIR.parent is still executed at line(s) {offenders}; it points outside the "
        "package whenever the file is packaged at the root"
    )
    assert "project_root" in text, "the server must delegate to the library resolver"


# ── 2. the package must ship schemas ─────────────────────────────────────
def test_the_package_plan_includes_the_schemas(builder) -> None:
    plan = builder.package_plan(PROJECT)
    destinations = [dest for _src, dest in plan]
    assert "schemas" in destinations, f"schemas missing from the plan: {destinations}"


def test_the_package_plan_includes_the_library_and_the_server(builder) -> None:
    plan = builder.package_plan(PROJECT)
    destinations = {dest for _src, dest in plan}
    assert "cbb_stage1" in destinations
    assert "server.py" in destinations


def test_every_planned_source_exists(builder) -> None:
    """A plan naming a path that does not exist builds a package with a hole in it."""
    missing = [
        str(src) for src, _dest in builder.package_plan(PROJECT) if not Path(src).exists()
    ]
    assert missing == [], f"plan names missing sources: {missing}"


def test_the_built_package_really_contains_schemas() -> None:
    """The end-to-end claim, checked against the artifact when it has been built."""
    dist = PROJECT / "dist" / "cbb-mcp"
    if not dist.is_dir():
        pytest.skip("package has not been built yet")
    schemas = dist / "schemas"
    assert schemas.is_dir(), "the built package has no schemas/"
    assert any(schemas.glob("*.schema.json")), "schemas/ is present but empty of schemas"


def test_the_resolver_finds_the_packaged_schemas(tmp_path: Path) -> None:
    """Shipping `schemas/` is not enough — the resolver has to look in the right place.

    `candidate_dirs()` derived the project root as `package.parents[1]`, which is the
    checkout layout `<root>/src/cbb_stage1`. In the package `<package>/cbb_stage1` it
    yields the package's *parent*, so every candidate missed and a fresh install reported
    `schema_dir: BLOCKED` while carrying 23 schema files. Measured on the real install.
    """
    from cbb_stage1.schemas import candidate_dirs

    package = tmp_path / "cbb-mcp"
    (package / "cbb_stage1").mkdir(parents=True)
    (package / "schemas").mkdir()
    (package / "cbb_stage1" / "__init__.py").write_text("", encoding="utf-8")

    dirs = candidate_dirs(package)
    assert package / "schemas" in dirs, f"the packaged schemas are not a candidate: {dirs}"
    assert package / "cbb_stage1" / "schemas" in dirs, "the nested candidate must survive"


def test_the_resolver_prefers_the_shipped_schemas_in_a_package(tmp_path: Path) -> None:
    """And it must actually be the one selected, not merely present in the tuple."""
    import shutil

    from cbb_stage1.schemas import schemas_dir

    package = tmp_path / "cbb-mcp"
    (package / "cbb_stage1").mkdir(parents=True)
    (package / "cbb_stage1" / "__init__.py").write_text("", encoding="utf-8")
    schemas = package / "schemas"
    schemas.mkdir()
    (schemas / "box.schema.json").write_text("{}", encoding="utf-8")

    # simulate the packaged module: copy the real resolver next to a stub __init__
    shutil.copy2(PROJECT / "src" / "cbb_stage1" / "schemas.py", package / "cbb_stage1" / "schemas.py")
    resolved = schemas_dir(package)
    assert Path(resolved) == schemas, f"resolved {resolved}, expected {schemas}"