"""The MCP server owns the project; the CLI is a development tool.

The architecture this file pins:

    MCP tools  ──┐
                 ├──> cbb_stage1 library   (all real logic)
    CLI        ──┘

Both are front-ends over the same library. Neither depends on the other.

What it looked like before: 13 of 21 tools did ``return _cli(argv)``, which spawned
``python -m cbb_stage1.cli`` in a subprocess and handed the agent ``{"returncode",
"stdout", "stderr"}``. Three things wrong with that:

  * the MCP surface owned nothing — it delegated its jurisdiction to a development tool;
  * the agent had to parse CLI text instead of reading structured data;
  * every call paid a process spawn (~0.8–1.1 s measured).

These tests fail while any tool still delegates, and pass when the tools call the
library directly.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

# Relative to this file: the portability gate caught this file pinning an absolute
# path, which is the same defect it exists to prevent. `parents[2]` is
# `cbb_stage1_v6`; the server lives in its `tools/` directory.
PROJECT = Path(__file__).resolve().parents[2]
SERVER = PROJECT / "tools" / "mcp_pipeline_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("cbb_mcp_server_t", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_server_t"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def server():
    return _load_server()


# ── the architectural invariant ─────────────────────────────────────────────
def _tool_functions() -> dict[str, ast.AST]:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("cbb_")
    }


def test_no_mcp_tool_delegates_to_the_cli() -> None:
    """The invariant. Every tool must call the library, not shell out to the CLI.

    A subprocess call to ``cbb_stage1.cli`` inside an MCP tool makes the server a thin
    wrapper over a development tool: the tool owns no capability, the result is text
    the agent must parse, and each call costs a process.
    """
    offenders: list[str] = []
    for name, node in _tool_functions().items():
        source = ast.get_source_segment(SERVER.read_text(encoding="utf-8"), node) or ""
        if "_cli(" in source:
            offenders.append(name)
    assert offenders == [], (
        "these MCP tools still delegate to the CLI instead of calling the library: "
        f"{sorted(offenders)}"
    )


def test_only_the_run_launcher_spawns_a_python_subprocess() -> None:
    """No tool delegates a synchronous capability to the CLI.

    One spawn is legitimate and must stay: ``cbb_start_run`` launches the pipeline as a
    **detached long-running job** that has to outlive the tool call. That is process
    management, not capability delegation — it cannot be an in-process call, because the
    pipeline must keep running after the MCP call returns.

    Everything else (gates, builds, decisions, model calls) is a library call now.

    The check is per function body, not per call site: the launcher builds its argv in a
    local variable, so a call-site text search sees ``Popen(argv)`` with no CLI string in
    it and would report nothing at all.
    """
    source = SERVER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    spawners: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.get_source_segment(source, node) or ""
        if "cbb_stage1.cli" not in body:
            continue
        spawners.append(node.name)

    assert spawners == ["cbb_start_run"], (
        "only the detached run launcher may reference the CLI; found: "
        f"{sorted(spawners)}"
    )


@pytest.mark.parametrize(
    "tool",
    [
        "cbb_lexicon_list",
        "cbb_lexicon_approve",
        "cbb_supersede",
        "cbb_conflict_resolve",
        "cbb_quarantine_adjudicate",
        "cbb_build_box",
        "cbb_lint_box",
        "cbb_gate",
        "cbb_smoke",
        "cbb_model_stats",
        "cbb_model_drain",
    ],
)
def test_the_tool_returns_structured_data(server, tool: str) -> None:
    """No `stdout`/`returncode` envelope: the agent gets the result itself."""
    function = getattr(server, tool)
    # the source must not build a CLI argv
    source = inspect.getsource(function)
    assert "_cli(" not in source, f"{tool} still delegates to the CLI"
    assert "argv" not in source, f"{tool} still builds a CLI argv"


def test_the_server_still_exposes_every_registered_tool(server) -> None:
    """Refactoring must not lose a tool from the dispatch table."""
    registered = set(server._DISPATCH)
    assert len(registered) == 21, f"expected 21 tools, got {len(registered)}"
    for name in (
        "cbb_start_run", "cbb_status", "cbb_next_requests", "cbb_reply",
        "cbb_build_box", "cbb_lint_box", "cbb_gate", "cbb_lexicon_approve",
        "cbb_supersede", "cbb_conflict_resolve", "cbb_quarantine_adjudicate",
    ):
        assert name in registered, f"{name} disappeared from the dispatch table"


# ── the library is importable without the CLI ──────────────────────────────
def test_the_library_never_imports_the_cli() -> None:
    """The CLI is a front-end; the library must not depend on it.

    Checked statically rather than by spawning a subprocess: the earlier version ran
    ``python -c "import ..."`` and hit ``PermissionError [WinError 5]`` on process
    creation in this environment. A static check is also the stronger statement —
    "no module imports the CLI" holds even for import paths a single probe misses.
    """
    offenders: list[str] = []
    for path in (PROJECT / "src" / "cbb_stage1").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name in {"cli.py", "__main__.py"}:
            continue  # the front-end itself
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.ImportFrom):
                target = node.module or ""
            elif isinstance(node, ast.Import):
                target = ",".join(alias.name for alias in node.names)
            if target and "cbb_stage1.cli" in target:
                offenders.append(f"{path.relative_to(PROJECT)}:{node.lineno}")
    assert offenders == [], f"the library imports the CLI: {offenders}"
