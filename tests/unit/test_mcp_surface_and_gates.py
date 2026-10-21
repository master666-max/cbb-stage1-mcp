"""Newly added gates and human-decision commands must be dispatchable.

A capability that only exists as a Python function is not available to the agent
driving the project — that is the same "written but not wired" failure one level
up. These tests pin the surface: MCP tools, CLI commands, and preflight probes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_server():
    path = ROOT / "tools" / "mcp_pipeline_server.py"
    spec = importlib.util.spec_from_file_location("mcp_pipeline_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mcp_exposes_the_new_gate_and_decision_tools() -> None:
    module = _load_server()
    names = {tool["name"] for tool in module.TOOLS}
    for expected in (
        "cbb_lexicon_list",
        "cbb_lexicon_approve",
        "cbb_supersede",
        "cbb_field_coverage",
        "cbb_call_wiring",
    ):
        assert expected in names, f"{expected} is not dispatchable over MCP"


def test_every_advertised_tool_has_a_handler() -> None:
    """A tool in the schema with no handler fails only when the agent calls it."""
    module = _load_server()
    advertised = {tool["name"] for tool in module.TOOLS}
    handled = set(module._DISPATCH)
    assert advertised == handled, advertised.symmetric_difference(handled)


def test_lexicon_approve_refuses_without_a_human_gate() -> None:
    module = _load_server()
    result = module._DISPATCH["cbb_lexicon_approve"](
        {"work": "x", "predicate": "初识"}
    )
    assert "error" in result
    assert "human gate" in result["error"] or "approved_by" in result["error"]


def test_supersede_refuses_without_a_reason() -> None:
    module = _load_server()
    result = module._DISPATCH["cbb_supersede"](
        {"work": "x", "loser": "a:1", "winner": "a:2", "by": "someone"}
    )
    assert "error" in result
    assert "reason" in result["error"]


def test_field_coverage_is_callable_over_mcp(tmp_path: Path) -> None:
    """The tool must be dispatchable. Assert on the dispatch wiring rather than on a
    subprocess exit code: spawning python from a test is flaky under the Windows
    sandbox (PermissionError on pipe creation), and that flakiness says nothing
    about whether the tool is reachable."""
    module = _load_server()
    assert "cbb_field_coverage" in module._DISPATCH
    handler = module._DISPATCH["cbb_field_coverage"]
    # A missing box is refused before any subprocess is spawned.
    refused = handler({"box": ""})
    assert "error" in refused
    assert "box" in refused["error"]


def test_call_wiring_is_callable_over_mcp() -> None:
    """Dispatch wiring, not subprocess behaviour: spawning python from a test is
    flaky under the Windows sandbox and proves nothing about reachability."""
    module = _load_server()
    assert "cbb_call_wiring" in module._DISPATCH
    # Exercise the underlying analysis directly instead of shelling out.
    import importlib.util
    import sys as _sys

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("cw_inline", root / "tools" / "call_wiring.py")
    assert spec and spec.loader
    inline = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = inline
    spec.loader.exec_module(inline)
    assert inline.analyse(root / "src" / "cbb_stage1") == []


def test_preflight_probes_the_gate_tools_themselves() -> None:
    """A missing gate tool is worse than no gate: acceptance silently skips it."""
    sys.path.insert(0, str(ROOT / "src"))
    from cbb_stage1.preflight import PROBES, _probe_gate_tools, _probe_schema_dir

    names = [name for name, _ in PROBES]
    assert "gate_tools" in names
    assert "schema_dir" in names

    tools = _probe_gate_tools()
    assert tools["status"] == "READY", tools
    assert tools["evidence"]["present"]["field_coverage.py"] is True
    assert tools["evidence"]["present"]["call_wiring.py"] is True

    schemas = _probe_schema_dir()
    assert schemas["status"] == "READY", schemas
    assert schemas["evidence"]["count"] >= 10


def test_field_coverage_refuses_to_report_clean_on_an_empty_scan(tmp_path: Path) -> None:
    """A green result from zero schemas is the illusion this tool exists to remove."""
    path = ROOT / "tools" / "field_coverage.py"
    spec = importlib.util.spec_from_file_location("fc_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    box = tmp_path / "box"
    box.mkdir()
    empty_schemas = tmp_path / "schemas"
    empty_schemas.mkdir()
    code = module.main([str(box), "--schema-dir", str(empty_schemas)])
    assert code == 2, "an empty schema dir must not produce a clean report"


def test_call_wiring_is_a_hard_gate_now() -> None:
    """It was a warning while the backlog existed; with zero gaps it must fail."""
    source = (ROOT / "src" / "cbb_stage1" / "qc" / "acceptance.py").read_text(
        encoding="utf-8"
    )
    assert "problems.extend(_call_level_problems())" in source
    assert "[warn] {warning}" not in source, "call-wiring must not still be a warning"
