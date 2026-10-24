"""Let the caller choose which postprocess stages run.

Measured gap
------------
`run_pipeline(postprocess_stages=...)` exists (`orchestrator/pipeline.py:509`) but its own docstring
says "Override the postprocess stage list (tests)" — the CLI has no `--stages` and the MCP surface has
no parameter. So the only way to run one stage is to write Python.

Why the surface needs it: when a run dies in `derive`, the only way to retry is the whole 8-stage
chain, which re-pays for `resolve`/`assemble`/`claims` and rewrites tables that were already correct.
An agent driving the pipeline has no way to say "just re-run derive".

The parse is strict on purpose: an unknown stage name is an error, not a silently shorter list — a
typo that quietly skips `validate` would produce a box with unvalidated assertions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cbb_stage1.orchestrator.postprocess import DEFAULT_STAGES, parse_stage_names

PROJECT = Path(__file__).resolve().parents[2]


def _spec(server):
    return next(tool for tool in server.TOOLS if tool["name"] == "cbb_start_run")


# ══ the parser ════════════════════════════════════════════════════════
def test_a_valid_subset_is_parsed() -> None:
    specs = parse_stage_names("resolve,derive")
    assert [spec.name for spec in specs] == ["resolve", "derive"]
    assert specs[0].module == "cbb_stage1.resolve.integration"


def test_order_follows_the_default_chain() -> None:
    """The stages have a fixed order; asking for them backwards must not run them backwards."""
    specs = parse_stage_names("derive,resolve")
    assert [spec.name for spec in specs] == ["resolve", "derive"], (
        "the caller's order was honoured, but later stages read earlier tables"
    )


def test_an_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_stage_names("resolve,nonsense")


def test_an_empty_selection_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_stage_names("   ")


def test_all_names_are_available() -> None:
    specs = parse_stage_names(",".join(spec.name for spec in DEFAULT_STAGES))
    assert len(specs) == len(DEFAULT_STAGES)


def test_duplicates_collapse() -> None:
    specs = parse_stage_names("resolve,resolve")
    assert [spec.name for spec in specs] == ["resolve"]


# ══ the CLI must expose it ════════════════════════════════════════════
def test_the_cli_has_a_stages_option() -> None:
    """Read the parser source rather than running `--help`.

    Spawning a subprocess just to inspect a parser makes the test depend on process creation, which
    is intermittently denied on this host (`WinError 5`) — the test then fails for a reason unrelated
    to the flag.
    """
    import ast as _ast

    tree = _ast.parse(
        (PROJECT / "src" / "cbb_stage1" / "cli.py").read_text(encoding="utf-8")
    )
    flags: set[str] = set()
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func = node.func
        if not (isinstance(func, _ast.Attribute) and func.attr == "add_argument"):
            continue
        for argument in node.args:
            if isinstance(argument, _ast.Constant) and isinstance(argument.value, str):
                flags.add(argument.value)
    assert "--stages" in flags, (
        f"--stages is missing from the CLI, so the stage selection is unreachable: {sorted(flags)}"
    )


# ══ the MCP surface must expose it ════════════════════════════════════
def test_the_mcp_tool_declares_stages() -> None:
    import importlib.util

    server_path = PROJECT / "tools" / "mcp_pipeline_server.py"
    spec = importlib.util.spec_from_file_location("cbb_mcp_stages", server_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_stages"] = module
    spec.loader.exec_module(module)

    properties = _spec(module)["inputSchema"]["properties"]
    assert "stages" in properties, (
        "cbb_start_run cannot select stages, so an agent can only re-run the whole chain"
    )
    assert "description" in properties["stages"], "the parameter must say what the names are"


def test_the_mcp_tool_passes_the_flag_through() -> None:
    source = (PROJECT / "tools" / "mcp_pipeline_server.py").read_text(encoding="utf-8")
    assert '"--stages"' in source, "the MCP handler never builds the --stages argument"
