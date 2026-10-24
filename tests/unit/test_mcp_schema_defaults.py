"""The advertised default must be the default the handler uses.

`cbb_start_run`'s schema says `"max_concurrency": {"default": 2}` while the handler writes
`int(args.get("max_concurrency") or CONCURRENCY_DEFAULT)` with `CONCURRENCY_DEFAULT = 8`. So an agent
that reads the schema — the only machine-readable description it has — plans for 2, and gets 8.

That gap matters more than a docstring being wrong, because the schema is what a client validates
against and what an agent reasons with. It is the same class as the `work` field missing from
`required`: the surface describes something other than what happens.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SERVER = PROJECT / "tools" / "mcp_pipeline_server.py"


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("cbb_mcp_schema_defaults", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_schema_defaults"] = module
    spec.loader.exec_module(module)
    return module


def _tool(server, name: str) -> dict:
    return next(tool for tool in server.TOOLS if tool["name"] == name)


# ══ the schema default is the effective default ═══════════════════════
def test_start_run_schema_default_matches_the_constant(server) -> None:
    properties = _tool(server, "cbb_start_run")["inputSchema"]["properties"]
    declared = properties["max_concurrency"].get("default")
    assert declared == server.CONCURRENCY_DEFAULT, (
        f"the schema advertises max_concurrency={declared} but the handler uses "
        f"{server.CONCURRENCY_DEFAULT}; an agent planning from the schema plans wrong"
    )


def test_the_declared_default_is_not_below_the_stage_worker_default(server) -> None:
    """The gateway must admit at least as many calls as the stages can spawn."""
    from cbb_stage1.config import Config

    declared = _tool(server, "cbb_start_run")["inputSchema"]["properties"]["max_concurrency"].get(
        "default"
    )
    assert declared >= Config.__dataclass_fields__["max_workers"].default, (
        "the advertised concurrency is below Config.max_workers, so threads would queue on the "
        "gateway semaphore"
    )


def test_policy_tool_does_not_advertise_a_misleading_default(server) -> None:
    """`cbb_model_policy` may leave the default unset; it must not disagree with the constant."""
    properties = _tool(server, "cbb_model_policy")["inputSchema"]["properties"]
    declared = properties["max_concurrency"].get("default")
    assert declared in (None, server.CONCURRENCY_DEFAULT)


def test_the_schema_description_mentions_the_unit(server) -> None:
    """A bare integer with no unit is how the 2-vs-8 confusion started."""
    properties = _tool(server, "cbb_start_run")["inputSchema"]["properties"]
    body = properties["max_concurrency"]
    text = f"{body.get('description') or ''}"
    assert text.strip(), "max_concurrency has no description at all"
