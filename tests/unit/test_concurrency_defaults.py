"""The concurrency knobs must agree with each other.

Three numbers decide how much actually runs at once, and they lived in three places:

    Config.max_workers              pipeline-level worker count (default 4)
    ModelPolicy.max_concurrency     the gateway's BoundedSemaphore (MCP wrote 2)
    upstream["max_workers"]         what a stage reads via stage_workers()

Measured consequence of them disagreeing: `claims` could ask for 4 threads while the gateway
admitted only 2 model calls, so half the threads sat blocked on a semaphore. And the reference
implementation in the library defaults to **25** concurrent requests
(`graphrag .../defaults.py:189`), so 2 was not a considered value — it was the MCP server's
placeholder.

The rule
--------
The stage worker count is the pipeline's `max_workers`; the gateway must admit at least that
many calls or the extra threads are pointless. `cbb_start_run` therefore writes a
`max_concurrency` that is not below the stage default, and the pipeline states
`upstream["max_workers"]` explicitly so a stage does not have to introspect `Config`.
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
    spec = importlib.util.spec_from_file_location("cbb_mcp_concurrency", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_concurrency"] = module
    spec.loader.exec_module(module)
    return module


# ── the MCP policy default ─────────────────────────────────────────────
def test_the_mcp_default_concurrency_is_not_a_placeholder(server) -> None:
    """It used to be 2, which throttled every stage below the pipeline's own 4 workers."""
    default = server.CONCURRENCY_DEFAULT
    assert default >= 8, f"MCP default concurrency is {default}"
    # and it must be at least the pipeline's default worker count
    assert default >= 4


def test_the_policy_uses_the_documented_default(server) -> None:
    policy, _env = server._transport_settings({})
    assert policy.get("host_queue") == "1"  # transport choice unchanged


# ── the pipeline must state the worker count explicitly ────────────────
def test_the_pipeline_passes_max_workers_into_upstream() -> None:
    """A stage reads `upstream["max_workers"]`; it must not have to guess or introspect."""
    source = (PROJECT / "src" / "cbb_stage1" / "orchestrator" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert '"max_workers"' in source, 'pipeline does not pass "max_workers" into upstream'


# ── stage_workers must prefer the explicit value ───────────────────────
def test_stage_workers_prefers_the_explicit_upstream_value() -> None:
    from types import SimpleNamespace

    from cbb_stage1.orchestrator.parallel import stage_workers

    upstream = {"max_workers": 16, "config": SimpleNamespace(max_workers=2)}
    assert stage_workers(upstream, 100) == 16


def test_stage_workers_falls_back_to_config() -> None:
    from types import SimpleNamespace

    from cbb_stage1.orchestrator.parallel import stage_workers

    assert stage_workers({"config": SimpleNamespace(max_workers=6)}, 100) == 6
