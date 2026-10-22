"""`cbb_start_run` must be able to start a run that does not need an agent in the loop.

The problem this fixes
----------------------
Every run started through the MCP surface writes ``host_queue: "1"`` into
``plan/model-policy.json`` and ``CBB_HOST_MODEL_QUEUE=1`` into the child's environment.
That means **the driving agent is the model**: every request becomes a turn of the
session's own conversation.

Measured on a 5000-character chapter: 38 requests had to be answered by hand, the driving
agent pulled 294,543 characters of prompt into its context to emit 28,710 characters of
answers (median answer 7 characters), the same chapter was echoed 28 times (73% of all
inbound text), two calls exceeded the 900-second queue timeout, and the run was still in
the claims stage after five hours. None of that is a defect in the pipeline — it is the
cost of the transport choice being hard-coded.

The pipeline already supports the alternative
---------------------------------------------
``model_gateway`` reads ``CBB_LLM_BASE_URL`` and, when ``host_queue`` is empty, talks to
that endpoint directly (``engines/ai_reader/llm_client.py`` and ``temporal/nli_client.py``
read the same variable). ``preflight`` already refuses a run that has neither
("no CBB_LLM_BASE_URL and not using the host queue").

So the fix is not new transport machinery — it is letting the caller choose.

The one rule that matters
-------------------------
``transport="http"`` **without** a base URL must refuse to start. Falling through to the
defaults would run the extraction against a mock engine and produce a box that looks
finished and contains nothing. That is the failure this project keeps writing tests against,
so it is the first thing pinned here.
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
    spec = importlib.util.spec_from_file_location("cbb_mcp_transport", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_transport"] = module
    spec.loader.exec_module(module)
    return module


# ── the default must not change ─────────────────────────────────────────
def test_host_is_still_the_default(server) -> None:
    """No transport argument keeps the previous behaviour exactly."""
    policy, env = server._transport_settings({})
    assert policy["transport"] == "host"
    assert policy["host_queue"] == "1"
    assert env["CBB_HOST_MODEL_QUEUE"] == "1"
    assert "CBB_LLM_BASE_URL" not in env


# ── the unattended path ─────────────────────────────────────────────────
def test_an_http_run_does_not_use_the_host_queue(server) -> None:
    policy, env = server._transport_settings(
        {"transport": "http", "llm_base_url": "http://127.0.0.1:8080/v1"}
    )
    assert policy["transport"] == "http"
    assert policy["host_queue"] == "", "an http run must not wait on the queue"
    assert "CBB_HOST_MODEL_QUEUE" not in env, (
        "inheriting CBB_HOST_MODEL_QUEUE would send the run back to the queue it was "
        "supposed to avoid"
    )


def test_an_http_run_passes_the_endpoint_through(server) -> None:
    policy, env = server._transport_settings(
        {
            "transport": "http",
            "llm_base_url": "http://127.0.0.1:8080/v1",
            "llm_model": "qwen2.5-7b-instruct",
        }
    )
    assert env["CBB_LLM_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["CBB_LLM_MODEL"] == "qwen2.5-7b-instruct"
    # whoever serves the reasoning has to be recorded, or the box cannot say
    assert policy["session_model_label"] == "qwen2.5-7b-instruct"


def test_the_caller_can_still_override_the_recorded_label(server) -> None:
    """An explicit label wins over the model id — the caller may know better."""
    policy, _env = server._transport_settings(
        {
            "transport": "http",
            "llm_base_url": "http://127.0.0.1:8080/v1",
            "llm_model": "qwen2.5-7b-instruct",
            "session_model_label": "lmstudio/qwen2.5-7b-instruct",
        }
    )
    assert policy["session_model_label"] == "lmstudio/qwen2.5-7b-instruct"


# ── refuse, never silently degrade ──────────────────────────────────────
def test_http_without_an_endpoint_is_refused(server) -> None:
    """A run with no endpoint and no queue would extract nothing — say so, don't start."""
    with pytest.raises(ValueError, match="llm_base_url"):
        server._transport_settings({"transport": "http"})


def test_an_unknown_transport_is_refused(server) -> None:
    """A typo must not silently mean 'host'."""
    with pytest.raises(ValueError, match="transport"):
        server._transport_settings({"transport": "htp", "llm_base_url": "http://x/v1"})


def test_the_two_supported_transports_are_named(server) -> None:
    assert set(server.TRANSPORTS) == {"host", "http"}
