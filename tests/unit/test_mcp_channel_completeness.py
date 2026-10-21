"""MCP must be a complete control channel, and the linters must actually run.

An agent-driven project whose agent cannot reach a capability is the same failure
as a function nobody calls, one level up. These tests pin:

  * every CLI command has an MCP tool (``cbb_capabilities`` reports missing=[]);
  * every advertised tool has a handler;
  * every tool response has the same envelope, so recovery logic is writable;
  * ruff / vulture / mypy are wired as hard gates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_server():
    path = ROOT / "tools" / "mcp_pipeline_server.py"
    spec = importlib.util.spec_from_file_location("mcp_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── channel completeness ─────────────────────────────────────────────────────
def test_every_cli_command_is_reachable_over_mcp() -> None:
    module = _load_server()
    envelope = module._DISPATCH["cbb_capabilities"]({})
    data = envelope["data"]
    assert data["missing"] == [], f"unreachable from MCP: {data['missing']}"


def test_every_advertised_tool_has_a_handler() -> None:
    module = _load_server()
    advertised = {tool["name"] for tool in module.TOOLS}
    assert advertised == set(module._DISPATCH)


def test_the_model_call_surface_exists() -> None:
    """The agent must be able to make one call without running the pipeline."""
    module = _load_server()
    for name in (
        "cbb_model_call",
        "cbb_model_stats",
        "cbb_model_drain",
        "cbb_model_policy",
        "cbb_smoke",
    ):
        assert name in module._DISPATCH, name


# ── uniform envelope ─────────────────────────────────────────────────────────
def test_success_response_has_the_envelope() -> None:
    module = _load_server()
    result = module._tool_result({"value": 1})
    envelope = result["structuredContent"]
    assert envelope["ok"] is True
    assert envelope["error"] is None
    assert envelope["data"] == {"value": 1}
    assert result["isError"] is False


def test_usage_errors_are_classified() -> None:
    module = _load_server()
    envelope = module._tool_result({"error": "box is required"})["structuredContent"]
    assert envelope["ok"] is False
    assert envelope["error"]["kind"] == "usage"


def test_blocked_errors_are_classified() -> None:
    module = _load_server()
    envelope = module._tool_result(
        {"error": "preflight blocked", "blocked": ["llm_endpoint"]}
    )["structuredContent"]
    assert envelope["error"]["kind"] == "blocked"


def test_backend_errors_are_classified() -> None:
    module = _load_server()
    envelope = module._tool_result(
        {"error": "connection refused to endpoint"}
    )["structuredContent"]
    assert envelope["error"]["kind"] == "backend"


def test_original_payload_is_preserved() -> None:
    """Existing callers read ``returncode``/``stdout``; the envelope must not break them."""
    module = _load_server()
    payload = {"returncode": 0, "stdout": "ok", "stderr": ""}
    envelope = module._tool_result(payload)["structuredContent"]
    assert envelope["data"]["returncode"] == 0
    assert envelope["data"]["stdout"] == "ok"


# ── linter gates ─────────────────────────────────────────────────────────────
def test_linters_are_hard_gates_in_acceptance() -> None:
    source = (ROOT / "src" / "cbb_stage1" / "qc" / "acceptance.py").read_text(
        encoding="utf-8"
    )
    assert "_linter_problems()" in source
    assert "ruff" in source
    assert "vulture" in source
    assert "union-attr" in source, "mypy risk gate missing"


def test_ruff_config_only_enables_hygiene_rules() -> None:
    """The default rule set is 465 findings; enabling it would bury the real ones."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.ruff.lint]" in text
    assert 'select = ["F401", "F811", "F841"]' in text


def test_vulture_whitelist_exists_and_explains_itself() -> None:
    whitelist = ROOT / ".vulture_whitelist.py"
    assert whitelist.is_file()
    text = whitelist.read_text(encoding="utf-8")
    assert "framework" in text.lower()
    assert "public" in text.lower()


def test_vulture_whitelist_covers_framework_magic() -> None:
    """The whitelist must name the framework hooks vulture cannot see.

    Asserted structurally rather than by spawning vulture: subprocess creation is
    flaky under the Windows sandbox, and the sandbox's behaviour says nothing about
    whether the whitelist is correct. The real gate runs in acceptance.
    """
    text = (ROOT / ".vulture_whitelist.py").read_text(encoding="utf-8")
    for expected in (
        "model_config",      # Pydantic class-creation hook
        "_validate_reason",  # Pydantic validator
        "__exit__",          # context manager protocol
    ):
        assert expected in text, f"whitelist does not cover {expected}"
