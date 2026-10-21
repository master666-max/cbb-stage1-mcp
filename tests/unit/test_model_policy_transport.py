from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.config import Config
from cbb_stage1.model_gateway import ModelPolicy, build_model_gateway, load_policy


def _cfg(work: Path) -> Config:
    # A live provider: ``mock`` intentionally builds no gateway at all.
    return Config(
        book_id="t",
        book_path=work / "book.txt",
        out_dir=work / "box",
        work_dir=work,
        provider="host",
        model="session",
        live=True,
    )


def _write_policy(work: Path, payload: dict) -> None:
    path = work / "plan" / "model-policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_policy_is_host_transport():
    assert ModelPolicy().transport == "host"


def test_policy_chooses_host_transport_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CBB_HOST_MODEL_QUEUE", raising=False)
    monkeypatch.delenv("CBB_MCP_MODEL_COMMAND", raising=False)
    _write_policy(tmp_path, {"transport": "host", "host_queue": "1"})
    gateway = build_model_gateway(_cfg(tmp_path), tmp_path)
    assert gateway is not None
    assert gateway.transport_kind == "host"
    assert load_policy(tmp_path).host_queue == "1"


def test_explicit_mcp_transport_is_honoured(tmp_path, monkeypatch):
    """MCP is the agent-driven transport; it is chosen only when declared."""
    monkeypatch.delenv("CBB_HOST_MODEL_QUEUE", raising=False)
    _write_policy(
        tmp_path,
        {"transport": "mcp", "mcp_command": "python x.py", "mcp_tool": "model_generate"},
    )
    spawned = {}

    def fake_build(command=None, tool=None, cwd=None):
        spawned["command"] = command
        spawned["tool"] = tool
        return object()

    import cbb_stage1.agents.mcp_model_client as mcp_module

    monkeypatch.setattr(mcp_module, "build_mcp_model_client", fake_build)
    gateway = build_model_gateway(_cfg(tmp_path), tmp_path)
    assert gateway is not None
    assert gateway.transport_kind == "mcp"
    assert spawned["tool"] == "model_generate"


def test_auto_without_any_config_stays_on_host(tmp_path, monkeypatch):
    """``auto`` must not degrade to a local endpoint when nothing is configured."""
    monkeypatch.delenv("CBB_HOST_MODEL_QUEUE", raising=False)
    monkeypatch.delenv("CBB_MCP_MODEL_COMMAND", raising=False)
    _write_policy(tmp_path, {"transport": "auto"})
    gateway = build_model_gateway(_cfg(tmp_path), tmp_path)
    assert gateway is not None
    assert gateway.transport_kind == "host"
