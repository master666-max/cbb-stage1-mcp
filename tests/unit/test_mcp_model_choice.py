from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[2] / "tools" / "mcp_model_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("cbb_mcp_server_autopick", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_chat_model_is_refused_not_guessed(monkeypatch):
    """A dev box lists a 1B model first; auto-picking it silently breaks runs."""
    module = _load()
    monkeypatch.setattr(module, "_MODEL", "")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "gemma-3-1b-it"},
                    {"id": "hermes3.6-35b-a3b"},
                    {"id": "text-embedding-qwen3-embedding-8b@q4_k_m"},
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return FakeResponse()

    class FakeHttpx:
        Client = FakeClient

    monkeypatch.setattr(module, "_http", lambda: FakeHttpx)

    with pytest.raises(RuntimeError) as excinfo:
        module._discover_model("chat")
    message = str(excinfo.value)
    assert "CBB_MCP_BACKEND_MODEL" in message
    # The candidates are listed so the operator can choose, but none is chosen.
    assert "gemma-3-1b-it" in message
    assert "hermes3.6-35b-a3b" in message
