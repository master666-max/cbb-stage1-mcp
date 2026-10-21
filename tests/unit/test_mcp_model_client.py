from __future__ import annotations

import asyncio

import pytest

from cbb_stage1.agents.mcp_model_client import MCPError, MCPModelClient, build_mcp_model_client


class FakeTransport:
    def __init__(self):
        self.calls = []

    def request(self, method, params):
        self.calls.append((method, dict(params)))
        return {"content": [{"type": "text", "text": "from-session"}]}

    def notify(self, method, params=None):
        pass

    def close(self):
        pass


def test_mcp_model_client_uses_tools_call():
    transport = FakeTransport()
    client = MCPModelClient(transport, tool="model_generate")
    text, usage = asyncio.run(client.generate("sys", "prompt", max_tokens=8))
    assert text == "from-session"
    assert usage is None
    method, params = transport.calls[0]
    assert method == "tools/call"
    assert params["name"] == "model_generate"
    assert params["arguments"]["system"] == "sys"


def test_mcp_client_rejects_empty_tool_output():
    class Empty(FakeTransport):
        def request(self, method, params):
            return {"content": []}

    with pytest.raises(MCPError):
        asyncio.run(MCPModelClient(Empty()).generate("s", "p"))


def test_build_mcp_client_disabled_without_command(monkeypatch):
    monkeypatch.delenv("CBB_MCP_MODEL_COMMAND", raising=False)
    assert build_mcp_model_client() is None
