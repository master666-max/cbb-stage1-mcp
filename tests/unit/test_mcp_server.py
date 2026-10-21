from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from cbb_stage1.agents.mcp_model_client import _normalize_command


def test_normalize_command_strips_windows_quotes():
    argv = _normalize_command(['"C:\\a b\\python.exe" tools/x.py'])
    assert argv[0] == "C:\\a b\\python.exe"
    assert argv[1] == "tools/x.py"


def test_shipped_mcp_server_speaks_tools_protocol(monkeypatch):
    """The repo ships a real MCP server; verify its protocol in-process.

    Driving ``main()`` directly avoids spawning a child process (Windows AV
    intermittently denies CreateProcess, which would make this flaky).
    """
    server = Path(__file__).resolve().parents[2] / "tools" / "mcp_model_server.py"
    assert server.exists()

    import importlib.util

    spec = importlib.util.spec_from_file_location("cbb_mcp_server_under_test", server)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert module.main() == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert replies[0]["result"]["serverInfo"]["name"] == "cbb-model-gateway"
    names = [tool["name"] for tool in replies[1]["result"]["tools"]]
    assert "model_generate" in names
    assert "model_embed" in names
    # Unknown tool is reported as an error result, never fabricated text.
    assert replies[2]["result"]["isError"] is True
