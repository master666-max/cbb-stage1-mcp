from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cbb_stage1.agents.mcp_model_client import MCPError, MCPModelClient


class _Transport:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.lock = threading.Lock()

    def request(self, method, params):
        with self.lock:
            self.calls += 1
        return self.reply

    def notify(self, method, params=None):
        pass

    def close(self):
        pass


def test_iserror_is_a_failure_not_a_model_answer():
    """A failed tool call must raise; its text is an error message, not output."""
    transport = _Transport(
        {
            "content": [{"type": "text", "text": "HTTPStatusError: 500"}],
            "isError": True,
        }
    )
    client = MCPModelClient(transport)
    with pytest.raises(MCPError) as excinfo:
        asyncio.run(client.generate("sys", "prompt"))
    assert "HTTPStatusError" in str(excinfo.value)


def test_ok_result_still_returns_text():
    transport = _Transport({"content": [{"type": "text", "text": "fine"}], "isError": False})
    client = MCPModelClient(transport)
    text, _ = asyncio.run(client.generate("sys", "prompt"))
    assert text == "fine"


class _Pipe:
    """Fake stdio pipe that answers out of order to expose cross-talk."""

    def __init__(self):
        self.lines: list[str] = []
        self.read_index = 0

    def readline(self):
        if self.read_index >= len(self.lines):
            return ""
        value = self.lines[self.read_index]
        self.read_index += 1
        return value


class _Process:
    def __init__(self, lines):
        self.stdin = _Stdin()
        self.stdout = _Pipe()
        self.stdout.lines = lines


class _Stdin:
    def write(self, data):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _transport_with_lines(lines):
    from cbb_stage1.agents.mcp_model_client import MCPStdioTransport

    transport = MCPStdioTransport.__new__(MCPStdioTransport)
    transport.timeout = 5.0
    transport._next_id = 1
    transport._pending = {}
    transport._lock = threading.Lock()
    transport.process = _Process(lines)
    return transport


def test_out_of_order_response_is_buffered_not_dropped():
    """stdio is one pipe: another caller's response must be kept, not discarded."""
    transport = _transport_with_lines(
        [
            '{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"second"}]}}\n',
            '{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"first"}]}}\n',
        ]
    )
    first = transport.request("tools/call", {"name": "x"})
    assert first["content"][0]["text"] == "first"
    # id=2 was seen while waiting for id=1 and must still be retrievable.
    second = transport.request("tools/call", {"name": "x"})
    assert second["content"][0]["text"] == "second"
