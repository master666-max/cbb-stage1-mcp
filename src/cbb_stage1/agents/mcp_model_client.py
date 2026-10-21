"""MCP client side for the host agent's model (Scheme A).

MCP is a tool protocol, not a model API: this client can call a **host-provided**
MCP server tool (default ``model_generate``) that is backed by the current
agent session's model.  If no host server is configured, the gateway must fall
back to the HTTP endpoint or report BLOCKED; this module never fabricates a
model answer.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

__all__ = [
    "MCPError",
    "MCPTransport",
    "MCPStdioTransport",
    "MCPModelClient",
    "build_mcp_model_client",
]


class MCPError(RuntimeError):
    pass


def _normalize_command(command: Sequence[str]) -> list[str]:
    """Split a shell-style command string and repair Windows path quoting.

    ``shlex.split(..., posix=False)`` keeps the surrounding quotes, which makes
    ``CreateProcess`` fail with WinError 2 on a quoted executable path.
    """
    argv: list[str] = []
    for item in command:
        for token in shlex.split(str(item), posix=os.name != "nt"):
            token = token.strip()
            if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
                token = token[1:-1]
            if token:
                argv.append(token)
    return argv


class MCPTransport(Protocol):
    def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]: ...
    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None: ...
    def close(self) -> None: ...


class MCPStdioTransport:
    """Minimal MCP JSON-RPC over stdio (newline-delimited JSON)."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = 120.0,
        cwd: str | Path | None = None,
    ) -> None:
        self.timeout = timeout
        self._next_id = 1
        self._pending: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            _normalize_command(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(cwd) if cwd else None,
        )
        self._initialize()

    def _read_response(self, request_id: int) -> dict[str, Any]:
        buffered = self._pending.pop(request_id, None)
        if buffered is not None:
            return self._unwrap(buffered)
        deadline = time.monotonic() + self.timeout
        assert self.process.stdout is not None
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                raise MCPError("MCP server closed stdout")
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # server logs on stdout are ignored, not trusted
            payload_id = payload.get("id")
            if payload_id != request_id:
                # Never drop another caller's response: buffer it by id so a
                # concurrent request can still find it (stdio is one pipe).
                if isinstance(payload_id, int):
                    self._pending[payload_id] = payload
                continue
            return self._unwrap(payload)
        raise MCPError(f"MCP request {request_id} timed out")

    @staticmethod
    def _unwrap(payload: Mapping[str, Any]) -> dict[str, Any]:
        if "error" in payload:
            raise MCPError(f"MCP error: {payload['error']}")
        result = payload.get("result")
        return dict(result) if isinstance(result, Mapping) else {"result": result}

    def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        # One pipe, many callers: the write/read pair must be atomic, otherwise
        # thread A can consume thread B's response line.
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            return self._read_response(request_id)

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cbb-stage1", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized")

    def close(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.process.kill()


class MCPModelClient:
    """Async client shaped like the existing LLM clients, backed by MCP."""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        tool: str = "model_generate",
        provider: str = "mcp-agent",
        model: str = "session",
    ) -> None:
        self.transport = transport
        self.tool = tool
        self.provider = provider
        self.model = model

    async def generate(self, system: str, prompt: str, **kwargs: Any) -> tuple[str, Any]:
        args = {
            "system": system,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature", 0.1),
            "timeout": kwargs.get("timeout"),
        }
        result = self.transport.request(
            "tools/call",
            {"name": self.tool, "arguments": {k: v for k, v in args.items() if v is not None}},
        )
        # A failed tool call is reported as a *normal* result with ``isError``;
        # treating its text as a model answer would put "HTTPStatusError: ..."
        # into the box while the audit log says status=ok.
        if result.get("isError"):
            detail = _tool_text(result)[0] or "MCP tool reported isError"
            raise MCPError(f"MCP model call failed: {detail}")
        text, usage = _tool_text(result)
        if not text:
            raise MCPError("MCP model tool returned no text")
        return text, usage

    def close(self) -> None:
        self.transport.close()


def _tool_text(result: Mapping[str, Any]) -> tuple[str, Any]:
    # MCP tools/call -> {content:[{type:"text",text:"..."}], structuredContent?}
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping):
        if isinstance(structured.get("text"), str):
            return structured["text"], structured.get("usage")
        if isinstance(structured.get("content"), str):
            return structured["content"], structured.get("usage")
    content = result.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        if parts:
            return "\n".join(parts), None
    if isinstance(result.get("text"), str):
        return result["text"], result.get("usage")
    return "", None


def build_mcp_model_client(
    command: str | None = None,
    tool: str | None = None,
    cwd: str | Path | None = None,
) -> MCPModelClient | None:
    """Build from policy values or CBB_MCP_MODEL_COMMAND."""
    command = (command if command is not None else os.environ.get("CBB_MCP_MODEL_COMMAND", "")).strip()
    if not command:
        return None
    tool = (tool if tool is not None else os.environ.get("CBB_MCP_MODEL_TOOL", "")).strip()
    tool = tool or "model_generate"
    transport = MCPStdioTransport(_normalize_command([command]), cwd=cwd)
    return MCPModelClient(transport, tool=tool)
