"""Model operations, callable from both front-ends.

These four operations used to live in ``cli.py``, and the MCP tools reached them by
spawning ``python -m cbb_stage1.cli``. That made the MCP surface a wrapper over a
development tool and cost a process per call. The logic lives here now; the CLI and the
MCP server are both thin adapters over it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = [
    "call_once",
    "drain_queue",
    "policy_payload",
    "summarise_calls",
    "write_policy",
]


def call_once(
    work: str | Path,
    *,
    provider: str,
    model: str,
    prompt: str,
    role: str | None = None,
    system: str = "",
    out: str | Path | None = None,
    book: str | Path | None = None,
) -> dict[str, Any]:
    """One model call through the gateway.

    Returns ``{"ok": ..., "data": {...}}`` or ``{"blocked": reason}`` — never raises for
    a mock/unconfigured provider, because that is a normal state an agent must see.
    """
    from cbb_stage1.config import Config
    from cbb_stage1.model_gateway import build_model_gateway

    work_root = Path(work)
    config = Config(
        book_id="model-call",
        book_path=Path(str(book or "book.txt")),
        out_dir=Path(str(out or "artifacts/box")),
        work_dir=work_root,
        provider=provider,
        model=model or "model",
    )
    gateway = build_model_gateway(config, work_root)
    if gateway is None:
        return {
            "ok": False,
            "blocked": "provider is mock/unconfigured; no model call dispatched",
        }

    result = gateway.call(prompt, role=role, system=system or "")
    payload = {
        "status": result.status,
        "role": result.role,
        "text": result.text,
        "error": result.error,
        "provider": result.provider,
        "model": result.model,
    }
    if out:
        target = Path(str(out))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result.__dict__, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {"ok": result.status == "ok", "data": payload}


def summarise_calls(work: str | Path) -> dict[str, Any]:
    """Count the recorded model calls by (role, status)."""
    path = Path(work) / "plan" / "model-calls.jsonl"
    if not path.exists():
        return {"ok": True, "data": {"calls": 0, "by_role_status": {}}}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = Counter((row.get("role"), row.get("status")) for row in rows)
    return {
        "ok": True,
        "data": {
            "calls": len(rows),
            "by_role_status": {
                f"{role}:{status}": count for (role, status), count in sorted(
                    counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
                )
            },
        },
    }


def drain_queue(
    work: str | Path,
    *,
    queue: str | Path | None = None,
) -> dict[str, Any]:
    """List pending host-model requests without needing a run id."""
    from cbb_stage1.agents.host_queue import HostQueueClient

    target = Path(str(queue)) if queue else Path(work) / "plan" / "host-model"
    client = HostQueueClient(target)
    rows = list(client.drain())
    return {
        "ok": True,
        "data": {"queue": str(target), "pending": rows, "count": len(rows)},
    }


def policy_payload(**kwargs: Any) -> dict[str, Any]:
    """Build the ``model-policy.json`` payload from keyword arguments."""
    role_models = kwargs.get("role_models") or {}
    if isinstance(role_models, str):
        role_models = json.loads(role_models) if role_models else {}
    return {
        "provider": kwargs.get("provider") or "",
        "model": kwargs.get("model") or "",
        "transport": kwargs.get("transport") or "host",
        "host_queue": kwargs.get("host_queue") or "",
        "mcp_command": kwargs.get("mcp_command") or "",
        "mcp_tool": kwargs.get("mcp_tool") or "model_generate",
        "embedding_base_url": kwargs.get("embedding_base_url") or "",
        "embedding_model": kwargs.get("embedding_model") or "",
        "max_concurrency": kwargs.get("max_concurrency"),
        "max_tokens": kwargs.get("max_tokens"),
        "timeout": kwargs.get("timeout"),
        "token_budget": kwargs.get("token_budget"),
        "role_models": role_models,
        "session_model_label": kwargs.get("session_model_label") or "",
        "judge_base_url": kwargs.get("judge_base_url") or "",
        "judge_model": kwargs.get("judge_model") or "",
        "nli_base_url": kwargs.get("nli_base_url") or "",
        "nli_model": kwargs.get("nli_model") or "",
        "whitebox_backend": bool(kwargs.get("whitebox_backend")),
    }


def write_policy(work: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Write ``plan/model-policy.json`` without starting a run."""
    target = Path(work) / "plan" / "model-policy.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = policy_payload(**kwargs)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"ok": True, "data": {"policy": str(target), "payload": payload}}
