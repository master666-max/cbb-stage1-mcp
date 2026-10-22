"""Host-agent model bridge (the agent running this session is the model host).

A Python pipeline cannot synchronously call the host conversation's model.  The
host can, however, be the model loop: the pipeline writes a request, the host
agent answers it with ``model-drain`` / ``model-reply`` (or an equivalent tool
loop), and the pipeline continues.  This is the file-queue transport used when
``CBB_HOST_MODEL_QUEUE`` is set.

It is intentionally a transport, not a fake model: a missing reply times out and
fails the call; no text is invented here.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "HostQueueError",
    "HostQueueTimeout",
    "HostQueueClient",
    "build_host_queue_client",
]

REQUESTS = "requests"
RESPONSES = "responses"


class HostQueueError(RuntimeError):
    pass


class HostQueueTimeout(HostQueueError):
    pass


class HostQueueClient:
    """Async generate() over work/plan/host-model/{requests,responses}."""

    def __init__(self, queue_dir: str | Path, *, timeout: float = 900.0, poll: float = 0.2) -> None:
        self.root = Path(queue_dir)
        self.timeout = float(timeout)
        self.poll = max(0.02, float(poll))
        (self.root / REQUESTS).mkdir(parents=True, exist_ok=True)
        (self.root / RESPONSES).mkdir(parents=True, exist_ok=True)
        self.provider = "host-agent"
        self.model = "session"

    async def generate(self, system: str, prompt: str, **kwargs: Any) -> tuple[str, Any]:
        request_id = f"{time.time_ns()}-{uuid.uuid4().hex[:12]}"
        request = {
            "request_id": request_id,
            "system": system,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature", 0.1),
            "timeout": kwargs.get("timeout"),
            "created_at": time.time(),
        }
        _write_json(self.root / REQUESTS / f"{request_id}.json", request)
        response_path = self.root / RESPONSES / f"{request_id}.json"
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if response_path.exists():
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                if payload.get("status") != "ok":
                    raise HostQueueError(payload.get("error") or "host model failed")
                return str(payload.get("text") or ""), payload.get("usage")
            await asyncio.sleep(self.poll)
        raise HostQueueTimeout(
            f"host model did not reply to {request_id} within {self.timeout}s"
        )

    def drain(self) -> list[dict[str, Any]]:
        """Return **unanswered** requests for the host agent (sorted, oldest first).

        "Pending" means no response file exists yet. This used to list every request
        regardless: on a real work directory it reported 16 pending of which 14 had already
        been answered, so an agent following it would answer them twice. `cbb_next_requests`
        compared against `responses/` and was right; the two tools disagreed about the same
        queue, which is worse than either being wrong alone.
        """
        rows: list[dict[str, Any]] = []
        for path in sorted((self.root / REQUESTS).glob("*.json")):
            if (self.root / RESPONSES / path.name).exists():
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            row["request_path"] = str(path)
            rows.append(row)
        return rows

    def reply(
        self,
        request_id: str,
        text: str,
        *,
        usage: Any = None,
        status: str = "ok",
        error: str | None = None,
    ) -> str:
        target = self.root / RESPONSES / f"{request_id}.json"
        _write_json(
            target,
            {
                "request_id": request_id,
                "status": status,
                "text": text,
                "usage": usage,
                "error": error,
                "replied_at": time.time(),
            },
        )
        return str(target)

    def close(self) -> None:
        return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_host_queue_client(work_root: str | Path | None, raw: str | None = None) -> HostQueueClient | None:
    """Build from an explicit policy value or CBB_HOST_MODEL_QUEUE."""
    raw = (raw if raw is not None else os.environ.get("CBB_HOST_MODEL_QUEUE", "")).strip()
    if not raw:
        return None
    if raw.lower() in {"1", "host", "on", "true"}:
        if work_root is None:
            raise ValueError("CBB_HOST_MODEL_QUEUE=1 requires a work_root")
        root = Path(work_root) / "plan" / "host-model"
    else:
        root = Path(raw)
        if not root.is_absolute() and work_root is not None:
            root = Path(work_root) / root
    timeout = float(os.environ.get("CBB_HOST_MODEL_TIMEOUT", "900"))
    return HostQueueClient(root, timeout=timeout)
