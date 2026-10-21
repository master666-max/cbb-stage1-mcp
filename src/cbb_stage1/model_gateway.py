"""Unified model gateway: every external model call goes through one scheduler.

The loaded-skill agent owns the policy (provider/model/concurrency/token budget)
and can dispatch one-off calls through the ``cli model`` command.  Extraction,
NLI and auxiliary passes all receive adapters from this gateway instead of
constructing their own clients.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from cbb_stage1.config import Config
from cbb_stage1.engines._async import run_coroutine_sync
from cbb_stage1.engines.ai_reader.llm_client import (
    get_llm_client,
    load_settings_from_env,
)

__all__ = [
    "DEFAULT_EMBEDDING_TIMEOUT",
    "EmbeddingError",
    "EmbeddingLookupWarning",
    "EmbeddingModelUnavailable",
    "ModelPolicy",
    "ModelRequest",
    "ModelResult",
    "ModelScheduler",
    "ModelGateway",
    "GatewayAsyncClient",
    "GatewaySyncLLM",
    "build_model_gateway",
    "embedding_timeout",
    "load_policy",
]

POLICY_RELATIVE = "plan/model-policy.json"
AUDIT_RELATIVE = "plan/model-calls.jsonl"

#: Embedding calls are slow on a cold local backend (a 4096-dim model measured
#: 43.7 s including load); the old hard-coded 60 s client aborted the derive
#: stage whenever the model had to be JIT reloaded. 180 s is the new default.
DEFAULT_EMBEDDING_TIMEOUT = 180.0


class EmbeddingError(RuntimeError):
    """An embedding call failed; the message carries model/url/elapsed context."""


class EmbeddingModelUnavailable(EmbeddingError):
    """No *loaded* embedding model could be resolved (never guessed)."""


class EmbeddingLookupWarning(UserWarning):
    """The embedding backend's state endpoint answered, but unusably."""


def _now() -> float:
    return time.monotonic()


def embedding_timeout() -> float:
    """Return the embedding HTTP timeout in seconds.

    Reads ``CBB_EMBEDDING_TIMEOUT`` (seconds, float accepted) and defaults to
    :data:`DEFAULT_EMBEDDING_TIMEOUT` (180 s) when unset. A malformed or
    non-positive value is an explicit :class:`EmbeddingError`, never a silent
    fallback to 60 s.
    """
    raw = (os.environ.get("CBB_EMBEDDING_TIMEOUT") or "").strip()
    if not raw:
        return DEFAULT_EMBEDDING_TIMEOUT
    try:
        value = float(raw)
    except ValueError as exc:
        raise EmbeddingError(
            f"CBB_EMBEDDING_TIMEOUT must be a number of seconds, got {raw!r}"
        ) from exc
    if value <= 0:
        raise EmbeddingError(
            f"CBB_EMBEDDING_TIMEOUT must be > 0 seconds, got {raw!r}"
        )
    return value


def _backend_root(base: str) -> str:
    """Backend root URL: ``http://host:port/v1`` -> ``http://host:port``.

    LM Studio exposes the model *state* on the unversioned admin API
    (``/api/v0/models``); appending it to an already-``/v1`` base produced
    ``/v1/api/v0/models``, which answers ``200`` with an error body.
    """
    root = (base or "").strip().rstrip("/")
    if root.lower().endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class ModelPolicy:
    provider: str = ""
    model: str = ""
    max_concurrency: int = 4
    max_tokens: int = 4096
    timeout: int = 120
    token_budget: int | None = None
    #: Default: the host agent's own session subagents serve the main reasoning.
    transport: str = "host"
    host_queue: str = ""
    mcp_command: str = ""
    mcp_tool: str = "model_generate"
    embedding_base_url: str = ""
    embedding_model: str = ""
    role_models: dict[str, str] = field(default_factory=dict)
    #: Human/agent-supplied label for whoever is serving the main reasoning.
    #:
    #: The host transport is "the agent answers", so the real model changes with
    #: the agent — the same run may be served by one model today and another
    #: tomorrow. Recording a fixed ``"session"`` made ``model-calls.jsonl`` unable
    #: to answer "which model produced this box", and it made the D4 decoupling
    #: check unanswerable: two calls could be the same model or different models
    #: and the audit could not tell. The label is the caller's declaration; it is
    #: not probed, because the host protocol does not expose model identity.
    session_model_label: str = ""
    #: Separate endpoints for the roles that must NOT reuse the main model.
    judge_base_url: str = ""
    judge_model: str = ""
    nli_base_url: str = ""
    nli_model: str = ""
    #: Whether the configured backend can return token-level signals at all.
    whitebox_backend: bool = False
    #: How many samples semantic entropy draws per assertion.
    #:
    #: This is the dominant cost of the validate stage: every assertion costs
    #: `confidence_samples` model calls. Measured on a 931-character unit, one run
    #: issued 92 restatement calls for 11 claims — 60% of all requests — because the
    #: default was 8. Entropy needs >= 2 samples; the paper uses 5-10. 4 is a
    #: defensible default for a pipeline that runs a whole book: lower it to 2 for
    #: smoke runs, raise it when a confidence number is being audited.
    confidence_samples: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelPolicy":
        return cls(
            provider=str(raw.get("provider") or ""),
            model=str(raw.get("model") or ""),
            transport=str(raw.get("transport") or "host"),
            host_queue=str(raw.get("host_queue") or ""),
            mcp_command=str(raw.get("mcp_command") or ""),
            mcp_tool=str(raw.get("mcp_tool") or "model_generate"),
            embedding_base_url=str(raw.get("embedding_base_url") or ""),
            embedding_model=str(raw.get("embedding_model") or ""),
            max_concurrency=max(1, int(raw.get("max_concurrency") or 4)),
            max_tokens=max(1, int(raw.get("max_tokens") or 4096)),
            timeout=max(1, int(raw.get("timeout") or 120)),
            token_budget=(
                None
                if raw.get("token_budget") is None
                else max(0, int(raw["token_budget"]))
            ),
            role_models={str(k): str(v) for k, v in (raw.get("role_models") or {}).items()},
            session_model_label=str(raw.get("session_model_label") or ""),
            judge_base_url=str(raw.get("judge_base_url") or ""),
            judge_model=str(raw.get("judge_model") or ""),
            nli_base_url=str(raw.get("nli_base_url") or ""),
            nli_model=str(raw.get("nli_model") or ""),
            whitebox_backend=bool(raw.get("whitebox_backend") or False),
            confidence_samples=max(2, int(raw.get("confidence_samples") or 4)),
        )


def load_policy(work_root: str | Path | None) -> ModelPolicy:
    if work_root is None:
        return ModelPolicy()
    path = Path(work_root) / POLICY_RELATIVE
    if not path.exists():
        return ModelPolicy()
    try:
        return ModelPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - malformed policy must not crash the run
        return ModelPolicy()


@dataclass
class ModelRequest:
    request_id: str
    kind: str  # chat | embed | nli
    role: str
    system: str
    prompt: str
    max_tokens: int | None = None
    temperature: float = 0.1
    timeout: int | None = None
    format: Any = None
    num_ctx: int | None = None
    #: Number of input texts for ``kind="embed"`` (0 for chat/nli).
    texts: int = 0


@dataclass
class ModelResult:
    request_id: str
    kind: str
    role: str
    status: str
    text: str = ""
    usage: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0
    provider: str = ""
    model: str = ""
    #: Number of texts sent (embedding calls only; audit field).
    texts: int | None = None
    #: Vector width of the first returned embedding (audit field).
    dimensions: int | None = None
    #: Returned vectors for embedding calls (never serialised to the audit).
    vectors: Any = None


class ModelScheduler:
    """Concurrency/token-budget policy plus an append-only audit of calls."""

    def __init__(
        self,
        policy: ModelPolicy,
        *,
        audit_path: str | Path | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        self.policy = policy
        self.audit_path = Path(audit_path) if audit_path else None
        self.provider = provider
        self.model = model
        self._sem = threading.BoundedSemaphore(policy.max_concurrency)
        self._async_sem: asyncio.Semaphore | None = None
        self._async_loop: Any = None
        self._lock = threading.Lock()
        self._used_tokens = 0
        self._calls = 0

    def _record(self, result: ModelResult) -> None:
        with self._lock:
            self._calls += 1
            if self.audit_path is not None:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                row: dict[str, Any] = {
                    "request_id": result.request_id,
                    "kind": result.kind,
                    "role": result.role,
                    "status": result.status,
                    "provider": result.provider,
                    "model": result.model,
                    "elapsed_ms": round(result.elapsed_ms, 2),
                    "error": result.error,
                }
                # Token accounting. Kept out of the row when absent so "not
                # reported" stays distinguishable from "zero tokens", and
                # normalised because transports hand back either a dict or a
                # SimpleNamespace (the latter is not JSON-serialisable and broke
                # the audit writer when passed through unchanged).
                usage = _usage_mapping(getattr(result, "usage", None))
                if usage is not None:
                    row["usage"] = usage
                # Embedding-specific audit fields (chat records keep their shape).
                if result.texts is not None:
                    row["texts"] = result.texts
                if result.dimensions is not None:
                    row["dimensions"] = result.dimensions
                with self.audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
        self._record_trace(result)

    def _record_trace(self, result: ModelResult) -> None:
        """Feed the LLM-call tracer/telemetry — the hook that had no caller.

        ``tracer.log_llm_call`` existed but nothing ever invoked it, so a run
        with 39 real chat calls still printed ``[trace] ended: (0 LLM calls)``
        and telemetry reported ``llm_calls: 0``. Every transport (host queue,
        MCP, HTTP) funnels through ``_record``, which makes it the only place
        that can count truthfully. The prompt is not available at this layer, so
        it is recorded empty rather than invented.
        """
        try:
            from cbb_stage1.orchestrator import telemetry as _telemetry
            from cbb_stage1.orchestrator.tracer import log_llm_call
        except Exception:  # noqa: BLE001 - tracing must never break a run
            return
        try:
            _telemetry.increment("llm_calls")
            _telemetry.increment(f"llm_calls_{result.kind}")
            log_llm_call(
                model_name=result.model or "",
                prompt="",
                response=result.text or "",
                duration_ms=float(result.elapsed_ms or 0.0),
                success=result.status == "ok",
                error=result.error,
            )
        except Exception:  # noqa: BLE001
            return

    def _check_budget(self) -> None:
        if self.policy.token_budget is not None and self._used_tokens >= self.policy.token_budget:
            raise RuntimeError(
                f"model token budget exhausted ({self._used_tokens}/{self.policy.token_budget})"
            )

    def _add_usage(self, usage: Any) -> None:
        total = getattr(usage, "total_tokens", None)
        if total is None and isinstance(usage, dict):
            total = usage.get("total_tokens")
        try:
            self._used_tokens += int(total or 0)
        except (TypeError, ValueError):
            pass

    async def achat(self, client: Any, request: ModelRequest) -> ModelResult:
        self._check_budget()
        # ``run_coroutine_sync`` runs every call on a fresh event loop, and an
        # asyncio.Semaphore is bound to the loop that first awaited it. Reusing
        # one across loops raises "bound to a different event loop" — rebuild it
        # whenever the running loop changes.
        loop = asyncio.get_running_loop()
        if self._async_sem is None or self._async_loop is not loop:
            self._async_sem = asyncio.Semaphore(self.policy.max_concurrency)
            self._async_loop = loop
        started = _now()
        async with self._async_sem:
            try:
                raw = await client.generate(
                    request.system,
                    request.prompt,
                    format=request.format,
                    max_tokens=request.max_tokens or self.policy.max_tokens,
                    temperature=request.temperature,
                    timeout=request.timeout or self.policy.timeout,
                    num_ctx=request.num_ctx,
                )
                text, usage = (raw if isinstance(raw, tuple) else (raw, None))
                result = ModelResult(
                    request.request_id,
                    request.kind,
                    request.role,
                    "ok",
                    str(text),
                    usage=usage,
                    elapsed_ms=(_now() - started) * 1000,
                    provider=self.provider,
                    model=self.model,
                )
            except Exception as exc:  # noqa: BLE001 - endpoint failure is a result
                result = ModelResult(
                    request.request_id,
                    request.kind,
                    request.role,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=(_now() - started) * 1000,
                    provider=self.provider,
                    model=self.model,
                )
        self._add_usage(result.usage)
        self._record(result)
        return result

    def chat(self, client: Any, request: ModelRequest) -> ModelResult:
        with self._sem:
            started = _now()
            try:
                raw = run_coroutine_sync(
                    client.generate(
                        request.system,
                        request.prompt,
                        format=request.format,
                        max_tokens=request.max_tokens or self.policy.max_tokens,
                        temperature=request.temperature,
                        timeout=request.timeout or self.policy.timeout,
                        num_ctx=request.num_ctx,
                    )
                )
                text, usage = (raw if isinstance(raw, tuple) else (raw, None))
                result = ModelResult(
                    request.request_id,
                    request.kind,
                    request.role,
                    "ok",
                    str(text),
                    usage=usage,
                    elapsed_ms=(_now() - started) * 1000,
                    provider=self.provider,
                    model=self.model,
                )
            except Exception as exc:  # noqa: BLE001
                result = ModelResult(
                    request.request_id,
                    request.kind,
                    request.role,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=(_now() - started) * 1000,
                    provider=self.provider,
                    model=self.model,
                )
        self._add_usage(result.usage)
        self._record(result)
        return result

    def embed(
        self,
        call: Callable[[], list[list[float]]],
        request: ModelRequest,
        *,
        model: str = "",
    ) -> ModelResult:
        """Run one embedding call under the scheduler and audit it.

        The HTTP mechanics live in :meth:`ModelGateway.embed`; the scheduler only
        owns concurrency/timing/audit so embeddings get the same
        ``plan/model-calls.jsonl`` record (``kind="embed"``) as chat calls:
        role, resolved model, text count, elapsed time, status and error.
        """
        if request.kind != "embed":
            raise ValueError(f"ModelScheduler.embed expects kind='embed', got {request.kind!r}")
        with self._sem:
            started = _now()
            vectors: list[list[float]] | None = None
            try:
                vectors = call()
                width = len(vectors[0]) if vectors else 0
                result = ModelResult(
                    request.request_id,
                    request.kind,
                    request.role,
                    "ok",
                    elapsed_ms=(_now() - started) * 1000,
                    provider=self.provider,
                    model=model or self.model,
                    texts=request.texts,
                    dimensions=width,
                    vectors=vectors,
                )
            except Exception as exc:  # noqa: BLE001 - endpoint failure is a result
                result = ModelResult(
                    request.request_id,
                    request.kind,
                    request.role,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=(_now() - started) * 1000,
                    provider=self.provider,
                    model=model or self.model,
                    texts=request.texts,
                )
        self._record(result)
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "used_tokens": self._used_tokens,
            "token_budget": self.policy.token_budget,
            "max_concurrency": self.policy.max_concurrency,
            "provider": self.provider,
            "model": self.model,
        }


class GatewayAsyncClient:
    """Async ``generate(system, prompt, ...)`` adapter for engines."""

    def __init__(self, gateway: "ModelGateway", role: str = "extraction") -> None:
        self.gateway = gateway
        self.role = role
        self.provider = gateway.provider
        self.model = gateway.model

    async def generate(self, system: str, prompt: str, **kwargs: Any) -> tuple[str, Any]:
        request = ModelRequest(
            request_id=f"{self.role}:{id(prompt)}",
            kind="chat",
            role=self.role,
            system=system,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens"),
            temperature=float(kwargs.get("temperature", 0.1)),
            timeout=kwargs.get("timeout"),
            format=kwargs.get("format"),
            num_ctx=kwargs.get("num_ctx"),
        )
        result = await self.gateway.scheduler.achat(self.gateway.client_for(self.role), request)
        if result.status != "ok":
            raise RuntimeError(result.error or "model call failed")
        return result.text, result.usage


class GatewaySyncLLM:
    """Sync ``generate``/``sample`` adapter for auxiliary integrations."""

    def __init__(self, gateway: "ModelGateway", role: str = "assistant") -> None:
        self.gateway = gateway
        self.role = role
        self.provider = f"gateway:{gateway.provider}"
        self.model = gateway.model

    def generate(self, prompt: str) -> str:
        request = ModelRequest(
            request_id=f"{self.role}:{id(prompt)}",
            kind="chat",
            role=self.role,
            system="",
            prompt=prompt,
        )
        result = self.gateway.scheduler.chat(self.gateway.client_for(self.role), request)
        if result.status != "ok":
            raise RuntimeError(result.error or "model call failed")
        return result.text

    def sample(self, prompt: str, n: int) -> list[str]:
        return [self.generate(prompt) for _ in range(max(1, int(n)))]


def _usage_mapping(usage: Any) -> dict[str, Any] | None:
    """Normalise a transport's usage payload into a JSON-safe dict.

    Returns ``None`` when there is nothing to record, so callers can keep the
    field absent rather than writing ``{}`` — "not reported" must stay
    distinguishable from "zero tokens".
    """
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return {str(key): value for key, value in usage.items()}
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:  # noqa: BLE001 - fall through to attribute scraping
            pass
    if hasattr(usage, "__dict__"):
        return {
            str(key): value
            for key, value in vars(usage).items()
            if not str(key).startswith("_")
        }
    return {"value": str(usage)}


class ModelGateway:
    def __init__(
        self,
        client: Any,
        *,
        cfg: Config,
        policy: ModelPolicy,
        work_root: str | Path | None = None,
        provider: str = "",
        model: str = "",
        transport_kind: str = "http",
        embedding_base_url: str = "",
        embedding_model: str = "",
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.provider = provider or cfg.provider
        self.model = model or cfg.model
        self.transport_kind = transport_kind
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.work_root = Path(work_root) if work_root is not None else None
        self.scheduler = ModelScheduler(
            policy,
            audit_path=(Path(work_root) / AUDIT_RELATIVE) if work_root else None,
            provider=self.provider,
            model=self.model,
        )
        self._role_clients: dict[str, Any] = {}

    def client_for(self, role: str) -> Any:
        """Role-specific model client when ``policy.role_models`` says so."""
        if self.transport_kind != "http":
            return self.client
        model = self.scheduler.policy.role_models.get(role)
        if not model or model == self.model:
            return self.client
        if model not in self._role_clients:
            settings = load_settings_from_env()
            settings.model = model
            if self.provider not in {"ai_reader", "ai-reader"}:
                settings.provider = self.provider
            self._role_clients[model] = get_llm_client(settings)
        return self._role_clients[model]

    @property
    def async_client(self) -> GatewayAsyncClient:
        return GatewayAsyncClient(self)

    def sync_llm(self, role: str = "assistant") -> GatewaySyncLLM:
        return GatewaySyncLLM(self, role=role)

    def embed(self, texts: list[str], *, role: str = "embedding") -> list[list[float]]:
        """OpenAI-compatible embeddings, executed through the model scheduler.

        Every call lands in ``plan/model-calls.jsonl`` as ``kind="embed"`` with
        the role, the *resolved* model id, the number of texts, the elapsed
        time, the status and any error. The resolved model is also written back
        to ``plan/model-policy.json`` so a finished run can answer "which model
        embedded this" without reading the code.

        The timeout comes from ``CBB_EMBEDDING_TIMEOUT`` (default 180 s): a cold
        local backend can take >60 s to load the model. Failures raise
        :class:`EmbeddingError` with model/url/elapsed context instead of a bare
        httpx exception.
        """
        values = [str(text) for text in texts]
        if not values:
            return []
        base = (
            self.embedding_base_url.strip()
            or os.environ.get("CBB_EMBEDDING_BASE_URL", "").strip()
            or os.environ.get("CBB_LLM_BASE_URL", "").strip()
            or "http://127.0.0.1:8080/v1"
        )
        key = os.environ.get("CBB_EMBEDDING_API_KEY") or os.environ.get("CBB_LLM_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        import httpx

        timeout = embedding_timeout()
        url = f"{base.rstrip('/')}/embeddings"
        with httpx.Client(timeout=timeout) as client:
            model = self.embedding_model.strip() or os.environ.get(
                "CBB_EMBEDDING_MODEL", ""
            ).strip()
            if not model:
                # LM Studio: pick an embedding model that is actually *loaded*.
                # ``/v1/models`` lists everything on disk; the unversioned
                # ``/api/v0/models`` carries the state. Never guess.
                model = _loaded_embedding_model(client, base, headers)

            request = ModelRequest(
                request_id=f"{role}:embed:{len(values)}:{int(_now() * 1000)}",
                kind="embed",
                role=role,
                system="",
                prompt="",
                texts=len(values),
            )

            def _call() -> list[list[float]]:
                response = client.post(
                    url,
                    headers=headers,
                    json={"model": model, "input": values},
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping) or payload.get("error"):
                    raise EmbeddingError(
                        f"embedding endpoint {url} returned an error body: {payload!r}"
                    )
                data = payload.get("data")
                if not isinstance(data, list):
                    raise EmbeddingError(
                        f"embedding endpoint {url} returned no data list: {payload!r}"
                    )
                vectors: list[list[float]] = []
                for item in data:
                    if not isinstance(item, Mapping) or "embedding" not in item:
                        raise EmbeddingError(
                            f"embedding endpoint {url} returned a malformed row: {item!r}"
                        )
                    vectors.append(list(map(float, item["embedding"])))
                if len(vectors) != len(values):
                    raise EmbeddingError(
                        f"embedding count mismatch on {url}: requested {len(values)}, "
                        f"got {len(vectors)}"
                    )
                return vectors

            result = self.scheduler.embed(_call, request, model=model)

        if result.status != "ok":
            raise EmbeddingError(
                f"embedding call failed: model={model!r} url={url} "
                f"texts={len(values)} timeout={timeout:g}s "
                f"elapsed={result.elapsed_ms / 1000.0:.1f}s error={result.error}"
            )
        self._remember_embedding_model(model, base)
        return list(result.vectors or [])

    def _remember_embedding_model(self, model: str, base: str) -> None:
        """Persist the resolved embedding model into ``plan/model-policy.json``."""
        self.embedding_model = model
        self.embedding_base_url = base
        self.scheduler.policy.embedding_model = model
        if self.work_root is None:
            return
        path = self.work_root / POLICY_RELATIVE
        try:
            payload: dict[str, Any] = {}
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    payload = raw
            payload["embedding_model"] = model
            payload["embedding_base_url"] = base
            _atomic_write_text(
                path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        except Exception as exc:  # noqa: BLE001 - a failed write must be visible
            warnings.warn(
                f"could not persist resolved embedding model {model!r} to {path}: "
                f"{type(exc).__name__}: {exc}",
                EmbeddingLookupWarning,
                stacklevel=2,
            )

    def call(self, prompt: str, *, role: str = "assistant", system: str = "") -> ModelResult:
        request = ModelRequest(
            request_id=f"{role}:{id(prompt)}",
            kind="chat",
            role=role,
            system=system,
            prompt=prompt,
        )
        return self.scheduler.chat(self.client_for(role), request)


def _loaded_embedding_model(client: Any, base: str, headers: Mapping[str, str]) -> str:
    """Return the id of a **loaded** embedding model; never guess.

    The state probe targets the backend *root* (``http://host:port``), because
    ``base`` is normally the OpenAI-compatible ``http://host:port/v1`` and
    ``/v1/api/v0/models`` answers ``200`` with an error body. When the probe is
    unusable (error key, no data list, HTTP failure) this emits a diagnosable
    :class:`EmbeddingLookupWarning` and raises :class:`EmbeddingModelUnavailable`
    -- it never silently falls back to ``/v1/models`` (which lists models on
    disk, loaded or not) and never picks an unloaded model.

    Raises:
        EmbeddingModelUnavailable: probe unusable, or no ``state=loaded``
            embedding model to pick.
    """
    root = _backend_root(base)
    url = f"{root}/api/v0/models"
    try:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - surfaced as an explicit error
        raise EmbeddingModelUnavailable(
            f"cannot determine a loaded embedding model: GET {url} failed "
            f"({type(exc).__name__}: {exc}); refusing to guess from /v1/models "
            "because it lists models on disk, not their state"
        ) from exc

    if not isinstance(payload, Mapping) or payload.get("error") or not isinstance(
        payload.get("data"), list
    ):
        detail = payload.get("error") if isinstance(payload, Mapping) else payload
        warnings.warn(
            f"embedding state probe {url} returned no model list (payload={detail!r}); "
            "refusing to fall back to /v1/models because that endpoint does not "
            "expose state=loaded",
            EmbeddingLookupWarning,
            stacklevel=2,
        )
        raise EmbeddingModelUnavailable(
            f"embedding state probe {url} is unusable (payload={detail!r}); "
            "cannot pick a loaded embedding model without guessing"
        )

    rows = [item for item in payload["data"] if isinstance(item, Mapping)]
    loaded = [
        str(item.get("id") or "")
        for item in rows
        if str(item.get("state") or "").lower() == "loaded"
        and (
            "embed" in str(item.get("id") or "").lower()
            or str(item.get("type") or "").lower() == "embeddings"
        )
    ]
    if loaded:
        return loaded[0]

    available = ", ".join(sorted(str(item.get("id") or "") for item in rows)) or "<none>"
    raise EmbeddingModelUnavailable(
        f"no loaded embedding model on {root}: GET {url} lists [{available}] and "
        "none has state=loaded; load one (LM Studio: `lms load <model>`) or set "
        "embedding_model in plan/model-policy.json"
    )


def build_model_gateway(cfg: Config, work_root: str | Path | None = None) -> ModelGateway | None:
    """Build the gateway from the host-owned model policy.

    The policy (``work/plan/model-policy.json``) chooses the transport and its
    parameters.  Environment variables are only a compatibility fallback for
    ``transport=auto``; the agent host can change routing without editing code.
    """
    policy = load_policy(work_root)
    transport = policy.transport.strip().lower() or "host"

    # Offline/mock runs never build a gateway (they use the mock engine).
    provider = (cfg.provider or "").strip().lower()
    if provider in {"", "mock"} or (cfg.model or "").strip().lower() == "mock":
        return None

    # Host-backed reasoning only makes sense for a live run: offline runs would
    # block waiting for a host reply that is never coming.
    if not getattr(cfg, "live", False):
        return None

    if transport == "auto":
        from cbb_stage1.agents.host_queue import build_host_queue_client
        from cbb_stage1.agents.mcp_model_client import build_mcp_model_client

        if policy.host_queue or os.environ.get("CBB_HOST_MODEL_QUEUE", "").strip():
            transport = "host"
        elif policy.mcp_command or os.environ.get("CBB_MCP_MODEL_COMMAND", "").strip():
            transport = "mcp"
        else:
            transport = "host"  # default: the host agent's session subagents

    if transport not in {"host", "mcp", "http"}:
        raise ValueError(f"unknown model transport: {transport!r}")

    if transport == "host":
        from cbb_stage1.agents.host_queue import build_host_queue_client

        raw_queue = policy.host_queue or os.environ.get("CBB_HOST_MODEL_QUEUE", "").strip()
        host_queue = build_host_queue_client(work_root, raw=raw_queue or "1")
        if host_queue is None:
            raise RuntimeError("model policy transport=host but host_queue is empty")
        return ModelGateway(
            host_queue,
            cfg=cfg,
            policy=policy,
            work_root=work_root,
            provider="host-agent",
            model=policy.session_model_label or "session",
            transport_kind="host",
            embedding_base_url=policy.embedding_base_url,
            embedding_model=policy.embedding_model,
        )

    if transport == "mcp":
        from cbb_stage1.agents.mcp_model_client import build_mcp_model_client

        mcp = build_mcp_model_client(policy.mcp_command or None, policy.mcp_tool or None)
        if mcp is None:
            raise RuntimeError("model policy transport=mcp but mcp_command is empty")
        return ModelGateway(
            mcp,
            cfg=cfg,
            policy=policy,
            work_root=work_root,
            provider="mcp-agent",
            model=policy.session_model_label or "session",
            transport_kind="mcp",
            embedding_base_url=policy.embedding_base_url,
            embedding_model=policy.embedding_model,
        )

    if transport != "http":
        raise ValueError(f"unknown model transport: {transport!r}")

    try:
        settings = load_settings_from_env()
        if cfg.model and cfg.model != "mock":
            settings.model = cfg.model
        if provider not in {"ai_reader", "ai-reader"}:
            settings.provider = provider
        client = get_llm_client(settings)
        return ModelGateway(
            client,
            cfg=cfg,
            policy=policy,
            work_root=work_root,
            provider=provider,
            model=settings.model,
            embedding_base_url=policy.embedding_base_url,
            embedding_model=policy.embedding_model,
        )
    except Exception:  # noqa: BLE001 - unconfigured live provider is explicit
        return None
