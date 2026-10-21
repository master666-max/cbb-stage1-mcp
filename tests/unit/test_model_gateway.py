from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from cbb_stage1.model_gateway import (
    ModelGateway,
    ModelPolicy,
    build_model_gateway,
)


class _FakeClient:
    async def generate(self, system, prompt, **kwargs):
        return f"echo:{prompt}", SimpleNamespace(total_tokens=3)


def _hub(**kwargs):
    """A gateway wired for embedding tests (no real network)."""
    cfg = SimpleNamespace(provider="host", model="session")
    return ModelGateway(
        _FakeClient(),
        cfg=cfg,
        policy=ModelPolicy(max_concurrency=2),
        work_root=kwargs.get("work_root"),
        provider="host-agent",
        model="session",
        transport_kind="host",
        embedding_model=kwargs.get("embedding_model", "fake-embed-8b"),
    )


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeHTTP:
    """Minimal ``httpx.Client`` stand-in that records requests/timeouts."""

    instances: list["_FakeHTTP"] = []

    def __init__(self, *, timeout=None, **_kwargs):
        self.timeout = timeout
        self.get_urls: list[str] = []
        self.posts: list[dict] = []
        self.post_error: BaseException | None = None
        self.get_payload = {"data": []}
        _FakeHTTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, *, headers=None):
        self.get_urls.append(url)
        return _FakeResponse(self.get_payload)

    def post(self, url, *, headers=None, json=None):
        self.posts.append({"url": url, "json": json})
        if self.post_error is not None:
            raise self.post_error
        items = json.get("input", [])
        return _FakeResponse(
            {"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in items]}
        )


@pytest.fixture
def fake_http(monkeypatch):
    _FakeHTTP.instances = []
    monkeypatch.setattr(httpx, "Client", _FakeHTTP)
    return _FakeHTTP


def test_gateway_sync_and_async_calls_share_scheduler(tmp_path):
    cfg = SimpleNamespace(provider="openai", model="m")
    gateway = ModelGateway(
        _FakeClient(),
        cfg=cfg,
        policy=ModelPolicy(max_concurrency=2),
        work_root=tmp_path,
        provider="openai",
        model="m",
    )
    result = gateway.call("hello", role="verifier")
    assert result.status == "ok"
    assert result.text == "echo:hello"
    assert gateway.scheduler.stats()["calls"] == 1
    async_text = asyncio.run(
        gateway.async_client.generate("sys", "hello2")
    )[0]
    assert async_text == "echo:hello2"
    assert gateway.scheduler.stats()["calls"] == 2
    lines = (tmp_path / "plan" / "model-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["role"] == "verifier"


def test_mock_config_builds_no_gateway():
    cfg = SimpleNamespace(provider="mock", model="mock")
    assert build_model_gateway(cfg, None) is None


# ---------------------------------------------------------------------------
# E1: embed goes through the scheduler and lands in model-calls.jsonl
# ---------------------------------------------------------------------------


def test_embed_is_audited_as_kind_embed_and_persists_model(tmp_path, fake_http):
    """E1 regression: ``embed`` must produce a ``kind="embed"`` audit record.

    Before the fix ``ModelGateway.embed`` posted directly with httpx, so
    ``plan/model-calls.jsonl`` had zero embedding rows and nothing wrote the
    resolved model back to ``plan/model-policy.json``.
    """
    gateway = _hub(work_root=tmp_path)

    vectors = gateway.embed(["文本一", "文本二"])

    assert len(vectors) == 2 and vectors[0] == [0.1, 0.2, 0.3]
    client = fake_http.instances[0]
    assert client.posts[0]["url"] == "http://127.0.0.1:8080/v1/embeddings"
    assert client.posts[0]["json"] == {"model": "fake-embed-8b", "input": ["文本一", "文本二"]}

    rows = [
        json.loads(line)
        for line in (tmp_path / "plan" / "model-calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    audit = rows[0]
    assert audit["kind"] == "embed"
    assert audit["role"] == "embedding"
    assert audit["model"] == "fake-embed-8b"
    assert audit["texts"] == 2
    assert audit["dimensions"] == 3
    assert audit["status"] == "ok"
    assert audit["error"] is None
    assert audit["elapsed_ms"] >= 0
    assert gateway.scheduler.stats()["calls"] == 1

    policy = json.loads((tmp_path / "plan" / "model-policy.json").read_text(encoding="utf-8"))
    assert policy["embedding_model"] == "fake-embed-8b"
    assert policy["embedding_base_url"] == "http://127.0.0.1:8080/v1"


def test_embed_failure_is_audited_with_status_failed(tmp_path, fake_http, monkeypatch):
    gateway = _hub(work_root=tmp_path)

    def _boom(*_args, **_kwargs):
        raise httpx.TimeoutException("read timeout")

    monkeypatch.setattr(fake_http, "post", _boom)

    with pytest.raises(Exception) as info:
        gateway.embed(["x"])
    assert "fake-embed-8b" in str(info.value)
    assert "http://127.0.0.1:8080/v1/embeddings" in str(info.value)
    assert "elapsed=" in str(info.value)
    audit = json.loads(
        (tmp_path / "plan" / "model-calls.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert audit["kind"] == "embed"
    assert audit["status"] == "failed"
    assert "TimeoutException" in audit["error"]


# ---------------------------------------------------------------------------
# E2: the loaded-state probe targets the backend ROOT and never guesses
# ---------------------------------------------------------------------------


def test_state_probe_targets_backend_root_not_v1_api_v0():
    """E2 regression: base ``.../v1`` must probe ``.../api/v0/models``.

    Before the fix the URL was ``.../v1/api/v0/models``, which answers 200 with
    an error body, so the function silently fell back to ``/v1/models``.
    """
    from cbb_stage1.model_gateway import _loaded_embedding_model

    seen: list[str] = []

    class Fake:
        def get(self, url, *, headers=None):
            seen.append(url)
            return _FakeResponse(
                {
                    "data": [
                        {"id": "llada-8b-instruct", "state": "not-loaded", "type": "llm"},
                        {
                            "id": "text-embedding-qwen3-embedding-8b@q4_k_m",
                            "state": "loaded",
                            "type": "embeddings",
                        },
                    ]
                }
            )

    picked = _loaded_embedding_model(Fake(), "http://127.0.0.1:8080/v1", {})
    assert seen == ["http://127.0.0.1:8080/api/v0/models"]
    assert picked == "text-embedding-qwen3-embedding-8b@q4_k_m"


def test_state_probe_error_body_warns_and_never_falls_back(monkeypatch):
    """E2 regression: an error body must warn + raise, not hit ``/v1/models``."""
    from cbb_stage1.model_gateway import _loaded_embedding_model

    seen: list[str] = []

    class Fake:
        def get(self, url, *, headers=None):
            seen.append(url)
            return _FakeResponse(
                {"error": "Unexpected endpoint or method. (GET /v1/api/v0/models)"}
            )

    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError):
            _loaded_embedding_model(Fake(), "http://127.0.0.1:8080/v1", {})
    assert seen == ["http://127.0.0.1:8080/api/v0/models"]


def test_state_probe_raises_when_no_model_is_loaded():
    """E2 regression: no ``state=loaded`` model is an error, not a guess."""
    from cbb_stage1.model_gateway import _loaded_embedding_model

    class Fake:
        def get(self, url, *, headers=None):
            return _FakeResponse(
                {
                    "data": [
                        {"id": "embed-small@q4", "state": "not-loaded", "type": "embeddings"},
                        {"id": "llada-8b-instruct", "state": "not-loaded", "type": "llm"},
                    ]
                }
            )

    with pytest.raises(RuntimeError, match="state=loaded"):
        _loaded_embedding_model(Fake(), "http://127.0.0.1:8080/v1", {})


# ---------------------------------------------------------------------------
# E3: configurable embedding timeout (CBB_EMBEDDING_TIMEOUT, default 180 s)
# ---------------------------------------------------------------------------


def test_embed_default_timeout_is_180(tmp_path, fake_http, monkeypatch):
    """E3 regression: the old client hard-coded ``timeout=60.0``."""
    monkeypatch.delenv("CBB_EMBEDDING_TIMEOUT", raising=False)
    _hub(work_root=tmp_path).embed(["x"])
    assert fake_http.instances[0].timeout == 180.0


def test_embed_timeout_reads_environment(tmp_path, fake_http, monkeypatch):
    monkeypatch.setenv("CBB_EMBEDDING_TIMEOUT", "300")
    _hub(work_root=tmp_path).embed(["x"])
    assert fake_http.instances[0].timeout == 300.0


def test_embed_timeout_error_carries_model_url_and_elapsed(tmp_path, fake_http, monkeypatch):
    gateway = _hub(work_root=tmp_path)

    def _boom(*_args, **_kwargs):
        raise httpx.TimeoutException("read timeout")

    monkeypatch.setattr(fake_http, "post", _boom)

    with pytest.raises(RuntimeError) as info:
        gateway.embed(["x"])
    message = str(info.value)
    assert "model='fake-embed-8b'" in message
    assert "url=http://127.0.0.1:8080/v1/embeddings" in message
    assert "timeout=180s" in message
    assert "elapsed=" in message


def test_embedding_timeout_rejects_malformed_value(monkeypatch):
    from cbb_stage1.model_gateway import EmbeddingError, embedding_timeout

    monkeypatch.setenv("CBB_EMBEDDING_TIMEOUT", "soon")
    with pytest.raises(EmbeddingError):
        embedding_timeout()
