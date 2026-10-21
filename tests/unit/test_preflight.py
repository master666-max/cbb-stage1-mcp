"""Preflight audit: every verdict must be backed by the probe that produced it.

The audit exists because the project had no dependency check at all: a dead
embedding endpoint or an already-existing box directory only surfaced at the
stage that needed it.  These tests pin the verdicts *and* the probe plumbing —
in particular that the LM Studio state endpoint is addressed at the root
(``/api/v0/models``) rather than appended to a base that already ends in ``/v1``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cbb_stage1 import preflight


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Records the URL it was asked for; serves a per-test payload."""

    payload: object = {"data": []}
    last_url: str = ""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def get(self, url: str):
        type(self).last_url = url
        return _FakeResponse(type(self).payload)


@pytest.fixture()
def fake_backend(monkeypatch):
    _FakeClient.last_url = ""
    _FakeClient.payload = {
        "data": [
            {
                "id": "text-embedding-qwen3-embedding-8b@q4_k_m",
                "type": "embeddings",
                "state": "loaded",
            },
            {"id": "some-llm", "type": "llm", "state": "not-loaded"},
        ]
    }
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return _FakeClient


def test_out_dir_that_already_exists_is_blocked(tmp_path):
    """A box is immutable; starting a run into an existing out guarantees failure."""
    out = tmp_path / "box"
    out.mkdir()
    report = preflight.run_preflight(
        book=None,
        work=tmp_path / "work",
        out=out,
        skip={"embedding_backend", "mcporter"},
    )
    assert report["status"] == "BLOCKED"
    assert "out_dir" in report["blocked"]
    check = next(item for item in report["checks"] if item["name"] == "out_dir")
    assert check["evidence"]["exists"] is True


def test_missing_book_is_blocked(tmp_path):
    report = preflight.run_preflight(
        book=tmp_path / "nope.epub",
        work=tmp_path / "work",
        out=tmp_path / "box",
        skip={"embedding_backend", "mcporter"},
    )
    assert "book" in report["blocked"]


def test_book_probe_carries_a_recomputable_digest(tmp_path):
    book = tmp_path / "book.epub"
    book.write_bytes(b"PK\x03\x04payload")
    report = preflight.run_preflight(
        book=book,
        work=tmp_path / "work",
        out=tmp_path / "box",
        skip={"embedding_backend", "mcporter"},
    )
    check = next(item for item in report["checks"] if item["name"] == "book")
    assert check["status"] == preflight.READY
    assert check["evidence"]["bytes"] == len(b"PK\x03\x04payload")
    assert len(check["evidence"]["sha256_16"]) == 16


def test_embedding_probe_hits_the_root_state_endpoint(tmp_path, fake_backend):
    """LM Studio's state endpoint lives at the ROOT, not under /v1.

    Appending ``/api/v0/models`` to a base of ``.../v1`` produced
    ``/v1/api/v0/models``, which LM Studio answers with HTTP 200 *and* an error
    body — so ``raise_for_status`` never fired and the "only pick a loaded model"
    guarantee silently degraded to "pick any model on disk".
    """
    check = preflight._probe_embedding("http://127.0.0.1:8080/v1")
    assert check["status"] == preflight.READY
    assert check["evidence"]["probe_url"].endswith("/api/v0/models")
    assert "/v1/api/v0/models" not in check["evidence"]["probe_url"]
    assert fake_backend.last_url == "http://127.0.0.1:8080/api/v0/models"
    assert check["evidence"]["loaded_embedding_models"] == [
        "text-embedding-qwen3-embedding-8b@q4_k_m"
    ]


def test_embedding_probe_blocks_when_nothing_is_loaded(fake_backend):
    fake_backend.payload = {
        "data": [
            {"id": "text-embedding-x", "type": "embeddings", "state": "not-loaded"},
            {"id": "some-llm", "type": "llm", "state": "loaded"},
        ]
    }
    check = preflight._probe_embedding("http://127.0.0.1:8080/v1")
    assert check["status"] == preflight.BLOCKED
    assert check["evidence"]["loaded_other_models"] == ["some-llm"]


def test_embedding_probe_blocks_when_backend_is_unreachable(monkeypatch):
    class Boom:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "Client", Boom)
    check = preflight._probe_embedding("http://127.0.0.1:8080/v1")
    assert check["status"] == preflight.BLOCKED
    assert "ConnectError" in check["evidence"]["error"]


def test_skipped_probes_are_reported_not_dropped(tmp_path, fake_backend):
    book = tmp_path / "book.epub"
    book.write_bytes(b"payload")
    report = preflight.run_preflight(
        book=book,
        work=tmp_path / "work",
        out=tmp_path / "box",
        skip={"mcporter"},
    )
    names = {item["name"] for item in report["checks"]}
    assert "mcporter" in names
    skipped = next(item for item in report["checks"] if item["name"] == "mcporter")
    assert skipped["status"] == preflight.N_A
    assert skipped["evidence"]["skipped"] is True


def test_ready_report_when_every_probe_passes(tmp_path, fake_backend, monkeypatch):
    # The host queue is the project's default transport: the agent answers model
    # calls, so no HTTP chat endpoint is needed. Without declaring it the audit
    # correctly blocks (a run with neither would fall back to the mock engine).
    monkeypatch.setenv("CBB_HOST_MODEL_QUEUE", "1")
    book = tmp_path / "book.epub"
    book.write_bytes(b"payload")
    report = preflight.run_preflight(
        book=book,
        work=tmp_path / "work",
        out=tmp_path / "box",
    )
    assert report["status"] == preflight.READY, json.dumps(report, ensure_ascii=False)
    assert report["blocked"] == []


def test_llm_endpoint_blocks_without_host_queue_or_base_url(tmp_path, fake_backend, monkeypatch):
    """Neither transport configured means the run silently extracts nothing."""
    monkeypatch.delenv("CBB_HOST_MODEL_QUEUE", raising=False)
    monkeypatch.delenv("CBB_LLM_BASE_URL", raising=False)
    check = preflight._probe_llm_endpoint(None, None)
    assert check["status"] == preflight.BLOCKED
    assert "mock engine" in check["detail"]


def test_llm_endpoint_accepts_host_queue_without_http(tmp_path, monkeypatch):
    monkeypatch.setenv("CBB_HOST_MODEL_QUEUE", "1")
    check = preflight._probe_llm_endpoint(None, None)
    assert check["status"] == preflight.READY
    assert check["evidence"]["host_queue"] is True


def test_llm_endpoint_blocks_when_the_model_is_not_listed(tmp_path, fake_backend, monkeypatch):
    monkeypatch.delenv("CBB_HOST_MODEL_QUEUE", raising=False)
    fake_backend.payload = {"data": [{"id": "some-other-model"}]}
    check = preflight._probe_llm_endpoint("http://127.0.0.1:8080/v1", "wanted-model")
    assert check["status"] == preflight.BLOCKED
    assert "not listed" in check["detail"]


def test_uq_checkpoint_probes_the_artifact_not_a_service(tmp_path, monkeypatch):
    """EXT-06 is a file. A reachable endpoint says nothing about it."""
    monkeypatch.delenv("CBB_UQ_HEAD_CHECKPOINT", raising=False)
    unset = preflight._probe_uq_checkpoint(None)
    assert unset["status"] == preflight.N_A

    missing = preflight._probe_uq_checkpoint(tmp_path / "nope.json")
    assert missing["status"] == preflight.BLOCKED

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    assert preflight._probe_uq_checkpoint(wrong)["status"] == preflight.BLOCKED

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"format": preflight.UQ_CHECKPOINT_FORMAT}), encoding="utf-8")
    ok = preflight._probe_uq_checkpoint(good)
    assert ok["status"] == preflight.READY
    assert ok["evidence"]["format"] == preflight.UQ_CHECKPOINT_FORMAT


def test_uq_train_data_rejects_single_class(tmp_path, monkeypatch):
    """EXT-07: the trainer refuses single-class input, so say it before it runs."""
    monkeypatch.delenv("CBB_UQ_TRAIN_DATA", raising=False)
    data = tmp_path / "labels.jsonl"
    data.write_text(
        "\n".join(json.dumps({"record": {}, "label": 1}) for _ in range(3)) + "\n",
        encoding="utf-8",
    )
    check = preflight._probe_labelled_data(data)
    assert check["status"] == preflight.BLOCKED
    assert "single-class" in check["detail"]

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        "\n".join(
            [json.dumps({"record": {}, "label": 1}), json.dumps({"record": {}, "label": 0})]
        )
        + "\n",
        encoding="utf-8",
    )
    assert preflight._probe_labelled_data(mixed)["status"] == preflight.READY


def test_budget_uses_the_documented_max_tokens_default():
    """19 清单 D.1 documents CBB_LLM_MAX_TOKENS = 8192; unset is not 'no ceiling'."""
    check = preflight._probe_budget(None, None)
    assert check["evidence"]["max_tokens"] == preflight.DEFAULT_LLM_MAX_TOKENS
    assert check["status"] == preflight.OBSERVED  # per-call bounded, per-run not

    assert preflight._probe_budget(8192, 100000)["status"] == preflight.READY


def test_wiring_reports_gaps_rather_than_pretending_live():
    """WIRED != LIVE: an endpoint being up does not mean the injection exists."""
    check = preflight._probe_wiring()
    assert check["status"] == preflight.OBSERVED
    assert check["evidence"]["gaps"]


def test_format_report_keeps_the_evidence_visible(tmp_path):
    out = tmp_path / "box"
    out.mkdir()
    report = preflight.run_preflight(
        book=None,
        work=tmp_path / "work",
        out=out,
        skip={"embedding_backend", "mcporter"},
    )
    text = preflight.format_report(report)
    assert "preflight: BLOCKED" in text
    assert "out_dir" in text
    assert "blocked:" in text
