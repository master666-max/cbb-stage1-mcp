"""The trace must count the calls that actually happened.

``tracer.log_llm_call`` existed but had **no caller**, so a run that made 39 real
chat calls still printed ``[trace] ended: (0 LLM calls)`` and telemetry reported
``llm_calls: 0``. Those numbers are what a reviewer reads, so a zero there is not
a cosmetic bug — it is a false statement about the run.

The fix hooks the one point every transport passes through
(``ModelScheduler._record``), so host-queue, MCP and HTTP calls are all counted.
"""

from __future__ import annotations

import threading

from cbb_stage1 import model_gateway as mg
from cbb_stage1.orchestrator import telemetry, tracer


class _FakeResult:
    def __init__(self, kind: str = "chat", status: str = "ok") -> None:
        self.request_id = "req-1"
        self.kind = kind
        self.role = "extraction"
        self.status = status
        self.text = "answer"
        self.usage = None
        self.elapsed_ms = 12.5
        self.provider = "host-agent"
        self.model = "session"
        self.error = None
        self.texts = 1 if kind == "embed" else None
        self.dimensions = 4096 if kind == "embed" else None


def _scheduler(tmp_path):
    scheduler = mg.ModelScheduler.__new__(mg.ModelScheduler)
    scheduler.audit_path = tmp_path / "plan" / "model-calls.jsonl"
    scheduler._lock = threading.Lock()
    scheduler._calls = 0
    return scheduler


def test_trace_counts_every_recorded_call(tmp_path):
    scheduler = _scheduler(tmp_path)
    tracer.start_trace({"probe": True})
    try:
        scheduler._record(_FakeResult())
        scheduler._record(_FakeResult())
    finally:
        trace = tracer.end_trace()
    assert trace is not None
    assert len(trace.llm_calls) == 2
    assert all(call.success for call in trace.llm_calls)


def test_failed_call_is_counted_and_marked(tmp_path):
    scheduler = _scheduler(tmp_path)
    tracer.start_trace({})
    try:
        scheduler._record(_FakeResult(status="failed"))
    finally:
        trace = tracer.end_trace()
    assert trace is not None
    assert len(trace.llm_calls) == 1
    assert trace.llm_calls[0].success is False


def test_embedding_calls_are_counted_separately(tmp_path):
    scheduler = _scheduler(tmp_path)
    telemetry.reset_telemetry()
    tracer.start_trace({})
    try:
        scheduler._record(_FakeResult(kind="chat"))
        scheduler._record(_FakeResult(kind="embed"))
    finally:
        trace = tracer.end_trace()
    assert trace is not None
    assert len(trace.llm_calls) == 2
    counters = telemetry.get_report()["counters"]
    # ``get_summary`` exposes a fixed key set, so per-kind counters are only
    # visible in the full report.
    assert counters["llm_calls"] == 2
    assert counters["llm_calls_chat"] == 1
    assert counters["llm_calls_embed"] == 1


def test_tracing_never_breaks_a_call(tmp_path, monkeypatch):
    """Tracing is observability; it must not be able to fail a run."""

    def boom(*args, **kwargs):
        raise RuntimeError("tracer exploded")

    monkeypatch.setattr(tracer, "log_llm_call", boom)
    scheduler = _scheduler(tmp_path)
    scheduler._record(_FakeResult())  # must not raise
    assert scheduler._calls == 1
    # The audit row is still written even when tracing fails.
    assert scheduler.audit_path is not None and scheduler.audit_path.exists()
