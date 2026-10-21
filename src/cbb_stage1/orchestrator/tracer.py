"""Tracing - request-scoped trace ids, spans, decisions, LLM-call records.

Ported from B09 ``src/observability/tracer.py`` (read-only source). Shape is
preserved (``Trace`` / ``TracedSpan`` / ``TracedDecision`` / ``TracedLLMCall`` /
``SpanContext`` / module-level active trace). Adaptation: the trace directory is
passed explicitly to :func:`save_trace` / :func:`load_trace` instead of being
read from a repo-global config, so the shell stays injectable for tests.

The trace is a work artifact; it never writes the box.
"""

from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "TracedDecision",
    "TracedLLMCall",
    "TracedSpan",
    "Trace",
    "SpanContext",
    "get_current_trace_id",
    "start_trace",
    "end_trace",
    "get_active_trace",
    "start_span",
    "end_span",
    "log_decision",
    "log_llm_call",
    "record_model_version",
    "save_trace",
    "load_trace",
]


# =============================================================================
# CONTEXT VARIABLE - propagates the trace id through the call stack
# =============================================================================

_current_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def get_current_trace_id() -> str | None:
    """Get the active trace id for the current execution context."""
    return _current_trace_id.get()


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TracedDecision:
    """Record of a decision made during processing."""

    decision_id: str
    category: str
    options_considered: list[str]
    option_chosen: str
    reason: str
    constraints: list[str] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "category": self.category,
            "options_considered": self.options_considered,
            "option_chosen": self.option_chosen,
            "reason": self.reason,
            "constraints": self.constraints,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class TracedLLMCall:
    """Record of an LLM call with prompt/response excerpt for replayability."""

    call_id: str
    model_name: str
    model_version: str | None = None
    prompt: str = ""
    response: str = ""
    temperature: float = 0.0
    seed: int | None = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_length": len(self.prompt),
            "response_length": len(self.response),
            "prompt": self.prompt[:500] + ("..." if len(self.prompt) > 500 else ""),
            "response": self.response[:500] + ("..." if len(self.response) > 500 else ""),
            "temperature": self.temperature,
            "seed": self.seed,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class TracedSpan:
    """A timed span within a trace for structured timing."""

    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        self.end_time = time.monotonic()
        self.status = status
        self.error = error

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


@dataclass
class Trace:
    """Complete trace for one pipeline run."""

    trace_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: str | None = None
    model_versions: dict[str, str] = field(default_factory=dict)
    spans: list[TracedSpan] = field(default_factory=list)
    decisions: list[TracedDecision] = field(default_factory=list)
    llm_calls: list[TracedLLMCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model_versions": self.model_versions,
            "spans": [s.to_dict() for s in self.spans],
            "decisions": [d.to_dict() for d in self.decisions],
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "metadata": self.metadata,
        }


# =============================================================================
# GLOBAL TRACE MANAGEMENT (one active trace per process)
# =============================================================================

_active_trace: Trace | None = None
_span_stack: list[TracedSpan] = []


def start_trace(metadata: dict[str, Any] | None = None) -> str:
    """Start a new trace for a pipeline run and return its hex trace id."""
    global _active_trace, _span_stack
    trace_id = uuid.uuid4().hex[:12]
    _current_trace_id.set(trace_id)
    _active_trace = Trace(trace_id=trace_id, metadata=metadata or {})
    _span_stack = []
    print(f"[trace] started: {trace_id}")
    return trace_id


def end_trace() -> Trace | None:
    """End the active trace and return it (``None`` when no trace is active)."""
    global _active_trace, _span_stack
    if _active_trace is None:
        return None

    _active_trace.ended_at = datetime.now().isoformat()
    trace = _active_trace
    _active_trace = None
    _span_stack = []
    _current_trace_id.set(None)

    print(
        f"[trace] ended: {trace.trace_id} "
        f"({len(trace.spans)} spans, {len(trace.decisions)} decisions, "
        f"{len(trace.llm_calls)} LLM calls)"
    )
    return trace


def get_active_trace() -> Trace | None:
    """Get the currently active trace, if any."""
    return _active_trace


# =============================================================================
# SPAN MANAGEMENT
# =============================================================================

def start_span(name: str, **attributes: Any) -> TracedSpan:
    """Start a span nested under the current span (if any)."""
    parent_id = _span_stack[-1].span_id if _span_stack else None
    span = TracedSpan(name=name, parent_span_id=parent_id, attributes=attributes)
    _span_stack.append(span)
    if _active_trace:
        _active_trace.spans.append(span)
    return span


def end_span(span: TracedSpan, status: str = "ok", error: str | None = None) -> None:
    """Finish a span and pop it from the stack if it is the current one."""
    span.finish(status=status, error=error)
    if _span_stack and _span_stack[-1] is span:
        _span_stack.pop()


class SpanContext:
    """Context manager for automatic span management."""

    def __init__(self, name: str, **attributes: Any):
        self.name = name
        self.attributes = attributes
        self.span: TracedSpan | None = None

    def __enter__(self) -> TracedSpan:
        self.span = start_span(self.name, **self.attributes)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: vulture - __exit__ signature
        if self.span:
            status = "error" if exc_type else "ok"
            error = str(exc_val) if exc_val else None
            end_span(self.span, status=status, error=error)


# =============================================================================
# DECISION LOGGING
# =============================================================================

def log_decision(
    category: str,
    options_considered: list[str],
    option_chosen: str,
    reason: str,
    constraints: list[str] | None = None,
    confidence: float = 1.0,
    **metadata: Any,
) -> TracedDecision:
    """Log a decision made during processing."""
    decision = TracedDecision(
        decision_id=uuid.uuid4().hex[:8],
        category=category,
        options_considered=options_considered,
        option_chosen=option_chosen,
        reason=reason,
        constraints=constraints or [],
        confidence=confidence,
        metadata=metadata,
    )
    if _active_trace:
        _active_trace.decisions.append(decision)
    return decision


# =============================================================================
# LLM CALL LOGGING
# =============================================================================

def log_llm_call(
    model_name: str,
    prompt: str,
    response: str,
    duration_ms: float = 0.0,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    temperature: float = 0.0,
    seed: int | None = None,
    model_version: str | None = None,
    success: bool = True,
    error: str | None = None,
) -> TracedLLMCall:
    """Log an LLM call with prompt/response excerpts for replayability."""
    call = TracedLLMCall(
        call_id=uuid.uuid4().hex[:8],
        model_name=model_name,
        model_version=model_version,
        prompt=prompt,
        response=response,
        duration_ms=duration_ms,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        temperature=temperature,
        seed=seed,
        success=success,
        error=error,
    )
    if _active_trace:
        _active_trace.llm_calls.append(call)
    return call


def record_model_version(model_name: str, digest: str) -> None:
    """Record the exact model digest for replayability."""
    if _active_trace:
        _active_trace.model_versions[model_name] = digest


# =============================================================================
# TRACE PERSISTENCE
# =============================================================================

def save_trace(trace: Trace | None, out_dir: str | Path) -> Path | None:
    """Save a trace as JSON under ``out_dir``; returns the saved path."""
    trace = trace or _active_trace
    if trace is None:
        return None

    traces_dir = Path(out_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{trace.trace_id}.json"
    trace_path.write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"[trace] saved: {trace_path}")
    return trace_path


def load_trace(trace_id: str, out_dir: str | Path) -> Trace | None:
    """Load a trace from ``out_dir`` for inspection."""
    trace_path = Path(out_dir) / f"{trace_id}.json"
    if not trace_path.exists():
        return None

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    return Trace(
        trace_id=data["trace_id"],
        started_at=data["started_at"],
        ended_at=data.get("ended_at"),
        model_versions=data.get("model_versions", {}),
        metadata=data.get("metadata", {}),
    )
