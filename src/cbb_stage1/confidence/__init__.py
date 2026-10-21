"""Confidence methods for stage-1 extraction (design ``01`` §5.15; absorption ``07`` B2–B5).

This package is the **implementation face** of the §5.15 confidence contract.  It
only produces a :class:`ConfidenceResult`; it never writes the box and never
fabricates a number:

* ``semantic_entropy`` (B2) — sample the same claim N times, cluster answers in
  meaning space, entropy → confidence.
* :mod:`cbb_stage1.confidence.sep` (B3) — white-box approximation of semantic
  entropy from hidden states (or token log-probs); unavailable without white-box
  access.  It reports ``method="semantic_entropy"`` with
  ``details["approximation"]="sep"`` because §5.15 has no separate ``sep`` value
  in its closed enum.
* :mod:`cbb_stage1.confidence.uqlm` (B4) — UQLM three ways: white-box token
  probabilities (``uqlm_whitebox``), black-box sampling agreement
  (``uqlm_blackbox``), and an LLM judge (closest closed-enum value is
  ``claim_qa``; ``details["uqlm_class"]="judge"`` keeps the UQLM name visible).
* :mod:`cbb_stage1.confidence.uq_heads` (B5) — inference adapter for an already
  trained uncertainty head.  Without a head/checkpoint+loader it records
  ``status="external"`` and leaves ``confidence`` as ``None`` (EXTERNAL is a
  real result, not a fabricated score).

Unified entry point::

    from cbb_stage1.confidence import score
    result = score(record, llm=llm)                    # auto-pick available method
    result = score(record, method="semantic_entropy", llm=llm)
    result = score(record, method="uqlm_blackbox", llm=llm)

Every result is one of:

======== ==============================================================
``ok``   a real score in ``[0, 1]`` with a method from the closed enum;
``unavailable`` the required input is missing (no LLM / no white-box / …);
``external`` the method needs an artifact outside this codebase (trained head);
``error``    the adapter ran but its output could not be trusted (never scored).
======== ==============================================================

The closed method enum is duplicated here (as a leaf module, to avoid importing
the assembly layer) and guarded against drift by
``tests/unit/test_confidence_methods.py::test_method_enum_matches_assemble``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "CONFIDENCE_METHODS",
    "IMPLEMENTED_METHODS",
    "STATUS_ERROR",
    "STATUS_EXTERNAL",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "ConfidenceResult",
    "available_methods",
    "score",
]

#: §5.15 closed enum (same values as ``assemble.assertions.CONFIDENCE_METHODS``).
CONFIDENCE_METHODS: tuple[str, ...] = (
    "engine_self",
    "evidence_span",
    "uqlm_whitebox",
    "uqlm_blackbox",
    "semantic_entropy",
    "uq_heads",
    "claim_qa",
    "human",
    "calibrated",
)

#: Methods this package actually implements (B2–B5).
IMPLEMENTED_METHODS: tuple[str, ...] = (
    "semantic_entropy",
    "uqlm_whitebox",
    "uqlm_blackbox",
    "claim_qa",
    "uq_heads",
)

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_EXTERNAL = "external"
STATUS_ERROR = "error"
_STATUSES = (STATUS_OK, STATUS_UNAVAILABLE, STATUS_EXTERNAL, STATUS_ERROR)
_SCOPES = ("assertion", "claim")


@dataclass(frozen=True)
class ConfidenceResult:
    """One confidence reading.  A score exists **only** when ``status == "ok"``."""

    method: str | None
    confidence: float | None
    status: str = STATUS_OK
    scope: str = "assertion"
    details: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError("status 不属闭集 %s：%r" % (_STATUSES, self.status))
        if self.scope not in _SCOPES:
            raise ValueError("scope 不属闭集 %s：%r" % (_SCOPES, self.scope))
        if self.status == STATUS_OK:
            if self.method not in CONFIDENCE_METHODS:
                raise ValueError("ok 结果的 method 必须在 §5.15 闭集内：%r" % (self.method,))
            value = self.confidence
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("ok 结果必须有 [0,1] 浮点 confidence：%r" % (value,))
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("confidence 超出 [0,1]：%r" % (value,))
        else:
            if self.confidence is not None:
                raise ValueError("%s 结果不许带分数（不伪造）：%r" % (self.status, self.confidence))
            if not self.reason:
                raise ValueError("%s 结果必须给 reason" % self.status)

    @property
    def available(self) -> bool:
        return self.status == STATUS_OK

    @classmethod
    def unavailable(cls, reason: str, *, method: str | None = None,
                    scope: str = "assertion", **details: Any) -> "ConfidenceResult":
        return cls(method=method, confidence=None, status=STATUS_UNAVAILABLE,
                   scope=scope, details=dict(details), reason=reason)

    @classmethod
    def external(cls, reason: str, *, method: str | None = None,
                 scope: str = "assertion", **details: Any) -> "ConfidenceResult":
        return cls(method=method, confidence=None, status=STATUS_EXTERNAL,
                   scope=scope, details=dict(details), reason=reason)

    @classmethod
    def error(cls, reason: str, *, method: str | None = None,
              scope: str = "assertion", **details: Any) -> "ConfidenceResult":
        return cls(method=method, confidence=None, status=STATUS_ERROR,
                   scope=scope, details=dict(details), reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "confidence": self.confidence,
            "status": self.status,
            "scope": self.scope,
            "reason": self.reason,
            "details": dict(self.details),
        }


# ── record / method helpers ────────────────────────────────────────────────
def as_mapping(record: Any) -> dict[str, Any]:
    """Accept a pydantic model or a Mapping (mirrors ``validate.confidence``)."""
    if hasattr(record, "model_dump"):
        return dict(record.model_dump())
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("record 必须是 Mapping 或 pydantic 模型，现得 %r" % (type(record),))


def _normalize_method(method: str | None) -> str | None:
    if not method:
        return None
    value = str(method).strip()
    if not value or value == "auto":
        return None
    return value


def _sampling_llm(llm: Any) -> bool:
    return llm is not None and any(hasattr(llm, name) for name in ("sample", "generate"))


def available_methods(llm: Any = None, **kwargs: Any) -> tuple[str, ...]:
    """Deterministic, input-driven list of methods that *could* run right now."""
    methods: list[str] = []
    if kwargs.get("head") is not None or kwargs.get("checkpoint"):
        methods.append("uq_heads")
    if kwargs.get("hidden_states") is not None or hasattr(llm, "hidden_states"):
        methods.append("semantic_entropy")  # sep path, approximation recorded in details
    if kwargs.get("logprobs") is not None or hasattr(llm, "token_logprobs") or hasattr(llm, "logprobs"):
        methods.append("uqlm_whitebox")
    if kwargs.get("sampler") is not None or _sampling_llm(llm):
        methods.append("semantic_entropy")
    if kwargs.get("judge_llm") is not None:
        methods.append("claim_qa")
    # de-dup, keep priority order
    return tuple(dict.fromkeys(methods))


def _auto_method(llm: Any, kwargs: Mapping[str, Any]) -> str | None:
    """Auto priority: trained head → white-box hidden → white-box logprob → sampling → judge."""
    if kwargs.get("head") is not None or kwargs.get("checkpoint"):
        return "uq_heads"
    if kwargs.get("hidden_states") is not None:
        return "sep"
    if kwargs.get("logprobs") is not None:
        return "uqlm_whitebox"
    if hasattr(llm, "hidden_states"):
        return "sep"
    if hasattr(llm, "token_logprobs") or hasattr(llm, "logprobs"):
        return "uqlm_whitebox"
    if kwargs.get("sampler") is not None or _sampling_llm(llm):
        return "semantic_entropy"
    if kwargs.get("judge_llm") is not None:
        return "claim_qa"
    return None


def score(record: Any, llm: Any = None, method: str | None = None, **kwargs: Any) -> ConfidenceResult:
    """Unified confidence entry (B2–B5).

    ``method`` may be an exact §5.15 value implemented here, ``"sep"`` (B3 white-box
    approximation), or ``None``/``"auto"``.  Methods outside this package's scope
    return an explicit ``unavailable`` result instead of raising, so callers can
    record the gap without a fabricated number.
    """
    rec = as_mapping(record)
    resolved = _normalize_method(method)
    if resolved is None:
        resolved = _auto_method(llm, kwargs)
        if resolved is None:
            return ConfidenceResult.unavailable(
                "没有可用的置信方法：需要 llm（可采样/白盒）、logprobs/hidden_states、"
                "judge_llm 或已训练头（本包不产生无来源的分数）",
            )
    if resolved == "sep":
        from . import sep as _sep
        return _sep.score(rec, llm, **kwargs)
    if resolved == "semantic_entropy":
        from . import semantic_entropy as _se
        return _se.score(rec, llm, **kwargs)
    if resolved in ("uqlm", "uqlm_whitebox", "uqlm_blackbox"):
        from . import uqlm as _uqlm
        mode = {"uqlm_whitebox": "whitebox", "uqlm_blackbox": "blackbox"}.get(resolved)
        if mode is not None and kwargs.get("mode") not in (None, mode):
            raise ValueError("method=%r 与 mode=%r 冲突" % (resolved, kwargs["mode"]))
        return _uqlm.score(rec, llm, mode=mode, **kwargs)
    if resolved == "claim_qa":
        from . import uqlm as _uqlm
        return _uqlm.score(rec, llm, mode="judge", **kwargs)
    if resolved == "uq_heads":
        from . import uq_heads as _uq
        return _uq.score(rec, llm, **kwargs)
    return ConfidenceResult.unavailable(
        "method=%r 不在本包实现范围（B2–B5：semantic_entropy/sep/uqlm_*/uq_heads；"
        "§5.15 的其余方法由别的组件填）" % (resolved,),
        method=resolved,
    )
