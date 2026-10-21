"""Abductive hypothesis skeleton (derived layer only, no LLM calls).

Implements the v2.2 ``derived/hypotheses.jsonl`` contract from box spec §5.20
and invariant I26:

* hypotheses live **only** in ``derived/``; they never enter ``knowledge/``;
* ``evidence_sufficiency`` is mandatory; ``missing_premises`` is allowed only
  when the evidence is ``insufficient`` (a *sufficient* case must not invent
  gaps);
* :func:`abduce` refuses to run on sufficient evidence — abduction starts only
  after the sufficiency check fails.

This is an interface skeleton: it deterministically builds, validates and
serialises candidate hypotheses. Choosing the hypothesis text / missing
premises needs a model and is a later pass; nothing here calls one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1 import ids

__all__ = [
    "HYPOTHESIS_SCOPES",
    "EVIDENCE_SUFFICIENCY",
    "CONSISTENCY_VALUES",
    "PLAUSIBILITY_VALUES",
    "HYPOTHESIS_STATUSES",
    "Hypothesis",
    "make_hypothesis_id",
    "abduce",
    "derive_hypotheses",
    "normalize_hypothesis_record",
]

#: Closed sets from box spec §5.20 / hypothesis.schema.json.
HYPOTHESIS_SCOPES = ("setting", "foreshadow", "motivation", "prediction")
EVIDENCE_SUFFICIENCY = ("sufficient", "insufficient")
CONSISTENCY_VALUES = ("pass", "fail", "unknown")
PLAUSIBILITY_VALUES = ("high", "medium", "low", "unknown")
HYPOTHESIS_STATUSES = ("candidate", "verified", "rejected")


def make_hypothesis_id(book_id: str, scope: str, claim: str) -> str:
    """Content-addressed hypothesis id: ``hy:{sha16(book|scope|claim)}``.

    Compatibility wrapper over :func:`cbb_stage1.ids.hypothesis_id` (same bytes
    as before).
    """
    return ids.hypothesis_id(book_id, scope, claim)


def _evidence_list(value: Iterable[Mapping[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in (value or ()))


def _validate_enum(value: str, allowed: Sequence[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"{label} must be one of {list(allowed)}, got {value!r}")
    return value


@dataclass(frozen=True)
class Hypothesis:
    """One derived hypothesis (mirrors ``hypothesis.schema.json``)."""

    book_id: str
    scope: str
    claim: str
    evidence_sufficiency: str
    missing_premises: tuple[str, ...] = ()
    support: tuple[dict[str, Any], ...] = ()
    against: tuple[dict[str, Any], ...] = ()
    consistency: str = "unknown"
    plausibility: str = "unknown"
    status: str = "candidate"
    confidence: float = 0.0
    method: str = "abductive_rag"
    derived_from: tuple[str, ...] = ()
    created_at: str | None = None
    hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.book_id):
            raise ValueError("hypothesis book_id is required")
        _validate_enum(self.scope, HYPOTHESIS_SCOPES, "scope")
        if not str(self.claim).strip():
            raise ValueError("hypothesis claim is required")
        _validate_enum(
            self.evidence_sufficiency, EVIDENCE_SUFFICIENCY, "evidence_sufficiency"
        )
        _validate_enum(self.consistency, CONSISTENCY_VALUES, "consistency")
        _validate_enum(self.plausibility, PLAUSIBILITY_VALUES, "plausibility")
        _validate_enum(self.status, HYPOTHESIS_STATUSES, "status")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("hypothesis confidence must be in [0, 1]")
        if self.evidence_sufficiency == "sufficient" and self.missing_premises:
            raise ValueError(
                "a sufficient hypothesis must not carry missing_premises (I26)"
            )
        if self.evidence_sufficiency == "insufficient" and not self.missing_premises:
            raise ValueError(
                "an insufficient hypothesis must name at least one missing premise (I26)"
            )
        if self.hypothesis_id is None:
            object.__setattr__(
                self,
                "hypothesis_id",
                make_hypothesis_id(self.book_id, self.scope, self.claim),
            )
        if not str(self.hypothesis_id or "").startswith("hy:"):
            raise ValueError("hypothesis_id must start with 'hy:'")

    def to_record(self) -> dict[str, Any]:
        """Return the §5.20 JSONL shape (JSON-serialisable, derived-only)."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "book_id": self.book_id,
            "scope": self.scope,
            "claim": self.claim,
            "evidence_sufficiency": self.evidence_sufficiency,
            "missing_premises": list(self.missing_premises),
            "support": [dict(item) for item in self.support],
            "against": [dict(item) for item in self.against],
            "consistency": self.consistency,
            "plausibility": self.plausibility,
            "status": self.status,
            "confidence": float(self.confidence),
            "method": self.method,
            "derived_from": list(self.derived_from),
            "created_at": self.created_at,
        }


def abduce(
    book_id: str,
    claim: str,
    *,
    scope: str = "setting",
    missing_premises: Iterable[str],
    support: Iterable[Mapping[str, Any]] = (),
    against: Iterable[Mapping[str, Any]] = (),
    consistency: str = "unknown",
    plausibility: str = "unknown",
    confidence: float = 0.0,
    derived_from: Iterable[str] = (),
    created_at: str | None = None,
    method: str = "abductive_rag",
) -> Hypothesis:
    """Build one *insufficient-evidence* candidate hypothesis.

    Abduction comes **after** the sufficiency check: this function raises
    ``ValueError`` when ``missing_premises`` is empty, so callers cannot use it
    to dress up a sufficient observation as a hypothesis.
    """
    premises = tuple(str(item) for item in missing_premises if str(item).strip())
    if not premises:
        raise ValueError(
            "abduce() requires missing_premises; use induction for sufficient evidence"
        )
    return Hypothesis(
        book_id=book_id,
        scope=scope,
        claim=claim,
        evidence_sufficiency="insufficient",
        missing_premises=premises,
        support=_evidence_list(support),
        against=_evidence_list(against),
        consistency=consistency,
        plausibility=plausibility,
        status="candidate",
        confidence=confidence,
        method=method,
        derived_from=tuple(str(item) for item in derived_from),
        created_at=created_at,
    )


def derive_hypotheses(
    book_id: str,
    observations: Iterable[Mapping[str, Any]],
    *,
    created_at: str | None = None,
) -> tuple[list[Hypothesis], list[dict[str, Any]]]:
    """Build candidate hypotheses from pre-shaped gap observations.

    Each observation may carry ``claim``, ``scope``, ``support``, ``against``,
    ``missing_premises``, ``consistency``, ``plausibility``, ``confidence`` and
    ``derived_from``. Observations whose evidence is ``sufficient`` (or that
    name no missing premise) are **skipped** and reported with a reason — a
    sufficient case goes through ordinary induction, not abduction.

    Returns:
        ``(hypotheses, skipped)``; both lists preserve input order.
    """
    hypotheses: list[Hypothesis] = []
    skipped: list[dict[str, Any]] = []
    for index, observation in enumerate(observations, 1):
        claim = str(observation.get("claim") or "").strip()
        premises = [
            str(item)
            for item in observation.get("missing_premises", ())
            if str(item).strip()
        ]
        sufficiency = str(observation.get("evidence_sufficiency") or "")
        if sufficiency == "sufficient":
            skipped.append(
                {"index": index, "claim": claim, "reason": "sufficient_evidence"}
            )
            continue
        if not claim:
            skipped.append({"index": index, "claim": claim, "reason": "empty_claim"})
            continue
        if not premises:
            skipped.append(
                {"index": index, "claim": claim, "reason": "no_missing_premises"}
            )
            continue
        hypotheses.append(
            abduce(
                book_id,
                claim,
                scope=str(observation.get("scope") or "setting"),
                missing_premises=premises,
                support=observation.get("support") or (),
                against=observation.get("against") or (),
                consistency=str(observation.get("consistency") or "unknown"),
                plausibility=str(observation.get("plausibility") or "unknown"),
                confidence=float(observation.get("confidence", 0.0) or 0.0),
                derived_from=observation.get("derived_from") or (),
                created_at=created_at,
            )
        )
    return hypotheses, skipped


def normalize_hypothesis_record(
    record: Mapping[str, Any],
    *,
    book_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Validate/normalise one hypothesis record for the derived table.

    Missing optional fields are filled with contract defaults; ids are
    content-addressed when absent. Raises ``ValueError``/``TypeError`` for
    invalid closed-set values or I26 violations so the caller can quarantine.

    This function never promotes a hypothesis into ``knowledge/``.
    """
    claim = str(record.get("claim") or "").strip()
    if not claim:
        raise ValueError("hypothesis claim is required")
    resolved_book = str(record.get("book_id") or book_id or "")
    scope = str(record.get("scope") or "setting")
    presence = record.get("missing_premises") or ()
    premises = tuple(str(item) for item in presence if str(item).strip())
    sufficiency = str(
        record.get("evidence_sufficiency")
        or ("insufficient" if premises else "sufficient")
    )
    hypothesis = Hypothesis(
        book_id=resolved_book,
        scope=scope,
        claim=claim,
        evidence_sufficiency=sufficiency,
        missing_premises=premises,
        support=_evidence_list(record.get("support")),
        against=_evidence_list(record.get("against")),
        consistency=str(record.get("consistency") or "unknown"),
        plausibility=str(record.get("plausibility") or "unknown"),
        status=str(record.get("status") or "candidate"),
        confidence=float(record.get("confidence", 0.0) or 0.0),
        method=str(record.get("method") or "abductive_rag"),
        derived_from=tuple(str(item) for item in record.get("derived_from") or ()),
        created_at=record.get("created_at") or created_at,
        hypothesis_id=record.get("hypothesis_id"),
    )
    return hypothesis.to_record()
