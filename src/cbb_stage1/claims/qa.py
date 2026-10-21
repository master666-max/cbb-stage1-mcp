"""Claim QA construction: statement -> question/answer pairs -> verified claim.

``01-箱规格-v2.md`` §5.19 (AVeriTeC + AFEV) and ``05-资料库前沿检索与吸收``
§3.1.  This module builds the ``knowledge/claims.jsonl`` object deterministically
from a ``claim_layer=statement`` assertion plus QA evidence collected by the
verifier:

* ``statement_assertion_id`` must point at a ``statement`` assertion (I22);
* every answer must fall back to ``unit_id + span + quote`` (and a quote hash);
  ``unanswerable`` is a legal stance and must **not** carry fabricated answers;
* entities mentioned by a claim must be resolved before any stance judgement
  (VeGraph order): unresolved surfaces are recorded in
  ``verified_by.unresolved_entities`` and force ``insufficient_evidence``;
* the four-valued verdict comes from :mod:`cbb_stage1.claims.verdicts`;
* ``claim_id`` is content-addressed: ``sha16(statement_assertion_id|normalized_claim)``,
  so reruns are idempotent.

No model call, no file write: the verifier (model/human/panel) supplies the
questions/answers; this module validates and shapes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1 import ids
from cbb_stage1.evidence.anchor import sha256_text
from cbb_stage1.models import Assertion, AtomicFact, Claim, ClaimQuestion

from .verdicts import (
    ClaimRuleError,
    aggregate_atomic_facts,
    verdict_from_stances,
)

__all__ = [
    "CLAIM_CONFIDENCE_METHODS",
    "CLAIM_TYPES",
    "CONTEXT_MODES",
    "EntityResolution",
    "ClaimBuildError",
    "build_claim",
    "claim_id_for",
    "make_answer",
    "make_question",
    "normalize_claim_text",
    "resolve_entities",
]

CLAIM_TYPES: tuple[str, ...] = ("attribute", "relation", "event", "setting")

#: Claim-level confidence methods (§5.15: claim scope is ``claim_qa`` / ``uq_heads``
#: / ``calibrated`` / ``human``).
CLAIM_CONFIDENCE_METHODS: frozenset[str] = frozenset({
    "claim_qa",
    "uq_heads",
    "calibrated",
    "human",
})

#: §5.19 rule 5: verifier context isolation.
CONTEXT_MODES: tuple[str, ...] = ("isolated_source", "isolated_claimed_only")

_QUESTION_STANCES = ("support", "refute", "irrelevant", "unanswerable")
_EVIDENCE_MODES = frozenset({
    "narration", "dialogue", "inner", "memory", "document", "letter",
})


class ClaimBuildError(ClaimRuleError):
    """A claim cannot be built from the given statement/questions."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _sha256_field(text: str) -> str:
    """Box-shaped quote hash (``sha256:<64 hex>``)."""
    return f"sha256:{sha256_text(text)}"


def _quote_digest(value: str) -> str:
    return value[len("sha256:"):] if value.startswith("sha256:") else value


def claim_id_for(statement_assertion_id: str, normalized_claim: str) -> str:
    """Thin wrapper over :func:`cbb_stage1.ids.claim_id` (same ``cl:`` bytes)."""
    return ids.claim_id(statement_assertion_id, normalized_claim)


def normalize_claim_text(statement: Assertion) -> str:
    """Deterministic fallback for ``normalized_claim``.

    The real normalizer should pass an explicit sentence; this fallback only
    gathers literal role values so a claim is still identifiable.
    """
    literals: list[str] = []
    for value in statement.roles.values():
        if isinstance(value, Mapping) and isinstance(value.get("literal"), str):
            literals.append(value["literal"])
        elif isinstance(value, str) and not value.startswith("e:"):
            literals.append(value)
    if literals:
        return f"{statement.predicate}：" + "，".join(literals)
    return statement.predicate


# ── answer / question builders ───────────────────────────────────


def make_answer(
    text: str,
    unit_id: str,
    start: int,
    end: int,
    quote: str,
    *,
    source: str = "narration",
    quote_sha256: str | None = None,
    unit_text: str | None = None,
) -> dict[str, Any]:
    """Build one answer map; it must fall back to ``unit_id + span + quote``."""
    if not isinstance(text, str) or not text:
        raise ClaimBuildError("bad_answer", "answer text must be a non-empty string")
    if not isinstance(unit_id, str) or not unit_id:
        raise ClaimBuildError("bad_answer", "answer unit_id is required")
    if not isinstance(quote, str) or not quote:
        raise ClaimBuildError("bad_answer", "answer quote must be non-empty")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ClaimBuildError("bad_answer", f"bad span [{start}, {end})")
    if end - start != len(quote):
        raise ClaimBuildError(
            "bad_answer",
            f"span length {end - start} != quote length {len(quote)}",
        )
    if unit_text is not None:
        if not (0 <= start < end <= len(unit_text)):
            raise ClaimBuildError("bad_answer", "answer span is outside the unit text")
        if unit_text[start:end] != quote:
            raise ClaimBuildError("bad_answer", "quote does not match unit_text[start:end]")

    expected_sha = _sha256_field(quote)
    if quote_sha256 is None:
        quote_sha256 = expected_sha
    elif _quote_digest(str(quote_sha256)) != sha256_text(quote):
        raise ClaimBuildError("bad_answer", "quote_sha256 does not match quote")

    if source not in _EVIDENCE_MODES:
        raise ClaimBuildError(
            "bad_answer", f"unknown answer source {source!r} (expected an evidence mode)"
        )
    return {
        "text": text,
        "unit_id": unit_id,
        "start": start,
        "end": end,
        "quote": quote,
        "quote_sha256": quote_sha256,
        "source": source,
    }


def make_question(
    qid: str,
    question: str,
    *,
    answers: Sequence[Mapping[str, Any]] = (),
    stance: str = "unanswerable",
) -> ClaimQuestion:
    """Build a validated question; ``unanswerable`` must not fabricate answers."""
    if not qid:
        raise ClaimBuildError("bad_question", "qid is required")
    if not question:
        raise ClaimBuildError("bad_question", "question text is required")
    if stance not in _QUESTION_STANCES:
        raise ClaimBuildError("bad_question", f"unknown stance {stance!r}")
    answer_list = [dict(answer) for answer in answers]
    if stance in ("unanswerable", "irrelevant") and answer_list:
        raise ClaimBuildError(
            "bad_question", f"stance={stance} must not carry answers"
        )
    if stance in ("support", "refute") and not answer_list:
        raise ClaimBuildError(
            "bad_question", f"stance={stance} needs at least one answer"
        )
    return ClaimQuestion(
        qid=qid,
        question=question,
        answers=answer_list,
        stance=stance,  # type: ignore[arg-type]
    )


def _validate_answer(
    answer: Mapping[str, Any] | ClaimQuestion,
    *,
    unit_texts: Mapping[str, str] | None,
) -> dict[str, Any]:
    payload = dict(answer) if isinstance(answer, Mapping) else answer.model_dump()
    required = ("text", "unit_id", "start", "end", "quote")
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ClaimBuildError("bad_answer", f"answer is missing {missing}")
    unit_id = str(payload["unit_id"])
    quote = str(payload["quote"])
    text = str(payload["text"])
    try:
        start = int(payload["start"])
        end = int(payload["end"])
    except (TypeError, ValueError) as exc:
        raise ClaimBuildError("bad_answer", f"span must be integer offsets: {exc}") from exc
    source = str(payload.get("source") or "narration")
    if unit_texts is not None:
        unit_text = unit_texts.get(unit_id)
        if unit_text is None:
            raise ClaimBuildError(
                "bad_answer", f"unit {unit_id!r} has no text to verify the answer span"
            )
        if not (0 <= start < end <= len(unit_text)) or unit_text[start:end] != quote:
            raise ClaimBuildError(
                "bad_answer",
                f"answer span [{start},{end}) does not fall back to unit {unit_id!r}",
            )
    if end - start != len(quote) or start < 0 or end <= start:
        raise ClaimBuildError("bad_answer", f"bad span [{start}, {end}) for quote")
    expected_sha = _sha256_field(quote)
    recorded = payload.get("quote_sha256")
    if recorded is not None and _quote_digest(str(recorded)) != sha256_text(quote):
        raise ClaimBuildError("bad_answer", "quote_sha256 does not match quote")
    if source not in _EVIDENCE_MODES:
        raise ClaimBuildError(
            "bad_answer", f"unknown answer source {source!r} (expected an evidence mode)"
        )
    return {
        "text": text,
        "unit_id": unit_id,
        "start": start,
        "end": end,
        "quote": quote,
        "quote_sha256": expected_sha,
        "source": source,
    }


def _validate_question(
    question: ClaimQuestion | Mapping[str, Any],
    *,
    unit_texts: Mapping[str, str] | None,
) -> ClaimQuestion:
    if isinstance(question, ClaimQuestion):
        qid, text, stance = question.qid, question.question, question.stance
        raw_answers = list(question.answers)
    elif isinstance(question, Mapping):
        qid = str(question.get("qid") or "")
        text = str(question.get("question") or "")
        stance = str(question.get("stance") or "unanswerable")
        raw_answers = list(question.get("answers") or ())
    else:
        raise ClaimBuildError(
            "bad_question",
            f"question must be a ClaimQuestion or mapping, got {type(question).__name__}",
        )
    if not qid:
        raise ClaimBuildError("bad_question", "qid is required")
    if not text:
        raise ClaimBuildError("bad_question", "question text is required")
    if stance not in _QUESTION_STANCES:
        raise ClaimBuildError("bad_question", f"unknown stance {stance!r}")
    answers = [_validate_answer(answer, unit_texts=unit_texts) for answer in raw_answers]
    if stance in ("unanswerable", "irrelevant") and answers:
        raise ClaimBuildError(
            "bad_question", f"stance={stance} must not carry answers"
        )
    if stance in ("support", "refute") and not answers:
        raise ClaimBuildError(
            "bad_question", f"stance={stance} needs at least one answer"
        )
    return ClaimQuestion(
        qid=qid,
        question=text,
        answers=answers,
        stance=stance,  # type: ignore[arg-type]
    )


# ── entity resolution before stance (VeGraph order) ──────────────


@dataclass(frozen=True)
class EntityResolution:
    """Surface -> entity id map plus the surfaces that did not resolve."""

    resolved: dict[str, str]
    unresolved: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.unresolved


def resolve_entities(
    surfaces: Iterable[str],
    *,
    name_to_id: Mapping[str, str],
    alias_map: Mapping[str, str] | None = None,
) -> EntityResolution:
    """Resolve claim surfaces through the alias map **before** any stance.

    ``name_to_id`` maps canonical entity names to entity ids; ``alias_map`` is
    the ``alias -> canonical_name`` map from ``cbb_stage1.resolve.alias``.
    """
    alias_map = alias_map or {}
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for surface in surfaces:
        if not surface:
            continue
        canonical = alias_map.get(surface, surface)
        entity_id = name_to_id.get(canonical) or name_to_id.get(surface)
        if entity_id:
            resolved[surface] = entity_id
        elif surface not in unresolved:
            unresolved.append(surface)
    return EntityResolution(resolved=resolved, unresolved=tuple(unresolved))


# ── claim builder ────────────────────────────────────────────────


def build_claim(
    statement: Assertion,
    *,
    questions: Sequence[ClaimQuestion | Mapping[str, Any]],
    normalized_claim: str | None = None,
    claim_type: str = "attribute",
    atomic_facts: Sequence[AtomicFact | Mapping[str, Any]] = (),
    book_id: str | None = None,
    confidence: float = 0.0,
    confidence_method: str = "claim_qa",
    calibration_version: str | None = None,
    verifier: str = "claim-verifier",
    model: str | None = None,
    context_mode: str = "isolated_source",
    unit_texts: Mapping[str, str] | None = None,
    unresolved_entities: Iterable[str] = (),
) -> Claim:
    """Validate QA output and build a ``knowledge/claims.jsonl`` record."""
    if statement.claim_layer != "statement":
        raise ClaimBuildError(
            "claim_layer_conflict",
            "statement_assertion_id must reference a claim_layer=statement assertion",
        )
    if not statement.assertion_id:
        raise ClaimBuildError("missing_assertion_id", "statement has no assertion_id")
    if claim_type not in CLAIM_TYPES:
        raise ClaimBuildError("bad_claim_type", f"claim_type {claim_type!r} not in {CLAIM_TYPES}")
    if context_mode not in CONTEXT_MODES:
        raise ClaimBuildError(
            "bad_context_mode", f"context_mode {context_mode!r} not in {CONTEXT_MODES}"
        )
    if confidence_method not in CLAIM_CONFIDENCE_METHODS:
        raise ClaimBuildError(
            "bad_confidence_method",
            f"claim confidence_method {confidence_method!r} not in "
            f"{sorted(CLAIM_CONFIDENCE_METHODS)}",
        )
    if confidence_method == "calibrated" and not calibration_version:
        raise ClaimBuildError(
            "missing_calibration_version",
            "confidence_method=calibrated requires calibration_version",
        )
    if not (0.0 <= float(confidence) <= 1.0):
        raise ClaimBuildError("bad_confidence", "confidence must be within [0, 1]")

    normalized = (normalized_claim or normalize_claim_text(statement)).strip()
    if not normalized:
        raise ClaimBuildError("bad_normalized_claim", "normalized_claim is empty")

    validated_questions: list[ClaimQuestion] = []
    seen_qids: set[str] = set()
    for question in questions:
        validated = _validate_question(question, unit_texts=unit_texts)
        if validated.qid in seen_qids:
            raise ClaimBuildError("duplicate_qid", f"qid {validated.qid!r} is repeated")
        seen_qids.add(validated.qid)
        validated_questions.append(validated)
    if not validated_questions:
        raise ClaimBuildError("no_questions", "a claim needs at least one question")

    unresolved = [str(item) for item in unresolved_entities if item]
    if unresolved and any(
        question.stance in ("support", "refute") for question in validated_questions
    ):
        raise ClaimBuildError(
            "entity_resolution_required",
            "unresolved entities must be resolved before support/refute stances "
            f"(unresolved={unresolved})",
        )

    atomic = [
        item if isinstance(item, AtomicFact) else AtomicFact.model_validate(dict(item))
        for item in atomic_facts
    ]

    if unresolved:
        verdict = "insufficient_evidence"
    else:
        verdict = verdict_from_stances(question.stance for question in validated_questions)
        if atomic:
            aggregated = aggregate_atomic_facts(atomic)
            if aggregated != verdict:
                raise ClaimBuildError(
                    "atomic_facts_inconsistent",
                    f"aggregate {aggregated!r} != stance verdict {verdict!r}",
                )

    verified_by: dict[str, Any] = {
        "verifier": verifier,
        "model": model,
        "context_mode": context_mode,
    }
    if unresolved:
        verified_by["unresolved_entities"] = unresolved
    if calibration_version:
        verified_by["calibration_version"] = calibration_version

    return Claim(
        claim_id=claim_id_for(statement.assertion_id, normalized),
        book_id=book_id or statement.book_id,
        statement_assertion_id=statement.assertion_id,
        normalized_claim=normalized,
        claim_type=claim_type,
        questions=validated_questions,
        atomic_facts=atomic,
        verdict=verdict,
        verdict_rule="av-eritec",
        justification=None,
        confidence=float(confidence),
        confidence_method=confidence_method,
        confidence_scope="claim",
        verified_by=verified_by,
        status="verified" if verdict != "insufficient_evidence" else "open",
    )
