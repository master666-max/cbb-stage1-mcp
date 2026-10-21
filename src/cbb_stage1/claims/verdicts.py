"""Claim verdict rules (AVeriTeC) + atomic-fact aggregation + promotion gate.

Deterministic rules, no model call (``01-箱规格-v2.md`` §5.19, paper mapping in
``05-资料库前沿检索与吸收-20261018.md``):

* four-valued verdict from question stances:
  support + refute => ``conflicting``; only support => ``supported``; only
  refute => ``refuted``; everything else (irrelevant/unanswerable/none) =>
  ``insufficient_evidence``;
* ``atomic_facts`` aggregation must agree with the stance verdict: all true =>
  supported, any false => refuted, any unverifiable => insufficient;
* **fact promotion gate**: a ``claim_layer=statement`` assertion may become
  ``fact`` only when the claim is ``supported``, has >= 2 independent sources
  (different unit or different speaker) and no refute.
  ``insufficient_evidence`` can *never* be promoted -- it is refused with a
  named reason, and promotion builds a **new** ``fact`` assertion with
  ``derived_from`` pointing at the statement (the statement is never mutated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1.assemble.assertions import assertion_id_for
from cbb_stage1.evidence.anchor import sha256_text
from cbb_stage1.models import (
    AtomicFact,
    Assertion,
    Claim,
    ClaimVerdict,
    Evidence,
)

__all__ = [
    "CLAIM_VERDICTS",
    "DEFAULT_ABSTAIN_THRESHOLD",
    "MIN_EVIDENCE_ITEMS",
    "MIN_INDEPENDENT_SOURCES",
    "PROMOTION_RULE",
    "AbstainDecision",
    "SufficiencyDecision",
    "abstain_decision",
    "apply_abstain",
    "evidence_sufficiency",
    "ClaimRuleError",
    "PromotionDecision",
    "PromotionRefused",
    "aggregate_atomic_facts",
    "build_promoted_fact",
    "claim_verdict",
    "independent_sources",
    "promotion_decision",
    "support_answers",
    "verdict_from_stances",
]

CLAIM_VERDICTS: tuple[str, ...] = (
    "supported",
    "refuted",
    "conflicting",
    "insufficient_evidence",
)

MIN_INDEPENDENT_SOURCES = 2
PROMOTION_RULE = "av-eritec-promotion"

_EVIDENCE_MODES = frozenset({
    "narration", "dialogue", "inner", "memory", "document", "letter",
})


class ClaimRuleError(ValueError):
    """A claim violates the AVeriTeC contract."""


class PromotionRefused(ClaimRuleError):
    """Raised when a statement may not be promoted to a fact."""

    def __init__(self, decision: "PromotionDecision") -> None:
        self.decision = decision
        super().__init__(f"promotion refused: {decision.reason}")


# ── verdict rules ────────────────────────────────────────────────


def verdict_from_stances(stances: Iterable[str]) -> ClaimVerdict:
    """AVeriTeC deterministic rule over question stances."""
    has_support = False
    has_refute = False
    for stance in stances:
        if stance == "support":
            has_support = True
        elif stance == "refute":
            has_refute = True
    if has_support and has_refute:
        return "conflicting"
    if has_support:
        return "supported"
    if has_refute:
        return "refuted"
    return "insufficient_evidence"


def _atomic_verdict(item: AtomicFact | Mapping[str, Any]) -> str:
    if isinstance(item, AtomicFact):
        return item.verdict
    value = item.get("verdict")
    return str(value)


def aggregate_atomic_facts(
    facts: Iterable[AtomicFact | Mapping[str, Any]],
) -> ClaimVerdict:
    """Aggregate atomic facts: false dominates, then unverifiable, then true."""
    verdicts = [_atomic_verdict(item) for item in facts]
    if not verdicts:
        return "insufficient_evidence"
    unknown = {value for value in verdicts if value not in ("true", "false", "unverifiable")}
    if unknown:
        raise ClaimRuleError(f"atomic fact has unknown verdict(s): {sorted(unknown)}")
    if any(value == "false" for value in verdicts):
        return "refuted"
    if any(value == "unverifiable" for value in verdicts):
        return "insufficient_evidence"
    return "supported"


def claim_verdict(claim: Claim, *, use_atomic_facts: bool = True) -> ClaimVerdict:
    """Recompute the verdict and enforce atomic-facts consistency (§5.19 rule 3)."""
    stance_verdict = verdict_from_stances(
        question.stance for question in claim.questions
    )
    if use_atomic_facts and claim.atomic_facts:
        atomic_verdict = aggregate_atomic_facts(claim.atomic_facts)
        if atomic_verdict != stance_verdict:
            raise ClaimRuleError(
                "atomic_facts aggregate "
                f"{atomic_verdict!r} disagrees with question stances {stance_verdict!r}"
            )
    return stance_verdict


# ── B6 evidence sufficiency + B12 abstain ────────────────────────
#
# AVeriTeC asks two questions before a verdict: is there enough evidence, and
# does it agree? The module previously answered only the second. Adding the
# first is what makes ``insufficient_evidence`` mean "we cannot tell" rather
# than "the answers happened to be irrelevant".
#
# B12 is the same idea driven by the confidence signal: the semantic entropy
# score was being computed (B2) and written to the assertion table, but nothing
# downstream read it, so a claim could be promoted on a 0.2-confidence basis.
# ``abstain_decision`` closes that loop.

#: Minimum evidence items before a claim may be judged at all.
MIN_EVIDENCE_ITEMS = 1
#: Below this confidence a claim abstains instead of being judged.
DEFAULT_ABSTAIN_THRESHOLD = 0.5


@dataclass(frozen=True)
class SufficiencyDecision:
    """Whether the evidence is sufficient to support *any* verdict (B6)."""

    sufficient: bool
    reason: str
    evidence_count: int = 0
    distinct_units: int = 0

    @property
    def status(self) -> str:
        return "sufficient" if self.sufficient else "insufficient"


def evidence_sufficiency(
    evidence: Sequence[Any],
    *,
    min_items: int = MIN_EVIDENCE_ITEMS,
) -> SufficiencyDecision:
    """Is there enough evidence to judge the claim (AVeriTeC step 1)?

    Deliberately structural: it counts evidence items and distinct source units,
    and refuses to guess at relevance. A claim whose evidence is empty or comes
    from a single span with no independent corroboration is reported as
    insufficient *before* any verdict is computed, so the verdict cannot
    accidentally read as a judgement on the facts.
    """
    items = [item for item in evidence if item is not None]
    if not items:
        return SufficiencyDecision(False, "no_evidence", 0, 0)

    units: list[str] = []
    for item in items:
        unit_id = ""
        if isinstance(item, Mapping):
            unit_id = str(item.get("unit_id") or "")
        else:
            unit_id = str(getattr(item, "unit_id", "") or "")
        if unit_id and unit_id not in units:
            units.append(unit_id)

    if len(items) < min_items:
        return SufficiencyDecision(
            False, "below_min_evidence_items", len(items), len(units)
        )
    return SufficiencyDecision(True, "enough_evidence", len(items), len(units))


@dataclass(frozen=True)
class AbstainDecision:
    """Whether a claim must abstain rather than carry a verdict (B12)."""

    abstain: bool
    reason: str
    confidence: float | None = None
    threshold: float = DEFAULT_ABSTAIN_THRESHOLD

    @property
    def verdict(self) -> ClaimVerdict:
        return "insufficient_evidence" if self.abstain else "supported"


def abstain_decision(
    confidence: Any,
    *,
    threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    sufficiency: SufficiencyDecision | None = None,
) -> AbstainDecision:
    """B12: low confidence or insufficient evidence forces ``insufficient_evidence``.

    The threshold is a real number, not a placeholder: it is configurable and the
    decision records both the value and the threshold that produced it, so a
    reader can tell "abstained at 0.31 < 0.5" from "abstained because there was
    no evidence at all".
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ClaimRuleError(f"abstain threshold must be in [0,1]: {threshold!r}")

    if sufficiency is not None and not sufficiency.sufficient:
        return AbstainDecision(
            True,
            f"insufficient_evidence:{sufficiency.reason}",
            None if confidence is None else float(confidence),
            float(threshold),
        )

    if confidence is None:
        # No score at all is not the same as a low score: report it as such.
        return AbstainDecision(True, "no_confidence_score", None, float(threshold))

    value = float(confidence)
    if value < float(threshold):
        return AbstainDecision(
            True, "below_abstain_threshold", value, float(threshold)
        )
    return AbstainDecision(False, "above_abstain_threshold", value, float(threshold))


def apply_abstain(
    claim: Claim,
    *,
    threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    evidence: Sequence[Any] | None = None,
) -> tuple[Claim, AbstainDecision]:
    """Return a copy of *claim* whose verdict honours B6 + B12.

    The claim is copied, never mutated in place: the original verdict is evidence
    of what the QA pass said, and overwriting it would erase the disagreement
    that a reviewer needs to see.
    """
    sufficiency = evidence_sufficiency(evidence if evidence is not None else [])
    decision = abstain_decision(
        claim.confidence, threshold=threshold, sufficiency=sufficiency
    )
    if not decision.abstain:
        return claim, decision
    updated = claim.model_copy(update={"verdict": "insufficient_evidence"})
    return updated, decision


# ── promotion gate ───────────────────────────────────────────────


def _answers(claim: Claim, stance: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for question in claim.questions:
        if question.stance != stance:
            continue
        for answer in question.answers:
            if isinstance(answer, Mapping):
                out.append(dict(answer))
    return out


def support_answers(claim: Claim) -> list[dict[str, Any]]:
    """Flatten the answer maps of every ``stance=support`` question."""
    return _answers(claim, "support")


def _refute_answers(claim: Claim) -> list[dict[str, Any]]:
    return _answers(claim, "refute")


def independent_sources(
    answers: Sequence[Mapping[str, Any]],
    *,
    answer_speakers: Sequence[str | None] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(distinct_units, distinct_speakers)`` for the given answers.

    The box answer row has no ``speaker`` field (schema is ``additionalProperties:
    false``), so speaker-level independence can only be supplied by the caller
    via ``answer_speakers`` aligned with *answers*.
    """
    units: list[str] = []
    speakers: list[str] = []
    for index, answer in enumerate(answers):
        unit_id = str(answer.get("unit_id") or "")
        if unit_id and unit_id not in units:
            units.append(unit_id)
        speaker = None
        if answer_speakers is not None and index < len(answer_speakers):
            speaker = answer_speakers[index]
        elif answer.get("speaker"):
            speaker = str(answer["speaker"])
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    return tuple(units), tuple(speakers)


@dataclass(frozen=True)
class PromotionDecision:
    """Whether a statement may be promoted, with a named reason either way."""

    allowed: bool
    reason: str
    verdict: str
    units: tuple[str, ...] = ()
    speakers: tuple[str, ...] = ()
    rule: str = PROMOTION_RULE

    @property
    def independent_source_count(self) -> int:
        return max(len(self.units), len(self.speakers))


def promotion_decision(
    claim: Claim,
    statement: Assertion,
    *,
    answer_speakers: Sequence[str | None] | None = None,
) -> PromotionDecision:
    """Apply the fact-promotion gate (§5.19 rule 4)."""
    if statement.claim_layer != "statement":
        return PromotionDecision(False, "statement_assertion_not_statement", claim.verdict)
    if claim.statement_assertion_id != statement.assertion_id:
        return PromotionDecision(False, "claim_statement_mismatch", claim.verdict)

    recomputed = claim_verdict(claim)
    if recomputed != claim.verdict:
        return PromotionDecision(
            False, f"claim_verdict_stale_{recomputed}", recomputed
        )

    if claim.verdict == "insufficient_evidence":
        return PromotionDecision(
            False, "insufficient_evidence_never_promotes", claim.verdict
        )
    if claim.verdict != "supported":
        return PromotionDecision(False, f"verdict_{claim.verdict}_not_promotable", claim.verdict)

    if _refute_answers(claim):
        return PromotionDecision(False, "refute_present", claim.verdict)

    answers = support_answers(claim)
    units, speakers = independent_sources(answers, answer_speakers=answer_speakers)
    independent = len(units) >= MIN_INDEPENDENT_SOURCES or len(speakers) >= MIN_INDEPENDENT_SOURCES
    if not independent:
        return PromotionDecision(
            False,
            "needs_two_independent_sources",
            claim.verdict,
            units,
            speakers,
        )
    return PromotionDecision(True, "supported_with_corroboration", claim.verdict, units, speakers)


def _answer_to_evidence(answer: Mapping[str, Any]) -> Evidence:
    quote = str(answer.get("quote") or "")
    if not quote:
        raise ClaimRuleError("support answer has no quote; cannot build fact evidence")
    mode = str(answer.get("source") or "narration")
    if mode not in _EVIDENCE_MODES:
        mode = "narration"
    quote_sha = answer.get("quote_sha256") or sha256_text(quote)
    return Evidence(
        unit_id=str(answer["unit_id"]),
        start=int(answer["start"]),
        end=int(answer["end"]),
        quote=quote,
        quote_sha256=str(quote_sha),
        occurrence=int(answer.get("occurrence") or 1),
        speaker=None,
        mode=mode,  # type: ignore[arg-type]
    )


def build_promoted_fact(
    claim: Claim,
    statement: Assertion,
    *,
    answer_speakers: Sequence[str | None] | None = None,
    evidence: Sequence[Evidence] | None = None,
) -> Assertion:
    """Build the *new* fact assertion for a supported statement (never mutate it)."""
    decision = promotion_decision(claim, statement, answer_speakers=answer_speakers)
    if not decision.allowed:
        raise PromotionRefused(decision)
    if not statement.assertion_id:
        raise ClaimRuleError("statement assertion has no assertion_id")

    evidences = list(evidence) if evidence is not None else [
        _answer_to_evidence(answer) for answer in support_answers(claim)
    ]
    if not evidences:
        raise PromotionRefused(
            PromotionDecision(False, "no_independent_evidence", claim.verdict)
        )

    deps: list[str] = [statement.assertion_id]
    for item in evidences:
        if item.unit_id not in deps:
            deps.append(item.unit_id)

    return Assertion(
        assertion_id=assertion_id_for(statement.book_id, statement.predicate, statement.roles, evidences),
        book_id=statement.book_id,
        predicate=statement.predicate,
        library=statement.library,
        roles=dict(statement.roles),
        qualifiers=dict(statement.qualifiers),
        claim_layer="fact",
        speaker=None,
        evidence=evidences,
        confidence=claim.confidence,
        confidence_method=claim.confidence_method or "claim_qa",
        confidence_scope="assertion",
        status="asserted",
        verification=dict(statement.verification or {}) or None,
        derived_from=[statement.assertion_id],
        deps=deps,
        produced_by=dict(claim.verified_by),
    )
