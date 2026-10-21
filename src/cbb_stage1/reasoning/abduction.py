"""Abductive RAG: evidence sufficiency -> missing premises -> consistency/plausibility.

Design refs: ``05-资料库前沿检索与吸收-20261018.md`` §3.5 (B8) and
``01-箱规格-v2.md`` §5.20 (``derived/hypotheses.jsonl``).

Rules enforced here (they are the design's, not new ones):

* sufficiency is checked **first**; ``sufficient`` => normal induction, **no
  hypothesis is fabricated**;
* only ``insufficient`` may propose ``missing_premises``;
* ``consistency``/``plausibility`` are the dual check of Abductive RAG; failing
  either keeps the hypothesis ``candidate`` (it never reaches ``verified``);
* the output targets ``derived/hypotheses.jsonl`` -- **derived layer only, never
  a fact card** (this module never writes it; it only shapes the record).

Boundary: no LLM call. Without an interpreter the only honest hypothesis is a
``candidate`` with no premises and an explicit "需补证据" claim. No file write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from cbb_stage1 import ids
from cbb_stage1.evidence.anchor import sha256_text
from cbb_stage1.models import Evidence

__all__ = [
    "DERIVED_TARGET",
    "LABELLED_CONSISTENCY",
    "LABELLED_PLAUSIBILITY",
    "MIN_SOURCES_DEFAULT",
    "SCOPES",
    "SUFFICIENCY",
    "VERIFIED_MIN_DEFAULT",
    "AbductionLLM",
    "AbductionResult",
    "AbductionRuleError",
    "Hypothesis",
    "abduce",
    "evidence_sufficiency",
]

#: Where the caller must write these records (derived layer; never ``core``).
DERIVED_TARGET = "derived/hypotheses.jsonl"

SCOPES: tuple[str, ...] = ("setting", "foreshadow", "motivation", "prediction")
SUFFICIENCY: tuple[str, ...] = ("sufficient", "insufficient")
HYPOTHESIS_STATUS: tuple[str, ...] = ("candidate", "verified", "rejected")

MIN_SOURCES_DEFAULT = 2
VERIFIED_MIN_DEFAULT = 0.6

LABELLED_CONSISTENCY: tuple[str, ...] = ("pass", "fail")
LABELLED_PLAUSIBILITY: tuple[str, ...] = ("high", "medium", "low")


class AbductionRuleError(ValueError):
    """The abductive interpreter returned something outside the contract."""


class AbductionLLM(Protocol):
    """The two model capabilities Abductive RAG needs."""

    def missing_premises(self, claim: str, evidence: Sequence[str]) -> Sequence[str]:
        """Propose the premises the text is missing for this claim."""

    def score(self, claim: str, premises: Sequence[str]) -> Mapping[str, Any]:
        """Return ``{"consistency": 0..1, "plausibility": 0..1}`` for the hypothesis."""


def hypothesis_id_for(book_id: str, scope: str, claim: str, premises: Sequence[str]) -> str:
    """Content-addressed hypothesis id: same inputs => same id (rerun idempotent).

    Compatibility wrapper over :func:`cbb_stage1.ids.abduction_hypothesis_id`;
    the public name/signature and the byte formula are unchanged.
    """
    return ids.abduction_hypothesis_id(book_id, scope, claim, premises)


def _unit_ids(evidence: Sequence[Evidence]) -> tuple[str, ...]:
    units: list[str] = []
    for item in evidence:
        if item.unit_id not in units:
            units.append(item.unit_id)
    return tuple(units)


def evidence_sufficiency(
    evidence: Sequence[Evidence],
    *,
    min_sources: int = MIN_SOURCES_DEFAULT,
) -> str:
    """Deterministic first check: distinct source units ``>= min_sources``.

    The design asks to check sufficiency *before* abduction; this is the
    mechanical form of that check (no model involved). A caller that has a
    stronger verifier may pass its own sufficiency judgement to :func:`abduce`
    via ``llm``, but the default never invents evidence.
    """
    if min_sources < 1:
        raise AbductionRuleError("min_sources must be >= 1")
    return "sufficient" if len(_unit_ids(evidence)) >= min_sources else "insufficient"


def _quote_row(item: Evidence) -> dict[str, Any]:
    return {
        "unit_id": item.unit_id,
        "quote": item.quote,
        "quote_sha256": "sha256:" + sha256_text(item.quote),
    }


def _evidence_quotes(evidence: Sequence[Evidence]) -> list[str]:
    return [item.quote for item in evidence]


def _clean_premises(values: Sequence[Any]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise AbductionRuleError(
                f"missing_premises() must return non-empty strings, got {value!r}"
            )
        text = value.strip()
        if text not in out:
            out.append(text)
    return tuple(out)


def _score(callable_llm: AbductionLLM, claim: str, premises: Sequence[str]) -> tuple[float, float]:
    raw = callable_llm.score(claim, list(premises))
    if not isinstance(raw, Mapping):
        raise AbductionRuleError(f"score() must return a mapping, got {type(raw).__name__}")
    values: list[float] = []
    for key in ("consistency", "plausibility"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AbductionRuleError(f"score()[{key!r}] must be a number in [0, 1], got {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise AbductionRuleError(f"score()[{key!r}]={value!r} is outside [0, 1]")
        values.append(float(value))
    return values[0], values[1]


def _consistency_label(value: float, verified_min: float) -> str:
    return "pass" if value >= verified_min else "fail"


def _plausibility_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


@dataclass(frozen=True)
class Hypothesis:
    """One ``derived/hypotheses.jsonl`` record (box §5.20 shape)."""

    book_id: str
    scope: str
    claim: str
    missing_premises: tuple[str, ...]
    support: tuple[Evidence, ...] = ()
    against: tuple[Evidence, ...] = ()
    consistency: str | None = None
    plausibility: str | None = None
    status: str = "candidate"
    confidence: float = 0.0
    method: str = "abductive_rag"
    derived_from: tuple[str, ...] = ()
    created_at: str | None = None
    hypothesis_id: str = ""
    target_file: str = DERIVED_TARGET

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise AbductionRuleError(f"scope {self.scope!r} not in {SCOPES}")
        if self.status not in HYPOTHESIS_STATUS:
            raise AbductionRuleError(f"status {self.status!r} not in {HYPOTHESIS_STATUS}")
        if self.consistency is not None and self.consistency not in LABELLED_CONSISTENCY:
            raise AbductionRuleError(f"consistency {self.consistency!r} not in {LABELLED_CONSISTENCY}")
        if self.plausibility is not None and self.plausibility not in LABELLED_PLAUSIBILITY:
            raise AbductionRuleError(f"plausibility {self.plausibility!r} not in {LABELLED_PLAUSIBILITY}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise AbductionRuleError(f"confidence {self.confidence!r} outside [0, 1]")
        if not self.hypothesis_id:
            object.__setattr__(
                self,
                "hypothesis_id",
                hypothesis_id_for(self.book_id, self.scope, self.claim, self.missing_premises),
            )

    def to_record(self) -> dict[str, Any]:
        """The box §5.20 row. ``created_at`` stays ``None`` unless the caller sets it."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "book_id": self.book_id,
            "scope": self.scope,
            "claim": self.claim,
            "evidence_sufficiency": "insufficient",
            "missing_premises": list(self.missing_premises),
            "support": [_quote_row(item) for item in self.support],
            "against": [_quote_row(item) for item in self.against],
            "consistency": self.consistency,
            "plausibility": self.plausibility,
            "status": self.status,
            "confidence": self.confidence,
            "method": self.method,
            "derived_from": list(self.derived_from),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AbductionResult:
    """Outcome of :func:`abduce`; ``hypothesis=None`` means "use induction instead"."""

    evidence_sufficiency: str
    hypothesis: Hypothesis | None
    reason: str
    target_file: str = DERIVED_TARGET

    @property
    def derived(self) -> bool:
        """These records only ever belong to the derived layer."""
        return True


def abduce(
    claim: str,
    *,
    book_id: str = "",
    scope: str = "setting",
    evidence: Sequence[Evidence] = (),
    llm: AbductionLLM | None = None,
    min_sources: int = MIN_SOURCES_DEFAULT,
    verified_min: float = VERIFIED_MIN_DEFAULT,
) -> AbductionResult:
    """Check sufficiency first; only then (and only if insufficient) hypothesise.

    Without ``llm`` the hypothesis is a *candidate* with no premises and an
    explicit "需补证据" claim -- nothing is invented.
    """
    if not isinstance(claim, str) or not claim.strip():
        raise AbductionRuleError("claim must be a non-empty string")
    if scope not in SCOPES:
        raise AbductionRuleError(f"scope {scope!r} not in {SCOPES}")
    if not 0.0 <= verified_min <= 1.0:
        raise AbductionRuleError("verified_min must be inside [0, 1]")
    claim = claim.strip()
    evidence = tuple(evidence)

    sufficiency = evidence_sufficiency(evidence, min_sources=min_sources)
    if sufficiency == "sufficient":
        return AbductionResult(
            evidence_sufficiency="sufficient",
            hypothesis=None,
            reason="sufficient_evidence_use_induction",
        )

    units = _unit_ids(evidence)
    if llm is None:
        hypothesis = Hypothesis(
            book_id=book_id,
            scope=scope,
            claim=f"{claim}（需补证据）",
            missing_premises=(),
            support=evidence,
            against=(),
            consistency=None,
            plausibility=None,
            status="candidate",
            confidence=0.0,
            derived_from=units,
        )
        return AbductionResult(
            evidence_sufficiency="insufficient",
            hypothesis=hypothesis,
            reason="no_llm_only_candidate_no_fabrication",
        )

    premises = _clean_premises(list(llm.missing_premises(claim, _evidence_quotes(evidence))))
    consistency, plausibility = _score(llm, claim, premises)
    status = "verified" if min(consistency, plausibility) >= verified_min else "candidate"
    hypothesis = Hypothesis(
        book_id=book_id,
        scope=scope,
        claim=claim,
        missing_premises=premises,
        support=evidence,
        against=(),
        consistency=_consistency_label(consistency, verified_min),
        plausibility=_plausibility_label(plausibility),
        status=status,
        confidence=round(min(consistency, plausibility), 3),
        derived_from=units,
    )
    return AbductionResult(
        evidence_sufficiency="insufficient",
        hypothesis=hypothesis,
        reason="insufficient_evidence_abduced",
    )
