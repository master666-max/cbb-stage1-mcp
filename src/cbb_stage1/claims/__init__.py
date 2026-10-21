"""Claim verification: AVeriTeC verdicts, atomic facts, fact-promotion gate.

* :mod:`cbb_stage1.claims.verdicts` -- deterministic four-valued verdict
  (``supported`` / ``refuted`` / ``conflicting`` / ``insufficient_evidence``),
  ``atomic_facts`` aggregation, and the promotion gate: only a supported
  statement with >= 2 independent sources may become a new ``fact`` assertion;
  ``insufficient_evidence`` never promotes.
* :mod:`cbb_stage1.claims.qa` -- build ``knowledge/claims.jsonl`` records:
  question/answer validation (span + quote + hash fallback), unanswerable must
  not fabricate answers, entity resolution before stance (VeGraph order),
  content-addressed ``claim_id``.

Boundary: ``claims`` reads assembled assertions and produces claims; it does not
write the box (``box/``) and does not call a model -- the verifier supplies the
QA pairs, this package enforces the rules.
"""

from __future__ import annotations

from .qa import (
    CLAIM_CONFIDENCE_METHODS,
    CLAIM_TYPES,
    CONTEXT_MODES,
    ClaimBuildError,
    EntityResolution,
    build_claim,
    claim_id_for,
    make_answer,
    make_question,
    normalize_claim_text,
    resolve_entities,
)
from .verdicts import (
    CLAIM_VERDICTS,
    MIN_INDEPENDENT_SOURCES,
    PROMOTION_RULE,
    ClaimRuleError,
    PromotionDecision,
    PromotionRefused,
    aggregate_atomic_facts,
    build_promoted_fact,
    claim_verdict,
    independent_sources,
    promotion_decision,
    support_answers,
    verdict_from_stances,
)
from .integration import run

__all__ = [
    # verdicts
    "CLAIM_VERDICTS",
    "MIN_INDEPENDENT_SOURCES",
    "PROMOTION_RULE",
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
    # qa
    "CLAIM_CONFIDENCE_METHODS",
    "CLAIM_TYPES",
    "CONTEXT_MODES",
    "ClaimBuildError",
    "EntityResolution",
    "build_claim",
    "claim_id_for",
    "make_answer",
    "make_question",
    "normalize_claim_text",
    "resolve_entities",
    # integration
    "run",
]
