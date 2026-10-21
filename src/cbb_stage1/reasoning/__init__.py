"""Reasoning adapters for box v2.2: AFEV / Abductive RAG / VeGraph (B7/B8/B9).

Design refs: ``设计/抽取段-v6/05-资料库前沿检索与吸收-20261018.md`` §3.4–3.5,
``01-箱规格-v2.md`` §5.19–§5.20.

Boundary (same as :mod:`cbb_stage1.claims`): **no model call and no file write in
this package.** The model is an injected ``Protocol``; when it is absent the
result is *unverifiable* / *insufficient_evidence* and nothing is fabricated.
The caller (box writer / orchestrator) decides whether to write the records.

* :mod:`cbb_stage1.reasoning.atomic` -- AFEV: iteratively decompose a complex
  claim into atomic facts, verify each, aggregate (false > unverifiable > true).
* :mod:`cbb_stage1.reasoning.abduction` -- Abductive RAG: check evidence
  sufficiency first; only ``insufficient`` may propose missing premises; output
  is a **derived-layer** hypothesis (never a fact card).
* :mod:`cbb_stage1.reasoning.vegraph` -- VeGraph: claim -> triples -> entity
  resolution (reuses :func:`cbb_stage1.claims.qa.resolve_entities`) -> per-triple
  sub-assertion verification; unresolved surfaces block stance judgement.
"""

from __future__ import annotations

from .abduction import (
    DERIVED_TARGET,
    SCOPES,
    SUFFICIENCY,
    AbductionLLM,
    AbductionResult,
    AbductionRuleError,
    Hypothesis,
    abduce,
    evidence_sufficiency,
)
from .atomic import (
    ATOMIC_VERDICTS,
    AFEVResult,
    AtomicLLM,
    AtomicRuleError,
    afev_verify,
)
from .vegraph import (
    STANCES,
    Triple,
    VeGraphLLM,
    VeGraphResult,
    VeGraphRuleError,
    vegraph_verify,
)

__all__ = [
    "ATOMIC_VERDICTS",
    "DERIVED_TARGET",
    "SCOPES",
    "STANCES",
    "SUFFICIENCY",
    "AFEVResult",
    "AbductionLLM",
    "AbductionResult",
    "AbductionRuleError",
    "AtomicLLM",
    "AtomicRuleError",
    "Hypothesis",
    "Triple",
    "VeGraphLLM",
    "VeGraphResult",
    "VeGraphRuleError",
    "abduce",
    "afev_verify",
    "evidence_sufficiency",
    "vegraph_verify",
]
