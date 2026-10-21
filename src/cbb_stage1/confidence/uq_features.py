"""Statistical features for the B5 UQ head (train / inference share this file).

The UQ head is a tiny claim-level uncertainty probe.  Feature extraction lives in
``cbb_stage1.confidence`` (not in ``tools/``) so the training tool and the runtime
adapter compute **exactly the same vector** — a train/infer feature skew would
silently make the head useless.

Feature schema ``1`` (12 numeric features, all finite; missing fields become 0):

============================  ==================================================
``claim_chars``               claim length in characters
``answer_chars``              predicted answer length
``evidence_count``            number of evidence quote items
``evidence_chars``            total characters across evidence quotes
``claim_bigram_recall``       share of claim char-bigrams covered by evidence
``answer_bigram_recall``      share of answer char-bigrams covered by evidence
``claim_answer_jaccard``      char-bigram Jaccard between claim and answer
``claim_has_number``          1.0 when the claim contains a digit
``claim_hedge_count``         count of hedging words (可能/似乎/…)
``claim_quote_marked``        1.0 when the claim carries quotation marks
``evidence_empty``            1.0 when no evidence was provided
``claim_is_question``         1.0 when the claim ends with ? / ？

The head itself is trained by ``tools/train_uq_head.py`` and loaded by
:mod:`cbb_stage1.confidence.uq_checkpoint`.  Nothing here fabricates a score.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from . import as_mapping
from .semantic_entropy import _char_bigrams

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "answer_text",
    "claim_text",
    "evidence_quotes",
    "extract_features",
    "feature_dict",
]

FEATURE_SCHEMA = 1

FEATURE_NAMES: tuple[str, ...] = (
    "claim_chars",
    "answer_chars",
    "evidence_count",
    "evidence_chars",
    "claim_bigram_recall",
    "answer_bigram_recall",
    "claim_answer_jaccard",
    "claim_has_number",
    "claim_hedge_count",
    "claim_quote_marked",
    "evidence_empty",
    "claim_is_question",
)

_CLAIM_FIELDS = ("normalized_claim", "claim", "statement", "text", "summary")
_ANSWER_FIELDS = ("answer", "predicted_answer", "prediction", "response", "qa_answer")
_HEDGES = ("可能", "似乎", "也许", "或许", "大概", "据说", "不确定", "应该是", "估计", "恐怕")
_QUOTE_MARKS = ("「", "」", "『", "』", "\"", "“", "”")


def claim_text(record: Any) -> str:
    """Claim/ assertion text only (never falls back to evidence — that would leak)."""
    rec = as_mapping(record)
    for name in _CLAIM_FIELDS:
        value = rec.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    predicate = rec.get("predicate") or rec.get("pred") or rec.get("pred_name")
    roles = rec.get("roles")
    if predicate:
        if isinstance(roles, Mapping) and roles:
            body = "；".join("%s=%s" % (k, v) for k, v in sorted(roles.items()))
            return "%s：%s" % (predicate, body)
        return str(predicate)
    return ""


def answer_text(record: Any) -> str:
    rec = as_mapping(record)
    for name in _ANSWER_FIELDS:
        value = rec.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping) and isinstance(value.get("text"), str):
            return value["text"].strip()
    return ""


def evidence_quotes(record: Any) -> list[str]:
    rec = as_mapping(record)
    raw = rec.get("evidence") or rec.get("quotes") or []
    quotes: list[str] = []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                quotes.append(item.strip())
            elif isinstance(item, Mapping):
                quote = item.get("quote")
                if isinstance(quote, str) and quote.strip():
                    quotes.append(quote.strip())
    return quotes


def _bigrams(text: str) -> set[str]:
    return _char_bigrams(text) if text else set()


def _recall(part: str, whole: str) -> float:
    bp, bw = _bigrams(part), _bigrams(whole)
    if not bp:
        return 0.0
    return len(bp & bw) / len(bp)


def _jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def extract_features(record: Any) -> list[float]:
    """Deterministic ``list[float]`` in :data:`FEATURE_NAMES` order."""
    rec = as_mapping(record)
    claim = claim_text(rec)
    answer = answer_text(rec)
    quotes = evidence_quotes(rec)
    evidence = " ".join(quotes)
    hedge = sum(1 for word in _HEDGES if word in claim)
    values = [
        float(len(claim)),
        float(len(answer)),
        float(len(quotes)),
        float(len(evidence)),
        float(_recall(claim, evidence)),
        float(_recall(answer, evidence)),
        float(_jaccard(claim, answer)),
        1.0 if any(ch.isdigit() for ch in claim) else 0.0,
        float(hedge),
        1.0 if any(mark in claim for mark in _QUOTE_MARKS) else 0.0,
        1.0 if not quotes else 0.0,
        1.0 if claim.rstrip().endswith(("?", "？")) else 0.0,
    ]
    # Defensive: features feed a linear model, so NaN/inf must never escape.
    return [0.0 if not math.isfinite(v) else float(v) for v in values]


def feature_dict(record: Any) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, extract_features(record)))
