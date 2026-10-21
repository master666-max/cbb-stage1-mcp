"""Deterministic evidence anchoring (quote -> exact span).

The AI Reader engine emits free-text `evidence` / `narrative_evidence` strings
with no character offsets.  This package is the only place allowed to turn a
quote into a span: exact substring match, nearest-to-anchor disambiguation,
1-based `occurrence`, and `quote_sha256`.  No model calls happen here.
"""

from cbb_stage1.evidence.anchor import (
    AnchoredQuote,
    anchor_quote,
    choose_occurrence,
    find_occurrences,
    locate_surface,
    make_evidence,
    sha256_text,
    verify_evidence,
)

__all__ = [
    "AnchoredQuote",
    "anchor_quote",
    "choose_occurrence",
    "find_occurrences",
    "locate_surface",
    "make_evidence",
    "sha256_text",
    "verify_evidence",
]
