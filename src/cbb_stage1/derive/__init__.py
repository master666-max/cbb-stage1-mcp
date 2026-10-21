"""Derived-layer builders (rebuildable, never part of the hard gates).

* :mod:`cbb_stage1.derive.summaries` -- summary-tree interface (§5.14): tree
  shape, parent closure, code-marked ``stale``. No LLM text in stage 1.
* :mod:`cbb_stage1.derive.hypotheses` -- Abductive skeleton (§5.20): derived
  hypotheses only; ``missing_premises`` is allowed only for insufficient
  evidence, and :func:`abduce` refuses sufficient evidence.

Everything here is pure; the box writer decides which rows enter
``derived/``.
"""

from __future__ import annotations

from .community import (
    build_community_reports,
    drilldown,
    rollup_communities,
    validate_community_hierarchy,
)
from .integration import run as run_derive
from .hypotheses import (
    CONSISTENCY_VALUES,
    EVIDENCE_SUFFICIENCY,
    HYPOTHESIS_SCOPES,
    HYPOTHESIS_STATUSES,
    PLAUSIBILITY_VALUES,
    Hypothesis,
    abduce,
    derive_hypotheses,
    make_hypothesis_id,
    normalize_hypothesis_record,
)
from .raptor import (
    EMBEDDING_METHOD,
    STRUCTURAL_METHOD,
    build_raptor_clusters,
    validate_raptor_tree,
)
from .summaries import (
    SUMMARY_CONTRACT,
    SUMMARY_SCOPES,
    SummaryNode,
    build_summary_tree,
    make_summary_id,
    mark_stale,
    summary_rows,
    validate_summary_tree,
)

__all__ = [
    # summaries
    "SUMMARY_CONTRACT",
    "SUMMARY_SCOPES",
    "SummaryNode",
    "make_summary_id",
    "build_summary_tree",
    "summary_rows",
    "validate_summary_tree",
    "mark_stale",
    # hypotheses
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
    # community hierarchy
    "build_community_reports",
    "rollup_communities",
    "drilldown",
    "validate_community_hierarchy",
    # RAPTOR recursive tree
    "STRUCTURAL_METHOD",
    "EMBEDDING_METHOD",
    "build_raptor_clusters",
    "validate_raptor_tree",
    # postprocess integration (P0-D)
    "run_derive",
]
