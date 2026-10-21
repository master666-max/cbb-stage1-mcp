"""P2-B temporal package: partial-order solving + injectable pairwise NLI + work wiring.

Box contract: ``knowledge/timeline.jsonl`` (design/01 §5.8), invariants I5/I20/I25.
Pure functions plus the postprocess integration (:mod:`cbb_stage1.temporal.integration`,
entry point ``run``) that turns work tables into ``work/assemble/timeline.jsonl``.
D11 NLI: :func:`build_nli` turns ``Config.nli_provider`` into a pairwise callable
(disabled/openai/ollama; offline by default, strict single-token answer parsing).
"""

from __future__ import annotations

from .ids import canonical_json, stable_id
from .integration import (
    append_jsonl,
    as_dict,
    first_table,
    from_outputs,
    outputs_list,
    read_jsonl,
    run,
    unit_order_from,
    write_jsonl,
)
from .nli import (
    DEFAULT_QUESTION_BY_KIND,
    NliResult,
    expected_answer,
    nli_pairwise,
    normalize_result,
    question_for,
    verify_constraint,
)
from .nli_client import NLI_PROVIDERS, build_nli, parse_answer
from .solver import (
    CONFLICT_KIND,
    NODE_KINDS,
    QUARANTINE_REASONS,
    TIMELINE_KINDS,
    TIMELINE_SOURCES,
    TIMELINE_STATUSES,
    ConflictItem,
    Constraint,
    Node,
    QuarantineItem,
    SolveResult,
    TimelineContractError,
    find_cycles,
    make_conflict,
    parse_constraint,
    parse_constraints,
    solve,
    topological_order,
)

__all__ = [
    "CONFLICT_KIND", "DEFAULT_QUESTION_BY_KIND", "NLI_PROVIDERS", "NODE_KINDS",
    "QUARANTINE_REASONS", "TIMELINE_KINDS", "TIMELINE_SOURCES", "TIMELINE_STATUSES",
    "ConflictItem", "Constraint", "NliResult", "Node", "QuarantineItem", "SolveResult",
    "TimelineContractError", "append_jsonl", "as_dict", "build_nli", "canonical_json",
    "expected_answer", "find_cycles", "first_table", "from_outputs", "make_conflict",
    "nli_pairwise", "normalize_result", "outputs_list", "parse_answer", "parse_constraint",
    "parse_constraints", "question_for", "read_jsonl", "run", "solve", "stable_id",
    "topological_order", "unit_order_from", "verify_constraint", "write_jsonl",
]
