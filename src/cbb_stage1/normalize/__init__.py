"""Normalizers: LLM payload cleanup, ChapterFact -> EngineOutput, A7 cleaning.

Everything here is deterministic code; no model calls.  The engine produces
`ChapterFact` (AI Reader nine-class schema), :mod:`normalize.facts` converts it
into the V5 `EngineOutput` contract used by Evidence/Resolver/Assembler, and
:mod:`normalize.cleaning` applies the ten A7 semantic-cleaning classes before
those stages (quarantine reasons from design/01 §5.12).
"""

from cbb_stage1.normalize.cleaning import (
    CLEANING_RULES,
    QUARANTINE_REASONS,
    CleaningReport,
    QuarantineItem,
    clean_engine_output,
)
from cbb_stage1.normalize.facts import (
    chapter_fact_to_engine_output,
    sanitize_mapped_output,
    validate_mapped_outputs,
)
from cbb_stage1.normalize.fields import normalize_llm_payload

__all__ = [
    "CLEANING_RULES",
    "QUARANTINE_REASONS",
    "CleaningReport",
    "QuarantineItem",
    "chapter_fact_to_engine_output",
    "clean_engine_output",
    "normalize_llm_payload",
    "sanitize_mapped_output",
    "validate_mapped_outputs",
]
