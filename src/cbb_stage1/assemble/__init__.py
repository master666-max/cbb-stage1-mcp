"""Assembly primitives: predicate registry + deterministic assertion assembly.

* :mod:`cbb_stage1.assemble.registry` -- ``lexicon/predicates.json`` machine
  form: six libraries, role schema (required/repeatable/type), digest for the
  manifest, unknown predicates -> proposals (never auto-registered).
* :mod:`cbb_stage1.assemble.assertions` -- engine assertion -> box assertion:
  registry library wins, role mismatch -> ``quarantine``, claim layer
  (fact/statement/inferred) derived from evidence modes, statement requires a
  speaker, confidence requires a method, speaker/mode pass through.

Not in scope: evidence anchoring (``evidence/``) and writing the box
(``box/``); the assembler returns accepted records and quarantine rows, it does
not persist them.
"""

from __future__ import annotations

from .assertions import (
    CONFIDENCE_METHODS,
    CONFIDENCE_METHOD_ALIASES,
    EVIDENCE_MODES,
    MODE_DEFAULT_CLAIM_LAYER,
    QUARANTINE_REASONS,
    AssembleResult,
    Assembler,
    LayerDowngrade,
    QuarantineItem,
    assemble_assertions,
    assertion_id_for,
    infer_claim_layer,
    normalize_confidence_method,
    resolve_claim_layer,
)
from .registry import (
    LIBRARIES,
    LIBRARY_ALIASES,
    SEED_LEXICON,
    PredicateProposal,
    PredicateRegistry,
    PredicateSpec,
    RegistryError,
    RoleCheck,
    RoleProblem,
    RoleSpec,
    normalize_library,
    propose_predicate,
    seed_registry,
)
from .integration import run

__all__ = [
    # registry
    "LIBRARIES",
    "LIBRARY_ALIASES",
    "SEED_LEXICON",
    "PredicateProposal",
    "PredicateRegistry",
    "PredicateSpec",
    "RegistryError",
    "RoleCheck",
    "RoleProblem",
    "RoleSpec",
    "normalize_library",
    "propose_predicate",
    "seed_registry",
    # assertions
    "CONFIDENCE_METHODS",
    "CONFIDENCE_METHOD_ALIASES",
    "EVIDENCE_MODES",
    "MODE_DEFAULT_CLAIM_LAYER",
    "QUARANTINE_REASONS",
    "AssembleResult",
    "Assembler",
    "LayerDowngrade",
    "QuarantineItem",
    "assemble_assertions",
    "assertion_id_for",
    "infer_claim_layer",
    "normalize_confidence_method",
    "resolve_claim_layer",
    # integration
    "run",
]
