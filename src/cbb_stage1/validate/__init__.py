"""P3 validation package: conflict detection + confidence/tier checks + P0-C wiring.

Pure validators plus the postprocess integration
(:mod:`cbb_stage1.validate.integration`, entry point ``run``) which is the single
writer of ``work/audit/{conflicts,decisions,quarantine}.jsonl``. File I/O and
pipeline wiring stay in the integration module; the validators are pure.
"""

from __future__ import annotations

from .confidence import (
    CALIBRATION_REQUIRED,
    CLAIM_VERDICTS,
    CONFIDENCE_METHODS,
    INFERENCE_SCOPE,
    TIERS,
    VERIFICATION_LEVELS,
    VERIFICATION_VERDICTS,
    Violation,
    confidence_method_error,
    inference_allowed,
    validate_assertion_record,
    validate_claim_promotion,
    validate_confidence,
    validate_tier,
    validate_verification,
    verification_level,
)
from .conflicts import (
    CONFLICT_KINDS,
    CONFLICT_STATUSES,
    DECISIONS,
    QUARANTINE_REASONS,
    SupersedePlan,
    SupersedeReport,
    TimelineAudit,
    apply_supersede,
    audit_timeline,
    check_supersede_chain,
    detect_attribute_conflicts,
    detect_relation_conflicts,
    detect_timeline_conflicts,
    make_conflict_record,
    record_id,
    retcon_history,
    supersede_chains,
    timeline_participation_violations,
    validate_conflict,
)
from .integration import run

__all__ = [
    "CALIBRATION_REQUIRED", "CLAIM_VERDICTS", "CONFIDENCE_METHODS", "CONFLICT_KINDS",
    "CONFLICT_STATUSES", "DECISIONS", "INFERENCE_SCOPE", "QUARANTINE_REASONS", "TIERS",
    "VERIFICATION_LEVELS", "VERIFICATION_VERDICTS", "SupersedePlan", "SupersedeReport",
    "TimelineAudit", "Violation", "apply_supersede", "audit_timeline", "check_supersede_chain",
    "confidence_method_error", "detect_attribute_conflicts", "detect_relation_conflicts",
    "detect_timeline_conflicts", "inference_allowed", "make_conflict_record", "record_id",
    "retcon_history", "run", "supersede_chains", "timeline_participation_violations",
    "validate_assertion_record", "validate_claim_promotion", "validate_confidence",
    "validate_conflict", "validate_tier", "validate_verification", "verification_level",
]
