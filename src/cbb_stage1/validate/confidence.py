"""Confidence / tier / verification validation (design/01 §5.15, invariants I19/I23).

Design intent:

* **No bare numbers**: a non-zero ``confidence`` without a ``confidence_method``
  is not trusted (I19); ``calibrated`` additionally needs ``calibration_version``.
* **Tier decides verification strength**: ``canonical`` -> CoVe + atomic-fact
  verification with a *different* model tier than extraction; ``major`` ->
  evidence-span checking; ``minor`` -> schema/format only. The bridge here is
  code, so the mapping cannot drift with prompts.
* **Claims**: ``insufficient_evidence`` must never be promoted to ``fact``
  (I23); promotion from ``statement`` needs >=2 independent source units and no
  ``against`` evidence (§5.16). This module only *checks* the promotion rule;
  the claim/QA pipeline that produces the verdict lives elsewhere (W3).

Pure functions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

TIERS = ("canonical", "major", "minor")

# §5.15 置信方法闭集
CONFIDENCE_METHODS = (
    "engine_self",       # 模型自报，低可信
    "evidence_span",     # 逐字证据，强
    "uqlm_whitebox",     # token 概率
    "uqlm_blackbox",     # 多次生成
    "semantic_entropy",  # 采样
    "uq_heads",          # 需训练，claim 级
    "claim_qa",          # QA 一致性
    "human",
    "calibrated",        # 金标校准后，必须附 calibration_version
)
CALIBRATION_REQUIRED = ("calibrated",)

#: §5.15: ``verification.level`` ∈ ``canonical/major/minor/none`` and is DERIVED
#: from ``entity.tier``. The level is therefore the tier itself — it names how much
#: verification this assertion is entitled to.
#:
#: This used to map tier -> a *strength name* (``cove_atomic`` / ``evidence_span`` /
#: ``schema_only``), which is a different concept: that is what the verifier should
#: DO, not which level the record IS. The box schema accepts only
#: ``canonical/major/minor/none``, so the strength name was silently dropped by the
#: box writer and every assertion reached the box with ``verification = null``
#: (measured 7/7) even after the stamping code was added.
VERIFICATION_LEVELS = {
    "canonical": "canonical",
    "major": "major",
    "minor": "minor",
}

#: What each level actually does (the strength, kept separate from the level name).
VERIFICATION_STRENGTH = {
    "canonical": "cove_atomic",     # 独立验证问题 + 原子断言验证 + 异档模型
    "major": "evidence_span",       # verifier 只判"原文是否支持"
    "minor": "schema_only",         # 仅 schema + 格式恢复
}
INFERENCE_SCOPE = {"canonical": "open", "major": "motivation_chain", "minor": "off"}
VERIFICATION_VERDICTS = ("support", "against", "unsure", "none")
CLAIM_VERDICTS = ("supported", "refuted", "conflicting", "insufficient_evidence")


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str
    field: str | None = None


def _dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump())
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("记录必须是 Mapping 或 pydantic 模型，现得 %r" % (type(record),))


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------
def confidence_method_error(record: Any) -> tuple[Violation, ...]:
    """I19: confidence present => method present/known; calibrated => version."""
    rec = _dict(record)
    confidence = rec.get("confidence")
    method = rec.get("confidence_method")
    violations: list[Violation] = []
    if confidence is None:
        return ()
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or \
            not 0.0 <= float(confidence) <= 1.0:
        violations.append(Violation("confidence_out_of_range",
                                    "confidence 必须在 [0,1]：%r" % (confidence,), "confidence"))
    if method is None or method == "":
        if confidence:
            violations.append(Violation("confidence_method_missing",
                                        "confidence=%r 但没有 confidence_method（不允许拍数，I19）"
                                        % (confidence,), "confidence_method"))
        return tuple(violations)
    if not isinstance(method, str) or method not in CONFIDENCE_METHODS:
        violations.append(Violation("confidence_method_unknown",
                                    "confidence_method 不属闭集 %s：%r"
                                    % (CONFIDENCE_METHODS, method), "confidence_method"))
        return tuple(violations)
    if method in CALIBRATION_REQUIRED:
        version = rec.get("calibration_version")
        if not (isinstance(version, str) and version.strip()):
            violations.append(Violation("calibration_version_missing",
                                        "method=calibrated 必须附 calibration_version（I19）",
                                        "calibration_version"))
    return tuple(violations)


def validate_confidence(record: Any) -> tuple[Violation, ...]:
    """Alias kept explicit for callers that only care about I19."""
    return confidence_method_error(record)


# ---------------------------------------------------------------------------
# tier -> verification strength
# ---------------------------------------------------------------------------
def verification_level(tier: str) -> str | None:
    return VERIFICATION_LEVELS.get(tier)


def inference_allowed(tier: str) -> str | None:
    """``open`` / ``motivation_chain`` / ``off``; ``None`` for unknown tier."""
    return INFERENCE_SCOPE.get(tier)


def validate_tier(tier: Any, field: str = "tier") -> tuple[Violation, ...]:
    if tier not in TIERS:
        return (Violation("tier_unknown", "tier 不属闭集 %s：%r" % (TIERS, tier), field),)
    return ()


def validate_verification(tier: Any, verification: Any, *,
                          extractor_model: str | None = None,
                          verifier_model: str | None = None) -> tuple[Violation, ...]:
    """§5.15 verification matrix: level derived from tier; canonical needs model decoupling.

    ``model_decoupled=false`` is allowed **only** as an explicit QC debt; a
    silent same-model self-check (extractor == verifier) is a violation.
    """
    violations: list[Violation] = list(validate_tier(tier))
    if violations:
        return tuple(violations)
    if verification is None:
        if tier == "canonical":
            violations.append(Violation("verification_missing",
                                        "canonical 必须有 verification（CoVe + 原子验证）", "verification"))
        return tuple(violations)
    if not isinstance(verification, Mapping):
        return (Violation("verification_shape", "verification 必须是对象", "verification"),)
    expected = VERIFICATION_LEVELS[tier]
    level = verification.get("level")
    if level is not None and level != expected:
        violations.append(Violation("verification_level_mismatch",
                                    "tier=%s 的验证强度应为 %r，现得 %r"
                                    % (tier, expected, level), "verification.level"))
    verdict = verification.get("verdict")
    if verdict not in VERIFICATION_VERDICTS:
        violations.append(Violation("verification_verdict_invalid",
                                    "verdict 不属闭集 %s：%r"
                                    % (VERIFICATION_VERDICTS, verdict), "verification.verdict"))
    if not (verification.get("verifier") or verification.get("model")):
        violations.append(Violation("verification_provenance_missing",
                                    "verification 必须落 verifier/model（可审计）",
                                    "verification.verifier"))
    if tier == "canonical" and extractor_model and verifier_model and \
            extractor_model == verifier_model:
        violations.append(Violation("model_not_decoupled",
                                    "canonical 的验证模型与抽取模型同档（%r）——同模型自证不算验证；"
                                    "要么换档，要么在 qc/report 显式记 model_decoupled=false"
                                    % (extractor_model,), "verification.model"))
    return tuple(violations)


# ---------------------------------------------------------------------------
# claim promotion
# ---------------------------------------------------------------------------
def validate_claim_promotion(*, claim_layer: Any, verdict: Any, source_units: Iterable[str] = (),
                             against: int = 0, target_layer: str = "fact") -> tuple[Violation, ...]:
    """I23/§5.16: only a supported, corroborated claim may be promoted to fact."""
    violations: list[Violation] = []
    units = {u for u in source_units if u}
    if verdict not in CLAIM_VERDICTS:
        violations.append(Violation("claim_verdict_unknown",
                                    "claim verdict 不属四值闭集 %s：%r"
                                    % (CLAIM_VERDICTS, verdict), "verdict"))
        return tuple(violations)
    if target_layer != "fact":
        return ()
    if verdict == "insufficient_evidence":
        violations.append(Violation("insufficient_promoted",
                                    "insufficient_evidence 不得升 fact（I23）", "verdict"))
    elif verdict == "refuted":
        violations.append(Violation("refuted_promoted", "refuted 不得升 fact", "verdict"))
    elif verdict == "conflicting":
        violations.append(Violation("conflicting_promoted",
                                    "conflicting 不得直接升 fact（须先裁决）", "verdict"))
    elif verdict == "supported":
        if len(units) < 2:
            violations.append(Violation("corroboration_insufficient",
                                        "statement→fact 至少 2 个独立来源单元（§5.16；现 %d）"
                                        % len(units), "source_units"))
        if against:
            violations.append(Violation("against_present",
                                        "有 %d 条 against 证据时不得升 fact（§5.16）" % against,
                                        "evidence"))
    return tuple(violations)


def validate_assertion_record(assertion: Any, *, tier: Any = None) -> tuple[Violation, ...]:
    """Convenience: I19 + (§5.15 if tier given) over one stored assertion."""
    rec = _dict(assertion)
    violations: list[Violation] = list(confidence_method_error(rec))
    if tier is not None:
        violations.extend(validate_verification(tier, rec.get("verification")))
        violations.extend(validate_tier(tier))
    return tuple(violations)
