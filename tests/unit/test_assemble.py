"""Unit tests for :mod:`cbb_stage1.assemble` (registry + assertion assembly).

Coverage follows the P2 acceptance points: the registry is the only source of
library/roles, unknown predicates go to proposals (never auto-registered), role
mismatch is ``role_mismatch`` quarantine (negative: wrong role), statement
requires a speaker (negative: statement without speaker), ``library`` cannot be
chosen by data, and confidence requires a method.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cbb_stage1.assemble import (
    CONFIDENCE_METHODS,
    EVIDENCE_MODES,
    LIBRARIES,
    MODE_DEFAULT_CLAIM_LAYER,
    QUARANTINE_REASONS,
    SEED_LEXICON,
    AssembleResult,
    Assembler,
    PredicateRegistry,
    PredicateSpec,
    QuarantineItem,
    RegistryError,
    RoleSpec,
    assemble_assertions,
    assertion_id_for,
    infer_claim_layer,
    normalize_confidence_method,
    normalize_library,
    propose_predicate,
    resolve_claim_layer,
    seed_registry,
)
from cbb_stage1.models import Assertion, Evidence, Speaker

REGISTRY = seed_registry()
SPEAKER = Speaker(kind="entity", entity_id="e:zj:p:1", name="甲")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _evidence(
    *,
    unit_id: str = "u:zj:v01:c001",
    mode: str = "narration",
    speaker: Speaker | None = None,
    quote: str = "拉法",
) -> Evidence:
    return Evidence(
        unit_id=unit_id,
        start=0,
        end=len(quote),
        quote=quote,
        quote_sha256=None,
        occurrence=1,
        speaker=speaker,
        mode=mode,  # type: ignore[arg-type]
    )


def _assertion(**overrides: object) -> Assertion:
    defaults: dict[str, object] = {
        "book_id": "zj",
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "e:zj:p:1", "受事": "e:zj:p:2"},
        "claim_layer": "fact",
        "evidence": [_evidence()],
        "confidence": 0.9,
        "confidence_method": "evidence_span",
    }
    defaults.update(overrides)
    return Assertion(**defaults)  # type: ignore[arg-type]


def _assemble(*assertions: Assertion, **kwargs: object) -> AssembleResult:
    return Assembler(REGISTRY, **kwargs).assemble(assertions)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# registry: libraries, digest, loading
# ---------------------------------------------------------------------------


def test_six_libraries_are_the_only_valid_names():
    assert LIBRARIES == ("实体", "关系", "事件", "埋线", "设定", "时序")
    assert normalize_library("relation") == "关系"
    assert normalize_library("setting") == "设定"
    assert normalize_library("timeline") == "时序"
    with pytest.raises(RegistryError):
        normalize_library("魔法")


def test_seed_registry_matches_the_box_default_registry():
    # The seed mirrors box/lexicon.py default_lexicon: work and box must agree
    # on which predicates are registered, or statements silently disappear.
    assert len(REGISTRY) == 6
    assert "敌对" in REGISTRY
    assert "亲属" in REGISTRY
    assert REGISTRY.library_of("敌对") == "关系"
    assert REGISTRY.library_of("亲属") == "关系"
    assert REGISTRY.library_of("参与") == "事件"
    assert REGISTRY.library_of("unregistered") is None
    assert set(REGISTRY.libraries_in_use()) == {"实体", "关系", "事件", "设定"}
    digest = REGISTRY.digest()
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64


def test_registry_from_lexicon_normalizes_english_library_alias():
    registry = PredicateRegistry.from_lexicon([
        {
            "pred": "亲属",
            "library": "relation",
            "roles": [{"name": "施事", "type": "entity"}, {"name": "受事", "type": "entity"}],
        }
    ])
    assert registry.library_of("亲属") == "关系"


def test_registry_from_path_and_roundtrip_digest(tmp_path):
    path = tmp_path / "predicates.json"
    path.write_text(json.dumps(SEED_LEXICON, ensure_ascii=False), encoding="utf-8")
    loaded = PredicateRegistry.from_path(path)
    assert loaded.digest() == REGISTRY.digest()
    assert PredicateRegistry.from_lexicon(loaded.to_lexicon()).digest() == REGISTRY.digest()


def test_registry_rejects_duplicate_predicate():
    with pytest.raises(RegistryError):
        PredicateRegistry.from_lexicon([
            {"pred": "性质", "library": "设定", "roles": [{"name": "主体"}]},
            {"pred": "性质", "library": "设定", "roles": [{"name": "主体"}]},
        ])


def test_registry_rejects_unknown_library():
    with pytest.raises((RegistryError, ValidationError)):
        PredicateRegistry.from_lexicon([
            {"pred": "X", "library": "魔法", "roles": [{"name": "主体"}]}
        ])


def test_registry_rejects_missing_predicates_list():
    with pytest.raises(RegistryError):
        PredicateRegistry.from_lexicon({"contract": "cbb-lexicon/1"})


def test_register_requires_human_approval_and_is_not_automatic():
    registry = seed_registry()
    spec = PredicateSpec(
        pred="赐予",
        library="关系",
        roles=[RoleSpec(name="施事"), RoleSpec(name="受事")],
    )
    with pytest.raises(RegistryError):
        registry.register(spec)  # no approved_by -> human gate missing
    registry.register(spec, approved_by="owner")
    assert "赐予" in registry
    with pytest.raises(RegistryError):
        registry.register(spec, approved_by="owner")


# ---------------------------------------------------------------------------
# role checking
# ---------------------------------------------------------------------------


def test_check_roles_accepts_registered_shape():
    check = REGISTRY.check_roles("敌对", {"施事": "e:zj:p:1", "受事": "e:zj:p:2"})
    assert check.ok is True
    assert check.library == "关系"
    assert check.problems == ()


def test_check_roles_unknown_predicate():
    check = REGISTRY.check_roles("不存在", {"施事": "e:zj:p:1"})
    assert check.ok is False
    assert check.library is None
    assert check.problems[0].kind == "unknown_role"


def test_check_roles_rejects_wrong_role_names():
    # negative: wrong roles (主体/对象 instead of 施事/受事)
    check = REGISTRY.check_roles("敌对", {"主体": "e:zj:p:1", "对象": "e:zj:p:2"})
    assert check.ok is False
    kinds = {problem.kind for problem in check.problems}
    assert {"unknown_role", "missing_role"} <= kinds
    assert check.reason == "role_mismatch"
    assert "'施事'" in check.detail()


def test_check_roles_rejects_wrong_value_types():
    entity_check = REGISTRY.check_roles("敌对", {"施事": "甲", "受事": "e:zj:p:2"})
    assert entity_check.ok is False
    assert "entity role expects" in entity_check.detail()

    # A bare string is rejected here on purpose: the value shape is owned by the
    # emitting end (`predicate_map.literal`). Tolerating it at this layer too
    # would give the contract two safety nets, and the full-chain gate would stop
    # being able to detect a break (verified by mutation).
    bare_literal = REGISTRY.check_roles("性质", {"主体": "e:zj:p:1", "值": "凡人"})
    assert bare_literal.ok is False
    assert "literal role expects" in bare_literal.detail()

    good_literal = REGISTRY.check_roles(
        "性质", {"主体": "e:zj:p:1", "值": {"literal": "凡人"}}
    )
    assert good_literal.ok is True

    # A non-string, non-wrapper value is still rejected.
    bad_literal = REGISTRY.check_roles("性质", {"主体": "e:zj:p:1", "值": 42})
    assert bad_literal.ok is False
    assert "literal role expects" in bad_literal.detail()


def test_check_roles_repeatable_contract():
    ok = REGISTRY.check_roles(
        "参与", {"事件": "e:ev:1", "参与者": ["e:zj:p:1", "e:zj:p:2"]}
    )
    assert ok.ok is True
    not_repeatable = REGISTRY.check_roles("敌对", {"施事": ["e:zj:p:1"], "受事": "e:zj:p:2"})
    assert not_repeatable.ok is False
    empty_repeatable = REGISTRY.check_roles("参与", {"事件": "e:ev:1", "参与者": []})
    assert empty_repeatable.ok is False


def test_check_roles_rejects_empty_mapping():
    check = REGISTRY.check_roles("敌对", {})
    assert check.ok is False
    assert check.problems[0].kind == "roles_shape"


# ---------------------------------------------------------------------------
# proposals: unknown predicates never auto-register
# ---------------------------------------------------------------------------


def test_propose_predicate_shape_and_determinism():
    proposal = propose_predicate(
        "赐予", book_id="zj", seen_in=["u:zj:v01:c002"], count=4,
        role_shape_examples=[{"施事": "e:a", "受事": "e:b"}],
    )
    assert proposal.status == "proposed"
    assert proposal.decided_by is None
    assert proposal.proposal_id.startswith("pp:")
    assert proposal.proposal_id == propose_predicate(
        "赐予", book_id="zj", seen_in=["u:zj:v01:c002"]
    ).proposal_id
    assert proposal.seen_in == ["u:zj:v01:c002"]


# ---------------------------------------------------------------------------
# claim-layer inference
# ---------------------------------------------------------------------------


def test_infer_claim_layer_from_evidence_modes():
    assert infer_claim_layer([_evidence(mode="narration")]) == "fact"
    assert infer_claim_layer([_evidence(mode="dialogue")]) == "statement"
    assert infer_claim_layer([_evidence(mode="inner")]) == "statement"
    assert infer_claim_layer([_evidence(mode="memory")]) == "statement"
    assert infer_claim_layer([]) == "fact"
    assert (
        infer_claim_layer([_evidence(mode="narration")], narrator_reliability="unreliable")
        == "statement"
    )
    assert MODE_DEFAULT_CLAIM_LAYER["narration"] == "fact"
    assert MODE_DEFAULT_CLAIM_LAYER["letter"] == "statement"


def test_resolve_claim_layer_downgrades_only_implicit_fact():
    evidence = [_evidence(mode="dialogue")]
    assert resolve_claim_layer("fact", evidence) == ("statement", True)
    assert resolve_claim_layer("statement", evidence) == ("statement", False)
    assert resolve_claim_layer("inferred", evidence) == ("inferred", False)
    assert resolve_claim_layer("fact", [_evidence(mode="narration")]) == ("fact", False)


# ---------------------------------------------------------------------------
# assembler: happy paths
# ---------------------------------------------------------------------------


def test_assembler_accepts_narration_fact_and_sets_registry_library():
    result = _assemble(_assertion())
    assert result.accepted_count == 1
    accepted = result.accepted[0]
    assert accepted.claim_layer == "fact"
    assert accepted.library == "关系"
    assert accepted.status == "asserted"
    assert accepted.assertion_id.startswith("a:") and len(accepted.assertion_id) == 18
    assert accepted.confidence_method == "evidence_span"
    assert accepted.evidence[0].mode == "narration"
    assert "u:zj:v01:c001" in accepted.deps
    assert {"e:zj:p:1", "e:zj:p:2"} <= set(accepted.deps)


def test_assembler_assertion_id_is_content_addressed():
    first = _assemble(_assertion()).accepted[0].assertion_id
    second = _assemble(_assertion()).accepted[0].assertion_id
    assert first == second
    different = _assemble(
        _assertion(roles={"施事": "e:zj:p:9", "受事": "e:zj:p:2"})
    ).accepted[0].assertion_id
    assert different != first


def test_assembler_downgrades_implicit_fact_with_dialogue_evidence():
    evidence = _evidence(mode="dialogue", speaker=SPEAKER)
    result = _assemble(_assertion(claim_layer="fact", evidence=[evidence]))
    assert result.quarantined == []
    assert result.downgrades[0].requested == "fact"
    assert result.downgrades[0].applied == "statement"
    accepted = result.accepted[0]
    assert accepted.claim_layer == "statement"
    # speaker/mode pass-through: speaker came from the evidence
    assert accepted.speaker is not None and accepted.speaker.entity_id == "e:zj:p:1"
    assert accepted.evidence[0].mode == "dialogue"
    assert accepted.evidence[0].speaker is not None


def test_assembler_passes_through_explicit_statement_speaker():
    result = _assemble(
        _assertion(
            claim_layer="statement",
            speaker=SPEAKER,
            evidence=[_evidence(mode="dialogue")],
        )
    )
    assert result.accepted[0].claim_layer == "statement"
    assert result.accepted[0].speaker == SPEAKER


def test_assembler_accepts_inferred_with_derived_from():
    result = _assemble(
        _assertion(claim_layer="inferred", derived_from=["a:0123456789abcdef"], evidence=[])
    )
    assert result.accepted[0].claim_layer == "inferred"
    assert result.accepted[0].status == "inferred"


def test_assembler_normalizes_legacy_confidence_method():
    assert normalize_confidence_method("engine_self_report") == "engine_self"
    assert normalize_confidence_method("handwave") is None
    result = _assemble(_assertion(confidence_method="engine_self_report"))
    assert result.accepted[0].confidence_method == "engine_self"


def test_assembler_accepts_english_library_alias_from_data():
    # Data says "relation", registry says 关系: same library, accepted.
    result = _assemble(_assertion(library="relation"))
    assert result.accepted_count == 1
    assert result.accepted[0].library == "关系"


def test_assembler_calibrated_requires_version_and_accepts_with_one():
    without = _assemble(_assertion(confidence_method="calibrated"))
    assert without.quarantined[0].reason == "schema"
    with_version = _assemble(
        _assertion(
            confidence_method="calibrated",
            produced_by={"calibration_version": "cal-2026-10"},
        )
    )
    assert with_version.accepted_count == 1


def test_assembler_deduplicates_identical_assertions():
    result = _assemble(_assertion(), _assertion())
    assert result.accepted_count == 1
    assert result.duplicates_dropped == 1


def test_assemble_assertions_convenience_wrapper():
    result = assemble_assertions([_assertion()], REGISTRY, book_id="zj")
    assert isinstance(result, AssembleResult)
    assert result.accepted_count == 1


# ---------------------------------------------------------------------------
# assembler: negatives
# ---------------------------------------------------------------------------


def test_assembler_quarantines_wrong_roles():
    # negative: role error must not be silently accepted
    result = _assemble(_assertion(roles={"主体": "e:zj:p:1", "对象": "e:zj:p:2"}))
    assert result.accepted_count == 0
    assert result.quarantined[0].reason == "role_mismatch"
    assert result.quarantined[0].kind == "assertion"


def test_assembler_quarantines_statement_without_speaker():
    # negative: statement missing speaker
    result = _assemble(
        _assertion(
            claim_layer="statement",
            speaker=None,
            evidence=[_evidence(mode="dialogue", speaker=None)],
        )
    )
    assert result.accepted_count == 0
    item = result.quarantined[0]
    assert item.reason == "claim_layer_conflict"
    assert "speaker" in item.detail


def test_unreliable_narration_is_kept_as_a_statement_attributed_to_the_narrator():
    """It used to be quarantined, so the flag deleted narration instead of lowering its layer.

    "The narrator claims X" is the honest reading of unreliable narration, and it is a claim a reader
    needs — dropping it loses the evidence that anything was said at all. The old assertion expected
    `accepted_count == 0`, which pinned the loss as the specification.
    """
    result = _assemble(
        _assertion(claim_layer="fact", speaker=None),
        narrator_reliability="unreliable",
    )
    assert result.accepted_count == 1
    assert not result.quarantined
    assert result.downgrades[0].applied == "statement"
    accepted = result.accepted[0]
    assert accepted.claim_layer == "statement"
    assert accepted.speaker is not None and accepted.speaker.kind == "narrator"


def test_assembler_unknown_predicate_goes_to_proposals_only():
    result = _assemble(_assertion(predicate="赐予", roles={"施事": "e:a", "受事": "e:b"}))
    assert result.accepted_count == 0
    assert result.proposal_count == 1
    assert result.proposals[0].predicate == "赐予"
    assert result.proposals[0].status == "proposed"
    assert result.quarantined[0].reason == "schema"
    assert "赐予" not in REGISTRY


def test_assembler_quarantines_data_chosen_library():
    result = _assemble(_assertion(library="设定"))
    assert result.accepted_count == 0
    assert result.quarantined[0].reason == "schema"
    assert "registry library" in result.quarantined[0].detail


def test_assembler_quarantines_unknown_library_value():
    result = _assemble(_assertion(library="魔法"))
    assert result.quarantined[0].reason == "schema"


def test_assembler_quarantines_asserted_without_evidence():
    result = _assemble(_assertion(evidence=[]))
    assert result.accepted_count == 0
    assert result.quarantined[0].reason == "evidence_not_found"


def test_assembler_quarantines_confidence_without_method():
    result = _assemble(_assertion(confidence=0.5, confidence_method=None))
    assert result.accepted_count == 0
    assert result.quarantined[0].reason == "schema"
    assert "not trusted" in result.quarantined[0].detail


def test_assembler_quarantines_invalid_confidence_method():
    result = _assemble(_assertion(confidence_method="handwave"))
    assert result.quarantined[0].reason == "schema"


def test_assembler_quarantines_inferred_without_premises():
    result = _assemble(_assertion(claim_layer="inferred", evidence=[], derived_from=[]))
    assert result.quarantined[0].reason == "claim_layer_conflict"


def test_assembler_quarantines_fact_derived_only_from_statements():
    result = _assemble(
        _assertion(
            claim_layer="fact",
            derived_from=["a:statement-one"],
            evidence=[_evidence(mode="dialogue", speaker=SPEAKER)],
            verification={"level": "major", "verdict": "support"},
        ),
        statement_ids={"a:statement-one"},
    )
    assert result.accepted_count == 0
    assert result.quarantined[0].reason == "claim_layer_conflict"
    assert "statement" in result.quarantined[0].detail


def test_assembler_accepts_statement_derivation_with_narration_evidence():
    result = _assemble(
        _assertion(claim_layer="fact", derived_from=["a:statement-one"]),
        statement_ids={"a:statement-one"},
    )
    assert result.accepted_count == 1
    assert result.accepted[0].claim_layer == "fact"


def test_quarantine_reason_is_closed_set():
    assert "role_mismatch" in QUARANTINE_REASONS
    assert "claim_layer_conflict" in QUARANTINE_REASONS
    with pytest.raises(ValidationError):
        QuarantineItem(item_id="a:1", reason="made_up", detail="x")


def test_confidence_methods_and_modes_match_contract():
    assert "claim_qa" in CONFIDENCE_METHODS
    assert "calibrated" in CONFIDENCE_METHODS
    assert EVIDENCE_MODES == ("narration", "dialogue", "inner", "memory", "document", "letter")
