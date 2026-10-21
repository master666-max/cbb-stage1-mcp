"""Unit tests for :mod:`cbb_stage1.validate` (conflicts + superseded + retcon + confidence).

Contract: design/01 §5.12/§5.15/§5.17, invariants I17/I19/I23. Negative cases
required by the Lead are explicit: **superseded cycle**, dangling/self pointers,
three conflict classes, and ``insufficient_evidence`` never promoted to fact.
"""

from __future__ import annotations

import copy

import pytest

from cbb_stage1.temporal import NliResult
from cbb_stage1.validate import (
    CONFLICT_KINDS, DECISIONS, apply_supersede, audit_timeline, check_supersede_chain,
    detect_attribute_conflicts, detect_relation_conflicts, detect_timeline_conflicts,
    inference_allowed, make_conflict_record, record_id, retcon_history, validate_claim_promotion,
    validate_confidence, validate_conflict, validate_tier, validate_verification,
    verification_level,
)


def assertion(aid, subject="e:1", predicate="年龄", literal="17", **over):
    rec = {
        "assertion_id": aid, "book_id": "zj", "predicate": predicate, "library": "实体",
        "roles": {"主体": subject, "值": {"literal": literal}},
        "claim_layer": "fact", "status": "asserted", "evidence": [], "confidence": 0.9,
        "confidence_method": "evidence_span", "derived_from": [], "superseded_by": None,
    }
    rec.update(over)
    return rec


def timeline_before(cid, left, right):
    return {"constraint_id": cid, "book_id": "zj", "kind": "before",
            "left": {"kind": "event", "id": left},
            "right": {"kind": "event", "id": right}}


def event(eid, participants, unit="u:1", **over):
    rec = {"event_id": eid, "book_id": "zj", "type": "活动", "summary": "",
           "participants": participants, "unit_id": unit, "status": "asserted"}
    rec.update(over)
    return rec


# ---------------------------------------------------------------------------
# attribute conflicts
# ---------------------------------------------------------------------------
def test_attribute_conflict_same_subject_predicate_different_literal():
    conflicts = detect_attribute_conflicts([assertion("a:1", literal="17"),
                                           assertion("a:2", literal="19")])
    assert len(conflicts) == 1
    record = conflicts[0]
    assert record["kind"] == "attribute"
    assert record["left"] == "a:1" and record["right"] == "a:2"
    assert "17" in record["detail"] and "19" in record["detail"]
    assert record["status"] == "open" and record["decision"] is None


def test_attribute_conflict_same_value_is_not_a_conflict():
    assert detect_attribute_conflicts([assertion("a:1"), assertion("a:2")]) == ()


def test_attribute_conflict_different_subjects_or_predicates_not_flagged():
    assert detect_attribute_conflicts([assertion("a:1", subject="e:1"),
                                       assertion("a:2", subject="e:2")]) == ()
    assert detect_attribute_conflicts([assertion("a:1", predicate="年龄"),
                                       assertion("a:2", predicate="身高")]) == ()


def test_attribute_conflict_does_not_cross_claim_layers():
    facts = [assertion("a:1", literal="17", claim_layer="fact")]
    statements = [assertion("a:2", literal="19", claim_layer="statement")]
    assert detect_attribute_conflicts(facts + statements) == ()


def test_attribute_conflict_ignores_non_asserted_and_relation_library():
    assert detect_attribute_conflicts([assertion("a:1"), assertion("a:2", status="inferred")]) == ()
    rel = assertion("a:3", library="关系", roles={"主体": "e:1", "值": {"literal": "17"}})
    assert detect_attribute_conflicts([assertion("a:1"), rel]) == ()


def test_attribute_conflict_ids_are_stable():
    one = detect_attribute_conflicts([assertion("a:1", literal="17"),
                                      assertion("a:2", literal="19")])
    two = detect_attribute_conflicts([assertion("a:2", literal="19"),
                                      assertion("a:1", literal="17")])
    assert one[0]["conflict_id"] == two[0]["conflict_id"]
    assert one[0]["conflict_id"].startswith("c:")


# ---------------------------------------------------------------------------
# relation conflicts
# ---------------------------------------------------------------------------
def relation(aid, roles, predicate="共立", library="关系", **over):
    rec = {"assertion_id": aid, "book_id": "zj", "predicate": predicate,
           "library": library, "roles": roles, "claim_layer": "fact",
           "status": "asserted"}
    rec.update(over)
    return rec


def test_relation_reversed_direction_is_conflict():
    conflicts = detect_relation_conflicts([
        relation("a:1", {"施事": "e:1", "受事": "e:2"}),
        relation("a:2", {"施事": "e:2", "受事": "e:1"}),
    ])
    assert len(conflicts) == 1
    assert conflicts[0]["kind"] == "relation"
    assert "方向相反" in conflicts[0]["detail"]


def test_relation_same_direction_is_not_conflict():
    assert detect_relation_conflicts([
        relation("a:1", {"施事": "e:1", "受事": "e:2"}),
        relation("a:2", {"施事": "e:1", "受事": "e:2"}),
    ]) == ()


def test_relation_exclusive_predicates_flagged():
    conflicts = detect_relation_conflicts([
        relation("a:1", {"施事": "e:1", "受事": "e:2"}, predicate="师兄"),
        relation("a:2", {"施事": "e:1", "受事": "e:2"}, predicate="师弟"),
    ], exclusive_predicates=[frozenset({"师兄", "师弟"})])
    assert len(conflicts) == 1
    assert "互斥" in conflicts[0]["detail"]


def test_relation_different_participants_not_flagged():
    assert detect_relation_conflicts([
        relation("a:1", {"施事": "e:1", "受事": "e:2"}),
        relation("a:2", {"施事": "e:3", "受事": "e:4"}),
    ]) == ()


# ---------------------------------------------------------------------------
# timeline conflicts (cycles / during / dead participation)
# ---------------------------------------------------------------------------
def test_timeline_cycle_becomes_conflict():
    records = [timeline_before("t:aa", "ev:a", "ev:b"),
               timeline_before("t:bb", "ev:b", "ev:a")]
    conflicts = detect_timeline_conflicts(records)
    assert conflicts and all(c["kind"] == "timeline" for c in conflicts)
    assert any("环" in c["detail"] for c in conflicts)


def test_timeline_during_contradiction_becomes_conflict():
    records = [timeline_before("t:aa", "ev:a", "ev:b"),
               {"constraint_id": "t:dd", "book_id": "zj", "kind": "during",
                "left": {"kind": "event", "id": "ev:a"},
                "right": {"kind": "event", "id": "ev:b"}}]
    conflicts = detect_timeline_conflicts(records)
    assert any("during" in c["detail"] for c in conflicts)


def test_timeline_audit_returns_quarantine_records():
    audit = audit_timeline([timeline_before("t:aa", "ev:a", "ev:a")])
    assert audit.result.has_cycle is True
    assert audit.quarantine and audit.quarantine[0]["reason"] == "timeline_cycle"
    assert set(audit.quarantine[0]) == {"item_id", "kind", "reason", "detail", "source_run"}


def test_timeline_nli_contradiction_becomes_conflict():
    audit = audit_timeline([timeline_before("t:aa", "ev:a", "ev:b")],
                           nli=lambda l, r, q: NliResult(question=q, result="no"),
                           verify=True)
    assert len(audit.conflicts) == 1
    assert "NLI" in audit.conflicts[0]["detail"]


def test_dead_entity_participation_is_timeline_conflict():
    events = [event("ev:1", ["e:1"], unit="u:3")]
    conflicts = detect_timeline_conflicts(
        [], events=events, entity_lifetimes={"e:1": {"died_at_unit": "u:2"}},
        unit_order={"u:2": 2, "u:3": 3})
    assert len(conflicts) == 1
    assert conflicts[0]["kind"] == "timeline"
    assert "已死亡" in conflicts[0]["detail"]


def test_dead_entity_lexical_fallback():
    events = [event("ev:1", ["e:1"], unit="u:9")]
    conflicts = detect_timeline_conflicts(
        [], events=events, entity_lifetimes={"e:1": "u:2"})
    assert len(conflicts) == 1


def test_live_entity_participation_not_flagged():
    events = [event("ev:1", ["e:1"], unit="u:2")]
    assert detect_timeline_conflicts(
        [], events=events, entity_lifetimes={"e:1": {"died_at_unit": "u:5"}},
        unit_order={"u:2": 2, "u:5": 5}) == ()


# ---------------------------------------------------------------------------
# superseded closure (I17)
# ---------------------------------------------------------------------------
def test_supersede_chain_valid_and_closed():
    records = [assertion("a:1", status="superseded", superseded_by="a:2"),
               assertion("a:2", derived_from=["a:1"])]
    report = check_supersede_chain(records)
    assert report.ok is True
    assert report.chains["a:1"] == ("a:1", "a:2")


def test_superseded_without_pointer_is_error():
    report = check_supersede_chain([assertion("a:1", status="superseded")])
    assert not report.ok
    assert [e.code for e in report.errors] == ["superseded_without_pointer"]


def test_superseded_dangling_pointer_is_error():
    report = check_supersede_chain([assertion("a:1", status="superseded",
                                              superseded_by="a:404")])
    assert [e.code for e in report.errors] == ["superseded_target_missing"]


def test_superseded_self_reference_is_error():
    report = check_supersede_chain([assertion("a:1", status="superseded",
                                              superseded_by="a:1")])
    assert [e.code for e in report.errors] == ["superseded_self_reference"]


def test_live_record_with_pointer_is_error():
    report = check_supersede_chain([assertion("a:1", superseded_by="a:2"),
                                    assertion("a:2")])
    assert "superseded_by_on_live_record" in [e.code for e in report.errors]


def test_superseded_cycle_is_error():
    records = [assertion("a:1", status="superseded", superseded_by="a:2"),
               assertion("a:2", status="superseded", superseded_by="a:1")]
    report = check_supersede_chain(records)
    assert not report.ok
    assert "superseded_cycle" in [e.code for e in report.errors]
    assert report.cycles and set(report.cycles[0]) == {"a:1", "a:2"}


def test_supersede_three_chain_ok():
    records = [assertion("a:1", status="superseded", superseded_by="a:2"),
               assertion("a:2", status="superseded", superseded_by="a:3"),
               assertion("a:3")]
    report = check_supersede_chain(records)
    assert report.ok is True
    assert report.chains["a:1"] == ("a:1", "a:2", "a:3")
    assert report.chains["a:2"] == ("a:2", "a:3")


# ---------------------------------------------------------------------------
# apply_supersede / retcon
# ---------------------------------------------------------------------------
def test_apply_supersede_keeps_both_records_and_links_them():
    old = assertion("a:1", literal="17")
    new = assertion("a:2", literal="19")
    plan = apply_supersede([old, new], loser_id="a:1", winner_id="a:2",
                           decided_by="arbiter", reason="新章更正", at="2026-10-18")
    by_id = {record_id(r): r for r in plan.records}
    assert by_id["a:1"]["status"] == "superseded"
    assert by_id["a:1"]["superseded_by"] == "a:2"
    assert "a:1" in by_id["a:2"]["derived_from"]
    assert len(plan.records) == 2                     # 不删除 (I8/I17)
    assert check_supersede_chain(plan.records).ok is True


def test_apply_supersede_is_pure():
    old = assertion("a:1")
    new = assertion("a:2")
    snapshot = copy.deepcopy([old, new])
    apply_supersede([old, new], loser_id="a:1", winner_id="a:2",
                    decided_by="arbiter", reason="r")
    assert [old, new] == snapshot


def test_apply_supersede_decision_record_shape():
    plan = apply_supersede([assertion("a:1"), assertion("a:2")], loser_id="a:1",
                           winner_id="a:2", decided_by="arbiter", reason="新章更正")
    decision = plan.decision
    assert decision["decision_id"].startswith("d:")
    assert decision["kind"] == "supersede"
    assert decision["subject"] == "a:1" and decision["choice"] == "superseded"
    assert decision["reason"] == "新章更正" and decision["decided_by"] == "arbiter"


def test_apply_supersede_retcon_has_old_and_new_values():
    plan = apply_supersede([assertion("a:1", literal="17"),
                            assertion("a:2", literal="19")],
                           loser_id="a:1", winner_id="a:2", decided_by="arbiter",
                           reason="新章更正")
    assert plan.retcon["old"]["value"] == "17"
    assert plan.retcon["new"]["value"] == "19"
    assert plan.retcon["reason"] == "新章更正"


def test_apply_supersede_rejects_same_or_missing_ids():
    with pytest.raises(ValueError):
        apply_supersede([assertion("a:1")], loser_id="a:1", winner_id="a:1",
                        decided_by="x", reason="r")
    with pytest.raises(KeyError):
        apply_supersede([assertion("a:1")], loser_id="a:404", winner_id="a:1",
                        decided_by="x", reason="r")


def test_retcon_history_renders_old_new_and_reason():
    plan = apply_supersede([assertion("a:1", literal="17"),
                            assertion("a:2", literal="19")],
                           loser_id="a:1", winner_id="a:2", decided_by="arbiter",
                           reason="新章更正", at="2026-10-18")
    rows = retcon_history(plan.records, decisions=[plan.decision])
    assert len(rows) == 1
    row = rows[0]
    assert row["old_id"] == "a:1" and row["new_id"] == "a:2"
    assert row["old"]["value"] == "17" and row["new"]["value"] == "19"
    assert row["reason"] == "新章更正" and row["decided_by"] == "arbiter"


def test_retcon_history_handles_missing_target():
    rows = retcon_history([assertion("a:1", status="superseded", superseded_by="a:404")])
    assert rows[0]["new"] is None


# ---------------------------------------------------------------------------
# conflict record validation
# ---------------------------------------------------------------------------
def test_valid_conflict_record_passes():
    record = make_conflict_record("zj", "attribute", "a:1", "a:2", "两值")
    assert validate_conflict(record) == ()
    assert record["status"] == "open" and record["decision"] is None


def test_conflict_record_closed_set_violations():
    bad = {"conflict_id": "c:x", "book_id": "zj", "kind": "vibe", "left": "", "right": None,
           "detail": "x", "status": "maybe", "decision": "coin"}
    codes = [v.code for v in validate_conflict(bad)]
    assert "conflict_kind_unknown" in codes
    assert "conflict_status_unknown" in codes
    assert "conflict_decision_unknown" in codes
    assert "conflict_endpoint_missing" in codes


def test_conflict_kind_and_decision_contracts():
    assert CONFLICT_KINDS == ("attribute", "relation", "timeline")
    assert "superseded" in DECISIONS
    with pytest.raises(ValueError):
        make_conflict_record("zj", "vibe", "a:1", "a:2", "x")


def test_resolved_conflict_needs_resolved_by():
    record = make_conflict_record("zj", "timeline", "ev:a", "ev:b", "环")
    record["status"] = "resolved"
    assert "resolved_without_resolved_by" in [v.code for v in validate_conflict(record)]


# ---------------------------------------------------------------------------
# confidence / tier / verification (I19, §5.15)
# ---------------------------------------------------------------------------
def test_confidence_requires_method():
    codes = [v.code for v in validate_confidence({"confidence": 0.9})]
    assert codes == ["confidence_method_missing"]


def test_confidence_zero_without_method_is_allowed():
    assert validate_confidence({"confidence": 0}) == ()


def test_confidence_valid_methods():
    assert validate_confidence({"confidence": 0.9, "confidence_method": "evidence_span"}) == ()
    assert validate_confidence({"confidence": 0.9, "confidence_method": "engine_self"}) == ()


def test_calibrated_requires_version():
    codes = [v.code for v in validate_confidence(
        {"confidence": 0.9, "confidence_method": "calibrated"})]
    assert codes == ["calibration_version_missing"]
    assert validate_confidence({"confidence": 0.9, "confidence_method": "calibrated",
                                "calibration_version": "cal-2026-10"}) == ()


def test_confidence_rejects_unknown_method_and_out_of_range():
    assert [v.code for v in validate_confidence(
        {"confidence": 0.9, "confidence_method": "vibes"})] == ["confidence_method_unknown"]
    assert "confidence_out_of_range" in [v.code for v in validate_confidence(
        {"confidence": 1.7, "confidence_method": "engine_self"})]


def test_tier_verification_level_mapping():
    # §5.15: verification.level ∈ canonical/major/minor/none and equals the tier.
    # The strength names (cove_atomic/evidence_span/schema_only) describe what the
    # verifier DOES and live in VERIFICATION_STRENGTH; conflating the two made the
    # box writer drop the field.
    assert verification_level("canonical") == "canonical"
    assert verification_level("major") == "major"
    assert verification_level("minor") == "minor"
    assert verification_level("boss") is None
    assert inference_allowed("canonical") == "open"
    assert inference_allowed("major") == "motivation_chain"
    assert inference_allowed("minor") == "off"
    assert [v.code for v in validate_tier("boss")] == ["tier_unknown"]


def test_canonical_requires_verification():
    assert [v.code for v in validate_verification("canonical", None)] == ["verification_missing"]


def test_verification_level_mismatch_and_bad_verdict():
    codes = [v.code for v in validate_verification("major", {
        "level": "canonical", "verdict": "maybe", "verifier": "v"})]
    assert "verification_level_mismatch" in codes
    assert "verification_verdict_invalid" in codes


def test_verification_provenance_required():
    assert "verification_provenance_missing" in [
        v.code for v in validate_verification("minor", {"verdict": "none"})]


def test_canonical_requires_model_decoupling():
    same = [v.code for v in validate_verification(
        "canonical", {"level": "canonical", "verdict": "support", "verifier": "v",
                      "model": "m1"},
        extractor_model="m1", verifier_model="m1")]
    assert same == ["model_not_decoupled"]
    different = validate_verification(
        "canonical", {"level": "canonical", "verdict": "support", "verifier": "v",
                      "model": "m2"},
        extractor_model="m1", verifier_model="m2")
    assert different == ()


# ---------------------------------------------------------------------------
# claim promotion (I23 / §5.16)
# ---------------------------------------------------------------------------
def test_insufficient_evidence_cannot_be_promoted():
    codes = [v.code for v in validate_claim_promotion(
        claim_layer="statement", verdict="insufficient_evidence")]
    assert codes == ["insufficient_promoted"]


def test_refuted_and_conflicting_cannot_be_promoted():
    assert [v.code for v in validate_claim_promotion(
        claim_layer="statement", verdict="refuted")] == ["refuted_promoted"]
    assert [v.code for v in validate_claim_promotion(
        claim_layer="statement", verdict="conflicting")] == ["conflicting_promoted"]


def test_supported_claim_needs_two_units_and_no_against():
    codes = [v.code for v in validate_claim_promotion(
        claim_layer="statement", verdict="supported", source_units=["u:1"])]
    assert "corroboration_insufficient" in codes
    codes = [v.code for v in validate_claim_promotion(
        claim_layer="statement", verdict="supported", source_units=["u:1", "u:2"])]
    assert codes == []
    codes = [v.code for v in validate_claim_promotion(
        claim_layer="statement", verdict="supported", source_units=["u:1", "u:2"],
        against=1)]
    assert codes == ["against_present"]


def test_unknown_claim_verdict_is_rejected():
    assert [v.code for v in validate_claim_promotion(
        claim_layer="statement", verdict="true")] == ["claim_verdict_unknown"]


def test_non_fact_target_is_not_promotion():
    assert validate_claim_promotion(claim_layer="statement", verdict="insufficient_evidence",
                                    target_layer="statement") == ()
