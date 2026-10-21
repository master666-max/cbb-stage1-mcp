"""Unit tests for :mod:`cbb_stage1.cues` (foreshadowing state machine).

Contract: design/01 §5.9/§5.18, invariants I18/I19/I24. Negative cases the Lead
asked for are explicit: **premature payoff** (payoff before ``triggered``) and
**missing guards** (falsified without reason, dormant without N, paid without
trigger/payoff evidence).
"""

from __future__ import annotations

import copy

import pytest

from cbb_stage1 import ids as contract_ids
from cbb_stage1.cues import (
    CUE_EVENTS, CueViolation, apply_event, can_transition, cue_id_of, transition_cue,
    validate_cue,
)


def evidence(unit="u:c1", quote="苍蓝色的影子"):
    return {"unit_id": unit, "start": 0, "end": len(quote), "quote": quote,
            "quote_sha256": "sha256:" + "0" * 64, "occurrence": 1}


def cue(**over):
    base = {
        "cue_id": "cue:0c63c5d8ea1c35b8", "book_id": "zj", "kind": "object",
        "subject": {"entity_id": None, "literal": "苍蓝色影子能力"},
        "setup": {"unit_id": "u:c1", "evidence": [evidence()]},
        "reinforce": [],
        "trigger": {"unit_id": None, "event_id": None, "condition": "能力代价出现",
                    "evidence": []},
        "payoff": {"event_id": None, "unit_id": None, "evidence": []},
        "status": "candidate", "activated_at": None, "triggered_at": None,
        "dormant_since": None, "falsified_reason": None, "premature_payoff": False,
        "expected": "能力来源/代价", "confidence": 0.7,
        "confidence_method": "engine_self", "deps": [],
    }
    base.update(over)
    return base


def active(**over):
    record = cue()
    record.update({"status": "active", "activated_at": "u:c1"})
    record.update(over)
    return record


def triggered(**over):
    record = cue()
    record.update({"status": "triggered", "activated_at": "u:c1", "triggered_at": "u:c2",
                   "trigger": {"unit_id": "u:c2", "event_id": "ev:1",
                               "condition": "代价出现",
                               "evidence": [evidence("u:c2")]}})
    record.update(over)
    return record


def paid(**over):
    record = triggered()
    record["status"] = "paid"
    record["payoff"] = {"event_id": "ev:9", "unit_id": "u:c9",
                        "evidence": [evidence("u:c9", "代价兑现")]}
    record.update(over)
    return record


# ---------------------------------------------------------------------------
# happy path: candidate -> active -> triggered -> paid
# ---------------------------------------------------------------------------
def test_confirmed_candidate_becomes_active():
    transition = apply_event(cue(), "confirm", unit_id="u:c1")
    assert transition.allowed is True
    assert transition.from_state == "candidate" and transition.to_state == "active"
    assert transition.cue["status"] == "active"
    assert transition.cue["activated_at"] == "u:c1"
    assert transition.violations == ()


def test_active_trigger_payoff_full_chain():
    t1 = apply_event(cue(), "confirm", unit_id="u:c1")
    t2 = apply_event(t1.cue, "trigger", unit_id="u:c2")
    assert t2.cue["status"] == "triggered" and t2.cue["triggered_at"] == "u:c2"
    t3 = apply_event(t2.cue, "payoff", unit_id="u:c9",
                     evidence=[evidence("u:c9", "代价兑现")])
    assert t3.allowed is True and t3.cue["status"] == "paid"
    assert t3.premature_payoff is False


def test_triggered_cue_accepts_payoff():
    transition = apply_event(triggered(), "payoff", unit_id="u:c9",
                             evidence=[evidence("u:c9")])
    assert transition.allowed is True
    assert transition.cue["status"] == "paid"


# ---------------------------------------------------------------------------
# guards: active needs setup evidence (I18)
# ---------------------------------------------------------------------------
def test_confirm_without_setup_evidence_is_rejected():
    record = cue(setup={"unit_id": "u:c1", "evidence": []})
    transition = apply_event(record, "confirm")
    assert transition.allowed is False
    assert transition.cue["status"] == "candidate"
    assert [v.code for v in transition.violations] == ["cue_active_without_setup"]


def test_confirm_from_non_candidate_is_rejected():
    transition = apply_event(active(), "confirm")
    assert transition.allowed is False
    assert transition.violations[0].code == "cue_transition_not_allowed"


# ---------------------------------------------------------------------------
# trigger guards
# ---------------------------------------------------------------------------
def test_trigger_needs_condition_or_evidence():
    record = active(trigger={"unit_id": None, "event_id": None, "condition": "",
                             "evidence": []})
    transition = apply_event(record, "trigger")
    assert transition.allowed is False
    assert [v.code for v in transition.violations] == ["cue_trigger_incomplete"]
    assert transition.cue["status"] == "active"


def test_trigger_accepts_evidence_without_condition():
    record = active(trigger={"unit_id": "u:c2", "event_id": None, "condition": None,
                             "evidence": [evidence("u:c2")]})
    transition = apply_event(record, "trigger")
    assert transition.allowed is True and transition.cue["status"] == "triggered"


def test_trigger_from_candidate_is_rejected():
    transition = apply_event(cue(), "trigger")
    assert transition.allowed is False
    assert transition.violations[0].code == "cue_transition_not_allowed"


# ---------------------------------------------------------------------------
# payoff guards: paid 前必须 triggered + premature_payoff (I18/I24)
# ---------------------------------------------------------------------------
def test_payoff_from_active_is_premature_and_not_paid():
    record = active()
    transition = apply_event(record, "payoff", unit_id="u:c9",
                             evidence=[evidence("u:c9")])
    assert transition.allowed is False
    assert transition.cue["status"] == "active"          # 不强行闭环
    assert transition.premature_payoff is True
    assert transition.cue["premature_payoff"] is True
    codes = [v.code for v in transition.violations]
    assert "cue_paid_without_trigger" in codes
    assert "cue_premature_payoff" in codes


def test_payoff_from_dormant_is_premature():
    record = active(status="dormant", dormant_since="u:c9")
    transition = apply_event(record, "payoff", evidence=[evidence("u:c9")])
    assert transition.allowed is False
    assert transition.premature_payoff is True
    assert transition.cue["status"] == "dormant"


def test_payoff_before_trigger_in_unit_order_is_premature_even_if_triggered():
    record = triggered()
    order = {"u:c1": 1, "u:c2": 5, "u:c3": 3}
    transition = apply_event(record, "payoff", unit_id="u:c3", unit_order=order,
                             evidence=[evidence("u:c3")])
    assert transition.allowed is False
    assert transition.cue["status"] == "triggered"
    assert transition.premature_payoff is True
    assert [v.code for v in transition.violations] == ["cue_premature_payoff"]


def test_payoff_needs_payoff_evidence():
    transition = apply_event(triggered(), "payoff")
    assert transition.allowed is False
    assert [v.code for v in transition.violations] == ["cue_paid_without_payoff_evidence"]


def test_premature_quarantine_records_are_box_shaped():
    transition = apply_event(active(), "payoff", evidence=[evidence("u:c9")])
    records = transition.quarantine_records()
    assert records
    record = records[0]
    assert record["kind"] == "cue" and record["reason"] == "contradiction"
    assert record["item_id"] == "cue:0c63c5d8ea1c35b8"
    assert set(record) == {"item_id", "kind", "reason", "detail", "source_run"}


# ---------------------------------------------------------------------------
# dormant / reinforce / falsify / close
# ---------------------------------------------------------------------------
def test_active_goes_dormant_after_n_units():
    transition = apply_event(active(), "dormant", unit_id="u:c9",
                             unit_order={"u:c1": 1, "u:c9": 9}, dormant_after_units=3)
    assert transition.allowed is True
    assert transition.cue["status"] == "dormant"
    assert transition.cue["dormant_since"] == "u:c9"


def test_active_stays_when_not_long_enough():
    transition = apply_event(active(), "dormant", unit_id="u:c2",
                             unit_order={"u:c1": 1, "u:c2": 2}, dormant_after_units=3)
    assert transition.allowed is True
    assert transition.cue["status"] == "active"


def test_dormant_requires_n_from_config():
    transition = apply_event(active(), "dormant", unit_id="u:c9",
                             unit_order={"u:c1": 1, "u:c9": 9})
    assert transition.allowed is False
    assert [v.code for v in transition.violations] == ["cue_dormant_threshold_missing"]


def test_dormant_requires_unit_order():
    transition = apply_event(active(), "dormant", dormant_after_units=2)
    assert transition.allowed is False
    assert [v.code for v in transition.violations] == ["cue_unit_order_missing"]


def test_dormant_to_active_needs_reinforce_evidence():
    record = active(status="dormant", dormant_since="u:c9")
    transition = apply_event(record, "reinforce", evidence=[evidence("u:c9")])
    assert transition.allowed is True
    assert transition.cue["status"] == "active"
    assert transition.cue["dormant_since"] is None
    assert len(transition.cue["reinforce"]) == 1


def test_dormant_reinforce_without_evidence_is_rejected():
    record = active(status="dormant", dormant_since="u:c9")
    transition = apply_event(record, "reinforce")
    assert transition.allowed is False
    assert transition.violations[0].code == "cue_transition_not_allowed"


def test_falsify_requires_reason():
    transition = apply_event(active(), "falsify")
    assert transition.allowed is False
    assert [v.code for v in transition.violations] == ["cue_falsified_without_reason"]
    ok = apply_event(active(), "falsify", reason="后文证明是另一个人的影子")
    assert ok.allowed is True
    assert ok.cue["status"] == "falsified"
    assert ok.cue["falsified_reason"] == "后文证明是另一个人的影子"


def test_close_marks_unresolved_and_keeps_paid_idempotent():
    transition = apply_event(active(), "close")
    assert transition.cue["status"] == "unresolved"
    already = apply_event(paid(), "close")
    assert already.allowed is True and already.cue["status"] == "paid"


def test_dormant_event_from_triggered_is_rejected():
    transition = apply_event(triggered(), "dormant", unit_id="u:c9",
                             unit_order={"u:c2": 2, "u:c9": 9}, dormant_after_units=1)
    assert transition.allowed is False
    assert transition.violations[0].code == "cue_transition_not_allowed"


def test_unknown_event_raises():
    with pytest.raises(ValueError):
        apply_event(cue(), "teleport")
    assert "teleport" not in CUE_EVENTS


def test_can_transition_shortcut():
    assert can_transition(cue(), "confirm") is True
    assert can_transition(cue(), "payoff", evidence=[evidence()]) is False


# ---------------------------------------------------------------------------
# purity / ids
# ---------------------------------------------------------------------------
def test_transition_is_pure_and_copies():
    record = cue()
    snapshot = copy.deepcopy(record)
    transition = apply_event(record, "confirm")
    assert record == snapshot
    transition.cue["status"] = "tampered"
    assert record["status"] == "candidate"


def test_cue_id_is_derived_and_stable_when_missing():
    record = cue()
    record.pop("cue_id")
    first = cue_id_of(record)
    assert first.startswith("cue:") and first == cue_id_of(copy.deepcopy(record))
    # 缺号用统一 cue: 公式（cbb_stage1.ids.cue_id）
    assert first == contract_ids.cue_id(
        record["book_id"], record["kind"], record["subject"], record["setup"]["unit_id"])
    assert transition_cue(record, "confirm")["cue_id"] == first


def test_existing_valid_cue_id_is_preserved_and_invalid_is_rebuilt():
    assert cue_id_of(cue()) == "cue:0c63c5d8ea1c35b8"     # 旧合法 id 原样保留
    broken = cue(cue_id="cue:not-a-hex-id")
    rebuilt = cue_id_of(broken)
    assert rebuilt != broken["cue_id"] and rebuilt.startswith("cue:")
    assert rebuilt == contract_ids.cue_id(
        broken["book_id"], broken["kind"], broken["subject"], broken["setup"]["unit_id"])


# ---------------------------------------------------------------------------
# validate_cue (read-only invariants)
# ---------------------------------------------------------------------------
def test_validate_valid_active_cue():
    assert validate_cue(active()) == ()


def test_validate_valid_paid_cue():
    assert validate_cue(paid()) == ()


def test_validate_paid_without_triggered_at():
    record = paid()
    record["triggered_at"] = None
    codes = [v.code for v in validate_cue(record)]
    assert "cue_paid_without_trigger" in codes


def test_validate_paid_without_payoff_evidence():
    record = paid()
    record["payoff"] = {"event_id": None, "unit_id": None, "evidence": []}
    codes = [v.code for v in validate_cue(record)]
    assert "cue_paid_without_payoff_evidence" in codes


def test_validate_falsified_without_reason():
    record = active(status="falsified")
    codes = [v.code for v in validate_cue(record)]
    assert "cue_falsified_without_reason" in codes


def test_validate_dormant_without_since():
    record = active(status="dormant")
    codes = [v.code for v in validate_cue(record)]
    assert "cue_dormant_without_since" in codes


def test_validate_premature_flag_missing():
    record = active()
    record["payoff"] = {"event_id": None, "unit_id": "u:c9", "evidence": [evidence("u:c9")]}
    codes = [v.code for v in validate_cue(record)]
    assert "cue_premature_flag_missing" in codes


def test_validate_premature_payoff_with_trigger_correctly_flagged():
    record = paid(status="active")
    record["triggered_at"] = None
    record["premature_payoff"] = True
    assert [v for v in validate_cue(record) if v.code == "cue_premature_payoff"] == []


def test_validate_confidence_method_violations():
    record = active(confidence=0.8, confidence_method=None)
    assert "confidence_method_missing" in [v.code for v in validate_cue(record)]
    record = active(confidence=0.8, confidence_method="calibrated")
    assert "calibration_version_missing" in [v.code for v in validate_cue(record)]
    record = active(confidence=0.8, confidence_method="made_up")
    assert "confidence_method_unknown" in [v.code for v in validate_cue(record)]


def test_validate_bad_state_and_kind():
    assert validate_cue(cue(status="maybe"))[0].code == "schema"
    assert validate_cue(cue(kind="vibe"))[0].code == "schema"


def test_violation_quarantine_reason_mapping():
    violation = CueViolation("cue_premature_payoff", "x")
    assert violation.quarantine_reason == "contradiction"
    assert CueViolation("cue_falsified_without_reason", "x").quarantine_reason == "schema"
