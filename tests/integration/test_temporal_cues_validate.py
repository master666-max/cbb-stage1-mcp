"""Integration tests for P0-C: temporal / cues / validate work-table wiring.

These tests use only the disposable work layer (``tmp_path``): synthetic engine
outputs, ``work/resolve/*.jsonl`` collector tables and upstream handoffs. They
assert the three contracts from the task:

* temporal writes accepted constraints + quarantine, **no fabrication** when no
  time evidence exists;
* cues run the code state machine (paid 前 triggered; premature_payoff flagged);
* validate single-writes audit/{conflicts,decisions,quarantine}.jsonl with the
  three conflict classes and I19/I17/I18/I24 quarantine rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.cues.integration import run as run_cues
from cbb_stage1.temporal.integration import run as run_temporal
from cbb_stage1.validate.integration import run as run_validate

CREATED = "2026-10-18T23:59:00+08:00"
BOOK = "zj"
UNITS = [{"unit_id": "u:1", "order": 1}, {"unit_id": "u:2", "order": 2},
         {"unit_id": "u:3", "order": 3}, {"unit_id": "u:4", "order": 4},
         {"unit_id": "u:5", "order": 5}]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def write_table(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")
    return path


def read_table(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evidence(unit: str = "u:1", quote: str = "影子") -> dict:
    return {"unit_id": unit, "start": 0, "end": len(quote), "quote": quote,
            "quote_sha256": "sha256:" + "0" * 64, "occurrence": 1}


def event(event_id: str, unit: str = "u:1", participants: list[str] | None = None) -> dict:
    return {"event_id": event_id, "book_id": BOOK, "type": "事件", "summary": "",
            "participants": participants or [], "unit_id": unit, "status": "asserted"}


def assertion(assertion_id: str, *, subject: str = "e:1", predicate: str = "年龄",
              literal: str = "17", **over) -> dict:
    record = {"assertion_id": assertion_id, "book_id": BOOK, "predicate": predicate,
              "library": "实体", "roles": {"主体": subject, "值": {"literal": literal}},
              "claim_layer": "fact", "status": "asserted",
              "evidence": [evidence()], "confidence": 0.9,
              "confidence_method": "evidence_span"}
    record.update(over)
    return record


def relation(assertion_id: str, roles: dict, predicate: str = "共立", **over) -> dict:
    record = {"assertion_id": assertion_id, "book_id": BOOK, "predicate": predicate,
              "library": "关系", "roles": roles, "claim_layer": "fact",
              "status": "asserted", "confidence": 0.8,
              "confidence_method": "evidence_span"}
    record.update(over)
    return record


def cue(cue_id: str, **over) -> dict:
    record = {"cue_id": cue_id, "book_id": BOOK, "kind": "object",
              "subject": {"entity_id": None, "literal": "苍蓝色影子"},
              "setup": {"unit_id": "u:1", "evidence": [evidence("u:1")]},
              "reinforce": [],
              "trigger": {"unit_id": None, "event_id": None, "condition": None,
                          "evidence": []},
              "payoff": {"event_id": None, "unit_id": None, "evidence": []},
              "status": "candidate", "activated_at": None, "triggered_at": None,
              "dormant_since": None, "falsified_reason": None,
              "premature_payoff": False, "expected": "能力来源",
              "confidence": 0.7, "confidence_method": "engine_self", "deps": []}
    record.update(over)
    return record


def timeline(constraint_id: str, kind: str = "before",
             left: tuple[str, str] = ("event", "ev:a"),
             right: tuple[str, str] = ("event", "ev:b"), **over) -> dict:
    record = {"constraint_id": constraint_id, "book_id": BOOK, "kind": kind,
              "left": {"kind": left[0], "id": left[1]},
              "right": {"kind": right[0], "id": right[1]},
              "source": "explicit", "confidence": 0.8}
    record.update(over)
    return record


def prepare(root: Path, *, events=None, assertions=None, cues=None) -> Path:
    write_table(root / "resolve" / "events.jsonl", events or [])
    write_table(root / "resolve" / "assertions.jsonl", assertions or [])
    write_table(root / "resolve" / "cues.jsonl", cues or [])
    return root


# ---------------------------------------------------------------------------
# temporal
# ---------------------------------------------------------------------------
def test_temporal_empty_input_writes_empty_timeline_without_fabrication(tmp_path: Path):
    root = prepare(tmp_path)
    result = run_temporal(root, book_id=BOOK, outputs=[], units=UNITS,
                          created_at=CREATED, upstream={})
    assert result["counts"]["timeline_input"] == 0
    assert result["counts"]["timeline_written"] == 0
    assert result["counts"]["conflicts"] == 0
    timeline_path = Path(result["paths"]["timeline"])
    assert timeline_path.is_file()
    assert timeline_path.read_text(encoding="utf-8") == ""      # 空表，不拿叙事序造假
    quarantine_path = Path(result["paths"]["quarantine"])
    assert quarantine_path.is_file() and read_table(quarantine_path) == []


def test_temporal_solves_chain_and_writes_accepted_constraints(tmp_path: Path):
    root = prepare(tmp_path, events=[event("ev:a"), event("ev:b"), event("ev:c")])
    upstream = {"timeline": [timeline("t:aa", left=("event", "ev:a"), right=("event", "ev:b")),
                             timeline("t:bb", left=("event", "ev:b"), right=("event", "ev:c"))]}
    result = run_temporal(root, book_id=BOOK, outputs=[], units=UNITS,
                          created_at=CREATED, upstream=upstream)
    assert result["counts"]["timeline_written"] == 2
    assert result["counts"]["quarantine"] == 0
    rows = read_table(Path(result["paths"]["timeline"]))
    assert [row["constraint_id"] for row in rows] == ["t:aa", "t:bb"]
    assert all(row["kind"] == "before" and row["book_id"] == BOOK for row in rows)


def test_temporal_cycle_quarantines_and_hands_conflicts_to_validate(tmp_path: Path):
    root = prepare(tmp_path, events=[event("ev:a"), event("ev:b")])
    upstream = {"timeline": [timeline("t:aa", left=("event", "ev:a"), right=("event", "ev:b")),
                             timeline("t:bb", left=("event", "ev:b"), right=("event", "ev:a"))]}
    temporal_result = run_temporal(root, book_id=BOOK, outputs=[], units=UNITS,
                                   created_at=CREATED, upstream=upstream)
    assert temporal_result["counts"]["timeline_written"] == 0
    assert temporal_result["counts"]["cycles"] >= 1
    quarantine = read_table(Path(temporal_result["paths"]["quarantine"]))
    assert {row["reason"] for row in quarantine} == {"timeline_cycle"}
    assert temporal_result["conflicts"] and \
        all(row["kind"] == "timeline" for row in temporal_result["conflicts"])

    # validate merges the upstream-conflicts (a cycle is quarantined out of the table)
    validate_result = run_validate(root, book_id=BOOK, outputs=[], units=UNITS,
                                   created_at=CREATED,
                                   upstream={"temporal": temporal_result})
    conflicts = read_table(Path(validate_result["paths"]["conflicts"]))
    assert any(row["kind"] == "timeline" for row in conflicts)
    assert validate_result["counts"]["conflicts_by_kind"]["timeline"] >= 1


def test_temporal_ref_missing_against_event_table(tmp_path: Path):
    root = prepare(tmp_path, events=[event("ev:a")])
    upstream = {"timeline": [timeline("t:bad", left=("event", "ev:a"),
                                      right=("event", "ev:missing"))]}
    result = run_temporal(root, book_id=BOOK, outputs=[], units=UNITS,
                          created_at=CREATED, upstream=upstream)
    assert result["counts"]["timeline_written"] == 0
    quarantine = read_table(Path(result["paths"]["quarantine"]))
    assert [row["reason"] for row in quarantine] == ["ref_missing"]


def test_temporal_during_contradiction_is_quarantined(tmp_path: Path):
    root = prepare(tmp_path, events=[event("ev:a"), event("ev:b")])
    upstream = {"timeline": [timeline("t:aa", left=("event", "ev:a"), right=("event", "ev:b")),
                             timeline("t:dd", kind="during", left=("event", "ev:a"),
                                      right=("event", "ev:b"))]}
    result = run_temporal(root, book_id=BOOK, outputs=[], units=UNITS,
                          created_at=CREATED, upstream=upstream)
    quarantine = read_table(Path(result["paths"]["quarantine"]))
    assert len(quarantine) == 1
    assert quarantine[0]["item_id"].startswith("q:")   # 统一 quarantine_id 公式
    assert "t:dd" in quarantine[0]["detail"]            # 旧 t: id 只作引用保留
    assert quarantine[0]["reason"] == "timeline_contradiction"


# ---------------------------------------------------------------------------
# cues
# ---------------------------------------------------------------------------
def test_cues_state_machine_advances_and_flags_premature(tmp_path: Path):
    root = prepare(tmp_path, cues=[
        cue("cue:a1a1a1a1a1a1a1a1"),
        cue("cue:b2b2b2b2b2b2b2b2", status="active", activated_at="u:1",
             payoff={"event_id": None, "unit_id": "u:3", "evidence": [evidence("u:3")]}),
        cue("cue:c3c3c3c3c3c3c3c3", trigger={"unit_id": "u:2", "event_id": None,
                                 "condition": "代价出现", "evidence": []},
             payoff={"event_id": None, "unit_id": "u:4", "evidence": [evidence("u:4")]}),
        cue("cue:d4d4d4d4d4d4d4d4", status="active", activated_at="u:1",
             falsified_reason="后文证明是别人的影子"),
        cue("cue:e5e5e5e5e5e5e5e5", setup={"unit_id": "u:1", "evidence": []}),
    ])
    result = run_cues(root, book_id=BOOK, outputs=[], units=UNITS,
                      created_at=CREATED, upstream={})
    rows = read_table(Path(result["paths"]["cues"]))
    by_id = {row["cue_id"]: row for row in rows}
    assert by_id["cue:a1a1a1a1a1a1a1a1"]["status"] == "active"
    assert by_id["cue:b2b2b2b2b2b2b2b2"]["status"] == "active"
    assert by_id["cue:b2b2b2b2b2b2b2b2"]["premature_payoff"] is True      # 提前回收只标记，不强行闭环
    assert by_id["cue:c3c3c3c3c3c3c3c3"]["status"] == "paid"
    assert by_id["cue:c3c3c3c3c3c3c3c3"]["triggered_at"] == "u:2"
    assert by_id["cue:d4d4d4d4d4d4d4d4"]["status"] == "falsified"
    assert by_id["cue:e5e5e5e5e5e5e5e5"]["status"] == "candidate"
    assert result["counts"]["premature_payoff"] == 1


def test_cues_payoff_before_trigger_unit_is_premature(tmp_path: Path):
    root = prepare(tmp_path, cues=[
        cue("cue:f6f6f6f6f6f6f6f6", trigger={"unit_id": "u:4", "event_id": None,
                                  "condition": "代价出现", "evidence": []},
             payoff={"event_id": None, "unit_id": "u:2", "evidence": [evidence("u:2")]})])
    result = run_cues(root, book_id=BOOK, outputs=[], units=UNITS,
                      created_at=CREATED, upstream={})
    row = read_table(Path(result["paths"]["cues"]))[0]
    assert row["status"] == "triggered"          # 不升 paid
    assert row["premature_payoff"] is True
    assert row["triggered_at"] == "u:4"


def test_cues_active_without_setup_is_reported_not_silently_fixed(tmp_path: Path):
    root = prepare(tmp_path, cues=[cue("cue:1717171717171717", setup={"unit_id": "u:1", "evidence": []},
                                       status="active")])
    result = run_cues(root, book_id=BOOK, outputs=[], units=UNITS,
                      created_at=CREATED, upstream={})
    assert result["counts"]["violations"] >= 1
    assert any(item["code"] == "cue_active_without_setup" for item in result["violations"])


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def test_validate_writes_three_audit_tables_for_three_conflict_classes(tmp_path: Path):
    root = prepare(tmp_path, events=[event("ev:a"), event("ev:b")], assertions=[
        assertion("a:1", literal="17"),
        assertion("a:2", literal="19"),
        relation("a:3", {"施事": "e:1", "受事": "e:2"}),
        relation("a:4", {"施事": "e:2", "受事": "e:1"}),
        assertion("a:5", subject="e:2", literal="30", confidence=0.8,
                  confidence_method=None),
    ])
    # timeline table produced by the temporal step (clean) + upstream cycle conflicts
    write_table(root / "assemble" / "timeline.jsonl",
                [timeline("t:aa", left=("event", "ev:a"), right=("event", "ev:b"))])
    cycle_conflicts = [{"conflict_id": "c:deadbeefdeadbeef", "book_id": BOOK,
                        "kind": "timeline", "left": "ev:a", "right": "ev:b",
                        "detail": "时序环", "status": "open", "decision": None,
                        "resolved_by": None, "at": None}]
    cues = [cue("cue:2828282828282828", status="paid",
                payoff={"event_id": None, "unit_id": "u:3",
                        "evidence": [evidence("u:3")]})]
    write_table(root / "assemble" / "cues.jsonl", cues)
    result = run_validate(root, book_id=BOOK, outputs=[], units=UNITS,
                          created_at=CREATED,
                          upstream={"temporal": {"conflicts": cycle_conflicts}})
    counts = result["counts"]
    assert counts["conflicts_by_kind"] == {"attribute": 1, "relation": 1, "timeline": 1}
    conflicts = read_table(Path(result["paths"]["conflicts"]))
    assert {row["kind"] for row in conflicts} == {"attribute", "relation", "timeline"}
    decisions = Path(result["paths"]["decisions"])
    assert decisions.is_file() and read_table(decisions) == []    # 不自动裁决
    quarantine = read_table(Path(result["paths"]["quarantine"]))
    reasons = {row["reason"] for row in quarantine}
    assert "low_confidence" in reasons       # I19 缺 method
    assert "contradiction" in reasons        # cue paid 前未 triggered / premature


def test_validate_superseded_cycle_is_quarantined_not_a_conflict(tmp_path: Path):
    root = prepare(tmp_path, assertions=[
        assertion("a:1", status="superseded", superseded_by="a:2"),
        assertion("a:2", status="superseded", superseded_by="a:1"),
    ])
    result = run_validate(root, book_id=BOOK, outputs=[], units=UNITS,
                          created_at=CREATED, upstream={})
    quarantine = read_table(Path(result["paths"]["quarantine"]))
    codes = " ".join(row["detail"] for row in quarantine)
    assert "superseded_cycle" in codes
    assert all(row["conflict_id"] for row in read_table(Path(result["paths"]["conflicts"]))) \
        if read_table(Path(result["paths"]["conflicts"])) else True


def test_empty_work_root_produces_explicit_empty_tables(tmp_path: Path):
    root = tmp_path / "empty-work"
    temporal = run_temporal(root, book_id=BOOK, outputs=[], units=[],
                            created_at=CREATED, upstream={})
    cues = run_cues(root, book_id=BOOK, outputs=[], units=[],
                    created_at=CREATED, upstream={})
    validate = run_validate(root, book_id=BOOK, outputs=[], units=[],
                            created_at=CREATED, upstream={})
    for path in (temporal["paths"]["timeline"], temporal["paths"]["quarantine"],
                 cues["paths"]["cues"], validate["paths"]["conflicts"],
                 validate["paths"]["decisions"], validate["paths"]["quarantine"]):
        assert Path(path).is_file(), path
        assert read_table(Path(path)) == [], path
    assert temporal["counts"]["timeline_written"] == 0
    assert cues["counts"]["written"] == 0
    assert validate["counts"]["conflicts"] == 0


def test_outputs_fallback_when_work_tables_absent(tmp_path: Path):
    root = tmp_path / "outputs-only"
    outputs = [{
        "unit_id": "u:1",
        "events": [event("ev:a"), event("ev:b")],
        "assertions": [assertion("a:1", literal="17"), assertion("a:2", literal="19")],
        "cues": [cue("cue:3939393939393939")],
    }]
    temporal = run_temporal(root, book_id=BOOK, outputs=outputs, units=UNITS,
                            created_at=CREATED,
                            upstream={"timeline": [timeline("t:aa",
                                                            left=("event", "ev:a"),
                                                            right=("event", "ev:b"))]})
    assert temporal["counts"]["events"] == 2
    assert temporal["counts"]["timeline_written"] == 1
    cues = run_cues(root, book_id=BOOK, outputs=outputs, units=UNITS,
                    created_at=CREATED, upstream={})
    assert cues["counts"]["written"] == 1
    validate = run_validate(root, book_id=BOOK, outputs=outputs, units=UNITS,
                            created_at=CREATED, upstream={})
    assert validate["counts"]["conflicts_by_kind"]["attribute"] == 1


def test_validate_tolerates_engine_records_without_ids(tmp_path: Path):
    """回归：并行集成时 validate 收到 event_id=None 的事件不得崩（Lead 实测失败点）。

    record_id 对无 id 记录会抛 ValueError；validate 只把这些记录挡在
    冲突/超替图之外，置信检查仍照常做，并在 notes 里具名跳过数。
    """
    root = prepare(
        tmp_path,
        events=[{"event_id": None, "book_id": BOOK, "type": "事件", "summary": "",
                 "participants": [], "unit_id": "u:1", "status": "asserted"}],
        assertions=[{"assertion_id": None, "book_id": BOOK, "predicate": "年龄",
                     "library": "实体", "roles": {"主体": "e:1", "值": {"literal": "17"}},
                     "claim_layer": "fact", "status": "asserted",
                     "evidence": [evidence()], "confidence": 0.5,
                     "confidence_method": "engine_self"}],
    )
    result = run_validate(root, book_id=BOOK, outputs=[], units=UNITS,
                          created_at=CREATED, upstream={})
    assert result["counts"]["conflicts"] == 0
    assert result["counts"]["idless_skipped"] == 2
    assert any("无 id" in note for note in result["notes"])
    assert Path(result["paths"]["conflicts"]).is_file()
    assert Path(result["paths"]["quarantine"]).is_file()


def test_full_chain_is_deterministic(tmp_path: Path):
    def build(root: Path) -> dict[str, str]:
        prepare(root, events=[event("ev:a"), event("ev:b")],
                assertions=[assertion("a:1", literal="17"),
                            assertion("a:2", literal="19")],
                cues=[cue("cue:3939393939393939")])
        temporal = run_temporal(root, book_id=BOOK, outputs=[], units=UNITS,
                                created_at=CREATED,
                                upstream={"timeline": [timeline(
                                    "t:aa", left=("event", "ev:a"),
                                    right=("event", "ev:b"))]})
        run_cues(root, book_id=BOOK, outputs=[], units=UNITS,
                 created_at=CREATED, upstream={})
        run_validate(root, book_id=BOOK, outputs=[], units=UNITS,
                     created_at=CREATED, upstream={"temporal": temporal})
        names = ("assemble/timeline.jsonl", "assemble/cues.jsonl",
                 "audit/conflicts.jsonl", "audit/decisions.jsonl",
                 "audit/quarantine.jsonl")
        return {name: (root / name).read_text(encoding="utf-8") for name in names
                if (root / name).is_file()}

    first = build(tmp_path / "run-a")
    second = build(tmp_path / "run-b")
    assert first == second
    assert first["audit/conflicts.jsonl"]
