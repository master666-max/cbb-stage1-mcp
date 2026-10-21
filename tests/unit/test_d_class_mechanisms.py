"""D-class stages, driven with material that actually exercises them.

The main run used a 2015-char section that happened to contain no foreshadowing, no
contradiction and no time progression, so D7/D8/D9/D10 were recorded as "wired but
not triggered". That is not the same as "works", and the difference matters: an
untested branch is an unverified claim.

Each test below supplies the input the stage needs and asserts on real output. Two
of them document a requirement that is easy to miss and was missed while writing
them:

* a cue's identity comes from ``subject`` + ``setup.unit_id`` — using invented
  field names collapses distinct cues onto one id;
* a timeline constraint needs two events with explicit time mentions **in the same
  unit**, and the unit must carry ``text_path`` so the text can be read at all.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cbb_stage1.cues.integration import run as cues_run
from cbb_stage1.temporal.extract import extract_time_mentions
from cbb_stage1.temporal.integration import run as temporal_run
from cbb_stage1.validate.integration import run as validate_run


def _work(texts: dict[str, str], units: list[dict]) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cbb-dclass-"))
    (work / "plan" / "texts").mkdir(parents=True, exist_ok=True)
    (work / "sources").mkdir(parents=True, exist_ok=True)
    for unit_id, text in texts.items():
        name = unit_id.replace(":", "_") + ".txt"
        (work / "plan" / "texts" / name).write_text(text, encoding="utf-8")
        for unit in units:
            if unit["unit_id"] == unit_id:
                unit["text_path"] = f"plan/texts/{name}"
    (work / "sources" / "units.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n",
        encoding="utf-8",
    )
    return work


def _ev(unit: str, quote: str) -> dict:
    return {
        "unit_id": unit,
        "start": 0,
        "end": len(quote),
        "quote": quote,
        "occurrence": 1,
        "mode": "narration",
    }


# ── D7 / D8 ──────────────────────────────────────────────────────────────────
def test_cue_walks_setup_trigger_payoff_and_reaches_paid() -> None:
    units = [
        {"unit_id": "u:1", "order": 1, "volume": 1, "chapter": 1},
        {"unit_id": "u:2", "order": 2, "volume": 1, "chapter": 1},
        {"unit_id": "u:3", "order": 3, "volume": 1, "chapter": 2},
    ]
    work = _work({"u:1": "钥匙。", "u:2": "三年后。", "u:3": "他想起了钥匙。"}, units)
    cue = {
        "kind": "object",
        "subject": {"surface": "铜钥匙", "entity_id": None},
        "setup": {"unit_id": "u:1", "evidence": [_ev("u:1", "钥匙")]},
        "reinforce": [],
        "trigger": {"unit_id": "u:2", "evidence": [_ev("u:2", "三年后")]},
        "payoff": {"unit_id": "u:3", "evidence": [_ev("u:3", "钥匙")]},
        "expected": "钥匙在北塔派上用场",
        "confidence": 0.7,
        "confidence_method": "semantic_entropy",
        "deps": ["u:1", "u:2", "u:3"],
    }
    result = cues_run(
        work,
        book_id="d78",
        outputs=[{"cues": [cue]}],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {u["unit_id"]: u["order"] for u in units}},
    )
    rows = [
        json.loads(line)
        for line in (work / "assemble" / "cues.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["status"] == "paid", rows[0]
    assert not rows[0].get("premature_payoff")


def test_payoff_before_trigger_is_flagged_premature() -> None:
    units = [
        {"unit_id": "u:1", "order": 1, "volume": 1, "chapter": 1},
        {"unit_id": "u:3", "order": 3, "volume": 1, "chapter": 2},
    ]
    work = _work({"u:1": "去北塔。", "u:3": "北塔的门开着。"}, units)
    cue = {
        "kind": "place",
        "subject": {"surface": "北塔", "entity_id": None},
        "setup": {"unit_id": "u:3", "evidence": [_ev("u:3", "北塔")]},
        "reinforce": [],
        "trigger": None,
        "payoff": {"unit_id": "u:1", "evidence": [_ev("u:1", "北塔")]},
        "expected": "北塔更早出现",
        "confidence": 0.6,
        "confidence_method": "semantic_entropy",
        "deps": ["u:1", "u:3"],
    }
    result = cues_run(
        work,
        book_id="d78",
        outputs=[{"cues": [cue]}],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {u["unit_id"]: u["order"] for u in units}},
    )
    assert result["counts"]["premature_payoff"] == 1
    rows = [
        json.loads(line)
        for line in (work / "assemble" / "cues.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0].get("premature_payoff") is True


def test_two_cues_with_different_subjects_get_different_ids() -> None:
    """Identity is subject + setup unit; invented field names collapse them."""
    units = [
        {"unit_id": "u:1", "order": 1, "volume": 1, "chapter": 1},
        {"unit_id": "u:2", "order": 2, "volume": 1, "chapter": 1},
    ]
    work = _work({"u:1": "钥匙与北塔。", "u:2": "钥匙与北塔。"}, units)

    def cue(kind: str, surface: str, unit: str) -> dict:
        return {
            "kind": kind,
            "subject": {"surface": surface, "entity_id": None},
            "setup": {"unit_id": unit, "evidence": [_ev(unit, surface)]},
            "reinforce": [],
            "trigger": None,
            "payoff": {"unit_id": unit, "evidence": [_ev(unit, surface)]},
            "expected": "x",
            "confidence": 0.5,
            "confidence_method": "semantic_entropy",
            "deps": [unit],
        }

    cues_run(
        work,
        book_id="d78",
        outputs=[{"cues": [cue("object", "铜钥匙", "u:1"), cue("place", "北塔", "u:2")]}],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {u["unit_id"]: u["order"] for u in units}},
    )
    rows = [
        json.loads(line)
        for line in (work / "assemble" / "cues.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len({row["cue_id"] for row in rows}) == 2


# ── D10 ──────────────────────────────────────────────────────────────────────
def test_time_mentions_are_read_from_the_text() -> None:
    assert [m.text for m in extract_time_mentions("三年后，他回到城里。")] == ["三年后"]
    assert [m.text for m in extract_time_mentions("次日清晨。")] == ["次日"]
    assert extract_time_mentions("他走了。") == []


def test_two_events_with_times_in_one_unit_produce_a_constraint() -> None:
    """Same-unit only, and the unit needs text_path for the text to be readable."""
    text = "三年前他离开，三年后他回到城里。"
    units = [{"unit_id": "u:1", "order": 1, "volume": 1, "chapter": 1}]
    work = _work({"u:1": text}, units)
    events = {
        "events": [
            {
                "event_id": "ev:1",
                "summary": "三年前他离开",
                "unit_id": "u:1",
                "evidence": [_ev("u:1", "三年前他离开")],
            },
            {
                "event_id": "ev:2",
                "summary": "三年后他回到城里",
                "unit_id": "u:1",
                "evidence": [_ev("u:1", "三年后他回到城里")],
            },
        ]
    }
    result = temporal_run(
        work,
        book_id="d10",
        outputs=[events],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )
    counts = result["counts"]
    assert counts["timeline_generated"] == 1, counts
    assert counts["timeline_written"] == 1
    rows = [
        json.loads(line)
        for line in (work / "assemble" / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["kind"] == "before"
    assert rows[0]["deps"] == ["ev:1", "ev:2"]


def test_events_without_time_mentions_produce_no_constraint() -> None:
    """Conservative by design: no mention means no constraint, never a guess."""
    units = [{"unit_id": "u:1", "order": 1, "volume": 1, "chapter": 1}]
    work = _work({"u:1": "他走了。她来了。"}, units)
    events = {
        "events": [
            {"event_id": "ev:1", "summary": "他走了", "unit_id": "u:1",
             "evidence": [_ev("u:1", "他走了")]},
            {"event_id": "ev:2", "summary": "她来了", "unit_id": "u:1",
             "evidence": [_ev("u:1", "她来了")]},
        ]
    }
    result = temporal_run(
        work,
        book_id="d10",
        outputs=[events],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )
    assert result["counts"]["timeline_generated"] == 0


# ── D9 ───────────────────────────────────────────────────────────────────────
def test_conflicting_attribute_is_detected() -> None:
    units = [{"unit_id": "u:1", "order": 1, "volume": 1, "chapter": 1}]
    work = _work({"u:1": "老人还活着。后来他死了。"}, units)

    def assertion(aid: str, literal: str) -> dict:
        return {
            "assertion_id": aid,
            "predicate": "性质",
            "library": "设定",
            "roles": {"主体": "e:d9:person:aaa", "值": {"literal": literal}},
            "claim_layer": "fact",
            "status": "asserted",
            "confidence": 0.8,
            "confidence_method": "semantic_entropy",
            "confidence_scope": "assertion",
            "evidence": [_ev("u:1", "老人")],
        }

    result = validate_run(
        work,
        book_id="d9",
        outputs=[{"assertions": [assertion("a:1", "活着"), assertion("a:2", "已死")]}],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )
    assert result["counts"]["conflicts"] == 1
    assert result["counts"]["conflicts_by_kind"]["attribute"] == 1
