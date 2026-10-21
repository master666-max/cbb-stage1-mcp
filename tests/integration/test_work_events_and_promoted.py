"""Integration tests: work-layer events producer + promoted-facts regression.

Covers task P1-A:

* the assemble stage writes ``work/assemble/events.jsonl`` from engine outputs,
  preserving ``unit/type/summary/participants/evidence/time`` and using the
  **same ``ev:`` content key as** ``box/records.py`` (the box derives its own
  event rows from the engine output; the ids must match);
* a supported claim written by the claims stage is promoted into
  ``work/assemble/promoted_facts.jsonl`` and the box writer consumes it as a new
  ``fact`` assertion with ``derived_from`` pointing at the statement;
* an ``insufficient_evidence`` claim never promotes: no promoted row, no fact
  assertion in the box.

No engine/model calls; synthetic EngineOutput dicts + on-disk units.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cbb_stage1.assemble import run as assemble_run
from cbb_stage1.box.writer import build_box
from cbb_stage1.claims import run as claims_run
from cbb_stage1.claims.qa import make_answer, make_question
from cbb_stage1.models import EngineOutput, Event, Evidence, Assertion, Entity
from cbb_stage1.resolve import run as resolve_run

BOOK = "evt"
CREATED = "2026-10-19T01:00:00+08:00"
UNIT1 = "u:evt:v01:c001"
UNIT2 = "u:evt:v01:c002"
TEXT1 = "贝塔心叶与拉法对战。贝塔心叶说了一句话。"
TEXT2 = "拉法离开了房间。"
EVENT_SUMMARY = "贝塔心叶与拉法对战"


# ---------------------------------------------------------------------------
# synthetic work layer
# ---------------------------------------------------------------------------


def _units() -> list[dict]:
    return [
        {
            "unit_id": UNIT1,
            "book_id": BOOK,
            "volume": 1,
            "chapter": 1,
            "order": 1,
            "title": "第一章",
            "text_path": "plan/texts/v01/u001.txt",
            "narrator_reliability": "reliable",
            "narrator": {"kind": "third_person", "entity_id": None, "name": "叙述者"},
        },
        {
            "unit_id": UNIT2,
            "book_id": BOOK,
            "volume": 1,
            "chapter": 2,
            "order": 2,
            "title": "第二章",
            "text_path": "plan/texts/v01/u002.txt",
            "narrator_reliability": "reliable",
            "narrator": {"kind": "third_person", "entity_id": None, "name": "叙述者"},
        },
    ]


def _evidence(unit_id: str, quote: str, *, mode: str = "narration") -> Evidence:
    start = (TEXT1 if unit_id == UNIT1 else TEXT2).find(quote)
    assert start >= 0
    return Evidence(
        unit_id=unit_id,
        start=start,
        end=start + len(quote),
        quote=quote,
        occurrence=1,
        mode=mode,
    )


def _outputs(*, extra_events: list[dict] | None = None) -> list[dict]:
    speaker = {"kind": "narrator", "entity_id": None, "name": "叙述者"}
    output1 = EngineOutput(
        unit_id=UNIT1,
        entities=[
            Entity(book_id=BOOK, type="person", canonical_name="贝塔心叶", tier="canonical"),
            Entity(book_id=BOOK, type="person", canonical_name="拉法", tier="major"),
        ],
        events=[
            Event(
                book_id=BOOK,
                type="战斗",
                summary=EVENT_SUMMARY,
                participants=["贝塔心叶", "拉法"],
                unit_id=UNIT1,
                evidence=[_evidence(UNIT1, EVENT_SUMMARY)],
                time={"anchor_unit": UNIT1, "expression": "那天"},
                importance="high",
            )
        ],
        assertions=[
            Assertion(
                book_id=BOOK,
                predicate="性质",
                library="设定",
                roles={"主体": "贝塔心叶", "值": {"literal": "说过话"}},
                claim_layer="statement",
                speaker=speaker,  # type: ignore[arg-type]
                evidence=[_evidence(UNIT1, "贝塔心叶说了一句话")],
                confidence=0.8,
                confidence_method="evidence_span",
                status="asserted",
            )
        ],
    )
    output2 = EngineOutput(
        unit_id=UNIT2,
        entities=[
            Entity(book_id=BOOK, type="person", canonical_name="拉法", tier="major"),
        ],
        events=list(extra_events or []),
    )
    return [output1.model_dump(), output2.model_dump()]


def _write_work(work: Path, outputs: list[dict], units: list[dict]) -> None:
    (work / "plan" / "texts" / "v01").mkdir(parents=True, exist_ok=True)
    (work / "plan" / "texts" / "v01" / "u001.txt").write_text(TEXT1, encoding="utf-8")
    (work / "plan" / "texts" / "v01" / "u002.txt").write_text(TEXT2, encoding="utf-8")
    with (work / "plan" / "units.jsonl").open("w", encoding="utf-8") as fh:
        for unit in units:
            fh.write(json.dumps(unit, ensure_ascii=False) + "\n")
    (work / "engine").mkdir(parents=True, exist_ok=True)
    for index, output in enumerate(outputs, 1):
        (work / "engine" / f"u{index:03d}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _run_resolve_assemble(tmp_path: Path) -> tuple[Path, list[dict], list[dict], dict, dict]:
    work = tmp_path / "work"
    units = _units()
    outputs = _outputs()
    _write_work(work, outputs, units)
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    resolved = resolve_run(work, **kwargs)
    assembled = assemble_run(work, **kwargs)
    return work, units, outputs, resolved, assembled


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_ids(work: Path) -> dict[str, str]:
    return {
        row["canonical_name"]: row["entity_id"]
        for row in _read_jsonl(work / "resolve" / "entity_candidates.jsonl")
    }


# ---------------------------------------------------------------------------
# events producer
# ---------------------------------------------------------------------------


def test_assemble_writes_work_events_with_box_content_key(tmp_path):
    work, _units, _outputs_, _resolved, assembled = _run_resolve_assemble(tmp_path)
    events = _read_jsonl(work / "assemble" / "events.jsonl")

    assert assembled["paths"]["events"] == "assemble/events.jsonl"
    assert assembled["counts"]["events_input"] == 1
    assert assembled["counts"]["events"] == 1
    assert len(events) == 1
    row = events[0]

    candidate_ids = _candidate_ids(work)
    assert row["unit_id"] == UNIT1
    assert row["type"] == "战斗"
    assert row["summary"] == EVENT_SUMMARY
    assert row["participants"] == [candidate_ids["贝塔心叶"], candidate_ids["拉法"]]
    assert row["time"]["expression"] == "那天"
    assert row["importance"] == "high"
    assert row["evidence"][0]["quote"] == EVENT_SUMMARY
    assert row["evidence"][0]["mode"] == "narration"
    assert row["evidence"][0]["quote_sha256"].startswith("sha256:")

    # same content key as box/records.normalize_event
    key = "|".join(
        [
            BOOK,
            row["type"],
            row["unit_id"],
            ",".join(sorted(row["participants"])),
            row["summary"],
        ]
    )
    expected = "ev:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    assert row["event_id"] == expected


def test_box_event_id_matches_the_work_event_table(tmp_path):
    work, units, outputs, _resolved, _assembled = _run_resolve_assemble(tmp_path)
    # run the claims stage so the work dir is complete, then build the box
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    claims_run(work, **kwargs)
    box = tmp_path / "box"
    result = build_box(work, box)
    assert result.status == "built"

    box_events = _read_jsonl(box / "knowledge" / "events.jsonl")
    work_events = _read_jsonl(work / "assemble" / "events.jsonl")
    assert len(box_events) == 1
    assert {row["event_id"] for row in box_events} == {
        row["event_id"] for row in work_events
    }
    assert box_events[0]["participants"] == work_events[0]["participants"]


def test_events_table_is_byte_idempotent(tmp_path):
    work, units, outputs, _resolved, _assembled = _run_resolve_assemble(tmp_path)
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    before = (work / "assemble" / "events.jsonl").read_bytes()
    assemble_run(work, **kwargs)
    after = (work / "assemble" / "events.jsonl").read_bytes()
    assert before == after


def test_events_producer_drops_unresolvable_and_dedupes(tmp_path):
    work = tmp_path / "work"
    units = _units()
    duplicate = Event(
        book_id=BOOK,
        type="战斗",
        summary=EVENT_SUMMARY,
        participants=["贝塔心叶", "拉法"],
        unit_id=UNIT1,
        evidence=[_evidence(UNIT1, EVENT_SUMMARY)],
    ).model_dump()
    invalid = Event(
        book_id=BOOK,
        type="战斗",
        summary="无参与者的战斗",
        participants=["查无此人"],
        unit_id=UNIT1,
        evidence=[_evidence(UNIT1, EVENT_SUMMARY)],
    ).model_dump()
    outputs = _outputs(extra_events=[duplicate, invalid])
    _write_work(work, outputs, units)
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    resolve_run(work, **kwargs)
    assembled = assemble_run(work, **kwargs)

    events = _read_jsonl(work / "assemble" / "events.jsonl")
    assert len(events) == 1  # duplicate deduped; unresolvable participant dropped
    assert assembled["counts"]["event_duplicates"] == 1
    assert assembled["counts"]["event_quarantined"] == 1
    reasons = [
        item["reason"]
        for item in assembled["quarantine"]
        if item.get("kind") == "event"
    ]
    assert reasons == ["ref_missing"]


# ---------------------------------------------------------------------------
# promoted facts
# ---------------------------------------------------------------------------


def _statement_row(work: Path) -> dict:
    return next(
        row
        for row in _read_jsonl(work / "assemble" / "assertions.jsonl")
        if row["claim_layer"] == "statement"
    )


def _supported_qa(work: Path) -> dict:
    statement_id = _statement_row(work)["assertion_id"]
    answers = [
        make_answer(EVENT_SUMMARY, UNIT1, TEXT1.find(EVENT_SUMMARY),
                    TEXT1.find(EVENT_SUMMARY) + len(EVENT_SUMMARY), EVENT_SUMMARY),
        make_answer("拉法离开了房间", UNIT2, 0, len("拉法离开了房间"), "拉法离开了房间"),
    ]
    questions = [
        make_question("q1", "贝塔心叶与拉法发生了什么？", answers=[answers[0]], stance="support"),
        make_question("q2", "贝塔心叶与拉法发生了什么？", answers=[answers[1]], stance="support"),
    ]
    return {statement_id: [question.model_dump() for question in questions]}


def test_supported_claim_promotes_into_box_fact(tmp_path):
    work, units, outputs, _resolved, _assembled = _run_resolve_assemble(tmp_path)
    statement = _statement_row(work)
    statement_id = statement["assertion_id"]
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    claimed = claims_run(work, **{**kwargs, "upstream": {"claims_qa": _supported_qa(work)}})

    assert claimed["counts"]["claims"] == 1
    assert claimed["counts"]["promotions_allowed"] == 1
    promoted = _read_jsonl(work / "assemble" / "promoted_facts.jsonl")
    assert len(promoted) == 1
    fact = promoted[0]
    assert fact["claim_layer"] == "fact"
    assert fact["derived_from"] == [statement_id]
    assert fact["status"] == "asserted"
    fact_id = fact["assertion_id"]

    box = tmp_path / "box"
    result = build_box(work, box)
    assert result.status == "built"

    box_assertions = _read_jsonl(box / "knowledge" / "assertions.jsonl")
    box_fact = next(row for row in box_assertions if row["assertion_id"] == fact_id)
    assert box_fact["claim_layer"] == "fact"
    assert box_fact["derived_from"] == [statement_id]
    assert box_fact["predicate"] == "性质"

    box_claims = _read_jsonl(box / "knowledge" / "claims.jsonl")
    assert len(box_claims) == 1
    assert box_claims[0]["verdict"] == "supported"
    assert box_claims[0]["statement_assertion_id"] == statement_id


def test_insufficient_claim_never_promotes_into_box(tmp_path):
    work, units, outputs, _resolved, _assembled = _run_resolve_assemble(tmp_path)
    statement_id = _statement_row(work)["assertion_id"]
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    claimed = claims_run(work, **kwargs)

    assert claimed["counts"]["by_verdict"] == {"insufficient_evidence": 1}
    assert claimed["counts"]["promotions_allowed"] == 0
    assert claimed["counts"]["promoted_facts"] == 0
    assert claimed["promotions"][0]["reason"] == "insufficient_evidence_never_promotes"
    assert _read_jsonl(work / "assemble" / "promoted_facts.jsonl") == []

    box = tmp_path / "box"
    result = build_box(work, box)
    assert result.status == "built"

    box_assertions = _read_jsonl(box / "knowledge" / "assertions.jsonl")
    assert not [
        row
        for row in box_assertions
        if row["claim_layer"] == "fact" and row["derived_from"] == [statement_id]
    ]
    box_claims = _read_jsonl(box / "knowledge" / "claims.jsonl")
    assert box_claims and all(
        row["verdict"] == "insufficient_evidence" for row in box_claims
    )
