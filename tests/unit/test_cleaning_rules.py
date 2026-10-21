"""A7 cleaning-rule tests: 10 semantic cleaning classes + 6 conflict input checks.

Each cleaning class gets its own red/green test through
:func:`cbb_stage1.normalize.cleaning.clean_engine_output`; each conflict
detector gets a positive and (where meaningful) negative case through
:func:`cbb_stage1.fact_validator.detect_input_conflicts`.  All records are
synthetic ``EngineOutput`` instances - no engine/LLM is involved.
"""

from __future__ import annotations

import pytest

from cbb_stage1.fact_validator import (
    CLEANING_RULES,
    CONFLICT_KINDS,
    CONFLICT_SUBTYPES,
    QUARANTINE_REASONS,
    InputConflict,
    detect_input_conflicts,
    validate_engine_output,
)
from cbb_stage1.models import (
    Assertion,
    EngineOutput,
    Entity,
    Event,
    Evidence,
    Mention,
    Speaker,
)
from cbb_stage1.normalize.cleaning import clean_engine_output

BOOK = "testbook"
UNIT = "u:testbook:v01:c001"


def _output(
    *,
    unit_id: str = UNIT,
    mentions: list[Mention] | None = None,
    entities: list[Entity] | None = None,
    assertions: list[Assertion] | None = None,
    events: list[Event] | None = None,
) -> EngineOutput:
    return EngineOutput(
        unit_id=unit_id,
        mentions=list(mentions or []),
        entities=list(entities or []),
        assertions=list(assertions or []),
        events=list(events or []),
    )


def _assertion(
    predicate: str = "叔侄",
    *,
    library: str = "relation",
    roles: dict | None = None,
    claim_layer: str = "statement",
    evidence: list[Evidence] | None = None,
    speaker: Speaker | None = None,
    status: str = "asserted",
    confidence: float = 0.7,
    confidence_method: str | None = None,
) -> Assertion:
    return Assertion(
        book_id=BOOK,
        predicate=predicate,
        library=library,
        roles=roles if roles is not None else {"person_a": "韩立", "person_b": "三叔"},
        claim_layer=claim_layer,  # type: ignore[arg-type]
        speaker=speaker,
        evidence=list(evidence or []),
        confidence=confidence,
        confidence_method=confidence_method,
        status=status,
    )


def _entity(name: str, etype: str = "person", aliases: list[str] | None = None) -> Entity:
    return Entity(
        book_id=BOOK,
        type=etype,
        canonical_name=name,
        aliases=list(aliases or []),
        confidence=0.8,
    )


# ── 10 cleaning classes ─────────────────────────────────────────


def test_rule_schema_drops_records_with_empty_required_fields() -> None:
    output = _output(
        mentions=[
            Mention(unit_id=UNIT, surface=" ", start=0, end=1),
            Mention(unit_id=UNIT, surface="韩立", start=0, end=2),
        ],
        entities=[_entity(""), _entity("韩立")],
        assertions=[
            _assertion(predicate=" "),
            _assertion(roles={}),
            _assertion(),
        ],
        events=[
            Event(book_id=BOOK, type="旅行", summary=" ", unit_id=UNIT),
            Event(book_id=BOOK, type="旅行", summary="出发", unit_id=UNIT),
        ],
    )
    report = clean_engine_output(output)

    assert report.rules["schema"] >= 4
    assert [m.surface for m in report.output.mentions] == ["韩立"]
    assert [e.canonical_name for e in report.output.entities] == ["韩立"]
    assert len(report.output.assertions) == 1
    assert [e.summary for e in report.output.events] == ["出发"]


def test_rule_name_clamp_strips_and_clamps_names() -> None:
    output = _output(
        entities=[_entity(" 韩立、二愣子 ", "person", [" 二愣子 ", "  "])],
        mentions=[Mention(unit_id=UNIT, surface=" 韩立 ", start=0, end=4)],
        events=[
            Event(book_id=BOOK, type="旅行", summary="  出发  ", unit_id=UNIT)
        ],
        assertions=[_assertion(roles={"person_a": " 韩立 ", "person_b": " 三叔 "})],
    )
    report = clean_engine_output(output)

    assert report.rules["name_clamp"] >= 3
    entity = report.output.entities[0]
    assert entity.canonical_name == "韩立"
    assert entity.aliases == ["二愣子"]
    assert report.output.mentions[0].surface == "韩立"
    assert report.output.events[0].summary == "出发"
    assert report.output.assertions[0].roles["person_a"] == "韩立"


def test_rule_generic_person_drops_bare_titles_and_generics() -> None:
    output = _output(
        entities=[
            _entity("大汉"),
            _entity("老者"),
            _entity("堂主"),
            _entity("韩立"),
        ]
    )
    report = clean_engine_output(output)

    assert report.rules["generic_person"] == 3
    assert [e.canonical_name for e in report.output.entities] == ["韩立"]
    reasons = {item.reason for item in report.quarantine}
    assert reasons <= QUARANTINE_REASONS


def test_rule_generic_location_drops_generic_places() -> None:
    output = _output(
        entities=[
            _entity("门口", "location"),
            _entity("山上", "location"),
            _entity("青牛镇", "location"),
        ]
    )
    report = clean_engine_output(output)

    assert report.rules["generic_location"] == 2
    assert [e.canonical_name for e in report.output.entities] == ["青牛镇"]


def test_rule_alias_safety_removes_unsafe_aliases() -> None:
    output = _output(
        entities=[
            _entity(
                "韩立",
                aliases=[
                    "二愣子",
                    "大哥",
                    "老人",
                    "齐天大圣",
                    "这是一个很长的别名超过六",
                    "韩立",
                ],
            )
        ]
    )
    report = clean_engine_output(output)

    entity = report.output.entities[0]
    assert entity.aliases == ["二愣子", "齐天大圣"]
    assert report.rules["alias_safety"] >= 4


def test_rule_role_shape_marks_fact_without_evidence() -> None:
    """A fact with no evidence keeps its layer and carries status=needs_evidence.

    It must NOT be relabelled `statement`: §5.16 requires a speaker on a statement,
    and the speaker lives on the evidence, so relabelling guaranteed a
    `claim_layer_conflict` quarantine. Measured on a real run: 4 of 9 assertions were
    lost this way.
    """
    output = _output(assertions=[_assertion(claim_layer="fact", evidence=[])])

    report = clean_engine_output(output)

    assertion = report.output.assertions[0]
    assert assertion.claim_layer == "fact"
    assert assertion.status == "needs_evidence"
    assert any(item.reason == "evidence_not_found" for item in report.quarantine)

def test_rule_event_shape_normalises_participants_and_enums() -> None:
    output = _output(
        events=[
            Event(
                book_id=BOOK,
                type="",
                summary="出发",
                participants=["韩立", "韩立", " "],
                location="门口",
                importance="urgent",
                unit_id=UNIT,
            )
        ]
    )
    report = clean_engine_output(output)

    event = report.output.events[0]
    assert event.participants == ["韩立"]
    assert event.importance == "medium"
    assert event.type == "其他"
    assert event.location is None
    assert report.rules["event_shape"] >= 3


def test_rule_evidence_exact_verifies_span_against_unit_text() -> None:
    text = "韩立站在青牛镇外。"
    good = Evidence(unit_id=UNIT, start=0, end=2, quote="韩立", occurrence=1)
    bad = Evidence(unit_id=UNIT, start=1, end=3, quote="韩立", occurrence=1)
    output = _output(
        assertions=[
            _assertion(claim_layer="fact", evidence=[good, bad]),
        ]
    )
    report = clean_engine_output(output, unit_text=text)

    kept = report.output.assertions[0].evidence
    assert len(kept) == 1
    assert kept[0].quote == "韩立"
    assert kept[0].quote_sha256
    assert report.rules["evidence_exact"] >= 1
    assert any(item.reason == "evidence_not_found" for item in report.quarantine)


def test_rule_evidence_exact_keeps_fact_when_all_evidence_invalid() -> None:
    """Dropping invalid evidence makes the assertion evidence-less, not a statement."""
    text = "韩立站在青牛镇外。"
    bad = Evidence(unit_id=UNIT, start=0, end=2, quote="三叔", occurrence=1)
    output = _output(assertions=[_assertion(claim_layer="fact", evidence=[bad])])

    report = clean_engine_output(output, unit_text=text)

    assertion = report.output.assertions[0]
    assert assertion.evidence == []
    assert assertion.claim_layer == "fact"
    assert assertion.status == "needs_evidence"

def test_rule_confidence_scope_clamps_and_normalises() -> None:
    output = _output(
        entities=[_entity("韩立")],
        assertions=[
            _assertion(
                confidence=1.5,
                confidence_method=None,
                status="bogus",
                speaker=Speaker(kind="narrator"),
            ),
        ],
    )
    output.entities[0].confidence = -1.0
    report = clean_engine_output(output)

    assert report.output.entities[0].confidence == 0.0
    assertion = report.output.assertions[0]
    assert assertion.confidence == 1.0
    assert assertion.confidence_method == "engine_self_report"
    assert assertion.status == "quarantined"
    assert report.rules["confidence_scope"] >= 3


def test_rule_duplicates_keeps_first_occurrence() -> None:
    mention = Mention(unit_id=UNIT, surface="韩立", start=0, end=2)
    entity = _entity("韩立")
    assertion = _assertion(roles={"person_a": "韩立", "person_b": "三叔"})
    event = Event(book_id=BOOK, type="旅行", summary="出发", unit_id=UNIT)
    output = _output(
        mentions=[mention, mention.model_copy()],
        entities=[entity, entity.model_copy()],
        assertions=[assertion, assertion.model_copy()],
        events=[event, event.model_copy()],
    )
    report = clean_engine_output(output)

    assert len(report.output.mentions) == 1
    assert len(report.output.entities) == 1
    assert len(report.output.assertions) == 1
    assert len(report.output.events) == 1
    assert report.rules["duplicates"] == 4


def test_cleaning_report_reasons_and_rules_are_closed_sets() -> None:
    output = _output(entities=[_entity("大汉")])
    report = clean_engine_output(output)

    assert set(report.rules) <= set(CLEANING_RULES)
    assert {item.reason for item in report.quarantine} <= QUARANTINE_REASONS
    summary = report.summary()
    assert summary["unit_id"] == UNIT
    assert summary["changed"] is True


# ── 6 conflict input checks ─────────────────────────────────────


def _conflict_output(unit: int, **kwargs) -> tuple[int, EngineOutput]:
    return (unit, _output(unit_id=f"u:testbook:v01:c{unit:03d}", **kwargs))


def test_conflict_ability_regression() -> None:
    records = [
        _conflict_output(
            1,
            assertions=[
                _assertion(
                    "能力获得", library="setting",
                    roles={"entity": "韩立", "dimension": "境界", "name": "炼气"},
                )
            ],
        ),
        _conflict_output(
            2,
            assertions=[
                _assertion(
                    "能力获得", library="setting",
                    roles={"entity": "韩立", "dimension": "境界", "name": "筑基"},
                )
            ],
        ),
        _conflict_output(
            3,
            assertions=[
                _assertion(
                    "能力获得", library="setting",
                    roles={"entity": "韩立", "dimension": "境界", "name": "炼气"},
                )
            ],
        ),
    ]
    conflicts = detect_input_conflicts(records)

    assert [c.subtype for c in conflicts] == ["ability"]
    assert conflicts[0].kind == "attribute"
    assert conflicts[0].units == (1, 2, 3)


def test_conflict_relation_hostile_to_family() -> None:
    records = [
        _conflict_output(1, assertions=[_assertion("敌对")]),
        _conflict_output(2, assertions=[_assertion("亲属")]),
    ]
    conflicts = detect_input_conflicts(records)

    assert [c.subtype for c in conflicts] == ["relation"]
    assert conflicts[0].kind == "relation"
    assert "敌对→亲属" in conflicts[0].description


def test_conflict_relation_flip_flop() -> None:
    records = [
        _conflict_output(1, assertions=[_assertion("朋友")]),
        _conflict_output(2, assertions=[_assertion("敌对")]),
        _conflict_output(3, assertions=[_assertion("朋友")]),
    ]
    conflicts = detect_input_conflicts(records)
    assert any("反复" in conflict.description for conflict in conflicts)


def test_conflict_location_hierarchy_minority_requires_two_units() -> None:
    def location_output(unit: int, parent: str):
        return _conflict_output(
            unit,
            assertions=[
                _assertion(
                    "地点属性", library="setting",
                    roles={"entity": "青牛镇", "parent": parent},
                )
            ],
        )

    records = [
        location_output(1, "元化国"),
        location_output(2, "元化国"),
        location_output(3, "乱星海"),
        location_output(4, "乱星海"),
    ]
    conflicts = detect_input_conflicts(records)
    assert [c.subtype for c in conflicts] == ["location"]
    assert conflicts[0].kind == "attribute"

    # A one-unit minority is treated as extraction noise, not a conflict.
    noisy = [
        location_output(1, "元化国"),
        location_output(2, "元化国"),
        location_output(3, "乱星海"),
    ]
    assert detect_input_conflicts(noisy) == ()


def test_conflict_death_continuity() -> None:
    records = [
        _conflict_output(
            1,
            assertions=[
                _assertion(
                    "组织事件", library="event",
                    roles={"member": "三叔", "action": "阵亡"},
                )
            ],
        ),
        _conflict_output(3, entities=[_entity("三叔")]),
    ]
    conflicts = detect_input_conflicts(records)

    assert [c.subtype for c in conflicts] == ["death"]
    assert conflicts[0].kind == "timeline"
    assert conflicts[0].units == (1, 3)


def test_conflict_direction_contradiction() -> None:
    def direction_output(unit: int, value: str):
        return _conflict_output(
            unit,
            assertions=[
                _assertion(
                    "空间关系", library="setting",
                    roles={
                        "source": "七玄门",
                        "target": "青牛镇",
                        "relation_type": "direction",
                        "value": value,
                    },
                )
            ],
        )

    conflicts = detect_input_conflicts(
        [direction_output(1, "north_of"), direction_output(2, "south_of")]
    )
    assert [c.subtype for c in conflicts] == ["direction"]
    assert conflicts[0].kind == "relation"


def test_conflict_distance_gap() -> None:
    def distance_output(unit: int, distance_class: str):
        return _conflict_output(
            unit,
            assertions=[
                _assertion(
                    "空间关系", library="setting",
                    roles={
                        "source": "青牛镇",
                        "target": "七玄门",
                        "relation_type": "distance",
                        "distance_class": distance_class,
                    },
                )
            ],
        )

    conflicts = detect_input_conflicts(
        [distance_output(1, "near"), distance_output(2, "far")]
    )
    assert [c.subtype for c in conflicts] == ["distance"]
    assert conflicts[0].details["gap"] == 2

    # One ordinal step (near -> medium) is tolerated.
    assert detect_input_conflicts(
        [distance_output(1, "near"), distance_output(2, "medium")]
    ) == ()


def test_conflict_alias_map_merges_surfaces_before_compare() -> None:
    records = [
        _conflict_output(
            1,
            assertions=[
                _assertion("敌对", roles={"person_a": "二愣子", "person_b": "三叔"})
            ],
        ),
        _conflict_output(
            2,
            assertions=[
                _assertion("亲属", roles={"person_a": "韩立", "person_b": "三叔"})
            ],
        ),
    ]
    assert detect_input_conflicts(records) == ()
    conflicts = detect_input_conflicts(records, alias_map={"二愣子": "韩立"})
    assert [c.subtype for c in conflicts] == ["relation"]


def test_conflict_records_follow_v22_shape() -> None:
    records = [
        _conflict_output(1, assertions=[_assertion("敌对")]),
        _conflict_output(2, assertions=[_assertion("亲属")]),
    ]
    conflict = detect_input_conflicts(records)[0]
    assert conflict.kind in CONFLICT_KINDS
    assert conflict.subtype in CONFLICT_SUBTYPES
    record = conflict.to_record(BOOK)
    assert record["conflict_id"].startswith("c:")
    assert record["book_id"] == BOOK
    assert record["kind"] in CONFLICT_KINDS
    assert record["subtype"] == conflict.subtype
    assert record["status"] == "open"
    assert record["decision"] is None


def test_validate_engine_output_cleans_then_detects() -> None:
    records = [
        (
            1,
            _output(
                mentions=[Mention(unit_id=UNIT, surface=" ", start=0, end=1)],
                assertions=[_assertion("敌对")],
            ),
        ),
        _conflict_output(2, assertions=[_assertion("亲属")]),
    ]
    report = validate_engine_output(records)

    assert len(report.cleanings) == 2
    assert report.cleanings[0].changed is True
    assert report.quarantine
    assert [c.subtype for c in report.conflicts] == ["relation"]
    summary = report.summary()
    assert summary["units"] == 2
    assert summary["changed_units"] == 2  # both units get a role/schema clean
    assert summary["rules"]["schema"] >= 1


def test_input_conflict_rejects_invalid_kind_or_subtype() -> None:
    with pytest.raises(ValueError):
        InputConflict(kind="bogus", subtype="ability", severity="一般",
                      description="x", units=(), entity="e")
    with pytest.raises(ValueError):
        InputConflict(kind="attribute", subtype="bogus", severity="一般",
                      description="x", units=(), entity="e")
