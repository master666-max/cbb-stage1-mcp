"""Regression tests for the two real-run defects D1 and D2.

D1 (evidence dropped from events)
    ``EventFact`` / ``ItemEventFact`` / ``OrgEventFact`` had no ``evidence``
    field, so ``normalize.facts`` had nothing to anchor and hard-coded
    ``status="needs_evidence"`` for all three event construction sites.  In the
    real run every row of ``resolve/events.jsonl`` had empty evidence and the
    box (``box/records.normalize_event``) rejected all of them as
    ``evidence_not_found`` -> ``knowledge/events.jsonl`` was 0 rows.

D2 (related / peers smuggled into aliases)
    ``_add_entity(concept.name, "concept", concept.related, ...)`` and
    ``_add_entity(loc.name, "location", list(loc.peers or []), ...)`` wrote
    semantically-related *entity names* into ``Entity.aliases``.  Aliases are
    other spellings of the same entity, so this made e.g. ``贝塔心叶`` an alias
    of every concept that merely mentioned him.  ``resolve.alias.surface_hosts``
    then counted >3 hosts and ``fanout_clean`` deleted the *canonical* entity
    (the protagonist), taking 26 events' participants with it.
"""

from __future__ import annotations

from cbb_stage1.engines.ai_reader.schema import (
    ChapterFact,
    EventFact,
    ItemEventFact,
    OrgEventFact,
    build_extraction_schema,
)
from cbb_stage1.normalize import chapter_fact_to_engine_output
from cbb_stage1.resolve import fanout_clean

UNIT_ID = "u:testbook:v01:c001"

# ---------------------------------------------------------------------------
# D1: evidence on EventFact / ItemEventFact / OrgEventFact
# ---------------------------------------------------------------------------

D1_TEXT = (
    "贝塔心叶推开石门，低语者的祭司冷冷地看着他。\n"
    "贝塔心叶从祭坛上取下了那枚青玉简。\n"
    "贝塔心叶加入了灰烬教团。\n"
)

D1_FACT = ChapterFact.model_validate(
    {
        "chapter_id": 1,
        "novel_id": "testbook",
        "characters": [{"name": "贝塔心叶"}],
        "item_events": [
            {
                "item_name": "青玉简",
                "item_type": "物品",
                "action": "获得",
                "actor": "贝塔心叶",
                "recipient": None,
                "description": "祭坛上的青玉简",
                "evidence": "贝塔心叶从祭坛上取下了那枚青玉简。",
            }
        ],
        "org_events": [
            {
                "org_name": "灰烬教团",
                "org_type": "组织",
                "member": "贝塔心叶",
                "role": "祭司",
                "action": "加入",
                "description": "贝塔心叶加入灰烬教团",
                "org_relation": None,
                "evidence": "贝塔心叶加入了灰烬教团。",
            }
        ],
        "events": [
            {
                "summary": "贝塔心叶推开石门",
                "type": "其他",
                "importance": "medium",
                "participants": ["贝塔心叶"],
                "location": None,
                "evidence": "贝塔心叶推开石门，低语者的祭司冷冷地看着他。",
            }
        ],
    }
)


def _event_output():
    return chapter_fact_to_engine_output(
        D1_FACT, unit_id=UNIT_ID, text=D1_TEXT, book_id="testbook"
    )


def test_event_facts_keep_evidence_and_are_asserted() -> None:
    output = _event_output()

    assert [event.type for event in output.events] == ["物品", "组织", "其他"]
    for event in output.events:
        assert event.evidence, f"{event.type} event lost its evidence"
        assert event.status == "asserted", (
            f"{event.type} event status is {event.status!r} even though its "
            "evidence quote is verbatim in the unit text"
        )
        for hit in event.evidence:
            assert hit.unit_id == UNIT_ID
            assert D1_TEXT[hit.start : hit.end] == hit.quote

    assert "evidence anchored: 3/3 quotes verbatim" in " ".join(output.notes)


def test_event_fact_models_accept_and_keep_evidence() -> None:
    quote = "贝塔心叶加入了灰烬教团。"
    assert EventFact(summary="s", type="其他", evidence=quote).evidence == quote
    assert (
        ItemEventFact(item_name="i", item_type="物品", action="获得", evidence=quote).evidence
        == quote
    )
    assert (
        OrgEventFact(org_name="o", org_type="组织", action="加入", evidence=quote).evidence
        == quote
    )


def test_extraction_schema_exposes_evidence_for_all_event_classes() -> None:
    defs = build_extraction_schema()["$defs"]
    for name in ("EventFact", "ItemEventFact", "OrgEventFact"):
        props = defs[name]["properties"]
        assert "evidence" in props, f"{name} has no evidence property in the schema"
        assert props["evidence"].get("default") == ""

    # The event classes keep their existing shape: participants is still pushed
    # to be non-empty (the D1 fix must not relax the recall constraint).
    assert defs["EventFact"]["properties"]["participants"]["minItems"] == 1


# ---------------------------------------------------------------------------
# D2: concept.related / location.peers are not aliases
# ---------------------------------------------------------------------------

D2_TEXT = "贝塔心叶站在佩尔西奥涅的祭坛前，桑克雷・奥库尔塔在低语。"

D2_FACT = ChapterFact.model_validate(
    {
        "chapter_id": 1,
        "novel_id": "testbook",
        "characters": [{"name": "贝塔心叶"}],
        "locations": [
            {
                "name": "佩尔西奥涅",
                "type": "地点",
                "parent": "异世界",
                "peers": ["贝塔心叶", "桑克雷・奥库尔塔"],
                "description": "低语者祭坛所在",
                "role": "setting",
            }
        ],
        "new_concepts": [
            {
                "name": "低语者",
                "category": "种族",
                "definition": "低语者能与死者对话。",
                "related": ["贝塔心叶", "桑克雷・奥库尔塔"],
            },
            {
                "name": "异世界转生",
                "category": "修炼体系",
                "definition": "死后转生到另一个世界。",
                "related": ["佩尔西奥涅", "贝塔心叶"],
            },
            {
                "name": "次元之门",
                "category": "修炼体系",
                "definition": "连接两个世界的门。",
                "related": ["佩尔西奥涅", "贝塔心叶"],
            },
            {
                "name": "转生之门",
                "category": "修炼体系",
                "definition": "转生者通过的门。",
                "related": ["佩尔西奥涅", "贝塔心叶"],
            },
        ],
    }
)


def _concept_output():
    return chapter_fact_to_engine_output(
        D2_FACT, unit_id=UNIT_ID, text=D2_TEXT, book_id="testbook"
    )


def test_related_and_peers_are_not_written_into_entity_aliases() -> None:
    output = _concept_output()

    assert any(
        entity.canonical_name == "贝塔心叶" and entity.type == "person"
        for entity in output.entities
    )
    for entity in output.entities:
        assert "贝塔心叶" not in entity.aliases, (
            f"{entity.type}:{entity.canonical_name} claims 贝塔心叶 is an alias "
            f"(aliases={entity.aliases})"
        )
        assert "桑克雷・奥库尔塔" not in entity.aliases
        assert "佩尔西奥涅" not in entity.aliases


def test_related_and_peers_survive_in_assertion_roles() -> None:
    output = _concept_output()

    # 概念定义 maps onto the registered L1 predicate 性质 (主体/值); the
    # concept's related list survives in qualifiers rather than being dropped.
    concept_qualifiers = {
        assertion.roles["主体"]: assertion.qualifiers
        for assertion in output.assertions
        if assertion.predicate == "性质" and "主体" in assertion.roles
    }
    assert concept_qualifiers["低语者"]["related"] == ["贝塔心叶", "桑克雷・奥库尔塔"]
    assert concept_qualifiers["次元之门"]["related"] == ["佩尔西奥涅", "贝塔心叶"]

    # 地点属性 maps onto 位于 (关系库: 主体/值); parent/peers live in qualifiers.
    location = next(
        assertion
        for assertion in output.assertions
        if assertion.predicate == "位于" and assertion.roles.get("主体") == "佩尔西奥涅"
    )
    assert location.qualifiers["parent"] == "异世界"
    assert location.qualifiers["peers"] == ["贝塔心叶", "桑克雷・奥库尔塔"]


def test_protagonist_entity_survives_fanout_clean() -> None:
    output = _concept_output()

    # DEFAULT_FANOUT == 3: the real run had 5 hosts for 贝塔心叶.
    result = fanout_clean(output.entities)

    kept_names = {entity.canonical_name for entity in result.kept}
    assert "贝塔心叶" in kept_names, (
        "fanout_clean deleted the protagonist entity; generic surfaces were "
        f"{sorted(result.generic_names)}"
    )
    assert "贝塔心叶" not in result.dropped_entities
    assert not [
        finding for finding in result.findings if finding.surface == "贝塔心叶"
    ]
