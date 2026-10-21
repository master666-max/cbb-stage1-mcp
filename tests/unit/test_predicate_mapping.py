"""Nine-class record names must map onto registered L1 predicates.

Measured before this module existed: ``normalize`` emitted 59 assertions across
12 predicate names, of which only 2 were registered — **57/59 = 96.6%
out-of-table**. The box quarantined them all and the knowledge layer stayed
empty.

The wrong fix is to register the 12 class names as predicates; ``00-架构总设计``
forbids exactly that ("谓词不是从数据里长出来的", "谓词注册表是唯一真源"). The right
fix is the mapping these tests pin:

* every emitted predicate is already in the registry;
* the extractor's own word is never lost (it moves to ``qualifiers``);
* a relation type with no honest mapping becomes a ``proposed_predicates`` row
  on disk, not a silent drop and not an auto-registration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cbb_stage1.assemble.integration import run as assemble_run
from cbb_stage1.box.lexicon import default_lexicon
from cbb_stage1.engines.ai_reader.schema import (
    AbilityGained,
    ChapterFact,
    CharacterFact,
    ConceptFact,
    EventFact,
    ItemEventFact,
    LocationFact,
    OrgEventFact,
    RelationshipFact,
    SpatialRelationship,
    WorldDeclaration,
)
from cbb_stage1.normalize import predicate_map as pmap
from cbb_stage1.normalize.facts import chapter_fact_to_engine_output

TEXT = (
    "贝塔心叶与拉斐尔是朋友。他穿着黑色夹克。神殿在太平洋之上。低语者是一种能力。"
)


def _full_fact() -> ChapterFact:
    """One fact touching every record class that produces an assertion."""
    return ChapterFact(
        chapter_id=1,
        novel_id="contract",
        characters=[
            CharacterFact(
                name="贝塔心叶",
                appearance="穿着黑色夹克",
                abilities_gained=[
                    AbilityGained(dimension="技能", name="低语者", description="读心")
                ],
                evidence="他穿着黑色夹克。",
            )
        ],
        relationships=[
            RelationshipFact(
                person_a="贝塔心叶",
                person_b="拉斐尔",
                relation_type="朋友",
                evidence="贝塔心叶与拉斐尔是朋友。",
            ),
            RelationshipFact(
                person_a="贝塔心叶",
                person_b="拉斐尔",
                relation_type="师徒",
                evidence="贝塔心叶与拉斐尔是朋友。",
            ),
        ],
        locations=[
            LocationFact(
                name="太平洋",
                type="海域",
                parent="异世界",
                evidence="神殿在太平洋之上。",
            )
        ],
        spatial_relationships=[
            SpatialRelationship(
                source="神殿",
                target="太平洋",
                relation_type="direction",
                value="north_of",
                narrative_evidence="神殿在太平洋之上。",
            )
        ],
        new_concepts=[
            ConceptFact(
                name="低语者",
                category="能力",
                definition="读心",
                evidence="低语者是一种能力。",
            )
        ],
        world_declarations=[
            WorldDeclaration(
                declaration_type="portal",
                content={"name": "门"},
                narrative_evidence="神殿在太平洋之上。",
            )
        ],
        item_events=[
            ItemEventFact(
                item_name="糖",
                item_type="食物",
                action="赠予",
                actor="贝塔心叶",
                evidence="他穿着黑色夹克。",
            )
        ],
        org_events=[
            OrgEventFact(
                org_name="黑手党",
                org_type="组织",
                member="贝塔心叶",
                action="逐出",
                evidence="他穿着黑色夹克。",
            )
        ],
        events=[
            EventFact(
                summary="两人交谈",
                type="社交",
                participants=["贝塔心叶"],
                evidence="贝塔心叶与拉斐尔是朋友。",
            )
        ],
    )


def _output():
    return chapter_fact_to_engine_output(
        _full_fact(), unit_id="u:contract:1", text=TEXT, book_id="contract"
    )


def test_no_out_of_table_predicate_is_emitted() -> None:
    """The whole point: 96.6% out-of-table must become 0%.

    ``朋友`` has no honest mapping, so it is expected to appear under its own
    name — but that is the *proposal* path, verified separately. Everything the
    mapper recognises must be registered.
    """
    lexicon = default_lexicon()
    output = _output()
    unmapped = {
        a.predicate
        for a in output.assertions
        if a.qualifiers.get("unmapped")
    }
    mapped = [a for a in output.assertions if not a.qualifiers.get("unmapped")]
    assert mapped, "no mapped assertions were produced; the fixture is stale"
    for assertion in mapped:
        assert lexicon.predicate(assertion.predicate) is not None, (
            f"{assertion.predicate!r} is not in the predicate registry"
        )
    # 朋友 and 世界观声明 both lack an honest L1 mapping:
    #   * 朋友      — a relation type the registry does not carry;
    #   * 世界观声明 — declarations have no subject entity, and every literal
    #                  predicate requires one, so forcing it would invent an
    #                  id that can never resolve.
    assert unmapped == {"朋友", "世界观声明"}, (
        f"unexpected unmapped predicates: {sorted(unmapped)}"
    )


def test_mapping_targets_are_all_registered() -> None:
    """The mapper may only target predicates the registry already knows."""
    lexicon = default_lexicon()
    for name in sorted(pmap.MAPPED_PREDICATES):
        assert lexicon.predicate(name) is not None, f"mapper targets unregistered {name!r}"


def test_two_party_predicates_use_registry_role_names() -> None:
    """Role names come from the registry, never from the extractor."""
    output = _output()
    for assertion in output.assertions:
        if assertion.predicate in pmap.TWO_PARTY_PREDICATES:
            assert set(assertion.roles) <= {"施事", "受事"}
    relation = next(
        a for a in output.assertions if a.predicate == "亲属"
    )
    assert relation.roles["施事"] == "贝塔心叶"
    assert relation.roles["受事"] == "拉斐尔"
    # The extractor's own word is preserved, not discarded.
    assert relation.qualifiers["relation_type"] == "师徒"


def test_extractor_wording_survives_in_qualifiers() -> None:
    """Mapping must not lose information the extractor provided."""
    output = _output()
    by_predicate: dict[str, list] = {}
    for assertion in output.assertions:
        by_predicate.setdefault(assertion.predicate, []).append(assertion)

    ability = by_predicate["性质"]
    assert any(a.qualifiers.get("ability_name") == "低语者" for a in ability)
    location = by_predicate["位于"]
    assert any(a.qualifiers.get("parent") == "异世界" for a in location)
    concept = by_predicate["性质"]
    assert any("category" in a.qualifiers for a in concept)
    event = by_predicate["参与"]
    assert any(a.qualifiers.get("item_name") == "糖" for a in event)


def test_unmapped_relation_reaches_proposed_predicates_on_disk() -> None:
    """I4: an unregistered predicate lives in proposed_predicates.jsonl.

    It used to be returned in memory only, so the box writer found an empty
    table and the proposal vanished without a trace (measured: proposals=1 in the
    return value, no file at all).
    """
    output = _output()
    work = Path(tempfile.mkdtemp(prefix="cbb-proposal-"))
    result = assemble_run(
        work,
        book_id="contract",
        outputs=[output.model_dump()],
        units=[{"unit_id": "u:contract:1"}],
        created_at="2026-01-01T00:00:00",
        upstream={},
    )
    assert result["counts"]["proposals"] >= 1

    proposal_path = work / "assemble" / "proposed_predicates.jsonl"
    assert proposal_path.exists(), "proposed_predicates.jsonl was not written"
    rows = [
        json.loads(line)
        for line in proposal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "proposed_predicates.jsonl is empty"
    predicates = {row["predicate"] for row in rows}
    assert "朋友" in predicates
    for row in rows:
        assert row["status"] == "proposed"
        assert row["decided_by"] is None  # never auto-registered


def test_absent_values_are_never_emitted_as_roles() -> None:
    """A null role is not a value; the box rejects it."""
    output = _output()
    for assertion in output.assertions:
        for role, value in (assertion.roles or {}).items():
            assert value is not None and value != [] and value != "", (
                f"{assertion.predicate}.{role} = {value!r}"
            )
