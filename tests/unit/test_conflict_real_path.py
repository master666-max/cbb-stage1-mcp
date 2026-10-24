"""The conflict detectors must see what the real normalize path actually writes.

Measured defect
---------------
`fact_validator._role` reads `assertion.roles` only. The real AI Reader path does not put those
values in `roles` — `normalize/predicate_map.py` writes:

    map_ability  -> roles: {主体: 角色, 值: 能力}
                    qualifiers: {dimension, ability_name, description}
    map_location -> roles: {主体: 地点, 值: 类型}
                    qualifiers: {parent, peers, description}

while the detectors ask for `entity`, `dimension`, `name`, `location`, `parent`. Three different
key names, none of which appear in either dict. So every one of the six input conflict checks
returns nothing in a real run: `input-conflicts.jsonl` is always empty and the audit table reports
zero conflicts no matter what the chapters say.

Measured before the fix: constructing 炼气 → 筑基 → 炼气 for one character in one dimension, plus a
2:2 parent split for one location, through `chapter_fact_to_engine_output`, `detect_input_conflicts`
returned **0**. Writing the same data in the unit-test shape (`roles={"entity": ..., "dimension":
...}`) returned 2.

What these tests pin
--------------------
The detectors must be reached **through the production path**, not through a hand-built
`Assertion`, because the hand-built shape is exactly the thing that hid the bug.
"""

from __future__ import annotations

from cbb_stage1.engines.ai_reader.schema import ChapterFact
from cbb_stage1.fact_validator import detect_input_conflicts
from cbb_stage1.models import Assertion, EngineOutput
from cbb_stage1.normalize.facts import chapter_fact_to_engine_output

UNIT = "u:tst:v01:c%03d"


def _chapter(unit_index: int, characters: list[dict], locations: list[dict] | None = None) -> dict:
    return {
        "chapter_id": unit_index,
        "novel_id": "tst",
        "characters": characters,
        "relationships": [],
        "locations": locations or [],
        "spatial_relationships": [],
        "item_events": [],
        "org_events": [],
        "events": [],
        "new_concepts": [],
        "world_declarations": [],
    }


def _character(name: str, abilities: list[tuple[str, str]]) -> dict:
    return {
        "name": name,
        "new_aliases": [],
        "appearance": "",
        "abilities_gained": [
            {"dimension": dimension, "name": ability, "description": ""}
            for dimension, ability in abilities
        ],
        "locations_in_chapter": [],
    }


def _location(name: str, parent: str) -> dict:
    return {
        "name": name,
        "type": "城市",
        "parent": parent,
        "peers": None,
        "description": "",
        "role": "setting",
    }


def _records(chapters: list[dict]) -> list[tuple[int, object]]:
    """Run the production converter, the way the pipeline does.

    The converter takes a validated `ChapterFact` and a text body, so the fixture goes through
    schema validation first — that is part of the production path and worth exercising.
    """
    records = []
    for index, chapter in enumerate(chapters, 1):
        fact = ChapterFact.model_validate(chapter)
        records.append(
            (
                index,
                chapter_fact_to_engine_output(
                    fact, unit_id=UNIT % index, text=f"第{index}章正文", book_id="tst"
                ),
            )
        )
    return records


# ══ the real path must reach the detectors ═════════════════════════════
def test_an_ability_regression_is_detected_through_the_real_path() -> None:
    """炼气 -> 筑基 -> 炼气 for one character in one dimension."""
    chapters = [
        _chapter(1, [_character("言万", [("境界", "炼气")])]),
        _chapter(2, [_character("言万", [("境界", "筑基")])]),
        _chapter(3, [_character("言万", [("境界", "炼气")])]),
    ]
    conflicts = detect_input_conflicts(_records(chapters))
    ability = [row for row in conflicts if row.subtype == "ability"]
    assert ability, (
        "the production path produced no ability conflict; the detectors cannot see the "
        f"qualifiers the normalizer writes. all subtypes seen: {[row.subtype for row in conflicts]}"
    )


def test_a_location_parent_split_is_detected_through_the_real_path() -> None:
    """One location with two different parents across units."""
    chapters = [
        _chapter(1, [_character("言万", [])], [_location("青牛镇", "元化国")]),
        _chapter(2, [_character("言万", [])], [_location("青牛镇", "元化国")]),
        _chapter(3, [_character("言万", [])], [_location("青牛镇", "南疆")]),
        _chapter(4, [_character("言万", [])], [_location("青牛镇", "南疆")]),
    ]
    conflicts = detect_input_conflicts(_records(chapters))
    location = [row for row in conflicts if row.subtype == "location"]
    assert location, (
        "the production path produced no location conflict; the detectors read `parent` from "
        f"roles while the normalizer writes it to qualifiers. seen: "
        f"{[row.subtype for row in conflicts]}"
    )


def test_no_conflict_is_reported_for_a_consistent_run() -> None:
    """The fix must not turn into a false-positive machine."""
    chapters = [
        _chapter(1, [_character("言万", [("境界", "炼气")])]),
        _chapter(2, [_character("言万", [("境界", "筑基")])]),
        _chapter(3, [_character("言万", [("境界", "金丹")])]),
    ]
    conflicts = detect_input_conflicts(_records(chapters))
    assert [row for row in conflicts if row.subtype == "ability"] == [], (
        "a monotonically increasing ability chain was reported as a regression"
    )


# ══ the legacy shape must keep working ═════════════════════════════════
def test_the_legacy_roles_shape_still_works() -> None:
    """Unit tests and older fixtures build Assertions with `roles={"entity": ...}`."""
    assertions = [
        Assertion(
            book_id="tst",
            predicate="性质",
            library="实体",
            roles={"entity": "言万", "dimension": "境界", "name": ability},
            qualifiers={},
            claim_layer="fact",
            evidence=[],
            confidence=0.9,
        )
        for ability in ("炼气", "筑基", "炼气")
    ]

    records = [
        (index, EngineOutput(unit_id=UNIT % index, assertions=[assertion]))
        for index, assertion in enumerate(assertions, 1)
    ]
    conflicts = detect_input_conflicts(records)
    assert [row for row in conflicts if row.subtype == "ability"], (
        "the legacy roles shape stopped being understood"
    )
