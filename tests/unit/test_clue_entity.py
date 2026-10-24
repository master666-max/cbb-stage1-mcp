"""A foreshadowing cue's subject must be an entity, not a bare string.

Measured gap (D12)
------------------
The lexicon has a `clue` entity type, and the cue carries `subject: {entity_id, literal}` — but
`normalize/facts.py` wrote `{"entity_id": None, "literal": ...}` unconditionally. Nothing ever
created a clue entity, so a foreshadowing subject could never get an id, aliases, a tier, mentions,
a merge, or a card: the first-class entity chain simply did not exist for it, and the cue fell back
to "literal only" every time.

The fix has two halves, and the second is what keeps it honest:

* when the subject names an entity that is already in this output (a character, location, item,
  concept), the cue points at **that** entity — no duplicate is created for the same name;
* otherwise a `clue` entity is registered for it, so the subject enters the entity chain.

Once `resolve` runs, entity ids come from `entity_id_for(book_id, type, canonical_name)`, so the
converter can compute the same id and the two agree.
"""

from __future__ import annotations

from cbb_stage1.engines.ai_reader.schema import ChapterFact
from cbb_stage1.normalize.facts import chapter_fact_to_engine_output
from cbb_stage1.resolve.integration import entity_id_for

UNIT = "u:tst:v01:c001"
TEXT = (
    "言万从怀里摸出一枚黑色令牌，令牌背面刻着一个‘柒’字。"
    "他抬头看了看残破的宗门石碑。"
)


def _chapter(cues: list[dict], characters: list[dict] | None = None) -> dict:
    return {
        "chapter_id": 1,
        "novel_id": "tst",
        "characters": characters or [],
        "relationships": [],
        "locations": [],
        "spatial_relationships": [],
        "item_events": [],
        "org_events": [],
        "events": [],
        "new_concepts": [],
        "world_declarations": [],
        "cues": cues,
    }


def _cue(subject: str, setup: str) -> dict:
    return {
        "kind": "object",
        "subject": subject,
        "setup": setup,
        "expected": "后面会再出现",
        "confidence": 0.7,
        "evidence": setup,
    }


def _convert(chapter: dict):
    fact = ChapterFact.model_validate(chapter)
    return fact, chapter_fact_to_engine_output(fact, unit_id=UNIT, text=TEXT, book_id="tst")


# ══ an unknown subject becomes a clue entity ═══════════════════════════
def test_an_unknown_cue_subject_becomes_a_clue_entity() -> None:
    _fact, output = _convert(_chapter([_cue("黑色令牌", "言万从怀里摸出一枚黑色令牌")]))

    clue_entities = [entity for entity in output.entities if entity.type == "clue"]
    assert clue_entities, (
        "no clue entity was created for the cue subject; D12's first-class chain (id/aliases/"
        "tier/mentions/merge/card) cannot start without one"
    )
    assert clue_entities[0].canonical_name == "黑色令牌"


def test_the_cue_points_at_the_clue_entity() -> None:
    _fact, output = _convert(_chapter([_cue("黑色令牌", "言万从怀里摸出一枚黑色令牌")]))
    subject = output.cues[0].subject
    assert subject.get("entity_id"), (
        f"the cue subject still has no entity_id: {subject!r}"
    )
    expected = entity_id_for("tst", "clue", "黑色令牌")
    assert subject["entity_id"] == expected, (
        f"{subject['entity_id']!r} != {expected!r}; the converter and resolve disagree on the id"
    )


def test_the_literal_is_kept_even_when_resolved() -> None:
    """The surface form is what a reader greps for; resolving must not erase it."""
    _fact, output = _convert(_chapter([_cue("黑色令牌", "言万从怀里摸出一枚黑色令牌")]))
    assert output.cues[0].subject.get("literal") == "黑色令牌"


# ══ a known subject reuses its entity, never duplicates it ═════════════
def test_a_subject_naming_an_existing_character_reuses_that_entity() -> None:
    chapter = _chapter(
        [_cue("言万", "言万从怀里摸出一枚黑色令牌")],
        characters=[
            {
                "name": "言万",
                "new_aliases": [],
                "appearance": "",
                "abilities_gained": [],
                "locations_in_chapter": [],
            }
        ],
    )
    _fact, output = _convert(chapter)

    person_ids = [
        entity.entity_id
        for entity in output.entities
        if entity.canonical_name == "言万" and entity.type == "person"
    ]
    assert person_ids, "fixture produced no person entity"
    assert not [entity for entity in output.entities if entity.type == "clue"], (
        "a clue entity was created for a name that is already an entity"
    )


def test_two_cues_with_the_same_subject_produce_one_clue_entity() -> None:
    chapter = _chapter(
        [
            _cue("黑色令牌", "言万从怀里摸出一枚黑色令牌"),
            _cue("黑色令牌", "令牌背面刻着一个‘柒’字"),
        ]
    )
    _fact, output = _convert(chapter)
    clue_entities = [entity for entity in output.entities if entity.type == "clue"]
    assert len(clue_entities) == 1, f"duplicated clue entities: {[e.canonical_name for e in clue_entities]}"


def test_cues_without_a_subject_match_stay_literal_only() -> None:
    """Guard: the cue still converts when nothing can be resolved."""
    _fact, output = _convert(_chapter([_cue("残破的宗门石碑", "他抬头看了看残破的宗门石碑")]))
    assert output.cues
    assert output.cues[0].subject.get("literal") == "残破的宗门石碑"
