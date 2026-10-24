"""ChapterFact schema (AI Reader nine-class records) + extraction JSON schema.

Ported from B06-AI-Reader-V2 `backend/src/models/chapter_fact.py` and
`models/entity_dict.py`.  The schema is kept deliberately loose on strings but
strict on structure: every record the LLM must return is a Pydantic model, and
``build_extraction_schema()`` re-adds ``minItems`` constraints that Pydantic
does not emit (the source patch behaviour).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


#: The verbatim-quote contract, stated where the model can read it.
#:
#: `evidence/make_evidence` requires `quote == text[start:end]` exactly: a
#: fuzzy match is NOT evidence, and an unanchored quote degrades the assertion to
#: `needs_evidence`. That rule lived only on the *consuming* side — the schema
#: merely said `{"type": "string"}` — so a model producing paraphrases or
#: spliced quotes had no way to know, and every such assertion was quarantined.
#: Measured on a real chapter: 40 assertions emitted, 0 into the box, all
#: `evidence_not_found`.
#: When the event happened, copied from the text.
#:
#: ``01-箱规格`` §5.7 requires events to carry ``time_expr`` so the temporal stage can
#: build a timeline. The extractor had no field for it, so every event reached the
#: box with ``time.expression = null`` and the timeline stayed empty — measured: 7/7
#: events hollow, ``timeline_generated = 0`` on material that contained 三年后.
#:
#: The value must be the text's own expression (``三年后`` / ``次日`` / ``当时``), not a
#: normalised date: the temporal stage re-reads the unit text and matches the
#: expression against it, so an invented date cannot be anchored.
TIME_DESCRIPTION = (
    "The time expression exactly as written in the text (e.g. 三年后, 次日, 当时)."
    " Leave empty when the text gives no explicit time. Do not invent or normalise"
    " a date."
)


#: Who is speaking in the quoted passage.
#:
#: ``01-箱规格`` §5.16 requires ``evidence[].speaker`` alongside ``mode``: a line of
#: dialogue or an inner monologue belongs to *someone*, and without an attribution
#: the assertion cannot honestly be ``claim_layer=statement`` — the assembler
#: quarantines a statement with no speaker (measured: every dialogue assertion in
#: a real chapter). The extractor therefore has to say who is talking, per quote,
#: because one assertion may carry several quotes from different speakers.
SPEAKER_DESCRIPTION = (
    "Name of the character who speaks or thinks this quoted passage, copied from"
    " the text exactly. Leave it empty for plain narration, and leave it empty"
    " rather than guessing when the text does not attribute the line."
)


EVIDENCE_DESCRIPTION = (
    "Verbatim quote copied from the chapter text. It MUST be a contiguous"
    " substring that appears character-for-character in the source: no"
    " paraphrase, no ellipsis, no joining two separate passages, no added"
    " quotation marks. Every record should carry one: 01-箱规格 §5.16 requires"
    " `asserted` records to have at least one exact evidence item, and a record"
    " without evidence is quarantined as `evidence_not_found`. Leave it empty"
    " ONLY when the chapter truly contains no quotable fragment for this record"
    " — never as a shortcut."
)


class AbilityGained(BaseModel):
    dimension: str  # "境界" / "技能" / "身份"
    name: str
    description: str = ""
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)



class CharacterFact(BaseModel):
    name: str
    new_aliases: list[str] = Field(default_factory=list)
    appearance: str | None = None
    abilities_gained: list[AbilityGained] = Field(default_factory=list)
    locations_in_chapter: list[str] = Field(default_factory=list)
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)



class RelationshipFact(BaseModel):
    person_a: str
    person_b: str
    relation_type: str
    is_new: bool = True
    previous_type: str | None = None
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)


class LocationFact(BaseModel):
    name: str
    type: str
    parent: str | None = None
    peers: list[str] | None = None  # same-level spatially adjacent/parallel entities
    description: str | None = None
    role: str | None = None  # "setting" | "referenced" | "boundary"
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)



class ItemEventFact(BaseModel):
    item_name: str
    item_type: str
    action: str  # 出现/获得/使用/赠予/消耗/丢失/损毁
    actor: str | None = None
    recipient: str | None = None
    description: str | None = None
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)


class OrgRelation(BaseModel):
    other_org: str
    type: str  # 盟友/敌对/从属/竞争


class OrgEventFact(BaseModel):
    org_name: str
    org_type: str
    member: str | None = None
    role: str | None = None
    action: str  # 加入/离开/晋升/阵亡/叛出/逐出
    description: str | None = None
    org_relation: OrgRelation | None = None
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)


class EventFact(BaseModel):
    summary: str
    type: str  # 战斗/成长/社交/旅行/其他
    importance: str = "medium"  # high/medium/low
    participants: list[str] = Field(default_factory=list)
    location: str | None = None
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)
    time_expr: str = Field(default="", description=TIME_DESCRIPTION)


class ConceptFact(BaseModel):
    name: str
    category: str  # 修炼体系/种族/货币/功法/...
    definition: str | None = ""
    related: list[str] = Field(default_factory=list)
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)



class SpatialRelationship(BaseModel):
    source: str
    target: str
    relation_type: str  # direction/distance/contains/adjacent/separated_by/terrain/in_between/travel_path/relative_scale/cluster
    value: str  # e.g. "north_of", "三天路程（步行）", "河流", "on_coast"
    confidence: str = "medium"  # high/medium/low
    narrative_evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)
    distance_class: str | None = None  # "near"/"medium"/"far"/"very_far"
    confidence_score: float | None = None  # 0.0-1.0, numeric confidence for solver
    waypoints: list[str] | None = None  # travel_path only: intermediate locations


class WorldDeclaration(BaseModel):
    declaration_type: str  # region_division / layer_exists / portal / region_position
    content: dict[str, Any]  # type-specific structured content
    narrative_evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)
    confidence: str = "medium"  # high / medium / low


class CueFact(BaseModel):
    """One foreshadowing cue as the extractor reports it.

    Deliberately thin: the extractor sees a setup and an expectation, not a state machine. `payoff`
    is filled only when the chapter already pays the cue off; the state machine in
    `cues/state.py` decides what happens to a cue over time.
    """

    kind: Literal["object", "line", "anomaly", "prophecy", "relationship"] = "object"
    subject: str = Field(min_length=1, description="伏笔的载体：物品/说法/异常/预言/关系的原文称呼")
    setup: str = Field(min_length=1, description="埋线处的原文逐字片段（必须能在本章原文中找到）")
    expected: str = Field(default="", description="预期后面如何回收")
    payoff: str = Field(default="", description="本章已回收则填回收处原文逐字片段；未回收留空")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = Field(default="", description=EVIDENCE_DESCRIPTION)
    evidence_speaker: str = Field(default="", description=SPEAKER_DESCRIPTION)


class ChapterFact(BaseModel):
    chapter_id: int
    novel_id: str
    characters: list[CharacterFact] = Field(default_factory=list)
    relationships: list[RelationshipFact] = Field(default_factory=list)
    locations: list[LocationFact] = Field(default_factory=list)
    spatial_relationships: list[SpatialRelationship] = Field(default_factory=list)
    item_events: list[ItemEventFact] = Field(default_factory=list)
    org_events: list[OrgEventFact] = Field(default_factory=list)
    events: list[EventFact] = Field(default_factory=list)
    new_concepts: list[ConceptFact] = Field(default_factory=list)
    world_declarations: list[WorldDeclaration] = Field(default_factory=list)
    #: Foreshadowing. Not in `_REQUIRED_NON_EMPTY_FIELDS`: a chapter may genuinely plant nothing,
    #: and demanding one would push the model to invent a promise the text never made.
    cues: list[CueFact] = Field(default_factory=list)


class EntityDictEntry(BaseModel):
    """One pre-scan candidate (= one entity_dictionary row)."""

    name: str
    entity_type: str = "unknown"
    frequency: int = 0
    confidence: str = "medium"  # low / medium / high
    aliases: list[str] = Field(default_factory=list)
    source: str  # freq / ngram / dialogue / title / naming / suffix / llm
    sample_context: str | None = None


# Arrays that must never come back empty: they are the recall-critical classes.
_REQUIRED_NON_EMPTY_FIELDS: tuple[str, ...] = (
    "characters",
    "relationships",
    "locations",
    "events",
)


def build_extraction_schema() -> dict[str, Any]:
    """Return the JSON schema sent to the LLM with ``minItems`` constraints.

    Pydantic emits ``"default": []`` and no ``minItems`` for list fields; the
    source extractor patched exactly that so the model is pushed to return
    non-empty characters/relationships/locations/events and event participants.
    """
    schema = ChapterFact.model_json_schema()
    defs = schema.get("$defs", {})

    event_fact = defs.get("EventFact", {})
    event_props = event_fact.get("properties", {})
    participants = event_props.get("participants")
    if isinstance(participants, dict):
        participants["minItems"] = 1
        participants.pop("default", None)
    location = event_props.get("location")
    if isinstance(location, dict):
        location.pop("default", None)

    root_props = schema.get("properties", {})
    for field in _REQUIRED_NON_EMPTY_FIELDS:
        prop = root_props.get(field)
        if isinstance(prop, dict):
            prop["minItems"] = 1
            prop.pop("default", None)

    return schema


EXTRACTION_SCHEMA: dict[str, Any] = build_extraction_schema()
