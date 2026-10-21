"""ChapterFact -> EngineOutput v1 mapping (deterministic, no model calls).

Implements the adapter matrix row from design/02 §4.1:

    characters          -> entities(type=person) + mentions
    relationships       -> assertions(predicate = relation_type, library=relation)
    locations           -> entities(type=location) + setting assertions
    new_concepts        -> entities(type=concept) + setting assertions
    world_declarations  -> setting assertions
    spatial_relationships -> setting assertions
    item_events/org_events -> events + event assertions
    events              -> events

Evidence policy: an AI Reader `evidence` / `narrative_evidence` string is
anchored verbatim against the unit text; when it cannot be located the record
is emitted with ``status="needs_evidence"`` (never "asserted") and the quote is
not smuggled into an Evidence object.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.engines.ai_reader.alias_resolver import alias_safety_level
from cbb_stage1.engines.ai_reader.schema import ChapterFact
from cbb_stage1.evidence.anchor import find_occurrences, make_evidence
from cbb_stage1.normalize import predicate_map as pmap
from .tier import assign_tiers

from cbb_stage1.models import (
    Assertion,
    EngineOutput,
    Entity,
    Event,
    Evidence,
    Mention,
    Speaker,
)

_MAX_MENTIONS_PER_SURFACE = 20


def _clean_aliases(aliases: list[str], owner: str) -> list[str]:
    """Keep only aliases the safety table considers usable (level 2)."""
    out: list[str] = []
    for alias in aliases:
        alias = (alias or "").strip()
        if not alias or alias == owner:
            continue
        if alias_safety_level(alias) < 2:
            continue
        if alias not in out:
            out.append(alias)
    return out


#: Characters that open a quoted passage in this corpus.
_DIALOGUE_OPENERS = "「『"
#: Characters that open an inner monologue (parenthesised thought).
_INNER_OPENERS = "（("
#: Matching closer for each dialogue opener.
_DIALOGUE_CLOSERS = {"「": "」", "『": "』"}


def infer_mode(quote: str) -> str:
    """Infer `evidence.mode` from the quote's own shape.

    `01-箱规格` §5.16 requires `mode ∈ narration/dialogue/inner/memory/document/
    letter` and says an assertion whose evidence is `dialogue|inner` defaults to
    `claim_layer=statement`. The normalizer never set it, so every citation
    carried `mode=null`, `infer_claim_layer` could never downgrade anything,
    and the whole narrative layer (D1/D2) plus the claims path (B6/B7, which only
    consumes `statement`) stayed empty.

    The classification is mechanical — it reads the punctuation the author
    actually used, and never guesses at meaning:

    * starts with 「 or 『  -> `dialogue`
    * starts with （ or (  -> `inner`
    * otherwise            -> `narration`

    `memory`/`document`/`letter` are not inferable from punctuation alone,
    so they are never claimed here; a caller that knows better can pass `mode`
    explicitly.
    """
    stripped = (quote or "").lstrip()
    if not stripped:
        return "narration"
    if stripped[0] in _INNER_OPENERS:
        return "inner"
    if stripped[0] in _DIALOGUE_OPENERS:
        # A leading quote is necessary but not sufficient. A passage such as
        # 「Susurrador」，意思是「低语者」。 opens with a quote and is still the
        # narrator explaining a word — marking it `dialogue` would downgrade
        # the assertion to `statement` and then quarantine it for lacking a
        # speaker, destroying a valid fact. Measured on real text: that exact
        # sentence is the only misclassification the naive rule produced.
        #
        # A genuine line of dialogue closes its quote and is followed by either
        # nothing (a standalone line) or by attribution/punctuation — never by a
        # comma continuing the narrator's sentence.
        closer = _DIALOGUE_CLOSERS.get(stripped[0])
        if closer:
            close_at = stripped.find(closer, 1)
            if close_at > 0:
                tail = stripped[close_at + 1:].lstrip()
                if tail.startswith(("，", ",", "、")):
                    return "narration"
        return "dialogue"
    return "narration"


#: Attribution verbs that may introduce a line of dialogue.
_SPEECH_VERBS = (
    "说道", "说", "道", "问道", "答道", "喊道", "叫道", "低语", "喃喃",
    "笑了", "回答", "追问", "自语",
)


def infer_speaker(text: str, start: int, end: int, mode: str) -> Speaker | None:
    """Best-effort speaker for a quoted line, or `None`.

    §5.16 requires `claim_layer=statement` to carry a speaker, and the
    assembler quarantines a statement without one. Dialogue evidence therefore
    needs an attribution; without it the record must not claim to be a statement.

    The rule is deliberately conservative — it looks for a name immediately
    followed by a speech verb in the same paragraph, on either side of the quote:

        「……」拉法说着        -> 拉法
        贝塔叶低声说道：「……」  -> 贝塔叶

    If nothing matches it returns `None` rather than guessing, and the caller
    falls back to narration evidence. A wrong speaker is worse than no speaker:
    it fabricates who claimed something.
    """
    if mode not in ("dialogue", "inner") or not text:
        return None

    # Paragraph bounds around the quote.
    para_start = text.rfind("\n", 0, start) + 1
    para_end = text.find("\n", end)
    if para_end < 0:
        para_end = len(text)
    paragraph = text[para_start:para_end]

    # Prefer a name immediately BEFORE the quote: 名字+说/道：「……
    before = paragraph[: start - para_start]
    after = paragraph[end - para_start:]

    def _match(segment: str) -> str | None:
        # Scan from the end of the segment backwards so the closest name wins.
        for verb in _SPEECH_VERBS:
            index = segment.rfind(verb)
            if index < 0:
                continue
            head = segment[:index]
            # A name is 2-4 CJK characters immediately before the verb.
            for size in (4, 3, 2):
                candidate = head[-size:]
                if len(candidate) == size and all(
                    "\u4e00" <= ch <= "\u9fff" or ch == "・" for ch in candidate
                ):
                    return candidate
        return None

    name = _match(before) or _match(after)
    if not name:
        return None
    return Speaker(kind="entity", entity_id=None, name=name)
def _anchor(
    unit_id: str,
    text: str,
    quote: str | None,
    speaker_name: str | None = None,
) -> list[Evidence]:
    """Anchor a quote, attaching ``mode`` and (when known) ``speaker``.

    ``speaker_name`` comes from the extractor's ``evidence_speaker`` field. When it
    is empty, ``infer_speaker`` tries the surrounding attribution; when that also
    fails the evidence carries no speaker, which is the honest outcome — a
    fabricated speaker is worse than none, because it invents who claimed it.
    """
    if not quote:
        return []
    mode = infer_mode(quote)
    ev = make_evidence(unit_id, text, quote, mode=mode)  # type: ignore[arg-type]
    if ev is None:
        return []
    speaker = _speaker_from_name(speaker_name)
    if speaker is None:
        speaker = infer_speaker(text, ev.start, ev.end, mode)
    if speaker is not None:
        ev = ev.model_copy(update={"speaker": speaker})
    return [ev]


def _speaker_from_name(name: str | None) -> Speaker | None:
    """Build a Speaker from the extractor's attribution, or ``None``."""
    cleaned = str(name or "").strip()
    if not cleaned:
        return None
    return Speaker(kind="entity", entity_id=None, name=cleaned)


def _status(evidence: list[Evidence]) -> str:
    return "asserted" if evidence else "needs_evidence"


def _claim_layer(evidence: list[Evidence]) -> str:
    """The narrative layer of an assertion (`01-箱规格` §5.16).

    * `fact`      — 世界事实（叙述者旁白直陈，或多源/原文核实后升格）
    * `statement` — 某角色/叙述者**声称**的内容；**必须有 speaker**
    * `inferred`  — 推断，必须有 derived_from 或 premises evidence

    "No quote" is none of those; it is `status=needs_evidence`, which the status
    field already expresses.

    What *does* decide the layer is the evidence mode: §5.16 says an assertion
    whose evidence is `dialogue|inner` defaults to `statement`. But a
    `statement` needs a speaker, and this normalizer does not extract speakers,
    so emitting `statement` here would produce records the box rejects
    (measured: 6 of 9 assertions quarantined as `claim_layer_conflict`).
    The honest move is to keep `fact` and record the gap in `notes` — the
    assembler can still downgrade later once a speaker is available.
    """
    # The mode decides the layer when the evidence actually says who spoke.
    #
    # §5.16: an assertion whose evidence is ``dialogue|inner`` defaults to
    # ``statement``. That is only emittable when a speaker exists — otherwise the
    # box quarantines it as ``claim_layer_conflict``. So the rule is: dialogue or
    # inner evidence **with** a speaker becomes ``statement``; without one it stays
    # ``fact`` and the gap is visible in notes.
    #
    # This is what unlocks claims (B6/B7): the claims stage only consumes
    # ``claim_layer == "statement"``, so as long as no speaker existed there were
    # never any statements and therefore never any claims.
    if not evidence:
        return "fact"
    if all(item.speaker is None for item in evidence):
        return "fact"
    if any(item.mode in ("dialogue", "inner") for item in evidence):
        return "statement"
    return "fact"


def chapter_fact_to_engine_output(
    fact: ChapterFact,
    *,
    unit_id: str,
    text: str,
    book_id: str = "",
    gleaning_round: int = 0,
    extra_notes: list[str] | None = None,
) -> EngineOutput:
    """Map one validated :class:`ChapterFact` into the V5 EngineOutput contract."""
    entities: list[Entity] = []
    mentions: list[Mention] = []
    assertions: list[Assertion] = []
    events: list[Event] = []
    # Relation types with no honest L1 mapping. Recorded so the omission is
    # visible in notes; assemble turns them into proposed_predicates rows.
    unmapped_relations: list[str] = []

    entity_keys: set[tuple[str, str]] = set()
    mention_keys: set[tuple[str, int]] = set()
    anchored_quotes = 0
    total_quotes = 0

    def _add_entity(
        name: str,
        etype: str,
        aliases: list[str],
        confidence: float,
        tier: str = "minor",
    ) -> None:
        key = (etype, name)
        if not name or key in entity_keys:
            return
        entity_keys.add(key)
        clean = _clean_aliases(aliases, name)
        entities.append(
            Entity(
                book_id=book_id,
                type=etype,
                canonical_name=name,
                aliases=clean,
                tier=tier,  # type: ignore[arg-type]
                status="active",
                confidence=confidence,
            )
        )
        for surface, mention_type, mconf in (
            [(name, "name", 0.9)]
            + [(alias, "alias", 0.6) for alias in clean]
        ):
            for idx, start in enumerate(find_occurrences(text, surface)):
                if idx >= _MAX_MENTIONS_PER_SURFACE:
                    break
                mkey = (surface, start)
                if mkey in mention_keys:
                    continue
                mention_keys.add(mkey)
                mentions.append(
                    Mention(
                        unit_id=unit_id,
                        surface=surface,
                        start=start,
                        end=start + len(surface),
                        occurrence=idx + 1,
                        mention_type=mention_type,
                        confidence=mconf,
                    )
                )

    def _add_assertion(
        predicate: str,
        library: str,
        roles: dict[str, Any],
        evidence: list[Evidence],
        confidence: float,
        qualifiers: dict[str, Any] | None = None,
    ) -> None:
        """Append a raw assertion. Only for predicates whose name is already the
        registry's own (an unmapped relation keeps its extractor name so assemble
        can propose it); everything else goes through :func:_add_mapped."""
        clean_roles = {
            key: value
            for key, value in roles.items()
            if value is not None and value != [] and value != ""
        }
        assertions.append(
            Assertion(
                book_id=book_id,
                predicate=predicate,
                library=library,
                roles=clean_roles,
                qualifiers=dict(qualifiers or {}),
                claim_layer=_claim_layer(evidence),  # type: ignore[arg-type]
                speaker=None,
                evidence=evidence,
                confidence=confidence,
                confidence_method="engine_self_report",
                status=_status(evidence),
            )
        )

    def _add_mapped(
        mapped: pmap.MappedAssertion,
        evidence: list[Evidence],
        confidence: float,
    ) -> None:
        """Append an assertion whose predicate/roles already came from the registry.

        Nine-class record names are not predicates (`02-引擎契约`: map them onto
        L1; unmapped goes to proposals).  Mapping happens in
        :mod:cbb_stage1.normalize.predicate_map, so this function only carries
        the result — it never invents a predicate name.
        """
        assertions.append(
            Assertion(
                book_id=book_id,
                predicate=mapped.predicate,
                library=mapped.library,
                roles=dict(mapped.roles),
                qualifiers=dict(mapped.qualifiers),
                claim_layer=_claim_layer(evidence),  # type: ignore[arg-type]
                speaker=None,
                evidence=evidence,
                confidence=confidence,
                confidence_method="engine_self_report",
                status=_status(evidence),
            )
        )

    # ── entity tiers (evidence-based, §5.15) ────────────────────
    #
    # ``tier`` decides how hard an assertion about this entity is verified. It used
    # to be the constant "major" for every character and location, which collapsed
    # the whole verification matrix onto one level. The tier is now derived from
    # what the chapter actually shows: how often the name appears, whether the
    # entity has aliases, whether it acts in an event, whether it is in a relation.
    tier_map = assign_tiers(
        [*({"name": c.name, "type": "person", "aliases": c.new_aliases} for c in fact.characters),
         *({"name": l.name, "type": "location", "aliases": []} for l in fact.locations)],
        unit_text=text,
        event_participants=[
            name for ev in fact.events for name in (ev.participants or [])
        ],
        relation_names=[
            name for rel in fact.relationships for name in (rel.person_a, rel.person_b)
        ],
    )

    def _tier_of(name: str, fallback: str = "minor") -> str:
        return str((tier_map.get(name) or {}).get("tier") or fallback)

    # ── characters -> person entities + mentions ────────────────
    for ch in fact.characters:
        _add_entity(ch.name, "person", ch.new_aliases, 0.8, tier=_tier_of(ch.name))
        if ch.appearance:
            total_quotes += 1
            appearance_evidence = _anchor(
                unit_id, text, ch.evidence, ch.evidence_speaker
            )
            anchored_quotes += 1 if appearance_evidence else 0
            _add_mapped(
                pmap.map_appearance(ch.name, ch.appearance),
                appearance_evidence, 0.6,
            )
        for ability in ch.abilities_gained:
            total_quotes += 1
            ability_evidence = _anchor(
                unit_id, text, ability.evidence, ability.evidence_speaker
            )
            anchored_quotes += 1 if ability_evidence else 0
            _add_mapped(
                pmap.map_ability(
                    ch.name, ability.dimension, ability.name, ability.description
                ),
                ability_evidence, 0.6,
            )

    # ── locations -> location entities + setting assertions ────
    for loc in fact.locations:
        # ``peers`` are neighbouring locations, not alternative spellings of
        # this one: they must never become aliases (fanout would delete the
        # canonical entity).  The names are preserved on the assertion below.
        _add_entity(loc.name, "location", [], 0.7, tier=_tier_of(loc.name))
        total_quotes += 1
        location_evidence = _anchor(
            unit_id, text, loc.evidence, loc.evidence_speaker
        )
        anchored_quotes += 1 if location_evidence else 0
        _add_mapped(
            pmap.map_location(
                loc.name, loc.type, loc.parent, loc.peers, loc.description
            ),
            location_evidence, 0.7,
        )

    # ── concepts -> concept entities + setting assertions ──────
    for concept in fact.new_concepts:
        # ``related`` lists semantically related entities, not aliases; keep
        # them in the assertion roles only.
        _add_entity(concept.name, "concept", [], 0.6)
        total_quotes += 1
        concept_evidence = _anchor(
            unit_id, text, concept.evidence, concept.evidence_speaker
        )
        anchored_quotes += 1 if concept_evidence else 0
        _add_mapped(
            pmap.map_concept(
                concept.name, concept.category, concept.definition, concept.related
            ),
            concept_evidence, 0.6,
        )

    # ── relationships -> relation assertions ───────────────────
    for rel in fact.relationships:
        total_quotes += 1
        evidence = _anchor(unit_id, text, rel.evidence, rel.evidence_speaker)
        anchored_quotes += 1 if evidence else 0
        mapped_relation = pmap.map_relationship(
            rel.relation_type, rel.person_a, rel.person_b
        )
        if mapped_relation is None:
            # No honest L1 mapping. Emit the assertion under its ORIGINAL
            # predicate name so assemble can do what the spec requires: an
            # unregistered predicate becomes a `proposed_predicates` row plus a
            # schema quarantine, never a silent drop and never an auto-register.
            # Swallowing it here would make the proposal vanish (measured:
            # proposals=0 and no proposed_predicates.jsonl at all).
            unmapped_relations.append(rel.relation_type)
            _add_assertion(
                rel.relation_type,
                "关系",
                {
                    "施事": rel.person_a,
                    "受事": rel.person_b,
                },
                evidence,
                0.75 if evidence else 0.5,
                qualifiers={
                    "is_new": rel.is_new,
                    "previous_type": rel.previous_type,
                    "unmapped": True,
                },
            )
        else:
            _add_mapped(
                pmap.MappedAssertion(
                    predicate=mapped_relation.predicate,
                    library=mapped_relation.library,
                    roles=mapped_relation.roles,
                    qualifiers={
                        **mapped_relation.qualifiers,
                        "is_new": rel.is_new,
                        "previous_type": rel.previous_type,
                    },
                ),
                evidence,
                0.75 if evidence else 0.5,
            )

    # ── spatial relationships -> setting assertions ────────────
    for srel in fact.spatial_relationships:
        total_quotes += 1
        evidence = _anchor(
            unit_id, text, srel.narrative_evidence, srel.evidence_speaker
        )
        anchored_quotes += 1 if evidence else 0
        _add_mapped(
            pmap.map_spatial(
                srel.source,
                srel.target,
                srel.relation_type,
                srel.value,
                distance_class=srel.distance_class,
                waypoints=srel.waypoints,
                confidence=srel.confidence,
                confidence_score=srel.confidence_score,
            ),
            evidence,
            0.7 if evidence else 0.45,
        )

    # ── world declarations -> setting assertions ───────────────
    for decl in fact.world_declarations:
        total_quotes += 1
        evidence = _anchor(
            unit_id, text, decl.narrative_evidence, decl.evidence_speaker
        )
        anchored_quotes += 1 if evidence else 0
        mapped_declaration = pmap.map_world_declaration(
            decl.declaration_type, decl.content
        )
        if mapped_declaration is None:
            # No L1 predicate can carry a declaration (they all require an
            # entity subject, and a declaration type is a category, not a
            # thing). Emit it under its own name so assemble records a
            # proposed_predicates row instead of inventing a subject.
            unmapped_relations.append(f"declaration:{decl.declaration_type}")
            _add_assertion(
                "世界观声明",
                "设定",
                {"类型": decl.declaration_type},
                evidence,
                0.65 if evidence else 0.4,
                qualifiers={
                    "declaration_type": decl.declaration_type,
                    "content": decl.content,
                    "unmapped": True,
                },
            )
        else:
            _add_mapped(
                mapped_declaration,
                evidence,
                0.65 if evidence else 0.4,
            )

    # ── item events -> assertions + events ─────────────────────
    for item in fact.item_events:
        participants = [p for p in (item.actor, item.recipient) if p]
        total_quotes += 1
        evidence = _anchor(unit_id, text, item.evidence, item.evidence_speaker)
        anchored_quotes += 1 if evidence else 0
        _add_mapped(
            pmap.map_item_event(
                item.item_name,
                item.item_type,
                item.action,
                item.actor,
                item.recipient,
                item.description,
            ),
            evidence, 0.5,
        )
        events.append(
            Event(
                book_id=book_id,
                type="物品",
                summary=f"{item.actor or '未知'} {item.action} {item.item_name}",
                participants=participants,
                location=None,
                unit_id=unit_id,
                evidence=evidence,
                status=_status(evidence),
            )
        )

    # ── org events -> assertions + events ──────────────────────
    for org in fact.org_events:
        participants = [p for p in (org.member,) if p]
        total_quotes += 1
        evidence = _anchor(unit_id, text, org.evidence, org.evidence_speaker)
        anchored_quotes += 1 if evidence else 0
        _add_mapped(
            pmap.map_org_event(
                org.org_name,
                org.org_type,
                org.member,
                org.role,
                org.action,
                org.description,
            ),
            evidence, 0.5,
        )
        events.append(
            Event(
                book_id=book_id,
                type="组织",
                summary=f"{org.member or org.org_name} {org.action} {org.org_name}",
                participants=participants,
                location=None,
                unit_id=unit_id,
                evidence=evidence,
                status=_status(evidence),
            )
        )

    # ── events -> V5 events ────────────────────────────────────
    for ev in fact.events:
        total_quotes += 1
        evidence = _anchor(unit_id, text, ev.evidence, ev.evidence_speaker)
        anchored_quotes += 1 if evidence else 0
        # ``time_expr`` is what the temporal stage anchors against: it re-reads the
        # unit text and requires the expression to appear there verbatim. Without
        # this field the box carried ``time.expression = null`` for every event and
        # no timeline could ever be built, no matter how explicit the prose was.
        time_record: dict[str, Any] = {}
        if ev.time_expr:
            time_record = {
                "time_expr": str(ev.time_expr),
                "anchor_unit": unit_id,
            }
        events.append(
            Event(
                book_id=book_id,
                type=ev.type,
                summary=ev.summary,
                participants=list(ev.participants),
                location=ev.location,
                time=time_record,
                importance=ev.importance,
                unit_id=unit_id,
                evidence=evidence,
                status=_status(evidence),
            )
        )

    # ── notes (self-description only; never facts) ─────────────
    notes: list[str] = list(extra_notes or [])
    notes.append(
        "ai-reader: "
        f"{len(fact.characters)} characters, {len(fact.locations)} locations, "
        f"{len(fact.relationships)} relationships, {len(fact.events)} events"
    )
    if total_quotes:
        notes.append(f"evidence anchored: {anchored_quotes}/{total_quotes} quotes verbatim")
    if unmapped_relations:
        distinct = sorted(set(unmapped_relations))
        notes.append(
            "relation types without an L1 predicate (left for assemble to propose): "
            + ", ".join(distinct)
        )
    if gleaning_round:
        notes.append(f"gleaning_round={gleaning_round}")

    mentions.sort(key=lambda m: (m.start, m.end))

    return EngineOutput(
        unit_id=unit_id,
        mentions=mentions,
        entities=entities,
        assertions=assertions,
        events=events,
        cues=[],
        summary=None,
        abstain=False,
        notes=notes,
        gleaning_round=gleaning_round,
    )


def sanitize_mapped_output(
    output: EngineOutput | dict,
    *,
    unit_text: str | None = None,
):
    """Run the A7 ten-class semantic cleaning over a mapped ``EngineOutput``.

    Thin entry point so callers that already hold the normalized output do not
    need to import the pipeline validator directly.  The cleaning rules live in
    :mod:`cbb_stage1.normalize.cleaning`.
    """
    from cbb_stage1.normalize.cleaning import clean_engine_output

    return clean_engine_output(output, unit_text=unit_text)


def validate_mapped_outputs(
    records,
    *,
    unit_texts=None,
    alias_map=None,
):
    """A7 pipeline validation: clean mapped outputs, then run the six conflict checks.

    Args:
        records: ``(unit_order, EngineOutput)`` pairs for the units.
        unit_texts: Optional ``unit_order -> exact unit text`` for verbatim
            evidence verification.
        alias_map: Optional alias -> canonical map for cross-unit comparison.

    Returns:
        :class:`cbb_stage1.fact_validator.ValidationReport`.
    """
    # Imported lazily: the pipeline validator imports this package's cleaning
    # rules, so a module-level import here would be circular (the wiring gate
    # still sees this import and keeps ``fact_validator`` reachable).
    from cbb_stage1.fact_validator import validate_engine_output

    return validate_engine_output(
        records, unit_texts=unit_texts, alias_map=alias_map
    )
