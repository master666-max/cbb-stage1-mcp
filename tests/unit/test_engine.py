"""Unit tests for the ported AI Reader extraction kernel (task-6 / P0-05).

Covers, with a fake LLM (no network):
* Pydantic nine-class schema + the patched ``minItems`` extraction schema;
* evidence anchoring (exact quote, occurrence, sha256, no fabrication);
* the alias safety table / Union-Find resolver (generic terms never bridge);
* fact_validator semantic cleaning;
* ChapterFact -> EngineOutput mapping;
* AiReaderEngine sync + async entry points, abstain path, gleaning rounds;
* DictionaryStore / FactStore adapters;
* the hard constraint: no ``src.db`` / ``fastapi`` / ``chromadb`` imports.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cbb_stage1.engines.ai_reader import (
    AiReaderEngine,
    AliasResolver,
    ChapterFact,
    CharacterFact,
    ContextSummaryBuilder,
    EntityDictEntry,
    EntityPreScanner,
    FactValidator,
    InMemoryDictionaryStore,
    InMemoryFactStore,
    JsonlDictionaryStore,
    JsonlFactStore,
    LlmUsage,
)
from cbb_stage1.engines.ai_reader.alias_resolver import (
    _alias_safety_level,
    alias_safety_level,
)
from cbb_stage1.engines.ai_reader.chapter_fact_extractor import _merge_chapter_facts
from cbb_stage1.engines.ai_reader.engine import ids_from_card
from cbb_stage1.engines.ai_reader.llm_client import (
    LLMError,
    _extract_json,
    _repair_truncated_json,
)
from cbb_stage1.engines.ai_reader.prompts import build_known_entities_section
from cbb_stage1.engines.ai_reader.schema import (
    EXTRACTION_SCHEMA,
    build_extraction_schema,
)
from cbb_stage1.evidence import anchor_quote, make_evidence, sha256_text, verify_evidence
from cbb_stage1.models import Entity
from cbb_stage1.normalize import chapter_fact_to_engine_output, normalize_llm_payload

FIXTURES = Path(__file__).parent.parent / "fixtures"
UNIT_ID = "u:testbook:v01:c001"
CARD = {
    "contract": "cbb-card/2",
    "book": {"id": "testbook", "title": "测试书"},
    "unit": {
        "unit_id": UNIT_ID,
        "volume": 1,
        "chapter": 1,
        "title": "第一章 青牛镇",
    },
}


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeLLM:
    """Async fake with the exact ``generate`` signature the clients expose."""

    is_cloud = True
    model = "fake-llm"

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate(
        self,
        system: str,
        prompt: str,
        format: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        num_ctx: int | None = None,
    ):
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "format": format,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "num_ctx": num_ctx,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLLM received more calls than expected")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, LlmUsage(prompt_tokens=11, completion_tokens=22, total_tokens=33)


# ── evidence ──────────────────────────────────────────────────────


def test_evidence_anchor_exact_span_and_occurrence() -> None:
    text = load_text("ai_reader_chapter.txt")
    quote = "韩立点了点头，跟着三叔向七玄门走去。"

    anchored = anchor_quote(text, quote)
    assert anchored is not None
    assert text[anchored.start:anchored.end] == quote
    assert anchored.occurrence == 1
    assert anchored.quote_sha256 == sha256_text(quote)

    evidence = make_evidence(UNIT_ID, text, quote)
    assert evidence is not None
    assert verify_evidence(text, evidence)

    # Unknown quote must never produce a fabricated offset.
    assert anchor_quote(text, "这句话不存在于原文") is None
    assert make_evidence(UNIT_ID, text, "这句话不存在于原文") is None


def test_evidence_choose_nearest_occurrence() -> None:
    text = "AB 甲乙 AB 丙丁 AB"
    anchored = anchor_quote(text, "AB", near=9)
    assert anchored is not None
    # second "AB" starts at offset 6; third at 12.
    assert anchored.start == 6
    assert anchored.occurrence == 2


def test_verify_evidence_rejects_tampered_span() -> None:
    text = load_text("ai_reader_chapter.txt")
    evidence = make_evidence(UNIT_ID, text, "韩立")
    assert evidence is not None
    tampered = evidence.model_copy(update={"start": evidence.start + 1})
    assert not verify_evidence(text, tampered)


# ── schema ────────────────────────────────────────────────────────


def test_extraction_schema_enforces_min_items() -> None:
    schema = build_extraction_schema()
    for field in ("characters", "relationships", "locations", "events"):
        assert schema["properties"][field]["minItems"] == 1
        assert "default" not in schema["properties"][field]

    event_props = schema["$defs"]["EventFact"]["properties"]
    assert event_props["participants"]["minItems"] == 1
    assert "default" not in event_props["participants"]
    assert "default" not in event_props["location"]
    assert EXTRACTION_SCHEMA == schema


def test_chapter_fact_schema_rejects_malformed_payload() -> None:
    with pytest.raises(ValidationError):
        ChapterFact.model_validate(
            {
                "chapter_id": 1,
                "novel_id": "b",
                "events": [{"summary": "缺少 type 字段"}],
            }
        )
    with pytest.raises(ValidationError):
        ChapterFact.model_validate(
            {
                "chapter_id": 1,
                "novel_id": "b",
                "item_events": [{"item_name": "剑"}],  # missing item_type/action
            }
        )


def test_normalize_llm_payload_fixes_field_aliases() -> None:
    payload = normalize_llm_payload(
        {
            "item_events": [{"name": "剑", "type": "武器", "action": "出现"}, "noise"],
            "org_events": [{"name": "七玄门", "type": "宗门", "action": "加入"}],
        }
    )
    assert payload["item_events"] == [
        {"item_name": "剑", "item_type": "武器", "action": "出现"}
    ]
    assert payload["org_events"][0]["org_name"] == "七玄门"

    # Top-level array from an object-schema model is unwrapped.
    assert normalize_llm_payload([{"chapter_id": 1, "novel_id": "b"}]) == {
        "chapter_id": 1,
        "novel_id": "b",
    }


# ── alias safety table / resolver ─────────────────────────────────


@pytest.mark.parametrize(
    "alias",
    ["大哥", "妈妈", "老人", "大汉", "堂主", "那厮", "他的剑", "她妈妈", "夫人"],
)
def test_alias_safety_blocked_terms(alias: str) -> None:
    assert _alias_safety_level(alias) == 0
    assert alias_safety_level(alias) < 2


@pytest.mark.parametrize("alias", ["众猴", "小", "一二三", "某某们"])
def test_alias_safety_suspicious_terms(alias: str) -> None:
    assert _alias_safety_level(alias) == 1


@pytest.mark.parametrize("alias", ["韩立", "二愣子", "孙悟空", "齐天大圣", "行者"])
def test_alias_safety_real_names(alias: str) -> None:
    assert _alias_safety_level(alias) == 2


def test_alias_resolver_merges_aliases_and_picks_full_canonical() -> None:
    dictionary = [
        EntityDictEntry(
            name="孙悟空", entity_type="person", frequency=10,
            aliases=["行者", "齐天大圣"], source="naming",
        )
    ]
    facts = [
        ChapterFact(
            chapter_id=1,
            novel_id="book",
            characters=[CharacterFact(name="孙悟空", new_aliases=["猴王"])],
        )
    ]
    alias_map = AliasResolver(
        InMemoryDictionaryStore(), InMemoryFactStore()
    ).build_from_sources(dictionary, facts)

    assert alias_map["行者"] == "孙悟空"
    assert alias_map["齐天大圣"] == "孙悟空"
    assert alias_map["猴王"] == "孙悟空"
    assert "孙悟空" not in alias_map  # canonical does not map to itself


def test_alias_resolver_generic_terms_never_bridge_entities() -> None:
    dictionary = [
        # A generic name entry must be skipped entirely (name + aliases).
        EntityDictEntry(
            name="大汉", entity_type="person", frequency=50,
            aliases=["李俊"], source="freq",
        ),
        EntityDictEntry(
            name="老人", entity_type="person", frequency=40,
            aliases=["李逵"], source="freq",
        ),
    ]
    facts = [
        ChapterFact(
            chapter_id=1,
            novel_id="book",
            characters=[
                CharacterFact(name="大汉", new_aliases=["李逵"]),
                CharacterFact(name="老人", new_aliases=["李俊"]),
            ],
        )
    ]
    alias_map = AliasResolver(
        InMemoryDictionaryStore(), InMemoryFactStore()
    ).build_from_sources(dictionary, facts)

    assert "李俊" not in alias_map
    assert "李逵" not in alias_map
    assert "大汉" not in alias_map


def test_alias_resolver_conflict_guard_keeps_established_groups_apart() -> None:
    dictionary = [
        EntityDictEntry(name="甲一", entity_type="person", frequency=10,
                        aliases=["甲二", "甲三"], source="dict"),
        EntityDictEntry(name="乙一", entity_type="person", frequency=10,
                        aliases=["乙二", "乙三"], source="dict"),
    ]
    facts = [
        ChapterFact(
            chapter_id=2,
            novel_id="book",
            characters=[CharacterFact(name="甲一", new_aliases=["乙一"])],
        )
    ]
    alias_map = AliasResolver(
        InMemoryDictionaryStore(), InMemoryFactStore()
    ).build_from_sources(dictionary, facts)

    assert alias_map["甲二"] == "甲一"
    assert alias_map["乙二"] == "乙一"
    # The cross-link between two size-3 groups must be refused.
    assert alias_map.get("乙一") != "甲一"
    assert alias_map.get("甲一") != "乙一"


# ── fact validator ────────────────────────────────────────────────


def test_fact_validator_cleans_generic_names_and_aliases() -> None:
    fact = ChapterFact.model_validate(
        {
            "chapter_id": 1,
            "novel_id": "book",
            "characters": [
                {"name": "李逵", "new_aliases": ["李俊", "大哥", "黑旋风"]},
                {"name": "李俊"},
                {"name": "大汉"},
            ],
            "locations": [
                {"name": "门口", "type": "地点"},
                {"name": "梁山泊", "type": "湖泊"},
                {"name": "李逵府邸", "type": "府邸"},
            ],
            "relationships": [
                {"person_a": "李逵", "person_b": "李俊", "relation_type": "兄弟",
                 "evidence": "李逵与李俊同来。"},
                {"person_a": "李逵", "person_b": "路人甲", "relation_type": "朋友",
                 "evidence": "李逵与路人甲同行。"},
            ],
            "events": [
                {"summary": "李逵与李俊同上梁山", "type": "旅行",
                 "participants": [], "location": None},
            ],
        }
    )
    cleaned = FactValidator().validate(fact)

    char_names = {ch.name for ch in cleaned.characters}
    # 大汉 is a generic person reference -> dropped.  李俊 is an independent
    # character that 李逵 explicitly lists as an alias, so the validator merges
    # it into 李逵 (explicit alias link) instead of keeping two entries.
    assert char_names == {"李逵"}
    li_kui = next(ch for ch in cleaned.characters if ch.name == "李逵")
    # The validator does not apply the alias safety table itself (that is the
    # resolver's job); it only merges/drops character-level conflicts here.
    assert set(li_kui.new_aliases) == {"李俊", "黑旋风", "大哥"}

    loc_names = {loc.name for loc in cleaned.locations}
    assert loc_names == {"梁山泊"}  # 门口 (generic) and 李逵府邸 (hallucinated) dropped

    # Relationship to an unknown person is dropped.
    relations = {(r.person_a, r.person_b) for r in cleaned.relationships}
    assert relations == {("李逵", "李俊")}

    # Empty event participants/location are backfilled from the character set.
    event = cleaned.events[0]
    assert set(event.participants) == {"李逵", "李俊"}
    assert event.location is None


def test_fact_validator_disambiguates_homonym_locations() -> None:
    fact = ChapterFact.model_validate(
        {
            "chapter_id": 3,
            "novel_id": "book",
            "locations": [
                {"name": "大观园", "type": "园林"},
                {"name": "夹道", "type": "通道", "parent": "大观园"},
            ],
        }
    )
    cleaned = FactValidator().validate(fact)
    loc_names = {loc.name for loc in cleaned.locations}
    assert "大观园·夹道" in loc_names  # parent prefix prevents cross-building poll
    assert "夹道" not in loc_names
    renamed = next(loc for loc in cleaned.locations if loc.name == "大观园·夹道")
    assert renamed.parent == "大观园"


def test_alias_resolver_store_api_and_cache_invalidation() -> None:
    dict_store = InMemoryDictionaryStore()
    fact_store = InMemoryFactStore()
    dict_store.replace_entries(
        "book",
        [
            EntityDictEntry(
                name="韩立", entity_type="person", frequency=5,
                aliases=["二愣子"], source="dict",
            )
        ],
    )
    resolver = AliasResolver(dict_store, fact_store)

    assert resolver.build_alias_map("book") == {"二愣子": "韩立"}
    resolver.invalidate("book")
    assert resolver.build_alias_map("book") == {"二愣子": "韩立"}
    assert resolver.build_alias_map("unknown-book") == {}


# ── normalization: ChapterFact -> EngineOutput ───────────────────


def test_mapping_filters_unsafe_aliases_and_marks_missing_evidence() -> None:
    text = load_text("ai_reader_chapter.txt")
    fact = ChapterFact.model_validate(
        {
            "chapter_id": 1,
            "novel_id": "testbook",
            "characters": [
                {"name": "韩立", "new_aliases": ["二愣子", "大哥"]},
                {"name": "三叔"},
            ],
            "relationships": [
                {"person_a": "韩立", "person_b": "三叔", "relation_type": "叔侄",
                 "evidence": "这句话不存在于原文"},
            ],
        }
    )
    output = chapter_fact_to_engine_output(
        fact, unit_id=UNIT_ID, text=text, book_id="testbook"
    )

    hanli = next(e for e in output.entities if e.canonical_name == "韩立")
    assert hanli.aliases == ["二愣子"]  # 大哥 is a kinship term -> filtered

    # 叔侄 maps onto the registered L1 predicate 亲属; with no quote the
    # Unanchored is expressed by `status`, not by the narrative layer:
    # `statement` means "a character claimed this" and requires a speaker,
    # which this assertion has none of.
    rel_assertion = next(a for a in output.assertions if a.predicate == "亲属")
    assert rel_assertion.status == "needs_evidence"
    assert rel_assertion.claim_layer == "fact"
    assert rel_assertion.speaker is None
    assert rel_assertion.evidence == []
    assert rel_assertion.qualifiers["relation_type"] == "叔侄"
    assert "evidence anchored: 0/1" in " ".join(output.notes)

    # Every emitted Evidence must match the text exactly (no fabricated offsets).
    for assertion in output.assertions:
        for evidence in assertion.evidence:
            assert text[evidence.start:evidence.end] == evidence.quote
            assert evidence.quote_sha256 == sha256_text(evidence.quote)
    for mention in output.mentions:
        assert text[mention.start:mention.end] == mention.surface


# ── engine end-to-end (fake LLM) ─────────────────────────────────


def test_engine_maps_fixture_to_engine_output() -> None:
    text = load_text("ai_reader_chapter.txt")
    llm = FakeLLM([load_json("ai_reader_fact.json")])
    engine = AiReaderEngine(llm=llm)

    output = engine.extract(UNIT_ID, text, CARD, [])

    assert output.unit_id == UNIT_ID
    assert output.abstain is False
    assert output.gleaning_round == 0

    names = {entity.canonical_name for entity in output.entities}
    assert {"韩立", "三叔", "青牛镇", "七玄门", "宗门"} <= names

    hanli = next(e for e in output.entities if e.canonical_name == "韩立")
    assert hanli.type == "person"
    assert hanli.aliases == ["二愣子"]
    assert hanli.book_id == "testbook"

    # Mentions are anchored to real text spans.
    hanli_mentions = [m for m in output.mentions if m.surface == "韩立"]
    assert hanli_mentions
    for mention in hanli_mentions:
        assert text[mention.start:mention.end] == "韩立"
        assert mention.unit_id == UNIT_ID
    alias_mentions = [m for m in output.mentions if m.surface == "二愣子"]
    assert alias_mentions and alias_mentions[0].mention_type == "alias"

    # Relationship assertion carries verbatim evidence.
    # Nine-class record names are not predicates: 叔侄 maps onto the registered
    # L1 predicate 亲属 (02-引擎契约), and the concrete relation_type survives in
    # qualifiers so no information is lost.
    rel = next(a for a in output.assertions if a.predicate == "亲属")
    assert rel.library == "关系"
    assert rel.status == "asserted"
    assert rel.claim_layer == "fact"
    assert rel.roles["施事"] == "韩立"
    assert rel.roles["受事"] == "三叔"
    assert rel.qualifiers["relation_type"] == "叔侄"
    assert len(rel.evidence) == 1
    assert rel.evidence[0].quote == "韩立点了点头，跟着三叔向七玄门走去。"
    assert verify_evidence(text, rel.evidence[0])

    # Location attributes map onto 位于 (关系库: 主体/值). This fixture has no
    # spatial_relationships, so the assertion comes from the location class.
    located = [a for a in output.assertions if a.predicate == "位于"]
    assert located, "no 位于 assertion was emitted for the location class"
    for assertion in located:
        assert assertion.library == "关系"
        assert "主体" in assertion.roles

    # Events: travel + item + org.
    summaries = [event.summary for event in output.events]
    assert "韩立与三叔前往七玄门" in summaries
    assert any("行囊" in s for s in summaries)
    assert any("七玄门" in s for s in summaries)
    for event in output.events:
        assert event.unit_id == UNIT_ID
        assert event.book_id == "testbook"

    # LLM call contract: schema with minItems passed as format; cloud schema
    # embedded in the system prompt; anti-fabrication wording preserved.
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["format"]["properties"]["characters"]["minItems"] == 1
    assert "绝对禁止编造" in call["system"]
    assert "输出 JSON Schema" in call["system"]
    assert "第 1 章" in call["prompt"]

    # Notes are self-description only.
    assert any("ai-reader:" in note for note in output.notes)


def test_engine_injects_known_entities_and_card_rolling_summary() -> None:
    text = load_text("ai_reader_chapter.txt")
    card = dict(CARD)
    card["rolling"] = {"summary": "前情：韩立离开家乡。"}
    llm = FakeLLM([load_json("ai_reader_fact.json")])
    engine = AiReaderEngine(llm=llm)
    known = [
        Entity(book_id="testbook", type="person", canonical_name="韩立", aliases=["二愣子"]),
    ]

    engine.extract(UNIT_ID, text, card, known)

    system = llm.calls[0]["system"]
    assert "前情：韩立离开家乡。" in system
    assert "### 本书已知实体" in system
    assert "二愣子" in system


def test_engine_abstains_when_llm_fails() -> None:
    text = load_text("ai_reader_chapter.txt")
    llm = FakeLLM([LLMError("boom"), LLMError("boom again")])
    engine = AiReaderEngine(llm=llm)

    output = engine.extract(UNIT_ID, text, CARD, [])

    assert output.abstain is True
    assert any("extraction failed" in note for note in output.notes)
    assert len(llm.calls) == 2  # first attempt + aggressive-truncation retry


def test_engine_abstains_on_empty_text() -> None:
    llm = FakeLLM([])
    output = AiReaderEngine(llm=llm).extract(UNIT_ID, "   ", CARD, [])
    assert output.abstain is True
    assert "no_text" in " ".join(output.notes)
    assert llm.calls == []


def test_engine_gleaning_round_adds_new_candidates() -> None:
    text = load_text("ai_reader_chapter.txt")
    llm = FakeLLM([load_json("ai_reader_fact.json"), load_json("ai_reader_fact_continue.json")])
    engine = AiReaderEngine(llm=llm, max_gleanings=1)

    output = engine.extract(UNIT_ID, text, CARD, [])

    assert len(llm.calls) == 2
    assert "续抽" in llm.calls[1]["prompt"]
    assert output.gleaning_round == 1
    summaries = [event.summary for event in output.events]
    assert "三叔叮嘱韩立莫要惹事" in summaries
    # The duplicate event from the continue round was de-duplicated.
    assert summaries.count("韩立与三叔前往七玄门") == 1
    assert any("gleaning_round=1" in note for note in output.notes)


def test_engine_gleaning_stops_when_nothing_new() -> None:
    text = load_text("ai_reader_chapter.txt")
    empty_round = {
        "chapter_id": 1,
        "novel_id": "testbook",
        "characters": [{"name": "韩立"}],
        "relationships": [],
        "locations": [],
        "spatial_relationships": [],
        "item_events": [],
        "org_events": [],
        "events": [],
        "new_concepts": [],
        "world_declarations": [],
    }
    llm = FakeLLM([load_json("ai_reader_fact.json"), empty_round])
    engine = AiReaderEngine(llm=llm, max_gleanings=2)

    output = engine.extract(UNIT_ID, text, CARD, [])

    # Round 1 produced no new candidates -> stop, do not burn round 2.
    assert len(llm.calls) == 2
    assert output.gleaning_round == 1
    assert any("no new candidates" in note for note in output.notes)


def test_engine_gleaning_disabled_by_default() -> None:
    text = load_text("ai_reader_chapter.txt")
    llm = FakeLLM([load_json("ai_reader_fact.json")])
    AiReaderEngine(llm=llm).extract(UNIT_ID, text, CARD, [])
    assert len(llm.calls) == 1


def test_engine_gleaning_failure_is_recorded_not_fatal() -> None:
    text = load_text("ai_reader_chapter.txt")
    llm = FakeLLM([load_json("ai_reader_fact.json"), LLMError("continue failed")])
    output = AiReaderEngine(llm=llm, max_gleanings=2).extract(UNIT_ID, text, CARD, [])

    assert output.abstain is False  # round 0 result is still delivered
    assert output.gleaning_round == 1
    assert any("gleaning round 1 failed" in note for note in output.notes)


def test_extractor_segments_long_chapter_and_merges() -> None:
    long_text = (load_text("ai_reader_chapter.txt") + "\n") * 60  # > 7000 chars
    assert len(long_text) > 7000
    llm = FakeLLM([load_json("ai_reader_fact.json"), load_json("ai_reader_fact_continue.json")])
    engine = AiReaderEngine(llm=llm)  # default 128K budget -> segmentation enabled

    output = engine.extract(UNIT_ID, long_text, CARD, [])

    assert len(llm.calls) == 2
    assert any("segmented extraction: 2 segments" in note for note in output.notes)
    assert "（第 1/2 部分）" in llm.calls[0]["prompt"]
    assert "（第 2/2 部分）" in llm.calls[1]["prompt"]
    assert {entity.canonical_name for entity in output.entities} >= {"韩立", "青牛镇"}


def test_engine_sync_extract_works_inside_running_event_loop() -> None:
    text = load_text("ai_reader_chapter.txt")

    async def _inner() -> str:
        llm = FakeLLM([load_json("ai_reader_fact.json")])
        engine = AiReaderEngine(llm=llm)
        output = engine.extract(UNIT_ID, text, CARD, [])  # sync call inside a loop
        return output.unit_id

    assert asyncio.run(_inner()) == UNIT_ID


def test_engine_aextract_is_async_native() -> None:
    text = load_text("ai_reader_chapter.txt")
    llm = FakeLLM([load_json("ai_reader_fact.json")])

    async def _run():
        return await AiReaderEngine(llm=llm).aextract(UNIT_ID, text, CARD, [])

    output = asyncio.run(_run())
    assert output.unit_id == UNIT_ID
    assert output.abstain is False


def test_ids_from_card_fallbacks() -> None:
    assert ids_from_card(CARD, UNIT_ID) == ("testbook", 1)
    assert ids_from_card({}, "u:book:v02:c007") == ("", 7)
    assert ids_from_card({"book_id": "b2", "chapter_id": 4}, "u:x") == ("b2", 4)


def test_engine_exposes_engine_protocol_plus_gleaning_slots() -> None:
    import inspect

    sig = inspect.signature(AiReaderEngine.extract)
    assert list(sig.parameters)[:5] == ["self", "unit_id", "text", "card", "known_entities"]
    assert sig.parameters["gleaning_round"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["gleaning_round"].default == 0
    assert sig.parameters["max_gleanings"].kind is inspect.Parameter.KEYWORD_ONLY
    # The async native entry point mirrors the sync one.
    async_sig = inspect.signature(AiReaderEngine.aextract)
    assert list(async_sig.parameters)[:5] == [
        "self", "unit_id", "text", "card", "known_entities"
    ]


# ── pre-scan / context summary / stores ─────────────────────────


def test_pre_scanner_phase1_and_dictionary_store() -> None:
    text = load_text("ai_reader_chapter.txt")
    chapters = ["\n".join([text] * 5)]
    store = InMemoryDictionaryStore()
    llm = FakeLLM([load_json("ai_reader_prescan.json")])
    scanner = EntityPreScanner(dictionary_store=store, llm=llm)

    candidates = asyncio.run(scanner.scan("testbook", chapters, ["第一章 青牛镇"]))

    names = {entry.name for entry in candidates}
    assert "韩立" in names  # naming-pattern signal (名叫韩立/绰号二愣子)
    assert "七玄门" in names  # n-gram + suffix signal
    assert store.get_status("testbook") == "completed"
    assert {entry.name for entry in store.list_entries("testbook")} == names

    # Phase 2 classification is merged back (七玄门 typed as org by the LLM).
    qixuan = next(entry for entry in store.list_entries("testbook") if entry.name == "七玄门")
    assert qixuan.entity_type == "org"


def test_context_summary_builder_sections() -> None:
    fact = ChapterFact.model_validate(load_json("ai_reader_fact.json"))
    fact_store = InMemoryFactStore([("testbook", fact)])
    dict_store = InMemoryDictionaryStore()
    dict_store.replace_entries(
        "testbook",
        [
            EntityDictEntry(name="韩立", entity_type="person", frequency=5,
                            source="naming", aliases=["二愣子"]),
            EntityDictEntry(name="青牛镇", entity_type="location", frequency=4,
                            source="freq"),
        ],
    )
    builder = ContextSummaryBuilder(fact_store, dict_store)

    summary = builder.build("testbook", chapter_num=2)

    assert "### 已知人物" in summary
    assert "韩立" in summary
    assert "### 已知关系" in summary
    assert "叔侄" in summary
    assert "### 已知地点" in summary
    assert "青牛镇" in summary
    assert "### 文中明确命名的实体" in summary
    assert "本书高频实体参考" in summary
    # Chapter 1 has no preceding facts.
    assert builder.build("testbook", chapter_num=1) != ""


def test_in_memory_fact_store_upserts_by_chapter() -> None:
    store = InMemoryFactStore()
    fact1 = ChapterFact.model_validate(load_json("ai_reader_fact.json"))
    fact2 = fact1.model_copy(update={"chapter_id": 2})
    store.save_fact("book", fact1)
    store.save_fact("book", fact2)
    store.save_fact("book", fact1)  # replace, not duplicate
    assert [f.chapter_id for f in store.list_facts("book")] == [1, 2]


def test_jsonl_stores_roundtrip(tmp_path: Path) -> None:
    fact = ChapterFact.model_validate(load_json("ai_reader_fact.json"))
    fact_store = JsonlFactStore(tmp_path / "facts.jsonl")
    fact_store.save_fact("book", fact)
    fact_store.save_fact("other", fact.model_copy(update={"chapter_id": 9}))
    loaded = fact_store.list_facts("book")
    assert len(loaded) == 1
    assert loaded[0].characters[0].name == "韩立"

    dict_store = JsonlDictionaryStore(tmp_path / "dict.jsonl")
    dict_store.set_status("book", "running")
    dict_store.replace_entries(
        "book", [EntityDictEntry(name="韩立", entity_type="person", frequency=3, source="freq")]
    )
    dict_store.set_status("book", "completed")
    assert dict_store.get_status("book") == "completed"
    assert dict_store.list_entries("book")[0].name == "韩立"


def test_known_entities_section_marks_aliases_as_hints() -> None:
    section = build_known_entities_section(
        [Entity(book_id="b", type="person", canonical_name="韩立", aliases=["二愣子"])],
        [{"name": "七玄门", "type": "org", "aliases": ["七玄"]}],
    )
    assert "韩立" in section and "七玄门" in section
    assert "需本章原文确认" in section
    assert "不同的人绝不能互为别名" in section


# ── LLM response cleanup ─────────────────────────────────────────


def test_extract_json_strips_thinking_and_fences() -> None:
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('思考中 <think>内部推理</think> 结果：{"a": 2} 完') == {"a": 2}


def test_repair_truncated_json_closes_brackets() -> None:
    broken = '{"a": {"b": [1, 2, 3], "c": "未完成'
    repaired = _repair_truncated_json(broken)
    assert json.loads(repaired) == {"a": {"b": [1, 2, 3]}}


# ── hard constraints ─────────────────────────────────────────────


def test_no_forbidden_imports_in_ported_packages() -> None:
    import ast

    package_root = Path(__file__).resolve().parents[2] / "src" / "cbb_stage1"
    forbidden_roots = {"src", "fastapi", "chromadb"}
    offenders: list[str] = []
    for subdir in ("engines", "evidence", "normalize"):
        for path in (package_root / subdir).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            modules: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
            for module in modules:
                if module.split(".")[0] in forbidden_roots:
                    offenders.append(f"{path.name}: imports {module}")
    assert offenders == []


def test_merge_chapter_facts_deduplicates() -> None:
    fact = ChapterFact.model_validate(load_json("ai_reader_fact.json"))
    merged = _merge_chapter_facts([fact, fact], "testbook", 1)
    assert len(merged.events) == len(fact.events)
    assert len(merged.relationships) == len(fact.relationships)
    assert len(merged.characters) == len(fact.characters)
    assert merged.characters[0].new_aliases == ["二愣子"]
