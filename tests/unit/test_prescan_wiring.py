"""Pre-scan / alias-resolver wiring tests for AiReaderEngine (P1-A).

Covers:
* ``prescan_book`` runs the EntityPreScanner, persists the dictionary and the
  next extraction prompt carries the dictionary hints;
* the pipeline-style path ``set_dictionary(book_id, entries)`` injects a
  pre-built dictionary (and wins over the store);
* ``AliasResolver`` is instantiated from the stores, feeds an alias hint
  section, and its cache is invalidated by a fresh pre-scan;
* **explicit degradation**: with no pre-scan / card glossary / known_entities
  the engine records it in ``EngineOutput.notes`` instead of pretending hints
  exist (and still extracts);
* a pre-scan request without a scanner is a hard ``EngineError``, not a silent
  no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.engines._async import run_coroutine_sync
from cbb_stage1.engines.ai_reader import (
    AiReaderEngine,
    AliasResolver,
    ChapterFact,
    EntityDictEntry,
    EntityPreScanner,
    InMemoryDictionaryStore,
    InMemoryFactStore,
)
from cbb_stage1.engines.base import EngineError
from cbb_stage1.models import Entity

FIXTURES = Path(__file__).parent.parent / "fixtures"
UNIT_ID = "u:testbook:v01:c001"
BOOK = "testbook"
CARD = {
    "contract": "cbb-card/2",
    "book": {"id": BOOK, "title": "测试书"},
    "unit": {"unit_id": UNIT_ID, "volume": 1, "chapter": 1, "title": "第一章 青牛镇"},
}


def load_text() -> str:
    return (FIXTURES / "ai_reader_chapter.txt").read_text(encoding="utf-8")


def load_fact_payload() -> dict:
    return json.loads((FIXTURES / "ai_reader_fact.json").read_text(encoding="utf-8"))


class FakeLLM:
    """Minimal async LLM fake for the chapter-fact extraction path."""

    is_cloud = True
    model = "fake-ai-reader"

    def __init__(self, responses: list, *, fail: bool = False) -> None:
        self.responses = list(responses)
        self.fail = fail
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
        from cbb_stage1.engines.ai_reader.llm_client import LlmUsage

        self.calls.append({"system": system, "prompt": prompt})
        if self.fail:
            raise RuntimeError("llm unavailable")
        if not self.responses:
            raise AssertionError("unexpected extra LLM call")
        return self.responses.pop(0), LlmUsage(prompt_tokens=1, completion_tokens=2)


class FakeScanner:
    """EntityPreScanner stand-in (async ``scan``) with call accounting."""

    def __init__(self, entries: list[EntityDictEntry]) -> None:
        self.entries = list(entries)
        self.calls: list[dict] = []

    async def scan(self, book_id: str, chapters: list[str], titles=None):
        self.calls.append({"book_id": book_id, "chapters": list(chapters), "titles": titles})
        return list(self.entries)


def dict_entry(name: str, aliases: list[str], *, frequency: int = 5) -> EntityDictEntry:
    return EntityDictEntry(
        name=name,
        entity_type="person",
        frequency=frequency,
        aliases=aliases,
        source="naming",
    )


# ── pre-scan wiring ─────────────────────────────────────────────


def test_engine_auto_instantiates_prescanner_and_alias_resolver() -> None:
    """A1/A6 WIRED: stores are enough to wire both components, no manual setup."""
    dict_store = InMemoryDictionaryStore()
    fact_store = InMemoryFactStore()
    engine = AiReaderEngine(
        llm=FakeLLM([]), dictionary_store=dict_store, fact_store=fact_store
    )
    assert isinstance(engine.prescanner, EntityPreScanner)
    assert isinstance(engine.alias_resolver, AliasResolver)

    # Dictionary-only engine still wires the pre-scanner; alias resolver needs
    # both stores (its Union-Find input is dictionary + chapter facts).
    dict_only = AiReaderEngine(llm=FakeLLM([]), dictionary_store=InMemoryDictionaryStore())
    assert isinstance(dict_only.prescanner, EntityPreScanner)
    assert dict_only.alias_resolver is None

    bare = AiReaderEngine(llm=FakeLLM([]))
    assert bare.prescanner is None and bare.alias_resolver is None


def test_real_entity_pre_scanner_runs_through_engine() -> None:
    """A1 WIRED end-to-end: real scanner -> store -> hints in the next prompt."""
    text = (FIXTURES / "ai_reader_chapter.txt").read_text(encoding="utf-8")
    store = InMemoryDictionaryStore()
    # First LLM response is consumed by the pre-scan Phase-2 classification (the
    # payload is not a classification dict, so Phase 1 candidates are kept).
    llm = FakeLLM([{"entities": [], "alias_groups": [], "rejected": []}, load_fact_payload()])
    engine = AiReaderEngine(llm=llm, dictionary_store=store)

    entries = engine.prescan_book(BOOK, ["\n".join([text] * 5)], ["第一章 青牛镇"])

    assert entries, "real pre-scan must produce candidates from the chapter text"
    assert store.get_status(BOOK) == "completed"
    assert {e.name for e in store.list_entries(BOOK)} == {e.name for e in entries}
    assert "韩立" in {e.name for e in entries}

    out = engine.extract(UNIT_ID, text, CARD, [])
    system = llm.calls[-1]["system"]
    assert "### 预扫词典" in system
    assert any("prescan: dictionary hints=" in note for note in out.notes)


def test_prescan_book_persists_dictionary_and_injects_hints() -> None:
    store = InMemoryDictionaryStore()
    scanner = FakeScanner([dict_entry("韩立", ["二愣子"]), dict_entry("七玄门", [])])
    llm = FakeLLM([load_fact_payload()])
    engine = AiReaderEngine(llm=llm, dictionary_store=store, prescanner=scanner)

    entries = engine.prescan_book(BOOK, ["第一章文本"], ["第一章 青牛镇"])

    assert len(entries) == 2
    assert scanner.calls and scanner.calls[0]["book_id"] == BOOK
    assert store.get_status(BOOK) == "completed"
    assert engine.dictionary_status(BOOK) == "completed"
    assert {e.name for e in engine.dictionary_for(BOOK)} == {"韩立", "七玄门"}

    out = engine.extract(UNIT_ID, load_text(), CARD, [])

    system = llm.calls[-1]["system"]
    assert "### 预扫词典" in system
    assert "韩立" in system and "二愣子" in system
    assert any("prescan: dictionary hints=2" in note for note in out.notes)


def test_pipeline_style_set_dictionary_injection() -> None:
    store = InMemoryDictionaryStore()  # empty store, dictionary arrives pre-built
    llm = FakeLLM([load_fact_payload()])
    engine = AiReaderEngine(llm=llm, dictionary_store=store)

    entries = [dict_entry("韩立", ["二愣子"], frequency=9), dict_entry("三叔", [])]
    engine.set_dictionary(BOOK, entries)

    assert engine.dictionary_status(BOOK) == "injected"
    assert [e.name for e in engine.dictionary_for(BOOK)] == ["韩立", "三叔"]
    # The injected dictionary is mirrored into the store so downstream users
    # (alias resolver / context builder) see the same table.
    assert {e.name for e in store.list_entries(BOOK)} == {"韩立", "三叔"}

    out = engine.extract(UNIT_ID, load_text(), CARD, [])
    system = llm.calls[-1]["system"]
    assert "### 预扫词典" in system and "出现9次" in system
    assert any("prescan: dictionary hints=2" in note for note in out.notes)


def test_prescan_without_store_keeps_in_memory_dictionary() -> None:
    scanner = FakeScanner([dict_entry("韩立", ["二愣子"])])
    llm = FakeLLM([load_fact_payload()])
    engine = AiReaderEngine(llm=llm, prescanner=scanner)  # no dictionary_store

    engine.prescan_book(BOOK, ["第一章文本"])

    assert engine.dictionary_status(BOOK) == "completed"
    assert [e.name for e in engine.dictionary_for(BOOK)] == ["韩立"]
    out = engine.extract(UNIT_ID, load_text(), CARD, [])
    assert "### 预扫词典" in llm.calls[-1]["system"]
    assert any("prescan: dictionary hints=1" in note for note in out.notes)


def test_prescan_request_without_scanner_is_a_hard_error() -> None:
    engine = AiReaderEngine(llm=FakeLLM([load_fact_payload()]))
    with pytest.raises(EngineError):
        engine.prescan_book(BOOK, ["第一章文本"])


# ── alias resolver wiring ───────────────────────────────────────


def test_alias_resolver_is_wired_and_hints_are_injected() -> None:
    dict_store = InMemoryDictionaryStore()
    dict_store.replace_entries(BOOK, [dict_entry("韩立", ["二愣子"])])
    fact_store = InMemoryFactStore(
        [(BOOK, ChapterFact.model_validate(load_fact_payload()))]
    )
    llm = FakeLLM([load_fact_payload()])
    engine = AiReaderEngine(
        llm=llm, dictionary_store=dict_store, fact_store=fact_store
    )

    assert engine.alias_resolver is not None
    assert engine.alias_map(BOOK) == {"二愣子": "韩立"}

    out = engine.extract(UNIT_ID, load_text(), CARD, [])

    system = llm.calls[-1]["system"]
    assert "### 别名映射" in system
    assert "二愣子 → 韩立" in system
    assert any("prescan: alias map=1" in note for note in out.notes)


def test_prescan_invalidates_alias_cache() -> None:
    dict_store = InMemoryDictionaryStore()
    dict_store.replace_entries(BOOK, [dict_entry("韩立", ["二愣子"])])
    fact_store = InMemoryFactStore()
    engine = AiReaderEngine(
        llm=FakeLLM([]), dictionary_store=dict_store, fact_store=fact_store
    )
    assert engine.alias_map(BOOK) == {"二愣子": "韩立"}

    scanner = FakeScanner([dict_entry("韩立", ["小韩"])])
    engine.prescanner = scanner
    engine.prescan_book(BOOK, ["第一章文本"])

    # The alias resolver cache must be refreshed from the new dictionary.
    assert engine.alias_map(BOOK) == {"小韩": "韩立"}


def test_injected_dictionary_builds_alias_map_without_stores() -> None:
    engine = AiReaderEngine(llm=FakeLLM([]))
    engine.set_dictionary(BOOK, [dict_entry("韩立", ["二愣子"])])

    assert engine.alias_resolver is None
    assert engine.alias_map(BOOK) == {"二愣子": "韩立"}
    # Cached: a second call does not rebuild.
    assert engine.alias_map(BOOK) == engine.alias_map(BOOK)


# ── explicit degradation ───────────────────────────────────────


def test_no_dictionary_degrades_explicitly_and_still_extracts() -> None:
    llm = FakeLLM([load_fact_payload()])
    engine = AiReaderEngine(llm=llm)

    out = engine.extract(UNIT_ID, load_text(), CARD, [])

    assert out.abstain is False  # extraction itself is unaffected
    assert out.entities  # candidates are still produced
    assert any("prescan: degraded — no dictionary available" in note for note in out.notes)
    assert "### 预扫词典" not in llm.calls[-1]["system"]
    assert "### 别名映射" not in llm.calls[-1]["system"]


def test_card_glossary_is_not_reported_as_degraded() -> None:
    card = json.loads(json.dumps(CARD))
    card["glossary"] = {
        "entities": [{"name": "韩立", "type": "person", "aliases": ["二愣子"]}],
        "predicates": [],
    }
    llm = FakeLLM([load_fact_payload()])
    engine = AiReaderEngine(llm=llm)

    out = engine.extract(UNIT_ID, load_text(), card, [])

    assert any("using card glossary/known_entities hints only" in note for note in out.notes)
    assert not any("prescan: degraded" in note for note in out.notes)


def test_known_entities_are_not_reported_as_degraded() -> None:
    known = [Entity(book_id=BOOK, type="person", canonical_name="韩立")]
    llm = FakeLLM([load_fact_payload()])
    out = AiReaderEngine(llm=llm).extract(UNIT_ID, load_text(), CARD, known)
    assert any("using card glossary/known_entities hints only" in note for note in out.notes)
    assert not any("prescan: degraded" in note for note in out.notes)


# ── async API / sync bridge ────────────────────────────────────


def test_aprescan_book_is_awaitable_and_sync_wrapper_matches() -> None:
    scanner = FakeScanner([dict_entry("韩立", ["二愣子"])])
    engine = AiReaderEngine(llm=FakeLLM([]), prescanner=scanner)

    async def _run():
        return await engine.aprescan_book(BOOK, ["第一章文本"], None)

    entries = run_coroutine_sync(_run())
    assert [e.name for e in entries] == ["韩立"]
    assert len(scanner.calls) == 1
