"""LPMM adapter + OpenIE mode tests (fake LLM, no network, no live calls).

Covers the P1-A acceptance surface:
* two-stage NER -> RDF maps to ``EngineOutput`` (entities/mentions/assertions);
* thread-pool batch processing really runs paragraphs concurrently;
* retry on bad JSON and tolerant repair of truncated/fenced JSON;
* per-paragraph checkpoint resume (and corrupt-cache recompute);
* explicit abstain when all paragraphs fail (never a silent mock);
* partial failure keeps the successful paragraphs;
* ``build_engine(provider="lpmm" | "openie" | ...)`` wiring;
* OpenIE chapter mode is a switchable AI Reader mode.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from cbb_stage1.engines._textsplit import split_paragraphs
from cbb_stage1.engines.ai_reader import AiReaderEngine
from cbb_stage1.engines.ai_reader.openie import (
    OpenIEError,
    OpenIEExtractor,
    filter_entities,
    filter_triples,
    parse_json_payload,
)
from cbb_stage1.engines.base import EngineError
from cbb_stage1.engines.factory import build_engine
from cbb_stage1.engines.lpmm import LpmmAdapter, LpmmEngine, LpmmSettings
from cbb_stage1.engines.mock import MockEngine

UNIT_ID = "u:testbook:v01:c001"
CARD = {
    "contract": "cbb-card/2",
    "book": {"id": "testbook", "title": "测试书"},
    "unit": {"unit_id": UNIT_ID, "volume": 1, "chapter": 1, "title": "第一章"},
}

CHAPTER_TEXT = """第一段：韩立站在青牛镇外，望着远处的七玄门。

第二段：三叔带着韩立走向七玄门，一路叮嘱他莫要惹事。

第三段：七玄门坐落于青牛镇北方三十里处，乃是附近最大的宗门。"""


def _usage_json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class RoutingLLM:
    """Fake LLM that routes by prompt content (paragraph marker + stage).

    * stage detection: RDF prompts contain ``实体列表：``, NER prompts do not;
    * each paragraph marker has a queue of responses, so a test can inject one
      bad JSON response followed by a good one (retry path);
    * ``fail_markers`` always raises for those paragraphs.
    """

    is_cloud = True
    model = "fake-lpmm"

    def __init__(
        self,
        *,
        entities: dict[str, list[str]],
        triples: dict[str, list[list[str]]],
        delay: float = 0.0,
        fail_markers: set[str] | None = None,
        bad_json: dict[str, list[str]] | None = None,
        strict: bool = False,
    ) -> None:
        self.entities = entities
        self.triples = triples
        self.delay = delay
        self.fail_markers = set(fail_markers or set())
        self.bad_json = {k: list(v) for k, v in (bad_json or {}).items()}
        self.strict = strict
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def _marker(self, prompt: str) -> str | None:
        for marker in self.entities:
            if marker in prompt:
                return marker
        return None

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
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls.append({"system": system, "prompt": prompt})
        try:
            if self.strict:
                raise AssertionError("LLM must not be called (checkpoint should cover all paragraphs)")
            if self.delay:
                await asyncio.sleep(self.delay)
            marker = self._marker(prompt)
            if marker is None:
                raise ValueError("unmatched prompt")
            if marker in self.fail_markers:
                raise RuntimeError(f"forced failure for {marker}")
            is_rdf = "实体列表：" in prompt
            with self._lock:
                queued = self.bad_json.get(marker)
                if queued:
                    bad = queued.pop(0)
                    is_rdf = False  # bad_json is injected on the NER stage
                else:
                    bad = None
            if bad is not None:
                return bad, _usage()
            payload = self.triples[marker] if is_rdf else self.entities[marker]
            return _usage_json(payload), _usage()
        finally:
            with self._lock:
                self._active -= 1


def _usage():
    from cbb_stage1.engines.ai_reader.llm_client import LlmUsage

    return LlmUsage(prompt_tokens=7, completion_tokens=11, total_tokens=18)


def default_llm(**kwargs) -> RoutingLLM:
    return RoutingLLM(
        entities={
            "第一段": ["韩立", "青牛镇", "你"],
            "第二段": ["三叔", "韩立", "七玄门"],
            "第三段": ["七玄门", "青牛镇"],
        },
        triples={
            "第一段": [["韩立", "位于", "青牛镇"], ["韩立", "位于", "青牛镇"]],
            "第二段": [["三叔", "带领", "韩立"], ["韩立", "前往", "七玄门"]],
            "第三段": [["七玄门", "位于", "青牛镇"]],
        },
        **kwargs,
    )


# ── LPMM engine: mapping ─────────────────────────────────────────


def test_lpmm_maps_two_stage_output_to_engine_output() -> None:
    llm = default_llm()
    engine = LpmmEngine(llm=llm, settings=LpmmSettings(max_workers=2))

    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert out.unit_id == UNIT_ID
    assert out.abstain is False
    names = {entity.canonical_name for entity in out.entities}
    assert {"韩立", "青牛镇", "三叔", "七玄门"} <= names
    # LPMM entity type is unknown; the Resolve stage classifies it.
    assert all(entity.type == "unknown" for entity in out.entities)
    # "你" is in the source INVALID_ENTITY list and must be filtered out.
    assert "你" not in names

    # Mentions are anchored to real spans of the unit text.
    assert out.mentions
    for mention in out.mentions:
        assert CHAPTER_TEXT[mention.start:mention.end] == mention.surface

    # Triples -> relation assertions with paragraph backfill anchors.
    assert len(out.assertions) == 4  # one duplicate triple was dropped
    by_pred = {(a.predicate, a.roles["subject"], a.roles["object"]) for a in out.assertions}
    assert ("带领", "三叔", "韩立") in by_pred
    for assertion in out.assertions:
        assert assertion.status == "needs_evidence"
        assert assertion.claim_layer == "statement"
        assert assertion.evidence == []  # LPMM has no quote; never fabricate one
        start, end = assertion.roles["paragraph_span"]
        paragraphs = split_paragraphs(CHAPTER_TEXT)
        assert (start, end) in {(p.start, p.end) for p in paragraphs}
        assert CHAPTER_TEXT[start:end].strip()
        assert len(assertion.roles["paragraph_sha256"]) == 64

    assert any(note.startswith("lpmm: 3/3 paragraphs") for note in out.notes)
    assert any("Evidence stage must backfill quotes" in note for note in out.notes)


def test_lpmm_paragraph_span_matches_source_text() -> None:
    llm = default_llm()
    out = LpmmEngine(llm=llm).extract(UNIT_ID, CHAPTER_TEXT, CARD, [])
    paragraphs = split_paragraphs(CHAPTER_TEXT)
    spans = {(p.start, p.end) for p in paragraphs}
    for assertion in out.assertions:
        span = tuple(assertion.roles["paragraph_span"])
        assert span in spans
        start, end = span
        assert CHAPTER_TEXT[start:end].strip()


# ── thread pool ─────────────────────────────────────────────────


def test_lpmm_thread_pool_runs_paragraphs_concurrently() -> None:
    llm = default_llm(delay=0.05)
    engine = LpmmEngine(llm=llm, settings=LpmmSettings(max_workers=3, retries=0))

    started = time.monotonic()
    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])
    elapsed = time.monotonic() - started

    assert out.abstain is False
    assert llm.max_active >= 2  # at least two paragraphs were in flight together
    # 3 paragraphs * 2 stages * 50ms serial would be >= 300ms; parallel is faster.
    assert elapsed < 0.28


# ── retry / JSON repair ─────────────────────────────────────────


def test_lpmm_retries_bad_json_then_succeeds() -> None:
    llm = default_llm(bad_json={"第一段": ["这不是JSON"]})
    engine = LpmmEngine(llm=llm, settings=LpmmSettings(max_workers=1, retries=2))

    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert out.abstain is False
    # 第一段 needed two NER attempts: one bad JSON, then valid.
    ner_calls_first = [
        call for call in llm.calls if "第一段" in call["prompt"] and "实体列表：" not in call["prompt"]
    ]
    assert len(ner_calls_first) == 2
    assert any("第一段" in call["prompt"] for call in llm.calls)


def test_openie_repairs_truncated_and_fenced_json() -> None:
    # truncated array (missing closing bracket) is repaired
    assert parse_json_payload('["韩立", "青牛镇"', ("entities",)) == ["韩立", "青牛镇"]
    # markdown fence + wrapper key
    assert parse_json_payload('```json\n{"entities": ["韩立"]}\n```', ("entities",)) == ["韩立"]
    # thinking block before the payload
    assert parse_json_payload('<think>x</think>[["A", "B", "C"]]', ("triples",)) == [
        ["A", "B", "C"]
    ]


# ── checkpoint / resume ─────────────────────────────────────────


def test_lpmm_checkpoint_resume_skips_completed_paragraphs(tmp_path: Path) -> None:
    first_llm = default_llm()
    first = LpmmEngine(
        llm=first_llm,
        settings=LpmmSettings(max_workers=2, checkpoint_dir=tmp_path),
    )
    first_out = first.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])
    assert first_out.abstain is False
    assert first_llm.calls

    # Second run with an LLM that fails if called: everything must come from cache.
    second_llm = default_llm(strict=True)
    second = LpmmEngine(
        llm=second_llm,
        settings=LpmmSettings(max_workers=2, checkpoint_dir=tmp_path),
    )
    second_out = second.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert second_llm.calls == []
    assert {e.canonical_name for e in second_out.entities} == {
        e.canonical_name for e in first_out.entities
    }
    assert {a.predicate for a in second_out.assertions} == {
        a.predicate for a in first_out.assertions
    }
    assert (tmp_path / f"{split_paragraphs(CHAPTER_TEXT)[0].sha256}.json").exists()


def test_lpmm_corrupt_checkpoint_is_recomputed(tmp_path: Path) -> None:
    paragraph = split_paragraphs(CHAPTER_TEXT)[0]
    (tmp_path / f"{paragraph.sha256}.json").write_text("{not json", encoding="utf-8")
    llm = default_llm()
    engine = LpmmEngine(llm=llm, settings=LpmmSettings(max_workers=1, checkpoint_dir=tmp_path))

    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert out.abstain is False
    assert llm.calls  # corrupt cache was discarded and recomputed
    repaired = json.loads((tmp_path / f"{paragraph.sha256}.json").read_text(encoding="utf-8"))
    assert repaired["passage"] == paragraph.text


# ── abstain / partial failure ───────────────────────────────────


def test_lpmm_abstains_when_all_paragraphs_fail() -> None:
    markers = {"第一段", "第二段", "第三段"}
    llm = default_llm(fail_markers=markers)
    engine = LpmmEngine(llm=llm, settings=LpmmSettings(max_workers=2, retries=1))

    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert out.abstain is True
    assert out.mentions == [] and out.entities == [] and out.assertions == []
    assert any("all 3 paragraphs failed" in note for note in out.notes)
    assert any("failed paragraph hashes" in note for note in out.notes)


def test_lpmm_partial_failure_keeps_successful_paragraphs() -> None:
    llm = default_llm(fail_markers={"第三段"})
    engine = LpmmEngine(llm=llm, settings=LpmmSettings(max_workers=2, retries=1))

    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert out.abstain is False
    assert {e.canonical_name for e in out.entities} >= {"韩立", "三叔", "七玄门"}
    assert any(note.startswith("lpmm: 2/3 paragraphs") for note in out.notes)
    assert any("1 failed" in note for note in out.notes)


def test_lpmm_empty_text_abstains_without_calling_llm() -> None:
    llm = default_llm()
    out = LpmmEngine(llm=llm).extract(UNIT_ID, "   ", CARD, [])
    assert out.abstain is True
    assert "no_text" in " ".join(out.notes)
    assert llm.calls == []


# ── factory / provider wiring ───────────────────────────────────


def test_factory_selects_engines_and_lpmm_is_usable() -> None:
    from cbb_stage1.engines import build_engine as lazy_build_engine

    assert isinstance(lazy_build_engine("mock"), MockEngine)

    llm = default_llm()
    lpmm = build_engine("lpmm", llm=llm, checkpoint_dir=None)
    assert isinstance(lpmm, LpmmEngine)
    assert lpmm.name == "lpmm"
    assert LpmmAdapter is LpmmEngine
    assert hasattr(lpmm, "extract")
    assert lpmm.extract(UNIT_ID, CHAPTER_TEXT, CARD, []).unit_id == UNIT_ID

    ai_reader = build_engine("ai_reader", llm=default_llm())
    assert isinstance(ai_reader, AiReaderEngine)
    assert ai_reader.name == "ai-reader"
    assert ai_reader.extraction_mode == "chapter_fact"

    openie_engine = build_engine("openie", llm=default_llm())
    assert isinstance(openie_engine, AiReaderEngine)
    assert openie_engine.extraction_mode == "openie"
    assert openie_engine.name == "ai-reader-openie"

    with pytest.raises(EngineError):
        build_engine("not-a-provider")


# ── OpenIE switchable mode ──────────────────────────────────────


def test_openie_mode_is_switchable_and_maps_to_engine_output() -> None:
    llm = default_llm()
    engine = AiReaderEngine(llm=llm, extraction_mode="openie")

    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert out.abstain is False
    assert out.entities and all(e.type == "unknown" for e in out.entities)
    assert out.assertions
    for assertion in out.assertions:
        assert assertion.status == "needs_evidence"
        assert assertion.roles["paragraph_span"]
    assert any(note.startswith("openie: 3 paragraphs") for note in out.notes)
    assert any("Evidence stage must backfill quotes" in note for note in out.notes)
    # Two stages per paragraph: 3 NER + 3 RDF calls.
    assert len(llm.calls) == 6


def test_openie_mode_abstains_on_llm_failure() -> None:
    llm = default_llm(fail_markers={"第一段", "第二段", "第三段"})
    engine = AiReaderEngine(llm=llm, extraction_mode="openie")

    out = engine.extract(UNIT_ID, CHAPTER_TEXT, CARD, [])

    assert out.abstain is True
    assert out.entities == []
    assert any("openie: extraction failed" in note for note in out.notes)


def test_openie_mode_can_be_constructed_with_extractor_and_validation() -> None:
    llm = default_llm()
    extractor = OpenIEExtractor(llm, retries=0)
    engine = AiReaderEngine(llm=llm, extraction_mode="openie", openie=extractor)
    assert engine.openie is extractor

    with pytest.raises(ValueError):
        AiReaderEngine(llm=llm, extraction_mode="nope")


def test_openie_filters_invalid_entities_and_triples() -> None:
    assert filter_entities(["你", "", "  ", "韩立", "韩立", 42]) == ["韩立"]
    assert filter_triples(
        [
            ["A", "B", "C"],
            ["A", "B", "C"],
            ["A", "", "C"],
            ["A", "B"],
            ("D", "E", "F"),
        ]
    ) == [["A", "B", "C"], ["D", "E", "F"]]
    with pytest.raises(OpenIEError):
        filter_entities("not-a-list")


def test_lpmm_sync_extract_inside_running_loop() -> None:
    async def _inner() -> str:
        llm = default_llm(delay=0.0)
        return LpmmEngine(llm=llm).extract(UNIT_ID, CHAPTER_TEXT, CARD, []).unit_id

    assert asyncio.run(_inner()) == UNIT_ID
