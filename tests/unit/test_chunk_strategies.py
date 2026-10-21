"""Chunk strategy tests (A23: structural / fixed / title / semantic).

Invariants checked per strategy:
* ``structural``: big-first for short chapters, paragraph aggregation +
  sentence fallback, reconstruction via ``"\\n\\n".join(body)``;
* ``fixed``: fixed windows of ``target_chars`` with overlap, hard cuts allowed,
  reconstruction via ``"".join(body)`` when overlap is 0;
* ``title``: section boundaries at headings, headings never cut from their
  section, reconstruction via ``"".join(body)``; no-heading fallback;
* ``semantic``: lexical (character-trigram) clustering groups same-topic
  paragraphs and splits on topic change; reconstruction via
  ``"\\n\\n".join(body)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cbb_stage1.ingest.chunk import (
    CHUNK_STRATEGIES,
    Chunk,
    chunk_chapter,
    chunk_text,
    normalize_paragraphs,
)
from cbb_stage1.ingest.epub import Chapter, ParsedBook
from cbb_stage1.ingest.units import build_units


def _structural_text(paragraphs: int = 12) -> str:
    return "\n\n".join(f"第{i}段。" + "内容" * 40 for i in range(paragraphs))


def _topic_text() -> str:
    combat = [
        "韩立手握长剑站在山巅，山风猎猎，剑光凛冽。",
        "韩立手握长剑站在山巅，山风呼啸，剑光凛冽。",
        "山巅之上韩立手握长剑，山风猎猎，剑光凛冽。",
    ]
    kitchen = [
        "三叔坐在灶台前翻炒青菜，饭菜香气弥漫院子。",
        "三叔坐在灶台前翻炒青菜，饭菜香气飘满院子。",
        "灶台前，三叔翻炒青菜，饭菜香气弥漫在院子。",
    ]
    return "\n\n".join(combat + kitchen)


def _paragraph(body: str) -> str:
    return body.replace("\n", " ")


# ── structural ──────────────────────────────────────────────────


def test_structural_big_first_and_reconstruction() -> None:
    short = "短章内容。" * 20
    chunks = chunk_text(short, target_chars=100, overlap_chars=10, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].split_reason == "big_first"
    assert chunks[0].strategy == "structural"

    text = _structural_text()
    chunks = chunk_text(text, target_chars=200, overlap_chars=40, max_chars=320)
    assert len(chunks) > 1
    assert "\n\n".join(chunk.body_text for chunk in chunks) == normalize_paragraphs(text)
    assert all(chunk.split_reason == "structural_aggregate" for chunk in chunks)
    assert all(chunk.strategy == "structural" for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.overlap_chars > 0
        assert current.text.startswith(previous.body_text[-current.overlap_chars:])


def test_structural_sentence_fallback_for_overlong_paragraph() -> None:
    long_paragraph = "字" * 600
    chunks = chunk_text(long_paragraph, target_chars=100, overlap_chars=20, max_chars=100)
    assert len(chunks) >= 2
    assert all(chunk.split_reason == "sentence_fallback" for chunk in chunks)
    assert "".join(chunk.body_text for chunk in chunks) == long_paragraph


# ── fixed ───────────────────────────────────────────────────────


def test_fixed_windows_without_overlap_reconstruct_exactly() -> None:
    text = "甲" * 1000
    chunks = chunk_text(text, target_chars=200, overlap_chars=0, max_chars=250, strategy="fixed")

    assert len(chunks) == 5
    assert "".join(chunk.body_text for chunk in chunks) == text
    assert all(chunk.strategy == "fixed" for chunk in chunks)
    assert all(chunk.split_reason == "fixed_window" for chunk in chunks)
    assert all(chunk.body_chars == 200 for chunk in chunks)


def test_fixed_windows_with_overlap_and_engine_window_guard() -> None:
    text = "乙" * 1000
    chunks = chunk_text(text, target_chars=200, overlap_chars=40, max_chars=250, strategy="fixed")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.body_chars <= 250
        assert chunk.body_text in text
    assert chunks[0].overlap_chars == 0
    for previous, current in zip(chunks, chunks[1:]):
        assert current.overlap_chars > 0
        assert current.overlap_chars <= 40
        assert current.text.startswith(previous.body_text[-current.overlap_chars:])
        assert current.text[current.overlap_chars:].lstrip("\n") == current.body_text


def test_fixed_merges_tiny_tail_window() -> None:
    text = "丙" * 260
    chunks = chunk_text(text, target_chars=200, overlap_chars=0, max_chars=300, strategy="fixed")
    assert len(chunks) == 1
    assert chunks[0].body_text == text


# ── title ───────────────────────────────────────────────────────


def test_title_splits_at_headings_without_cutting_them() -> None:
    text = (
        "开场段落，雪落长街，行人稀少。\n\n"
        "一、山雨欲来\n\n"
        + "山雨欲来风满楼，乌云压城，街头一片寂静。\n\n"
        "二、风起云涌\n\n"
        + "风起云涌之际，江湖各路人物齐聚长安。"
    )
    normalized = normalize_paragraphs(text)
    chunks = chunk_text(
        text, target_chars=30, overlap_chars=8, max_chars=200, strategy="title"
    )

    assert len(chunks) > 1
    assert all(chunk.strategy == "title" for chunk in chunks)
    assert all(chunk.split_reason == "title_section" for chunk in chunks)
    assert "".join(chunk.body_text for chunk in chunks) == normalized
    # A section heading starts its own chunk (never split from the section).
    for heading in ("一、山雨欲来", "二、风起云涌"):
        assert any(chunk.body_text.startswith(heading) for chunk in chunks), heading


def test_title_falls_back_to_structural_without_headings() -> None:
    text = _structural_text(paragraphs=8)
    chunks = chunk_text(text, target_chars=200, overlap_chars=20, max_chars=320, strategy="title")
    assert len(chunks) > 1
    assert all(chunk.strategy == "title" for chunk in chunks)
    assert all(
        chunk.split_reason in ("structural_aggregate", "sentence_fallback", "big_first")
        for chunk in chunks
    )
    assert "\n\n".join(chunk.body_text for chunk in chunks) == normalize_paragraphs(text)


# ── semantic ────────────────────────────────────────────────────


def test_semantic_groups_same_topic_and_splits_on_topic_change() -> None:
    text = _topic_text()
    normalized = normalize_paragraphs(text)
    chunks = chunk_text(
        text, target_chars=1800, overlap_chars=0, max_chars=2000, strategy="semantic"
    )

    assert len(chunks) == 2
    assert all(chunk.strategy == "semantic" for chunk in chunks)
    assert all(chunk.split_reason == "semantic_cluster" for chunk in chunks)
    assert "\n\n".join(chunk.body_text for chunk in chunks) == normalized
    first, second = chunks
    assert "长剑" in first.body_text and "灶台" not in first.body_text
    assert "灶台" in second.body_text and "长剑" not in second.body_text


def test_semantic_threshold_controls_cluster_granularity() -> None:
    text = _topic_text()
    coarse = chunk_text(
        text, target_chars=1800, overlap_chars=0, max_chars=2000,
        strategy="semantic", semantic_threshold=0.12,
    )
    fine = chunk_text(
        text, target_chars=1800, overlap_chars=0, max_chars=2000,
        strategy="semantic", semantic_threshold=0.99,
    )
    assert len(fine) > len(coarse)


def test_semantic_single_paragraph_falls_back_to_structural() -> None:
    text = "字" * 600
    chunks = chunk_text(
        text, target_chars=100, overlap_chars=0, max_chars=150, strategy="semantic"
    )
    assert chunks
    assert all(chunk.strategy == "semantic" for chunk in chunks)
    assert all(chunk.body_chars <= 150 for chunk in chunks)


# ── API surface / wiring ───────────────────────────────────────


def test_chunk_chapter_forwards_strategy_and_token_budget() -> None:
    text = "丁" * 1200
    chunks = chunk_chapter(
        text,
        target_tokens=100,
        overlap_tokens=10,
        max_tokens=200,
        strategy="fixed",
    )
    assert chunks
    assert all(chunk.strategy == "fixed" for chunk in chunks)
    assert all(chunk.body_chars <= int(round(200 * 1.5)) for chunk in chunks)
    assert chunks[0].body_chars == int(round(100 * 1.5))


def test_unknown_chunk_strategy_raises() -> None:
    assert CHUNK_STRATEGIES == ("structural", "fixed", "title", "semantic")
    with pytest.raises(ValueError):
        chunk_text("内容" * 50, strategy="nope")
    with pytest.raises(ValueError):
        chunk_chapter("内容" * 50, strategy="nope")


def test_chunk_to_dict_includes_strategy() -> None:
    chunk = chunk_text("内容" * 20, strategy="fixed")[0]
    assert isinstance(chunk, Chunk)
    payload = chunk.to_dict()
    assert payload["strategy"] == "fixed"
    assert payload["split_reason"] == "fixed_window"


def _parsed_book(content: str) -> ParsedBook:
    return ParsedBook(
        file_path=Path("synthetic.txt"),
        title="测试书",
        author="tester",
        book_id="testbook",
        content_hash="sha256:0",
        chapters=[
            Chapter(index=0, title="第一章 测试", content=content, word_count=len(content))
        ],
        format="text",
    )


def test_build_units_passes_chunk_strategy(tmp_path: Path) -> None:
    content = ("句子内容。" * 120)  # long enough to split at 100 tokens (150 chars)
    units = build_units(
        _parsed_book(content),
        tmp_path,
        book_id="testbook",
        chunk_target_tokens=30,
        chunk_overlap_tokens=5,
        chunk_max_tokens=100,
        chunk_strategy="fixed",
    )

    assert len(units) > 1
    assert all(unit.split_reason == "fixed_window" for unit in units)
    assert [unit.order for unit in units] == list(range(1, len(units) + 1))
    for index, unit in enumerate(units):
        text_file = tmp_path / unit.text_path
        assert text_file.exists()
        assert text_file.read_text(encoding="utf-8") == unit.text
        assert unit.prev == (units[index - 1].unit_id if index else None)
        assert unit.next == (units[index + 1].unit_id if index + 1 < len(units) else None)


def test_build_units_title_strategy_produces_title_sections(tmp_path: Path) -> None:
    content = (
        "楔子内容，交代背景。\n\n"
        "一、初入江湖\n\n" + "韩立步入江湖，前路未卜。" * 8 + "\n\n"
        "二、风起云涌\n\n" + "江湖风波骤起，各方势力登场。" * 8
    )
    units = build_units(
        _parsed_book(content),
        tmp_path,
        book_id="testbook",
        chunk_target_tokens=20,
        chunk_overlap_tokens=4,
        chunk_max_tokens=60,
        chunk_strategy="title",
    )
    assert any(unit.split_reason == "title_section" for unit in units)


def test_build_units_rejects_unknown_chunk_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_units(
            _parsed_book("内容" * 200),
            tmp_path,
            book_id="testbook",
            chunk_strategy="semantic-ish",
        )
