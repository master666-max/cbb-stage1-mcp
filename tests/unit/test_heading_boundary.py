"""A heading must not be orphaned at the end of a chunk.

Measured gap (A13 ②)
--------------------
The default `structural` strategy splits on blank lines only (`chunk.py:300`). Headings are recognised
by `_HEADING_LINE` but only the `title` strategy uses them, so in the default path a heading line is just
another paragraph: the block boundary can fall immediately after it, leaving a chunk whose last line is
"第三章 出发" and whose text lives in the next chunk.

Two consequences, both real:

* the chunk that carries the heading has no content to extract from it, so the heading's own information
  (chapter/section titles are entity and event cues) is wasted;
* the chunk that carries the text has no heading, so a reader — or the model — cannot tell which section
  it belongs to. That is the "节级定位丢失" in the audit.

What the fix does, and does not
-------------------------------
The fix keeps a heading attached to the paragraph that follows it. It does **not** switch the default to
section-based chunking: the structural strategy's "big-first, only split when necessary" rule is a
v6 decision and changing it would alter chunk boundaries for every existing book. Only the orphan case is
prevented.
"""

from __future__ import annotations

from cbb_stage1.ingest.chunk import chunk_text


def _chapter(sections: int, body_chars: int = 400) -> str:
    parts: list[str] = ["# 第三章 出发"]
    for index in range(sections):
        parts.append(f"小节{index + 1}")
        parts.append("言万推开木门，看见拉法正在院中练剑。" * (body_chars // 18))
    return "\n\n".join(parts)


def _chunks(text: str, *, target: int = 300, max_chars: int = 600) -> list[str]:
    result = chunk_text(
        text,
        strategy="structural",
        target_chars=target,
        max_chars=max_chars,
    )
    return [chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in result]


def _ends_with_heading(chunk: str) -> bool:
    """True when the last non-empty line is a heading, i.e. it has been orphaned."""
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    if not lines:
        return False
    last = lines[-1]
    return bool(
        last.startswith("#")
        or last.startswith("第三章")
        or (len(last) <= 12 and not last.endswith(("。", "！", "？", "」")))
    )


# ══ the orphan case ═════════════════════════════════════════════════
def test_no_chunk_ends_with_an_orphaned_heading() -> None:
    chunks = _chunks(_chapter(sections=12))
    assert len(chunks) > 1, "the fixture did not produce multiple chunks"
    for index, chunk in enumerate(chunks):
        assert not _ends_with_heading(chunk), (
            f"chunk {index} ends with a heading, so the text it introduces is in the next chunk: "
            f"{chunk.strip()[-60:]!r}"
        )


def test_the_opening_heading_travels_with_its_body() -> None:
    chunks = _chunks(_chapter(sections=12))
    assert "第三章" in chunks[0], "the chapter heading was moved out of the first chunk"
    assert "言万" in chunks[0], "the first chunk carries the heading but none of its text"


def test_a_heading_before_an_oversized_paragraph_is_not_left_alone() -> None:
    """The case that actually orphans: a heading whose paragraph exceeds max_chars.

    Measured: with a heading followed by a 1080-character paragraph and max_chars=600, the paragraph is
    split away and the heading is emitted as its own 8-character chunk — a chunk with no text to extract
    from, and a body chunk with no idea which section it belongs to.
    """
    body = "言万推开木门，看见拉法正在院中练剑。" * 60
    text = "# 第三章 出发\n\n" + body + "\n\n尾声段落。"
    result = chunk_text(
        text, strategy="structural", target_chars=300, max_chars=600, overlap_chars=0
    )
    chunks = [chunk.text for chunk in result]
    assert len(chunks) > 1, "the fixture did not split"
    assert not any(
        len(chunk.strip()) <= 20 and chunk.strip().startswith("#") for chunk in chunks
    ), (
        "a heading was emitted as its own chunk: " + repr(chunks[0][:40])
    )


# ══ the defaults are unchanged ══════════════════════════════════════
def test_a_short_chapter_is_still_one_chunk() -> None:
    """The structural strategy's big-first rule must not be weakened."""
    text = "# 第一章 起\n\n言万遇见拉法。"
    chunks = _chunks(text, target=10_000, max_chars=10_000)
    assert len(chunks) == 1


def test_text_without_headings_chunks_as_before() -> None:
    text = "\n\n".join("言万推开木门。" * 30 for _ in range(8))
    chunks = _chunks(text)
    assert len(chunks) > 1
    assert all("言万" in chunk for chunk in chunks)


# ══ content is preserved ════════════════════════════════════════════
def test_no_text_is_lost_when_overlap_is_off() -> None:
    """With overlap disabled every passage must appear exactly once.

    Overlap is a deliberate feature (overlap_chars, default non-zero) that carries the previous chunk's
    tail into the next one, so duplication is expected by default. Turning it off isolates the question
    this test cares about: does the boundary logic itself drop anything?
    """
    text = _chapter(sections=6, body_chars=200)
    result = chunk_text(
        text, strategy="structural", target_chars=300, max_chars=600, overlap_chars=0
    )
    joined = "".join(chunk.text for chunk in result)
    assert joined.count("言万") == text.count("言万"), (
        "chunking lost or duplicated content with overlap disabled"
    )
