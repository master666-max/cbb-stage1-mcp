"""Deterministic text splitting shared by the two-stage engines (LPMM / OpenIE).

Splits a unit into paragraph chunks while keeping **exact source offsets**, so
the Evidence stage can always fall back to ``text[start:end]``.  Splitting is
pure: blank-line paragraph boundaries first, then sentence-boundary chunking
for paragraphs longer than ``target_chars``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cbb_stage1.ids import sha256_hex_text

_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")


@dataclass(frozen=True)
class Paragraph:
    """A chunk with exact source span and content hash."""

    text: str
    start: int
    end: int
    sha256: str

    @property
    def idx(self) -> str:
        return self.sha256


def _sha256(text: str) -> str:
    """Bare hex fingerprint (delegates to ``cbb_stage1.ids``)."""
    return sha256_hex_text(text)


def _split_long_block(block: str) -> list[str]:
    """Split an oversized paragraph at sentence boundaries, packing greedily."""
    pieces = [match.group(0) for match in _SENTENCE.finditer(block)]
    if not pieces:
        return [block]
    return pieces


def split_paragraphs(
    text: str,
    *,
    target_chars: int = 800,
    min_chars: int = 1,
) -> list[Paragraph]:
    """Return paragraph chunks covering *text* with exact offsets.

    Blocks are separated by blank lines; a block longer than ``target_chars`` is
    packed from sentence pieces into chunks of at most ~``target_chars``.
    Whitespace-only blocks are dropped.  Offsets always satisfy
    ``paragraph.text == text[start:end]``.
    """
    if not text:
        return []

    blocks: list[tuple[int, int]] = []
    cursor = 0
    for match in _BLANK_LINE.finditer(text):
        blocks.append((cursor, match.start()))
        cursor = match.end()
    blocks.append((cursor, len(text)))

    paragraphs: list[Paragraph] = []
    for start, end in blocks:
        # Trim leading/trailing whitespace while keeping offsets honest.
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue

        block = text[start:end]
        if len(block) <= target_chars:
            chunks = [(start, end)]
        else:
            chunks = []
            offset = start
            buffer_start = start
            buffer_len = 0
            for piece in _split_long_block(block):
                if buffer_len and buffer_len + len(piece) > target_chars:
                    chunks.append((buffer_start, offset))
                    buffer_start = offset
                    buffer_len = 0
                offset += len(piece)
                buffer_len += len(piece)
            if buffer_start < offset:
                chunks.append((buffer_start, offset))

        for chunk_start, chunk_end in chunks:
            chunk_text = text[chunk_start:chunk_end]
            if len(chunk_text) < min_chars or not chunk_text.strip():
                continue
            paragraphs.append(
                Paragraph(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_end,
                    sha256=_sha256(chunk_text),
                )
            )

    return paragraphs
