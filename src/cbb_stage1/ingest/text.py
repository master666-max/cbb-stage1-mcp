"""Plain-text / Markdown parsing into the same ``ParsedBook`` shape as EPUB.

Novels in the v6 corpus also arrive as ``.txt`` / ``.md``. Chapter boundaries
are, in priority order:

1. Markdown headings (``# ...``; if none, ``## ...``);
2. Chinese chapter headings (``第一章 ...``, ``第12回 ...``);
3. the whole file as a single chapter.

The output dataclasses are shared with :mod:`cbb_stage1.ingest.epub`, so the
chunker and unit builder do not care which format produced the book.
"""

from __future__ import annotations

import re
from pathlib import Path

from .epub import (
    Chapter,
    ParsedBook,
    compute_content_hash,
    generate_book_id,
    normalize_text,
)

__all__ = ["parse_text"]

_MD_H1_RE = re.compile(r"^#\s+(.+?)\s*$")
_MD_ANY_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$")
_CN_CHAPTER_RE = re.compile(
    r"^\s*(第[0-9一二三四五六七八九十百千零两]+[章节回卷][^\n]{0,60})\s*$"
)


def _split_markdown(lines: list[str], pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []
    for line in lines:
        match = pattern.match(line)
        if match:
            if current_title is not None:
                sections.append((current_title, current_body))
            current_title = match.group(1).strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line)
    if current_title is not None:
        sections.append((current_title, current_body))
    return [(title, "\n".join(body)) for title, body in sections]


def parse_text(
    file_path: Path | str,
    *,
    book_id: str | None = None,
    title: str | None = None,
    author: str = "Unknown Author",
) -> ParsedBook:
    """Parse a UTF-8 ``.txt`` / ``.md`` file into a ``ParsedBook``.

    Args:
        file_path: Path to the text file.
        book_id: Optional explicit id; otherwise derived from title/author.
        title: Optional book title; otherwise the file stem.
        author: Optional author name.

    Returns:
        ``ParsedBook`` with ``format="text"``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    raw = file_path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    book_title = title or file_path.stem

    sections = _split_markdown(lines, _MD_H1_RE)
    if not sections:
        sections = _split_markdown(lines, _MD_ANY_RE)
    if not sections:
        # Chinese chapter headings: split while keeping the heading as title.
        current_title: str | None = None
        current_body: list[str] = []
        found: list[tuple[str, str]] = []
        for line in lines:
            if _CN_CHAPTER_RE.match(line):
                if current_title is not None:
                    found.append((current_title, "\n".join(current_body)))
                current_title = line.strip()
                current_body = []
            elif current_title is not None:
                current_body.append(line)
        if current_title is not None:
            found.append((current_title, "\n".join(current_body)))
        sections = found

    chapters: list[Chapter] = []
    if not sections:
        body = normalize_text(raw)
        if body:
            sections = [(book_title, raw)]

    for index, (chapter_title, body) in enumerate(sections):
        content = normalize_text(body)
        if not content:
            continue
        chapters.append(
            Chapter(
                index=len(chapters),
                title=chapter_title.strip() or f"Chapter {index + 1}",
                content=content,
                word_count=len(content.split()),
                href=file_path.name,
                nav_index=len(chapters),
            )
        )

    resolved_id = book_id or generate_book_id(book_title, author)
    full_text = "\n\n".join(ch.content for ch in chapters)
    return ParsedBook(
        file_path=file_path,
        title=book_title,
        author=author,
        book_id=resolved_id,
        content_hash=compute_content_hash(full_text),
        chapters=chapters,
        format="text",
    )
