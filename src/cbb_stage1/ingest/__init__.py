"""Ingestion: book files -> chapters -> structural chunks -> extraction units."""

from __future__ import annotations

from .chunk import Chunk, chunk_chapter, chunk_text
from .epub import Chapter, ParsedBook, parse_epub
from .loader import load_book
from .text import parse_text
from .units import Unit, build_units, select_chapters

__all__ = [
    "Chapter",
    "ParsedBook",
    "Chunk",
    "Unit",
    "chunk_chapter",
    "chunk_text",
    "build_units",
    "load_book",
    "parse_epub",
    "parse_text",
    "select_chapters",
]
