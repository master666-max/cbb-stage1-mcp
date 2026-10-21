"""Book loader - dispatch EPUB vs plain text into one ``ParsedBook`` shape."""

from __future__ import annotations

from pathlib import Path

from .epub import ParsedBook, parse_epub
from .text import parse_text

__all__ = ["load_book"]

_EPUB_SUFFIXES = {".epub"}
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}


def load_book(
    path: str | Path,
    *,
    book_id: str | None = None,
    title: str | None = None,
    author: str = "Unknown Author",
) -> ParsedBook:
    """Load a book from ``.epub`` / ``.txt`` / ``.md``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: For unsupported suffixes.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"book file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in _EPUB_SUFFIXES:
        return parse_epub(file_path)
    if suffix in _TEXT_SUFFIXES:
        return parse_text(file_path, book_id=book_id, title=title, author=author)
    raise ValueError(
        f"unsupported book format {suffix!r} for {file_path} "
        "(supported: .epub, .txt, .md)"
    )
