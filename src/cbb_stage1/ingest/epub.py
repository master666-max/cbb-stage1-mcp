"""EPUB parsing - chapters and text, ported from B09 ``src/ingestion/epub_parser.py``.

The source is read-only reference. The public shape is preserved
(``Chapter`` / ``ParsedBook`` / ``normalize_text`` / ``html_to_text`` /
``extract_chapter_title`` / ``generate_book_id`` / ``compute_content_hash`` /
``parse_epub``). Two deliberate deviations:

* ``html_to_text`` no longer double-counts text when block elements are nested
  (the source selected both ``div`` and its child ``p``, duplicating body text);
* ``generate_book_id`` falls back to a hash-based id when the title/author are
  non-Latin, so the id still satisfies ``^[a-z0-9_-]+$`` from
  ``schemas/unit.schema.json``.

No Ollama / FastAPI imports. EPUB files are read with ``ebooklib`` (declared in
``pyproject.toml``) and HTML is flattened with ``beautifulsoup4``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from cbb_stage1.ids import sha256_hex_text

__all__ = [
    "Chapter",
    "ParsedBook",
    "normalize_text",
    "html_to_text",
    "extract_chapter_title",
    "generate_book_id",
    "compute_content_hash",
    "parse_epub",
]

#: Block-level tags whose text becomes a paragraph.
_BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div"]

#: Chapters shorter than this are treated as navigation/metadata (source rule).
_MIN_CHAPTER_CHARS = 100


@dataclass
class Chapter:
    """A chapter extracted from an EPUB.

    Attributes:
        index: Zero-based reading-order position of the chapter.
        title: Chapter title (from metadata, heading, or generated fallback).
        content: Plain text content, whitespace-normalized.
        word_count: Whitespace-separated token count (kept from the source).
        href: Document path inside the EPUB container.
        nav_index: Zero-based navigation index (used in ``source_locator``).
    """

    index: int
    title: str
    content: str
    word_count: int
    href: str = ""
    nav_index: int = 0

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class ParsedBook:
    """A fully parsed book (EPUB or plain text).

    Attributes:
        file_path: Filesystem path to the source file.
        title: Book title.
        author: Author name.
        book_id: Deterministic slug derived from title + author.
        content_hash: SHA-256 digest of all chapter content.
        chapters: Ordered list of parsed chapters.
        format: Source format (``epub`` or ``text``).
    """

    file_path: Path
    title: str
    author: str
    book_id: str
    content_hash: str
    chapters: list[Chapter] = field(default_factory=list)
    format: str = "epub"

    @property
    def full_text(self) -> str:
        """Concatenate all chapter content into a single string."""
        return "\n\n".join(ch.content for ch in self.chapters)

    @property
    def total_words(self) -> int:
        """Total whitespace-separated token count across all chapters."""
        return sum(ch.word_count for ch in self.chapters)

    @property
    def total_chars(self) -> int:
        """Total character count across all chapters."""
        return sum(ch.char_count for ch in self.chapters)


def normalize_text(text: str) -> str:
    """Normalize whitespace and unicode (NFC) while preserving paragraphs."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html_content: str) -> str:
    """Convert XHTML content to paragraph-separated plain text.

    Strips ``script`` / ``style`` / ``nav`` / ``header`` / ``footer``. Text is
    taken from the innermost block elements only, so ``<div><p>x</p></div>``
    yields ``x`` once (the source shell duplicated it).
    """
    soup = BeautifulSoup(html_content, "html.parser")

    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()

    text_parts: list[str] = []
    for tag in soup.find_all(_BLOCK_TAGS):
        # Skip wrappers: their block children are emitted instead.
        if tag.find(_BLOCK_TAGS) is not None:
            continue
        text = tag.get_text(separator=" ", strip=True)
        if text:
            text_parts.append(text)

    if not text_parts:
        fallback = soup.get_text(separator=" ", strip=True)
        if fallback:
            text_parts = [fallback]

    return "\n\n".join(text_parts)


def extract_chapter_title(item: epub.EpubItem, index: int) -> str:
    """Extract a chapter title from an EPUB item.

    Tries the item ``title``, then ``h1``-``h3`` / ``title`` tags, then falls
    back to ``Chapter {index + 1}``.
    """
    title = getattr(item, "title", None)
    if title:
        return str(title)

    raw = item.get_content().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in ("h1", "h2", "h3", "title"):
        heading = soup.find(tag)
        if heading:
            text = heading.get_text(strip=True)
            if text and len(text) < 100:
                return text

    return f"Chapter {index + 1}"


def generate_book_id(title: str, author: str) -> str:
    """Generate a deterministic, URL-safe book id from title and author.

    Keeps ``[a-z0-9]`` and converts spaces to underscores. If the result is
    empty (e.g. a pure-CJK title), returns ``book_<hash10>`` so the id still
    matches the box schema pattern ``^[a-z0-9_-]+$``.
    """
    combined = f"{title}_{author}".lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", combined)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")[:50]
    if not cleaned:
        digest = sha256_hex_text(f"{title}_{author}", 10)
        cleaned = f"book_{digest}"
    return cleaned


def compute_content_hash(text: str) -> str:
    """Return the full SHA-256 hex digest of ``text``."""
    return sha256_hex_text(text)


def _first_metadata(book: epub.EpubBook, field_name: str) -> str | None:
    values = book.get_metadata("DC", field_name)
    if values:
        value = values[0][0]
        if value:
            return str(value).strip()
    return None


def parse_epub(file_path: Path | str) -> ParsedBook:
    """Parse an EPUB file and extract structured chapters.

    Args:
        file_path: Path to the ``.epub`` file.

    Returns:
        ``ParsedBook`` with metadata, ordered chapters, and content hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not an ``.epub``.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {file_path}")
    if file_path.suffix.lower() != ".epub":
        raise ValueError(f"Not an EPUB file: {file_path}")

    book = epub.read_epub(str(file_path), options={"ignore_ncx": True})

    title = _first_metadata(book, "title") or "Unknown Title"
    author = _first_metadata(book, "creator") or "Unknown Author"
    book_id = generate_book_id(title, author)

    chapters: list[Chapter] = []
    chapter_index = 0

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        html_content = item.get_content().decode("utf-8", errors="ignore")
        normalized = normalize_text(html_to_text(html_content))
        # Drop navigation / metadata documents (source rule).
        if len(normalized) < _MIN_CHAPTER_CHARS:
            continue

        chapters.append(
            Chapter(
                index=chapter_index,
                title=extract_chapter_title(item, chapter_index),
                content=normalized,
                word_count=len(normalized.split()),
                href=item.get_name(),
                nav_index=chapter_index,
            )
        )
        chapter_index += 1

    full_text = "\n\n".join(ch.content for ch in chapters)
    return ParsedBook(
        file_path=file_path,
        title=title,
        author=author,
        book_id=book_id,
        content_hash=compute_content_hash(full_text),
        chapters=chapters,
        format="epub",
    )
