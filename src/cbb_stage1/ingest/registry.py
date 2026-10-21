"""Ingestion registry - track processed books for incremental ingestion.

Ported from B09 ``src/ingestion/registry.py`` (read-only source). Shape is
preserved (``BookRecord`` / ``IngestionRegistry`` / load / save / create record).
Adaptations: stdlib ``json`` instead of ``orjson`` (not a declared dependency),
and the registry path is explicit rather than a repo-global config constant.

The registry lives under ``work/``: it is bookkeeping for incremental runs, not
box content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "BookRecord",
    "IngestionRegistry",
    "load_registry",
    "save_registry",
    "create_book_record",
]

#: Default location relative to the work root's parent (pipeline passes explicit).
DEFAULT_REGISTRY_PATH = Path("work/plan/ingestion_log.json")


@dataclass
class BookRecord:
    """Record of a processed book."""

    book_id: str
    file_path: str
    content_hash: str
    title: str
    author: str
    chapter_count: int
    word_count: int
    processed_at: str
    entities_extracted: int = 0


@dataclass
class IngestionRegistry:
    """Registry tracking all processed books."""

    processed_books: dict[str, BookRecord] = field(default_factory=dict)

    def is_processed(self, content_hash: str) -> bool:
        return any(rec.content_hash == content_hash for rec in self.processed_books.values())

    def get_record_by_hash(self, content_hash: str) -> BookRecord | None:
        for record in self.processed_books.values():
            if record.content_hash == content_hash:
                return record
        return None

    def get_record_by_id(self, book_id: str) -> BookRecord | None:
        return self.processed_books.get(book_id)

    def add_record(self, record: BookRecord) -> None:
        self.processed_books[record.book_id] = record

    def update_entity_count(self, book_id: str, count: int) -> None:
        if book_id in self.processed_books:
            self.processed_books[book_id].entities_extracted = count

    def to_dict(self) -> dict:
        return {
            "processed_books": {
                book_id: {
                    "book_id": record.book_id,
                    "file_path": record.file_path,
                    "content_hash": record.content_hash,
                    "title": record.title,
                    "author": record.author,
                    "chapter_count": record.chapter_count,
                    "word_count": record.word_count,
                    "processed_at": record.processed_at,
                    "entities_extracted": record.entities_extracted,
                }
                for book_id, record in self.processed_books.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IngestionRegistry":
        registry = cls()
        for _, record_data in data.get("processed_books", {}).items():
            registry.processed_books[record_data["book_id"]] = BookRecord(
                book_id=record_data["book_id"],
                file_path=record_data["file_path"],
                content_hash=record_data["content_hash"],
                title=record_data["title"],
                author=record_data["author"],
                chapter_count=record_data["chapter_count"],
                word_count=record_data["word_count"],
                processed_at=record_data["processed_at"],
                entities_extracted=record_data.get("entities_extracted", 0),
            )
        return registry


def load_registry(path: str | Path | None = None) -> IngestionRegistry:
    """Load the registry; missing/corrupt files yield an empty registry."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        return IngestionRegistry()
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        return IngestionRegistry.from_dict(data)
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        print(f"warning: cannot parse registry {registry_path}, starting fresh: {exc}")
        return IngestionRegistry()


def save_registry(
    registry: IngestionRegistry,
    path: str | Path | None = None,
) -> Path:
    """Persist the registry atomically (temp file + rename)."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(registry_path)
    return registry_path


def create_book_record(
    book_id: str,
    file_path: Path,
    content_hash: str,
    title: str,
    author: str,
    chapter_count: int,
    word_count: int,
) -> BookRecord:
    """Create a new book record stamped with the current time."""
    return BookRecord(
        book_id=book_id,
        file_path=str(file_path),
        content_hash=content_hash,
        title=title,
        author=author,
        chapter_count=chapter_count,
        word_count=word_count,
        processed_at=datetime.now().isoformat(),
    )
