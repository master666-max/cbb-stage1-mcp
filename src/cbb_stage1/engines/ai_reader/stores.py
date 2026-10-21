"""Store protocols that replace B06's ``src.db`` SQLite layer.

The source modules read/write three tables directly:
``entity_dictionary`` (pre-scan candidates), ``chapter_facts`` (extracted facts),
and a status flag.  This module freezes those needs as two small Protocols and
ships deterministic adapters:

* :class:`InMemoryDictionaryStore` / :class:`InMemoryFactStore` — tests, CI,
  single-process smoke runs (default when no persistence is wired).
* :class:`JsonlDictionaryStore` / :class:`JsonlFactStore` — UTF-8 JSONL files
  under ``work/``, matching the design's "work is disposable JSONL" decision.

No SQLite, no Chroma, no FastAPI.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from cbb_stage1.engines.ai_reader.schema import ChapterFact, EntityDictEntry

PRESCAN_PENDING = "pending"
PRESCAN_RUNNING = "running"
PRESCAN_COMPLETED = "completed"
PRESCAN_FAILED = "failed"


@runtime_checkable
class DictionaryStore(Protocol):
    """The slice of ``entity_dictionary_store`` the engine actually uses."""

    def list_entries(self, book_id: str) -> list[EntityDictEntry]:
        """All pre-scan candidates for *book_id*, highest frequency first."""

    def replace_entries(self, book_id: str, entries: Sequence[EntityDictEntry]) -> None:
        """Replace the whole candidate set (source: delete_all + insert_batch)."""

    def get_status(self, book_id: str) -> str:
        """One of pending/running/completed/failed."""

    def set_status(self, book_id: str, status: str) -> None:
        """Record pre-scan progress."""


@runtime_checkable
class FactStore(Protocol):
    """The slice of ``chapter_fact_store`` the engine actually uses."""

    def list_facts(self, book_id: str) -> list[ChapterFact]:
        """All extracted chapter facts, ordered by chapter_id."""

    def save_fact(self, book_id: str, fact: ChapterFact) -> None:
        """Insert or replace the fact for ``fact.chapter_id``."""


# ── In-memory adapters (default) ─────────────────────────────────


class InMemoryDictionaryStore:
    """Process-local dictionary store; the zero-dependency default."""

    def __init__(self) -> None:
        self._entries: dict[str, list[EntityDictEntry]] = {}
        self._status: dict[str, str] = {}

    def list_entries(self, book_id: str) -> list[EntityDictEntry]:
        return sorted(
            self._entries.get(book_id, []),
            key=lambda e: e.frequency,
            reverse=True,
        )

    def replace_entries(self, book_id: str, entries: Sequence[EntityDictEntry]) -> None:
        self._entries[book_id] = list(entries)

    def get_status(self, book_id: str) -> str:
        return self._status.get(book_id, PRESCAN_PENDING)

    def set_status(self, book_id: str, status: str) -> None:
        self._status[book_id] = status


class InMemoryFactStore:
    """Process-local fact store keyed by ``(book_id, chapter_id)``."""

    def __init__(self, facts: Iterable[tuple[str, ChapterFact]] | None = None) -> None:
        self._facts: dict[tuple[str, int], ChapterFact] = {}
        for book_id, fact in facts or []:
            self._facts[(book_id, fact.chapter_id)] = fact

    def list_facts(self, book_id: str) -> list[ChapterFact]:
        rows = [fact for (bid, _chapter), fact in self._facts.items() if bid == book_id]
        return sorted(rows, key=lambda f: f.chapter_id)

    def save_fact(self, book_id: str, fact: ChapterFact) -> None:
        self._facts[(book_id, fact.chapter_id)] = fact


# ── JSONL adapters (work/ persistence) ───────────────────────────


class JsonlFactStore:
    """Append-only JSONL fact store (latest record per chapter wins on read)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list_facts(self, book_id: str) -> list[ChapterFact]:
        latest: dict[int, ChapterFact] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("book_id") != book_id:
                        continue
                    fact = ChapterFact.model_validate(record["fact"])
                    latest[fact.chapter_id] = fact
        return [latest[k] for k in sorted(latest)]

    def save_fact(self, book_id: str, fact: ChapterFact) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"book_id": book_id, "fact": fact.model_dump()}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class JsonlDictionaryStore:
    """Snapshot-per-book JSONL dictionary store (one line per ``replace``)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _read_latest(self, book_id: str) -> tuple[list[EntityDictEntry], str]:
        entries: list[EntityDictEntry] = []
        status = PRESCAN_PENDING
        if not self.path.exists():
            return entries, status
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("book_id") != book_id:
                    continue
                if "entries" in record:
                    entries = [
                        EntityDictEntry.model_validate(item) for item in record["entries"]
                    ]
                if "status" in record:
                    status = record["status"]
        return entries, status

    def list_entries(self, book_id: str) -> list[EntityDictEntry]:
        entries, _ = self._read_latest(book_id)
        return sorted(entries, key=lambda e: e.frequency, reverse=True)

    def _append(self, book_id: str, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"book_id": book_id, **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def replace_entries(self, book_id: str, entries: Sequence[EntityDictEntry]) -> None:
        self._append(book_id, {"entries": [e.model_dump() for e in entries]})

    def get_status(self, book_id: str) -> str:
        _, status = self._read_latest(book_id)
        return status

    def set_status(self, book_id: str, status: str) -> None:
        self._append(book_id, {"status": status})
