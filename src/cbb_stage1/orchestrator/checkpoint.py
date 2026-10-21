"""Checkpointing - safe incremental saves and resume for unit extraction.

Ported from B09 ``src/observability/checkpoint.py`` (read-only source). Shape is
preserved: ``ExtractionCheckpoint`` dataclass, interval-based saves, a
``CheckpointManager`` context manager that saves on error and clears on success.

Adaptations required by the v6 work layout:

* one checkpoint file per book under ``work/plan/`` (the source hard-coded a
  repo-global ``CHECKPOINT_FILE``); the store takes an explicit path, so tests
  and parallel books do not share state;
* the unit of progress is a **unit** (chapter or chunk), matching
  ``sources/units.jsonl`` rather than a chapter title.
"""

from __future__ import annotations

from typing import Literal

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

__all__ = [
    "DEFAULT_CHECKPOINT_INTERVAL",
    "ExtractionCheckpoint",
    "CheckpointStore",
    "CheckpointManager",
]

DEFAULT_CHECKPOINT_INTERVAL = 5


@dataclass
class ExtractionCheckpoint:
    """Checkpoint state for an in-progress extraction.

    Attributes:
        book_id: Deterministic book slug.
        book_title: Human-readable book title.
        total_units: Number of units selected for this run.
        completed_units: Unit ids already extracted.
        errors: Error messages recorded while the run was failing.
        last_updated: ISO-8601 timestamp of the last save.
    """

    book_id: str
    book_title: str
    total_units: int
    completed_units: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def progress_percent(self) -> float:
        if self.total_units == 0:
            return 0.0
        return (len(self.completed_units) / self.total_units) * 100

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionCheckpoint":
        return cls(**data)


class CheckpointStore:
    """Persist one book's checkpoint at an explicit path (atomic write)."""

    def __init__(self, path: str | Path, interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        self.path = Path(path)
        self.interval = max(1, int(interval))

    def load(self, book_id: str) -> ExtractionCheckpoint | None:
        """Load the checkpoint if it exists and belongs to ``book_id``."""
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("book_id") != book_id:
            return None
        try:
            return ExtractionCheckpoint.from_dict(data)
        except TypeError:
            return None

    def save(self, checkpoint: ExtractionCheckpoint) -> Path:
        """Persist ``checkpoint`` atomically."""
        from .files import atomic_write_json

        checkpoint.last_updated = datetime.now().isoformat()
        return atomic_write_json(self.path, checkpoint.to_dict())

    def clear(self) -> None:
        """Delete the checkpoint file (called after a successful run)."""
        if self.path.exists():
            self.path.unlink()

    def should_save(self, units_completed: int) -> bool:
        """True when ``units_completed`` is a positive multiple of ``interval``."""
        return units_completed > 0 and units_completed % self.interval == 0


class CheckpointManager:
    """Context manager for checkpointed extraction.

    Usage::

        with CheckpointManager(store, book_id, title, total_units) as mgr:
            for unit in units:
                if mgr.is_unit_done(unit.unit_id):
                    continue
                extract(unit)
                mgr.mark_complete(unit.unit_id)

    On exception the checkpoint is saved for recovery; on success it is
    cleared, because a completed run needs no resume state.
    """

    def __init__(
        self,
        store: CheckpointStore,
        book_id: str,
        book_title: str,
        total_units: int,
    ):
        self.store = store
        self.book_id = book_id
        self.book_title = book_title
        self.total_units = total_units
        self.checkpoint: ExtractionCheckpoint | None = None

    def __enter__(self) -> "CheckpointManager":
        self.checkpoint = self.store.load(self.book_id)
        if self.checkpoint is not None:
            done = len(self.checkpoint.completed_units)
            print(f"[checkpoint] resuming: {done}/{self.total_units} units done")
            # A resumed run speaks for the current unit plan; keep the larger total.
            self.checkpoint.total_units = max(self.checkpoint.total_units, self.total_units)
        else:
            self.checkpoint = ExtractionCheckpoint(
                book_id=self.book_id,
                book_title=self.book_title,
                total_units=self.total_units,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:  # noqa: vulture - __exit__ signature
        if self.checkpoint is None:
            return False
        if exc_val is not None:
            self.checkpoint.errors.append(str(exc_val))
            self.store.save(self.checkpoint)
            print("[checkpoint] error - saved for recovery")
        else:
            self.store.clear()
        return False

    def _checkpoint(self) -> "ExtractionCheckpoint":
        """The live checkpoint, creating it if absent.

        ``__exit__`` already treats a missing checkpoint as a normal state, so the
        rest of the class must too. Reaching into ``self.checkpoint`` directly was
        a latent None-dereference (mypy flagged six of them).
        """
        if self.checkpoint is None:
            self.checkpoint = ExtractionCheckpoint(
                book_id=self.book_id,
                book_title=self.book_title,
                total_units=self.total_units,
            )
        return self.checkpoint

    def is_unit_done(self, unit_id: str) -> bool:
        """True when ``unit_id`` is already recorded as complete."""
        return unit_id in self._checkpoint().completed_units

    def mark_complete(self, unit_id: str) -> None:
        """Mark a unit complete and save at the configured interval."""
        checkpoint = self._checkpoint()
        if unit_id in checkpoint.completed_units:
            return
        checkpoint.completed_units.append(unit_id)
        if self.store.should_save(len(checkpoint.completed_units)):
            self.store.save(checkpoint)

    @property
    def completed_count(self) -> int:
        return len(self._checkpoint().completed_units)

    def get_completed_units(self) -> list[str]:
        return list(self._checkpoint().completed_units)
