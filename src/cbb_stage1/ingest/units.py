"""Units - struct-level text units with narrator / reliability field slots.

A unit is the extraction granularity recorded in ``sources/units.jsonl``
(``01-箱规格-v2.md`` §5.2). This module turns ``ParsedBook`` chapters into
units: chapters that fit the engine window stay one unit (big-first), longer
chapters become several chunks with tail overlap (see :mod:`.chunk`).

The unit record keeps the narrator slot required by v2.2:

* ``narrator.kind`` ∈ ``first_person / third_person / third_limited / unknown``
  with ``entity_id`` (always ``None`` here; identity is resolved downstream) and
  ``name``;
* ``narrator_reliability`` ∈ ``reliable / unreliable / unknown``. The value is
  a **slot**: narration detection here only fills ``narrator.kind``
  heuristically, reliability defaults to ``unknown`` until L2 summary / human
  labelling sets it (per ``01`` §5.16). Nothing here claims a chapter is
  reliable.

Unit ids follow ``^u:[a-z0-9_-]+:v[0-9]{2}:c[0-9]{3}(:s[0-9]{2})?$``; a chapter
split into several chunks uses the bare chapter id for the first chunk and
``:s01``, ``:s02``… for the rest, so the chapter-level id stays stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from cbb_stage1.ids import sha256_text as _ids_sha256_text
from cbb_stage1.ids import unit_id as _ids_unit_id

from .chunk import CHARS_PER_TOKEN, chunk_chapter
from .epub import ParsedBook

__all__ = [
    "BOOK_ID_RE",
    "UNIT_ID_RE",
    "NARRATIVE_MODES",
    "NARRATOR_KINDS",
    "NARRATOR_RELIABILITIES",
    "Unit",
    "detect_narrator",
    "select_chapters",
    "make_unit_id",
    "make_text_path",
    "build_units",
]

BOOK_ID_RE = re.compile(r"^[a-z0-9_-]+$")
UNIT_ID_RE = re.compile(r"^u:[a-z0-9_-]+:v[0-9]{2}:c[0-9]{3}(:s[0-9]{2})?$")

NARRATIVE_MODES = ("present", "flashback", "prophecy", "mixed")
NARRATOR_KINDS = ("first_person", "third_person", "third_limited", "unknown")
NARRATOR_RELIABILITIES = ("reliable", "unreliable", "unknown")

#: First-person pronouns/determiners used by the kind heuristic.
_FIRST_PERSON_RE = re.compile(r"[我俺咱]")


def _validate_book_id(book_id: str) -> str:
    if not isinstance(book_id, str) or not BOOK_ID_RE.match(book_id):
        raise ValueError(
            f"invalid book_id {book_id!r}: must match {BOOK_ID_RE.pattern} "
            "(ascii lowercase letters, digits, '_' or '-')"
        )
    return book_id


def make_unit_id(book_id: str, volume: int, chapter: int, seq: int = 0) -> str:
    """Build a schema-valid unit id (compatibility wrapper).

    Delegates to :func:`cbb_stage1.ids.unit_id` - the single formula source for
    every contract ID.  The old name is kept for existing call sites/tests.

    Args:
        book_id: Slug matching ``^[a-z0-9_-]+$``.
        volume: 1-based volume (1-99; schema uses two digits).
        chapter: 1-based chapter (1-999; schema uses three digits).
        seq: Zero-based chunk index within the chapter; 0 omits the ``:sNN``
            suffix so the chapter-level id stays the stable as-of key.

    Raises:
        ValueError: On out-of-range parts or an invalid book id.
    """
    return _ids_unit_id(book_id, volume, chapter, seq)


def make_text_path(volume: int, chapter: int, seq: int = 0) -> str:
    """Relative work path for a unit's text file (under the work root)."""
    suffix = "" if seq == 0 else f"_s{seq:02d}"
    return f"plan/texts/v{int(volume):02d}/u{int(chapter):03d}{suffix}.txt"


def text_sha256(text: str) -> str:
    """Return ``sha256:<64 hex>`` for exact unit text (delegates to ids)."""
    return _ids_sha256_text(text)


def detect_narrator(text: str) -> dict:
    """Heuristically fill the ``narrator`` slot for one unit.

    This is a *kind* heuristic only (first-person pronoun density). It never
    claims reliability; ``narrator_reliability`` is a separate slot owned by
    L2/human labels. Ambiguous density maps to ``unknown`` rather than guessing.
    """
    if not text:
        return {"kind": "unknown", "entity_id": None, "name": None}
    ratio = len(_FIRST_PERSON_RE.findall(text)) / max(1, len(text))
    if ratio >= 0.02:
        return {"kind": "first_person", "entity_id": None, "name": "我"}
    if ratio >= 0.005:
        return {"kind": "unknown", "entity_id": None, "name": None}
    return {"kind": "third_person", "entity_id": None, "name": "叙述者"}


def select_chapters(
    selector: str | int | None,
    available: list[int],
) -> list[int]:
    """Resolve a chapter selector against available 1-based chapter numbers.

    Supported forms: ``None`` / ``"all"``, ``"3"``, ``"1,3,5"``, ``"1-3"``,
    and combinations such as ``"1-3,7"``.

    Raises:
        ValueError: On malformed selectors or no intersection with ``available``.
    """
    available_set = sorted({int(n) for n in available})
    if not available_set:
        raise ValueError("no chapters available to select")

    if selector is None or (isinstance(selector, str) and selector.strip().lower() in ("", "all")):
        return available_set

    if isinstance(selector, int):
        requested = [int(selector)]
    elif isinstance(selector, str):
        requested = []
        for token in selector.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                left, _, right = token.partition("-")
                try:
                    start, end = int(left), int(right)
                except ValueError as exc:
                    raise ValueError(f"invalid chapter range {token!r}") from exc
                if start < 1 or end < start:
                    raise ValueError(f"invalid chapter range {token!r}")
                requested.extend(range(start, end + 1))
            else:
                try:
                    number = int(token)
                except ValueError as exc:
                    raise ValueError(f"invalid chapter selector {token!r}") from exc
                if number < 1:
                    raise ValueError(f"chapter numbers are 1-based: {token!r}")
                requested.append(number)
    else:
        raise ValueError(f"unsupported chapter selector type: {type(selector)!r}")

    selected = sorted({n for n in requested if n in set(available_set)})
    if not selected:
        high = available_set[-1]
        raise ValueError(
            f"chapter selector {selector!r} matched no chapters "
            f"(available: 1-{high})"
        )
    return selected


@dataclass
class Unit:
    """One extraction unit (schema v2.2 §5.2 shape + internal work fields)."""

    unit_id: str
    book_id: str
    volume: int
    chapter: int
    title: str
    order: int
    prev: str | None
    next: str | None
    text_path: str
    chars: int
    sha256: str
    narrative_mode: str
    narrator: dict
    narrator_reliability: str
    source_locator: dict
    scenes: list[str]
    created_at: str
    text: str = field(repr=False)
    seq: int = 0
    chapter_unit_id: str = ""
    split_reason: str = "big_first"

    def to_record(self) -> dict:
        """Serialize exactly the ``sources/units.jsonl`` fields (no internals)."""
        return {
            "unit_id": self.unit_id,
            "book_id": self.book_id,
            "volume": self.volume,
            "chapter": self.chapter,
            "title": self.title,
            "order": self.order,
            "prev": self.prev,
            "next": self.next,
            "text_path": self.text_path,
            "chars": self.chars,
            "sha256": self.sha256,
            "narrative_mode": self.narrative_mode,
            "narrator": self.narrator,
            "narrator_reliability": self.narrator_reliability,
            "source_locator": self.source_locator,
            "scenes": self.scenes,
            "created_at": self.created_at,
        }


def build_units(
    book: ParsedBook,
    work_root: str | Path,
    *,
    book_id: str | None = None,
    chapters: str | int | None = None,
    chunk_target_tokens: int = 1200,
    chunk_overlap_tokens: int = 100,
    chunk_max_tokens: int = 8000,
    chars_per_token: float = CHARS_PER_TOKEN,
    chunk_strategy: str = "structural",
    narrative_mode: str = "present",
    narrator_reliability: str = "unknown",
    narrator: dict | None = None,
) -> list[Unit]:
    """Build ordered units for ``book`` and write their text files under work.

    Args:
        book: Parsed book (EPUB or text).
        work_root: Work root (``Config.work_dir``); texts go to
            ``<work_root>/plan/texts/vNN/uNNN.txt``.
        book_id: Override for the book id (``Config.book_id``); defaults to the
            parser-derived id.
        chapters: Chapter selector passed to :func:`select_chapters`.
        chunk_target_tokens / chunk_overlap_tokens: chunk sizing knobs.
        chunk_max_tokens: engine-window guard; chapters above it are split.
        chunk_strategy: chunking strategy (``Config.chunk_strategy``):
            ``structural`` / ``fixed`` / ``title`` / ``semantic``.
        narrative_mode: Slot value for every unit (default ``present``).
        narrator_reliability: Slot value for every unit; default ``unknown``.
        narrator: Optional explicit narrator dict override for all units.

    Returns:
        Units ordered by ``order`` with closed ``prev``/``next`` links.

    Raises:
        ValueError: On invalid book id / narrator slot values / empty selection.
    """
    if narrative_mode not in NARRATIVE_MODES:
        raise ValueError(f"invalid narrative_mode {narrative_mode!r}")
    if narrator_reliability not in NARRATOR_RELIABILITIES:
        raise ValueError(f"invalid narrator_reliability {narrator_reliability!r}")
    if narrator is not None and narrator.get("kind") not in NARRATOR_KINDS:
        raise ValueError(f"invalid narrator.kind {narrator.get('kind')!r}")
    if not book.chapters:
        raise ValueError(f"book has no chapters: {book.file_path}")

    resolved_book_id = _validate_book_id(book_id or book.book_id)
    available = [chapter.index + 1 for chapter in book.chapters]
    selected = select_chapters(chapters, available)

    work_root = Path(work_root)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    units: list[Unit] = []
    chapter_by_number = {chapter.index + 1: chapter for chapter in book.chapters}

    for chapter_number in selected:
        chapter = chapter_by_number[chapter_number]
        chunks = chunk_chapter(
            chapter.content,
            target_tokens=chunk_target_tokens,
            overlap_tokens=chunk_overlap_tokens,
            max_tokens=chunk_max_tokens,
            chars_per_token=chars_per_token,
            strategy=chunk_strategy,
        )
        if not chunks:
            continue
        chapter_unit_id = make_unit_id(resolved_book_id, 1, chapter_number, 0)
        for chunk in chunks:
            unit_id = make_unit_id(resolved_book_id, 1, chapter_number, chunk.seq)
            relative_path = make_text_path(1, chapter_number, chunk.seq)
            text_file = work_root / relative_path
            text_file.parent.mkdir(parents=True, exist_ok=True)
            text_file.write_text(chunk.text, encoding="utf-8", newline="\n")

            units.append(
                Unit(
                    unit_id=unit_id,
                    book_id=resolved_book_id,
                    volume=1,
                    chapter=chapter_number,
                    title=chapter.title,
                    order=0,  # filled below, monotonically
                    prev=None,
                    next=None,
                    text_path=relative_path,
                    chars=len(chunk.text),
                    sha256=text_sha256(chunk.text),
                    narrative_mode=narrative_mode,
                    narrator=dict(narrator) if narrator is not None else detect_narrator(chunk.text),
                    narrator_reliability=narrator_reliability,
                    source_locator={
                        "href": chapter.href,
                        "nav_index": chapter.nav_index,
                    },
                    scenes=[],
                    created_at=created_at,
                    text=chunk.text,
                    seq=chunk.seq,
                    chapter_unit_id=chapter_unit_id,
                    split_reason=chunk.split_reason,
                )
            )

    for index, unit in enumerate(units):
        unit.order = index + 1
        unit.prev = units[index - 1].unit_id if index > 0 else None
        unit.next = units[index + 1].unit_id if index + 1 < len(units) else None

    return units
