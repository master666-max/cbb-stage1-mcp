"""Summary-tree interface for the derived layer (no LLM calls).

Implements the v2.1 summary-tree shape from ``01-箱规格-v2`` §5.14: one summary
row per target (``scene/chapter/volume/book``) with ``parent_summary_id`` links,
``level`` from 0, and ``stale`` marked by code when the subtree changed.

Stage 1 deliberately does **not** synthesise text: with no ``unit_summaries``
mapping every row carries ``text=None`` and ``status="derived"``. That is the
interface skeleton; LLM summarisation belongs to a later pass and must fill
``text`` without changing the tree shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from cbb_stage1 import ids

__all__ = [
    "SUMMARY_CONTRACT",
    "SUMMARY_SCOPES",
    "SummaryNode",
    "make_summary_id",
    "build_summary_tree",
    "summary_rows",
    "validate_summary_tree",
    "mark_stale",
]

#: Contract marker for ``derived/summary_tree.json``.
SUMMARY_CONTRACT = "cbb-summary-tree/1"

#: Closed set from box spec §5.14.
SUMMARY_SCOPES = ("scene", "chapter", "volume", "book")


def make_summary_id(book_id: str, scope: str, target: str) -> str:
    """Content-addressed summary id: ``sm:{sha16(book|scope|target)}``.

    Compatibility wrapper over :func:`cbb_stage1.ids.summary_id`; the byte
    formula is unchanged and the scope stays a closed set here.
    """
    if scope not in SUMMARY_SCOPES:
        raise ValueError(f"unknown summary scope: {scope!r}")
    return ids.summary_id(book_id, scope, target)


@dataclass(frozen=True)
class SummaryNode:
    """One node of the summary tree (mirrors a ``derived/summaries.jsonl`` row)."""

    summary_id: str
    book_id: str
    scope: str
    target: str
    level: int
    parent_summary_id: str | None = None
    cluster_id: str | None = None
    text: str | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    status: str = "derived"
    stale: bool = False
    produced_by: str = "box:derive"

    def __post_init__(self) -> None:
        if self.scope not in SUMMARY_SCOPES:
            raise ValueError(f"unknown summary scope: {self.scope!r}")
        if self.level < 0:
            raise ValueError("summary level must be >= 0")
        if self.parent_summary_id == self.summary_id:
            raise ValueError("summary cannot be its own parent")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("summary text must be str or None")

    def to_record(self) -> dict[str, Any]:
        """Return the §5.14 JSONL shape (JSON-serialisable)."""
        return {
            "summary_id": self.summary_id,
            "book_id": self.book_id,
            "scope": self.scope,
            "target": self.target,
            "level": self.level,
            "parent_summary_id": self.parent_summary_id,
            "cluster_id": self.cluster_id,
            "text": self.text,
            "evidence": list(self.evidence),
            "status": self.status,
            "stale": self.stale,
            "produced_by": self.produced_by,
        }


def _unit_volume(unit: Mapping[str, Any], fallback_no: int) -> int:
    value = unit.get("volume", fallback_no)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback_no
    return max(1, number)


def build_summary_tree(
    units: Iterable[Mapping[str, Any]],
    *,
    book_id: str | None = None,
    unit_summaries: Mapping[str, str] | None = None,
    volume_titles: Mapping[int, str] | None = None,
    produced_by: str = "box:derive",
) -> dict[str, Any]:
    """Build a deterministic book → volume → unit summary tree.

    Args:
        units: Unit records from ``work/plan/units.jsonl`` (``order`` is used to
            sort; ``volume``/``chapter``/``title`` label the nodes).
        book_id: Book id; defaults to the first unit's ``book_id``.
        unit_summaries: Optional unit_id -> summary text. Missing units stay
            ``text=None`` (the stage-1 skeleton never fabricates a summary).
        volume_titles: Optional volume number -> title.
        produced_by: Provenance string for the ``produced_by`` column.

    Returns:
        ``{"contract", "book_id", "root", "nodes"}`` where nodes are §5.14
        record dicts in level order (book, volumes, units).

    Raises:
        ValueError: no units, mixed book ids, or duplicate unit ids.
    """
    ordered: list[tuple[int, dict[str, Any]]] = []
    seen_units: set[str] = set()
    seen_books: set[str] = set()
    for index, unit in enumerate(units, 1):
        uid = str(unit.get("unit_id") or "")
        if not uid:
            raise ValueError("unit record is missing unit_id")
        if uid in seen_units:
            raise ValueError(f"duplicate unit_id: {uid}")
        seen_units.add(uid)
        seen_books.add(str(unit.get("book_id") or ""))
        try:
            order = int(unit.get("order", index))
        except (TypeError, ValueError):
            order = index
        ordered.append((order, dict(unit)))
    if not ordered:
        raise ValueError("cannot build a summary tree without units")
    if len(seen_books) > 1:
        raise ValueError(f"mixed book_id values: {sorted(seen_books)}")

    resolved_book = book_id or next(iter(seen_books), "") or "book"
    summaries = dict(unit_summaries or {})
    # `volume_titles` is accepted but SummaryNode has no title field, so the value
    # has nowhere to go. Recorded as a known gap rather than silently dropped: the
    # caller passes it expecting it to appear somewhere.
    _unused_volume_titles = {
        int(k): str(v) for k, v in (volume_titles or {}).items()
    }

    root = SummaryNode(
        summary_id=make_summary_id(resolved_book, "book", resolved_book),
        book_id=resolved_book,
        scope="book",
        target=resolved_book,
        level=0,
        parent_summary_id=None,
        produced_by=produced_by,
    )
    nodes: list[SummaryNode] = [root]

    by_volume: dict[int, list[dict[str, Any]]] = {}
    for _, unit in sorted(ordered, key=lambda item: item[0]):
        volume = _unit_volume(unit, 1)
        by_volume.setdefault(volume, []).append(unit)

    volumes: dict[int, SummaryNode] = {}
    for volume_no in sorted(by_volume):
        volume_node = SummaryNode(
            summary_id=make_summary_id(resolved_book, "volume", f"v{volume_no:02d}"),
            book_id=resolved_book,
            scope="volume",
            target=f"v{volume_no:02d}",
            level=1,
            parent_summary_id=root.summary_id,
            text=None,
            produced_by=produced_by,
        )
        volumes[volume_no] = volume_node
        nodes.append(volume_node)

    for volume_no in sorted(by_volume):
        for unit in by_volume[volume_no]:
            uid = str(unit["unit_id"])
            unit_node = SummaryNode(
                summary_id=make_summary_id(resolved_book, "chapter", uid),
                book_id=resolved_book,
                scope="chapter",
                target=uid,
                level=2,
                parent_summary_id=volumes[volume_no].summary_id,
                text=summaries.get(uid),
                produced_by=produced_by,
            )
            nodes.append(unit_node)

    return {
        "contract": SUMMARY_CONTRACT,
        "book_id": resolved_book,
        "root": root.summary_id,
        "nodes": [node.to_record() for node in nodes],
    }


def summary_rows(tree: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the ``derived/summaries.jsonl`` rows from a summary tree."""
    nodes = tree.get("nodes") or []
    return [dict(node) for node in nodes]


def validate_summary_tree(tree: Mapping[str, Any]) -> list[str]:
    """Check summary-tree closure (ids unique, parents exist, no cycle)."""
    errors: list[str] = []
    nodes = tree.get("nodes") or []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, node in enumerate(nodes, 1):
        sid = str(node.get("summary_id") or "")
        if not sid:
            errors.append(f"node {index}: missing summary_id")
            continue
        if sid in by_id:
            errors.append(f"node {index}: duplicate summary_id {sid}")
        by_id[sid] = node
    root = tree.get("root")
    if root not in by_id:
        errors.append(f"root summary id missing from nodes: {root!r}")
    for sid, node in by_id.items():
        parent = node.get("parent_summary_id")
        if parent is not None and parent not in by_id:
            errors.append(f"summary {sid}: parent missing {parent}")
    # cycle detection over parent links
    for sid in by_id:
        seen: set[str] = set()
        current: str | None = sid
        while current is not None:
            if current in seen:
                errors.append(f"summary {sid}: parent cycle at {current}")
                break
            seen.add(current)
            parent = by_id.get(current, {}).get("parent_summary_id")
            current = str(parent) if parent is not None else None
    return errors


def mark_stale(
    tree: Mapping[str, Any], changed_units: Iterable[str]
) -> dict[str, Any]:
    """Return a copy of ``tree`` with changed units and their ancestors stale.

    Incremental rule from spec §5.14: only code marks ``stale`` (not the model).
    The input tree is not mutated. Unknown unit ids are ignored.
    """
    changed = {str(uid) for uid in changed_units if uid}
    nodes = [dict(node) for node in tree.get("nodes") or []]
    by_id = {str(node.get("summary_id")): node for node in nodes}
    stale_ids: set[str] = set()
    for node in nodes:
        if node.get("target") in changed and node.get("scope") in ("chapter", "scene"):
            stale_ids.add(str(node.get("summary_id")))
    changed_any = bool(stale_ids)
    while changed_any:
        parents: set[str] = set()
        for sid in stale_ids:
            parent = by_id.get(sid, {}).get("parent_summary_id")
            if parent is not None and parent not in stale_ids:
                parents.add(str(parent))
        changed_any = bool(parents)
        stale_ids |= parents
    for node in nodes:
        node["stale"] = str(node.get("summary_id")) in stale_ids
    result = dict(tree)
    result["nodes"] = nodes
    return result
