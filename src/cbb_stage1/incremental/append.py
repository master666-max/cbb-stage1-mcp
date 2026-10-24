"""Append-based incremental update: shift the delta's ids, never touch the old rows.

Ported from `库/code/B-项目案例/B02-graphrag/packages/graphrag/graphrag/index/update/communities.py:47-73`,
which is what the source project ships instead of the "old table + delta patch" the audit asked for.

The three rules and their failure modes:

* shift by ``old_max + 1`` — a recompute would move ids that other tables already reference;
* never shift the ``-1`` sentinel — shifting it turns "no community" into a real community;
* shift ``parent`` with ``community`` — otherwise the hierarchy points at rows that no longer exist.

Plus the source's shape repair: a column added in a later version is backfilled with ``None`` on both sides
before concatenation. Same lesson as A19: a missing column is a mismatch to repair deliberately, not an error.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "SENTINEL",
    "append_communities",
    "shift_ids",
]

#: The "no parent / no community" sentinel. It never shifts.
SENTINEL = -1

#: Columns the source backfills when an older table lacks them.
_BACKFILL_COLUMNS: tuple[str, ...] = ("size", "period")


def shift_ids(value: Any, offset: int) -> Any:
    """Move one id by ``offset``; ``None`` and the sentinel are left alone."""
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return value
    if number == SENTINEL:
        return SENTINEL
    return number + int(offset)


def _max_id(rows: Sequence[Mapping[str, Any]], key: str = "community") -> int:
    best = SENTINEL
    for row in rows:
        try:
            value = int(row.get(key))
        except (TypeError, ValueError):
            continue
        if value > best:
            best = value
    return best


def _backfill(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy rows with the late-added columns present (``None`` when absent)."""
    filled: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        for column in _BACKFILL_COLUMNS:
            copy.setdefault(column, None)
        filled.append(copy)
    return filled


def append_communities(
    old_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Concatenate the delta onto the old communities, shifting the delta's ids.

    Returns ``(merged_rows, mapping)``. The mapping is returned because the report table has to apply the
    **same** shift — two different offsets would break the join between a community and its report.
    """
    old = _backfill(old_rows)
    delta = _backfill(delta_rows)
    if not delta:
        return old, {}
    offset = _max_id(old) + 1 if old else 0

    mapping: dict[int, int] = {}
    for row in delta:
        try:
            original = int(row.get("community"))
        except (TypeError, ValueError):
            continue
        mapping[original] = shift_ids(original, offset)

    merged: list[dict[str, Any]] = []
    for row in delta:
        copy = dict(row)
        copy["community"] = shift_ids(row.get("community"), offset)
        copy["parent"] = shift_ids(row.get("parent"), offset)
        merged.append(copy)
    return old + merged, mapping
