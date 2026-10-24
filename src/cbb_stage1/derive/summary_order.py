"""How far an aggregate text reaches — derived from its tests.

The gate withholds a summary whole rather than trimming it, because a half summary misleads. That needs the row
to state its furthest source position, and `evidence` alone is not reliable for it (it may be trimmed to a few
representative units). So `max_order` is computed when the text is produced.

Unknown stays unknown: a row whose units cannot be ordered gets `None`, which the gate treats as a reason to
withhold.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1.disclose.as_of import parse_unit_order

__all__ = [
    "MAX_ORDER_FIELD",
    "annotate_summaries",
    "max_order_of",
    "unit_order_map",
]

MAX_ORDER_FIELD = "max_order"

#: Fields a summary row may carry its units in.
_UNIT_FIELDS: tuple[str, ...] = ("unit_ids", "member_units", "units")


def unit_order_map(units: Iterable[Any]) -> dict[str, int]:
    """``{unit_id: order}`` from unit records, using the declared order where present."""
    order: dict[str, int] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        try:
            value = int(unit.get("order"))
        except (TypeError, ValueError):
            parsed = parse_unit_order(unit_id)
            if parsed is None:
                continue
            value = parsed
        order[unit_id] = value
    return order


def _unit_ids(record: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for name in _UNIT_FIELDS:
        value = record.get(name)
        if isinstance(value, (list, tuple)):
            ids.extend(str(item) for item in value if str(item))
    scope = record.get("scope")
    if isinstance(scope, Mapping):
        for name in ("unit_ids", "units"):
            value = scope.get(name)
            if isinstance(value, (list, tuple)):
                ids.extend(str(item) for item in value if str(item))
    for item in record.get("evidence") or ():
        unit_id = str(item.get("unit_id") or "") if isinstance(item, Mapping) else str(item or "")
        if unit_id:
            ids.append(unit_id)
    # Measured: real summary rows carry `evidence == []`. A chapter-scope row points at its unit through
    # `target`, which is the only anchor available.
    if str(record.get("scope") or "") == "chapter":
        target = str(record.get("target") or "")
        if target:
            ids.append(target)
    return list(dict.fromkeys(ids))


def max_order_of(record: Mapping[str, Any], order: Mapping[str, int] | None = None) -> int | None:
    """The furthest position this record's text reaches, or ``None``.

    When a map is supplied it is authoritative: a unit missing from it is unknown rather than parsed from its id,
    so a row cannot borrow a position from a map that deliberately excludes its units.
    """
    resolved: list[int] = []
    for unit_id in _unit_ids(record):
        if order is not None:
            if unit_id in order:
                resolved.append(int(order[unit_id]))
            continue
        parsed = parse_unit_order(unit_id)
        if parsed is not None:
            resolved.append(parsed)
    return max(resolved) if resolved else None


def annotate_summaries(
    rows: Sequence[Mapping[str, Any]],
    order: Mapping[str, int] | None = None,
    *,
    field: str = MAX_ORDER_FIELD,
) -> list[dict[str, Any]]:
    """Copy each row with ``max_order`` set, inheriting through ``parent_summary_id``.

    A volume or book summary has no units of its own — measured, its `evidence` list is empty — so it is only as
    far-reaching as its children. Inheritance runs bottom-up and terminates on a cycle, leaving those rows
    unknown rather than hanging.

    Order is preserved and the input is never mutated.
    """
    annotated: list[dict[str, Any]] = [dict(row) for row in rows]
    by_id: dict[str, int] = {}
    for row in annotated:
        summary_id = str(row.get("summary_id") or "")
        value = max_order_of(row, order)
        row[field] = value
        if summary_id and value is not None:
            by_id[summary_id] = value

    children_of: dict[str, list[str]] = {}
    for row in annotated:
        parent_id = str(row.get("parent_summary_id") or "")
        child_id = str(row.get("summary_id") or "")
        if parent_id and child_id:
            children_of.setdefault(parent_id, []).append(child_id)

    # Bottom-up: repeatedly push a child's value into its parent until nothing changes. Bounded so a corrupt
    # parent cycle cannot spin forever.
    for _round in range(len(annotated) + 1):
        changed = False
        for row in annotated:
            summary_id = str(row.get("summary_id") or "")
            if not summary_id:
                continue
            child_values = [by_id[child] for child in children_of.get(summary_id, ()) if child in by_id]
            if not child_values:
                continue
            candidate = max(child_values)
            current = row.get(field)
            if current is None or candidate > int(current):
                row[field] = candidate
                by_id[summary_id] = candidate
                changed = True
        if not changed:
            break
    return annotated
