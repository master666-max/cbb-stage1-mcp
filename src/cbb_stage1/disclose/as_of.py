"""When a record becomes knowable.

*Without Spoilers* states the data-side rule this module implements: "every paragraph is tagged with the chapter
it is drawn from … the filtering is mechanical … not a guess". Every box table already carries such an anchor;
this turns it into one comparable number.

Anchors differ by record shape, and the difference is deliberate:

* ``evidence[...]`` — an assertion is supported by several quotes, and you cannot evaluate it fully until you
  have read **all** of them, so the anchor is the **latest** evidence. Taking the earliest would show a claim
  while part of its support is still in the reader's future;
* ``first_unit`` — an entity becomes visible the moment it appears, so the anchor is the **earliest** unit.
  Using its last mention would hide a character for the whole book.

A missing anchor is reported, never guessed: `known=False` makes the gate withhold, which is recoverable;
showing something too early is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "AsOf",
    "as_of",
    "build_unit_order",
    "parse_unit_order",
]

#: `u:<book>:v<volume>:c<chapter>`, the unit id shape the pipeline emits.
_UNIT_RE = re.compile(r":v(\d+):c(\d+)\s*$")

#: Volume and chapter are folded into one comparable integer.
_CHAPTER_SPAN = 1000


@dataclass(frozen=True)
class AsOf:
    """The earliest reading position at which a record may be shown."""

    unit_id: str | None = None
    order: int | None = None
    source: str = "unknown"
    known: bool = False
    candidates: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "order": self.order,
            "source": self.source,
            "known": self.known,
            "candidates": list(self.candidates),
        }


def parse_unit_order(unit_id: Any) -> int | None:
    """Reading order from a structured unit id, or ``None``.

    Numeric, not lexical: ``c010`` must sort after ``c009``. A unit id that does not follow the shape yields
    ``None`` rather than a guess, because a wrong order is a spoiler and a missing one is only a withheld row.
    """
    match = _UNIT_RE.search(str(unit_id or ""))
    if not match:
        return None
    return int(match.group(1)) * _CHAPTER_SPAN + int(match.group(2))


def build_unit_order(unit_ids: Iterable[Any]) -> dict[str, int]:
    """``{unit_id: order}`` for the ids that parse; unparseable ids are omitted."""
    order: dict[str, int] = {}
    for unit_id in unit_ids:
        text = str(unit_id or "")
        value = parse_unit_order(text)
        if text and value is not None:
            order[text] = value
    return order


def _unit_ids_from_evidence(evidence: Any) -> list[str]:
    """Unit ids from an evidence list, accepting mappings or bare ids."""
    ids: list[str] = []
    if not isinstance(evidence, (list, tuple)):
        return ids
    for item in evidence:
        if isinstance(item, Mapping):
            unit_id = str(item.get("unit_id") or "")
        else:
            unit_id = str(item or "")
        if unit_id:
            ids.append(unit_id)
    return ids


def _resolve(unit_ids: Iterable[str], unit_order: Mapping[str, int] | None) -> list[tuple[str, int]]:
    """Pairs of ``(unit_id, order)`` that can be resolved, explicit map first then the id itself."""
    resolved: list[tuple[str, int]] = []
    for unit_id in unit_ids:
        # An explicit map is authoritative: a unit missing from it is unknown, not "parsed from its id". Mixing
        # the two would let a record borrow a position from a map that deliberately does not contain it.
        if unit_order is not None:
            if unit_id in unit_order:
                resolved.append((unit_id, int(unit_order[unit_id])))
            continue
        parsed = parse_unit_order(unit_id)
        if parsed is not None:
            resolved.append((unit_id, parsed))
    return resolved


def as_of(record: Any, *, unit_order: Mapping[str, int] | None = None) -> AsOf:
    """The disclosure position of one record.

    Anchor precedence follows the record's shape rather than a single global rule: ``first_unit`` is the
    entity's appearance and takes priority when present, ``evidence`` is a claim's support and is dated by its
    latest item, and a bare ``unit_id`` is a unit-bound row.
    """
    data: Mapping[str, Any] = record if isinstance(record, Mapping) else {}

    # An aggregate row states its reach explicitly, because its evidence may be empty or trimmed. `None` is a
    # computed answer ("undeterminable") and must not fall through to an optimistic anchor.
    if "max_order" in data:
        raw = data.get("max_order")
        if raw is None:
            return AsOf(
                unit_id=None,
                order=None,
                source="max_order_unknown",
                known=False,
                candidates=(),
            )
        try:
            return AsOf(
                unit_id=str(data.get("target") or "") or None,
                order=int(raw),
                source="max_order",
                known=True,
                candidates=(str(data.get("target") or ""),),
            )
        except (TypeError, ValueError):
            pass  # malformed: fall through to the other anchors rather than fail the box

    first_unit = str(data.get("first_unit") or "")
    evidence_ids = _unit_ids_from_evidence(data.get("evidence"))
    own_unit = str(data.get("unit_id") or "")
    candidates = tuple(dict.fromkeys([first_unit] + evidence_ids + [own_unit]))

    if first_unit:
        resolved = _resolve([first_unit], unit_order)
        if resolved:
            unit_id, order = resolved[0]
            return AsOf(
                unit_id=unit_id,
                order=order,
                source="first_unit",
                known=True,
                candidates=candidates,
            )

    if evidence_ids:
        resolved = _resolve(evidence_ids, unit_order)
        if resolved:
            unit_id, order = max(resolved, key=lambda pair: pair[1])
            return AsOf(
                unit_id=unit_id,
                order=order,
                source="evidence",
                known=True,
                candidates=candidates,
            )

    if own_unit:
        resolved = _resolve([own_unit], unit_order)
        if resolved:
            unit_id, order = resolved[0]
            return AsOf(
                unit_id=unit_id,
                order=order,
                source="unit_id",
                known=True,
                candidates=candidates,
            )

    return AsOf(unit_id=None, order=None, source="unknown", known=False, candidates=candidates)
