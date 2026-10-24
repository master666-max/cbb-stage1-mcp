"""One gate for every exit.

The precedents disagree about presentation and agree about architecture: *Without Spoilers* filters "every
article, infobox, link, and the character timeline" and excludes unmet characters from search; Thulr filters
"within each article, on overview pages, in each link and alias reveal, and even in the text snippets shown on
Google search"; Discourse's leak was an exit nobody wired.

So this module has one entry point and a **closed** surface set. `surface` is a required keyword so that adding
an exit is a visible act: the alternative is a default that quietly applies another exit's policy.

Two policies, chosen per surface because they protect different things:

* **keep-and-mark** (`card`, `summary`) — the client needs to know something is there in order to render a
  placeholder and to unlock it when the reader advances. The record stays; its payload is withheld and
  `reveal_at` says when it becomes available.
* **omit** (`index`, `alias`, `search`, `prompt`, `timeline`) — the row's *existence* is itself the disclosure.
  An alias reveals an identity, an index entry reveals that a character exists, and a prompt is quotable.

Unknown dates fail closed. A record we cannot date is withheld, because showing something too early cannot be
taken back while withholding one paragraph can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .as_of import AsOf, as_of, parse_unit_order

__all__ = [
    "KEEP_AND_MARK",
    "OMIT",
    "SURFACES",
    "Disclosed",
    "disclose",
    "disclose_rows",
    "horizon_from_unit",
]

#: Exits that keep the row and mark it withheld, so a client can render a placeholder and unlock later.
KEEP_AND_MARK: frozenset[str] = frozenset({"card", "summary"})

#: Exits where the row's existence is itself the disclosure, so it is dropped.
OMIT: frozenset[str] = frozenset({"index", "alias", "search", "prompt", "timeline"})

#: The closed set of exits this repository has.
SURFACES: tuple[str, ...] = ("card", "index", "alias", "summary", "search", "prompt", "timeline")

#: Fields whose value is the disclosure payload, withheld on the keep-and-mark surfaces.
_PAYLOAD_FIELDS: tuple[str, ...] = ("text", "summary", "description", "content")


@dataclass
class Disclosed:
    """One record after the gate."""

    record: dict[str, Any] | None
    withheld: bool
    reason: str = ""
    as_of: AsOf = field(default_factory=AsOf)
    surface: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "withheld": self.withheld,
            "reason": self.reason,
            "as_of": self.as_of.as_dict(),
            "record": self.record,
        }


def horizon_from_unit(unit_id: Any, *, unit_order: Mapping[str, int] | None = None) -> int:
    """The reader's position as a comparable order.

    An unresolvable position is an error rather than a default: a typo in "I have read up to here" must not
    release the whole book.
    """
    text = str(unit_id or "")
    if unit_order is not None and text in unit_order:
        return int(unit_order[text])
    parsed = parse_unit_order(text)
    if parsed is None:
        raise ValueError(
            f"unresolvable horizon {unit_id!r}; expected a unit id like 'u:<book>:v01:c003' "
            "or an entry in unit_order"
        )
    return parsed


def _withheld_payload(record: Mapping[str, Any], order: int | None, unit_id: str | None) -> dict[str, Any]:
    """A copy of the record with its payload removed and the lock described."""
    kept = dict(record)
    for name in _PAYLOAD_FIELDS:
        if name in kept:
            kept[name] = None
    kept["withheld"] = True
    kept["reveal_at"] = order
    kept["reveal_unit"] = unit_id
    return kept


def disclose(
    record: Any,
    horizon: int,
    *,
    surface: str,
    unit_order: Mapping[str, int] | None = None,
) -> Disclosed:
    """Apply the gate to one record for one exit.

    Args:
        record: A box row (assertion, entity, mention, citation, summary, card).
        horizon: The reader's position as an order value.
        surface: Which exit is asking. Required, and must be in :data:`SURFACES`.
        unit_order: Optional authoritative unit→order map.
    """
    if surface not in SURFACES:
        raise ValueError(
            f"unknown surface {surface!r}; expected one of {list(SURFACES)}. "
            "A new exit must be named here so its absence cannot pass as a policy."
        )
    if not isinstance(record, Mapping):
        return Disclosed(
            record=None,
            withheld=True,
            reason="record is not a mapping; nothing can be released",
            as_of=AsOf(),
            surface=surface,
        )

    position = as_of(record, unit_order=unit_order)
    if not position.known:
        return _withhold(
            record,
            surface,
            "record has no resolvable date (as_of unknown); withheld rather than released",
            position,
            None,
            None,
        )

    order = position.order
    if order is None or order <= int(horizon):
        return Disclosed(
            record=dict(record),
            withheld=False,
            reason="",
            as_of=position,
            surface=surface,
        )

    reason = f"record becomes available at order {order}, past the reader's horizon {horizon}"
    return _withhold(record, surface, reason, position, order, position.unit_id)


def _withhold(
    record: Mapping[str, Any],
    surface: str,
    reason: str,
    position: AsOf,
    order: int | None,
    unit_id: str | None,
) -> Disclosed:
    if surface in KEEP_AND_MARK:
        return Disclosed(
            record=_withheld_payload(record, order, unit_id),
            withheld=True,
            reason=reason,
            as_of=position,
            surface=surface,
        )
    return Disclosed(
        record=None,
        withheld=True,
        reason=reason,
        as_of=position,
        surface=surface,
    )


def disclose_rows(
    rows: Sequence[Mapping[str, Any]],
    horizon: int,
    *,
    surface: str,
    unit_order: Mapping[str, int] | None = None,
    report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply the gate to a list, preserving order and recording how much was withheld.

    Order is preserved because a caller may rely on it (a timeline is ordered by construction), and the count is
    recorded because "the list looks short" is exactly the symptom a reader cannot distinguish from a bug.
    """
    kept: list[dict[str, Any]] = []
    released = 0
    withheld = 0
    for row in rows:
        result = disclose(row, horizon, surface=surface, unit_order=unit_order)
        if result.record is None:
            withheld += 1
            continue
        if result.withheld:
            withheld += 1
        else:
            released += 1
        kept.append(result.record)
    if report is not None:
        report["surface"] = surface
        report["horizon"] = int(horizon)
        report["released"] = released
        report["withheld"] = withheld
        report["total"] = len(rows)
    return kept
