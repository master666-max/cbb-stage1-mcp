"""The disclosure artifact: coverage in numbers, and the gate exercised on real rows.

F4's data-side obligation is "every record says where it becomes knowable". A claim is not evidence, so the box
carries `qc/disclosure.json` with per-table coverage and a gate run at a chosen horizon.

Two numbers matter and neither may be flattered:

* **undatable** — records the gate must withhold because nothing anchors them. Reported, not hidden, because this
  is the residual risk the audit asked about;
* **withheld at the chosen horizon** — proof the gate is wired to real data rather than only to fixtures.

An undatable record is withheld at every horizon, including an unlimited one: the report must not imply that
raising the horizon reveals everything.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .as_of import as_of, build_unit_order
from .gate import SURFACES, disclose_rows, horizon_from_unit

__all__ = [
    "CONTRACT",
    "REPORT_PATH",
    "disclosure_report",
    "table_coverage",
]

#: Named so a reader can tell which format this record holds.
CONTRACT = "cbb-disclosure/1"

#: Where the artifact lives inside a box.
REPORT_PATH = "qc/disclosure.json"


def table_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    unit_order: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """How many records in one table can be dated."""
    total = len(rows)
    datable = sum(1 for row in rows if as_of(row, unit_order=unit_order).known)
    undatable = total - datable
    return {
        "total": total,
        "datable": datable,
        "undatable": undatable,
        "ratio": round(datable / total, 4) if total else 0.0,
    }


def _unit_order_from(units: Iterable[Any]) -> dict[str, int] | None:
    """Order map from unit records; ``None`` when no unit declares one (the id parser then applies)."""
    order: dict[str, int] = {}
    for unit in units or ():
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        try:
            order[unit_id] = int(unit.get("order"))
        except (TypeError, ValueError):
            continue
    return order or build_unit_order(
        str(unit.get("unit_id") or "") for unit in units or () if isinstance(unit, Mapping)
    ) or None


def disclosure_report(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    units: Iterable[Any] = (),
    horizon: int | str = 0,
) -> dict[str, Any]:
    """Coverage per table plus a gate run per surface at ``horizon``.

    The gate is exercised on every table together, so a surface that withholds nothing is visible as a zero
    rather than as an untested claim.
    """
    unit_order = _unit_order_from(units)
    # A reader states a position as a unit ("I have read to c003"); an internal caller already has the order.
    resolved_horizon = (
        int(horizon)
        if isinstance(horizon, int) and not isinstance(horizon, bool)
        else horizon_from_unit(horizon, unit_order=unit_order)
    )
    coverage = {
        name: table_coverage(rows, unit_order=unit_order) for name, rows in sorted(tables.items())
    }

    combined: list[Mapping[str, Any]] = [
        row for _name, rows in sorted(tables.items()) for row in rows
    ]

    surfaces: dict[str, dict[str, Any]] = {}
    for surface in SURFACES:
        report: dict[str, Any] = {}
        disclose_rows(combined, resolved_horizon, surface=surface, unit_order=unit_order, report=report)
        surfaces[surface] = report

    total = sum(item["total"] for item in coverage.values())
    datable = sum(item["datable"] for item in coverage.values())
    return {
        "contract": CONTRACT,
        "horizon": resolved_horizon,
        "tables": coverage,
        "totals": {
            "records": total,
            "datable": datable,
            "undatable": total - datable,
            "ratio": round(datable / total, 4) if total else 0.0,
        },
        "surfaces": surfaces,
    }
