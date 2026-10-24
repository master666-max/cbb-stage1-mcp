"""F4 acceptance artifact: is the data datable, and does the gate behave?

The audit's complaint about F4 was "机制只到数据底座，未到披露行为". The data side is the half this repository
owes, so it needs an artifact that answers it in numbers rather than assertion:

* **coverage** — per table, how many records carry a resolvable date. "The box can be filtered" is only true to
  the extent this number is the total;
* **gate exercise** — at a chosen horizon, how many rows each surface releases and withholds. A gate nobody runs
  on real data is a claim, not a mechanism.

`qc/disclosure.json` is that artifact. It is also what makes the package reachable from the build entrypoint
instead of sitting unimported — the wiring gate is right to insist on a real caller.

Failure policy matches the gate's: a record that cannot be dated counts as **undatable and withheld**, never as
released. The report must not flatter the coverage.
"""

from __future__ import annotations

import json

from cbb_stage1.disclose.report import (
    CONTRACT,
    disclosure_report,
    table_coverage,
)

UNITS = [
    {"unit_id": "u:t:v01:c001", "order": 1},
    {"unit_id": "u:t:v01:c002", "order": 2},
    {"unit_id": "u:t:v01:c003", "order": 3},
]

ASSERTIONS = [
    {"assertion_id": "a:1", "evidence": [{"unit_id": "u:t:v01:c001"}]},
    {"assertion_id": "a:2", "evidence": [{"unit_id": "u:t:v01:c003"}]},
]

ENTITIES = [
    {"entity_id": "e:1", "first_unit": "u:t:v01:c001"},
    {"entity_id": "e:2"},  # no anchor at all
]


def _tables() -> dict:
    return {"knowledge/assertions.jsonl": ASSERTIONS, "identity/entities.jsonl": ENTITIES}


# ══ the contract ════════════════════════════════════════════════════
def test_the_contract_is_named() -> None:
    assert CONTRACT == "cbb-disclosure/1"


# ══ coverage is reported honestly ═══════════════════════════════════
def test_coverage_counts_datable_and_undatable() -> None:
    coverage = table_coverage(ASSERTIONS, unit_order=None)
    assert coverage["total"] == 2
    assert coverage["datable"] == 2
    assert coverage["undatable"] == 0


def test_a_record_with_no_anchor_is_undatable() -> None:
    coverage = table_coverage(ENTITIES, unit_order=None)
    assert coverage["total"] == 2
    assert coverage["datable"] == 1
    assert coverage["undatable"] == 1, "a record with no anchor was counted as datable"


def test_coverage_reports_its_ratio() -> None:
    coverage = table_coverage(ENTITIES, unit_order=None)
    assert coverage["ratio"] == 0.5


def test_coverage_of_an_empty_table_is_not_a_division_error() -> None:
    coverage = table_coverage([], unit_order=None)
    assert coverage["total"] == 0
    assert coverage["ratio"] == 0.0


# ══ the report ══════════════════════════════════════════════════════
def test_the_report_covers_every_table() -> None:
    report = disclosure_report(_tables(), units=UNITS)
    assert set(report["tables"]) == set(_tables())


def test_the_report_exercises_the_gate_per_surface() -> None:
    import cbb_stage1.disclose.gate as gate

    report = disclosure_report(_tables(), units=UNITS, horizon=1)
    for surface in gate.SURFACES:
        assert surface in report["surfaces"], f"{surface} was never exercised"


def test_the_gate_result_is_plausible_at_a_mid_horizon() -> None:
    """At order 1, the record dated 3 must be withheld and the one dated 1 released."""
    report = disclosure_report(_tables(), units=UNITS, horizon=1)
    counts = report["surfaces"]["card"]
    assert counts["released"] >= 1
    assert counts["withheld"] >= 1, (
        "at horizon 1 the record dated 3 was not withheld: " + json.dumps(counts, ensure_ascii=False)
    )


def test_an_undatable_record_is_withheld_not_released() -> None:
    report = disclosure_report(_tables(), units=UNITS, horizon=10_000)
    entities = report["tables"]["identity/entities.jsonl"]
    assert entities["undatable"] == 1
    # Even at an unlimited horizon, the undatable row stays withheld.
    card = report["surfaces"]["card"]
    assert card["withheld"] >= 1


def test_the_report_is_serialisable() -> None:
    json.dumps(disclosure_report(_tables(), units=UNITS, horizon=1), ensure_ascii=False)


def test_the_report_states_its_horizon() -> None:
    report = disclosure_report(_tables(), units=UNITS, horizon=2)
    assert report["horizon"] == 2


def test_an_empty_box_produces_a_report_rather_than_failing() -> None:
    report = disclosure_report({}, units=UNITS)
    assert report["tables"] == {}

# ══ the horizon may be given the way a reader states it ═════════════
def test_the_horizon_may_be_a_unit_id() -> None:
    """A reader says "I have read up to here", which is a unit, not an internal order number."""
    report = disclosure_report(_tables(), units=UNITS, horizon="u:t:v01:c001")
    assert report["horizon"] == 1


def test_a_unit_id_horizon_withholds_what_lies_beyond_it() -> None:
    report = disclosure_report(_tables(), units=UNITS, horizon="u:t:v01:c001")
    card = report["surfaces"]["card"]
    assert card["withheld"] >= 1


def test_an_unresolvable_horizon_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="horizon"):
        disclosure_report(_tables(), units=UNITS, horizon="nonsense")
