"""Summaries must carry the furthest position their text reaches.

F4 step 4. The gate withholds an aggregate text whole rather than trimming it, because a half summary misleads.
That decision needs the row to state how far it reaches, and the stored `evidence` list is not reliable for it —
it may be trimmed to a few representative units.

So `max_order` is computed when the text is produced, and `None` when it cannot be — which the gate treats as
unknown and withholds.

Pinned here
-----------
* `max_order` is the **maximum** order among the units that fed the text, not the first;
* units are found in whichever field the row uses (`unit_ids`, `member_units`, `units`, `scope`, `evidence`);
* a row with no resolvable units gets `None`, never a fabricated number;
* the input rows are not mutated;
* ids are kept in order.
"""

from __future__ import annotations

from cbb_stage1.derive.summary_order import (
    MAX_ORDER_FIELD,
    annotate_summaries,
    max_order_of,
    unit_order_map,
)

UNITS = [
    {"unit_id": "u:t:v01:c001", "order": 1},
    {"unit_id": "u:t:v01:c002", "order": 2},
    {"unit_id": "u:t:v02:c001", "order": 30},
]


def test_the_field_is_named() -> None:
    assert MAX_ORDER_FIELD == "max_order"


# ══ the map ═════════════════════════════════════════════════════════
def test_the_map_uses_the_declared_order() -> None:
    assert unit_order_map(UNITS)["u:t:v02:c001"] == 30


def test_the_map_falls_back_to_the_unit_id_when_order_is_missing() -> None:
    order = unit_order_map([{"unit_id": "u:t:v01:c003"}])
    assert order["u:t:v01:c003"] == 1003


def test_an_unparseable_unit_is_omitted_from_the_map() -> None:
    assert unit_order_map([{"unit_id": "nonsense"}]) == {}


# ══ the anchor ══════════════════════════════════════════════════════
def test_the_maximum_wins_not_the_first() -> None:
    row = {"summary_id": "s:1", "unit_ids": ["u:t:v01:c001", "u:t:v02:c001"]}
    assert max_order_of(row, unit_order_map(UNITS)) == 30


def test_units_are_found_in_member_units() -> None:
    row = {"summary_id": "s:1", "member_units": ["u:t:v01:c002"]}
    assert max_order_of(row, unit_order_map(UNITS)) == 2


def test_units_are_found_in_scope() -> None:
    row = {"summary_id": "s:1", "scope": {"unit_ids": ["u:t:v01:c002"]}}
    assert max_order_of(row, unit_order_map(UNITS)) == 2


def test_units_are_found_in_evidence() -> None:
    row = {"summary_id": "s:1", "evidence": [{"unit_id": "u:t:v01:c001"}]}
    assert max_order_of(row, unit_order_map(UNITS)) == 1


def test_a_row_with_no_units_is_unknown() -> None:
    assert max_order_of({"summary_id": "s:1"}, unit_order_map(UNITS)) is None


def test_an_unresolvable_unit_makes_it_unknown() -> None:
    row = {"summary_id": "s:1", "unit_ids": ["u:t:v09:c099"]}
    assert max_order_of(row, unit_order_map(UNITS)) is None, (
        "an unresolvable unit must not fall through to a parsed position while a map is authoritative"
    )


# ══ annotating ══════════════════════════════════════════════════════
def test_annotate_sets_the_field() -> None:
    rows = [{"summary_id": "s:1", "unit_ids": ["u:t:v01:c001"]}]
    out = annotate_summaries(rows, unit_order_map(UNITS))
    assert out[0]["max_order"] == 1


def test_annotate_keeps_order() -> None:
    rows = [
        {"summary_id": "s:1", "unit_ids": ["u:t:v01:c001"]},
        {"summary_id": "s:2", "unit_ids": ["u:t:v02:c001"]},
    ]
    out = annotate_summaries(rows, unit_order_map(UNITS))
    assert [row["summary_id"] for row in out] == ["s:1", "s:2"]


def test_annotate_does_not_mutate_the_input() -> None:
    rows = [{"summary_id": "s:1", "unit_ids": ["u:t:v01:c001"]}]
    annotate_summaries(rows, unit_order_map(UNITS))
    assert MAX_ORDER_FIELD not in rows[0], "the caller's rows were modified in place"


def test_annotate_preserves_unknown_as_none() -> None:
    out = annotate_summaries([{"summary_id": "s:1"}], unit_order_map(UNITS))
    assert out[0][MAX_ORDER_FIELD] is None

# ══ real summary rows carry no evidence; the anchor is target + parent ═
def test_a_chapter_summary_is_dated_by_its_target() -> None:
    """Measured: every summary level has vidence == []. For a chapter row the unit is 	arget."""
    row = {"summary_id": "sm:1", "scope": "chapter", "target": "u:t:v01:c002"}
    assert max_order_of(row, unit_order_map(UNITS)) == 2


def test_a_chapter_target_that_is_not_a_unit_is_unknown() -> None:
    row = {"summary_id": "sm:1", "scope": "chapter", "target": "v01"}
    assert max_order_of(row, unit_order_map(UNITS)) is None


def test_a_volume_summary_inherits_the_maximum_of_its_children() -> None:
    """A volume row has no units of its own; it is as far-reaching as its children."""
    rows = [
        {"summary_id": "sm:v", "scope": "volume", "target": "v01"},
        {"summary_id": "sm:c1", "scope": "chapter", "target": "u:t:v01:c001",
         "parent_summary_id": "sm:v"},
        {"summary_id": "sm:c2", "scope": "chapter", "target": "u:t:v02:c001",
         "parent_summary_id": "sm:v"},
    ]
    out = annotate_summaries(rows, unit_order_map(UNITS))
    assert out[0]["max_order"] == 30, (
        "the volume row was dated without looking at its children: " + str(out[0]["max_order"])
    )


def test_a_book_summary_inherits_through_two_levels() -> None:
    rows = [
        {"summary_id": "sm:b", "scope": "book", "target": "t"},
        {"summary_id": "sm:v", "scope": "volume", "target": "v01",
         "parent_summary_id": "sm:b"},
        {"summary_id": "sm:c", "scope": "chapter", "target": "u:t:v02:c001",
         "parent_summary_id": "sm:v"},
    ]
    out = annotate_summaries(rows, unit_order_map(UNITS))
    assert out[0]["max_order"] == 30, "the book row did not inherit through the volume level"


def test_a_childless_parent_stays_unknown() -> None:
    rows = [{"summary_id": "sm:b", "scope": "book", "target": "t"}]
    out = annotate_summaries(rows, unit_order_map(UNITS))
    assert out[0]["max_order"] is None


def test_inheritance_terminates_on_a_cycle() -> None:
    """A corrupt parent link must not hang the build."""
    rows = [
        {"summary_id": "a", "scope": "volume", "target": "v01", "parent_summary_id": "b"},
        {"summary_id": "b", "scope": "book", "target": "t", "parent_summary_id": "a"},
    ]
    out = annotate_summaries(rows, unit_order_map(UNITS))
    assert all(row["max_order"] is None for row in out)
