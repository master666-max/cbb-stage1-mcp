"""A community report needs the fields the source's reports carry.

Measured gap (A19)
------------------
`derive/community.py:121-135` builds a level-1 community row with:

    report_id, book_id, level, community_id, parent_id, member_entities, title, summary,
    evidence, stale, produced_by

The source's `COMMUNITIES_FINAL_COLUMNS` also carries `relationship_ids`, `text_unit_ids`, `size` and
`period`. Without them a report cannot answer the questions a community report exists for:

* **which relations** hold the community together (`relationship_ids`) — `member_entities` says who is in
  it, not what binds them;
* **which text units** support it (`text_unit_ids`) — without this there is no way to check the summary
  against the book;
* **how big** it is (`size`) — a reader has to count the member list, and a size that disagrees with the
  list is a signal;
* **when** (`period`) — a community that spans a hundred chapters and one that spans two are different
  findings.

The detection algorithm itself (connected components vs Leiden) is a declared substitution and is not
changed here. This is about the report's shape: a row that omits what the source's report carries cannot
be compared with one that has it.

What is pinned
--------------
* every community row carries the four fields;
* `size` equals the member count (so the two cannot disagree);
* `relationship_ids` are the assertions that joined the members, not an arbitrary sample;
* `text_unit_ids` are the distinct units those assertions came from;
* `period` is the min/max unit span, and is explicit when it is unknown rather than silently absent.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.derive.community import build_community_reports

BOOK = "t"
A = "e:t:person:aaaaaaaaaaaa"
B = "e:t:person:bbbbbbbbbbbb"
C = "e:t:person:cccccccccccc"


def _assertion(assertion_id: str, left: str, right: str, unit: str) -> dict:
    return {
        "assertion_id": assertion_id,
        "book_id": BOOK,
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": left, "受事": right},
        "claim_layer": "fact",
        "confidence": 0.9,
        "confidence_method": "evidence_span",
        "status": "asserted",
        "evidence": [
            {"unit_id": unit, "start": 0, "end": 4, "quote": "对阵", "occurrence": 1}
        ],
    }


def _rows() -> list[dict[str, Any]]:
    assertions = [
        _assertion("a:1", A, B, "u:t:v01:c001"),
        _assertion("a:2", B, C, "u:t:v01:c002"),
        # A separate component.
        _assertion("a:3", "e:t:person:dddddddddddd", "e:t:person:eeeeeeeeeeee", "u:t:v01:c003"),
    ]
    return build_community_reports(BOOK, assertions, levels=(1,))


# ══ the four fields exist ════════════════════════════════════════════
def test_every_community_row_carries_the_report_fields() -> None:
    rows = _rows()
    assert rows, "no communities were built"
    for row in rows:
        for field in ("relationship_ids", "text_unit_ids", "size", "period"):
            assert field in row, f"the community row is missing {field!r}: {sorted(row)}"


# ══ size agrees with the member list ═════════════════════════════════
def test_size_equals_the_member_count() -> None:
    for row in _rows():
        assert row["size"] == len(row["member_entities"]), (
            f"size {row['size']} disagrees with {len(row['member_entities'])} members"
        )


# ══ the relations that bind the community ═══════════════════════════
def test_relationship_ids_are_the_assertions_that_joined_the_members() -> None:
    rows = _rows()
    joined = {row["community_id"]: set(row["relationship_ids"]) for row in rows}
    # The component {A, B, C} is held by a:1 and a:2.
    component = next(row for row in rows if set(row["member_entities"]) == {A, B, C})
    assert set(component["relationship_ids"]) == {"a:1", "a:2"}, (
        f"the community's relations are {component['relationship_ids']}, not the two that joined it"
    )
    assert all(ids for ids in joined.values()), "a community carries no relationship ids"


def test_text_unit_ids_are_the_units_the_relations_came_from() -> None:
    component = next(
        row for row in _rows() if set(row["member_entities"]) == {A, B, C}
    )
    assert sorted(component["text_unit_ids"]) == ["u:t:v01:c001", "u:t:v01:c002"]


# ══ the span ════════════════════════════════════════════════════════
def test_period_is_the_unit_span() -> None:
    component = next(
        row for row in _rows() if set(row["member_entities"]) == {A, B, C}
    )
    period = component["period"]
    assert period["first_unit"] == "u:t:v01:c001"
    assert period["last_unit"] == "u:t:v01:c002"


def test_a_single_unit_community_has_a_degenerate_period() -> None:
    rows = _rows()
    component = next(
        row
        for row in rows
        if set(row["member_entities"]) == {"e:t:person:dddddddddddd", "e:t:person:eeeeeeeeeeee"}
    )
    assert component["period"]["first_unit"] == component["period"]["last_unit"]
