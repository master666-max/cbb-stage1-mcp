"""A20/D6: append-based incremental update, the pattern the source project actually ships.

Measured gap, and the correction
--------------------------------
The audit's A20 asked for `update_communities` / `update_community_reports` equivalents — "旧表+delta 表合并
的补丁". GraphRAG, the project this series ports from, does something simpler and it is in production:
`graphrag/index/update/communities.py:47-73`

    old_max = old["community"].max()
    mapping = {v: v + old_max + 1 for v in delta["community"]}
    mapping.update({-1: -1})                      # the "no community" sentinel must not shift
    delta["community"] = delta["community"].apply(mapping)
    delta["parent"]    = delta["parent"].apply(mapping)
    merged = pd.concat([old, delta])

Three lessons worth carrying, each with a failure mode if dropped:

* **Do not recompute.** The delta's ids shift by an offset; the old rows are untouched. A "patch" that
  re-derives communities would change ids that other tables already reference.
* **The `-1` sentinel must not shift.** If it does, "no community" silently becomes a real community id.
* **`parent` shifts with `community`**, or the hierarchy points at rows that no longer exist.

`entities.py:26-35` adds a fourth: columns added in a later version (`size`, `period`) are backfilled with
None on both sides before concat. That is the same lesson as the A19 fix — a missing column is not an error,
it is a shape mismatch to repair deliberately.

What is pinned
--------------
* ids shift by `old_max + 1` and old rows are byte-identical afterwards;
* `-1` stays `-1`, in both `community` and `parent`;
* the mapping is returned so later tables (reports) can use the same shift;
* an empty delta returns the old rows unchanged, not an error;
* appending twice is idempotent in the sense that no id collides with an existing one.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.incremental.append import (
    SENTINEL,
    append_communities,
    shift_ids,
)


def _community(cid: int, parent: int | None, members: list[str]) -> dict[str, Any]:
    return {
        "community": cid,
        "parent": parent,
        "level": 1 if parent == SENTINEL else 2,
        "member_entities": members,
        "title": f"Community {cid}",
    }


# ══ the shift ═══════════════════════════════════════════════════════
def test_the_shift_moves_every_id_by_the_offset() -> None:
    assert shift_ids(3, 5) == 8


def test_the_sentinel_does_not_move() -> None:
    assert shift_ids(SENTINEL, 5) == SENTINEL, (
        "shifting the sentinel turns 'no community' into a real community id"
    )


def test_a_missing_parent_stays_missing() -> None:
    assert shift_ids(None, 5) is None


# ══ appending communities ═══════════════════════════════════════════
def test_the_delta_is_shifted_and_the_old_rows_are_untouched() -> None:
    old = [_community(0, SENTINEL, ["a"]), _community(1, 0, ["b"])]
    delta = [_community(0, SENTINEL, ["x"]), _community(1, 0, ["y"])]
    merged, mapping = append_communities(old, delta)

    assert [row["community"] for row in old] == [0, 1], "the old rows were modified"
    assert sorted(row["community"] for row in merged) == [0, 1, 2, 3]
    assert mapping == {0: 2, 1: 3}


def test_the_deltas_parent_is_shifted_too() -> None:
    old = [_community(0, SENTINEL, ["a"])]
    delta = [_community(0, SENTINEL, ["x"]), _community(1, 0, ["y"])]
    merged, _mapping = append_communities(old, delta)
    child = next(row for row in merged if row["member_entities"] == ["y"])
    assert child["parent"] == 1, (
        f"the child points at {child['parent']!r}, which is an old id or the unshifted delta id"
    )
    assert any(row["community"] == child["parent"] for row in merged)


def test_an_empty_delta_returns_the_old_rows() -> None:
    old = [_community(0, SENTINEL, ["a"])]
    merged, mapping = append_communities(old, [])
    assert [row["community"] for row in merged] == [0], "the old rows moved"
    assert mapping == {}


def test_an_empty_old_table_keeps_the_delta_ids() -> None:
    """With no previous rows the offset is zero, so nothing should move."""
    delta = [_community(0, SENTINEL, ["x"])]
    merged, _mapping = append_communities([], delta)
    assert [row["community"] for row in merged] == [0]


def test_a_missing_column_is_backfilled_rather_than_failing() -> None:
    """The source adds `size`/`period` as None when an older table lacks them."""
    old = [{"community": 0, "parent": SENTINEL, "member_entities": ["a"]}]
    delta = [{"community": 0, "parent": SENTINEL, "member_entities": ["x"], "size": 1, "period": {"first_unit": None, "last_unit": None}}]
    merged, _mapping = append_communities(old, delta)
    assert all("size" in row for row in merged), "the column was not backfilled"
    assert all("period" in row for row in merged)


# ══ no id collision ever ═══════════════════════════════════════════
def test_two_appends_never_collide() -> None:
    old = [_community(0, SENTINEL, ["a"])]
    first, mapping = append_communities(old, [_community(0, SENTINEL, ["x"])])
    second, _mapping = append_communities(first, [_community(0, SENTINEL, ["z"])])
    ids = [row["community"] for row in second]
    assert len(ids) == len(set(ids)), f"ids collided after two appends: {ids}"
