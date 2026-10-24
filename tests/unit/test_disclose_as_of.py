"""F4 step 1: every record must say when it becomes knowable.

The three precedents agree on the data side: *Without Spoilers* tags "every paragraph … with the chapter it is
drawn from", and calls the filtering "mechanical … not a guess". So the first thing this package owes is a
function that answers, from fields the box already carries, **the earliest reading position at which a record
may be shown**.

Anchors, by record shape (measured against real box tables):

    assertions   evidence[{unit_id, …}]      the claim is supported by these; you cannot evaluate it fully
                                             until you have read them all => the LATEST evidence
    entities     first_unit, last_unit       the entity becomes visible when it first appears => first_unit
    mentions     unit_id                     the mention lives in one unit
    citations    unit_id                     same
    summaries    evidence                    aggregate; the latest evidence is where it becomes fully known

Unknown must not be guessed. A record whose anchors cannot be resolved returns `known=False`, and the gate will
withhold it rather than show it — a spoiler is unrecoverable, a withheld paragraph is not.

What is pinned
--------------
* each anchor field is used, and only where it exists;
* assertions take the **latest** evidence while entities take `first_unit` — the two differ on purpose;
* order is resolved from an explicit map first, then parsed from the unit id (`u:<book>:vNN:cNNN`);
* an unresolvable unit makes `known=False` rather than yielding a fabricated number;
* a record with no anchors at all is `unknown`;
* `as_of` reports which anchor it used, so a thin result can be explained.
"""

from __future__ import annotations

from cbb_stage1.disclose.as_of import (
    AsOf,
    as_of,
    build_unit_order,
    parse_unit_order,
)

ORDER = {
    "u:t:v01:c001": 1001,
    "u:t:v01:c002": 1002,
    "u:t:v01:c003": 1003,
    "u:t:v02:c001": 2001,
}


def _assertion(*unit_ids: str) -> dict:
    return {
        "assertion_id": "a:1",
        "evidence": [{"unit_id": unit_id, "quote": "…"} for unit_id in unit_ids],
    }


# ══ order resolution ════════════════════════════════════════════════
def test_order_parses_from_the_unit_id() -> None:
    assert parse_unit_order("u:t:v01:c003") == 1003


def test_a_later_volume_sorts_after_an_earlier_one() -> None:
    assert parse_unit_order("u:t:v02:c001") > parse_unit_order("u:t:v01:c009")


def test_chapter_ordering_is_numeric_not_lexical() -> None:
    """`c010` must sort after `c009`, which string comparison would get wrong."""
    assert parse_unit_order("u:t:v01:c010") > parse_unit_order("u:t:v01:c009")


def test_an_unparseable_unit_id_yields_none() -> None:
    assert parse_unit_order("not-a-unit") is None
    assert parse_unit_order("") is None
    assert parse_unit_order(None) is None


def test_build_unit_order_covers_every_id() -> None:
    order = build_unit_order(["u:t:v01:c001", "u:t:v01:c002"])
    assert order == {"u:t:v01:c001": 1001, "u:t:v01:c002": 1002}


# ══ assertions use the latest evidence ══════════════════════════════
def test_an_assertion_is_dated_by_its_latest_evidence() -> None:
    """You cannot evaluate a claim fully until you have read all of its support."""
    result = as_of(_assertion("u:t:v01:c001", "u:t:v01:c003"), unit_order=ORDER)
    assert result.order == 1003
    assert result.source == "evidence"
    assert result.unit_id == "u:t:v01:c003"
    assert result.known is True


def test_a_single_evidence_assertion_uses_that_unit() -> None:
    result = as_of(_assertion("u:t:v01:c002"), unit_order=ORDER)
    assert result.order == 1002


def test_evidence_given_as_plain_ids_is_accepted() -> None:
    record = {"assertion_id": "a:1", "evidence": ["u:t:v01:c001", "u:t:v01:c002"]}
    assert as_of(record, unit_order=ORDER).order == 1002


# ══ entities use first_unit ═════════════════════════════════════════
def test_an_entity_is_dated_by_first_unit() -> None:
    """An entity becomes visible when it appears, not when it last does."""
    record = {"entity_id": "e:1", "first_unit": "u:t:v01:c002", "last_unit": "u:t:v02:c001"}
    result = as_of(record, unit_order=ORDER)
    assert result.order == 1002
    assert result.source == "first_unit"


def test_first_unit_wins_over_evidence_for_an_entity() -> None:
    """Both anchors present: the entity's own appearance is the disclosure point."""
    record = {
        "entity_id": "e:1",
        "first_unit": "u:t:v01:c002",
        "evidence": [{"unit_id": "u:t:v02:c001"}],
    }
    result = as_of(record, unit_order=ORDER)
    assert result.order == 1002
    assert result.source == "first_unit"


# ══ unit-bound records ══════════════════════════════════════════════
def test_a_mention_is_dated_by_its_unit() -> None:
    result = as_of({"mention_id": "m:1", "unit_id": "u:t:v01:c001"}, unit_order=ORDER)
    assert result.order == 1001
    assert result.source == "unit_id"


def test_a_citation_is_dated_by_its_unit() -> None:
    result = as_of({"citation_id": "c:1", "unit_id": "u:t:v01:c003"}, unit_order=ORDER)
    assert result.order == 1003


# ══ unknown is reported, never guessed ══════════════════════════════
def test_a_record_with_no_anchors_is_unknown() -> None:
    result = as_of({"assertion_id": "a:1"}, unit_order=ORDER)
    assert result.known is False
    assert result.order is None
    assert result.source == "unknown"


def test_an_unresolvable_unit_is_unknown_not_guessed() -> None:
    """A unit missing from the map must not borrow a neighbour's position."""
    record = {"mention_id": "m:1", "unit_id": "u:t:v09:c099"}
    result = as_of(record, unit_order=ORDER)
    assert result.known is False
    assert result.order is None


def test_an_empty_evidence_list_is_unknown() -> None:
    assert as_of({"assertion_id": "a:1", "evidence": []}, unit_order=ORDER).known is False


def test_the_unknown_result_is_serialisable() -> None:
    import json

    payload = as_of({"assertion_id": "a:1"}, unit_order=ORDER).as_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["known"] is False


# ══ order can be derived when no map is supplied ════════════════════
def test_order_is_derived_from_the_id_when_no_map_is_given() -> None:
    """Box consumers do not carry a unit-order table; the id is structured, so it is not a guess."""
    result = as_of(_assertion("u:t:v01:c001", "u:t:v01:c003"))
    assert result.order == 1003
    assert result.known is True


def test_an_explicit_map_wins_over_the_parsed_order() -> None:
    result = as_of({"mention_id": "m:1", "unit_id": "u:t:v01:c001"}, unit_order={"u:t:v01:c001": 42})
    assert result.order == 42


# ══ the anchor is reported ══════════════════════════════════════════
def test_the_candidates_are_reported_for_audit() -> None:
    """A thin release list needs to be explainable, not guessed at."""
    result = as_of(_assertion("u:t:v01:c001", "u:t:v01:c003"), unit_order=ORDER)
    assert "u:t:v01:c001" in result.candidates
    assert "u:t:v01:c003" in result.candidates


def test_as_of_is_a_dataclass_with_the_documented_fields() -> None:
    assert isinstance(as_of({"mention_id": "m", "unit_id": "u:t:v01:c001"}, unit_order=ORDER), AsOf)

# ══ aggregate rows: max_order is authoritative ══════════════════════
def test_a_record_with_max_order_is_dated_by_it() -> None:
    """A summary carries a precomputed position, because its evidence list may be trimmed to nothing."""
    row = {"summary_id": "sm:1", "max_order": 1002, "text": "摘要"}
    result = as_of(row, unit_order=ORDER)
    assert result.order == 1002
    assert result.source == "max_order"
    assert result.known is True


def test_max_order_wins_over_evidence() -> None:
    """The stored position was computed from the units that actually fed the text."""
    row = {"summary_id": "sm:1", "max_order": 1003, "evidence": [{"unit_id": "u:t:v01:c001"}]}
    assert as_of(row, unit_order=ORDER).order == 1003


def test_max_order_none_does_not_fall_through_to_an_optimistic_anchor() -> None:
    """An explicitly unknown aggregate must not be dated from a trimmed evidence list."""
    row = {"summary_id": "sm:1", "max_order": None, "evidence": [{"unit_id": "u:t:v01:c001"}]}
    result = as_of(row, unit_order=ORDER)
    assert result.known is False


def test_a_non_integer_max_order_is_ignored() -> None:
    row = {"summary_id": "sm:1", "max_order": "not-a-number", "unit_id": "u:t:v01:c001"}
    assert as_of(row, unit_order=ORDER).order == 1001
