"""Edge strength must be an aggregate, not one assertion's self-report.

Measured gap (D3)
-----------------
`box/projection_tables.py:226` sets:

    "strength": float(assertion.get("confidence") or 0.0)

and each edge is keyed by `left|right|predicate|assertion_id`, so there is exactly one assertion behind
every edge and `strength` is that assertion's confidence verbatim. §5.15 asks for the opposite: a
relation's strength aggregates over the **distinct source units** that assert it — one unit repeating
itself is not corroboration, two units agreeing is.

The practical consequence: a claim asserted once by a low-confidence extraction and a claim asserted in
five chapters look identical to a reader, and a single repeated unit inflates a relation as if it had
been corroborated.

What is pinned
--------------
* an edge groups by `(from, to, predicate)` — the assertion id is provenance, not identity;
* strength rises with distinct units and does **not** rise when one unit repeats itself;
* the aggregate is a stated formula, and the row says which formula produced it;
* `support_units` is carried, so a reader can see what the number rests on.
"""

from __future__ import annotations

from cbb_stage1.box.projection_tables import build_graph_edges

BOOK = "t"


def _assertion(
    assertion_id: str,
    unit_id: str,
    confidence: float,
    *,
    left: str = "e:t:person:aaaaaaaaaaaa",
    right: str = "e:t:person:bbbbbbbbbbbb",
    predicate: str = "敌对",
) -> dict:
    return {
        "assertion_id": assertion_id,
        "book_id": BOOK,
        "predicate": predicate,
        "library": "关系",
        "roles": {"施事": left, "受事": right},
        "claim_layer": "fact",
        "confidence": confidence,
        "confidence_method": "evidence_span",
        "status": "asserted",
        "evidence": [
            {
                "unit_id": unit_id,
                "start": 0,
                "end": 4,
                "quote": "对阵",
                "occurrence": 1,
            }
        ],
    }


# ══ edges group by the relation, not the assertion ═══════════════════
def test_two_assertions_of_the_same_relation_make_one_edge() -> None:
    rows = build_graph_edges(
        BOOK,
        [
            _assertion("a:1", "u:t:v01:c001", 0.9),
            _assertion("a:2", "u:t:v01:c002", 0.7),
        ],
    )
    assert len(rows) == 1, f"the same relation produced {len(rows)} edges"


def test_the_edge_records_every_assertion_behind_it() -> None:
    rows = build_graph_edges(
        BOOK,
        [
            _assertion("a:1", "u:t:v01:c001", 0.9),
            _assertion("a:2", "u:t:v01:c002", 0.7),
        ],
    )
    row = rows[0]
    assert row.get("assertion_ids"), "the edge does not list the assertions it aggregates"
    assert len(row["assertion_ids"]) == 2


# ══ corroboration raises strength; repetition does not ════════════════
def test_more_distinct_units_raise_strength() -> None:
    one = build_graph_edges(BOOK, [_assertion("a:1", "u:t:v01:c001", 0.6)])[0]["strength"]
    three = build_graph_edges(
        BOOK,
        [
            _assertion("a:1", "u:t:v01:c001", 0.6),
            _assertion("a:2", "u:t:v01:c002", 0.6),
            _assertion("a:3", "u:t:v01:c003", 0.6),
        ],
    )[0]["strength"]
    assert three > one, (
        f"three units asserting the same relation ({three}) is not stronger than one ({one}); "
        "corroboration is the whole point of aggregating"
    )


def test_repeating_one_unit_does_not_raise_strength() -> None:
    """One chapter saying it twice is not two chapters saying it."""
    once = build_graph_edges(BOOK, [_assertion("a:1", "u:t:v01:c001", 0.6)])[0]["strength"]
    twice = build_graph_edges(
        BOOK,
        [
            _assertion("a:1", "u:t:v01:c001", 0.6),
            _assertion("a:2", "u:t:v01:c001", 0.6),
        ],
    )[0]["strength"]
    assert twice == once, (
        f"a repeated unit raised strength from {once} to {twice}; distinct units are what count"
    )


# ══ the row says how the number was made ═════════════════════════════
def test_the_edge_states_the_aggregation_method() -> None:
    row = build_graph_edges(BOOK, [_assertion("a:1", "u:t:v01:c001", 0.6)])[0]
    assert row.get("strength_method"), "the edge does not say how strength was computed"


def test_the_edge_carries_its_support_units() -> None:
    rows = build_graph_edges(
        BOOK,
        [
            _assertion("a:1", "u:t:v01:c001", 0.9),
            _assertion("a:2", "u:t:v01:c002", 0.7),
        ],
    )
    assert sorted(rows[0]["support_units"]) == ["u:t:v01:c001", "u:t:v01:c002"]


def test_strength_stays_within_the_probability_range() -> None:
    rows = build_graph_edges(
        BOOK, [_assertion(f"a:{i}", f"u:t:v01:c{i:03d}", 1.0) for i in range(1, 8)]
    )
    assert 0.0 <= rows[0]["strength"] <= 1.0


def test_a_single_assertion_keeps_its_own_confidence() -> None:
    """Aggregating one item must not move the number."""
    row = build_graph_edges(BOOK, [_assertion("a:1", "u:t:v01:c001", 0.42)])[0]
    assert abs(row["strength"] - 0.42) < 1e-9
