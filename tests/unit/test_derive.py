"""Unit tests for the derived layer and incremental dependency closure.

Covers the interfaces added in P4:

* ``derive/summaries.py`` -- summary tree shape/closure, code-marked staleness;
* ``derive/hypotheses.py`` -- Abductive skeleton (derived-only, I26 rules);
* ``incremental/deps.py`` -- entity-level deps + transitive dirty closure.
"""

from __future__ import annotations

import pytest

from cbb_stage1.derive.hypotheses import (
    HYPOTHESIS_SCOPES,
    Hypothesis,
    abduce,
    derive_hypotheses,
    make_hypothesis_id,
    normalize_hypothesis_record,
)
from cbb_stage1.derive.summaries import (
    SUMMARY_CONTRACT,
    SummaryNode,
    build_summary_tree,
    make_summary_id,
    mark_stale,
    summary_rows,
    validate_summary_tree,
)
from cbb_stage1.incremental.deps import (
    build_dependency_graph,
    dirty_closure,
    entity_deps,
)

U1 = "u:book:v01:c001"
U2 = "u:book:v01:c002"
U3 = "u:book:v02:c003"
E1 = "e:book:person:aaaaaaaaaaaa"
E2 = "e:book:location:bbbbbbbbbbbb"


def unit(uid: str, order: int, volume: int = 1, chapter: int | None = None) -> dict:
    return {
        "unit_id": uid,
        "book_id": "book",
        "volume": volume,
        "chapter": chapter or order,
        "title": f"第{order}章",
        "order": order,
    }


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def test_summary_tree_shape_parents_and_levels():
    tree = build_summary_tree(
        [unit(U1, 1), unit(U2, 2), unit(U3, 3, volume=2)],
        unit_summaries={U2: "第二章摘要"},
    )
    assert tree["contract"] == SUMMARY_CONTRACT
    assert tree["book_id"] == "book"
    nodes = tree["nodes"]
    by_scope: dict[str, list[dict]] = {}
    for node in nodes:
        by_scope.setdefault(node["scope"], []).append(node)
    assert len(by_scope["book"]) == 1
    assert len(by_scope["volume"]) == 2
    assert len(by_scope["chapter"]) == 3

    root = by_scope["book"][0]
    assert root["level"] == 0 and root["parent_summary_id"] is None
    assert tree["root"] == root["summary_id"]
    volume_one = next(v for v in by_scope["volume"] if v["target"] == "v01")
    assert volume_one["level"] == 1
    assert volume_one["parent_summary_id"] == root["summary_id"]
    chapter_two = next(c for c in by_scope["chapter"] if c["target"] == U2)
    assert chapter_two["level"] == 2
    assert chapter_two["parent_summary_id"] == volume_one["summary_id"]
    assert chapter_two["text"] == "第二章摘要"
    assert chapter_two["status"] == "derived"
    assert chapter_two["stale"] is False
    # Stage-1 skeleton: no LLM text for unprovided units.
    chapter_one = next(c for c in by_scope["chapter"] if c["target"] == U1)
    assert chapter_one["text"] is None
    assert validate_summary_tree(tree) == []
    assert summary_rows(tree) == nodes


def test_summary_id_is_content_addressed_and_scope_checked():
    assert make_summary_id("book", "chapter", U1) == make_summary_id("book", "chapter", U1)
    assert make_summary_id("book", "chapter", U1) != make_summary_id("book", "chapter", U2)
    assert make_summary_id("book", "chapter", U1).startswith("sm:")
    with pytest.raises(ValueError):
        make_summary_id("book", "paragraph", U1)
    with pytest.raises(ValueError):
        SummaryNode(
            summary_id="sm:x",
            book_id="book",
            scope="paragraph",
            target=U1,
            level=0,
        )


def test_summary_tree_rejects_bad_units():
    with pytest.raises(ValueError):
        build_summary_tree([])
    with pytest.raises(ValueError):
        build_summary_tree([unit(U1, 1), unit(U1, 2)])
    with pytest.raises(ValueError):
        build_summary_tree([unit(U1, 1), {**unit(U2, 2), "book_id": "other"}])


def test_mark_stale_marks_leaf_and_ancestors_without_mutating():
    tree = build_summary_tree([unit(U1, 1), unit(U2, 2)])
    stale = mark_stale(tree, [U2, "u:book:v01:missing"])
    by_target = {node["target"]: node for node in stale["nodes"]}
    assert by_target[U2]["stale"] is True
    assert by_target["v01"]["stale"] is True
    assert by_target["book"]["stale"] is True
    assert by_target[U1]["stale"] is False
    # input tree untouched
    assert all(node["stale"] is False for node in tree["nodes"])


def test_validate_summary_tree_detects_missing_parent_and_cycle():
    broken = {
        "contract": SUMMARY_CONTRACT,
        "root": "sm:root",
        "nodes": [
            {"summary_id": "sm:root", "parent_summary_id": "sm:a", "scope": "book"},
            {"summary_id": "sm:a", "parent_summary_id": "sm:root", "scope": "volume"},
            {"summary_id": "sm:b", "parent_summary_id": "sm:missing", "scope": "chapter"},
        ],
    }
    errors = validate_summary_tree(broken)
    assert any("cycle" in error for error in errors)
    assert any("parent missing" in error for error in errors)
    with pytest.raises(ValueError):
        SummaryNode(
            summary_id="sm:u",
            book_id="book",
            scope="chapter",
            target=U1,
            level=2,
            parent_summary_id="sm:u",
        )


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------


def test_abduce_requires_missing_premises():
    with pytest.raises(ValueError):
        abduce("book", "黑之魔王与言万心叶敌对", missing_premises=[])


def test_abduce_creates_derived_only_candidate_record():
    hypothesis = abduce(
        "book",
        "黑之魔王与言万心叶敌对",
        scope="foreshadow",
        missing_premises=["两人的直接冲突场景"],
        support=[{"unit_id": U1, "quote": "黑之魔王站在船头"}],
        consistency="pass",
        plausibility="medium",
        confidence=0.4,
        derived_from=[U1],
        created_at="2026-10-18T20:00:00+08:00",
    )
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.hypothesis_id == make_hypothesis_id(
        "book", "foreshadow", "黑之魔王与言万心叶敌对"
    )
    record = hypothesis.to_record()
    assert record["hypothesis_id"].startswith("hy:")
    assert record["scope"] == "foreshadow"
    assert record["evidence_sufficiency"] == "insufficient"
    assert record["missing_premises"] == ["两人的直接冲突场景"]
    assert record["status"] == "candidate"
    assert record["method"] == "abductive_rag"
    assert record["derived_from"] == [U1]
    assert record["created_at"] == "2026-10-18T20:00:00+08:00"


def test_hypothesis_rules_reject_sufficient_with_missing_and_insufficient_without():
    with pytest.raises(ValueError):
        Hypothesis(
            book_id="book",
            scope="setting",
            claim="x",
            evidence_sufficiency="sufficient",
            missing_premises=("gap",),
        )
    with pytest.raises(ValueError):
        Hypothesis(
            book_id="book",
            scope="setting",
            claim="x",
            evidence_sufficiency="insufficient",
        )
    for bad_scope in ("paragraph", ""):
        with pytest.raises(ValueError):
            Hypothesis(
                book_id="book",
                scope=bad_scope,
                claim="x",
                evidence_sufficiency="sufficient",
            )
    with pytest.raises(ValueError):
        Hypothesis(
            book_id="book",
            scope="setting",
            claim="x",
            evidence_sufficiency="sufficient",
            confidence=1.5,
        )


def test_derive_hypotheses_skips_sufficient_and_empty_gaps():
    observations = [
        {"claim": "有充分证据的归纳", "evidence_sufficiency": "sufficient"},
        {"claim": "没有缺口的候选", "missing_premises": []},
        {"claim": "", "missing_premises": ["x"]},
        {
            "claim": "需要补前提的假设",
            "scope": "motivation",
            "missing_premises": ["动机段"],
            "support": [{"unit_id": U1, "quote": "q"}],
        },
    ]
    hypotheses, skipped = derive_hypotheses("book", observations)
    assert [item.claim for item in hypotheses] == ["需要补前提的假设"]
    assert hypotheses[0].scope == "motivation"
    assert {item["reason"] for item in skipped} == {
        "sufficient_evidence",
        "no_missing_premises",
        "empty_claim",
    }


def test_normalize_hypothesis_record_defaults_and_validation():
    record = normalize_hypothesis_record(
        {
            "claim": "灵石汇率稳定",
            "missing_premises": ["多次交易场景"],
            "book_id": "book",
            "scope": "setting",
        },
        created_at="2026-10-18T20:00:00+08:00",
    )
    assert record["hypothesis_id"].startswith("hy:")
    assert record["status"] == "candidate"
    assert record["consistency"] == "unknown"
    assert record["created_at"] == "2026-10-18T20:00:00+08:00"
    assert set(record) == {
        "hypothesis_id",
        "book_id",
        "scope",
        "claim",
        "evidence_sufficiency",
        "missing_premises",
        "support",
        "against",
        "consistency",
        "plausibility",
        "status",
        "confidence",
        "method",
        "derived_from",
        "created_at",
    }
    with pytest.raises(ValueError):
        normalize_hypothesis_record({"claim": "x", "scope": "bad"})
    with pytest.raises(ValueError):
        normalize_hypothesis_record(
            {"claim": "x", "evidence_sufficiency": "sufficient", "missing_premises": ["y"]}
        )
    assert HYPOTHESIS_SCOPES == ("setting", "foreshadow", "motivation", "prediction")


# ---------------------------------------------------------------------------
# dependency graph / dirty closure
# ---------------------------------------------------------------------------


def _graph_records() -> dict:
    return {
        "units": [
            {"unit_id": U1, "book_id": "book"},
            {"unit_id": U2, "book_id": "book"},
        ],
        "entities": [{"entity_id": E1}, {"entity_id": E2}],
        "mentions": [
            {"mention_id": "m:1", "unit_id": U1, "entity_id": E1},
            {"mention_id": "m:2", "unit_id": U2, "entity_id": E2},
        ],
        "assertions": [
            {
                "assertion_id": "a:1",
                "deps": ["m:1"],
                "roles": {"主体": E1, "值": {"literal": "x"}},
                "evidence": [{"unit_id": U1}],
            },
            {
                "assertion_id": "a:2",
                "roles": {"主体": E2, "值": {"literal": "y"}},
                "evidence": [{"unit_id": U2}],
            },
        ],
        "events": [
            {
                "event_id": "ev:1",
                "unit_id": U1,
                "participants": [E1],
                "evidence": [{"unit_id": U1}],
            }
        ],
        "cues": [
            {
                "cue_id": "cue:1",
                "subject": {"entity_id": E1},
                "setup": {"unit_id": U1, "evidence": [{"unit_id": U1}]},
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "hy:1",
                "derived_from": ["a:1"],
                "support": [{"unit_id": U1}],
            }
        ],
    }


def test_build_dependency_graph_forward_reverse_and_entity_refs():
    graph = build_dependency_graph(**_graph_records())
    assert "m:1" in graph.deps["a:1"]
    assert E1 in graph.deps["a:1"]
    assert "a:1" in graph.dependents["m:1"]
    assert "a:1" in graph.dependents[E1]
    assert "hy:1" in graph.dependents["a:1"]
    assert graph.entity_refs["a:1"] == (E1,)
    assert graph.entity_refs["ev:1"] == (E1,)


def test_dirty_closure_from_changed_unit_reaches_entities_assertions_events_cues():
    closure = dirty_closure(changed_units=[U1], **_graph_records())
    assert closure.changed_units == (U1,)
    assert U1 in closure.units
    assert "m:1" in closure.mentions
    assert E1 in closure.entities
    assert "a:1" in closure.assertions
    assert "ev:1" in closure.events
    assert "cue:1" in closure.cues
    assert "hy:1" in closure.hypotheses
    # U2-only records stay clean.
    assert "a:2" not in closure.assertions
    assert E2 not in closure.entities
    assert closure.to_manifest_fields() == {
        "changed_units": [U1],
        "dirty_entities": [E1],
    }
    assert closure.reasons["m:1"] == (U1,)


def test_dirty_closure_from_changed_entity_uses_reverse_edges():
    closure = dirty_closure(changed_entities=[E1], **_graph_records())
    assert closure.entities == (E1,)
    assert "a:1" in closure.assertions
    assert "ev:1" in closure.events
    assert "cue:1" in closure.cues
    assert "hy:1" in closure.hypotheses
    assert "a:2" not in closure.assertions


def test_dirty_closure_is_cycle_safe_and_ignores_unknown_seeds():
    records = _graph_records()
    records["assertions"] = [
        {"assertion_id": "a:1", "deps": ["a:2"], "roles": {"主体": E1}},
        {"assertion_id": "a:2", "deps": ["a:1"], "roles": {"主体": E1}},
    ]
    closure = dirty_closure(changed_entities=[E1], **records)
    assert closure.assertions == ("a:1", "a:2")
    unknown = dirty_closure(changed_units=["u:book:v09:c999"], **records)
    assert unknown.changed_units == ("u:book:v09:c999",)
    assert unknown.dirty_ids == ("u:book:v09:c999",)
    assert unknown.to_dict()["changed_units"] == ["u:book:v09:c999"]


def test_entity_deps_lists_dependent_records_per_entity():
    records = _graph_records()
    deps = entity_deps(
        mentions=records["mentions"],
        assertions=records["assertions"],
        events=records["events"],
        cues=records["cues"],
        hypotheses=records["hypotheses"],
    )
    assert deps[E1] == ("a:1", "cue:1", "ev:1", "m:1")
    assert deps[E2] == ("a:2", "m:2")
