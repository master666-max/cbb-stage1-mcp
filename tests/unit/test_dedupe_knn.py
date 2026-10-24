"""Unit tests for the A12 KNN/cluster half of ``cbb_stage1.dedupe``.

The greedy :func:`dedupe_items` (no transitive union) is covered by
``test_dedupe.py`` and stays untouched.  This file covers the additive A12
machinery from the v6 design (``00`` §4.12: semhash / KNN clustering + in-cluster
LM): deterministic candidate pairs, connected-component clusters, the optional
merger protocol, and the integration wiring that records candidates + clusters in
``dedupe_map`` while keeping every merge reversible.

Mandated negatives: with no merger the result says ``lm_merger="unavailable"`` and
never invents a merge decision or score; a merger that returns a non-member is a
rule error, not a silent pass.
"""

from __future__ import annotations

import json

import pytest

from cbb_stage1.dedupe.integration import run as run_dedupe
from cbb_stage1.dedupe.semhash import (
    ClusterDedupeResult,
    DedupeRuleError,
    KnnCandidate,
    dedupe_clusters,
    dedupe_items,
    knn_candidates,
    similarity,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeMerger:
    """Records offered clusters and replays pre-set decisions (``None`` = hold)."""

    def __init__(self, choices):
        self._choices = list(choices)
        self.calls: list[tuple[str, ...]] = []

    def merge(self, cluster):
        self.calls.append(tuple(cluster))
        return self._choices.pop(0)


def _output(unit_id: str, entities: list[dict]) -> dict:
    for entity in entities:
        entity.setdefault("type", "person")
        entity.setdefault("aliases", [])
        entity.setdefault("units", [unit_id])
        entity.setdefault("mention_count", 1)
    return {"unit_id": unit_id, "entities": entities}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# knn_candidates
# ---------------------------------------------------------------------------


def test_knn_candidates_are_deterministic_thresholded_pairs():
    rows = knn_candidates(
        ["New York", "New York City", "new york", "新约"], threshold=0.6
    )

    assert [(row.left_index, row.right_index) for row in rows] == [(0, 2), (0, 1), (1, 2)]
    assert [round(row.similarity, 4) for row in rows] == [1.0, 0.6667, 0.6667]
    assert all(isinstance(row, KnnCandidate) for row in rows)
    assert rows[0].as_record() == {
        "a": "New York",
        "b": "new york",
        "similarity": 1.0,
    }
    # No relation to 新约; candidate rows are symmetric hints, never merges.
    assert all("新约" not in (row.left, row.right) for row in rows)


def test_knn_candidates_top_k_limits_per_left_item():
    items = ["t1 t2 t3", "t1 t2 t3 t4", "t1 t2 t3 t5", "t1 t2 t3 t6 t7"]
    everything = knn_candidates(items, threshold=0.6)
    top_one = knn_candidates(items, threshold=0.6, top_k=1)

    assert len(everything) > len(top_one)
    assert [(row.left_index, row.right_index) for row in top_one] == [(0, 1), (1, 2)]
    # top_k is applied per left item, then the flat list is re-sorted deterministically.
    assert [row.left_index for row in top_one] == [0, 1]


def test_knn_candidates_empty_and_validation():
    assert knn_candidates([]) == ()
    with pytest.raises(ValueError):
        knn_candidates(["x"], threshold=0.0)
    with pytest.raises(TypeError):
        knn_candidates(["x"], threshold="0.9")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        knn_candidates(["x", 1])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        knn_candidates(["x"], top_k=0)
    with pytest.raises(TypeError):
        knn_candidates(["x"], top_k="1")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# dedupe_clusters
# ---------------------------------------------------------------------------


def test_dedupe_clusters_unions_transitively_while_greedy_does_not():
    items = ["t1 t2 t3 t4", "t1 t2 t3", "t1 t2 t3 t5"]

    clustered = dedupe_clusters(items, threshold=0.65)
    greedy = dedupe_items(items, threshold=0.65)

    assert similarity(items[0], items[2]) < 0.65  # a~c is below the threshold
    assert clustered.selected == ("t1 t2 t3 t4",)
    assert clustered.clusters == (("t1 t2 t3 t4", "t1 t2 t3", "t1 t2 t3 t5"),)
    assert clustered.mapping == {
        "t1 t2 t3 t4": "t1 t2 t3 t4",
        "t1 t2 t3": "t1 t2 t3 t4",
        "t1 t2 t3 t5": "t1 t2 t3 t4",
    }
    assert greedy.selected == ("t1 t2 t3 t4", "t1 t2 t3 t5")  # no union


def test_dedupe_clusters_is_reversible_and_stats_are_consistent():
    result = dedupe_clusters(["A", "a", "A "])

    assert isinstance(result, ClusterDedupeResult)
    assert result.selected == ("A",)
    assert result.reversible is True
    assert result.total_items == 3
    assert result.deduplicated_items == 1
    assert result.duplicate_items == 2
    assert result.reduction == pytest.approx(200 / 3)
    assert result.canonical_of("a") == "A"
    assert result.rollback("A") == ("A", "a", "A ")
    assert "Method: semhash" in result.stats()
    assert "LM merger: unavailable" in result.stats()


def test_dedupe_clusters_merger_chooses_another_member():
    merger = FakeMerger(["t1 t2 t3"])

    result = dedupe_clusters(
        ["t1 t2 t3 t4", "t1 t2 t3", "t1 t2 t3 t5"], threshold=0.65, merger=merger
    )

    assert merger.calls == [("t1 t2 t3 t4", "t1 t2 t3", "t1 t2 t3 t5")]
    assert result.selected == ("t1 t2 t3",)
    assert result.mapping["t1 t2 t3 t4"] == "t1 t2 t3"
    assert result.mapping["t1 t2 t3 t5"] == "t1 t2 t3"
    assert result.lm_merger == "provided"
    assert result.merger_calls == 1
    assert result.mergers_applied == 1
    # The cluster audit trail is unchanged by the merger's choice.
    assert result.clusters == (("t1 t2 t3 t4", "t1 t2 t3", "t1 t2 t3 t5"),)
    assert result.rollback("t1 t2 t3") == ("t1 t2 t3 t4", "t1 t2 t3", "t1 t2 t3 t5")


def test_dedupe_clusters_merger_none_falls_back_to_first_member():
    merger = FakeMerger([None])

    result = dedupe_clusters(["A", "a"], merger=merger)

    assert result.selected == ("A",)
    assert result.lm_merger == "provided"
    assert result.merger_calls == 1
    assert result.mergers_applied == 0


def test_dedupe_clusters_merger_must_return_a_member():
    with pytest.raises(DedupeRuleError):
        dedupe_clusters(["A", "a"], merger=FakeMerger(["B"]))
    with pytest.raises(DedupeRuleError):
        dedupe_clusters(["A", "a"], merger=FakeMerger([42]))  # type: ignore[list-item]


def test_dedupe_clusters_without_merger_is_explicitly_unavailable():
    result = dedupe_clusters(["A", "a", "B"])

    assert result.lm_merger == "unavailable"
    assert result.merger_calls == 0
    assert result.mergers_applied == 0
    assert result.notes and "lm_merger unavailable" in result.notes[0]

    singleton = dedupe_clusters(["A", "B"])
    assert singleton.lm_merger == "unavailable"
    assert singleton.notes == ()  # no multi-member cluster: no caveat needed


def test_dedupe_clusters_empty_and_validation():
    empty = dedupe_clusters([])
    assert empty.selected == ()
    assert empty.total_items == 0
    assert empty.mapping == {}
    assert empty.lm_merger == "unavailable"

    with pytest.raises(ValueError):
        dedupe_clusters(["x"], threshold=1.5)
    with pytest.raises(TypeError):
        dedupe_clusters(["x", 1])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        dedupe_clusters(["x"], merger=object())  # type: ignore[arg-type]


def test_dedupe_clusters_is_deterministic_across_runs():
    items = ["New York", "New York City", "new york", "New Jersey"]
    first = dedupe_clusters(items, threshold=0.5)
    second = dedupe_clusters(items, threshold=0.5)

    assert first.selected == second.selected
    assert first.mapping == second.mapping
    assert first.clusters == second.clusters


# ---------------------------------------------------------------------------
# dedupe.integration wiring
# ---------------------------------------------------------------------------


def test_run_records_knn_candidates_and_clusters_reversibly(tmp_path):
    work = tmp_path / "work"
    outputs = [
        _output("u:t:v01:c001", [{"canonical_name": "言万心叶"}, {"canonical_name": "黑之魔王"}]),
        _output("u:t:v01:c002", [{"canonical_name": "言万心叶 "}]),
    ]

    result = run_dedupe(work, book_id="t", outputs=outputs, units=[], created_at="2026-10-19T00:00:00")

    assert result["stage"] == "dedupe"
    assert result["status"] == "ok"
    assert result["lm_merger"] == "unavailable"
    assert result["counts"]["entity_cards"] == 2
    assert result["counts"]["entity_surfaces"] == 3
    assert result["counts"]["entity_duplicates"] == 1
    assert result["counts"]["entity_knn_pairs"] == 1

    cards = _read_jsonl(work / "assemble" / "entity_cards.jsonl")
    assert [card["canonical_name"] for card in cards] == ["言万心叶", "黑之魔王"]

    dedupe_map = _read_json(work / "assemble" / "dedupe_map.json")
    assert dedupe_map["contract"] == "cbb-dedupe-map/1"
    entities = dedupe_map["entities"]
    assert entities["method"] == "semhash"
    assert entities["similarity"] == "jaccard"
    assert entities["lm_merger"] == "unavailable"
    assert entities["reversible"] is True
    assert entities["mapping"]["言万心叶 "] == "言万心叶"
    assert entities["clusters"] == [["言万心叶", "言万心叶 "], ["黑之魔王"]]
    assert entities["knn_candidates"] == [
        {"a": "言万心叶", "b": "言万心叶 ", "similarity": 1.0}
    ]
    assert entities["knn"] == {"pairs": 1, "merger_calls": 0, "mergers_applied": 0}
    assert entities["stats"] == {"total": 3, "kept": 2, "duplicates": 1, "reduction": 33.3333}
    # Reversibility: the whole cluster stays recoverable from the map.
    assert entities["mapping"]["言万心叶"] == "言万心叶"
    assert any("lm merger unavailable" in warning for warning in result["warnings"])


def test_run_with_merger_uses_the_lm_decision(tmp_path):
    work = tmp_path / "work"
    merger = FakeMerger(["言万心叶 "])
    outputs = [
        _output("u:t:v01:c001", [{"canonical_name": "言万心叶"}, {"canonical_name": "黑之魔王"}]),
        _output("u:t:v01:c002", [{"canonical_name": "言万心叶 "}]),
    ]

    result = run_dedupe(
        work, book_id="t", outputs=outputs, units=[],
        created_at="2026-10-19T00:00:00", merger=merger,
    )

    assert merger.calls == [("言万心叶", "言万心叶 ")]
    assert result["lm_merger"] == "provided"
    assert result["counts"]["entity_merger_applied"] == 1

    cards = _read_jsonl(work / "assemble" / "entity_cards.jsonl")
    assert [card["canonical_name"] for card in cards] == ["言万心叶 ", "黑之魔王"]

    entities = _read_json(work / "assemble" / "dedupe_map.json")["entities"]
    assert entities["lm_merger"] == "provided"
    assert entities["knn"]["mergers_applied"] == 1
    assert entities["mapping"] == {
        "言万心叶": "言万心叶 ",
        "言万心叶 ": "言万心叶 ",
        "黑之魔王": "黑之魔王",
    }
    assert entities["clusters"] == [["言万心叶", "言万心叶 "], ["黑之魔王"]]


def test_run_with_threshold_override_changes_clustering(tmp_path):
    outputs = [_output("u:t:v01:c001", [{"canonical_name": "New York"},
                                        {"canonical_name": "New York City"}])]

    default_work = tmp_path / "default"
    run_dedupe(default_work, book_id="t", outputs=outputs, units=[], created_at="")
    assert _read_jsonl(default_work / "assemble" / "entity_cards.jsonl")[0:] and len(
        _read_jsonl(default_work / "assemble" / "entity_cards.jsonl")
    ) == 2

    loose_work = tmp_path / "loose"
    run_dedupe(loose_work, book_id="t", outputs=outputs, units=[], created_at="",
               knn_threshold=0.6)
    assert len(_read_jsonl(loose_work / "assemble" / "entity_cards.jsonl")) == 1
    entities = _read_json(loose_work / "assemble" / "dedupe_map.json")["entities"]
    assert entities["threshold"] == 0.6
    assert entities["knn_candidates"] == [
        {"a": "New York", "b": "New York City", "similarity": 0.666667}
    ]


def test_run_without_input_keeps_the_explicit_empty_shape(tmp_path):
    work = tmp_path / "work"

    result = run_dedupe(work, book_id="t", outputs=[], units=[], created_at="")

    assert result["counts"]["entity_cards"] == 0
    assert result["counts"]["entity_knn_pairs"] == 0
    assert _read_jsonl(work / "assemble" / "entity_cards.jsonl") == []
    entities = _read_json(work / "assemble" / "dedupe_map.json")["entities"]
    assert entities["clusters"] == []
    assert entities["knn_candidates"] == []
    assert entities["lm_merger"] == "unavailable"
    assert entities["method"] == "semhash"
    assert (
        _read_json(work / "assemble" / "dedupe_map.json")["descriptions"]["method"]
        == "semhash"
    )
