"""E4 regression: optional embedding KNN for entity/description dedupe.

The A12 half of ``cbb_stage1.dedupe`` used to ship a deliberate stand-in: its
``knn_candidates`` docstring said "deterministic, no embeddings" and described
itself as a "pure semhash stand-in for the upstream embedding KNN", while the
artifact said ``method="knn_cluster"``. These tests pin the contract:

* **no embedder** -> the deterministic token-bag path runs and the artifact
  records ``method="semhash"`` (never the misleading ``knn_cluster``);
* **embedder** -> vectors are embedded once and candidates use cosine KNN,
  recorded as ``method="embedding_knn"``;
* the embedder reaches the stage through ``upstream["embedder"]`` (the
  orchestrator's hand-off used by ``derive`` as well).

Mandated negatives: a missing embedder never fabricates embeddings; the
embedder is only called when supplied.
"""

from __future__ import annotations

import json

import pytest

from cbb_stage1.dedupe.integration import run as run_dedupe
from cbb_stage1.dedupe.semhash import (
    dedupe_clusters,
    dedupe_graph,
    knn_candidates,
)


class MappingEmbedder:
    """Explicit text -> vector map (callable) that records every batch it sees."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        missing = [text for text in texts if text not in self.vectors]
        if missing:
            raise AssertionError(f"test embedder has no vector for {missing!r}")
        return [list(self.vectors[text]) for text in texts]


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# knn_candidates / dedupe_clusters: pure functions
# ---------------------------------------------------------------------------


def test_knn_candidates_without_embedder_is_semhash():
    rows = knn_candidates(["Alpha One", "Alpha Two"], threshold=0.95)
    assert rows == ()  # token Jaccard 1/3, far below 0.95


def test_knn_candidates_with_embedder_uses_cosine():
    embedder = MappingEmbedder(
        {"Alpha One": [1.0, 1.0], "Alpha Two": [1.0, 1.0], "Beta Nine": [1.0, 0.0]}
    )

    rows = knn_candidates(
        ["Alpha One", "Alpha Two", "Beta Nine"], threshold=0.9, embedder=embedder
    )

    assert [(row.left_index, row.right_index) for row in rows] == [(0, 1)]
    assert rows[0].similarity == pytest.approx(1.0)
    # One batch for all items: the embedder is not called per pair.
    assert embedder.calls == [["Alpha One", "Alpha Two", "Beta Nine"]]


def test_dedupe_clusters_without_embedder_records_semhash():
    result = dedupe_clusters(["Alpha One", "Alpha Two"], threshold=0.95)

    assert result.method == "semhash"
    assert result.selected == ("Alpha One", "Alpha Two")
    assert result.knn_pairs == 0
    assert result.candidates == ()
    assert "Method: semhash" in result.stats()


def test_dedupe_clusters_with_embedder_records_embedding_knn():
    embedder = MappingEmbedder(
        {"Alpha One": [1.0, 1.0], "Alpha Two": [1.0, 1.0], "Beta Nine": [1.0, 0.0]}
    )

    result = dedupe_clusters(
        ["Alpha One", "Alpha Two", "Beta Nine"], threshold=0.9, embedder=embedder
    )

    assert result.method == "embedding_knn"
    assert result.selected == ("Alpha One", "Beta Nine")
    assert result.mapping["Alpha Two"] == "Alpha One"
    assert result.knn_pairs == 1
    # Candidate hints stay reusable (the integration no longer re-embeds).
    assert [(row.left_index, row.right_index) for row in result.candidates] == [(0, 1)]
    assert "Method: embedding_knn" in result.stats()


def test_embedder_validation_and_zero_vector():
    with pytest.raises(TypeError):
        dedupe_clusters(["A"], embedder=object())  # type: ignore[arg-type]

    class Short:
        def __call__(self, texts):
            return [[1.0]]

    with pytest.raises(ValueError, match="vectors for"):
        dedupe_clusters(["A", "B"], embedder=Short())

    zero = MappingEmbedder({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    result = dedupe_clusters(["A", "B"], threshold=0.9, embedder=zero)
    assert result.selected == ("A", "B")  # zero vectors never count as similar


def test_dedupe_graph_methods_need_their_prerequisites():
    with pytest.raises(NotImplementedError, match="embedder"):
        dedupe_graph([], [], [], method="lm_based")

    embedder = MappingEmbedder(
        {"A": [1.0, 0.0], "Alpha One": [1.0, 0.0], "Alpha Two": [0.0, 1.0]}
    )
    with pytest.raises(NotImplementedError, match="merger"):
        dedupe_graph(["A"], [], [], method="lm_based", embedder=embedder)

    graph = dedupe_graph(["Alpha One", "Alpha Two"], [], [], embedder=embedder)
    assert graph.method == "embedding_knn"
    assert graph.entities == ("Alpha One", "Alpha Two")  # orthogonal vectors


# ---------------------------------------------------------------------------
# dedupe.integration: artifact method must be truthful
# ---------------------------------------------------------------------------


def test_integration_without_embedder_writes_semhash_method(tmp_path):
    work = tmp_path / "work"
    outputs = [
        _output("u:t:v01:c001", [{"canonical_name": "Alpha One"}]),
        _output("u:t:v01:c002", [{"canonical_name": "Alpha Two"}]),
    ]

    result = run_dedupe(work, book_id="t", outputs=outputs, units=[], created_at="")

    assert result["method"] == "semhash"
    dedupe_map = _read_json(work / "assemble" / "dedupe_map.json")
    entities = dedupe_map["entities"]
    assert entities["method"] == "semhash"
    assert entities["similarity"] == "jaccard"
    assert entities["knn_candidates"] == []
    assert dedupe_map["descriptions"]["method"] == "semhash"
    assert dedupe_map["descriptions"]["similarity"] == "exact_fragment"
    assert len(_read_jsonl(work / "assemble" / "entity_cards.jsonl")) == 2


def test_integration_with_embedder_writes_embedding_knn_method(tmp_path):
    work = tmp_path / "work"
    embedder = MappingEmbedder(
        {
            "Alpha One": [1.0, 1.0],
            "Alpha Two": [1.0, 1.0],
            "Beta Nine": [1.0, 0.0],
            "他笑了": [1.0, 0.0],
            "他 笑了": [1.0, 0.0],
        }
    )
    outputs = [
        _output(
            "u:t:v01:c001",
            [
                {"canonical_name": "Alpha One", "description": "他笑了"},
                {"canonical_name": "Beta Nine"},
            ],
        ),
        _output(
            "u:t:v01:c002",
            [
                {"canonical_name": "Alpha Two", "description": "他 笑了"},
            ],
        ),
    ]

    result = run_dedupe(
        work, book_id="t", outputs=outputs, units=[], created_at="", embedder=embedder
    )

    assert result["method"] == "embedding_knn"
    dedupe_map = _read_json(work / "assemble" / "dedupe_map.json")
    entities = dedupe_map["entities"]
    assert entities["method"] == "embedding_knn"
    assert entities["similarity"] == "cosine"
    assert entities["mapping"]["Alpha Two"] == "Alpha One"
    assert entities["knn"] == {"pairs": 1, "merger_calls": 0, "mergers_applied": 0}
    assert entities["knn_candidates"] == [
        {"a": "Alpha One", "b": "Alpha Two", "similarity": 1.0}
    ]

    descriptions = dedupe_map["descriptions"]
    assert descriptions["method"] == "embedding_knn"
    assert descriptions["similarity"] == "cosine"
    assert descriptions["duplicates"] == {"他 笑了": "他笑了"}

    cards = _read_jsonl(work / "assemble" / "entity_cards.jsonl")
    assert [card["canonical_name"] for card in cards] == ["Alpha One", "Beta Nine"]


def test_integration_inherits_embedder_from_upstream(tmp_path):
    """The orchestrator hands the embedder over through ``upstream``."""
    work = tmp_path / "work"
    embedder = MappingEmbedder({"Alpha One": [1.0, 1.0], "Alpha Two": [1.0, 1.0]})
    outputs = [
        _output("u:t:v01:c001", [{"canonical_name": "Alpha One"}]),
        _output("u:t:v01:c002", [{"canonical_name": "Alpha Two"}]),
    ]

    result = run_dedupe(
        work,
        book_id="t",
        outputs=outputs,
        units=[],
        created_at="",
        upstream={"embedder": embedder},
    )

    assert result["method"] == "embedding_knn"
    assert _read_json(work / "assemble" / "dedupe_map.json")["entities"]["method"] == (
        "embedding_knn"
    )
