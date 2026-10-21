"""Recursive RAPTOR-style clustering tree (structural by default).

RAPTOR's paper recursively clusters embeddings and summarises each cluster.  The
stage-1 runtime has no embedding model by default, so this module builds the
*cluster tree* deterministically:

* level 1 = units clustered per volume;
* levels 2..N = recursive agglomeration of the previous level's clusters;
* ``parent_cluster_id`` points from a child to its parent (top-level rows have
  ``None``); ``child_cluster_ids`` on parents lists the children;
* ``text`` is always ``None`` here -- a later synthesis pass writes summaries;
  no prose is invented.

Similarity:

* an ``embedder`` (callable ``texts -> vectors`` or object with ``.encode``)
  enables ``method="embedding_cosine"``; unit texts are embedded once and
  higher-level clusters use their centroid vectors;
* without an embedder the fallback is deterministic token-Jaccard over the
  union of member unit tokens, ``method="structural_token_jaccard"``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1 import ids

__all__ = [
    "STRUCTURAL_METHOD",
    "EMBEDDING_METHOD",
    "build_raptor_clusters",
    "validate_raptor_tree",
]

STRUCTURAL_METHOD = "structural_token_jaccard"
EMBEDDING_METHOD = "embedding_cosine"

_TOKEN = re.compile(r"[\u4e00-\u9fff]{1,4}|[A-Za-z]{2,}")
_DEFAULT_STRUCTURAL_THRESHOLD = 0.18
_DEFAULT_EMBEDDING_THRESHOLD = 0.75


def _bag(text: str) -> set[str]:
    return set(_TOKEN.findall(text or ""))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid(vectors: Iterable[Sequence[float]]) -> list[float]:
    gathered = [list(vector) for vector in vectors]
    if not gathered:
        return []
    size = len(gathered[0])
    return [sum(vector[i] for vector in gathered) / len(gathered) for i in range(size)]


def _embed(embedder: Any, texts: list[str]) -> list[list[float]]:
    """Call a callable/``.encode`` embedder and validate the vector matrix."""
    if hasattr(embedder, "encode") and callable(embedder.encode):
        raw = embedder.encode(texts)
    elif callable(embedder):
        raw = embedder(texts)
    else:
        raise TypeError("embedder must be callable or expose .encode(texts)")
    vectors = [list(map(float, vector)) for vector in raw]
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(texts)} texts"
        )
    if vectors:
        width = len(vectors[0])
        if width == 0 or any(len(vector) != width for vector in vectors):
            raise ValueError("embedder vectors must be non-empty and equal length")
    return vectors


def _cluster_indices(
    scores: list[list[float]],
    count: int,
    threshold: float,
) -> list[list[int]]:
    """Greedy first-fit clustering: each item joins its most similar open cluster."""
    clusters: list[list[int]] = []
    for index in range(count):
        best = -1
        best_score = 0.0
        for cluster_index, members in enumerate(clusters):
            score = max((scores[index][member] for member in members), default=0.0)
            if score > best_score:
                best, best_score = cluster_index, score
        if best >= 0 and best_score >= threshold:
            clusters[best].append(index)
        else:
            clusters.append([index])
    return clusters


class _LevelItem:
    """One clusterable item: member units + structural bag + embedding centroid."""

    __slots__ = ("member_units", "bag", "vector", "cluster_id")

    def __init__(
        self,
        member_units: Iterable[str],
        bag: set[str],
        vector: list[float] | None = None,
        cluster_id: str | None = None,
    ) -> None:
        self.member_units = sorted(set(member_units))
        self.bag = set(bag)
        self.vector = vector
        self.cluster_id = cluster_id


def build_raptor_clusters(
    book_id: str,
    units: Iterable[Mapping[str, Any]],
    unit_texts: Mapping[str, str],
    *,
    embedder: Any | None = None,
    max_levels: int = 4,
    structural_threshold: float = _DEFAULT_STRUCTURAL_THRESHOLD,
    embedding_threshold: float = _DEFAULT_EMBEDDING_THRESHOLD,
    threshold_decay: float = 0.6,
) -> list[dict[str, Any]]:
    """Build a recursive cluster tree over units (per volume).

    Args:
        book_id: Book id used for content-addressed cluster ids.
        units: Unit records (``unit_id``/``order``/``volume``).
        unit_texts: unit_id -> text (missing text => empty token bag).
        embedder: Optional callable/``.encode`` embedding backend. When given,
            unit texts are embedded once and clustering uses cosine similarity.
        max_levels: Maximum tree height (level 1 leaves; ``>=1``).
        structural_threshold: Jaccard threshold for the structural fallback.
        embedding_threshold: Cosine threshold for the embedding path.
        threshold_decay: Per-level multiplier for the similarity threshold. A
            strictly same-threshold greedy tree collapses to one level (any
            cross-level merge candidate would already have merged at level 1),
            so higher levels relax the threshold by this factor.

    Returns:
        Flat rows sorted by ``(level, cluster_id)``. Rows with
        ``level > 1`` carry ``child_cluster_ids``; children carry
        ``parent_cluster_id``. ``text`` is always ``None``.

    Raises:
        ValueError: non-positive ``max_levels`` or a malformed embedder result.
        TypeError: ``embedder`` is neither callable nor ``.encode``.
    """
    if int(max_levels) < 1:
        raise ValueError("max_levels must be >= 1")
    if not 0.0 < float(threshold_decay) <= 1.0:
        raise ValueError("threshold_decay must be in (0, 1]")
    method = EMBEDDING_METHOD if embedder is not None else STRUCTURAL_METHOD
    produced_by = "derive:raptor_embedding" if embedder is not None else "derive:raptor_structural"

    ordered_units = sorted(
        units,
        key=lambda unit: (int(unit.get("volume") or 1), int(unit.get("order") or 0), str(unit.get("unit_id") or "")),
    )
    by_volume: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in ordered_units:
        try:
            volume = int(unit.get("volume") or 1)
        except (TypeError, ValueError):
            volume = 1
        by_volume[volume].append(unit)

    unit_ids = [str(unit.get("unit_id") or "") for unit in ordered_units]
    vectors: dict[str, list[float]] = {}
    if embedder is not None and unit_ids:
        embedded = _embed(embedder, [unit_texts.get(uid, "") for uid in unit_ids])
        vectors = dict(zip(unit_ids, embedded))

    rows: list[dict[str, Any]] = []
    for volume in sorted(by_volume):
        volume_units = by_volume[volume]
        items: list[_LevelItem] = []
        for unit in volume_units:
            unit_id = str(unit.get("unit_id") or "")
            bag = _bag(unit_texts.get(unit_id, ""))
            vector = vectors.get(unit_id) if embedder is not None else None
            if embedder is not None and vector is None:
                raise ValueError(f"embedder returned no vector for unit {unit_id}")
            items.append(_LevelItem([unit_id], bag, vector))

        level = 1
        while True:
            scores = _pair_scores(items, embedder is not None)
            base = embedding_threshold if embedder is not None else structural_threshold
            threshold = float(base) * (float(threshold_decay) ** (level - 1))
            groups = _cluster_indices(scores, len(items), threshold)
            if len(groups) >= len(items) and level > 1:
                # No higher-level merge happened: the previous level is the root.
                break
            level_rows = _emit_level_rows(
                book_id, volume, level, groups, items, method, produced_by
            )
            rows.extend(level_rows)
            if len(groups) <= 1 or level >= int(max_levels):
                break
            items = _next_level_items(groups, items, embedder is not None)
            level += 1

    rows.sort(key=lambda row: (int(row["level"]), str(row["cluster_id"])))
    _link_parents(rows)
    return rows


def _pair_scores(items: list[_LevelItem], embedding: bool) -> list[list[float]]:
    size = len(items)
    scores = [[0.0] * size for _ in range(size)]
    for i in range(size):
        scores[i][i] = 1.0
        for j in range(i + 1, size):
            if embedding:
                value = _cosine(items[i].vector or [], items[j].vector or [])
            else:
                value = _jaccard(items[i].bag, items[j].bag)
            scores[i][j] = scores[j][i] = value
    return scores


def _emit_level_rows(
    book_id: str,
    volume: int,
    level: int,
    groups: list[list[int]],
    items: list[_LevelItem],
    method: str,
    produced_by: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        member_units = sorted({uid for index in group for uid in items[index].member_units})
        cluster_id = ids.raptor_cluster_id(book_id, volume, level, member_units)
        rows.append(
            {
                "cluster_id": cluster_id,
                "book_id": book_id,
                "level": level,
                "volume": volume,
                "parent_cluster_id": None,
                "child_cluster_ids": [],
                "member_units": member_units,
                "text": None,
                "method": method,
                "produced_by": produced_by,
            }
        )
    return rows


def _next_level_items(
    groups: list[list[int]],
    items: list[_LevelItem],
    embedding: bool,
) -> list[_LevelItem]:
    next_items: list[_LevelItem] = []
    for group in groups:
        members = [items[index] for index in group]
        member_units = sorted({uid for item in members for uid in item.member_units})
        bag: set[str] = set()
        for item in members:
            bag |= item.bag
        vector = _centroid(item.vector or [] for item in members) if embedding else None
        next_items.append(_LevelItem(member_units, bag, vector))
    return next_items


def _link_parents(rows: list[dict[str, Any]]) -> None:
    """Set ``parent_cluster_id``/``child_cluster_ids`` from member containment."""
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[int(row["level"])].append(row)
    for level in sorted(by_level):
        parents = by_level.get(level + 1)
        if not parents:
            continue
        for child in by_level[level]:
            members = set(child["member_units"])
            for parent in parents:
                if members and members <= set(parent["member_units"]):
                    child["parent_cluster_id"] = parent["cluster_id"]
                    if child["cluster_id"] not in parent["child_cluster_ids"]:
                        parent["child_cluster_ids"].append(child["cluster_id"])
                    break
        for parent in parents:
            parent["child_cluster_ids"].sort()


def validate_raptor_tree(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Check id uniqueness, parent closure, levels, and child coverage."""
    errors: list[str] = []
    rows = list(rows)
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        cluster_id = str(row.get("cluster_id") or "")
        if not cluster_id:
            errors.append(f"row {index}: missing cluster_id")
            continue
        if cluster_id in by_id:
            errors.append(f"row {index}: duplicate cluster_id {cluster_id}")
        by_id[cluster_id] = row
    for cluster_id, row in by_id.items():
        parent = row.get("parent_cluster_id")
        if parent is None:
            continue
        if parent not in by_id:
            errors.append(f"cluster {cluster_id}: parent missing {parent}")
            continue
        if int(by_id[parent].get("level") or 0) <= int(row.get("level") or 0):
            errors.append(f"cluster {cluster_id}: parent level must be higher")
        if cluster_id not in (by_id[parent].get("child_cluster_ids") or ()):
            errors.append(f"cluster {cluster_id}: parent does not list it as a child")
    return errors
