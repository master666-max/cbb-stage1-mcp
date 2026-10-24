"""Postprocess integration for the dedupe stage.

Unified interface (design P0-A/P0-D)::

    run(work_root, *, book_id, outputs, units, created_at, upstream) -> dict

Inputs come from the pipeline work layer, never from the box:

* ``outputs`` -- engine candidate dicts (``unit_id`` + ``entities`` at least);
* ``units`` -- unit records from ``work/plan/units.jsonl``;
* ``work/resolve/entity_candidates.jsonl`` -- the resolve stage's exact-name
  candidate table (optional).

Outputs (work layer only, the box writer consumes them):

* ``work/assemble/entity_cards.jsonl`` -- deterministic entity cards with
  semantic-hash deduplicated names, exact-deduplicated observed descriptions,
  and ``description=None`` (no LLM synthesis in this pass);
* ``work/assemble/dedupe_map.json`` -- the reversible mapping: near-duplicate
  surfaces -> canonical surface, description fragments -> kept fragment, plus
  the A12 KNN half: candidate pairs (hints only), connected-component clusters,
  and ``lm_merger`` (``"unavailable"`` when no in-cluster merger was injected --
  the deterministic first-seen member is used and no decision is invented).
  ``method`` is truthful: ``"semhash"`` for the deterministic token-bag path,
  ``"embedding_knn"`` when an optional ``embedder`` produced cosine KNN.
  Nothing is deleted from the source silently; this file is the audit trail for
  every merge decision.

The embedder is optional and injected (or read from ``upstream["embedder"]`` by
the postprocess orchestrator). Without it the pre-existing deterministic path
runs unchanged; embedding failures are loud, never a silent downgrade.

Boundaries: no box imports, no LLM, no fabrication. If there is no input the
stage writes an explicit empty table + empty mapping, not invented rows.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1.dedupe.describe import (
    MapReduceSynthesizer,
    Observation,
    dedupe_fragments,
    describe,
)
from cbb_stage1.orchestrator.parallel import stage_workers
from cbb_stage1.dedupe.semhash import (
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_METHOD,
    SEMHASH_METHOD,
    ClusterDedupeResult,
    ClusterMerger,
    dedupe_clusters,
    normalize_text,
)

__all__ = ["CONTRACT", "run"]

#: Contract marker for ``work/assemble/dedupe_map.json``.
CONTRACT = "cbb-dedupe-map/1"

DEFAULT_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD


def _row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump())
    raise TypeError(f"expected mapping/pydantic model, got {type(value).__name__}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: JSONL row must be an object")
        out.append(value)
    return out


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False) for row in rows]
    _atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _clean_list(values: Any) -> list[str]:
    out: list[str] = []
    for value in values or ():
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _collect_entity_data(
    outputs: Iterable[Any],
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate name/type/alias/unit/description evidence by canonical name."""
    data: dict[str, dict[str, Any]] = {}

    def entry(name: str) -> dict[str, Any]:
        return data.setdefault(
            name,
            {
                "canonical_name": name,
                "type": "unknown",
                "aliases": [],
                "units": [],
                "mention_count": 0,
                "descriptions": [],
            },
        )

    for output in outputs:
        record = _row(output)
        unit_id = str(record.get("unit_id") or "")
        for raw in record.get("entities") or ():
            entity = _row(raw)
            name = str(entity.get("canonical_name") or entity.get("name") or "")
            if not name.strip():
                continue
            item = entry(name)
            type_value = str(entity.get("type") or "").strip()
            if type_value and item["type"] == "unknown":
                item["type"] = type_value
            for alias in _clean_list(entity.get("aliases")):
                if alias != name and alias not in item["aliases"]:
                    item["aliases"].append(alias)
            if unit_id and unit_id not in item["units"]:
                item["units"].append(unit_id)
            description = entity.get("description")
            if isinstance(description, str) and description.strip():
                if description not in item["descriptions"]:
                    item["descriptions"].append(description)

    for raw in candidates:
        candidate = _row(raw)
        name = str(candidate.get("canonical_name") or "")
        if not name.strip():
            continue
        item = entry(name)
        type_value = str(candidate.get("type") or "").strip()
        if type_value and item["type"] == "unknown":
            item["type"] = type_value
        for alias in _clean_list(candidate.get("aliases")):
            if alias != name and alias not in item["aliases"]:
                item["aliases"].append(alias)
        for unit_id in _clean_list(candidate.get("units")):
            if unit_id not in item["units"]:
                item["units"].append(unit_id)
        item["mention_count"] += int(candidate.get("mention_count") or 0)
    return data


def _merge_canonical(
    data: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int | None = None,
    merger: ClusterMerger | None = None,
    embedder: Any | None = None,
    embedding_threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    merge_workers: int = 1,
) -> tuple[dict[str, dict[str, Any]], ClusterDedupeResult]:
    """KNN-cluster the entity names; merge evidence into each canonical name."""
    surfaces = sorted(data)
    result = dedupe_clusters(
        surfaces,
        threshold=threshold if embedder is None else embedding_threshold,
        top_k=top_k,
        merger=merger,
        embedder=embedder,
        merge_workers=merge_workers,
    )
    merged: dict[str, dict[str, Any]] = {}
    for cluster, canonical in zip(result.clusters, result.selected):
        target = {
            "canonical_name": canonical,
            "type": "unknown",
            "aliases": [],
            "units": [],
            "mention_count": 0,
            "descriptions": [],
        }
        for member in cluster:
            item = data[member]
            if target["type"] == "unknown" and item["type"] != "unknown":
                target["type"] = item["type"]
            # A whitespace/case/width variant is not a real alias: keep it in
            # the reversible mapping only, not in the card's alias list.
            if (
                member != canonical
                and normalize_text(member) != normalize_text(canonical)
                and member not in target["aliases"]
            ):
                target["aliases"].append(member)
            for alias in item["aliases"]:
                if (
                    normalize_text(alias) != normalize_text(canonical)
                    and alias not in target["aliases"]
                ):
                    target["aliases"].append(alias)
            for unit_id in item["units"]:
                if unit_id not in target["units"]:
                    target["units"].append(unit_id)
            target["mention_count"] += int(item["mention_count"] or 0)
            for description in item["descriptions"]:
                if description not in target["descriptions"]:
                    target["descriptions"].append(description)
        merged[canonical] = target
    return merged, result


def _unit_order_map(units: Iterable[Any]) -> dict[str, int]:
    order: dict[str, int] = {}
    for index, raw in enumerate(units or (), 1):
        unit = _row(raw)
        unit_id = str(unit.get("unit_id") or "")
        if unit_id:
            order[unit_id] = int(unit.get("order") or index)
    return order


class LLMClusterMerger:
    """Ask the auxiliary LLM to choose one canonical surface from a cluster."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def merge(self, cluster: Sequence[str]) -> str | None:
        members = list(cluster)
        prompt = (
            "只从下列候选名中选一个规范名，只输出这个名字，不要解释。\n"
            + "\n".join(f"- {item}" for item in members)
        )
        answer = str(self.llm.generate(prompt)).strip().strip("\"'`")
        return answer if answer in members else None


def _dedupe_description_fragments(
    fragments: Sequence[str],
    *,
    embedder: Any | None,
    embedding_threshold: float,
    top_k: int | None,
) -> dict[str, Any]:
    """Reversible description-fragment dedupe (exact, or embedding KNN).

    Without an embedder this is the pre-existing exact (character-identical
    modulo whitespace/NFKC) path. With one, fragments are clustered by cosine
    KNN; ``method`` says which actually ran.
    """
    if embedder is None:
        result = dedupe_fragments(fragments)
        return {
            "method": SEMHASH_METHOD,
            "similarity": "exact_fragment",
            "mapping": dict(result.mapping),
            "duplicates": dict(result.duplicates),
            "kept": list(result.kept),
            "dropped": list(result.dropped),
            "ignored": list(result.ignored),
            "clusters": [list(cluster) for cluster in result.clusters],
            "knn_candidates": [],
            "knn": {"pairs": 0},
            "stats": {
                "total": result.total_items,
                "kept": result.deduplicated_items,
                "duplicates": result.duplicate_items,
                "reduction": round(result.reduction, 4),
            },
        }

    clustered = dedupe_clusters(
        fragments,
        threshold=embedding_threshold,
        top_k=top_k,
        embedder=embedder,
    )
    dropped = [member for cluster in clustered.clusters for member in cluster[1:]]
    return {
        "method": EMBEDDING_METHOD,
        "similarity": "cosine",
        "mapping": dict(clustered.mapping),
        "duplicates": dict(clustered.duplicates),
        "kept": list(clustered.selected),
        "dropped": dropped,
        "ignored": [],
        "clusters": [list(cluster) for cluster in clustered.clusters],
        "knn_candidates": [candidate.as_record() for candidate in clustered.candidates],
        "knn": {"pairs": clustered.knn_pairs},
        "stats": {
            "total": clustered.total_items,
            "kept": clustered.deduplicated_items,
            "duplicates": clustered.duplicate_items,
            "reduction": round(clustered.reduction, 4),
        },
    }


def run(
    work_root: str | Path,
    *,
    book_id: str,
    outputs: Iterable[Any] = (),
    units: Iterable[Any] = (),
    created_at: str = "",
    upstream: Mapping[str, Any] | None = None,
    knn_threshold: float = DEFAULT_THRESHOLD,
    knn_top_k: int | None = None,
    knn_embedding_threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    merger: ClusterMerger | None = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Dedupe names/descriptions and write ``work/assemble/entity_cards.jsonl``.

    Entity names go through the A12 KNN half: candidate pairs are recorded as
    hints in ``dedupe_map.entities.knn_candidates``, and connected-component
    clusters are merged with an optional in-cluster ``merger`` (the LM decision).
    Without a merger the map says ``lm_merger="unavailable"`` -- no score or
    merge decision is invented. The mapping/clusters stay reversible.

    Args:
        embedder: Optional ``texts -> vectors`` callable (or object exposing
            ``.encode``). When supplied (directly or via
            ``upstream["embedder"]``) entity *and* description dedupe use cosine
            KNN and the artifact records ``method="embedding_knn"``; otherwise
            the deterministic semhash path runs and ``method="semhash"``.

    Returns:
        A JSON-serialisable receipt ``{"stage","status","method","counts",
        "paths","warnings"}`` for the postprocess wiring log.
    """
    work_root = Path(work_root)
    units = list(units or ())
    outputs = list(outputs or ())
    order = _unit_order_map(units)
    upstream = upstream or {}
    inherited_embedder = upstream.get("embedder") if isinstance(upstream, Mapping) else None
    embedder = embedder if embedder is not None else inherited_embedder

    candidates = _read_jsonl(work_root / "resolve" / "entity_candidates.jsonl")
    data = _collect_entity_data(outputs, candidates)
    # How many cluster merges to ask for at once is this run's decision, not the library's:
    # the worker count is the same one the other postprocess stages use.
    merged, name_result = _merge_canonical(
        data,
        threshold=knn_threshold,
        top_k=knn_top_k,
        merger=merger,
        embedder=embedder,
        embedding_threshold=knn_embedding_threshold,
        merge_workers=stage_workers(upstream, len(data)),
    )
    # The clustering result already carries its candidate pairs: reusing them
    # avoids a second embedding round-trip for the same surfaces.
    knn_hints = name_result.candidates

    warnings: list[str] = []
    cards: list[dict[str, Any]] = []
    all_descriptions: list[str] = []
    config = upstream.get("config")
    aux_llm = upstream.get("llm")
    merger = merger or upstream.get("merger")
    if merger is None and aux_llm is not None:
        merger = LLMClusterMerger(aux_llm)
    synthesizer = None
    if (
        aux_llm is not None
        and config is not None
        and bool(getattr(config, "description_synthesis", False))
    ):
        synthesizer = MapReduceSynthesizer(aux_llm.generate)
    # A14: batch entity merging, opt-in because it costs one LLM call per 80 candidates. The fan-out filter
    # inside `batch_merge` runs first, so the model is only asked about names that survived it — that order
    # is the source's, and it is what stops transitive closure absorbing generic terms.
    batch_merge_map: dict[str, Any] | None = None
    if aux_llm is not None and bool(upstream.get("batch_merge")):
        from cbb_stage1.dedupe.batch_merge import batch_merge as _batch_merge

        candidates = [
            {
                "name": canonical,
                "type": item.get("type") or "concept",
                "desc": "",
                "aliases": sorted(item.get("aliases") or ()),
                "freq": len(item.get("units") or ()),
            }
            for canonical, item in merged.items()
        ]
        try:
            merged_result = _batch_merge(candidates, aux_llm.generate)
        except Exception as exc:  # noqa: BLE001 - a failed merge must not lose the stage
            warnings.append(f"batch_merge failed: {type(exc).__name__}: {exc}")
            merged_result = None
        if merged_result is not None:
            batch_merge_map = merged_result
            _write_jsonl(
                work_root / "assemble" / "merge_groups.jsonl",
                merged_result["groups"],
            )
            counts["batch_merge_groups"] = len(merged_result["groups"])
            counts["batch_merge_candidates"] = merged_result["candidates"]
            counts["batch_merge_batches"] = merged_result["batches"]
            counts["batch_merge_generic"] = len(merged_result["generic"])
            if merged_result["failures"]:
                warnings.extend(
                    f"batch_merge {item}" for item in merged_result["failures"][:3]
                )

    for canonical in sorted(merged):
        item = merged[canonical]
        observations = tuple(Observation(text=text) for text in item["descriptions"])
        described = describe(
            f"dedupe:{canonical}", observations, synthesizer=synthesizer
        )
        observed = [observation.text for observation in described.observations]
        # The dedupe map is computed over the raw fragments (so dropped
        # duplicates stay auditable), while the card shows the kept set.
        all_descriptions.extend(item["descriptions"])
        unit_ids = sorted(set(item["units"]), key=lambda uid: order.get(uid, 10**9))
        cards.append(
            {
                "canonical_name": canonical,
                "type": item["type"],
                "aliases": sorted(set(item["aliases"])),
                "observed_descriptions": observed,
                "description": described.text,
                "description_mode": described.mode,
                "units": unit_ids,
                "mention_count": int(item["mention_count"]),
                "produced_by": "dedupe:integration",
            }
        )

    description_result = _dedupe_description_fragments(
        all_descriptions,
        embedder=embedder,
        embedding_threshold=knn_embedding_threshold,
        top_k=knn_top_k,
    )
    dedupe_map: dict[str, Any] = {
        "contract": CONTRACT,
        "book_id": book_id,
        "created_at": created_at,
        "threshold": knn_threshold,
        "entities": {
            "method": name_result.method,
            "similarity": "cosine" if embedder is not None else "jaccard",
            "threshold": knn_threshold if embedder is None else knn_embedding_threshold,
            "top_k": knn_top_k,
            "lm_merger": name_result.lm_merger,
            "reversible": name_result.reversible,
            "mapping": dict(name_result.mapping),
            "duplicates": dict(name_result.duplicates),
            "clusters": [list(cluster) for cluster in name_result.clusters],
            "knn_candidates": [candidate.as_record() for candidate in knn_hints],
            "knn": {
                "pairs": name_result.knn_pairs,
                "merger_calls": name_result.merger_calls,
                "mergers_applied": name_result.mergers_applied,
            },
            "notes": list(name_result.notes),
            "stats": {
                "total": name_result.total_items,
                "kept": name_result.deduplicated_items,
                "duplicates": name_result.duplicate_items,
                "reduction": round(name_result.reduction, 4),
            },
        },
        "descriptions": description_result,
        # A14's merge decisions, so a reader can audit which names the model grouped and which the fan-out
        # filter removed. `None` when the pass was not requested.
        "batch_merge": batch_merge_map,
    }

    cards_path = work_root / "assemble" / "entity_cards.jsonl"
    map_path = work_root / "assemble" / "dedupe_map.json"
    _write_jsonl(cards_path, cards)
    _write_json(map_path, dedupe_map)

    if not cards:
        warnings.append("no entity names in resolve/engine inputs; entity_cards is explicitly empty")
    if not outputs:
        warnings.append("no engine outputs supplied to dedupe integration")
    if merger is None and any(len(cluster) > 1 for cluster in name_result.clusters):
        warnings.append(
            "lm merger unavailable: multi-member entity clusters kept the first-seen "
            "member (dedupe_map.entities.lm_merger='unavailable'; no LM decision made)"
        )

    return {
        "stage": "dedupe",
        "status": "ok",
        "method": name_result.method,
        "lm_merger": name_result.lm_merger,
        "counts": {
            "entity_cards": len(cards),
            "entity_surfaces": name_result.total_items,
            "entity_duplicates": name_result.duplicate_items,
            "entity_knn_pairs": name_result.knn_pairs,
            "entity_merger_applied": name_result.mergers_applied,
            "description_fragments": description_result["stats"]["total"],
            "description_duplicates": description_result["stats"]["duplicates"],
        },
        "paths": {
            "entity_cards": cards_path.relative_to(work_root).as_posix(),
            "dedupe_map": map_path.relative_to(work_root).as_posix(),
        },
        "warnings": warnings,
    }
