"""P0-B integration for identity resolution (``run`` contract).

Signature (shared by resolve/assemble/claims integrations)::

    run(work_root, *, book_id, outputs, units, created_at, upstream) -> dict

``outputs`` are engine JSON dicts produced for each unit; ``units`` are the
``plan/units.jsonl`` records.  This stage is deterministic and writes only the
work-side identity tables:

* ``resolve/alias_map.json`` -- the engine's alias -> canonical-name map plus
  the fanout-clean report;
* ``resolve/entity_candidates.jsonl`` -- one record per surviving entity
  (``canonical_name``/``type``/``aliases``/``units``/``confidence``/``tier``;
  the box writer consumes exactly these fields);
* ``resolve/merges.jsonl`` -- reversible, acyclic alias merges
  (``identity/merges.jsonl`` shape) produced by :class:`MergeLedger`.

Identity ids are computed with the same content-addressed formula as the box
builder (``box/records.py`` ``make_entity_id``: ``e:{book}:{type_slug}:{sha256
(slug|name)[:12]}``) so the merge records reference ids the box will actually
create.  The formula is mirrored here on purpose: this stage must not import
``box.writer``; a test cross-checks the two formulas.

Nothing is fabricated: units that are not in the unit list are dropped, and a
candidate with no valid unit is skipped (reported in ``counts``/``notes``).
The work tables are rewritten atomically and are byte-stable for the same
inputs + ``created_at``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1 import ids
from cbb_stage1.models import Entity

from .alias import BLACKLIST, DEFAULT_FANOUT, build_alias_map_from_entities, fanout_clean
from .merges import MergeError, MergeLedger, make_merge

__all__ = ["STAGE", "alias_map_path", "entity_id_for", "entity_candidates_path", "merges_path", "run"]

STAGE = "resolve"
ALIAS_MAP_CONTRACT = "cbb-alias-map/1"

_TIERS = ("minor", "major", "canonical")
_TIER_RANK = {tier: index for index, tier in enumerate(_TIERS)}


# ── small path/shape helpers ────────────────────────────────────


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce mappings / pydantic models / dataclasses to a plain dict."""
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    raise TypeError(f"expected mapping/model/unit record, got {type(value).__name__}")


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - defensive
        return path.as_posix()


def _write_text_atomic(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _write_json_atomic(path: Path, payload: Any) -> Path:
    return _write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    data = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    return _write_text_atomic(path, data)


def alias_map_path(work_root: str | Path) -> Path:
    return Path(work_root) / "resolve" / "alias_map.json"


def entity_candidates_path(work_root: str | Path) -> Path:
    return Path(work_root) / "resolve" / "entity_candidates.jsonl"


def merges_path(work_root: str | Path) -> Path:
    return Path(work_root) / "resolve" / "merges.jsonl"


# ── box-compatible entity ids ───────────────────────────────────


def entity_id_for(book_id: str, entity_type: str, canonical_name: str) -> str:
    """Thin wrapper over :func:`cbb_stage1.ids.entity_id` (same bytes)."""
    return ids.entity_id(book_id, entity_type, canonical_name)


# ── unit / output ordering ──────────────────────────────────────


def _unit_records(units: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for unit in units or ():
        try:
            records.append(_as_dict(unit))
        except TypeError:
            continue
    return records


def _ordered_outputs(
    outputs: Sequence[Any], unit_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    order = {
        str(unit.get("unit_id")): index
        for index, unit in enumerate(unit_records)
        if unit.get("unit_id")
    }
    indexed: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for index, output in enumerate(outputs or ()):
        try:
            record = _as_dict(output)
        except TypeError:
            continue
        unit_id = str(record.get("unit_id") or "")
        indexed.append(((order.get(unit_id, len(order) + index), index), record))
    indexed.sort(key=lambda item: item[0])
    return [record for _, record in indexed]


# ── entity collection ──────────────────────────────────────────


@dataclass
class _SeenEntity:
    name: str
    etype: str
    aliases: list[str]
    units: list[str]
    confidence: float
    tier: str


def _clean_aliases(values: Iterable[Any], owner: str) -> list[str]:
    out: list[str] = []
    for value in values or ():
        alias = str(value or "").strip()
        if not alias or alias == owner or alias in out:
            continue
        out.append(alias)
    return out


def _collect_entities(
    ordered: Sequence[Mapping[str, Any]],
    *,
    valid_units: set[str],
    unit_order: Mapping[str, int],
) -> tuple[list[Entity], dict[tuple[str, str], _SeenEntity], int, int]:
    seen: dict[tuple[str, str], _SeenEntity] = {}
    skipped_no_unit = 0
    for output in ordered:
        unit_id = str(output.get("unit_id") or "")
        for row in output.get("entities") or ():
            try:
                entity = _as_dict(row)
            except TypeError:
                continue
            name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
            if not name:
                continue
            etype = str(entity.get("type") or entity.get("entity_type") or "unknown")
            slug = ids.entity_type_slug(etype)
            key = (name, slug)
            entry = seen.get(key)
            if entry is None:
                entry = _SeenEntity(
                    name=name,
                    etype=etype,
                    aliases=[],
                    units=[],
                    confidence=0.0,
                    tier="minor",
                )
                seen[key] = entry
            for alias in _clean_aliases(entity.get("aliases") or (), name):
                if alias not in entry.aliases:
                    entry.aliases.append(alias)
            if unit_id and unit_id in valid_units and unit_id not in entry.units:
                entry.units.append(unit_id)
            try:
                entry.confidence = max(entry.confidence, float(entity.get("confidence") or 0.0))
            except (TypeError, ValueError):
                pass
            tier = str(entity.get("tier") or "minor")
            if tier in _TIER_RANK and _TIER_RANK[tier] > _TIER_RANK.get(entry.tier, 0):
                entry.tier = tier

    for entry in seen.values():
        if not entry.units:
            skipped_no_unit += 1
        entry.units.sort(key=lambda uid: unit_order.get(uid, 10**9))
        entry.confidence = _clamp01(entry.confidence)

    entities = [
        Entity(
            book_id="",  # filled below by run()
            type=entry.etype,
            canonical_name=entry.name,
            aliases=list(entry.aliases),
            tier=entry.tier,  # type: ignore[arg-type]
            status="active",
            confidence=entry.confidence,
        )
        for entry in seen.values()
        if entry.units
    ]
    return entities, seen, skipped_no_unit, len(seen)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _collect_mention_candidates(
    ordered: Sequence[Mapping[str, Any]],
    *,
    book_id: str,
    valid_units: set[str],
    unit_order: Mapping[str, int],
    existing_names: set[str],
) -> list[dict[str, Any]]:
    """Low-confidence entity candidates for units that extracted no entities.

    The mock/spine engines only emit mentions; without this fallback the box
    would have no entities at all.  These candidates are explicitly marked
    ``candidate_source="mention"`` with type ``unknown`` and the AI Reader
    safety table is **not** applied here (the box/quality gate decides); the
    caller records a warning so this is never silent.
    """
    grouped: dict[str, list[str]] = {}
    confidence: dict[str, float] = {}
    for output in ordered:
        if output.get("entities"):
            continue  # this unit already produced real entity candidates
        unit_id = str(output.get("unit_id") or "")
        if unit_id and unit_id not in valid_units:
            continue
        for raw in output.get("mentions") or ():
            try:
                mention = _as_dict(raw)
            except TypeError:
                continue
            surface = str(mention.get("surface") or "").strip()
            if not surface or surface in existing_names:
                continue
            if str(mention.get("mention_type") or "name") == "pronoun":
                continue
            units = grouped.setdefault(surface, [])
            if unit_id and unit_id not in units:
                units.append(unit_id)
            try:
                confidence[surface] = max(
                    confidence.get(surface, 0.0), float(mention.get("confidence") or 0.0)
                )
            except (TypeError, ValueError):
                pass

    rows: list[dict[str, Any]] = []
    for surface in sorted(grouped):
        units = sorted(grouped[surface], key=lambda uid: unit_order.get(uid, 10**9))
        if not units:
            continue
        rows.append(
            {
                "book_id": book_id,
                "entity_id": entity_id_for(book_id, "unknown", surface),
                "canonical_name": surface,
                "type": "unknown",
                "type_slug": "unknown",
                "aliases": [],
                "units": units,
                "confidence": _clamp01(confidence.get(surface, 0.0)),
                "tier": "minor",
                "candidate_source": "mention",
                "produced_by": "integration:resolve",
            }
        )
    return rows


# ── the stage entry point ───────────────────────────────────────


def run(
    work_root: str | Path,
    *,
    book_id: str,
    outputs: list[dict],
    units: list[dict],
    created_at: str,
    upstream: dict,
) -> dict:
    """Build alias map + entity candidates + reversible merges under ``work/``."""
    root = Path(work_root)
    upstream = upstream or {}
    unit_records = _unit_records(units or [])
    ordered = _ordered_outputs(outputs or [], unit_records)
    unit_order = {
        str(unit.get("unit_id")): index
        for index, unit in enumerate(unit_records)
        if unit.get("unit_id")
    }
    valid_units = set(unit_order) or {
        str(output.get("unit_id")) for output in ordered if output.get("unit_id")
    }

    notes: list[str] = []
    if not ordered:
        notes.append("resolve: no engine outputs; empty identity tables written")
    if not valid_units:
        notes.append("resolve: no unit ids available; no entity candidates emitted")

    entities, seen, skipped_no_unit, collected = _collect_entities(
        ordered, valid_units=valid_units, unit_order=unit_order
    )
    for entity in entities:
        entity.book_id = book_id

    threshold = int(upstream.get("fanout_threshold") or DEFAULT_FANOUT)
    blacklist = set(BLACKLIST) | {str(item) for item in upstream.get("alias_blacklist") or ()}
    fanout = fanout_clean(entities, fanout_threshold=threshold, blacklist=blacklist)

    kept: list[Entity] = list(fanout.kept)
    candidates: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    skipped_after_fanout = 0
    ordered_kept = sorted(kept, key=lambda item: (item.canonical_name, ids.entity_type_slug(item.type)))
    for entity in ordered_kept:
        key = (entity.canonical_name, ids.entity_type_slug(entity.type))
        entry = seen.get(key)
        units_for_entity = list(entry.units) if entry else []
        if not units_for_entity:
            skipped_after_fanout += 1
            continue
        candidate = {
            "book_id": book_id,
            "entity_id": entity_id_for(book_id, entity.type, entity.canonical_name),
            "canonical_name": entity.canonical_name,
            "type": entity.type,
            "type_slug": ids.entity_type_slug(entity.type),
            "aliases": list(entity.aliases),
            "units": units_for_entity,
            "confidence": _clamp01(entity.confidence),
            "tier": entity.tier,
            "produced_by": "integration:resolve",
        }
        candidates.append(candidate)
        by_name.setdefault(candidate["canonical_name"], candidate)

    mention_candidates = _collect_mention_candidates(
        ordered,
        book_id=book_id,
        valid_units=valid_units,
        unit_order=unit_order,
        existing_names=set(by_name),
    )
    for candidate in mention_candidates:
        candidates.append(candidate)
        by_name.setdefault(candidate["canonical_name"], candidate)

    alias_map = build_alias_map_from_entities(ordered_kept)

    # alias map surface -> canonical name; only surfaces that are themselves a
    # surviving canonical name become an entity-id merge.
    merge_pairs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for surface in sorted(alias_map):
        canonical = alias_map[surface]
        source = by_name.get(surface)
        target = by_name.get(canonical)
        if source is None or target is None:
            continue
        if source["entity_id"] == target["entity_id"]:
            continue
        merge_pairs.append((source["canonical_name"], target["canonical_name"], source, target))

    ledger = MergeLedger(book_id=book_id)
    merge_rows: list[dict[str, Any]] = []
    for _source_name, _target_name, source, target in sorted(
        merge_pairs, key=lambda item: (item[3]["entity_id"], item[2]["entity_id"])
    ):
        record = make_merge(
            book_id,
            [source["entity_id"]],
            target["entity_id"],
            kind="alias",
            decided_by="rule",
            confidence=min(source["confidence"], target["confidence"]),
            evidence=[],
            at=created_at,
        )
        try:
            stored = ledger.apply(record)
        except MergeError as exc:  # pragma: no cover - union-find groups cannot cycle
            raise RuntimeError(f"resolve: non-reversible merge rejected: {exc}") from exc
        merge_rows.append(stored.box_dict())

    # Reversibility / acyclicity gate: every source resolves to its target and
    # the merged-into map is finite.
    merged_into = ledger.merged_into_map()
    for source_id, target_id in merged_into.items():
        if ledger.resolve(source_id) != target_id:
            raise RuntimeError(f"resolve: merge resolution mismatch for {source_id}")
    expected = len({source["entity_id"] for _, _, source, _ in merge_pairs})
    if len(merged_into) != expected:
        raise RuntimeError(
            f"resolve: expected {expected} merged ids, ledger has {len(merged_into)}"
        )

    alias_payload = {
        "contract": ALIAS_MAP_CONTRACT,
        "book_id": book_id,
        "stage": STAGE,
        "created_at": created_at,
        "entity_count": len(candidates),
        "alias_count": len(alias_map),
        "alias_map": dict(sorted(alias_map.items())),
        "entities": {candidate["canonical_name"]: candidate["entity_id"] for candidate in candidates},
        "fanout": {
            "threshold": fanout.threshold,
            "generic_names": sorted(fanout.generic_names),
            "dropped_entities": list(fanout.dropped_entities),
            "dropped_aliases": {
                name: list(aliases) for name, aliases in sorted(fanout.dropped_aliases.items())
            },
            "findings": [
                {
                    "surface": finding.surface,
                    "host_count": finding.host_count,
                    "hosts": list(finding.hosts),
                    "reason": finding.reason,
                }
                for finding in fanout.findings
            ],
        },
    }

    alias_path = _write_json_atomic(alias_map_path(root), alias_payload)
    candidates_path = _write_jsonl_atomic(entity_candidates_path(root), candidates)
    merge_path = _write_jsonl_atomic(merges_path(root), merge_rows)

    if skipped_no_unit:
        notes.append(
            f"resolve: {skipped_no_unit} entity surface(s) skipped (no unit in the unit list)"
        )
    if skipped_after_fanout:
        notes.append(
            f"resolve: {skipped_after_fanout} fanout-cleaned entity record(s) dropped "
            "(no valid unit)"
        )
    if fanout.dropped_entities:
        notes.append(
            f"resolve: fanout dropped {len(fanout.dropped_entities)} generic-named entities"
        )
    if mention_candidates:
        notes.append(
            f"resolve: {len(mention_candidates)} low-confidence entity candidate(s) derived "
            "from mentions in units without extracted entities "
            "(candidate_source=mention; safety table not applied)"
        )

    return {
        "stage": STAGE,
        "status": "ok",
        "book_id": book_id,
        "counts": {
            "outputs": len(ordered),
            "entity_surfaces": collected,
            "candidates": len(candidates),
            "mention_candidates": len(mention_candidates),
            "aliases": len(alias_map),
            "merges": len(merge_rows),
            "merged_entities": len(merged_into),
            "fanout_dropped_entities": len(fanout.dropped_entities),
            "fanout_dropped_aliases": fanout.dropped_alias_count,
            "skipped_no_unit": skipped_no_unit,
            "skipped_after_fanout": skipped_after_fanout,
        },
        "paths": {
            "alias_map": _rel(root, alias_path),
            "entity_candidates": _rel(root, candidates_path),
            "merges": _rel(root, merge_path),
        },
        "notes": notes,
    }
