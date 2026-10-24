"""Deterministic collectors for pipeline stages 4-7.

The B09 shell ended with alias resolution, LLM summarisation, and co-occurrence
clustering. In v6 those are separate deterministic/LLM stages owned by other
modules (``resolve/``, ``assemble/``, ``enrich-merge``). This orchestrator must
still run the **9-stage shape** end to end and leave auditable work artifacts,
so these collectors:

* flatten engine candidates into work JSONL (mentions / entity candidates /
  assertions / events / cues) with exact, deterministic ordering;
* build entity-card and co-occurrence shells **without calling an LLM**;
* never write the box.

They are explicitly labelled ``shell`` in the trace decisions: counts are real,
semantic merging/description synthesis is not claimed here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from cbb_stage1.models import EngineOutput

from .files import unit_file_stem, write_jsonl

__all__ = [
    "collect_candidates",
    "build_entity_cards",
    "build_connections",
    "build_engine_index",
]


def _outputs_for(units, outputs: dict[str, EngineOutput]):
    for unit in units:
        output = outputs.get(unit.unit_id)
        if output is not None:
            yield unit, output


def collect_candidates(units: list, outputs: dict[str, EngineOutput]) -> dict[str, list]:
    """Flatten ``EngineOutput`` candidates into deterministic work records.

    Returns a dict with ``mentions``, ``entities``, ``assertions``, ``events``,
    ``cues`` lists. ``entities`` is an exact-name candidate table (no fuzzy
    merge, no union-find): identity resolution belongs to ``resolve/``.
    """
    mentions: list[dict] = []
    seen_mentions: set[tuple] = set()
    entity_table: dict[str, dict] = {}
    assertions: list[dict] = []
    events: list[dict] = []
    cues: list[dict] = []

    for unit, output in _outputs_for(units, outputs):
        for mention in output.mentions:
            key = (mention.unit_id, mention.surface, mention.start, mention.end)
            if key in seen_mentions:
                continue
            seen_mentions.add(key)
            mentions.append(
                {
                    "unit_id": unit.unit_id,
                    "surface": mention.surface,
                    "start": mention.start,
                    "end": mention.end,
                    "occurrence": mention.occurrence,
                    "mention_type": mention.mention_type,
                    "entity_id": mention.entity_id,
                    "confidence": mention.confidence,
                }
            )
            entry = entity_table.setdefault(
                mention.surface,
                {
                    "canonical_name": mention.surface,
                    "type": "unknown",
                    "aliases": set(),
                    "types": Counter(),
                    "mention_count": 0,
                    "first_unit": unit.unit_id,
                    "units": [],
                },
            )
            entry["mention_count"] += 1
            entry["units"].append(unit.unit_id)

        for entity in output.entities:
            name = entity.canonical_name
            entry = entity_table.setdefault(
                name,
                {
                    "canonical_name": name,
                    "type": entity.type,
                    "aliases": set(),
                    "types": Counter(),
                    "mention_count": 0,
                    "first_unit": unit.unit_id,
                    "units": [],
                },
            )
            if entry["type"] == "unknown":
                entry["type"] = entity.type
            entry["types"][entity.type] += 1
            entry["aliases"].update(entity.aliases)
            if unit.unit_id not in entry["units"]:
                entry["units"].append(unit.unit_id)

        for assertion in output.assertions:
            record = assertion.model_dump()
            record["unit_id"] = unit.unit_id
            assertions.append(record)
        for event in output.events:
            record = event.model_dump()
            record["unit_id"] = unit.unit_id
            events.append(record)
        for cue in output.cues:
            record = cue.model_dump()
            record["unit_id"] = unit.unit_id
            cues.append(record)

    entities: list[dict] = []
    for name in sorted(entity_table):
        entry = entity_table[name]
        entities.append(
            {
                "canonical_name": name,
                "type": entry["type"],
                "aliases": sorted(entry["aliases"]),
                "types": sorted(entry["types"].items()),
                "mention_count": entry["mention_count"],
                "first_unit": entry["first_unit"],
                "units": sorted(set(entry["units"])),
            }
        )

    return {
        "mentions": mentions,
        "entities": entities,
        "assertions": assertions,
        "events": events,
        "cues": cues,
    }


def build_entity_cards(
    entity_candidates: list[dict],
    outputs: dict[str, EngineOutput],
) -> list[dict]:
    """Build deterministic entity-card shell records (no LLM synthesis).

    ``description`` stays ``None``: merging descriptions across units is the
    ``enrich-merge`` task. The card carries the observed unit set so downstream
    work can synthesise with evidence.
    """
    descriptions: dict[str, list[str]] = {}
    for output in outputs.values():
        for entity in output.entities:
            if entity.description:
                descriptions.setdefault(entity.canonical_name, []).append(entity.description)

    cards: list[dict] = []
    for candidate in sorted(entity_candidates, key=lambda item: item["canonical_name"]):
        name = candidate["canonical_name"]
        observed = sorted(set(descriptions.get(name, [])))
        cards.append(
            {
                "canonical_name": name,
                "type": candidate["type"],
                "aliases": candidate["aliases"],
                "observed_descriptions": observed,
                "description": None,
                "units": candidate["units"],
                "mention_count": candidate["mention_count"],
                "produced_by": "orchestrator:descriptions-shell",
            }
        )
    return cards


def build_connections(
    units: list,
    outputs: dict[str, EngineOutput],
    *,
    min_weight: int = 2,
) -> list[dict]:
    """Build co-occurrence connections between entity names within a unit.

    Deterministic shell of the B09 ``connections`` stage: pair weight is the
    number of units in which two canonical names both appear. Pairs below
    ``min_weight`` are dropped.

    **source default: 1; this stage uses 2.** The deviation is deliberate — a pair that co-occurs
    in a single unit is usually one scene, and an edge for every one of those makes the connection
    graph noise. It is written down here rather than left implicit because this comment used to
    claim "source default: 2", which was simply wrong: it made a deviation read as fidelity.
    ``test_documented_defaults.py`` binds this sentence to the parameter so the two cannot diverge
    again without a test going red.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_units: dict[tuple[str, str], list[str]] = {}

    for unit, output in _outputs_for(units, outputs):
        names = {entity.canonical_name for entity in output.entities}
        names.update(mention.surface for mention in output.mentions)
        ordered = sorted(names)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                pair = (left, right)
                pair_counts[pair] += 1
                pair_units.setdefault(pair, []).append(unit.unit_id)

    connections: list[dict] = []
    for (left, right), weight in sorted(
        pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    ):
        if weight < min_weight:
            continue
        connections.append(
            {
                "a": left,
                "b": right,
                "weight": weight,
                "units": pair_units.get((left, right), []),
            }
        )
    return connections


def build_engine_index(
    units: list,
    outputs: dict[str, EngineOutput],
    engine_dir: str | Path,
    work_root: str | Path,
) -> list[dict]:
    """Index engine outputs by unit (path + sha of the artifact on disk)."""
    from cbb_stage1.ids import sha256_hex_bytes

    engine_dir = Path(engine_dir)
    work_root = Path(work_root)
    records: list[dict] = []
    for unit in units:
        path = engine_dir / f"{unit_file_stem(unit.unit_id)}.json"
        sha = None
        if path.exists():
            sha = sha256_hex_bytes(path.read_bytes())
        output = outputs.get(unit.unit_id)
        records.append(
            {
                "unit_id": unit.unit_id,
                "order": unit.order,
                "engine_output_path": str(path.relative_to(work_root)).replace("\\", "/"),
                "engine_output_sha256": sha,
                "mentions": len(output.mentions) if output else 0,
                "entities": len(output.entities) if output else 0,
                "assertions": len(output.assertions) if output else 0,
                "events": len(output.events) if output else 0,
                "cues": len(output.cues) if output else 0,
                "abstain": bool(output.abstain) if output else False,
            }
        )
    return records


def write_candidate_files(
    candidates: dict[str, list],
    resolve_dir: str | Path,
) -> dict[str, str]:
    """Write candidate JSONL files under ``resolve/`` and return their paths."""
    resolve_dir = Path(resolve_dir)
    paths = {
        "mentions": write_jsonl(resolve_dir / "mentions.jsonl", candidates["mentions"]),
        "entities": write_jsonl(
            resolve_dir / "entity_candidates.jsonl", candidates["entities"]
        ),
        "assertions": write_jsonl(
            resolve_dir / "assertions.jsonl", candidates["assertions"]
        ),
        "events": write_jsonl(resolve_dir / "events.jsonl", candidates["events"]),
        "cues": write_jsonl(resolve_dir / "cues.jsonl", candidates["cues"]),
    }
    return {name: str(path) for name, path in paths.items()}
