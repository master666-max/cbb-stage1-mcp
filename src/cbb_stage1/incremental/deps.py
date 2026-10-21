"""Entity-level dependency graph and dirty closure (incremental layer).

Implements design ``00`` §4.16 / box spec §5.6 ``deps`` + invariant I13:

* every record carries ``deps`` pointing at ``u:/e:/a:/m:`` ids (or at
  ``ev:/cue:`` for derived records);
* a changed unit (or entity) dirties exactly its transitive dependents —
  mentions → assertions/events/cues → hypotheses — via the *reverse* edges;
* changed units map to dirty **entities** so the next incremental box can
  declare ``incremental.dirty_entities``.

Everything is pure and deterministic: sorted inputs, sorted outputs, cycle-safe
BFS, no I/O. Unknown dependency ids are kept as graph nodes so an upstream id
that is not in this batch still propagates correctly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "DependencyGraph",
    "DirtyClosure",
    "build_dependency_graph",
    "dirty_closure",
    "entity_deps",
]

#: Node prefixes understood by the closure classifier.
UNIT_PREFIX = "u:"
MENTION_PREFIX = "m:"
ENTITY_PREFIX = "e:"
ASSERTION_PREFIX = "a:"
EVENT_PREFIX = "ev:"
CUE_PREFIX = "cue:"
HYPOTHESIS_PREFIX = "hy:"
CLAIM_PREFIX = "cl:"


def _as_records(records: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    return [record for record in (records or ()) if isinstance(record, Mapping)]


def _string_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, (str, bytes)):
        return [str(values)]
    return [str(value) for value in values if value]


def _evidence_units(record: Mapping[str, Any]) -> list[str]:
    units: list[str] = []
    for evidence in _as_records(record.get("evidence")):
        unit_id = evidence.get("unit_id")
        if unit_id:
            units.append(str(unit_id))
    return units


def _evidence_blocks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    blocks: list[Mapping[str, Any]] = []
    for key in ("setup", "payoff", "trigger"):
        block = record.get(key)
        if isinstance(block, Mapping):
            blocks.append(block)
            for evidence in _as_records(block.get("evidence")):
                blocks.append(evidence)
    return blocks


def _entity_ids(value: Any) -> list[str]:
    """Collect ``e:`` ids from strings / role dicts / lists."""
    found: list[str] = []
    if isinstance(value, str):
        if value.startswith(ENTITY_PREFIX):
            found.append(value)
    elif isinstance(value, Mapping):
        for key in ("entity", "entity_id"):
            item = value.get(key)
            if isinstance(item, str) and item.startswith(ENTITY_PREFIX):
                found.append(item)
        for item in value.values():
            found.extend(_entity_ids(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found.extend(_entity_ids(item))
    return found


def _record_node(record: Mapping[str, Any], kind: str) -> str | None:
    key = {
        "unit": "unit_id",
        "mention": "mention_id",
        "entity": "entity_id",
        "assertion": "assertion_id",
        "event": "event_id",
        "cue": "cue_id",
        "hypothesis": "hypothesis_id",
        "claim": "claim_id",
    }[kind]
    value = record.get(key)
    return str(value) if value else None


def _record_deps(record: Mapping[str, Any], kind: str) -> tuple[list[str], list[str]]:
    """Return ``(deps, entity_refs)`` for one record."""
    deps: list[str] = _string_list(record.get("deps"))
    entities: list[str] = []

    if kind == "unit":
        pass
    elif kind == "mention":
        deps.extend(_string_list(record.get("unit_id")))
        entities.extend(_entity_ids(record.get("entity_id")))
    elif kind == "entity":
        pass
    elif kind == "assertion":
        deps.extend(_evidence_units(record))
        entities.extend(_entity_ids(record.get("roles")))
        # speaker.entity_id / qualifiers.location are also entity references.
        speaker = record.get("speaker")
        if isinstance(speaker, Mapping):
            entities.extend(_entity_ids(speaker.get("entity_id")))
        qualifiers = record.get("qualifiers")
        if isinstance(qualifiers, Mapping):
            entities.extend(_entity_ids(qualifiers.get("location")))
    elif kind == "event":
        deps.extend(_string_list(record.get("unit_id")))
        deps.extend(_evidence_units(record))
        entities.extend(_entity_ids(record.get("participants")))
        entities.extend(_entity_ids(record.get("location")))
        time = record.get("time")
        if isinstance(time, Mapping):
            deps.extend(_string_list(time.get("anchor_unit")))
            deps.extend(_string_list(time.get("constraints")))
    elif kind == "cue":
        deps.extend(_evidence_units(record))
        entities.extend(_entity_ids(record.get("subject")))
        for block in _evidence_blocks(record):
            deps.extend(_evidence_units(block))
            deps.extend(_string_list(block.get("unit_id")))
            deps.extend(_string_list(block.get("event_id")))
    elif kind == "hypothesis":
        # hypothesis.schema.json has no ``deps`` field: closure uses
        # derived_from + support/against evidence.
        deps.extend(_string_list(record.get("derived_from")))
        deps.extend(_evidence_units(record))
        for block in _as_records(record.get("support")) + _as_records(
            record.get("against")
        ):
            deps.extend(_string_list(block.get("unit_id")))
            deps.extend(_string_list(block.get("assertion_id")))
    elif kind == "claim":
        deps.extend(_string_list(record.get("statement_assertion_id")))
        for question in _as_records(record.get("questions")):
            for answer in _as_records(question.get("answers")):
                deps.extend(_string_list(answer.get("unit_id")))
        for atomic in _as_records(record.get("atomic_facts")):
            deps.extend(_evidence_units(atomic))

    seen: set[str] = set()
    unique_deps = []
    for dep in deps:
        if dep and dep not in seen:
            seen.add(dep)
            unique_deps.append(dep)
    return unique_deps, sorted(set(entities))


@dataclass(frozen=True)
class DependencyGraph:
    """Forward deps, reverse dependents, and record→entity references."""

    deps: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dependents: dict[str, tuple[str, ...]] = field(default_factory=dict)
    entity_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self.deps))


_KIND_FIELDS = {
    "unit": "units",
    "mention": "mentions",
    "entity": "entities",
    "assertion": "assertions",
    "event": "events",
    "cue": "cues",
    "hypothesis": "hypotheses",
    "claim": "claims",
}


def build_dependency_graph(
    *,
    units: Iterable[Mapping[str, Any]] | None = None,
    mentions: Iterable[Mapping[str, Any]] | None = None,
    entities: Iterable[Mapping[str, Any] | str] | None = None,
    assertions: Iterable[Mapping[str, Any]] | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    cues: Iterable[Mapping[str, Any]] | None = None,
    hypotheses: Iterable[Mapping[str, Any]] | None = None,
    claims: Iterable[Mapping[str, Any]] | None = None,
) -> DependencyGraph:
    """Build the dependency graph from any subset of box/work tables."""
    records: dict[str, list[Mapping[str, Any]]] = {
        kind: [] for kind in _KIND_FIELDS
    }
    for kind, value in (
        ("unit", units),
        ("mention", mentions),
        ("assertion", assertions),
        ("event", events),
        ("cue", cues),
        ("hypothesis", hypotheses),
        ("claim", claims),
    ):
        records[kind].extend(_as_records(value))

    for item in entities or ():
        if isinstance(item, str):
            records["entity"].append({"entity_id": item})
        elif isinstance(item, Mapping):
            records["entity"].append(item)

    deps_map: dict[str, set[str]] = {}
    entity_refs: dict[str, set[str]] = {}
    for kind in _KIND_FIELDS:
        for record in records[kind]:
            node = _record_node(record, kind)
            if not node:
                continue
            deps_map.setdefault(node, set())
            entity_refs.setdefault(node, set())
            deps, refs = _record_deps(record, kind)
            deps_map[node].update(dep for dep in deps if dep and dep != node)
            # A record depends on the entities it references: changing/merging an
            # entity therefore dirties every record that mentions it.
            deps_map[node].update(ref for ref in refs if ref != node)
            entity_refs[node].update(refs)
            # Entity nodes exist even when no entity table was supplied.
            for entity_id in refs:
                deps_map.setdefault(entity_id, set())
                entity_refs.setdefault(entity_id, set())

    dependents: dict[str, set[str]] = {node: set() for node in deps_map}
    for node, dependencies in deps_map.items():
        for dependency in dependencies:
            dependents.setdefault(dependency, set()).add(node)

    return DependencyGraph(
        deps={node: tuple(sorted(deps_map[node])) for node in sorted(deps_map)},
        dependents={
            node: tuple(sorted(dependents.get(node, set())))
            for node in sorted(set(deps_map) | set(dependents))
        },
        entity_refs={
            node: tuple(sorted(entity_refs.get(node, set())))
            for node in sorted(deps_map)
        },
    )


@dataclass(frozen=True)
class DirtyClosure:
    """Result of a unit/entity change: every node that must be recomputed.

    ``reasons`` maps each dirty node to the immediate upstream dirty nodes that
    pulled it in, so a run report can explain the closure without re-running.
    """

    changed_units: tuple[str, ...] = ()
    changed_entities: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    mentions: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    assertions: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def dirty_ids(self) -> tuple[str, ...]:
        """Every dirty node id, sorted (excluding the seed reason entries)."""
        groups = (
            self.units,
            self.mentions,
            self.entities,
            self.assertions,
            self.events,
            self.cues,
            self.hypotheses,
            self.claims,
        )
        return tuple(sorted({node for group in groups for node in group}))

    def to_manifest_fields(self) -> dict[str, list[str]]:
        """Fields for ``manifest.incremental`` (§5.1)."""
        return {
            "changed_units": list(self.changed_units),
            "dirty_entities": list(self.entities),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_units": list(self.changed_units),
            "changed_entities": list(self.changed_entities),
            "units": list(self.units),
            "mentions": list(self.mentions),
            "entities": list(self.entities),
            "assertions": list(self.assertions),
            "events": list(self.events),
            "cues": list(self.cues),
            "hypotheses": list(self.hypotheses),
            "claims": list(self.claims),
            "reasons": {key: list(value) for key, value in self.reasons.items()},
        }


def dirty_closure(
    *,
    changed_units: Iterable[str] = (),
    changed_entities: Iterable[str] = (),
    graph: DependencyGraph | None = None,
    units: Iterable[Mapping[str, Any]] | None = None,
    mentions: Iterable[Mapping[str, Any]] | None = None,
    entities: Iterable[Mapping[str, Any] | str] | None = None,
    assertions: Iterable[Mapping[str, Any]] | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    cues: Iterable[Mapping[str, Any]] | None = None,
    hypotheses: Iterable[Mapping[str, Any]] | None = None,
    claims: Iterable[Mapping[str, Any]] | None = None,
) -> DirtyClosure:
    """Compute the transitive dirty closure of changed units/entities.

    Args:
        changed_units: Unit ids whose text/records changed.
        changed_entities: Entity ids whose identity/merge changed.
        graph: Pre-built :class:`DependencyGraph`; built from the record kwargs
            when omitted.
        units/mentions/entities/...: Records used when ``graph`` is omitted.

    Returns:
        Sorted node groups plus a per-node ``reasons`` map. An unknown seed is
        still echoed in ``changed_units``/``changed_entities`` but dirties
        nothing (it is not a dependency of any record here).
    """
    seeds = sorted(
        {str(uid) for uid in changed_units if uid}
        | {str(eid) for eid in changed_entities if eid}
    )
    if graph is None:
        graph = build_dependency_graph(
            units=units,
            mentions=mentions,
            entities=entities,
            assertions=assertions,
            events=events,
            cues=cues,
            hypotheses=hypotheses,
            claims=claims,
        )

    reasons: dict[str, list[str]] = {seed: [] for seed in seeds}
    dirty: set[str] = set(seeds)
    queue: deque[str] = deque(seeds)
    while queue:
        node = queue.popleft()
        for dependent in graph.dependents.get(node, ()):
            if dependent not in dirty:
                dirty.add(dependent)
                reasons.setdefault(dependent, []).append(node)
                queue.append(dependent)
            elif node not in reasons.get(dependent, []):
                reasons[dependent].append(node)

    # Entities referenced by dirty records are dirty too (a changed assertion
    # means its entities' cards must be recomputed).
    dirty_entities: set[str] = set()
    for node in dirty:
        dirty_entities.update(graph.entity_refs.get(node, ()))
    for node in dirty:
        if node.startswith(ENTITY_PREFIX):
            dirty_entities.add(node)
    dirty |= dirty_entities

    def _pick(prefix: str) -> tuple[str, ...]:
        return tuple(sorted(node for node in dirty if node.startswith(prefix)))

    changed = sorted({str(uid) for uid in changed_units if uid})
    changed_ent = sorted({str(eid) for eid in changed_entities if eid})
    return DirtyClosure(
        changed_units=tuple(changed),
        changed_entities=tuple(changed_ent),
        units=_pick(UNIT_PREFIX),
        mentions=_pick(MENTION_PREFIX),
        entities=tuple(sorted(dirty_entities)),
        assertions=_pick(ASSERTION_PREFIX),
        events=_pick(EVENT_PREFIX),
        cues=_pick(CUE_PREFIX),
        hypotheses=_pick(HYPOTHESIS_PREFIX),
        claims=_pick(CLAIM_PREFIX),
        reasons={
            node: tuple(sorted(values)) for node, values in sorted(reasons.items())
        },
    )


def entity_deps(
    *,
    mentions: Iterable[Mapping[str, Any]] | None = None,
    assertions: Iterable[Mapping[str, Any]] | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    cues: Iterable[Mapping[str, Any]] | None = None,
    hypotheses: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Entity id → record ids that reference it (entity-level deps, I13).

    Useful for the incremental writer: a changed entity's dependent records are
    exactly ``entity_deps(...)[entity_id]``. Output is sorted for determinism.
    """
    graph = build_dependency_graph(
        mentions=mentions,
        assertions=assertions,
        events=events,
        cues=cues,
        hypotheses=hypotheses,
    )
    reverse: dict[str, set[str]] = {}
    for node, entity_refs in graph.entity_refs.items():
        for entity_id in entity_refs:
            reverse.setdefault(entity_id, set()).add(node)
    return {entity: tuple(sorted(nodes)) for entity, nodes in sorted(reverse.items())}
