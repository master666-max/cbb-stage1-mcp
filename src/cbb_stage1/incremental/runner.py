"""Incremental runner (design ``00`` §4.16 / box spec §5.6, task A20/D6).

The pipeline is still a *full* runner: every invocation extracts every unit.
This module is the **planning half** of the incremental path and deliberately
does not touch ``orchestrator/pipeline.py`` (the shell is wired by W1 later).
Given the previous state (a box or the previous work layer) and the new unit
plan, it answers, deterministically and without side effects:

* which units must be extracted again (new / changed / renamed),
* which unchanged units can reuse their old ``work/engine/*.json`` artifact,
* which downstream records the change dirties (``incremental.deps`` closure),
* which work-table rows have to be replaced and which files cleaned up,
* whether incremental is even *provable* (``deps`` must exist — otherwise the
  plan fails loudly instead of silently falling back to a full run).

Design invariants:

* **Pure planning.** No writes to the work root, no engine calls, no clock in the
  plan payload: two calls with the same inputs produce the same ``to_dict()``.
* **Deterministic ordering.** Every tuple is sorted; JSON keys are sorted by
  ``json.dumps(sort_keys=True)`` in hashes, so plan equality is byte-stable.
* **Fail loud, not fallback.** A previous state that cannot be closed over (no
  dependency records at all) raises :class:`MissingDependenciesError`; it never
  degrades to "re-extract everything and call it incremental".
* **No new dependency layer.** Work-table paths and the engine-file stem are
  duplicated here on purpose: ``incremental`` sits below the orchestrator and
  must stay importable without it (and without the pipeline imports that pull
  every integration module).

Input shapes accepted for ``units``:

* ``list[dict]`` — ``Unit.to_record()`` records exactly as written to
  ``work/plan/units.jsonl`` (the normal case), or
* ``Mapping[str, Mapping]`` — ``unit_id -> record``.

``previous_state`` accepts ``None`` (first run → full), a :class:`PreviousState`,
a box directory (``manifest.json`` + ``sources/units.jsonl``), a work root
(``plan/units.jsonl``), a ``units.jsonl`` / JSON file path, a previous plan's
``to_dict()`` payload, or a mapping / sequence of unit records.

Usage::

    from cbb_stage1.incremental import plan_incremental
    plan = plan_incremental(work_root, units_new, previous_state=previous_box)
    if not plan.full_run:
        for unit_id in plan.extract_units: ...
    plan.write(work_root / "plan" / "incremental-plan.json")

``plan.manifest_fields()`` returns the ``manifest.incremental`` object expected
by ``box.writer.build_box``. The pipeline shell is expected to call this only
when ``Config.incremental`` is true (the flag is already plumbed); this module
itself never reads the config, so it stays testable without a ``Config``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1.ids import sha256_hex_text

from .deps import (
    DependencyGraph,
    DirtyClosure,
    dirty_closure,
    build_dependency_graph,
)

__all__ = [
    "IncrementalPlan",
    "MissingDependenciesError",
    "PreviousState",
    "load_previous_state",
    "plan_incremental",
    "unit_fingerprint",
    "work_engine_path",
]

#: Prefixes understood by :mod:`cbb_stage1.incremental.deps`.
UNIT_PREFIX = "u:"

#: Work tables that carry dependency-relevant records (relative to the work
#: root, POSIX separators). Mirrors the paths the eight integrations write.
_GRAPH_TABLES: tuple[tuple[str, str], ...] = (
    ("units", "plan/units.jsonl"),
    ("mentions", "resolve/mentions.jsonl"),
    ("entities", "resolve/entity_candidates.jsonl"),
    ("assertions", "assemble/assertions.jsonl"),
    ("events", "assemble/events.jsonl"),
    ("cues", "assemble/cues.jsonl"),
    ("hypotheses", "derive/hypotheses.jsonl"),
    ("claims", "assemble/claims.jsonl"),
)

#: Dirty-closure group -> ``(work table, primary key)`` used for the replace plan.
_REPLACE_TABLES: tuple[tuple[str, str, str], ...] = (
    ("units", "plan/units.jsonl", "unit_id"),
    ("mentions", "resolve/mentions.jsonl", "mention_id"),
    ("entities", "resolve/entity_candidates.jsonl", "entity_id"),
    ("assertions", "assemble/assertions.jsonl", "assertion_id"),
    ("events", "assemble/events.jsonl", "event_id"),
    ("cues", "assemble/cues.jsonl", "cue_id"),
    ("hypotheses", "derive/hypotheses.jsonl", "hypothesis_id"),
    ("claims", "assemble/claims.jsonl", "claim_id"),
)

#: Unit-record fields that participate in the unit identity fingerprint.
#: ``text_path`` is excluded because the box layer names texts differently from
#: the work layer; content identity is carried by ``sha256``.
#: ``prev``/``next`` are refreshed chain metadata and ``source_locator`` carries
#: the source file name — neither changes what the engine reads, so both are
#: excluded (renaming the book file must not dirty every unit). Changing the
#: ``prev``/``next`` chain does not require re-extraction; the units table itself
#: is rebuilt from the new plan on every run.
_FP_FIELDS: tuple[str, ...] = (
    "unit_id",
    "book_id",
    "volume",
    "chapter",
    "title",
    "order",
    "chars",
    "sha256",
    "narrative_mode",
    "narrator",
    "narrator_reliability",
    "scenes",
)


class MissingDependenciesError(RuntimeError):
    """Raised when an incremental plan cannot be proven from the work layer.

    The usual causes are a work root with no records, or records that carry no
    ``deps``/unit references at all. Incremental then has nothing to close over
    and must fail loudly instead of pretending that everything is clean.
    """


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return sha256_hex_text(text)


def unit_fingerprint(record: Mapping[str, Any]) -> str:
    """Deterministic identity hash of one unit record.

    Two units with the same fingerprint are interchangeable for incremental
    purposes. Content identity is carried by ``sha256``; the other fields cover
    the structural metadata that downstream stages consume (title, order,
    ``prev``/``next`` chain, narrator slots).
    """
    payload = {key: record.get(key) for key in _FP_FIELDS}
    return _sha256_text(_canonical(payload))


def work_engine_path(unit_id: str) -> str:
    """Relative path of the engine artifact for ``unit_id``.

    Mirrors ``orchestrator.files.unit_file_stem`` (``:`` is illegal on Windows);
    duplicated so this package does not import the orchestrator.
    """
    return "engine/%s.json" % unit_id.replace(":", "_")


def _normalize_sha(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text


def _as_unit_mapping(units: Any) -> list[Mapping[str, Any]]:
    """Normalize the ``units`` argument into a list of mapping records."""
    if units is None:
        return []
    if isinstance(units, Mapping):
        records: list[Mapping[str, Any]] = []
        for unit_id, value in units.items():
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("unit_id", unit_id)
            else:
                record = {"unit_id": str(unit_id), "sha256": value}
            records.append(record)
        return records
    if isinstance(units, (str, bytes)):
        raise TypeError("units must be a sequence of unit records, not a string")
    records = [record for record in units if isinstance(record, Mapping)]
    return records


def _current_units(units: Any) -> dict[str, dict[str, Any]]:
    """Return ``unit_id -> normalized`` for the new unit plan, order-checked."""
    current: dict[str, dict[str, Any]] = {}
    for record in _as_unit_mapping(units):
        unit_id = str(record.get("unit_id") or "").strip()
        if not unit_id:
            raise ValueError("unit record without 'unit_id' in units argument")
        if unit_id in current:
            raise ValueError(f"duplicate unit_id in units argument: {unit_id}")
        current[unit_id] = {
            "unit_id": unit_id,
            "sha256": _normalize_sha(record.get("sha256")),
            "fingerprint": unit_fingerprint(record),
            "order": record.get("order"),
            "title": record.get("title"),
            "text_path": record.get("text_path"),
        }
    return current


def _previous_from_records(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    previous: dict[str, dict[str, Any]] = {}
    for record in records:
        unit_id = str(record.get("unit_id") or record.get("id") or "").strip()
        if not unit_id:
            raise ValueError("previous unit record without 'unit_id'")
        # A bare ``{unit_id: sha256}`` state has no structural fields, so no
        # fingerprint can be derived; identity then falls back to sha256.
        structural = any(
            key not in ("unit_id", "id", "sha256", "hash", "fingerprint")
            for key in record
        )
        explicit_fp = record.get("fingerprint")
        previous[unit_id] = {
            "unit_id": unit_id,
            "sha256": _normalize_sha(record.get("sha256") or record.get("hash")),
            "fingerprint": explicit_fp if explicit_fp else (
                unit_fingerprint(record) if structural else None
            ),
            "order": record.get("order"),
            "title": record.get("title"),
            "text_path": record.get("text_path"),
        }
    return previous


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class PreviousState:
    """Normalized previous snapshot: unit identity hashes plus provenance."""

    units: dict[str, dict[str, Any]] = field(default_factory=dict)
    parent: str | None = None
    state_hash: str | None = None
    source: str = "explicit"
    changed_entities: tuple[str, ...] = ()

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.units))

    @property
    def empty(self) -> bool:
        return not self.units


def _from_units_payload(payload: Any) -> tuple[list[Mapping[str, Any]], str | None, str | None]:
    """Extract ``(unit records, parent, state_hash)`` from a loose payload."""
    parent = None
    state_hash = None
    if isinstance(payload, Mapping):
        parent = payload.get("parent") or payload.get("box_id")
        state_hash = (
            payload.get("state_hash")
            or payload.get("current_state_hash")
            or payload.get("merkle_root")
        )
        units = payload.get("units", payload.get("unit_hashes"))
        if units is None and any(str(key).startswith(UNIT_PREFIX) for key in payload):
            # A bare ``{unit_id: sha256}`` mapping (no envelope keys).
            units = payload
        if units is None and "unit_fingerprints" in payload:
            ids = payload.get("unit_sha256") or {}
            fingerprints = payload.get("unit_fingerprints") or {}
            records = []
            for unit_id in sorted(set(ids) | set(fingerprints)):
                records.append(
                    {
                        "unit_id": unit_id,
                        "sha256": ids.get(unit_id),
                        "fingerprint": fingerprints.get(unit_id),
                    }
                )
            return records, parent, state_hash
        return units, parent, state_hash
    return payload, parent, state_hash



def _payload_entities(payload: Any) -> tuple[str, ...]:
    """Read declared dirty entities from a manifest/plan payload."""
    if not isinstance(payload, Mapping):
        return ()
    raw = payload.get("dirty_entities") or payload.get("changed_entities") or ()
    if isinstance(raw, (str, bytes)):
        return (str(raw),)
    return tuple(sorted({str(item) for item in raw if item}))


def load_previous_state(source: Any) -> PreviousState | None:
    """Normalize every accepted ``previous_state`` shape into :class:`PreviousState`.

    ``None`` means "no previous state" (full run). A directory is probed as a
    box first (``manifest.json`` + ``sources/units.jsonl``) and then as a work
    root (``plan/units.jsonl``). A file may be ``units.jsonl`` or a previous
    plan's JSON payload. Mappings/sequences of unit records are accepted
    directly (tests and hand-written state).
    """
    if source is None:
        return None
    if isinstance(source, PreviousState):
        return source
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            manifest_path = path / "manifest.json"
            if manifest_path.is_file():
                manifest = _read_json(manifest_path) or {}
                parent = manifest.get("box_id") or (
                    (manifest.get("book") or {}).get("book_id")
                    if isinstance(manifest.get("book"), Mapping)
                    else None
                )
                hashes = manifest.get("hashes") if isinstance(manifest.get("hashes"), Mapping) else {}
                state_hash = hashes.get("merkle_root") if isinstance(hashes, Mapping) else None
                records = _read_jsonl(path / "sources" / "units.jsonl")
                incremental = manifest.get("incremental")
                dirty_entities = ()
                if isinstance(incremental, Mapping):
                    raw_entities = incremental.get("dirty_entities") or ()
                    dirty_entities = tuple(sorted({str(item) for item in raw_entities if item}))
                return PreviousState(
                    units=_previous_from_records(records),
                    parent=str(parent) if parent else None,
                    state_hash=str(state_hash) if state_hash else None,
                    source=str(path),
                    changed_entities=dirty_entities,
                )
            work_units = path / "plan" / "units.jsonl"
            if work_units.is_file():
                records = _read_jsonl(work_units)
                previous = _previous_from_records(records)
                return PreviousState(
                    units=previous,
                    parent=None,
                    state_hash=_state_hash(previous) if previous else None,
                    source=str(path),
                )
            raise ValueError(
                f"previous_state directory is neither a box nor a work root: {path} "
                f"(expected manifest.json or plan/units.jsonl)"
            )
        if path.is_file():
            if path.suffix.lower() == ".jsonl":
                previous = _previous_from_records(_read_jsonl(path))
                return PreviousState(
                    units=previous,
                    parent=None,
                    state_hash=_state_hash(previous) if previous else None,
                    source=str(path),
                )
            payload = _read_json(path)
            if payload is None:
                raise ValueError(f"previous_state file is not readable JSON: {path}")
            units, parent, state_hash = _from_units_payload(payload)
            records = list(_as_unit_mapping(units))
            previous = _previous_from_records(records)
            return PreviousState(
                units=previous,
                parent=str(parent) if parent else None,
                state_hash=str(state_hash) if state_hash else (_state_hash(previous) if previous else None),
                source=str(path),
                changed_entities=_payload_entities(payload),
            )
        raise ValueError(f"previous_state path does not exist: {path}")
    if isinstance(source, Mapping) or isinstance(source, Sequence):
        units, parent, state_hash = _from_units_payload(source)
        records = list(_as_unit_mapping(units))
        previous = _previous_from_records(records)
        return PreviousState(
            units=previous,
            parent=str(parent) if parent else None,
            state_hash=str(state_hash) if state_hash else (_state_hash(previous) if previous else None),
            source="mapping",
            changed_entities=_payload_entities(source),
        )
    raise TypeError(f"unsupported previous_state type: {type(source)!r}")


def _same_identity(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Compare a normalized previous unit with a current one.

    Full unit records carry a structural fingerprint. A bare ``{unit_id: sha256}``
    previous state cannot produce one, so identity falls back to the content hash
    (``sha256``) — the only comparable field both sides have.
    """
    previous_fp = previous.get("fingerprint")
    current_fp = current.get("fingerprint")
    if previous_fp and current_fp:
        return previous_fp == current_fp
    return (previous.get("sha256") or "") == (current.get("sha256") or "")


def _state_hash(units: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [
        [unit_id, units[unit_id].get("fingerprint") or units[unit_id].get("sha256") or ""]
        for unit_id in sorted(units)
    ]
    return _sha256_text(_canonical(rows))


def _load_graph_records(work_root: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for kind, relative in _GRAPH_TABLES:
        records[kind] = _read_jsonl(work_root / relative)
    return records


def _augment_unit_entity_edges(
    graph: DependencyGraph, entity_rows: Iterable[Mapping[str, Any]]
) -> tuple[DependencyGraph, int]:
    """Add ``unit -> entity-candidate`` edges from the ``units`` field.

    Returns the augmented graph and the number of added edges. Entity records
    keep their identity (``entity_id``) as the node; the dependent is the entity
    candidate row itself, whose replace key is ``entity_id``.
    """
    deps = {node: set(values) for node, values in graph.deps.items()}
    dependents = {node: set(values) for node, values in graph.dependents.items()}
    entity_refs = {node: set(values) for node, values in graph.entity_refs.items()}
    added = 0
    for row in entity_rows:
        entity_id = str(row.get("entity_id") or "")
        if not entity_id:
            continue
        deps.setdefault(entity_id, set())
        dependents.setdefault(entity_id, set())
        entity_refs.setdefault(entity_id, set())
        for unit_id in row.get("units") or ():
            unit = str(unit_id)
            if not unit:
                continue
            if unit not in deps.get(entity_id, set()):
                deps[entity_id].add(unit)
                dependents.setdefault(unit, set()).add(entity_id)
                added += 1
    return (
        DependencyGraph(
            deps={node: tuple(sorted(values)) for node, values in sorted(deps.items())},
            dependents={
                node: tuple(sorted(values)) for node, values in sorted(dependents.items())
            },
            entity_refs={
                node: tuple(sorted(values)) for node, values in sorted(entity_refs.items())
            },
        ),
        added,
    )


def _table_ids(work_root: Path, relative: str, key: str) -> set[str]:
    return {
        str(record.get(key))
        for record in _read_jsonl(work_root / relative)
        if record.get(key)
    }


@dataclass(frozen=True)
class IncrementalPlan:
    """Deterministic result of :func:`plan_incremental`.

    Attributes:
        work_root: The old work layer the plan was computed against.
        full_run: ``True`` when no previous state was given (nothing reusable).
        parent: Previous box id, when the previous state came from a box.
        previous_state_hash / current_state_hash: State hashes for audit.
        changed_units: New, content-changed, or renamed-current unit ids.
        unchanged_units: Unit ids whose content identity did not change (no dirty
            closure needed); ``extract_units`` may still include one of them when
            its old engine artifact is missing (see ``warnings``).
        removed_units: Unit ids that disappeared (cleanup + dependent records).
        renamed_units: ``(old_id, new_id)`` pairs matched by ``sha256``.
        extract_units: Units to send to the engine (changed + missing artifacts).
        reuse_engine_outputs: ``(unit_id, relative_engine_path)`` for reuse.
        cleanup_files: Relative files to delete (engine artifacts / old texts).
        dirty: Dirty-closure groups (``units``/``mentions``/... → ids).
        dirty_reasons: ``node -> upstream nodes`` explanation per dirty node.
        changed_entities: Entity ids to declare in ``manifest.incremental``.
        replace: Work-table path → record ids to drop before recomputing.
        warnings: Explicit notes (e.g. missing engine artifacts, no deps).
        counts: Flat integer counters for run reports.
        unit_sha256 / unit_fingerprints: Per-unit identity of the *current* plan,
            so the plan itself can serve as the next ``previous_state``.
        dependency_edges: Number of dependency edges seen in the work layer.
    """

    work_root: str
    full_run: bool
    parent: str | None
    previous_state_hash: str | None
    current_state_hash: str
    changed_units: tuple[str, ...] = ()
    unchanged_units: tuple[str, ...] = ()
    removed_units: tuple[str, ...] = ()
    renamed_units: tuple[tuple[str, str], ...] = ()
    extract_units: tuple[str, ...] = ()
    reuse_engine_outputs: tuple[tuple[str, str], ...] = ()
    cleanup_files: tuple[str, ...] = ()
    dirty: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dirty_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    changed_entities: tuple[str, ...] = ()
    replace: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)
    unit_sha256: dict[str, str] = field(default_factory=dict)
    unit_fingerprints: dict[str, str] = field(default_factory=dict)
    dependency_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready payload (deterministic key/order; usable as next state)."""
        return {
            "work_root": self.work_root,
            "full_run": self.full_run,
            "parent": self.parent,
            "previous_state_hash": self.previous_state_hash,
            "current_state_hash": self.current_state_hash,
            "changed_units": list(self.changed_units),
            "unchanged_units": list(self.unchanged_units),
            "removed_units": list(self.removed_units),
            "renamed_units": [list(pair) for pair in self.renamed_units],
            "extract_units": list(self.extract_units),
            "reuse_engine_outputs": [list(pair) for pair in self.reuse_engine_outputs],
            "cleanup_files": list(self.cleanup_files),
            "dirty": {kind: list(ids) for kind, ids in sorted(self.dirty.items())},
            "dirty_reasons": {
                node: list(reasons) for node, reasons in sorted(self.dirty_reasons.items())
            },
            "changed_entities": list(self.changed_entities),
            "replace": {path: list(ids) for path, ids in sorted(self.replace.items())},
            "warnings": list(self.warnings),
            "counts": dict(sorted(self.counts.items())),
            "unit_sha256": dict(sorted(self.unit_sha256.items())),
            "unit_fingerprints": dict(sorted(self.unit_fingerprints.items())),
            "dependency_edges": self.dependency_edges,
        }

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write the plan as pretty JSON (atomic replace, no timestamps)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
        return target

    def manifest_fields(self) -> dict[str, Any]:
        """``manifest.incremental`` object for ``box.writer.build_box``."""
        return {
            "parent": self.parent,
            "changed_units": list(self.changed_units),
            "dirty_entities": list(self.changed_entities),
            "replay_base": None,
        }


def _detect_renames(
    previous: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    removed: set[str],
    added: set[str],
) -> list[tuple[str, str]]:
    """Pair removed old ids with added new ids that carry the same ``sha256``.

    A chapter renumbering changes unit ids while the text stays byte-identical.
    Pairing by content hash keeps the old artifact reusable and turns the pair
    into an explicit ``renamed_units`` entry instead of a silent delete+add.
    """
    by_sha: dict[str, list[str]] = {}
    for unit_id in sorted(removed):
        sha = previous[unit_id].get("sha256") or ""
        if sha:
            by_sha.setdefault(sha, []).append(unit_id)
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    for unit_id in sorted(added):
        sha = current[unit_id].get("sha256") or ""
        if not sha:
            continue
        candidates = [old for old in by_sha.get(sha, []) if old not in used]
        if candidates:
            old = candidates[0]
            used.add(old)
            pairs.append((old, unit_id))
    return pairs


def plan_incremental(
    work_root: str | os.PathLike[str],
    units: Any,
    previous_state: Any = None,
) -> IncrementalPlan:
    """Plan the incremental delta for ``units`` against ``previous_state``.

    Args:
        work_root: Existing work layer (the previous run's ``work/``). Graph
            tables are read from here; nothing is written. May be missing only
            when ``previous_state`` is ``None`` (first/full run).
        units: New unit plan (``Unit.to_record()`` list, or id→record mapping).
        previous_state: Previous box / work root / units file / plan payload /
            mapping / sequence; ``None`` means full run.

    Returns:
        A deterministic :class:`IncrementalPlan`.

    Raises:
        MissingDependenciesError: The old work layer has no dependency records,
            so the downstream dirty closure cannot be proven. This is the
            explicit failure mode required by design ``00`` §4.16 — never a
            silent full-run fallback.
        ValueError: Malformed units/previous state or a missing work root while
            a previous state was supplied.
    """
    root = Path(work_root)
    current = _current_units(units)
    current_hash = _state_hash(current)

    previous = load_previous_state(previous_state)
    if previous_state is not None and not root.is_dir():
        raise MissingDependenciesError(
            f"previous_state was supplied but the work layer is missing: {root}; "
            f"cannot reuse artifacts or compute the dirty closure (run full first)"
        )

    warnings: list[str] = []
    if previous is None or previous.empty:
        full_run = True
        changed = tuple(sorted(current))
        unchanged: tuple[str, ...] = ()
        removed: tuple[str, ...] = ()
        renamed: tuple[tuple[str, str], ...] = ()
        closure = DirtyClosure()
        dependency_edges = 0
        records_loaded = 0
    else:
        full_run = False
        previous_units = previous.units
        added = set(current) - set(previous_units)
        gone = set(previous_units) - set(current)
        renamed_pairs = _detect_renames(previous_units, current, gone, added)
        renamed_old = {old for old, _new in renamed_pairs}
        removed_set = gone - renamed_old

        changed_ids = []
        for unit_id in sorted(current):
            if unit_id not in previous_units:
                changed_ids.append(unit_id)  # includes renamed_new
                continue
            if not _same_identity(previous_units[unit_id], current[unit_id]):
                changed_ids.append(unit_id)
        changed = tuple(changed_ids)
        unchanged = tuple(sorted(set(current) - set(changed)))
        removed = tuple(sorted(removed_set))
        renamed = tuple(renamed_pairs)

        records = _load_graph_records(root)
        records_loaded = sum(len(rows) for rows in records.values())
        graph = build_dependency_graph(
            units=records["units"],
            mentions=records["mentions"],
            entities=records["entities"],
            assertions=records["assertions"],
            events=records["events"],
            cues=records["cues"],
            hypotheses=records["hypotheses"],
            claims=records["claims"],
        )
        # ``deps.build_dependency_graph`` does not read the entity-candidate
        # ``units`` field (the ``entity`` kind has no deps). Entity cards are
        # still dirty when one of their units changes, so wire that edge in
        # here: unit --dep--> entity candidate. Without it, a chapter that only
        # produces entities would leave stale entity candidates unreplaced.
        graph, entity_edges = _augment_unit_entity_edges(graph, records["entities"])
        dependency_edges = sum(len(deps) for deps in graph.deps.values())

        needs_closure = bool(changed or removed or renamed)
        if needs_closure and records_loaded == 0:
            raise MissingDependenciesError(
                f"incremental plan for {root} cannot be proven: the work layer has no "
                f"records at all (0 rows across {len(_GRAPH_TABLES)} tables), so the "
                f"downstream dirty closure is unknown. Re-run full once (deps must be "
                f"generated at extraction time), then switch to incremental."
            )
        if needs_closure and dependency_edges == 0:
            raise MissingDependenciesError(
                f"incremental plan for {root} cannot be proven: {records_loaded} work "
                f"records exist but none carries a dependency edge (no deps/unit refs); "
                f"refusing to fall back to a silent full run. Deps must be produced by "
                f"extraction/resolve/assemble (design 00 §4.16, invariant I13)."
            )

        seeds = set(changed) | set(removed) | renamed_old
        closure = dirty_closure(
            changed_units=sorted(seeds),
            changed_entities=previous.changed_entities,
            graph=graph,
        )

    # ---- extract vs reuse ---------------------------------------------------
    extract: list[str] = []
    reuse: list[tuple[str, str]] = []
    planned_extract = set(changed)
    for unit_id in sorted(current):
        relative = work_engine_path(unit_id)
        exists = (root / relative).is_file()
        if unit_id in planned_extract:
            extract.append(unit_id)
        elif exists:
            reuse.append((unit_id, relative))
        else:
            # Unchanged unit whose artifact is missing: it is not dirty, but it
            # cannot be reused either. Extract it and say why.
            extract.append(unit_id)
            warnings.append(
                f"unchanged unit {unit_id} has no engine artifact at {relative}; "
                f"scheduled for extraction (artifact missing, not content change)"
            )
    warnings.sort()

    # ---- cleanup (removed + renamed-away ids) -------------------------------
    cleanup: set[str] = set()
    if previous is not None:
        for unit_id in removed:
            cleanup.add(work_engine_path(unit_id))
            text_path = previous.units.get(unit_id, {}).get("text_path")
            if text_path:
                cleanup.add(str(text_path))
        for old, _new in renamed:
            cleanup.add(work_engine_path(old))
            text_path = previous.units.get(old, {}).get("text_path")
            if text_path:
                cleanup.add(str(text_path))

    # ---- replace plan (closure groups -> work tables) -----------------------
    replace: dict[str, tuple[str, ...]] = {}
    groups = {
        "units": closure.units,
        "mentions": closure.mentions,
        "entities": closure.entities,
        "assertions": closure.assertions,
        "events": closure.events,
        "cues": closure.cues,
        "hypotheses": closure.hypotheses,
        "claims": closure.claims,
    }
    for kind, relative, key in _REPLACE_TABLES:
        ids = set(groups.get(kind, ()))
        if kind == "units":
            ids |= set(removed)
        if not ids:
            continue
        if not (root / relative).is_file():
            continue
        present = _table_ids(root, relative, key)
        overlap = tuple(sorted(ids & present))
        if overlap:
            replace[relative] = overlap

    current_sha = {unit_id: data["sha256"] for unit_id, data in sorted(current.items())}
    current_fp = {unit_id: data["fingerprint"] for unit_id, data in sorted(current.items())}
    counts = {
        "units_total": len(current),
        "changed": len(changed),
        "unchanged": len(unchanged),
        "removed": len(removed),
        "renamed": len(renamed),
        "extract": len(extract),
        "reuse": len(reuse),
        "cleanup_files": len(cleanup),
        "replace_records": sum(len(ids) for ids in replace.values()),
        "dirty_ids": len(closure.dirty_ids),
        "changed_entities": len(closure.entities),
        "records_loaded": records_loaded if not full_run else 0,
        "dependency_edges": dependency_edges,
    }

    return IncrementalPlan(
        work_root=str(root),
        full_run=full_run,
        parent=previous.parent if previous is not None else None,
        previous_state_hash=(
            previous.state_hash
            if previous is not None and previous.state_hash
            else (_state_hash(previous.units) if previous is not None and previous.units else None)
        ),
        current_state_hash=current_hash,
        changed_units=changed,
        unchanged_units=unchanged,
        removed_units=removed,
        renamed_units=renamed,
        extract_units=tuple(extract),
        reuse_engine_outputs=tuple(reuse),
        cleanup_files=tuple(sorted(cleanup)),
        dirty={
            "units": closure.units,
            "mentions": closure.mentions,
            "entities": closure.entities,
            "assertions": closure.assertions,
            "events": closure.events,
            "cues": closure.cues,
            "hypotheses": closure.hypotheses,
            "claims": closure.claims,
        },
        dirty_reasons=closure.reasons,
        changed_entities=closure.entities,
        replace=replace,
        warnings=tuple(warnings),
        counts=counts,
        unit_sha256=current_sha,
        unit_fingerprints=current_fp,
        dependency_edges=dependency_edges,
    )
