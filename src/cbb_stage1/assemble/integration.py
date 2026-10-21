"""P0-B integration for assertion assembly (``run`` contract).

Signature::

    run(work_root, *, book_id, outputs, units, created_at, upstream) -> dict

Reads engine output dicts (``unit_id`` / ``entities`` / ``assertions`` / ...),
plus the work-side identity tables written by :mod:`cbb_stage1.resolve.integration`
when present, and writes ``assemble/assertions.jsonl``.

The integration layer only does *plumbing*:

* role surfaces (``"贝塔心叶"``) are mapped to entity ids through the resolve
  candidate/alias tables, and merged ids are redirected to their surviving id --
  identity mapping, not validation;
* every rule (registered predicate, library, role names/types, claim layer,
  speaker, confidence method) is delegated to
  :class:`cbb_stage1.assemble.assertions.Assembler` and
  :class:`cbb_stage1.assemble.registry.PredicateRegistry` -- nothing is
  re-implemented here.

The predicate registry is read from ``work/lexicon/predicates.json`` when the
pipeline provides one; otherwise the design's minimal :data:`SEED_LEXICON` is
used and the source is reported in the result (never silently pretending it is
the human-owned profile).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import ValidationError

from cbb_stage1 import ids
from cbb_stage1.evidence.anchor import sha256_text as quote_sha_hex
from cbb_stage1.models import Assertion

from .assertions import Assembler
from .registry import SEED_LEXICON, PredicateRegistry

__all__ = ["STAGE", "assertions_path", "cues_path", "events_path", "run"]

STAGE = "assemble"
_RELIABLE = "reliable"

#: ``ev:`` id pattern from box spec (``schemas/event.schema.json``).
_EVENT_ID_RE = re.compile(r"^ev:[0-9a-f]{16,64}$")
_EVENT_IMPORTANCE = ("high", "medium", "low")


# ── shape/path helpers ─────────────────────────────────────────


def _as_dict(value: Any) -> dict[str, Any]:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, Mapping) else None


def _write_text_atomic(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    data = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    return _write_text_atomic(path, data)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover
        return path.as_posix()


def assertions_path(work_root: str | Path) -> Path:
    return Path(work_root) / "assemble" / "assertions.jsonl"


def events_path(work_root: str | Path) -> Path:
    return Path(work_root) / "assemble" / "events.jsonl"


def cues_path(work_root: str | Path) -> Path:
    return Path(work_root) / "resolve" / "cues.jsonl"


def proposed_predicates_path(work_root: str | Path) -> Path:
    """Where I4 keeps out-of-table predicates (never auto-registered)."""
    return Path(work_root) / "assemble" / "proposed_predicates.jsonl"


def assemble_quarantine_path(work_root: str | Path) -> Path:
    """Rejections recorded at assemble time (not deferred to build_box)."""
    return Path(work_root) / "assemble" / "quarantine.jsonl"


# ── identity index from the resolve stage ──────────────────────


@dataclass
class _Resolution:
    """Surface -> entity id index + merge redirects, loaded from ``work/resolve``."""

    surface_to_id: dict[str, str]
    redirect: dict[str, str]
    candidates: int
    source: str

    @property
    def has_tables(self) -> bool:
        return self.candidates > 0

    def resolve_id(self, entity_id: str) -> str:
        seen: set[str] = set()
        current = entity_id
        while current in self.redirect:
            if current in seen:
                raise RuntimeError(f"assemble: merge cycle at {current}")
            seen.add(current)
            current = self.redirect[current]
        return current

    def map_surface(self, value: str) -> str:
        if value.startswith("e:"):
            return self.resolve_id(value)
        mapped = self.surface_to_id.get(value)
        return self.resolve_id(mapped) if mapped else value

    def map_speaker(self, speaker: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(speaker)
        entity_id = result.get("entity_id")
        if isinstance(entity_id, str) and entity_id:
            result["entity_id"] = self.map_surface(entity_id)
        elif not entity_id and result.get("name"):
            mapped = self.surface_to_id.get(str(result["name"]))
            if mapped:
                result["entity_id"] = self.resolve_id(mapped)
        return result

    def map_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.map_surface(value)
        if isinstance(value, Mapping):
            if "literal" in value:
                return dict(value)
            return {key: self.map_value(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.map_value(inner) for inner in value]
        return value


def _load_resolution(root: Path) -> _Resolution:
    candidates_path = root / "resolve" / "entity_candidates.jsonl"
    alias_path = root / "resolve" / "alias_map.json"
    merge_path = root / "resolve" / "merges.jsonl"
    candidates = _read_jsonl(candidates_path)

    surface_to_id: dict[str, str] = {}
    ambiguous: set[str] = set()
    for candidate in candidates:
        entity_id = str(candidate.get("entity_id") or "")
        if not entity_id:
            continue
        surfaces = [candidate.get("canonical_name"), *(candidate.get("aliases") or [])]
        for surface in surfaces:
            surface = str(surface or "").strip()
            if not surface:
                continue
            if surface in surface_to_id and surface_to_id[surface] != entity_id:
                ambiguous.add(surface)
            else:
                surface_to_id[surface] = entity_id
    for surface in ambiguous:
        surface_to_id.pop(surface, None)

    alias_payload = _read_json(alias_path) or {}
    entities = alias_payload.get("entities")
    if isinstance(entities, Mapping):
        for name, entity_id in entities.items():
            if isinstance(entity_id, str) and entity_id and str(name) not in ambiguous:
                surface_to_id.setdefault(str(name), entity_id)
    alias_map = alias_payload.get("alias_map")
    if isinstance(alias_map, Mapping):
        for surface, canonical in alias_map.items():
            target = surface_to_id.get(str(canonical)) or (
                str(entities.get(canonical)) if isinstance(entities, Mapping) else None
            )
            if target:
                surface_to_id[str(surface)] = target

    redirect: dict[str, str] = {}
    for row in _read_jsonl(merge_path):
        target = str(row.get("to") or "")
        if not target:
            continue
        for source in row.get("from") or ():
            source = str(source or "")
            if source and source != target:
                redirect[source] = target

    source = "work/resolve" if candidates or alias_map else "none"
    return _Resolution(
        surface_to_id=surface_to_id,
        redirect=redirect,
        candidates=len(candidates),
        source=source,
    )


# ── registry / outputs ─────────────────────────────────────────


def _load_registry(root: Path) -> tuple[PredicateRegistry, str]:
    path = root / "lexicon" / "predicates.json"
    if path.exists():
        return PredicateRegistry.from_path(path), _rel(root, path)
    return PredicateRegistry.from_lexicon(SEED_LEXICON), "builtin:SEED_LEXICON"


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


def _unit_reliabilities(units: Sequence[Any]) -> dict[str, str]:
    rel: dict[str, str] = {}
    for unit in units or ():
        try:
            record = _as_dict(unit)
        except TypeError:
            continue
        unit_id = str(record.get("unit_id") or "")
        if unit_id:
            value = str(record.get("narrator_reliability") or _RELIABLE)
            rel[unit_id] = value if value in ("reliable", "unreliable") else "unknown"
    return rel


# ── row coercion ───────────────────────────────────────────────


def _normalize_evidence_rows(
    rows: Any,
    *,
    default_unit: str,
    resolution: _Resolution,
) -> list[Any]:
    """Default the unit, remap speakers and recompute box-style quote hashes."""
    evidence: list[Any] = []
    for raw in rows or ():
        item = dict(raw) if isinstance(raw, Mapping) else None
        if item is None:
            evidence.append(raw)  # let pydantic reject it
            continue
        if not item.get("unit_id"):
            item["unit_id"] = default_unit
        quote = item.get("quote")
        if isinstance(quote, str) and quote:
            # the box recomputes the quote hash from the located text; do the
            # same here so assertion/event ids match the box table ids
            item["quote_sha256"] = f"sha256:{quote_sha_hex(quote)}"
        speaker = item.get("speaker")
        if isinstance(speaker, Mapping):
            item["speaker"] = resolution.map_speaker(speaker)
        evidence.append(item)
    return evidence


def _map_roles(
    roles: Mapping[str, Any],
    *,
    resolution: _Resolution,
    registry: "PredicateRegistry | None" = None,
    predicate: str = "",
) -> dict[str, Any]:
    """Map role values according to what the REGISTRY says each role is.

    The old code ran every role value through `resolution.map_value`, which
    cannot know whether a value is an entity surface or a literal. So an entity
    name that resolve had never seen came back unchanged and was rejected as
    "entity role expects an 'e:' id" — a message that hides the real cause
    (unresolved name) — and a literal came back as a bare string that the
    registry's literal branch rejected.

    The registry is the single source of truth for role types, so the mapping
    is decided here:

    * `entity`  -> resolve the surface to an `e:` id (unresolved stays a
      bare name and is reported precisely by the assembler);
    * `literal` -> wrap into `{"literal": str}`;
    * role not declared by the registry (unknown predicate) -> leave untouched
      and let the proposal path handle it.

    Returns the mapped roles. `None` values are dropped: a null role is not a
    value and the box rejects it.
    """
    spec = registry.get(predicate) if (registry is not None and predicate) else None
    declared = {role.name: role for role in spec.roles} if spec is not None else {}

    mapped: dict[str, Any] = {}
    for name, value in roles.items():
        key = str(name)
        if value is None or value == [] or value == "":
            continue
        role = declared.get(key)
        if role is None:
            # Unknown predicate / undeclared role: keep the raw value; the
            # assembler reports it as role_mismatch and the proposal table
            # records the predicate.
            mapped[key] = resolution.map_value(value)
            continue
        if role.type == "literal":
            # Deliberately NOT re-wrapping here. The emitting end owns the value
            # shape (predicate_map.literal); if it hands us a bare string that is
            # a real contract break and the assembler must reject it. Wrapping it
            # here would hide the break behind a second safety net — measured:
            # with both layers tolerant, no single-point mutation could make the
            # full-chain gate fail, i.e. the gate was decorative.
            mapped[key] = value
            continue
        # entity role
        if isinstance(value, (list, tuple)):
            mapped[key] = [
                resolution.map_value(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            mapped[key] = resolution.map_value(value)
    return mapped


def _as_literal(value: Any) -> dict[str, Any]:
    """Box-shaped literal: `{"literal": str}` (matches box/records.py)."""
    if isinstance(value, Mapping):
        if "literal" in value:
            return dict(value)
        return {"literal": str(value)}
    if isinstance(value, (list, tuple)):
        return {"literal": "／".join(str(item) for item in value)}
    return {"literal": str(value)}


def _normalize_assertion_row(
    row: Mapping[str, Any],
    *,
    book_id: str,
    unit_id: str,
    resolution: _Resolution,
    registry: "PredicateRegistry | None" = None,
) -> Assertion:
    data = dict(row)
    data["book_id"] = str(data.get("book_id") or book_id)
    data["library"] = str(data.get("library") or "")
    layer = str(data.get("claim_layer") or "fact")
    data["claim_layer"] = layer if layer in ("fact", "statement", "inferred") else "fact"
    roles = data.get("roles") or {}
    data["roles"] = (
        _map_roles(
            roles,
            resolution=resolution,
            registry=registry,
            predicate=str(data.get("predicate") or ""),
        )
        if isinstance(roles, Mapping)
        else roles
    )

    evidence = _normalize_evidence_rows(
        data.get("evidence"), default_unit=unit_id, resolution=resolution
    )
    data["evidence"] = evidence

    speaker = data.get("speaker")
    if isinstance(speaker, Mapping):
        data["speaker"] = resolution.map_speaker(speaker)
    if isinstance(data.get("deps"), (list, tuple)):
        data["deps"] = [resolution.map_value(value) for value in data["deps"]]

    for key, default in (
        ("qualifiers", {}),
        ("derived_from", []),
        ("deps", []),
        ("produced_by", {}),
        ("verification", None),
    ):
        if data.get(key) is None:
            data[key] = default
    if data.get("confidence") is None:
        data["confidence"] = 0.0
    if data.get("status") is None:
        data["status"] = "asserted"
    if data.get("confidence_scope") is None:
        data["confidence_scope"] = "assertion"
    if data.get("superseded_by") is None:
        data["superseded_by"] = None
    return Assertion.model_validate(data)


# ── events producer (work/assemble/events.jsonl) ──────────────


def _box_event_id(
    book_id: str,
    event_type: str,
    unit_id: str,
    participants: Sequence[str],
    summary: str,
) -> str:
    """Same content key as ``box/records.py`` ``normalize_event`` (via ids)."""
    return ids.event_id(book_id, event_type, unit_id, participants, summary)


def _normalize_event_row(
    row: Mapping[str, Any],
    *,
    book_id: str,
    unit_id: str,
    resolution: _Resolution,
    run_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Coerce one engine event into the work event row (box-compatible id).

    Returns ``(record, problems)``.  Rules mirror ``box/records.py`` for the
    fields that shape the ``ev:`` content key: non-empty type/summary and
    entity-resolvable participants (names -> ids, merged ids redirected).
    """
    data = dict(row)
    event_type = str(data.get("type") or "").strip()
    summary = str(data.get("summary") or "").strip()
    if not event_type or not summary:
        return None, [
            {
                "item_id": ids.invalid_id(
                    "ev", {"unit_id": unit_id, "event_type": event_type, "summary": summary}
                ),
                "kind": "event",
                "reason": "schema",
                "detail": "event needs type and summary",
                "source_run": run_id,
            }
        ]

    event_unit = str(data.get("unit_id") or unit_id or "")
    participants: list[str] = []
    for raw in data.get("participants") or ():
        value: Any = raw
        if isinstance(value, Mapping):
            value = value.get("entity") or value.get("entity_id")
        if not isinstance(value, str) or not value.strip():
            continue
        mapped = resolution.map_value(value.strip())
        if isinstance(mapped, str) and mapped.startswith("e:") and mapped not in participants:
            participants.append(mapped)
    if not participants:
        return None, [
            {
                "item_id": ids.invalid_id("ev", {"unit_id": unit_id, "summary": summary}),
                "kind": "event",
                "reason": "ref_missing",
                "detail": "event has no resolvable participants",
                "source_run": run_id,
            }
        ]

    evidence = _normalize_evidence_rows(
        data.get("evidence"), default_unit=event_unit, resolution=resolution
    )

    location: str | None = None
    raw_location = data.get("location")
    if raw_location:
        mapped_location = resolution.map_value(raw_location)
        if isinstance(mapped_location, str) and mapped_location.startswith("e:"):
            location = mapped_location

    time_row = data.get("time")
    time_row = dict(time_row) if isinstance(time_row, Mapping) else {}
    importance = str(data.get("importance") or "medium")
    if importance not in _EVENT_IMPORTANCE:
        importance = "medium"
    status = str(data.get("status") or "asserted")

    event_id = str(data.get("event_id") or "")
    if not _EVENT_ID_RE.match(event_id):
        event_id = _box_event_id(book_id, event_type, event_unit, participants, summary)

    evidence_units = {
        str(item.get("unit_id")) for item in evidence if isinstance(item, Mapping)
    }
    deps = sorted({event_unit} | set(participants) | evidence_units)
    return (
        {
            "event_id": event_id,
            "book_id": book_id,
            "type": event_type,
            "summary": summary,
            "participants": participants,
            "location": location,
            "time": time_row,
            "importance": importance,
            "unit_id": event_unit,
            "evidence": evidence,
            "status": status,
            "superseded_by": None,
            "deps": deps,
            "produced_by": {"run_id": run_id, "engine": "integration:assemble"},
        },
        [],
    )


# ── the stage entry point ─────────────────────────────────────


def run(
    work_root: str | Path,
    *,
    book_id: str,
    outputs: list[dict],
    units: list[dict],
    created_at: str,
    upstream: dict,
) -> dict:
    """Assemble engine assertions into ``assemble/assertions.jsonl``."""
    root = Path(work_root)
    upstream = upstream or {}
    resolution = _load_resolution(root)
    registry, registry_source = _load_registry(root)
    unit_records = []
    for unit in units or ():
        try:
            unit_records.append(_as_dict(unit))
        except TypeError:
            continue
    ordered = _ordered_outputs(outputs or [], unit_records)
    reliabilities = _unit_reliabilities(units or [])
    run_id = str(upstream.get("run_id") or f"run:{book_id}")

    quarantined: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    inputs = 0

    # Group raw rows per unit so each unit gets the right narrator reliability.
    per_unit: dict[str, list[Assertion]] = {}
    for output in ordered:
        unit_id = str(output.get("unit_id") or "")
        for index, raw in enumerate(output.get("assertions") or ()):
            inputs += 1
            try:
                row = _as_dict(raw)
            except TypeError as exc:
                quarantined.append(
                    {
                        "item_id": ids.invalid_id(
                            "a", {"unit_id": unit_id, "index": index}
                        ),
                        "kind": "assertion",
                        "reason": "schema",
                        "detail": f"assertion row is not a mapping: {exc}",
                        "source_run": run_id,
                    }
                )
                continue
            try:
                assertion = _normalize_assertion_row(
                    row,
                    book_id=book_id,
                    unit_id=unit_id,
                    resolution=resolution,
                    registry=registry,
                )
            except (ValidationError, TypeError, ValueError) as exc:
                quarantined.append(
                    {
                        "item_id": ids.invalid_id(
                            "a", {"unit_id": unit_id, "index": index}
                        ),
                        "kind": "assertion",
                        "reason": "schema",
                        "detail": f"assertion shape rejected: {exc}",
                        "source_run": run_id,
                    }
                )
                continue
            per_unit.setdefault(unit_id, [])
            per_unit[unit_id].append(assertion)

    # Dedupe by FACT, not by assertion_id.
    #
    # `assertion_id` hashes the evidence quote, so two rows stating the same fact with
    # slightly different quotes get different ids and both survive. Measured on a real
    # run: "从我手腕伸出的丝线，变幻成了银色的西洋剑" and the same sentence with a
    # trailing "。" produced two assertions with identical predicate and roles. Each was
    # then scored separately by semantic entropy (8 samples each), doubling the cost of
    # one fact, and the box would have carried both.
    #
    # The quote is still part of the id — different evidence for the same fact is
    # meaningful — but the two rows must be MERGED rather than dropped: both quotes are
    # real, and discarding one loses evidence.
    accepted_by_fact: dict[tuple, list[dict]] = {}
    duplicates = 0
    merged_evidence = 0

    def _fact_key(row: dict) -> tuple[str, str, str, tuple]:
        """Identity of a FACT for dedupe purposes.

        Three parts, each earned by a measured failure:

        1. ``predicate`` + ``roles`` — the fact itself.
        2. ``claim_layer`` — a `fact` ("X and Y are enemies") and a `statement`
           ("someone says so") are different records under §5.16; merging them would
           silently promote a claim to a world fact.
        3. the evidence SPANS — two rows with identical roles can still be different
           events. Measured in the seam test: `参与` with the same mapped roles for
           "贝塔心叶穿着黑色夹克" (span 12-23) and "黑手党把贝塔心叶逐出" (span 41-52).
           Merging those destroyed one real event.

        What is deliberately NOT in the key is the quote TEXT: the same sentence with
        and without a trailing "。" is one fact with two spans that overlap, which is
        the case that motivated this dedupe in the first place.
        """
        spans = tuple(
            sorted(
                (ev.get("start"), ev.get("end"))
                for ev in (row.get("evidence") or [])
                if ev.get("start") is not None and ev.get("end") is not None
            )
        )
        return (
            str(row.get("predicate") or ""),
            json.dumps(row.get("roles") or {}, ensure_ascii=False, sort_keys=True),
            str(row.get("claim_layer") or ""),
            spans,
        )

    def _overlaps(existing: dict, incoming: dict, *, tolerance: int = 2) -> bool:
        """True when two rows point at the same piece of text.

        Exact span equality is too strict: the same sentence quoted once with and once
        without its trailing "。" gives spans 729-749 and 729-750 — the same fact, two
        quotes. Treat spans that overlap within ``tolerance`` characters as one.

        But NOT any overlap at all: the seam case has "贝塔心叶穿着黑色夹克" (12-23) and
        "黑手党把贝塔心叶逐出" (41-52) under identical mapped roles, which are two
        different events. Distance is what separates them.
        """
        left = [
            (ev.get("start"), ev.get("end"))
            for ev in (existing.get("evidence") or [])
            if ev.get("start") is not None and ev.get("end") is not None
        ]
        right = [
            (ev.get("start"), ev.get("end"))
            for ev in (incoming.get("evidence") or [])
            if ev.get("start") is not None and ev.get("end") is not None
        ]
        for start_a, end_a in left:
            for start_b, end_b in right:
                if abs(start_a - start_b) <= tolerance and abs(end_a - end_b) <= tolerance:
                    return True
        return False

    for unit_id in sorted(
        per_unit,
        key=lambda uid: next(
            (index for index, unit in enumerate(unit_records) if unit.get("unit_id") == uid),
            len(unit_records),
        ),
    ):
        reliability = reliabilities.get(unit_id, "unknown")
        if reliability == "unknown":
            reliability = _RELIABLE
        assembler = Assembler(
            registry,
            book_id=book_id,
            narrator_reliability=reliability,
            run_id=run_id,
        )
        for raw in per_unit[unit_id]:
            result = assembler.assemble_one(raw)
            for row in result.accepted:
                dumped = row.model_dump(mode="json")
                key = _fact_key(dumped)
                # Same fact AND same text location -> merge. Key alone is not enough:
                # spans are part of the key, so an overlapping-but-unequal span lands
                # under a different key and needs the tolerance check.
                bucket = accepted_by_fact.setdefault(key[:3], [])
                existing = next(
                    (candidate for candidate in bucket if _overlaps(candidate, dumped)),
                    None,
                )
                if existing is not None:
                    duplicates += 1
                    # Merge the evidence lists, keeping distinct (start, end, quote)
                    # triples. The fact is the same; the support is additive.
                    seen = {
                        (ev.get("start"), ev.get("end"), ev.get("quote"))
                        for ev in existing.get("evidence") or []
                    }
                    for ev in dumped.get("evidence") or []:
                        fingerprint = (ev.get("start"), ev.get("end"), ev.get("quote"))
                        if fingerprint in seen:
                            continue
                        seen.add(fingerprint)
                        existing.setdefault("evidence", []).append(ev)
                        merged_evidence += 1
                    continue
                bucket.append(dumped)
                accepted.append(dumped)
            quarantined.extend(item.model_dump() for item in result.quarantined)
            proposals.extend(proposal.model_dump() for proposal in result.proposals)

    # events producer: box-compatible ev: ids so temporal/validate/box can read
    # the same rows without re-deriving a parallel event table.
    events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    events_input = 0
    event_duplicates = 0
    for output in ordered:
        unit_id = str(output.get("unit_id") or "")
        for index, raw in enumerate(output.get("events") or ()):
            events_input += 1
            try:
                row = _as_dict(raw)
            except TypeError as exc:
                quarantined.append(
                    {
                        "item_id": ids.invalid_id(
                            "ev", {"unit_id": unit_id, "index": index}
                        ),
                        "kind": "event",
                        "reason": "schema",
                        "detail": f"event row is not a mapping: {exc}",
                        "source_run": run_id,
                    }
                )
                continue
            record, problems = _normalize_event_row(
                row, book_id=book_id, unit_id=unit_id, resolution=resolution, run_id=run_id
            )
            if problems:
                quarantined.extend(problems)
                continue
            if record is None:  # pragma: no cover - defensive
                continue
            if record["event_id"] in seen_event_ids:
                event_duplicates += 1
                continue
            seen_event_ids.add(record["event_id"])
            events.append(record)

    # Raw cue producer: the cue state machine reads resolve/cues.jsonl; without
    # this table a replay without engine outputs has no cue universe. Rows are
    # raw candidates (state/IDs are assigned by cues.integration).
    raw_cues: list[dict[str, Any]] = []
    cues_input = 0
    for output in ordered:
        unit_id = str(output.get("unit_id") or "")
        for raw in output.get("cues") or ():
            cues_input += 1
            try:
                row = _as_dict(raw)
            except TypeError as exc:
                quarantined.append(
                    {
                        "item_id": ids.invalid_id(
                            "cue", {"unit_id": unit_id, "index": cues_input}
                        ),
                        "kind": "cue",
                        "reason": "schema",
                        "detail": f"cue row is not a mapping: {exc}",
                        "source_run": run_id,
                    }
                )
                continue
            row.setdefault("unit_id", unit_id)
            raw_cues.append(row)

    path = _write_jsonl_atomic(assertions_path(root), accepted)
    event_path = _write_jsonl_atomic(events_path(root), events)
    cue_path = _write_jsonl_atomic(cues_path(root), raw_cues)
    # I4: unregistered predicates must land in proposed_predicates.jsonl. They
    # were only ever returned in-memory, so the box writer found an empty table
    # and the proposals silently vanished (measured: proposals=1 in the return
    # value, no file on disk). Persist them where the contract says they live.
    # Side effect only: the file is what matters, the returned path is not used.
    _write_jsonl_atomic(proposed_predicates_path(root), proposals)
    # Persist rejections here, not only inside build_box. The audit measured a
    # run that died before box construction: resolve/assertions.jsonl held 43KB of
    # raw assertions, assemble/assertions.jsonl was 0 bytes, and quarantine.jsonl
    # did not exist at all — so every rejection reason evaporated and the run was
    # un-diagnosable. Writing them at the end of assemble keeps the evidence even
    # when the box is never built.
    _write_jsonl_atomic(assemble_quarantine_path(root), quarantined)
    by_layer: dict[str, int] = {}
    by_library: dict[str, int] = {}
    for row in accepted:
        by_layer[str(row.get("claim_layer"))] = by_layer.get(str(row.get("claim_layer")), 0) + 1
        by_library[str(row.get("library"))] = by_library.get(str(row.get("library")), 0) + 1

    # Per-predicate reconciliation: emitted vs accepted vs quarantined.
    #
    # A table-level check ("events and assertions are not both empty") cannot see
    # a partial loss: a box keeping 1 of 9 assertions still has a non-empty table
    # and a low absent ratio, so it would ship green. Counting per predicate makes
    # the shape of the loss visible and gives the box gate something exact to
    # compare. Emitted is counted from the engine outputs, accepted from the box
    # table, quarantined from the rejections — three numbers that must add up.
    emitted_by_predicate: dict[str, int] = {}
    for output in ordered:
        for raw in output.get("assertions") or ():
            try:
                row = _as_dict(raw)
            except TypeError:
                continue
            predicate = str(row.get("predicate") or "")
            if predicate:
                emitted_by_predicate[predicate] = emitted_by_predicate.get(predicate, 0) + 1
    accepted_by_predicate: dict[str, int] = {}
    for row in accepted:
        predicate = str(row.get("predicate") or "")
        accepted_by_predicate[predicate] = accepted_by_predicate.get(predicate, 0) + 1
    quarantined_by_predicate: dict[str, int] = {}
    for item in quarantined:
        if item.get("kind") != "assertion":
            continue
        detail = str(item.get("detail") or "")
        # Rejections name the predicate for the two shapes the assembler emits.
        for predicate in emitted_by_predicate:
            if f"'{predicate}'" in detail or f'"{predicate}"' in detail:
                quarantined_by_predicate[predicate] = (
                    quarantined_by_predicate.get(predicate, 0) + 1
                )
                break
    predicate_reconciliation = {
        predicate: {
            "emitted": emitted_by_predicate.get(predicate, 0),
            "accepted": accepted_by_predicate.get(predicate, 0),
            "quarantined": quarantined_by_predicate.get(predicate, 0),
        }
        for predicate in sorted(emitted_by_predicate)
    }

    notes: list[str] = []
    if not resolution.has_tables:
        notes.append(
            "assemble: no resolve tables found; role surfaces were left as-is "
            "(name-valued roles will be role_mismatch)"
        )
    if registry_source == "builtin:SEED_LEXICON":
        notes.append(
            "assemble: work/lexicon/predicates.json missing; used builtin SEED_LEXICON "
            "(smoke only, not the human-owned profile)"
        )
    if not quarantined and not accepted and inputs == 0:
        notes.append("assemble: no engine assertions in the given outputs")

    return {
        "stage": STAGE,
        "status": "ok",
        "book_id": book_id,
        "counts": {
            "outputs": len(ordered),
            "inputs": inputs,
            "accepted": len(accepted),
            "quarantined": len(quarantined),
            "proposals": len(proposals),
            "duplicates": duplicates,
        "merged_evidence": merged_evidence,
            "events_input": events_input,
            "events": len(events),
            "event_duplicates": event_duplicates,
            "event_quarantined": sum(
                1 for item in quarantined if item.get("kind") == "event"
            ),
            "cues_input": cues_input,
            "cues": len(raw_cues),
            "by_claim_layer": dict(sorted(by_layer.items())),
            "by_library": dict(sorted(by_library.items())),
            "predicate_reconciliation": predicate_reconciliation,
        },
        "registry": {
            "source": registry_source,
            "digest": registry.digest(),
            "predicates": len(registry),
        },
        "resolution": {
            "source": resolution.source,
            "candidates": resolution.candidates,
            "surface_index": len(resolution.surface_to_id),
            "merge_redirects": len(resolution.redirect),
        },
        "quarantine": quarantined,
        "proposals": proposals,
        "paths": {
            "assertions": _rel(root, path),
            "events": _rel(root, event_path),
            "cues": _rel(root, cue_path),
        },
        "notes": notes,
    }
