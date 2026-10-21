"""Map engine candidates / work records into v2.2 box table records.

This is the strict translation layer between ``work/`` (disposable, candidate
records) and the box (immutable, schema-valid). Nothing here writes files.

Policy (box spec §5.6 / §6):

* spans are *repaired* by locating the quote/surface at the declared occurrence;
  if the text does not contain it the record is not silently kept — the caller
  turns the returned problem into an ``audit/quarantine`` row;
* the predicate registry decides ``library`` and role names; an unregistered
  predicate becomes a ``proposed_predicates`` row, never an assertion (I4);
* ids are content-addressed: ``m:``/``e:``/``a:``/``ev:``/``cue:``/``t:``/``cl:``;
* evidence always carries a recomputed ``quote_sha256`` (never trust the input).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1.box.lexicon import Lexicon, PredicateSpec, entity_type_slug
from cbb_stage1.ids import (
    assertion_id as _ids_assertion_id,
    claim_id as _ids_claim_id,
    cue_id as _ids_cue_id,
    entity_id as _ids_entity_id,
    event_id as _ids_event_id,
    mention_id as _ids_mention_id,
    merge_id as _ids_merge_id,
    proposal_id as _ids_proposal_id,
    sha256_text,
    timeline_id as _ids_timeline_id,
)

__all__ = [
    "Problem",
    "RunMeta",
    "CLOSED_REASONS",
    "normalize_surface",
    "line_span",
    "make_mention_id",
    "make_entity_id",
    "normalize_entity",
    "normalize_mention",
    "normalize_evidence",
    "normalize_assertion",
    "normalize_event",
    "normalize_cue",
    "normalize_claim",
    "normalize_timeline_constraint",
    "normalize_merge",
    "proposal_for_predicate",
    "to_engine_contract",
]

#: Closed quarantine reason set from box spec §5.12.
CLOSED_REASONS = frozenset(
    {
        "schema",
        "evidence_not_found",
        "role_mismatch",
        "ref_missing",
        "timeline_cycle",
        "contradiction",
        "low_confidence",
        "duplicate",
        "unresolved_quote",
        "claim_layer_conflict",
    }
)

#: ``model_dump`` enums from cbb_stage1.models that are box-valid.
_ASSERTION_STATUSES = ("asserted", "inferred", "disputed", "retracted", "superseded")
_CLAIM_LAYERS = ("fact", "statement", "inferred")
_CONFIDENCE_METHODS = (
    "engine_self",
    "evidence_span",
    "uqlm_whitebox",
    "uqlm_blackbox",
    "semantic_entropy",
    "uq_heads",
    "claim_qa",
    "human",
    "calibrated",
)
_QUALIFIER_MODALITIES = ("asserted", "inferred", "hypothetical", "quoted")
_EVIDENCE_MODES = ("narration", "dialogue", "inner", "memory", "document", "letter")
_MENTION_TYPES = ("name", "alias", "pronoun", "title", "description")
_EVENT_IMPORTANCE = ("high", "medium", "low")
_CUE_KINDS = ("object", "line", "anomaly", "prophecy", "relationship")
_CUE_STATUSES = (
    "candidate",
    "active",
    "triggered",
    "dormant",
    "paid",
    "falsified",
    "unresolved",
)
_TIMELINE_KINDS = ("before", "after", "during", "overlap")
_TIMELINE_SOURCES = ("explicit", "inferred")
_TIMELINE_STATUSES = ("asserted", "inferred", "disputed")
_MERGE_KINDS = ("alias", "coref", "type_unify")
_MERGE_DECIDERS = ("rule", "llm", "human", "panel")
_CLAIM_TYPES = ("attribute", "relation", "event", "setting")
_CLAIM_STANCES = ("support", "refute", "irrelevant", "unanswerable")
_CLAIM_VERDICTS = ("supported", "refuted", "conflicting", "insufficient_evidence")
_CLAIM_STATUSES = ("open", "verified", "rejected")

_ENTITY_ID_RE = re.compile(r"^e:[a-z0-9_-]+:[a-z_]+:.+$")
_ASSERTION_ID_RE = re.compile(r"^a:[0-9a-f]{16,64}$")
_EVENT_ID_RE = re.compile(r"^ev:[0-9a-f]{16,64}$")
_CUE_ID_RE = re.compile(r"^cue:[0-9a-f]{16,64}$")
_TIMELINE_ID_RE = re.compile(r"^t:[0-9a-f]{16,64}$")
_CLAIM_ID_RE = re.compile(r"^cl:[0-9a-f]{16,64}$")
_MERGE_ID_RE = re.compile(r"^mg:[0-9a-f]{16,64}$")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One rejected candidate: closed ``reason`` + human-readable detail."""

    reason: str
    detail: str

    def __post_init__(self) -> None:
        if self.reason not in CLOSED_REASONS:
            raise ValueError(f"unknown quarantine reason: {self.reason!r}")


@dataclass(frozen=True)
class RunMeta:
    """Provenance stamped on every box record (``produced_by``)."""

    run_id: str
    box_id: str
    engine: str
    model: str
    prompt_sha256: str
    started_at: str
    ended_at: str

    def produced_by(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine": self.engine,
            "prompt_sha256": self.prompt_sha256,
        }


def normalize_surface(text: str) -> str:
    """NFKC + whitespace-collapse key for name/alias linking."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def line_span(text: str, index: int) -> tuple[int, int]:
    """Character span of the line containing ``index`` (for sentence fields)."""
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end == -1:
        end = len(text)
    return start, end


def make_mention_id(unit_id: str, start: int, end: int, surface: str) -> str:
    return _ids_mention_id(unit_id, start, end, surface)


def make_entity_id(book_id: str, type_slug: str, canonical_name: str) -> str:
    return _ids_entity_id(book_id, type_slug, canonical_name)


def _row(value: Any) -> dict[str, Any]:
    """Normalise a pydantic model / mapping into a plain dict."""
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump())
    raise TypeError(f"expected a mapping or pydantic model, got {type(value).__name__}")


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _occurrences(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    if not needle:
        return positions
    start = 0
    while True:
        found = text.find(needle, start)
        if found == -1:
            return positions
        positions.append(found)
        start = found + max(1, len(needle))


def _locate(
    text: str, needle: str, start: Any, end: Any, occurrence: Any
) -> tuple[int, int, int] | None:
    """Return ``(start, end, occurrence)`` for ``needle`` in ``text`` or None."""
    if not needle:
        return None
    positions = _occurrences(text, needle)
    if not positions:
        return None
    declared_start = _as_int(start, -1)
    declared_end = _as_int(end, -1)
    if (
        0 <= declared_start < declared_end <= len(text)
        and text[declared_start:declared_end] == needle
    ):
        if declared_start in positions:
            occurrence_no = positions.index(declared_start) + 1
        else:  # narrowed by Python slicing semantics; keep declared occurrence
            occurrence_no = max(1, _as_int(occurrence, 1))
        return declared_start, declared_end, occurrence_no
    occurrence_no = max(1, _as_int(occurrence, 1))
    index = occurrence_no - 1 if occurrence_no - 1 < len(positions) else 0
    found = positions[index]
    return found, found + len(needle), index + 1


_EVIDENCE_CORE_KEYS = ("unit_id", "start", "end", "quote", "quote_sha256", "occurrence")


def _core_evidence(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Strip ``speaker``/``mode`` for tables whose schema forbids them (cue/timeline)."""
    return [
        {key: row[key] for key in _EVIDENCE_CORE_KEYS if key in row}
        for row in rows
    ]


def _speaker(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    kind = str(value.get("kind") or "narrator")
    if kind not in ("entity", "narrator"):
        return None
    entity_id = value.get("entity_id")
    if entity_id is not None and not str(entity_id).startswith("e:"):
        entity_id = None
    name = value.get("name")
    return {
        "kind": kind,
        "entity_id": str(entity_id) if entity_id else None,
        "name": str(name) if name is not None else None,
    }


# ---------------------------------------------------------------------------
# evidence / mentions / entities
# ---------------------------------------------------------------------------


def normalize_evidence(
    rows: Iterable[Any] | None,
    unit_texts: Mapping[str, str],
    *,
    default_unit: str,
) -> tuple[list[dict[str, Any]], list[Problem]]:
    """Validate/repair evidence rows against the copied source text.

    The returned evidence is box-shaped (``unit_id/start/end/quote/
    quote_sha256/occurrence`` and optional ``speaker``/``mode``). Quotes that
    cannot be located in their unit are dropped with an ``unresolved_quote``
    problem; the caller decides whether the owning record survives.
    """
    records: list[dict[str, Any]] = []
    problems: list[Problem] = []
    for row in rows or ():
        try:
            item = _row(row)
        except TypeError as exc:
            problems.append(Problem("schema", str(exc)))
            continue
        unit_id = str(item.get("unit_id") or default_unit or "")
        if unit_id not in unit_texts:
            problems.append(Problem("ref_missing", f"evidence unit not in box: {unit_id}"))
            continue
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote:
            problems.append(Problem("unresolved_quote", f"empty evidence quote in {unit_id}"))
            continue
        located = _locate(
            unit_texts[unit_id],
            quote,
            item.get("start"),
            item.get("end"),
            item.get("occurrence"),
        )
        if located is None:
            problems.append(
                Problem("unresolved_quote", f"quote not found in {unit_id}: {quote!r}")
            )
            continue
        start, end, occurrence = located
        evidence: dict[str, Any] = {
            "unit_id": unit_id,
            "start": start,
            "end": end,
            "quote": quote,
            "quote_sha256": sha256_text(quote),
            "occurrence": occurrence,
        }
        speaker = _speaker(item.get("speaker"))
        if speaker is not None:
            evidence["speaker"] = speaker
        mode = str(item.get("mode") or "narration")
        evidence["mode"] = mode if mode in _EVIDENCE_MODES else "narration"
        records.append(evidence)
    return records, problems


def normalize_mention(
    row: Any,
    unit_text: str,
    unit_id: str,
    *,
    produced_by: str,
) -> tuple[dict[str, Any] | None, list[Problem]]:
    """Map one engine mention into ``identity/mentions.jsonl`` shape."""
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))]
    problems: list[Problem] = []
    surface = str(item.get("surface") or "").strip()
    if not surface:
        return None, [Problem("schema", "mention without surface")]
    located = _locate(
        unit_text,
        surface,
        item.get("start"),
        item.get("end"),
        item.get("occurrence"),
    )
    if located is None:
        return None, [
            Problem(
                "unresolved_quote",
                f"mention surface not in {unit_id}: {surface!r}",
            )
        ]
    start, end, occurrence = located
    mention_type = str(item.get("mention_type") or "name")
    if mention_type not in _MENTION_TYPES:
        mention_type = "name"
    sentence_start, sentence_end = line_span(unit_text, start)
    raw_entity_id = item.get("entity_id")
    entity_id = str(raw_entity_id) if raw_entity_id and str(raw_entity_id).startswith("e:") else None
    if raw_entity_id is not None and entity_id is None:
        problems.append(Problem("schema", f"mention entity_id ignored: {raw_entity_id!r}"))
    record = {
        "mention_id": make_mention_id(unit_id, start, end, surface),
        "unit_id": unit_id,
        "surface": surface,
        "start": start,
        "end": end,
        "occurrence": occurrence,
        "sentence_span": [sentence_start, sentence_end],
        "sentence": unit_text[sentence_start:sentence_end],
        "entity_id": entity_id,
        "mention_type": mention_type,
        "confidence": _clamp01(item.get("confidence", 0.0)),
        "produced_by": produced_by,
    }
    return record, problems


def normalize_entity(
    row: Any,
    *,
    book_id: str,
    output_unit: str,
    produced_by: str,
) -> tuple[dict[str, Any] | None, list[Problem]]:
    """Map one engine entity into ``identity/entities.jsonl`` shape."""
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))]
    name = str(item.get("canonical_name") or item.get("name") or "").strip()
    if not name:
        return None, [Problem("schema", "entity without canonical_name")]
    type_slug = entity_type_slug(item.get("type"))
    tier = str(item.get("tier") or "minor")
    if tier not in ("canonical", "major", "minor"):
        tier = "minor"
    aliases: list[str] = []
    for alias in item.get("aliases") or ():
        alias = str(alias).strip()
        if alias and alias != name and alias not in aliases:
            aliases.append(alias)
    supplied_id = item.get("entity_id")
    entity_id = (
        str(supplied_id)
        if supplied_id and _ENTITY_ID_RE.match(str(supplied_id))
        else make_entity_id(book_id, type_slug, name)
    )
    status = "inactive" if item.get("status") == "inactive" else "active"
    record = {
        "entity_id": entity_id,
        "book_id": book_id,
        "type": type_slug,
        "canonical_name": name,
        "aliases": aliases,
        "tier": tier,
        "name_history": [
            {"name": name, "from_unit": output_unit, "reason": "first_seen"}
        ],
        "first_unit": output_unit,
        "last_unit": output_unit,
        "status": status,
        "merged_into": None,
        "confidence": _clamp01(item.get("confidence", 0.0)),
        "produced_by": produced_by,
    }
    return record, []


# ---------------------------------------------------------------------------
# assertions / events / cues
# ---------------------------------------------------------------------------


def _resolve_role_value(
    value: Any,
    role_type: str,
    entities_by_id: Mapping[str, Mapping[str, Any]],
    name_index: Mapping[str, str],
) -> tuple[Any | None, Problem | None]:
    if role_type == "entity":
        if isinstance(value, Mapping):
            inner = value.get("entity") or value.get("entity_id")
            if inner is None:
                return None, Problem("role_mismatch", "entity role given a literal object")
            value = inner
        if not isinstance(value, str) or not value.strip():
            return None, Problem("role_mismatch", "entity role must be an id/name string")
        raw = value.strip()
        if raw in entities_by_id:
            return raw, None
        linked = name_index.get(normalize_surface(raw))
        if linked:
            return linked, None
        if raw.startswith("e:"):
            return None, Problem("ref_missing", f"entity id not in identity table: {raw}")
        return None, Problem("ref_missing", f"entity name not resolved: {raw}")

    # literal role
    if isinstance(value, Mapping):
        if "literal" not in value:
            return None, Problem("role_mismatch", "literal role without 'literal'")
        literal = {"literal": str(value.get("literal"))}
        datatype = value.get("datatype")
        if datatype in ("string", "number", "time", "boolean"):
            literal["datatype"] = datatype
        return literal, None
    if isinstance(value, str):
        return {"literal": value}, None
    return None, Problem("role_mismatch", "literal role must be a string or {literal}")


def _normalize_roles(
    spec: PredicateSpec,
    roles: Any,
    entities_by_id: Mapping[str, Mapping[str, Any]],
    name_index: Mapping[str, str],
) -> tuple[dict[str, Any] | None, list[Problem]]:
    if not isinstance(roles, Mapping) or not roles:
        return None, [Problem("role_mismatch", "roles must be a non-empty object")]
    declared = spec.role_by_name
    problems: list[Problem] = []
    resolved: dict[str, Any] = {}
    for role_name, raw_value in roles.items():
        role = declared.get(str(role_name))
        if role is None:
            problems.append(
                Problem("role_mismatch", f"{spec.pred!r} has no role {role_name!r}")
            )
            continue
        if isinstance(raw_value, list):
            if not role.repeatable:
                problems.append(
                    Problem("role_mismatch", f"role {role_name!r} is not repeatable")
                )
                continue
            values: list[Any] = []
            for item in raw_value:
                value, problem = _resolve_role_value(
                    item, role.type, entities_by_id, name_index
                )
                if problem:
                    problems.append(problem)
                elif value is not None:
                    values.append(value)
            if values:
                resolved[str(role_name)] = values
            continue
        value, problem = _resolve_role_value(
            raw_value, role.type, entities_by_id, name_index
        )
        if problem:
            problems.append(problem)
        elif value is not None:
            resolved[str(role_name)] = value
    for required in spec.required_roles:
        if required not in resolved:
            problems.append(Problem("role_mismatch", f"missing required role {required!r}"))
    if problems:
        return None, problems
    return resolved, []


def normalize_assertion(
    row: Any,
    *,
    output_unit: str,
    unit_texts: Mapping[str, str],
    lexicon: Lexicon,
    entities_by_id: Mapping[str, Mapping[str, Any]],
    name_index: Mapping[str, str],
    run_meta: RunMeta,
    book_id: str,
) -> tuple[dict[str, Any] | None, list[Problem], dict[str, Any] | None]:
    """Map one engine assertion.

    Returns ``(record, problems, proposal)``. Exactly one of ``record`` /
    ``proposal`` is set when the input is usable; unregistered predicates are
    returned as a ``proposed_predicates`` proposal (I4).
    """
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))], None

    predicate = str(item.get("predicate") or "").strip()
    if not predicate:
        return None, [Problem("schema", "assertion without predicate")], None
    spec = lexicon.predicate(predicate)
    if spec is None:
        return None, [], proposal_for_predicate(
            book_id, predicate, output_unit, item.get("roles")
        )

    evidence, evidence_problems = normalize_evidence(
        item.get("evidence"), unit_texts, default_unit=output_unit
    )
    problems = list(evidence_problems)
    if not evidence:
        problems.append(Problem("evidence_not_found", "assertion has no resolvable evidence"))
        return None, problems, None

    roles, role_problems = _normalize_roles(
        spec, item.get("roles"), entities_by_id, name_index
    )
    if roles is None:
        return None, problems + role_problems, None

    status = str(item.get("status") or "asserted")
    if status not in _ASSERTION_STATUSES:
        status = "inferred" if item.get("claim_layer") == "inferred" else "asserted"
    raw_layer = str(item.get("claim_layer") or "")
    claim_layer = raw_layer if raw_layer in _CLAIM_LAYERS else (
        "inferred" if status == "inferred" else "fact"
    )
    if claim_layer == "statement":
        speaker = _speaker(item.get("speaker"))
        if speaker is None or (
            speaker["kind"] == "entity" and not speaker.get("entity_id")
        ):
            problems.append(
                Problem("claim_layer_conflict", "statement assertion without valid speaker")
            )
            return None, problems, None
    else:
        speaker = _speaker(item.get("speaker"))

    qualifiers = item.get("qualifiers")
    if not isinstance(qualifiers, Mapping):
        qualifiers = {}
    modality = str(qualifiers.get("modality") or "asserted")
    if modality not in _QUALIFIER_MODALITIES:
        modality = "inferred" if claim_layer == "inferred" else "asserted"
    time_value = qualifiers.get("time")
    if not isinstance(time_value, Mapping):
        time_value = None
    location = qualifiers.get("location")
    location_id = (
        str(location)
        if isinstance(location, str) and location in entities_by_id
        else None
    )
    qualifiers_record = {
        "negated": bool(qualifiers.get("negated", False)),
        "modality": modality,
        "time": dict(time_value) if time_value else None,
        "location": location_id,
    }

    confidence = _clamp01(item.get("confidence", 0.0))
    method = str(item.get("confidence_method") or "")
    if method not in _CONFIDENCE_METHODS:
        method = "engine_self" if confidence else "evidence_span"
    verification = item.get("verification")
    if not (
        isinstance(verification, Mapping)
        and verification.get("level") in ("canonical", "major", "minor", "none")
        and verification.get("verdict") in ("support", "against", "unsure", "none")
    ):
        verification = None

    derived_from = [
        str(value)
        for value in item.get("derived_from") or ()
        if str(value).startswith(("u:", "a:", "e:", "m:", "ev:", "cue:"))
    ]
    deps = sorted(
        {
            evidence_item["unit_id"]
            for evidence_item in evidence
        }
        | {
            value
            for value in _entity_ids_in_roles(roles)
        }
        | set(derived_from)
    )
    assertion_id = str(item.get("assertion_id") or "")
    if not _ASSERTION_ID_RE.match(assertion_id):
        assertion_id = _ids_assertion_id(
            book_id, predicate, roles, evidence[0]["quote_sha256"]
        )
    record = {
        "assertion_id": assertion_id,
        "book_id": book_id,
        "predicate": predicate,
        "library": spec.library,
        "roles": roles,
        "qualifiers": qualifiers_record,
        "claim_layer": claim_layer,
        "speaker": speaker,
        "evidence": evidence,
        "confidence": confidence,
        "confidence_method": method,
        "confidence_scope": "assertion",
        "verification": verification,
        "superseded_by": (
            str(item.get("superseded_by"))
            if item.get("superseded_by")
            and str(item.get("superseded_by")).startswith("a:")
            else None
        ),
        "status": status,
        "derived_from": derived_from,
        "deps": deps,
        "produced_by": run_meta.produced_by(),
        "record_sha256": sha256_text(
            json.dumps(
                [predicate, roles, evidence[0]["quote_sha256"]],
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
    }
    return record, problems, None


def _entity_ids_in_roles(roles: Mapping[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value.startswith("e:"):
                found.append(value)
        elif isinstance(value, Mapping):
            for inner in value.values():
                walk(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                walk(inner)

    walk(roles)
    return found


def normalize_event(
    row: Any,
    *,
    output_unit: str,
    unit_texts: Mapping[str, str],
    entities_by_id: Mapping[str, Mapping[str, Any]],
    name_index: Mapping[str, str],
    run_meta: RunMeta,
    book_id: str,
) -> tuple[dict[str, Any] | None, list[Problem]]:
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))]
    problems: list[Problem] = []
    event_type = str(item.get("type") or "").strip()
    summary = str(item.get("summary") or "").strip()
    if not event_type or not summary:
        return None, [Problem("schema", "event needs type and summary")]

    participants: list[str] = []
    for raw in item.get("participants") or ():
        value, problem = _resolve_role_value(raw, "entity", entities_by_id, name_index)
        if problem:
            problems.append(problem)
        elif value and value not in participants:
            participants.append(str(value))
    if not participants:
        problems.append(Problem("ref_missing", "event has no resolvable participants"))
        return None, problems

    unit_id = str(item.get("unit_id") or output_unit)
    if unit_id not in unit_texts:
        return None, problems + [Problem("ref_missing", f"event unit missing: {unit_id}")]
    evidence, evidence_problems = normalize_evidence(
        item.get("evidence"), unit_texts, default_unit=unit_id
    )
    problems.extend(evidence_problems)
    if not evidence:
        problems.append(Problem("evidence_not_found", "event has no resolvable evidence"))
        return None, problems

    location = item.get("location")
    location_id = None
    if location:
        value, problem = _resolve_role_value(
            location, "entity", entities_by_id, name_index
        )
        if problem:
            problems.append(problem)
        elif value:
            location_id = str(value)

    time_row = item.get("time")
    time_row = dict(time_row) if isinstance(time_row, Mapping) else {}
    anchor = time_row.get("anchor_unit") or unit_id
    if anchor not in unit_texts:
        anchor = unit_id
    story_time = time_row.get("story_time")
    story_time = dict(story_time) if isinstance(story_time, Mapping) else None
    expression = time_row.get("expression")
    if expression is None:
        expression = time_row.get("time_expr")
    expression = str(expression) if expression else None
    time_record = {
        "anchor_unit": anchor,
        "expression": expression,
        "story_time": story_time,
        "constraints": [],
    }
    if expression and expression in unit_texts[unit_id]:
        time_record["time_expr"] = expression

    importance = str(item.get("importance") or "medium")
    if importance not in _EVENT_IMPORTANCE:
        importance = "medium"
    status = str(item.get("status") or "asserted")
    if status not in _ASSERTION_STATUSES:
        status = "asserted"
    event_id = str(item.get("event_id") or "")
    if not _EVENT_ID_RE.match(event_id):
        event_id = _ids_event_id(
            book_id, event_type, unit_id, participants, summary
        )
    record = {
        "event_id": event_id,
        "book_id": book_id,
        "type": event_type,
        "summary": summary,
        "participants": participants,
        "location": location_id,
        "time": time_record,
        "importance": importance,
        "unit_id": unit_id,
        "evidence": evidence,
        "status": status,
        "superseded_by": None,
        "deps": sorted({unit_id} | set(participants) | {ev["unit_id"] for ev in evidence}),
        "produced_by": run_meta.produced_by(),
    }
    return record, problems


def normalize_cue(
    row: Any,
    *,
    output_unit: str,
    unit_texts: Mapping[str, str],
    entities_by_id: Mapping[str, Mapping[str, Any]],
    run_meta: RunMeta,
    book_id: str,
) -> tuple[dict[str, Any] | None, list[Problem]]:
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))]
    problems: list[Problem] = []
    kind = str(item.get("kind") or "object")
    if kind not in _CUE_KINDS:
        problems.append(Problem("schema", f"unknown cue kind: {kind!r}"))
        kind = "object"

    subject_row = item.get("subject")
    subject_row = dict(subject_row) if isinstance(subject_row, Mapping) else {}
    subject_entity = subject_row.get("entity_id")
    if subject_entity is not None and str(subject_entity) not in entities_by_id:
        problems.append(
            Problem("ref_missing", f"cue subject entity missing: {subject_entity}")
        )
        subject_entity = None
    subject_literal = subject_row.get("literal")
    subject = {
        "entity_id": str(subject_entity) if subject_entity else None,
        "literal": str(subject_literal) if subject_literal is not None else None,
    }
    if not subject["entity_id"] and not subject["literal"]:
        return None, problems + [Problem("schema", "cue needs subject entity or literal")]

    setup_row = item.get("setup")
    setup_row = dict(setup_row) if isinstance(setup_row, Mapping) else {}
    setup_unit = str(setup_row.get("unit_id") or output_unit)
    if setup_unit not in unit_texts:
        return None, problems + [Problem("ref_missing", f"cue setup unit missing: {setup_unit}")]
    setup_evidence, setup_problems = normalize_evidence(
        setup_row.get("evidence"), unit_texts, default_unit=setup_unit
    )
    problems.extend(setup_problems)
    if not setup_evidence:
        problems.append(Problem("evidence_not_found", "cue setup without resolvable evidence"))
        return None, problems

    reinforce_evidence, reinforce_problems = normalize_evidence(
        item.get("reinforce"), unit_texts, default_unit=output_unit
    )
    problems.extend(reinforce_problems)

    payoff_row = item.get("payoff")
    payoff_row = dict(payoff_row) if isinstance(payoff_row, Mapping) else {}
    payoff_event = payoff_row.get("event_id")
    payoff_unit = payoff_row.get("unit_id")
    if payoff_unit is not None and str(payoff_unit) not in unit_texts:
        problems.append(Problem("ref_missing", f"cue payoff unit missing: {payoff_unit}"))
        payoff_unit = None
    payoff_evidence, payoff_problems = normalize_evidence(
        payoff_row.get("evidence"), unit_texts, default_unit=output_unit
    )
    problems.extend(payoff_problems)

    status = str(item.get("status") or "active")
    if status not in _CUE_STATUSES:
        status = "active" if setup_evidence else "candidate"
    trigger_row = item.get("trigger")
    trigger = None
    if isinstance(trigger_row, Mapping):
        trigger_evidence, trigger_problems = normalize_evidence(
            trigger_row.get("evidence"), unit_texts, default_unit=output_unit
        )
        problems.extend(trigger_problems)
        trigger = {
            "unit_id": str(trigger_row["unit_id"])
            if trigger_row.get("unit_id") in unit_texts
            else None,
            "event_id": str(trigger_row["event_id"])
            if trigger_row.get("event_id")
            else None,
            "condition": trigger_row.get("condition"),
            "evidence": _core_evidence(trigger_evidence),
        }
    if status in ("triggered", "paid") and not (trigger and trigger.get("condition")):
        problems.append(Problem("schema", f"cue status {status} without trigger condition"))
        status = "active"
    if status == "paid":
        if not payoff_evidence:
            problems.append(Problem("evidence_not_found", "paid cue without payoff evidence"))
            status = "triggered"
        elif item.get("triggered_at") is None:
            problems.append(Problem("schema", "paid cue before triggered"))
            status = "triggered"
    if status == "falsified" and not item.get("falsified_reason"):
        problems.append(Problem("schema", "falsified cue without reason"))
        status = "active"
    # Keep the state machine lint-clean after downgrades.
    if status == "triggered" and not (trigger and trigger.get("condition")):
        status = "active"
    if status == "paid":
        if not (trigger and trigger.get("condition")):
            status = "active"
        elif item.get("triggered_at") is None and (item.get("activated_at") or setup_unit):
            item = dict(item)
            item["triggered_at"] = item.get("activated_at") or setup_unit

    record: dict[str, Any] = {
        "cue_id": str(item.get("cue_id") or ""),
        "book_id": book_id,
        "kind": kind,
        "subject": subject,
        "setup": {"unit_id": setup_unit, "evidence": _core_evidence(setup_evidence)},
        "reinforce": _core_evidence(reinforce_evidence),
        "payoff": {
            "event_id": str(payoff_event) if payoff_event else None,
            "unit_id": str(payoff_unit) if payoff_unit else None,
            "evidence": _core_evidence(payoff_evidence),
        },
        "status": status,
        "activated_at": item.get("activated_at") or setup_unit,
        "triggered_at": item.get("triggered_at"),
        "dormant_since": item.get("dormant_since"),
        "falsified_reason": item.get("falsified_reason"),
        "premature_payoff": bool(item.get("premature_payoff", False)),
        "expected": str(item.get("expected") or ""),
        "confidence": _clamp01(item.get("confidence", 0.0)),
        "deps": sorted({setup_unit} | {ev["unit_id"] for ev in setup_evidence}),
    }
    if trigger is not None:
        record["trigger"] = trigger
    if not _CUE_ID_RE.match(record["cue_id"]):
        record["cue_id"] = _ids_cue_id(book_id, kind, subject, setup_unit)
    return record, problems


# ---------------------------------------------------------------------------
# claims / timeline / merges / proposals
# ---------------------------------------------------------------------------


def _normalize_claim_answer(
    row: Any, unit_texts: Mapping[str, str]
) -> tuple[dict[str, Any] | None, Problem | None]:
    try:
        item = _row(row)
    except TypeError as exc:
        return None, Problem("schema", str(exc))
    unit_id = str(item.get("unit_id") or "")
    if unit_id not in unit_texts:
        return None, Problem("ref_missing", f"claim answer unit missing: {unit_id}")
    quote = item.get("quote")
    if not isinstance(quote, str) or not quote:
        return None, Problem("unresolved_quote", "claim answer without quote")
    located = _locate(
        unit_texts[unit_id], quote, item.get("start"), item.get("end"), item.get("occurrence", 1)
    )
    if located is None:
        return None, Problem("unresolved_quote", f"claim answer quote missing: {quote!r}")
    start, end, _ = located
    return (
        {
            "text": str(item.get("text") or quote),
            "unit_id": unit_id,
            "start": start,
            "end": end,
            "quote": quote,
            "quote_sha256": sha256_text(quote),
            "source": str(item.get("source") or "narration"),
        },
        None,
    )


def normalize_claim(
    row: Any,
    *,
    book_id: str,
    unit_texts: Mapping[str, str],
    assertion_layers: Mapping[str, str],
    run_meta: RunMeta,
) -> tuple[dict[str, Any] | None, list[Problem]]:
    """Validate a claim/QA row (I22-I23); recomputes the four-value verdict."""
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))]
    problems: list[Problem] = []
    statement_id = str(item.get("statement_assertion_id") or "")
    if statement_id not in assertion_layers:
        return None, [Problem("ref_missing", f"claim statement missing: {statement_id}")]
    if assertion_layers[statement_id] != "statement":
        return None, [
            Problem(
                "claim_layer_conflict",
                f"claim points at {assertion_layers[statement_id]!r}, not 'statement'",
            )
        ]

    questions: list[dict[str, Any]] = []
    for question in item.get("questions") or ():
        if not isinstance(question, Mapping):
            problems.append(Problem("schema", "claim question must be an object"))
            continue
        answers: list[dict[str, Any]] = []
        for answer in question.get("answers") or ():
            record, problem = _normalize_claim_answer(answer, unit_texts)
            if problem:
                problems.append(problem)
            elif record:
                answers.append(record)
        stance = str(question.get("stance") or "unanswerable")
        if stance not in _CLAIM_STANCES:
            stance = "unanswerable"
        questions.append(
            {
                "qid": str(question.get("qid") or f"q{len(questions) + 1}"),
                "question": str(question.get("question") or ""),
                "answers": answers,
                "stance": stance,
            }
        )
    if not questions:
        return None, problems + [Problem("schema", "claim needs at least one question")]

    stances = [question["stance"] for question in questions]
    has_support = "support" in stances
    has_refute = "refute" in stances
    verdict = (
        "conflicting"
        if has_support and has_refute
        else "supported"
        if has_support
        else "refuted"
        if has_refute
        else "insufficient_evidence"
    )

    atomic_facts: list[dict[str, Any]] = []
    for atomic in item.get("atomic_facts") or ():
        if not isinstance(atomic, Mapping):
            problems.append(Problem("schema", "atomic fact must be an object"))
            continue
        atomic_verdict = atomic.get("verdict")
        if atomic_verdict not in ("true", "false", "unverifiable"):
            problems.append(Problem("schema", "bad atomic fact verdict"))
            continue
        atomic_facts.append(
            {
                "fact_id": str(atomic.get("fact_id") or f"af:{len(atomic_facts) + 1}"),
                "text": str(atomic.get("text") or ""),
                "verdict": atomic_verdict,
                "evidence": list(atomic.get("evidence") or []),
                "rationale": atomic.get("rationale"),
            }
        )
    if any(fact["verdict"] == "false" for fact in atomic_facts) and verdict == "supported":
        # I23 + box_lint conflict: a supported claim cannot contain a false
        # atomic fact. Reject for adjudication instead of writing an invalid row.
        return None, problems + [
            Problem("contradiction", "supported verdict with a false atomic fact")
        ]

    normalized = str(item.get("normalized_claim") or "").strip()
    if not normalized:
        return None, problems + [Problem("schema", "claim needs normalized_claim")]
    claim_id = str(item.get("claim_id") or "")
    if not _CLAIM_ID_RE.match(claim_id):
        # Byte-compatible with the previous formula: an invalid supplied id was
        # historically folded into the hash input, so keep that exact input.
        claim_id = _ids_claim_id(statement_id, claim_id or normalized)
    claim_type = str(item.get("claim_type") or "attribute")
    if claim_type not in _CLAIM_TYPES:
        claim_type = "attribute"
    confidence_method = str(item.get("confidence_method") or "claim_qa")
    record = {
        "claim_id": claim_id,
        "book_id": book_id,
        "statement_assertion_id": statement_id,
        "normalized_claim": normalized,
        "claim_type": claim_type,
        "questions": questions,
        "atomic_facts": atomic_facts,
        "verdict": verdict,
        "verdict_rule": str(item.get("verdict_rule") or "av-eritec"),
        "justification": item.get("justification"),
        "confidence": _clamp01(item.get("confidence", 0.0)),
        "confidence_method": confidence_method,
        "confidence_scope": "claim",
        "verified_by": dict(item.get("verified_by") or {}),
        "status": (
            str(item.get("status"))
            if item.get("status") in _CLAIM_STATUSES
            else "open"
        ),
        "created_at": item.get("created_at") or run_meta.ended_at,
    }
    return record, problems


def normalize_timeline_constraint(
    row: Any,
    *,
    book_id: str,
    unit_texts: Mapping[str, str],
    unit_ids: set[str],
    event_ids: set[str],
) -> tuple[dict[str, Any] | None, list[Problem]]:
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))]
    problems: list[Problem] = []
    kind = str(item.get("kind") or "")
    if kind not in _TIMELINE_KINDS:
        return None, [Problem("schema", f"bad timeline kind: {kind!r}")]

    def _node(value: Any) -> dict[str, str] | None:
        if not isinstance(value, Mapping):
            return None
        node_kind = str(value.get("kind") or "")
        node_id = str(value.get("id") or "")
        if node_kind == "event" and node_id in event_ids:
            return {"kind": "event", "id": node_id}
        if node_kind == "unit" and node_id in unit_ids:
            return {"kind": "unit", "id": node_id}
        if node_kind == "time_point" and node_id:
            return {"kind": "time_point", "id": node_id}
        return None

    left = _node(item.get("left"))
    right = _node(item.get("right"))
    if left is None or right is None:
        return None, problems + [Problem("ref_missing", "timeline node missing")]
    source = str(item.get("source") or "inferred")
    if source not in _TIMELINE_SOURCES:
        source = "inferred"
    status = str(item.get("status") or "inferred")
    if status not in _TIMELINE_STATUSES:
        status = "inferred"
    evidence, evidence_problems = normalize_evidence(item.get("evidence"), unit_texts, default_unit="")
    problems.extend(evidence_problems)
    verify = item.get("verify")
    if not (
        isinstance(verify, Mapping)
        and verify.get("method") in ("nli_pairwise", "explicit_expression", "topological")
        and verify.get("result") in ("yes", "no", "unknown")
    ):
        verify = None
    constraint_id = str(item.get("constraint_id") or "")
    if not _TIMELINE_ID_RE.match(constraint_id):
        constraint_id = _ids_timeline_id(
            book_id,
            kind,
            left["kind"],
            left["id"],
            right["kind"],
            right["id"],
        )
    record = {
        "constraint_id": constraint_id,
        "book_id": book_id,
        "kind": kind,
        "left": left,
        "right": right,
        "source": source,
        "confidence": _clamp01(item.get("confidence", 0.0)),
        "verify": verify,
        "evidence": _core_evidence(evidence),
        "status": status,
        "deps": sorted(
            {
                node["id"]
                for node in (left, right)
                if node["kind"] in ("event", "unit")
            }
            | {ev["unit_id"] for ev in evidence}
        ),
    }
    return record, problems


def normalize_merge(
    row: Any,
    *,
    book_id: str,
    entity_ids: set[str],
) -> tuple[dict[str, Any] | None, list[Problem]]:
    try:
        item = _row(row)
    except TypeError as exc:
        return None, [Problem("schema", str(exc))]
    problems: list[Problem] = []
    from_ids = [str(value) for value in item.get("from") or () if str(value) in entity_ids]
    if not from_ids:
        return None, [Problem("ref_missing", "merge has no existing 'from' entity")]
    to_id = str(item.get("to") or "")
    if to_id not in entity_ids:
        return None, [Problem("ref_missing", f"merge 'to' missing: {to_id}")]
    kind = str(item.get("kind") or "alias")
    if kind not in _MERGE_KINDS:
        return None, [Problem("schema", f"bad merge kind: {kind!r}")]
    decided_by = str(item.get("decided_by") or "rule")
    if decided_by not in _MERGE_DECIDERS:
        decided_by = "rule"
    merge_id = str(item.get("merge_id") or "")
    if not _MERGE_ID_RE.match(merge_id):
        merge_id = _ids_merge_id(book_id, kind, from_ids, to_id)
    at = item.get("at") or "1970-01-01T00:00:00+00:00"
    record = {
        "merge_id": merge_id,
        "book_id": book_id,
        "kind": kind,
        "from": from_ids,
        "to": to_id,
        "decided_by": decided_by,
        "confidence": _clamp01(item.get("confidence", 1.0)),
        "evidence": list(item.get("evidence") or ()),
        "at": at,
        "reversible": True,
    }
    return record, problems


def proposal_for_predicate(
    book_id: str, predicate: str, unit_id: str, roles: Any
) -> dict[str, Any]:
    """Build a ``knowledge/proposed_predicates.jsonl`` row (I4)."""
    return {
        "proposal_id": _ids_proposal_id(
            book_id, predicate, [unit_id] if unit_id else []
        ),
        "book_id": book_id,
        "predicate": predicate,
        "seen_in": [unit_id] if unit_id else [],
        "count": 1,
        "role_shape_examples": [dict(roles)] if isinstance(roles, Mapping) else [],
        "status": "proposed",
        "decided_by": None,
        "first_seen": None,
    }


# ---------------------------------------------------------------------------
# engine-output audit conversion (cbb-engine-output/1)
# ---------------------------------------------------------------------------


def _contract_evidence(rows: Iterable[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or ():
        try:
            item = _row(row)
        except TypeError:
            continue
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote:
            continue
        span = item.get("span")
        if isinstance(span, Sequence) and not isinstance(span, (str, bytes)) and len(span) == 2:
            start, end = _as_int(span[0], 0), _as_int(span[1], 0)
        else:
            start = _as_int(item.get("start"), 0)
            end = _as_int(item.get("end"), start + len(quote))
        out.append({"quote": quote, "span": [start, end]})
    return out


def _contract_assertion(row: Any) -> dict[str, Any] | None:
    try:
        item = _row(row)
    except TypeError:
        return None
    predicate = str(item.get("predicate") or "")
    if not predicate:
        return None
    roles = item.get("roles")
    roles = dict(roles) if isinstance(roles, Mapping) else {}
    subject = ""
    for key in ("主体", "施事", "subject"):
        value = roles.get(key)
        if isinstance(value, str) and value:
            subject = value
            break
    if not subject:
        for value in roles.values():
            if isinstance(value, str) and value:
                subject = value
                break
    obj: dict[str, Any] | None = None
    for key in ("值", "受事", "object"):
        value = roles.get(key)
        if isinstance(value, Mapping) and "literal" in value:
            obj = {"literal": str(value.get("literal"))}
            break
        if isinstance(value, str) and value and key != subject:
            obj = {"literal": value}
            break
    if obj is None and item.get("object") is not None:
        raw = _row(item["object"]) if isinstance(item["object"], Mapping) else {}
        if "literal" in raw:
            obj = {"literal": str(raw["literal"])}
    if obj is None:
        return None
    inferred = item.get("claim_layer") == "inferred" or item.get("status") == "inferred"
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence": _contract_evidence(item.get("evidence")),
        "confidence": _clamp01(item.get("confidence", 0.0)),
        "inferred": inferred,
    }


def to_engine_contract(
    output: Any,
    *,
    unit_text: str,
    engine_name: str,
    engine_version: str,
    engine_model: str,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert a pydantic ``EngineOutput`` artifact to ``cbb-engine-output/1``.

    The work artifact is the pydantic model dump (``mentions`` without sentence
    fields, ``abstain`` as bool, ``summary`` as str). The audit copy must match
    the box's engine-output schema, so this function fills the gaps
    deterministically. It never invents extraction content: missing summaries
    stay ``null`` and a true ``abstain`` maps to the documented ``ambiguous``
    reason only when no more specific reason is available.
    """
    dump = _row(output)
    unit_id = str(dump.get("unit_id") or "")
    mentions: list[dict[str, Any]] = []
    for row in dump.get("mentions") or ():
        item = _row(row)
        surface = str(item.get("surface") or "")
        start = _as_int(item.get("start"), 0)
        end = _as_int(item.get("end"), start + len(surface))
        sentence_start, sentence_end = line_span(unit_text, start)
        mentions.append(
            {
                "surface": surface,
                "span": [start, end],
                "sentence_span": [sentence_start, sentence_end],
                "sentence": unit_text[sentence_start:sentence_end],
                "hint_type": None,
                "hint_alias_of": None,
            }
        )
    entities: list[dict[str, Any]] = []
    for index, row in enumerate(dump.get("entities") or (), 1):
        item = _row(row)
        name = str(item.get("canonical_name") or item.get("name") or "")
        if not name:
            continue
        tier = item.get("tier")
        entities.append(
            {
                "temp_id": str(item.get("entity_id") or f"t{index}"),
                "name": name,
                "type": item.get("type"),
                "aliases": [str(alias) for alias in item.get("aliases") or ()],
                "description": item.get("description"),
                "description_quote": None,
                "tier_hint": tier if tier in ("canonical", "major", "minor") else "minor",
            }
        )
    assertions = [
        record
        for record in (_contract_assertion(row) for row in dump.get("assertions") or ())
        if record is not None
    ]
    events: list[dict[str, Any]] = []
    for index, row in enumerate(dump.get("events") or (), 1):
        item = _row(row)
        summary = str(item.get("summary") or "")
        if not summary:
            continue
        events.append(
            {
                "temp_id": str(item.get("event_id") or f"e{index}"),
                "type": str(item.get("type") or "event"),
                "summary": summary,
                "participants": [str(p) for p in item.get("participants") or ()],
                "evidence": _contract_evidence(item.get("evidence")),
            }
        )
    cues: list[dict[str, Any]] = []
    for row in dump.get("cues") or ():
        item = _row(row)
        setup = item.get("setup")
        setup_quote = ""
        if isinstance(setup, Mapping):
            evidence_rows = setup.get("evidence") or ()
            for evidence_row in evidence_rows:
                quote = _row(evidence_row).get("quote")
                if isinstance(quote, str) and quote:
                    setup_quote = quote
                    break
        cues.append(
            {
                "kind": item.get("kind") if item.get("kind") in _CUE_KINDS else "object",
                "subject_literal": str(item.get("subject_literal") or ""),
                "setup_quote": setup_quote,
                "expected": str(item.get("expected") or ""),
                "confidence": _clamp01(item.get("confidence", 0.0)),
            }
        )
    summary = dump.get("summary")
    if isinstance(summary, str) and summary:
        summary_record: dict[str, Any] | None = {"text": summary, "evidence": []}
    else:
        summary_record = None
    abstain_value = dump.get("abstain")
    if isinstance(abstain_value, Mapping):
        reason = abstain_value.get("reason")
        abstain_record = (
            {"reason": reason}
            if reason in ("no_text", "too_long", "ambiguous", "schema_failed", "policy_refused")
            else None
        )
    elif abstain_value:
        abstain_record = {"reason": "no_text" if not unit_text else "ambiguous"}
    else:
        abstain_record = None

    contract: dict[str, Any] = {
        "contract": "cbb-engine-output/1",
        "unit_id": unit_id,
        "engine": {
            "name": engine_name,
            "version": engine_version,
            "model": engine_model,
        },
        "mentions": mentions,
        "entities": entities,
        "assertions": assertions,
        "events": events,
        "cues": cues,
        "summary": summary_record,
        "abstain": abstain_record,
        "notes": [str(note) for note in (list(notes) + list(dump.get("notes") or ()))],
    }
    gleaning_round = dump.get("gleaning_round")
    if isinstance(gleaning_round, int) and gleaning_round > 0:
        contract["gleaning_round"] = gleaning_round
    return contract
