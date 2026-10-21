"""Single-writer v2.2 box builder (work/ -> immutable box).

Reads the disposable work layer of the 9-stage pipeline:

* ``work/plan/units.jsonl`` (unit list; text lives next to it under
  ``work/plan/texts/``),
* ``work/plan/run.json`` (book/engine/run identity),
* ``work/engine/*.json`` (``EngineOutput`` candidates),
* the postprocess work tables (the writer consumes them, it does not produce
  parallel records):
  ``assemble/entity_cards.jsonl`` (dedupe), ``assemble/{claims,merges,timeline,
  cues}.jsonl``, ``resolve/{merges,timeline,cues}.jsonl`` fallbacks,
  ``audit/conflicts.jsonl`` (validate) and ``derive/{summaries,forecasts,
  hypotheses,community,raptor}.jsonl`` (derive);
* ``assemble/events.jsonl`` (assemble); engine events are a legacy fallback
  that always records a warning;
* media references in unit text are extracted into ``sources/assets.jsonl``
  (no media => explicit empty, which is valid).

Missing optional tables stay explicitly empty. The writer never synthesises a
claim, merge, conflict, hypothesis or cue row to fill a gap.

and writes a box matching ``01-箱规格-v2.md`` v2.2. The write is atomic:
everything goes into a hidden temp sibling directory, the manifest/CHECKSUMS
are computed over the real bytes, the canonical ``tools/box_lint.py`` gate runs
against the temp directory, and only then is the directory renamed to ``out``.
Boxes are immutable — :func:`build_box` refuses to overwrite an existing path.

Boundaries:

* this module never modifies ``work/`` and never calls an engine;
* only the builder writes the box (single-writer rule);
* unrepresentable candidates are quarantined (closed reason set) instead of
  being silently dropped or smuggled into another table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1.box import atomic, projection_tables, records
from cbb_stage1.box.gates import GATE_NAMES, measure_hard_gates
from cbb_stage1.box.lexicon import Lexicon, load_lexicon
from cbb_stage1.box.lint import canonical_lint_errors
from cbb_stage1.box.records import Problem, RunMeta
from cbb_stage1.derive.hypotheses import normalize_hypothesis_record
from cbb_stage1.derive.summaries import (
    SUMMARY_CONTRACT,
    build_summary_tree,
    mark_stale,
    summary_rows,
    validate_summary_tree,
)
from cbb_stage1.ids import (
    conflict_id as _ids_conflict_id,
    sha256_text as _ids_sha256_text,
    short_hash as _ids_short_hash,
)
from cbb_stage1.incremental.deps import dirty_closure

__all__ = ["BuildError", "BuildResult", "build_box", "canonical_lint_errors", "main"]

BOX_SPEC = "cbb-box/2"
SPEC_VERSION = "2.2.0"

#: Table file -> JSON Schema used by ``tools/box_lint.py``.
TABLE_SCHEMAS = {
    "sources/units.jsonl": "unit.schema.json",
    "identity/mentions.jsonl": "mention.schema.json",
    "identity/entities.jsonl": "entity.schema.json",
    "identity/merges.jsonl": "merge.schema.json",
    "knowledge/assertions.jsonl": "assertion.schema.json",
    "knowledge/claims.jsonl": "claim.schema.json",
    "knowledge/citations.jsonl": "citation.schema.json",
    "knowledge/narrative.jsonl": "narrative.schema.json",
    "knowledge/graph_edges.jsonl": "graph-edge.schema.json",
    "knowledge/conflicts.jsonl": "conflict.schema.json",
    "sources/assets.jsonl": "asset.schema.json",
    "knowledge/events.jsonl": "event.schema.json",
    "knowledge/timeline.jsonl": "timeline.schema.json",
    "knowledge/cues.jsonl": "cue.schema.json",
    "derived/hypotheses.jsonl": "hypothesis.schema.json",
    "derived/research.jsonl": "research.schema.json",
    "audit/runs.jsonl": "run.schema.json",
}

_LOCAL_TZ = timezone(timedelta(hours=8))
_UNIT_ID_RE = re.compile(r"^u:[a-z0-9_-]+:v[0-9]{2}:c[0-9]{3}(:s[0-9]{2})?$")
_BOOK_ID_RE = re.compile(r"^[a-z0-9_-]+$")
_NARRATIVE_MODES = ("present", "flashback", "prophecy", "mixed")
_NARRATOR_KINDS = ("first_person", "third_person", "third_limited", "unknown")
_NARRATOR_RELIABILITIES = ("reliable", "unreliable", "unknown")


class BuildError(RuntimeError):
    """Raised when a box cannot be built (work missing/invalid or gate failed)."""

    def __init__(self, message: str, errors: Iterable[str] = ()) -> None:
        self.message = message
        self.errors = tuple(errors)
        detail = message if not self.errors else f"{message}\n" + "\n".join(self.errors)
        super().__init__(detail)


@dataclass
class BuildResult:
    """Successful build summary (returned only after the box is published)."""

    box: Path
    box_id: str
    status: str
    counts: dict[str, int]
    quarantine: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    lint_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": str(self.box),
            "box_id": self.box_id,
            "status": self.status,
            "counts": dict(self.counts),
            "quarantined": len(self.quarantine),
            "warnings": list(self.warnings),
            "lint_errors": list(self.lint_errors),
        }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _iso(value: Any, fallback: str) -> str:
    """Return ``value`` as an ISO 8601 string with an offset when possible."""
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            return text[:-1] + "+00:00"
        if re.search(r"[+-][0-9]{2}:[0-9]{2}$", text):
            return text
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_LOCAL_TZ)
        return parsed.isoformat(timespec="seconds")
    return fallback


def _now_iso() -> str:
    return datetime.now(_LOCAL_TZ).isoformat(timespec="seconds")


def _sanitize_book_id(raw: Any, fallback: str = "book") -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text).strip("_")
    return text if text and _BOOK_ID_RE.match(text) else fallback


def _box_text_rel(unit: Mapping[str, Any], order: int) -> str:
    raw = str(unit.get("text_path") or "").replace("\\", "/")
    if "texts/" in raw:
        relative = raw.split("texts/", 1)[1]
    else:
        volume = max(1, _int(unit.get("volume"), 1))
        relative = f"v{volume:02d}/u{order:03d}.txt"
    return "sources/texts/" + relative.lstrip("/")


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _engine_info(run: Mapping[str, Any]) -> dict[str, str]:
    raw = run.get("engine") or "unknown"
    if isinstance(raw, Mapping):
        name = str(raw.get("name") or raw.get("primary") or "unknown")
        version = str(raw.get("version") or "0.1.0")
        model = str(raw.get("model") or "unknown")
    else:
        text = str(raw)
        if ":" in text:
            name, model = text.split(":", 1)
        else:
            name, model = text, "unknown"
        version = str(run.get("engine_version") or "0.1.0")
    prompt = _ids_sha256_text(f"engine:{name}|model:{model}|stage:extract")
    return {
        "name": name.strip() or "unknown",
        "version": version.strip() or "0.1.0",
        "model": model.strip() or "unknown",
        "prompt_sha256": prompt,
    }


def _schemas_dir() -> Path:
    # writer.py -> box -> cbb_stage1 -> src -> cbb_stage1 -> V5
    # Resolved by the chain in ``cbb_stage1.schemas``: shipped ``schemas/`` first, the
    # legacy design-tree sibling last. The old line went straight to the sibling, which
    # a published checkout does not have.
    from cbb_stage1.schemas import schemas_dir

    return schemas_dir()


def _quarantine_row(
    item_id: str, kind: str, problem: Problem, source_run: str
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "kind": kind,
        "reason": problem.reason,
        "detail": problem.detail,
        "source_run": source_run,
    }


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# unit / source normalisation
# ---------------------------------------------------------------------------


def _load_units(
    work_root: Path,
    book_id: str,
    created_at: str,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    units = atomic.read_jsonl(work_root / "plan" / "units.jsonl")
    if not units:
        raise BuildError(f"no units found under {work_root / 'plan' / 'units.jsonl'}")

    ordered = sorted(
        units,
        key=lambda unit: (
            _int(unit.get("order"), 10**9),
            _int(unit.get("chapter"), 10**9),
            str(unit.get("unit_id") or ""),
        ),
    )
    unit_texts: dict[str, str] = {}
    records_out: list[dict[str, Any]] = []
    for rank, source in enumerate(ordered, 1):
        unit_id = str(source.get("unit_id") or "")
        if not _UNIT_ID_RE.match(unit_id):
            raise BuildError(f"unit_id does not match box schema: {unit_id!r}")
        text_rel = source.get("text_path")
        if not text_rel:
            raise BuildError(f"unit {unit_id} has no text_path")
        text_path = work_root / str(text_rel)
        if not text_path.exists():
            raise BuildError(f"unit {unit_id} text missing: {text_path}")
        text = text_path.read_text(encoding="utf-8")
        if not text:
            raise BuildError(f"unit {unit_id} text file is empty")
        if len(text) != _int(source.get("chars"), len(text)):
            warnings.append(
                f"unit {unit_id}: chars {source.get('chars')} != actual {len(text)}; recomputed"
            )
        expected_sha = str(source.get("sha256") or "")
        actual_sha = _ids_sha256_text(text)
        if expected_sha and expected_sha != actual_sha:
            warnings.append(
                f"unit {unit_id}: sha256 mismatch; recomputed from disk text"
            )
        if unit_id in unit_texts:
            raise BuildError(f"duplicate unit_id: {unit_id}")
        unit_texts[unit_id] = text

        narrative_mode = str(source.get("narrative_mode") or "present")
        if narrative_mode not in _NARRATIVE_MODES:
            narrative_mode = "present"
        narrator = source.get("narrator")
        normalized_narrator: dict[str, Any] | None = None
        if isinstance(narrator, Mapping):
            kind = str(narrator.get("kind") or "unknown")
            if kind not in _NARRATOR_KINDS:
                kind = "unknown"
            entity_id = narrator.get("entity_id")
            normalized_narrator = {
                "kind": kind,
                "entity_id": str(entity_id)
                if entity_id and str(entity_id).startswith("e:")
                else None,
                "name": narrator.get("name"),
            }
        reliability = str(source.get("narrator_reliability") or "unknown")
        if reliability not in _NARRATOR_RELIABILITIES:
            reliability = "unknown"
        locator = source.get("source_locator")
        if not isinstance(locator, Mapping):
            locator = {}
        href = str(locator.get("href") or work_root.name)
        nav_index = _int(locator.get("nav_index"), rank - 1)
        if nav_index < 0:
            nav_index = rank - 1
        volume = max(1, _int(source.get("volume"), 1))
        chapter = max(1, _int(source.get("chapter"), rank))
        record: dict[str, Any] = {
            "unit_id": unit_id,
            "book_id": book_id,
            "volume": volume,
            "chapter": chapter,
            "title": str(source.get("title") or f"第{chapter}章"),
            "order": rank,
            "prev": None,
            "next": None,
            "text_path": _box_text_rel(source, rank),
            "chars": len(text),
            "sha256": actual_sha,
            "narrative_mode": narrative_mode,
            "narrator_reliability": reliability,
            "source_locator": {"href": href, "nav_index": nav_index},
            "scenes": [str(scene) for scene in source.get("scenes") or ()],
            "created_at": _iso(source.get("created_at"), created_at),
        }
        if normalized_narrator is not None:
            record["narrator"] = normalized_narrator
        records_out.append(record)

    for index, record in enumerate(records_out):
        record["prev"] = records_out[index - 1]["unit_id"] if index else None
        record["next"] = (
            records_out[index + 1]["unit_id"] if index + 1 < len(records_out) else None
        )
    return records_out, unit_texts


# ---------------------------------------------------------------------------
# entity table
# ---------------------------------------------------------------------------


def _add_entity(
    table: dict[str, dict[str, Any]],
    record: dict[str, Any],
    *,
    unit_id: str,
) -> None:
    existing = table.get(record["entity_id"])
    if existing is None:
        table[record["entity_id"]] = record
        return
    for alias in record["aliases"]:
        if alias not in existing["aliases"]:
            existing["aliases"].append(alias)
    if record["type"] != "unknown" and existing["type"] == "unknown":
        existing["type"] = record["type"]
    if unit_id and unit_id != existing["first_unit"]:
        existing["last_unit"] = unit_id
    existing["confidence"] = max(existing["confidence"], record["confidence"])
    if existing.get("name_history") and record.get("name_history"):
        history = existing["name_history"]
        if record["name_history"][0] not in history:
            history.append(record["name_history"][0])


def _build_entity_table(
    *,
    book_id: str,
    outputs: list[tuple[str, Any, Path]],
    units: list[dict[str, Any]],
    unit_texts: Mapping[str, str],
    work_root: Path,
    run_meta: RunMeta,
    warnings: list[str],
    quarantine: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None], dict[str, list[str]]]:
    table: dict[str, dict[str, Any]] = {}
    descriptions: dict[str, list[str]] = {}
    for output_unit, output, _ in outputs:
        dump = output.model_dump() if hasattr(output, "model_dump") else dict(output)
        for row in dump.get("entities") or ():
            record, problems = records.normalize_entity(
                row,
                book_id=book_id,
                output_unit=output_unit,
                produced_by=run_meta.run_id,
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(
                        f"entity:{output_unit}", "entity", problem, run_meta.run_id
                    )
                )
            if record is None:
                continue
            _add_entity(table, record, unit_id=output_unit)
            description = row.get("description") if isinstance(row, Mapping) else None
            if description:
                descriptions.setdefault(record["entity_id"], []).append(str(description))

    # Optional work-side identity candidates (the orchestrator's resolve shell).
    candidates_path = work_root / "resolve" / "entity_candidates.jsonl"
    if candidates_path.exists():
        valid_units = {unit["unit_id"] for unit in units}
        for candidate in atomic.read_jsonl(candidates_path):
            name = str(candidate.get("canonical_name") or "").strip()
            if not name:
                continue
            type_slug = records.entity_type_slug(candidate.get("type"))
            existing_id = None
            for entity_id, record in table.items():
                names = [record["canonical_name"], *record["aliases"]]
                if name in names:
                    existing_id = entity_id
                    break
            source_units = [
                str(unit) for unit in candidate.get("units") or () if str(unit) in valid_units
            ]
            if not source_units:
                continue
            if existing_id is not None:
                existing = table[existing_id]
                existing["first_unit"] = min(
                    [existing["first_unit"], *source_units],
                    key=lambda uid: next(
                        (u["order"] for u in units if u["unit_id"] == uid), 10**9
                    ),
                )
                existing["last_unit"] = max(
                    [existing["last_unit"], *source_units],
                    key=lambda uid: next(
                        (u["order"] for u in units if u["unit_id"] == uid), -1
                    ),
                )
                continue
            entity_id = records.make_entity_id(book_id, type_slug, name)
            table[entity_id] = {
                "entity_id": entity_id,
                "book_id": book_id,
                "type": type_slug,
                "canonical_name": name,
                "aliases": [str(alias) for alias in candidate.get("aliases") or ()],
                "tier": "minor",
                "name_history": [
                    {"name": name, "from_unit": source_units[0], "reason": "first_seen"}
                ],
                "first_unit": source_units[0],
                "last_unit": source_units[-1],
                "status": "active",
                "merged_into": None,
                "confidence": min(1.0, max(0.0, float(candidate.get("confidence", 0.0) or 0.0))),
                "produced_by": run_meta.run_id,
            }

    # Derive first/last units from resolved mentions later; for now reject
    # entities whose declared units are not in the box.
    valid_units = {unit["unit_id"] for unit in units}
    for entity_id in list(table):
        entity = table[entity_id]
        if entity["first_unit"] not in valid_units or entity["last_unit"] not in valid_units:
            quarantine.append(
                _quarantine_row(
                    entity_id,
                    "entity",
                    Problem("ref_missing", "entity first/last unit is not in the box"),
                    run_meta.run_id,
                )
            )
            del table[entity_id]

    return table, _name_index_from_entities(table), descriptions


def _name_index_from_entities(
    table: Mapping[str, Mapping[str, Any]],
) -> dict[str, str | None]:
    """Canonical names + aliases -> entity id; ambiguous keys become ``None``."""
    index: dict[str, str | None] = {}
    for entity_id, entity in table.items():
        for name in [str(entity.get("canonical_name") or ""), *(entity.get("aliases") or ())]:
            key = records.normalize_surface(name)
            if not key:
                continue
            if key in index and index[key] != entity_id:
                index[key] = None  # ambiguous alias: do not auto-link
            else:
                index[key] = entity_id
    return index


def _entities_from_mentions(
    book_id: str,
    mentions: list[dict[str, Any]],
    run_meta: RunMeta,
) -> dict[str, dict[str, Any]]:
    """Conservative identity fallback when resolve produced no candidates.

    This is the explicit bridge requested by the lead for mock engines: mentions
    with no linked entity become ``type=unknown``/``tier=minor`` entities using
    the same content-addressed id formula as the box. No aliases are inferred and
    no merge is performed; the caller records a warning.
    """
    table: dict[str, dict[str, Any]] = {}
    for mention in mentions:
        surface = str(mention.get("surface") or "").strip()
        if not surface:
            continue
        unit_id = str(mention.get("unit_id") or "")
        entity_id = records.make_entity_id(book_id, "unknown", surface)
        existing = table.get(entity_id)
        if existing is None:
            table[entity_id] = {
                "entity_id": entity_id,
                "book_id": book_id,
                "type": "unknown",
                "canonical_name": surface,
                "aliases": [],
                "tier": "minor",
                "name_history": [
                    {"name": surface, "from_unit": unit_id, "reason": "mention_fallback"}
                ],
                "first_unit": unit_id,
                "last_unit": unit_id,
                "status": "active",
                "merged_into": None,
                "confidence": float(mention.get("confidence") or 0.0),
                "produced_by": run_meta.run_id,
            }
            continue
        existing["confidence"] = max(
            float(existing["confidence"]), float(mention.get("confidence") or 0.0)
        )
        if unit_id and unit_id != existing["last_unit"]:
            existing["last_unit"] = unit_id
    return table


# ---------------------------------------------------------------------------
# merges
# ---------------------------------------------------------------------------


def _apply_merges(
    merges: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    mentions: list[dict[str, Any]],
    assertions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    cues: list[dict[str, Any]],
) -> None:
    target: dict[str, str] = {}
    for merge in merges:
        for source in merge["from"]:
            target[source] = merge["to"]

    def resolve(entity_id: str) -> str:
        seen: set[str] = set()
        current = entity_id
        while current in target and current not in seen:
            seen.add(current)
            current = target[current]
        return current

    for entity_id, entity in entities.items():
        resolved = resolve(entity_id)
        if resolved != entity_id:
            entity["status"] = "merged"
            entity["merged_into"] = resolved
    for mention in mentions:
        if mention.get("entity_id"):
            mention["entity_id"] = resolve(mention["entity_id"])
    for assertion in assertions:
        roles = assertion.get("roles") or {}

        def walk(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("e:"):
                return resolve(value)
            if isinstance(value, Mapping):
                return {key: walk(inner) for key, inner in value.items()}
            if isinstance(value, list):
                return [walk(inner) for inner in value]
            return value

        assertion["roles"] = walk(roles)
        assertion["deps"] = sorted(
            {
                resolve(dep) if isinstance(dep, str) and dep.startswith("e:") else dep
                for dep in assertion.get("deps") or ()
            }
        )
        speaker = assertion.get("speaker")
        if isinstance(speaker, Mapping) and speaker.get("entity_id"):
            speaker = dict(speaker)
            speaker["entity_id"] = resolve(speaker["entity_id"])
            assertion["speaker"] = speaker
    for event in events:
        event["participants"] = sorted(
            {resolve(pid) for pid in event.get("participants") or ()}
        )
        if event.get("location"):
            event["location"] = resolve(event["location"])
    for cue in cues:
        subject = cue.get("subject")
        if isinstance(subject, Mapping) and subject.get("entity_id"):
            subject = dict(subject)
            subject["entity_id"] = resolve(subject["entity_id"])
            cue["subject"] = subject


# ---------------------------------------------------------------------------
# per-unit collection
# ---------------------------------------------------------------------------


def _load_engine_outputs(
    work_root: Path,
    units: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    warnings: list[str],
) -> list[tuple[str, Any, Path]]:
    from cbb_stage1.models import EngineOutput

    engine_dir = work_root / "engine"
    if not engine_dir.is_dir():
        warnings.append(f"engine directory missing: {engine_dir}")
        return []
    unit_ids = {unit["unit_id"] for unit in units}
    outputs: list[tuple[str, Any, Path]] = []
    for path in sorted(engine_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            quarantine.append(
                _quarantine_row(
                    f"engine:{path.name}",
                    "engine_output",
                    Problem("schema", f"invalid engine JSON: {exc}"),
                    f"engine:{path.name}",
                )
            )
            continue
        unit_id = str(raw.get("unit_id") or "")
        if unit_id not in unit_ids:
            quarantine.append(
                _quarantine_row(
                    f"engine:{path.name}",
                    "engine_output",
                    Problem("ref_missing", f"engine unit not in box: {unit_id}"),
                    f"engine:{path.name}",
                )
            )
            continue
        try:
            output = EngineOutput.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError et al.
            quarantine.append(
                _quarantine_row(
                    f"engine:{path.name}",
                    "engine_output",
                    Problem("schema", f"engine output does not match contract: {exc}"),
                    f"engine:{path.name}",
                )
            )
            continue
        outputs.append((unit_id, output, path))
    outputs.sort(key=lambda item: item[0])
    return outputs


def _collect_mentions(
    outputs: list[tuple[str, Any, Path]],
    unit_texts: Mapping[str, str],
    entities: Mapping[str, dict[str, Any]],
    name_index: Mapping[str, str | None],
    run_meta: RunMeta,
    quarantine: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit_id, output, _ in outputs:
        dump = output.model_dump() if hasattr(output, "model_dump") else dict(output)
        for row in dump.get("mentions") or ():
            record, problems = records.normalize_mention(
                row,
                unit_texts[unit_id],
                unit_id,
                produced_by=run_meta.run_id,
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(
                        f"mention:{unit_id}", "mention", problem, run_meta.run_id
                    )
                )
            if record is None:
                continue
            explicit = record.get("entity_id")
            if explicit and explicit in entities:
                record["entity_id"] = explicit
            else:
                linked = name_index.get(records.normalize_surface(record["surface"]))
                record["entity_id"] = linked if linked in entities else None
            if record["mention_id"] in seen:
                continue
            seen.add(record["mention_id"])
            mentions.append(record)
    return mentions


def _assertion_content_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    """Semantic identity used to avoid duplicating work rows with fallbacks."""
    evidence = record.get("evidence") or []
    primary = ""
    if evidence and isinstance(evidence[0], Mapping):
        primary = str(evidence[0].get("quote_sha256") or "")
    return (
        str(record.get("predicate") or ""),
        json.dumps(record.get("roles") or {}, ensure_ascii=False, sort_keys=True),
        primary,
    )


def _load_work_assertions(
    work_root: Path,
    *,
    unit_texts: Mapping[str, str],
    lexicon: Lexicon,
    entities: Mapping[str, Mapping[str, Any]],
    name_index: Mapping[str, str | None],
    run_meta: RunMeta,
    book_id: str,
    quarantine: list[dict[str, Any]],
    default_unit: str,
) -> list[dict[str, Any]] | None:
    """Read assemble's authoritative ``work/assemble/assertions.jsonl``.

    Returns ``None`` when the table does not exist (pre-postprocess work dirs),
    or the normalised rows when it does. Claim records reference these
    assertion ids, so the writer must use them instead of re-deriving ids from
    engine candidates.
    """
    path = work_root / "assemble" / "assertions.jsonl"
    if not path.exists():
        return None
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    usable_index = {key: value for key, value in name_index.items() if value}
    for index, row in enumerate(atomic.read_jsonl(path), 1):
        evidence = row.get("evidence") or []
        output_unit = default_unit
        if evidence and isinstance(evidence[0], Mapping) and evidence[0].get("unit_id"):
            output_unit = str(evidence[0]["unit_id"])
        record, problems, proposal = records.normalize_assertion(
            row,
            output_unit=output_unit,
            unit_texts=unit_texts,
            lexicon=lexicon,
            entities_by_id=entities,
            name_index=usable_index,
            run_meta=run_meta,
            book_id=book_id,
        )
        for problem in problems:
            quarantine.append(
                _quarantine_row(
                    f"work-assertion:{row.get('assertion_id') or index}",
                    "assertion",
                    problem,
                    run_meta.run_id,
                )
            )
        if proposal is not None:
            quarantine.append(
                _quarantine_row(
                    f"work-assertion:{row.get('assertion_id') or index}",
                    "assertion",
                    Problem("schema", f"predicate {proposal['predicate']!r} not in writer lexicon"),
                    run_meta.run_id,
                )
            )
            continue
        if record is None or record["assertion_id"] in seen:
            continue
        seen.add(record["assertion_id"])
        accepted.append(record)
    return accepted


def _load_work_events(
    work_root: Path,
    *,
    unit_texts: Mapping[str, str],
    entities: Mapping[str, Mapping[str, Any]],
    name_index: Mapping[str, str | None],
    run_meta: RunMeta,
    book_id: str,
    quarantine: list[dict[str, Any]],
    default_unit: str,
) -> list[dict[str, Any]]:
    """Read assemble's authoritative ``work/assemble/events.jsonl``."""
    path = work_root / "assemble" / "events.jsonl"
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    usable_index = {key: value for key, value in name_index.items() if value}
    for index, row in enumerate(atomic.read_jsonl(path), 1):
        output_unit = str(row.get("unit_id") or default_unit)
        evidence = row.get("evidence") or []
        if (
            output_unit == default_unit
            and evidence
            and isinstance(evidence[0], Mapping)
            and evidence[0].get("unit_id")
        ):
            output_unit = str(evidence[0]["unit_id"])
        record, problems = records.normalize_event(
            row,
            output_unit=output_unit,
            unit_texts=unit_texts,
            entities_by_id=entities,
            name_index=usable_index,
            run_meta=run_meta,
            book_id=book_id,
        )
        for problem in problems:
            quarantine.append(
                _quarantine_row(
                    f"work-event:{row.get('event_id') or index}",
                    "event",
                    problem,
                    run_meta.run_id,
                )
            )
        if record is None or record["event_id"] in seen:
            continue
        seen.add(record["event_id"])
        accepted.append(record)
    return accepted


def _resolve_payoff_references(
    cues: list[dict[str, Any]],
    events: list[dict[str, Any]],
    unit_ids: set[str],
    quarantine: list[dict[str, Any]],
    run_id: str,
) -> None:
    event_ids = {event["event_id"] for event in events}
    for cue in cues:
        payoff = cue.get("payoff") or {}
        if payoff.get("event_id") and payoff["event_id"] not in event_ids:
            quarantine.append(
                _quarantine_row(
                    cue["cue_id"],
                    "cue",
                    Problem("ref_missing", f"payoff event missing: {payoff['event_id']}"),
                    run_id,
                )
            )
            payoff["event_id"] = None
        if payoff.get("unit_id") and payoff["unit_id"] not in unit_ids:
            quarantine.append(
                _quarantine_row(
                    cue["cue_id"],
                    "cue",
                    Problem("ref_missing", f"payoff unit missing: {payoff['unit_id']}"),
                    run_id,
                )
            )
            payoff["unit_id"] = None
        trigger = cue.get("trigger")
        if isinstance(trigger, Mapping):
            # Copy before mutating: ``trigger`` may be a read-only view supplied by
            # the caller, and writing through a Mapping was a latent runtime error
            # (mypy flagged both assignments).
            trigger = dict(trigger)
            if trigger.get("event_id") and trigger["event_id"] not in event_ids:
                trigger["event_id"] = None
            if trigger.get("unit_id") and trigger["unit_id"] not in unit_ids:
                trigger["unit_id"] = None
            cue["trigger"] = trigger


def _timeline_cycle_nodes(timeline: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return node ids participating in a ``before``/``after`` cycle."""
    edges: dict[str, set[str]] = {}
    for row in timeline:
        if row["kind"] in ("before", "after"):
            left, right = row["left"]["id"], row["right"]["id"]
            if row["kind"] == "after":
                left, right = right, left
            edges.setdefault(left, set()).add(right)

    color: dict[str, int] = {}
    cycle_nodes: set[str] = set()

    def dfs(node: str, stack: list[str]) -> None:
        color[node] = 1
        for child in edges.get(node, ()):
            if color.get(child) == 1:
                cycle_nodes.add(child)
                cycle_nodes.update(stack)
            elif color.get(child, 0) == 0:
                dfs(child, stack + [node])
        color[node] = 2

    for node in list(edges):
        if color.get(node, 0) == 0:
            dfs(node, [])
    return tuple(sorted(cycle_nodes))


def _assert_timeline_acyclic(timeline: list[dict[str, Any]]) -> None:
    """Reject a cyclic work timeline; never silently repair/drop it.

    Cycle resolution belongs to ``temporal`` (the work producer). If a cycle
    reaches the box, the honest behaviour is a failed build naming the nodes,
    not a silently pruned table.
    """
    cycle_nodes = _timeline_cycle_nodes(timeline)
    if cycle_nodes:
        raise BuildError(
            "work/assemble/timeline.jsonl contains a cycle; "
            "fix the temporal integration instead of letting the box drop constraints",
            [f"cycle nodes: {', '.join(cycle_nodes)}"],
        )


# ---------------------------------------------------------------------------
# work-table loaders (postprocess -> box; no parallel producers)
# ---------------------------------------------------------------------------


def _load_work_entity_cards(work_root: Path) -> list[dict[str, Any]]:
    """Read dedupe's ``work/assemble/entity_cards.jsonl`` (empty when absent)."""
    path = work_root / "assemble" / "entity_cards.jsonl"
    if not path.exists():
        return []
    rows = atomic.read_jsonl(path)
    cards: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not row.get("canonical_name"):
            raise BuildError(
                f"{path}:{index}: entity card without canonical_name",
            )
        cards.append(dict(row))
    return cards


def _load_work_summaries(
    work_root: Path,
    units: list[dict[str, Any]],
    *,
    book_id: str,
    volume_titles: Mapping[int, str] | None,
    changed_units: Iterable[str],
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read derive's summary tree, or fall back to the explicit skeleton tree."""
    path = work_root / "derive" / "summaries.jsonl"
    if path.exists():
        rows = atomic.read_jsonl(path)
        if not rows:
            raise BuildError(
                f"{path} exists but is empty; the derive integration must write the tree or omit the file"
            )
        root = next(
            (
                row
                for row in rows
                if row.get("scope") == "book" and not row.get("parent_summary_id")
            ),
            None,
        )
        if root is None:
            raise BuildError(f"{path}: no book-level root summary")
        tree = {
            "contract": SUMMARY_CONTRACT,
            "book_id": book_id,
            "root": root.get("summary_id"),
            "nodes": rows,
        }
        errors = validate_summary_tree(tree)
        if errors:
            raise BuildError("work summary tree is not closed", errors)
    else:
        # Explicit skeleton: unit-derived tree with no synthesised text.
        tree = build_summary_tree(
            units,
            book_id=book_id,
            volume_titles=volume_titles,
            produced_by="box:derive",
        )
        tree["created_at"] = created_at
    if list(changed_units):
        tree = mark_stale(tree, changed_units)
    return tree, summary_rows(tree)


def _load_work_cues(
    path: Path,
    *,
    unit_texts: Mapping[str, str],
    entities: Mapping[str, Mapping[str, Any]],
    run_meta: RunMeta,
    book_id: str,
    default_unit: str,
    quarantine: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise the cues integration's ``work/assemble/cues.jsonl`` rows."""
    records_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in atomic.read_jsonl(path):
        setup = row.get("setup")
        setup_unit = (
            str(setup.get("unit_id"))
            if isinstance(setup, Mapping) and setup.get("unit_id")
            else str(row.get("unit_id") or default_unit)
        )
        record, problems = records.normalize_cue(
            row,
            output_unit=setup_unit,
            unit_texts=unit_texts,
            entities_by_id=entities,
            run_meta=run_meta,
            book_id=book_id,
        )
        for problem in problems:
            quarantine.append(
                _quarantine_row(f"cue:{path.name}", "cue", problem, run_meta.run_id)
            )
        if record is None or record["cue_id"] in seen:
            continue
        seen.add(record["cue_id"])
        records_out.append(record)
    return records_out


def _load_work_conflicts(
    work_root: Path,
    *,
    book_id: str,
    assertion_ids: set[str],
    created_at: str,
) -> list[dict[str, Any]]:
    """Read validate's ``work/audit/conflicts.jsonl``; never detect conflicts here."""
    path = work_root / "audit" / "conflicts.jsonl"
    if not path.exists():
        return []
    kinds = ("attribute", "relation", "timeline")
    decisions = ("left", "right", "merge", "disputed", "superseded")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(atomic.read_jsonl(path), 1):
        left = str(row.get("left") or "")
        right = str(row.get("right") or "")
        if left not in assertion_ids or right not in assertion_ids:
            raise BuildError(
                f"{path}:{index}: conflict references a missing assertion",
                [f"left={left!r} right={right!r}"],
            )
        kind = str(row.get("kind") or "")
        if kind not in kinds:
            raise BuildError(f"{path}:{index}: bad conflict kind {kind!r}")
        status = str(row.get("status") or "open")
        if status not in ("open", "resolved"):
            raise BuildError(f"{path}:{index}: bad conflict status {status!r}")
        detail = str(row.get("detail") or "").strip()
        if not detail:
            raise BuildError(f"{path}:{index}: conflict without detail")
        decision = row.get("decision")
        if decision is not None and decision not in decisions:
            raise BuildError(f"{path}:{index}: bad conflict decision {decision!r}")
        conflict_id = str(row.get("conflict_id") or "")
        if not conflict_id:
            conflict_id = _ids_conflict_id(book_id, kind, left, right, detail)
        rows.append(
            {
                "conflict_id": conflict_id,
                "book_id": str(row.get("book_id") or book_id),
                "kind": kind,
                "left": left,
                "right": right,
                "detail": detail,
                "status": status,
                "decision": decision,
                "resolved_by": row.get("resolved_by"),
                "at": row.get("at") or (created_at if status == "resolved" else None),
            }
        )
    return sorted(rows, key=lambda item: item["conflict_id"])


def _load_passthrough_derived(
    work_root: Path, relative: str
) -> list[dict[str, Any]] | None:
    """Read an optional derived work table verbatim (``None`` when absent)."""
    path = work_root / relative
    if not path.exists():
        return None
    rows = atomic.read_jsonl(path)
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise BuildError(f"{path}:{index}: derived row must be an object")
    return [dict(row) for row in rows]



# ---------------------------------------------------------------------------
# hard-gate validation
# ---------------------------------------------------------------------------


def _collect_box_hashes(root: Path) -> tuple[dict[str, str], str]:
    """Hash every box file except the manifest/CHECKSUMS/SPEC bootstrap trio."""
    files: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in ("manifest.json", "CHECKSUMS.json", "SPEC.md"):
            continue
        files[relative] = atomic.sha256_bytes(path.read_bytes())
    return files, atomic.merkle_root(files)


def _write_manifest_and_checksums(
    root: Path, manifest: dict[str, Any]
) -> dict[str, str]:
    """Recompute hashes over the real bytes and write manifest + CHECKSUMS."""
    files, merkle = _collect_box_hashes(root)
    manifest["hashes"] = {"files": files, "merkle_root": merkle}
    atomic.write_json(root / "CHECKSUMS.json", {"files": files, "merkle_root": merkle})
    atomic.write_json(root / "manifest.json", manifest)
    return files


def _soft_gate_report(work_root: Path | None = None) -> dict[str, Any]:
    """Soft gates are report fields, not build measurements.

    The previous values (``unresolved_rate: 0.0`` / ``cost_usd: 0.0``) were
    literals — a fake green. They are now ``null`` with an explicit "not
    measured" note; no soft gate is claimed without a measurement behind it.
    """
    decoupled, decoupled_note = _measure_model_decoupling(work_root)
    return {
        "relation_precision": None,
        "merge_precision": None,
        "event_recall": None,
        "unresolved_rate": None,
        "cost_usd": None,
        "model_decoupled": decoupled,
        "notes": {
            "relation_precision": "not measured：构建期没有关系金标集",
            "merge_precision": "not measured：构建期没有合并金标集",
            "event_recall": "not measured：构建期没有事件金标集",
            "unresolved_rate": "not measured：未接入 unresolved 率的规范测量",
            "cost_usd": "not measured：audit/runs.jsonl 没有引擎 token 计费",
            "model_decoupled": decoupled_note,
        },
    }


def _measure_model_decoupling(
    work_root: Path | None,
) -> tuple[bool | None, str]:
    """D4: is the verifier model different from the extractor model?

    `01-箱规格` §5.15 / D4 require the verifying model to be a different tier
    from the extracting one, otherwise the verifier is just re-reading its own
    output. This used to be reported as `not measured` even though
    `plan/model-calls.jsonl` records the model of every call, including the
    verifier's — so the fact was available and simply not read.

    Returns `(value, note)`:
      * `True`  — verifier and extractor ran on different models;
      * `False` — they ran on the same model (the QC debt the spec names);
      * `None`  — no verifier calls, so decoupling cannot be assessed.
    """
    if work_root is None:
        return None, "not measured：没有 work 目录"
    path = Path(work_root) / "plan" / "model-calls.jsonl"
    if not path.exists():
        return None, "not measured：plan/model-calls.jsonl 不存在"
    models: dict[str, set[str]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            role = str(row.get("role") or "")
            model = str(row.get("model") or "")
            if role and model:
                models.setdefault(role, set()).add(model)
    except (OSError, json.JSONDecodeError):
        return None, "not measured：model-calls.jsonl 不可读"

    extractor = models.get("extraction") or models.get("assistant") or set()
    verifier = models.get("verifier") or set()
    judge = models.get("judge") or set()

    if not verifier:
        return None, "not measured：本轮没有 verifier 调用"
    if not extractor:
        return None, "not measured：本轮没有抽取调用"

    # A dedicated judge counts as decoupling even when the inline verifier shares
    # the extractor's model. §5.15 asks that the *verification* of a claim not be
    # performed by the same model that produced it; a separate judge model
    # satisfies that, and ignoring it made the field permanently false under the
    # agent-driven transport, where verifier and extractor are always the same
    # agent by construction.
    if judge and not (judge & extractor):
        return True, (
            "judge=%s 与 extractor=%s 异档（inline verifier 共用 %s，但 §5.15 的"
            "独立性由独立判卷模型满足）"
            % (
                ", ".join(sorted(judge)),
                ", ".join(sorted(extractor)),
                ", ".join(sorted(verifier & extractor)) or "none",
            )
        )

    if verifier & extractor:
        shared = ", ".join(sorted(verifier & extractor))
        hint = (
            "；已配置 judge=%s 但抽取模型也用它，未构成异档" % ", ".join(sorted(judge))
            if judge
            else "；配置一个独立的 judge 模型即可满足"
        )
        return False, (
            "verifier 与 extractor 同档（共用 %s）—— §5.15 要求验证模型与抽取模型异档；"
            "本轮属 QC 债务，记 false%s" % (shared, hint)
        )

    return True, "verifier=%s / extractor=%s，异档" % (
        ", ".join(sorted(verifier)),
        ", ".join(sorted(extractor)),
    )


def _work_table_problems(work_root: Path | None) -> list[str]:
    """GAP work tables must exist and be non-empty before a box is built.

    ``qc.tables.check_work_tables`` existed with exactly this rule and nothing
    called it, so ``build_box`` would happily assemble a box from a work directory
    whose assertion table had been emptied — producing a structurally valid box
    that describes nothing.
    """
    if work_root is None:
        return []
    try:
        from cbb_stage1.qc.tables import check_work_tables
    except ImportError:  # pragma: no cover - contract file is shipped
        return []
    try:
        return list(check_work_tables(Path(work_root)))
    except Exception as exc:  # noqa: BLE001 - a broken contract must not block a box
        return [f"check_work_tables failed: {type(exc).__name__}: {exc}"]


def _capability_gaps(
    work_root: Path | None, soft_gates: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Which mechanisms degraded this run because an input format was missing.

    The audit needed this list assembled by hand: "B3 unavailable" and "B5
    external" were only visible by reading individual receipts. They are all
    derivable from two files — ``plan/model-calls.jsonl`` (what was called) and
    the soft gates (what could not be measured) — so they are derived here rather
    than restated.

    A gap is reported with the reason, not just a flag: "no logprobs" and "no
    judge model" are different problems with different fixes.
    """
    gaps: dict[str, dict[str, Any]] = {}

    calls: list[dict[str, Any]] = []
    if work_root is not None:
        path = Path(work_root) / "plan" / "model-calls.jsonl"
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, Mapping):
                        calls.append(row)
            except (OSError, json.JSONDecodeError):
                calls = []

    methods = {
        str(row.get("method") or "")
        for row in calls
        if row.get("method")
    }
    kinds = {str(row.get("kind") or "") for row in calls}
    roles = {str(row.get("role") or "") for row in calls}

    # B3 / B4-whitebox need token-level signals; the host transport has none.
    if "uqlm_whitebox" not in methods and "sep" not in methods:
        gaps["b3_sep"] = {
            "status": "unavailable",
            "reason": "no token logprobs or hidden states: the host transport returns text only",
        }
        gaps["b4_uqlm_whitebox"] = {
            "status": "unavailable",
            "reason": "same white-box signal missing",
        }

    # B4 judge needs a second model.
    if "claim_qa" not in methods and "judge" not in roles:
        gaps["b4_uqlm_judge"] = {
            "status": "unavailable",
            "reason": "no judge model configured (a second model is required; the main one is not reused)",
        }

    # B5 needs a trained checkpoint.
    if "uq_heads" not in methods:
        gaps["b5_uq_heads"] = {
            "status": "external",
            "reason": "no cbb-uq-head/1 checkpoint; confidence fell back to a sampling method",
        }

    # D4: decoupling is only satisfiable with a separate verifier model.
    decoupled = soft_gates.get("model_decoupled")
    if decoupled is False:
        gaps["d4_decoupled"] = {
            "status": "false",
            "reason": "verifier and extractor share one model; §5.15 wants different tiers",
        }
    elif decoupled is None:
        gaps["d4_decoupled"] = {
            "status": "not measured",
            "reason": "no verifier calls in this run",
        }

    # D11: NLI verdicts produced by the main model are a fallback, not an NLI model.
    if "nli" not in kinds and "nli" not in roles:
        gaps["d11_nli"] = {
            "status": "unavailable",
            "reason": "no temporal constraints to verify, or no NLI backend configured",
        }

    # Mechanisms that are NOT implemented at all, as opposed to "implemented but
    # starved of an input". The distinction matters: the entries above can be
    # fixed by configuring a backend, these cannot be fixed by configuration.
    gaps.update(_not_implemented_gaps())
    return gaps


#: Mechanisms the inventory records as "未实现" (no code at all).
#:
#: Kept as a literal on purpose. Deriving this from source scanning produced false
#: positives (a variable named ``ending`` matched "100-Endings"), so the list is
#: maintained by hand and reviewed against ``07-吸收机制总清单-v2``. Every entry is
#: something a reader can check in that document.
NOT_IMPLEMENTED: dict[str, str] = {
    "b1_cove": (
        "CoVe four-step loop (draft -> verification questions -> independent "
        "answers -> revision) is not implemented; only the independent-context "
        "half exists"
    ),
    # d15_motive used to read "motive chain / emotion curve / theme / style are not
    # implemented; derived/inference holds placeholders only". That stopped being true
    # when the four tables landed, and a stale gap entry is worse than no entry: it
    # sends the next reader looking for work that is already done. The key is kept (a
    # reader may reference it) but now describes only what remains.
    "d15_inference_partial": (
        "derived/inference now carries stylometry, themes, emotion curves (lexical "
        "polarity, NOT UED — that paper is not in the resource library) and motivation "
        "chains (Maslow closed set, opt-in). Still missing: the published forecast "
        "method (E07 generates 100 continuations per sentence, ~180k calls), author "
        "identification from stylometry, and any fan-theory support."
    ),
}


def _not_implemented_gaps() -> dict[str, dict[str, Any]]:
    """Report mechanisms that have no implementation behind them at all."""
    return {
        name: {"status": "not_implemented", "reason": reason}
        for name, reason in NOT_IMPLEMENTED.items()
    }


def _qc_report_payload(
    *,
    box_id: str,
    hard_gates: Mapping[str, str],
    soft_gates: Mapping[str, Any],
    assertions: Sequence[Mapping[str, Any]],
    merges: Sequence[Mapping[str, Any]],
    generated_at: str,
    hard_gate_errors: Mapping[str, Sequence[str]] | None = None,
    capability_gaps: Mapping[str, Mapping[str, Any]] | None = None,
    work_table_problems: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "box_id": box_id,
        "capability_gaps": {
            name: dict(entry) for name, entry in (capability_gaps or {}).items()
        },
        "work_table_problems": list(work_table_problems or ()),
        "hard_gates": dict(hard_gates),
        "hard_gate_errors": {
            gate: list(messages)
            for gate, messages in (hard_gate_errors or {}).items()
            if messages
        },
        "soft_gates": dict(soft_gates),
        "samples": {
            "assertions": len(assertions),
            "merges": len(merges),
            "low_confidence": sum(
                1 for row in assertions if row["confidence"] < 0.5
            ),
        },
        "human_reviewed_units": 0,
        "status": "pass" if all(status == "pass" for status in hard_gates.values()) else "fail",
        "generated_at": generated_at,
    }


def _validate_schema_rows(
    tables: Mapping[str, list[dict[str, Any]]],
    manifest: Mapping[str, Any],
    engine_contracts: Mapping[str, dict[str, Any]],
) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise BuildError("jsonschema is required to gate the box") from exc
    schemas_dir = _schemas_dir()
    if not schemas_dir.is_dir():
        raise BuildError(f"box schemas not found: {schemas_dir}")

    def load(name: str) -> Any:
        return json.loads((schemas_dir / name).read_text(encoding="utf-8"))

    errors: list[str] = []
    validator = Draft202012Validator(load("box.schema.json"))
    for error in sorted(validator.iter_errors(manifest), key=lambda e: list(e.path)):
        errors.append(f"manifest: {'/'.join(map(str, error.path))}: {error.message}")
    for relative, schema_name in TABLE_SCHEMAS.items():
        rows = tables.get(relative)
        if rows is None:
            continue
        row_validator = Draft202012Validator(load(schema_name))
        for number, row in enumerate(rows, 1):
            for error in sorted(
                row_validator.iter_errors(row), key=lambda e: list(e.path)
            ):
                errors.append(
                    f"{relative}:{number}: {'/'.join(map(str, error.path))}: {error.message}"
                )
    engine_validator = Draft202012Validator(load("engine-output.schema.json"))
    for name, contract in engine_contracts.items():
        for error in sorted(
            engine_validator.iter_errors(contract), key=lambda e: list(e.path)
        ):
            errors.append(f"audit/engine/{name}: {'/'.join(map(str, error.path))}: {error.message}")
    return errors


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _load_work_media(work_root: Path) -> list[dict[str, Any]]:
    """Read the pipeline's real media extraction table (EPUB images/audio/video)."""
    path = work_root / "plan" / "media.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in atomic.read_jsonl(path):
        if isinstance(row, dict) and row.get("asset_id") and row.get("path"):
            rows.append(dict(row))
    return rows


def _merge_media_assets(
    references: list[dict[str, Any]],
    media_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge real extracted assets over text-reference rows, schema-clean."""
    allowed = {
        "asset_id",
        "book_id",
        "path",
        "sha256",
        "media_type",
        "source_unit_id",
        "alt",
    }
    merged = {str(row["asset_id"]): dict(row) for row in references}
    for row in media_rows:
        merged[str(row["asset_id"])] = {
            key: row.get(key) for key in allowed if key in row
        }
    return [merged[key] for key in sorted(merged)]


def _predicate_reconciliation(
    work_root: Path, assertions: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    """Emitted (assemble stage) vs accepted (this box) per predicate.

    Why this exists: a box keeping 1 of 9 assertions has a non-empty assertions
    table and a low absent ratio, so the table-level invariant in `box_lint`
    passes it. That is the "partial hollowing" shape the external audit flagged
    (B-P2-3). Counting per predicate gives the gate an exact number to compare.

    `emitted` comes from the assemble stage's own record; when the work tree has
    no assemble stage the block is empty rather than guessed.
    """
    counts: dict[str, dict[str, int]] = {}
    stage_path = Path(work_root) / "assemble" / "assertions.jsonl"
    emitted: dict[str, int] = {}
    if stage_path.exists():
        try:
            for line in stage_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                predicate = str(row.get("predicate") or "")
                if predicate:
                    emitted[predicate] = emitted.get(predicate, 0) + 1
        except (OSError, json.JSONDecodeError):
            emitted = {}
    accepted: dict[str, int] = {}
    for row in assertions:
        predicate = str(row.get("predicate") or "")
        if predicate:
            accepted[predicate] = accepted.get(predicate, 0) + 1
    for predicate in sorted(set(emitted) | set(accepted)):
        counts[predicate] = {
            "emitted": emitted.get(predicate, 0),
            "accepted": accepted.get(predicate, 0),
        }
    return counts


def build_box(
    work_root: str | Path,
    out: str | Path,
    *,
    book_title: str | None = None,
    box_id: str | None = None,
    created_at: str | None = None,
    parent: str | None = None,
    incremental: Mapping[str, Any] | None = None,
    lexicon: Lexicon | None = None,
    lexicon_dir: str | Path | None = None,
    volume_titles: Mapping[int, str] | None = None,
    changed_units: Iterable[str] = (),
    changed_entities: Iterable[str] = (),
    series_id: str | None = None,
    volume_no: int | None = None,
    verify: bool = True,
) -> BuildResult:
    """Build an immutable v2.2 box from the pipeline work layer.

    Args:
        work_root: ``work/`` directory produced by the 9-stage pipeline.
        out: Destination box directory; must not exist (boxes are immutable).
        book_title: Optional title override (default from ``work/plan/run.json``).
        box_id: Optional box id (default ``<book>-<created_at>``).
        created_at: Optional ISO timestamp; fixed timestamps make builds
            byte-reproducible in tests.
        parent: Parent box id for incremental boxes.
        incremental: Optional manifest ``incremental`` object; when omitted a
            minimal object with ``parent`` is written.
        lexicon: Pre-loaded lexicon (takes precedence over ``lexicon_dir``).
        lexicon_dir: Directory with genre/predicates/types JSON.
        volume_titles: Optional volume number -> title map.
        changed_units / changed_entities: Incremental seeds; they are closed
            over the work dependency graph and recorded in ``manifest.incremental``.
        series_id / volume_no: Multi-volume grouping keys (optional); projection
            uses ``book.series_id`` instead of treating each volume as a book.
        verify: Enforce the canonical ``tools/box_lint.py`` gate before
            publishing. The canonical lint always runs so the seven hard gates
            are measured; with ``verify=False`` a failing box is published with
            ``qc.hard_gates = "fail"`` instead of raising ``BuildError``.

    Raises:
        FileExistsError: ``out`` already exists.
        BuildError: work is missing/invalid, a hard gate fails, or lint fails.
    """
    work_root = Path(work_root)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing box: {out}")
    if not work_root.is_dir():
        raise BuildError(f"work root does not exist: {work_root}")

    # GAP work tables are checked and reported, not enforced here.
    #
    # The only GAP table is ``plan/wiring.json``, which the pipeline writes at the
    # end of a run; a box built from a hand-assembled work fixture legitimately has
    # none. Raising here turned 33 existing tests red for a reason unrelated to what
    # they test. The finding is surfaced in ``qc.report`` instead, where a reader
    # can see "this box was built from incomplete work" without the build refusing.
    work_table_problems = _work_table_problems(work_root)

    resolved_created = _iso(created_at, _now_iso()) if created_at else _now_iso()
    run = atomic.read_json(work_root / "plan" / "run.json", default={}) or {}
    book_id = _sanitize_book_id(run.get("book_id"), fallback="book")
    if book_id == "book":
        units_probe = atomic.read_jsonl(work_root / "plan" / "units.jsonl")
        if units_probe:
            book_id = _sanitize_book_id(units_probe[0].get("book_id"), fallback="book")
    resolved_title = (
        book_title
        or run.get("book_title")
        or run.get("book")
        or book_id
    )
    warnings: list[str] = []
    quarantine: list[dict[str, Any]] = []

    units, unit_texts = _load_units(work_root, book_id, resolved_created, warnings)
    unit_ids = [unit["unit_id"] for unit in units]

    if lexicon is None:
        if lexicon_dir is not None:
            lexicon = load_lexicon(lexicon_dir)
        else:
            work_lexicon = work_root / "lexicon"
            lexicon = load_lexicon(work_lexicon) if work_lexicon.is_dir() else None
    if lexicon is None:
        from cbb_stage1.box.lexicon import default_lexicon

        lexicon = default_lexicon()

    engine = _engine_info(run)
    trace_id = str(run.get("trace_id") or _ids_short_hash(str(work_root), 12))
    run_id = f"run:{trace_id}"
    started_at = _iso(run.get("started_at"), resolved_created)
    ended_at = _iso(run.get("ended_at"), resolved_created)
    run_meta = RunMeta(
        run_id=run_id,
        box_id="",  # filled after the box id is known
        engine=engine["name"],
        model=engine["model"],
        prompt_sha256=engine["prompt_sha256"],
        started_at=started_at,
        ended_at=ended_at,
    )
    resolved_box_id = box_id or f"{book_id}-{re.sub(r'[^0-9A-Za-z]', '', resolved_created)[:15]}"
    if len(resolved_box_id) < 3:
        resolved_box_id = f"{book_id}-box"
    run_meta = RunMeta(
        run_id=run_id,
        box_id=resolved_box_id,
        engine=engine["name"],
        model=engine["model"],
        prompt_sha256=engine["prompt_sha256"],
        started_at=started_at,
        ended_at=ended_at,
    )

    outputs = _load_engine_outputs(work_root, units, quarantine, warnings)
    entities, name_index, _descriptions = _build_entity_table(
        book_id=book_id,
        outputs=outputs,
        units=units,
        unit_texts=unit_texts,
        work_root=work_root,
        run_meta=run_meta,
        warnings=warnings,
        quarantine=quarantine,
    )
    mentions = _collect_mentions(
        outputs, unit_texts, entities, name_index, run_meta, quarantine
    )

    # Conservative identity bridge (lead-requested): if resolve wrote no
    # entity_candidates at all (e.g. a mock engine with only mentions), derive
    # type=unknown/minor entities from the mention surfaces instead of silently
    # producing an entity-less box.
    if not entities and mentions:
        entities = _entities_from_mentions(book_id, mentions, run_meta)
        name_index = _name_index_from_entities(entities)
        for mention in mentions:
            explicit = mention.get("entity_id")
            if explicit and explicit in entities:
                continue
            linked = name_index.get(records.normalize_surface(mention["surface"]))
            mention["entity_id"] = linked if linked in entities else None
        warnings.append(
            "resolve/entity_candidates.jsonl is empty or absent; generated "
            f"{len(entities)} conservative entity record(s) from mentions "
            "(type=unknown, tier=minor, no alias merge)"
        )

    # Recompute entity first/last units from linked mentions (identity evidence).
    entity_units: dict[str, list[str]] = {}
    for mention in mentions:
        if mention.get("entity_id") in entities:
            entity_units.setdefault(mention["entity_id"], []).append(mention["unit_id"])
    order_by_unit = {unit["unit_id"]: unit["order"] for unit in units}
    for entity_id, linked_units in entity_units.items():
        unique = sorted(set(linked_units), key=lambda uid: order_by_unit.get(uid, 10**9))
        entities[entity_id]["first_unit"] = unique[0]
        entities[entity_id]["last_unit"] = unique[-1]

    assertions: list[dict[str, Any]] = []
    engine_assertions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    engine_events: list[dict[str, Any]] = []
    engine_cues: list[dict[str, Any]] = []
    proposals: dict[str, dict[str, Any]] = {}
    seen_assertions: set[str] = set()
    seen_events: set[str] = set()
    seen_cues: set[str] = set()
    for unit_id, output, _ in outputs:
        dump = output.model_dump() if hasattr(output, "model_dump") else dict(output)
        for row in dump.get("assertions") or ():
            record, problems, proposal = records.normalize_assertion(
                row,
                output_unit=unit_id,
                unit_texts=unit_texts,
                lexicon=lexicon,
                entities_by_id=entities,
                name_index={k: v for k, v in name_index.items() if v},
                run_meta=run_meta,
                book_id=book_id,
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(
                        f"assertion:{unit_id}", "assertion", problem, run_meta.run_id
                    )
                )
            if proposal is not None:
                predicate = proposal["predicate"]
                existing = proposals.get(predicate)
                if existing is None:
                    proposal["first_seen"] = resolved_created
                    proposals[predicate] = proposal
                else:
                    existing["count"] = int(existing.get("count", 1)) + 1
                    if unit_id not in existing["seen_in"]:
                        existing["seen_in"].append(unit_id)
                    existing["role_shape_examples"].append(proposal["role_shape_examples"][0])
                continue
            if record is None or record["assertion_id"] in seen_assertions:
                continue
            seen_assertions.add(record["assertion_id"])
            engine_assertions.append(record)

        for row in dump.get("events") or ():
            record, problems = records.normalize_event(
                row,
                output_unit=unit_id,
                unit_texts=unit_texts,
                entities_by_id=entities,
                name_index={k: v for k, v in name_index.items() if v},
                run_meta=run_meta,
                book_id=book_id,
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(f"event:{unit_id}", "event", problem, run_meta.run_id)
                )
            if record is None or record["event_id"] in seen_events:
                continue
            seen_events.add(record["event_id"])
            engine_events.append(record)

        for row in dump.get("cues") or ():
            record, problems = records.normalize_cue(
                row,
                output_unit=unit_id,
                unit_texts=unit_texts,
                entities_by_id=entities,
                run_meta=run_meta,
                book_id=book_id,
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(f"cue:{unit_id}", "cue", problem, run_meta.run_id)
                )
            if record is None or record["cue_id"] in seen_cues:
                continue
            seen_cues.add(record["cue_id"])
            engine_cues.append(record)

    # Assemble's events table is authoritative when present (W3 produces it).
    # Engine events are a legacy fallback only, and never silent.
    work_events_path = work_root / "assemble" / "events.jsonl"
    if work_events_path.exists():
        events = _load_work_events(
            work_root,
            unit_texts=unit_texts,
            entities=entities,
            name_index=name_index,
            run_meta=run_meta,
            book_id=book_id,
            quarantine=quarantine,
            default_unit=unit_ids[0],
        )
    else:
        events = engine_events
        if events:
            warnings.append(
                "work/assemble/events.jsonl missing; using legacy engine-output "
                f"event fallback ({len(events)} event(s))"
            )

    # Assemble's work table is authoritative when present: claims reference its
    # assertion ids. Engine-derived records are only a fallback for rows the
    # work table does not cover (e.g. legacy work dirs, or a predicate that the
    # assemble seed lexicon proposed instead of accepting).
    work_assertions = _load_work_assertions(
        work_root,
        unit_texts=unit_texts,
        lexicon=lexicon,
        entities=entities,
        name_index=name_index,
        run_meta=run_meta,
        book_id=book_id,
        quarantine=quarantine,
        default_unit=unit_ids[0],
    )
    if work_assertions is None:
        assertions = engine_assertions
    else:
        # The work table is authoritative — full stop.
        #
        # There used to be an "engine fallback" here: any engine candidate whose
        # content key was absent from the work table got re-admitted. That is
        # wrong twice over. The work table is what the Assembler accepted after
        # role/value/evidence checks; a candidate missing from it was rejected on
        # purpose, and re-admitting it resurrects exactly what the stage threw
        # away. Measured: assemble quarantined a 性质 assertion as role_mismatch
        # (主体 was a concept name that cannot resolve), the writer re-admitted it
        # from the engine output, and the box ended up with 10 assertions while
        # the stage accepted 9 — which also made predicate_reconciliation report
        # accepted > emitted, i.e. the reconciliation itself stopped adding up.
        #
        # The content key cannot be used to "diff" the two sides either: the
        # stage stores post-resolution roles (e: ids) while the engine output
        # holds pre-resolution surfaces, so the keys never match and every
        # rejected row would look like a legitimate fallback.
        assertions = work_assertions
        skipped = len(engine_assertions) - len(
            [
                record
                for record in engine_assertions
                if _assertion_content_key(record)
                in {_assertion_content_key(row) for row in work_assertions}
            ]
        )
        if skipped > 0:
            warnings.append(
                "assemble/assertions.jsonl is authoritative; kept "
                f"{len(work_assertions)} accepted assertion(s) and did NOT "
                f"re-admit {skipped} engine candidate(s) the stage rejected "
                "(see audit/quarantine.jsonl for the reasons)"
            )

    # Fix dangling superseded_by pointers and superseded-without-pointer status.
    assertion_ids = {record["assertion_id"] for record in assertions}
    promoted_path = work_root / "assemble" / "promoted_facts.jsonl"
    promoted_added = 0
    if promoted_path.exists():
        for row in atomic.read_jsonl(promoted_path):
            record, problems, _ = records.normalize_assertion(
                row,
                output_unit=str(row.get("unit_id") or unit_ids[0]),
                unit_texts=unit_texts,
                lexicon=lexicon,
                entities_by_id=entities,
                name_index={k: v for k, v in name_index.items() if v},
                run_meta=run_meta,
                book_id=book_id,
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(
                        "assertion:promoted", "assertion", problem, run_meta.run_id
                    )
                )
            if record is not None and record["assertion_id"] not in assertion_ids:
                assertion_ids.add(record["assertion_id"])
                assertions.append(record)
                promoted_added += 1
        if promoted_added:
            warnings.append(
                f"claims promoted {promoted_added} supported statement(s) into fact assertions"
            )
    for record in assertions:
        pointer = record.get("superseded_by")
        if pointer and pointer not in assertion_ids:
            record["superseded_by"] = None
        if record["status"] == "superseded" and not record.get("superseded_by"):
            record["status"] = "disputed"

    # Prefer the cues integration's table; engine cues are only the fallback for
    # work directories produced before postprocess wiring existed. An existing
    # (even empty) work table wins: the writer never re-derives cue state.
    work_cues_path = work_root / "assemble" / "cues.jsonl"
    if work_cues_path.exists():
        cues = _load_work_cues(
            work_cues_path,
            unit_texts=unit_texts,
            entities=entities,
            run_meta=run_meta,
            book_id=book_id,
            default_unit=unit_ids[0],
            quarantine=quarantine,
        )
    else:
        cues = engine_cues

    _resolve_payoff_references(
        cues, events, set(unit_ids), quarantine, run_meta.run_id
    )

    # Time-expression producer: explicit mentions only, inferred before edges.
    from cbb_stage1.temporal.extract import (
        annotate_event_times,
        build_timeline_from_events,
    )

    events = annotate_event_times(events, unit_texts)
    generated_timeline = build_timeline_from_events(book_id, events, unit_texts)

    # Optional work tables.
    timeline_path = _first_existing(
        [
            work_root / "assemble" / "timeline.jsonl",
            work_root / "resolve" / "timeline.jsonl",
        ]
    )
    timeline: list[dict[str, Any]] = []
    if timeline_path is not None:
        for row in atomic.read_jsonl(timeline_path):
            record, problems = records.normalize_timeline_constraint(
                row,
                book_id=book_id,
                unit_texts=unit_texts,
                unit_ids=set(unit_ids),
                event_ids={event["event_id"] for event in events},
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(
                        f"timeline:{timeline_path.name}", "timeline", problem, run_meta.run_id
                    )
                )
            if record is not None:
                timeline.append(record)
        _assert_timeline_acyclic(timeline)

    # Work timeline is authoritative when the table exists (even empty); the
    # box-level extractor is only a legacy fallback for pre-postprocess work.
    if timeline_path is None:
        for row in generated_timeline:
            record, problems = records.normalize_timeline_constraint(
                row,
                book_id=book_id,
                unit_texts=unit_texts,
                unit_ids=set(unit_ids),
                event_ids={event["event_id"] for event in events},
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row("timeline:extracted", "timeline", problem, run_meta.run_id)
                )
            if record is not None:
                timeline.append(record)
        if generated_timeline:
            _assert_timeline_acyclic(timeline)

    merges_path = _first_existing(
        [
            work_root / "assemble" / "merges.jsonl",
            work_root / "resolve" / "merges.jsonl",
        ]
    )
    merges: list[dict[str, Any]] = []
    if merges_path is not None:
        entity_ids = set(entities)
        for row in atomic.read_jsonl(merges_path):
            record, problems = records.normalize_merge(
                row, book_id=book_id, entity_ids=entity_ids
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(
                        f"merge:{merges_path.name}", "merge", problem, run_meta.run_id
                    )
                )
            if record is not None:
                merges.append(record)
        if merges:
            _apply_merges(merges, entities, mentions, assertions, events, cues)

    claims_path = _first_existing(
        [
            work_root / "assemble" / "claims.jsonl",
            work_root / "resolve" / "claims.jsonl",
        ]
    )
    claims: list[dict[str, Any]] = []
    if claims_path is not None:
        assertion_layers = {
            record["assertion_id"]: record["claim_layer"] for record in assertions
        }
        for row in atomic.read_jsonl(claims_path):
            record, problems = records.normalize_claim(
                row,
                book_id=book_id,
                unit_texts=unit_texts,
                assertion_layers=assertion_layers,
                run_meta=run_meta,
            )
            for problem in problems:
                quarantine.append(
                    _quarantine_row(
                        f"claim:{claims_path.name}", "claim", problem, run_meta.run_id
                    )
                )
            if record is not None:
                claims.append(record)
    # No skeleton fill: a statement without a claims-integration row simply has
    # no claim. The explicit empty table is the honest output.

    hypotheses_path = _first_existing(
        [
            work_root / "derive" / "hypotheses.jsonl",
            work_root / "assemble" / "hypotheses.jsonl",
        ]
    )
    hypotheses: list[dict[str, Any]] = []
    if hypotheses_path is not None:
        for row in atomic.read_jsonl(hypotheses_path):
            try:
                record = normalize_hypothesis_record(
                    row, book_id=book_id, created_at=resolved_created
                )
            except (ValueError, TypeError) as exc:
                quarantine.append(
                    _quarantine_row(
                        f"hypothesis:{hypotheses_path.name}",
                        "hypothesis",
                        Problem("schema", str(exc)),
                        run_meta.run_id,
                    )
                )
                continue
            hypotheses.append(record)

    # ---- textual research aggregate (cues/hypotheses/conflicts) ------------
    research_path = _first_existing(
        [work_root / "derive" / "research.jsonl"]
    )
    research_rows = atomic.read_jsonl(research_path) if research_path is not None else []

    # ---- derived layer -----------------------------------------------------
    # Summaries come from derive's work table when present; the fallback is the
    # explicit text-less skeleton tree (never a synthesised chapter summary).
    summary_tree, summary_table = _load_work_summaries(
        work_root,
        units,
        book_id=book_id,
        volume_titles=volume_titles,
        changed_units=changed_units,
        created_at=resolved_created,
    )
    entity_cards = _load_work_entity_cards(work_root)
    assertions_by_entity: dict[str, list[str]] = {}
    for assertion in assertions:
        for entity_id in _assertion_entity_ids(assertion):
            assertions_by_entity.setdefault(entity_id, []).append(
                assertion["assertion_id"]
            )
    mentions_by_entity: dict[str, list[str]] = {}
    for mention in mentions:
        if mention.get("entity_id"):
            mentions_by_entity.setdefault(mention["entity_id"], []).append(
                mention["unit_id"]
            )

    # ---- projection-facing tables (v2.2 addendum) --------------------------
    narrative_rows = projection_tables.build_narrative(units, unit_texts)
    projection_tables.attach_narrative_anchors(narrative_rows)
    citations = projection_tables.build_citations(
        book_id, units, assertions, events, cues, claims, narrative_rows
    )
    narrative = [
        {
            "seg_id": row["seg_id"],
            "unit_id": row["unit_id"],
            "entity_id": row["entity_id"],
            "order": row["order"],
            "type": row["type"],
            "text": row["text"],
            "anchor": row["anchor"],
            "content_hash": row["content_hash"],
        }
        for row in narrative_rows
    ]
    graph_edges = projection_tables.build_graph_edges(book_id, assertions)
    # Conflicts are produced by validate (work/audit/conflicts.jsonl); the box
    # only normalises and carries them. Missing table => explicit empty.
    conflicts = _load_work_conflicts(
        work_root,
        book_id=book_id,
        assertion_ids={record["assertion_id"] for record in assertions},
        created_at=resolved_created,
    )
    hypotheses = sorted(hypotheses, key=lambda row: row["hypothesis_id"])
    assets = projection_tables.extract_assets(
        book_id,
        units,
        unit_texts,
        media_roots=[
            work_root,
            work_root / "assets",
            work_root / "plan",
            work_root / "plan" / "texts",
        ],
    )
    media_rows = _load_work_media(work_root)
    if media_rows:
        assets = _merge_media_assets(assets, media_rows)
        warnings.append(
            f"sources/assets.jsonl carries {len(media_rows)} real EPUB media file(s)"
        )
    if assets:
        warnings.append(
            f"sources/assets.jsonl contains {len(assets)} asset row(s) "
            "(extracted media + text references)"
        )
    coverage = projection_tables.build_coverage(
        book_id,
        list(entities.values()),
        assertions,
        events,
        timeline,
        cues,
        claims,
        summary_table,
        citations,
        graph_edges,
        conflicts,
        generated_at=resolved_created,
    )
    # Per-predicate reconciliation: emitted (engine) vs accepted (box table).
    # The table-level invariant cannot see a partial loss, so this block is what
    # lets the gate distinguish "1 of 9 survived" from "all 9 survived".
    coverage["predicate_reconciliation"] = _predicate_reconciliation(
        work_root, assertions
    )

    # ---- incremental closure (§4.16 / manifest.incremental) ----------------
    closure = dirty_closure(
        changed_units=changed_units,
        changed_entities=changed_entities,
        units=units if changed_units else (),
        entities=list(entities.values()) if changed_units or changed_entities else (),
        mentions=mentions if changed_units or changed_entities else (),
        assertions=assertions if changed_units or changed_entities else (),
        events=events if changed_units or changed_entities else (),
        cues=cues if changed_units or changed_entities else (),
        hypotheses=hypotheses if changed_units or changed_entities else (),
    )
    if incremental is not None:
        incremental_record = dict(incremental)
    else:
        incremental_record = {
            "parent": parent,
            "changed_units": list(closure.changed_units),
            "dirty_entities": list(closure.entities),
            "replay_base": None,
        }
    if changed_units or changed_entities:
        incremental_record.setdefault("parent", parent)
        incremental_record["changed_units"] = list(closure.changed_units)
        incremental_record["dirty_entities"] = list(closure.entities)

    # ---- audit -------------------------------------------------------------
    runs: list[dict[str, Any]] = []
    for rank, (unit_id, output, engine_path) in enumerate(outputs, 1):
        unit = next(unit for unit in units if unit["unit_id"] == unit_id)
        runs.append(
            {
                "run_id": f"{run_meta.run_id}-{rank:04d}",
                "box_id": resolved_box_id,
                "stage": "extract",
                "unit_id": unit_id,
                "engine": run_meta.engine,
                "model": run_meta.model,
                "prompt_sha256": run_meta.prompt_sha256,
                "input_sha256": unit["sha256"],
                "output_sha256": atomic.sha256_bytes(engine_path.read_bytes()),
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "started_at": run_meta.started_at,
                "ended_at": run_meta.ended_at,
                "status": "ok",
                "error": None,
            }
        )

    engine_contracts: dict[str, dict[str, Any]] = {}
    for unit_id, output, _ in outputs:
        contract = records.to_engine_contract(
            output,
            unit_text=unit_texts[unit_id],
            engine_name=run_meta.engine,
            engine_version=engine["version"],
            engine_model=run_meta.model,
        )
        name = f"{unit_id.replace(':', '_')}.json"
        engine_contracts[name] = contract

    provenance: list[dict[str, Any]] = []
    for kind, rows in (
        ("assertion", assertions),
        ("event", events),
        ("cue", cues),
        ("claim", claims),
        ("hypothesis", hypotheses),
        ("merge", merges),
    ):
        id_key = {
            "assertion": "assertion_id",
            "event": "event_id",
            "cue": "cue_id",
            "claim": "claim_id",
            "hypothesis": "hypothesis_id",
            "merge": "merge_id",
        }[kind]
        for row in rows:
            provenance.append(
                {
                    "record_id": row[id_key],
                    "inputs": {},
                    "derivation": [
                        {
                            "engine": run_meta.engine,
                            "run_id": run_meta.run_id,
                            "prompt_sha256": run_meta.prompt_sha256,
                        }
                    ],
                }
            )

    # ---- tables ------------------------------------------------------------
    tables: dict[str, list[dict[str, Any]]] = {
        "sources/units.jsonl": units,
        "identity/mentions.jsonl": mentions,
        "identity/entities.jsonl": sorted(entities.values(), key=lambda item: item["entity_id"]),
        "identity/merges.jsonl": merges,
        "knowledge/assertions.jsonl": assertions,
        "knowledge/claims.jsonl": claims,
        "knowledge/citations.jsonl": citations,
        "knowledge/narrative.jsonl": narrative,
        "knowledge/graph_edges.jsonl": graph_edges,
        "knowledge/conflicts.jsonl": conflicts,
        "knowledge/events.jsonl": events,
        "knowledge/timeline.jsonl": timeline,
        "knowledge/cues.jsonl": cues,
        "knowledge/proposed_predicates.jsonl": sorted(
            proposals.values(), key=lambda item: item["predicate"]
        ),
        "derived/hypotheses.jsonl": hypotheses,
        "derived/research.jsonl": research_rows,
        "audit/runs.jsonl": runs,
        "audit/provenance.jsonl": provenance,
        "audit/decisions.jsonl": [],
        "audit/quarantine.jsonl": quarantine,
        "audit/conflicts.jsonl": conflicts,
        "sources/assets.jsonl": assets,
        "derived/summaries.jsonl": summary_table,
        "derived/entity_cards.jsonl": entity_cards,
    }
    if claims:
        tables["knowledge/claims.jsonl"] = claims

    # Optional derived-layer pass-through tables (work -> box, verbatim).
    for work_relative, box_relative in (
        ("derive/forecasts.jsonl", "derived/inference/forecasts.jsonl"),
    # §5.20 inference tables. Without a copy entry the work table is computed and then
    # silently dropped on the way to the box — the "静默丢弃" shape the wiring audit
    # documented.
    ("derive/stylometry.jsonl", "derived/inference/stylometry.jsonl"),
    ("derive/themes.jsonl", "derived/inference/themes.jsonl"),
    ("derive/emotion_curves.jsonl", "derived/inference/emotion_curves.jsonl"),
    ("derive/motivation_chains.jsonl", "derived/inference/motivation_chains.jsonl"),
        ("derive/community.jsonl", "derived/community_reports.jsonl"),
        ("derive/raptor.jsonl", "derived/raptor.jsonl"),
    ):
        rows = _load_passthrough_derived(work_root, work_relative)
        if rows is not None:
            tables[box_relative] = rows

    counts = {
        "units": len(units),
        "mentions": len(mentions),
        "entities": len(entities),
        "assertions": len(assertions),
        "statements": sum(
            1 for row in assertions if row.get("claim_layer") == "statement"
        ),
        "claims": len(claims),
        "citations": len(citations),
        "narratives": len(narrative),
        "graph_edges": len(graph_edges),
        "assets": len(assets),
        "events": len(events),
        "cues": len(cues),
        "conflicts": len(conflicts),
        "superseded": sum(
            1 for row in assertions if row.get("status") == "superseded"
        ),
        "quarantined": len(quarantine),
    }

    volumes: dict[int, int] = {}
    for unit in units:
        volumes[unit["volume"]] = volumes.get(unit["volume"], 0) + 1
    volume_rows = [
        {
            "no": number,
            "title": (volume_titles or {}).get(number) or f"第{number}卷",
            "units": volumes[number],
        }
        for number in sorted(volumes)
    ]

    source_format = str(run.get("book_format") or "text").lower()
    source_format = {"text": "txt", "epub": "epub", "mixed": "mixed"}.get(
        source_format, "txt"
    )

    # Write the temp box (atomic publish happens after the lint gate).
    temp = atomic.new_temp_dir(out)
    try:
        for row in media_rows:
            source_rel = str(row.get("source_path") or "")
            if not source_rel:
                continue
            source = work_root / source_rel
            if not source.is_file():
                warnings.append(f"media source missing: {source_rel}")
                continue
            target = temp / str(row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        for relative, rows in tables.items():
            atomic.write_jsonl(temp / relative, rows)

        # lexicon (profiles hashes are computed from the written bytes)
        atomic.write_json(temp / "lexicon" / "genre.json", lexicon.genre_file())
        atomic.write_json(temp / "lexicon" / "predicates.json", lexicon.predicates_file())
        atomic.write_json(temp / "lexicon" / "types.json", lexicon.types_file())
        atomic.write_json(temp / "derived" / "summary_tree.json", summary_tree)

        for unit in units:
            text = unit_texts[unit["unit_id"]]
            atomic.write_text(temp / unit["text_path"], text)

        for name, contract in engine_contracts.items():
            atomic.write_json(temp / "audit" / "engine" / name, contract)

        source_files = [
            {"path": unit["text_path"], "sha256": unit["sha256"]} for unit in units
        ]
        soft_gates = _soft_gate_report(work_root)
        # Provisional report only. The canonical validator reads qc/report.json
        # while it measures, so the file must exist first; the flags written
        # here are replaced by the measured values before anything is published.
        provisional_gates = {name: "pass" for name in GATE_NAMES}
        qc_report = _qc_report_payload(
            box_id=resolved_box_id,
            hard_gates=provisional_gates,
            soft_gates=soft_gates,
            assertions=assertions,
            merges=merges,
            generated_at=resolved_created,
            capability_gaps=_capability_gaps(work_root, soft_gates),
            work_table_problems=work_table_problems,
        )
        atomic.write_json(temp / "qc" / "report.json", qc_report)
        atomic.write_json(temp / "qc" / "coverage.json", coverage)
        atomic.write_jsonl(
            temp / "qc" / "gates.jsonl",
            [{"gate": gate, "status": status} for gate, status in provisional_gates.items()],
        )
        (temp / "qc" / "samples").mkdir(parents=True, exist_ok=True)

        # index views (materialised, rebuildable)
        by_entity: dict[str, dict[str, Any]] = {}
        for entity_id in sorted(entities):
            by_entity[entity_id] = {
                "assertions": sorted(assertions_by_entity.get(entity_id, [])),
                "mentions": sorted(
                    mention["mention_id"]
                    for mention in mentions
                    if mention.get("entity_id") == entity_id
                ),
                "units": sorted(set(mentions_by_entity.get(entity_id, []))),
            }
        by_unit: dict[str, dict[str, Any]] = {}
        for unit in units:
            uid = unit["unit_id"]
            by_unit[uid] = {
                "mentions": sum(1 for mention in mentions if mention["unit_id"] == uid),
                "entities": sorted(
                    {
                        mention["entity_id"]
                        for mention in mentions
                        if mention["unit_id"] == uid and mention.get("entity_id")
                    }
                ),
                "assertions": sorted(
                    assertion["assertion_id"]
                    for assertion in assertions
                    if uid in assertion.get("deps", [])
                ),
                "events": sorted(
                    event["event_id"]
                    for event in events
                    if event["unit_id"] == uid
                ),
                "cues": sorted(
                    cue["cue_id"]
                    for cue in cues
                    if (cue.get("setup") or {}).get("unit_id") == uid
                ),
            }
        by_predicate: dict[str, list[str]] = {}
        for assertion in assertions:
            by_predicate.setdefault(assertion["predicate"], []).append(
                assertion["assertion_id"]
            )
        atomic.write_json(temp / "index" / "by_entity.json", by_entity)
        atomic.write_json(temp / "index" / "by_unit.json", by_unit)
        atomic.write_json(
            temp / "index" / "by_predicate.json",
            {key: sorted(value) for key, value in sorted(by_predicate.items())},
        )
        atomic.write_json(
            temp / "index" / "timeline.json",
            {
                "events": sorted(event["event_id"] for event in events),
                "constraints": sorted(row["constraint_id"] for row in timeline),
            },
        )

        atomic.write_text(
            temp / "SPEC.md",
            "# 箱\n\n"
            f"- box: `{resolved_box_id}`\n"
            f"- spec: `{BOX_SPEC}` / `{SPEC_VERSION}`\n"
            f"- book: `{book_id}` — {resolved_title}\n"
            f"- units: {len(units)}; mentions: {len(mentions)}; "
            f"entities: {len(entities)}; assertions: {len(assertions)}; "
            f"events: {len(events)}; cues: {len(cues)}\n"
            "- 读法：`sources` → `identity` → `knowledge` → `derived` → `audit` → `index`。\n",
        )

        # hashes over every file except manifest/CHECKSUMS/SPEC
        files, merkle = _collect_box_hashes(temp)

        manifest = {
            "spec": BOX_SPEC,
            "spec_version": SPEC_VERSION,
            "box_id": resolved_box_id,
            "book": {
                "id": book_id,
                "title": str(resolved_title),
                "language": str(run.get("language") or "zh-CN"),
                "volumes": volume_rows,
                "series_id": series_id or run.get("series_id"),
                "volume_no": volume_no if volume_no is not None else run.get("volume_no"),
            },
            "created_at": resolved_created,
            "source": {
                "format": source_format,
                "files": source_files,
                "unit_count": len(units),
                "char_count": sum(unit["chars"] for unit in units),
            },
            "profiles": {
                "genre_id": lexicon.genre_id,
                "genre_sha256": files["lexicon/genre.json"],
                "predicates_sha256": files["lexicon/predicates.json"],
                "types_sha256": files["lexicon/types.json"],
            },
            "engine": {
                "primary": engine["name"],
                "version": engine["version"],
                "model": engine["model"],
                "prompt_sha256": engine["prompt_sha256"],
            },
            "counts": counts,
            "hashes": {"files": files, "merkle_root": merkle},
            "status": "built",
            "parent": parent,
            "incremental": incremental_record,
            "qc": {
                # Provisional; replaced by the measured value below before the
                # box is published.
                "hard_gates": "pass",
                "soft_gates": soft_gates,
                "human_reviewed_units": 0,
            },
        }
        atomic.write_json(temp / "manifest.json", manifest)
        atomic.write_json(temp / "CHECKSUMS.json", {"files": files, "merkle_root": merkle})

        schema_errors = _validate_schema_rows(
            tables, manifest, engine_contracts
        )

        # ---- measured hard gates (no literals) ----------------------------
        # The canonical box_lint implementation is the single measurement: its
        # real error list is bucketed onto the seven gate names, so an empty box
        # can no longer report 7/7 pass.
        gate_report = measure_hard_gates(temp, extra_errors=schema_errors)
        hard_gates = gate_report.gates
        qc_report = _qc_report_payload(
            box_id=resolved_box_id,
            hard_gates=hard_gates,
            soft_gates=soft_gates,
            assertions=assertions,
            merges=merges,
            generated_at=resolved_created,
            hard_gate_errors=gate_report.by_gate,
            capability_gaps=_capability_gaps(work_root, soft_gates),
            work_table_problems=work_table_problems,
        )
        atomic.write_json(temp / "qc" / "report.json", qc_report)
        atomic.write_jsonl(
            temp / "qc" / "gates.jsonl",
            [{"gate": gate, "status": status} for gate, status in hard_gates.items()],
        )
        qc_block = manifest.get("qc")
        if isinstance(qc_block, dict):
            qc_block["hard_gates"] = "pass" if gate_report.ok else "fail"
        # qc/report.json and qc/gates.jsonl are part of the hash set: recompute
        # the manifest hashes/CHECKSUMS over the measured report.
        _write_manifest_and_checksums(temp, manifest)

        if schema_errors:
            raise BuildError("box hard gates failed (schema)", schema_errors)

        lint_errors: list[str] = list(gate_report.errors)
        if verify and gate_report.errors:
            raise BuildError("canonical box_lint gate failed", gate_report.errors)

        atomic.publish_dir(temp, out)
    except BaseException:
        atomic.remove_tree(temp)
        raise

    return BuildResult(
        box=out,
        box_id=resolved_box_id,
        status="built",
        counts=counts,
        quarantine=tuple(quarantine),
        warnings=tuple(warnings),
        lint_errors=tuple(lint_errors),
    )


def _assertion_entity_ids(assertion: Mapping[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str) and value.startswith("e:"):
            found.append(value)
        elif isinstance(value, Mapping):
            for inner in value.values():
                walk(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                walk(inner)

    walk(assertion.get("roles") or {})
    return found


# ---------------------------------------------------------------------------
# module entry point (``python -m cbb_stage1.box.writer``)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cbb_stage1.box.writer",
        description="Build an immutable v2.2 box from a pipeline work directory.",
    )
    parser.add_argument("--work", required=True, help="work/ directory")
    parser.add_argument("--out", required=True, help="destination box directory")
    parser.add_argument("--book-title")
    parser.add_argument("--box-id")
    parser.add_argument("--created-at")
    parser.add_argument("--parent")
    parser.add_argument("--lexicon")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="publish even if measured hard gates fail (qc records fail)",
    )
    args = parser.parse_args(argv)
    try:
        result = build_box(
            args.work,
            args.out,
            book_title=args.book_title,
            box_id=args.box_id,
            created_at=args.created_at,
            parent=args.parent,
            lexicon_dir=args.lexicon,
            verify=not args.no_verify,
        )
    except (BuildError, FileExistsError) as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
