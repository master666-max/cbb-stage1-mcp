"""Field-coverage check: does every field the spec REQUIRES actually carry a value?

This exists because three gates all passed while ``verification.level`` was
permanently null:

  * ``wiring`` only proves every module is *imported* from an entry point;
  * ``pytest`` only proves the tests I wrote (to match my own implementation) pass;
  * ``counts`` only reports fields someone remembered to count.

A required field that is always null satisfies all three. The only way to catch it
is to compare the schema against the produced artifact, field by field.

Usage::

    python tools/field_coverage.py <box-dir>
    python tools/field_coverage.py <box-dir> --json

Exit code 1 when a required field is empty in every row, because that is a real
defect rather than a warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: schema file -> box table (relative to the box root).
SCHEMA_TABLES: dict[str, str] = {
    "assertion.schema.json": "knowledge/assertions.jsonl",
    "citation.schema.json": "knowledge/citations.jsonl",
    "claim.schema.json": "knowledge/claims.jsonl",
    "conflict.schema.json": "knowledge/conflicts.jsonl",
    "cue.schema.json": "knowledge/cues.jsonl",
    "event.schema.json": "knowledge/events.jsonl",
    "graph-edge.schema.json": "knowledge/graph_edges.jsonl",
    "narrative.schema.json": "knowledge/narrative.jsonl",
    "entity.schema.json": "identity/entities.jsonl",
    "mention.schema.json": "identity/mentions.jsonl",
    "merge.schema.json": "identity/merges.jsonl",
    "timeline.schema.json": "knowledge/timeline.jsonl",
    "unit.schema.json": "sources/units.jsonl",
    "asset.schema.json": "sources/assets.jsonl",
    "hypothesis.schema.json": "derived/hypotheses.jsonl",
    "research.schema.json": "derived/research.jsonl",
    # The inference layer (§5.20). Without these entries its required fields are
    # invisible to this gate, which is exactly how a required-but-always-null field
    # survived three green gates before (see the module docstring).
    "stylometry.schema.json": "derived/inference/stylometry.jsonl",
    "themes.schema.json": "derived/inference/themes.jsonl",
    "motivation_chain.schema.json": "derived/inference/motivation_chains.jsonl",
    "emotion_curve.schema.json": "derived/inference/emotion_curves.jsonl",
    "run.schema.json": "audit/runs.jsonl",
}

#: Fields that are legitimately allowed to be empty in some rows. Each entry needs
#: a reason, otherwise this list becomes a place to hide defects.
ALLOWED_EMPTY: dict[str, str] = {
    "assertion.superseded_by": "only set when a later assertion replaces this one",
    "assertion.speaker": "only claim_layer=statement carries a speaker",
    "assertion.verification": "only assertions that went through a verifier have it",
    "assertion.derived_from": "only promoted/derived assertions have provenance",
    "event.time": "only events with an explicit time expression carry it",
    "event.location": "only events with a resolved place carry it",
    "cue.trigger": "a cue may be paid off without an explicit trigger",
    "cue.payoff": "an unresolved cue has no payoff yet",
    "entity.aliases": "an entity with a single surface has no aliases",
    "timeline.verify": "only verified constraints carry an NLI verdict",
    "unit.narrator": "only units with a declared narrator carry one",
}

#: Box tables that are legitimately empty for a single-unit run.
EMPTY_TABLE_OK: dict[str, str] = {
    "knowledge/claims.jsonl": "no statement-layer assertions in this material",
    "knowledge/cues.jsonl": "no foreshadowing in this material",
    "knowledge/timeline.jsonl": "no explicit time progression in this material",
    "knowledge/conflicts.jsonl": "no contradictions in this material",
    "knowledge/graph_edges.jsonl": "projection view, rebuilt from assertions",
    "identity/merges.jsonl": "no duplicate entities found",
    "derived/hypotheses.jsonl": "no insufficient-evidence claims to abduce from",
    "derived/research.jsonl": "no cues/hypotheses to aggregate",
    "sources/assets.jsonl": "text-only material has no media assets",
    "audit/runs.jsonl": "written by the engine adapter, not by build_box",
}


@dataclass
class FieldStat:
    table: str
    field: str
    rows: int = 0
    empty: int = 0
    required: bool = False
    allowed_reason: str | None = None
    #: True when the schema permits ``null`` for this field. A required field that
    #: allows null and is always null is a *suspicious* pattern, not a defect: the
    #: schema author explicitly made it optional-in-value. Only a field that must
    #: carry a non-null value and never does is a real defect.
    nullable: bool = False

    @property
    def fill_rate(self) -> float:
        return 1.0 if self.rows == 0 else 1.0 - self.empty / self.rows

    @property
    def is_defect(self) -> bool:
        return (
            self.required
            and self.rows > 0
            and self.empty == self.rows
            and self.allowed_reason is None
            and not self.nullable
        )

    @property
    def is_suspicious(self) -> bool:
        """Required, nullable, and never populated — worth a human look."""
        return (
            self.required
            and self.rows > 0
            and self.empty == self.rows
            and self.allowed_reason is None
            and self.nullable
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def _schema_fields(schema: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return (required, optional) property names for a table schema."""
    props = schema.get("properties") or {}
    required = [str(name) for name in schema.get("required") or [] if name in props]
    optional = [str(name) for name in props if name not in required]
    return required, optional


def _allows_null(schema: Mapping[str, Any], field: str) -> bool:
    """Does the schema accept ``null`` for this field?

    Handles the three shapes used in this repo: ``type: ["string", "null"]``,
    ``anyOf`` with a null branch, and an explicit ``"null"`` in a type list.
    """
    spec = (schema.get("properties") or {}).get(field)
    if not isinstance(spec, Mapping):
        return False
    declared = spec.get("type")
    if isinstance(declared, list) and "null" in declared:
        return True
    if declared == "null":
        return True
    for branch in spec.get("anyOf") or ():
        if isinstance(branch, Mapping) and branch.get("type") == "null":
            return True
    for branch in spec.get("oneOf") or ():
        if isinstance(branch, Mapping) and branch.get("type") == "null":
            return True
    return False


def check_box(box: Path, schema_dir: Path) -> tuple[list[FieldStat], list[str]]:
    stats: list[FieldStat] = []
    notes: list[str] = []

    for schema_name, table_rel in sorted(SCHEMA_TABLES.items()):
        schema_path = schema_dir / schema_name
        if not schema_path.exists():
            notes.append(f"schema missing: {schema_name}")
            continue
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            notes.append(f"schema unreadable: {schema_name} ({exc})")
            continue

        rows = _read_jsonl(box / table_rel)
        if not rows:
            reason = EMPTY_TABLE_OK.get(table_rel)
            if reason is None:
                notes.append(f"EMPTY TABLE (no stated reason): {table_rel}")
            continue

        required, optional = _schema_fields(schema)
        for name in required + optional:
            counter = Counter()
            for row in rows:
                counter["empty" if _is_empty(row.get(name)) else "filled"] += 1
            stat = FieldStat(
                table=table_rel,
                field=name,
                rows=len(rows),
                empty=counter.get("empty", 0),
                required=name in required,
                nullable=_allows_null(schema, name),
                allowed_reason=ALLOWED_EMPTY.get(f"{Path(table_rel).stem.rstrip('s')}.{name}")
                or ALLOWED_EMPTY.get(f"{_stem_key(table_rel)}.{name}"),
            )
            stats.append(stat)

    return stats, notes


def _stem_key(table_rel: str) -> str:
    """``knowledge/assertions.jsonl`` -> ``assertion`` (schema-ish singular)."""
    stem = Path(table_rel).stem
    if stem.endswith("ies"):
        return stem[:-3] + "y"
    if stem.endswith("s"):
        return stem[:-1]
    return stem


def format_report(stats: list[FieldStat], notes: list[str]) -> str:
    defects = [stat for stat in stats if stat.is_defect]
    empties = [
        stat
        for stat in stats
        if stat.required and stat.rows and stat.empty == stat.rows and stat.allowed_reason
    ]
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("FIELD COVERAGE — required fields that are always empty are defects")
    lines.append("=" * 78)

    suspicious = [stat for stat in stats if stat.is_suspicious]

    if defects:
        lines.append("")
        lines.append(f"DEFECTS ({len(defects)}) — required, non-nullable, always empty:")
        for stat in defects:
            lines.append(
                f"  {stat.table:34s} {stat.field:22s} empty in {stat.empty}/{stat.rows} rows"
            )
    else:
        lines.append("")
        lines.append("DEFECTS: none")

    if suspicious:
        lines.append("")
        lines.append(
            f"SUSPICIOUS ({len(suspicious)}) — schema allows null, but never populated:"
        )
        for stat in suspicious:
            lines.append(
                f"  {stat.table:34s} {stat.field:22s} null in {stat.empty}/{stat.rows} rows"
            )

    if empties:
        lines.append("")
        lines.append(f"allowed-empty ({len(empties)}):")
        for stat in empties:
            lines.append(
                f"  {stat.table:34s} {stat.field:22s} — {stat.allowed_reason}"
            )

    if notes:
        lines.append("")
        lines.append("notes:")
        for note in notes:
            lines.append(f"  {note}")

    lines.append("")
    lines.append(f"tables scanned: {len({s.table for s in stats})}")
    lines.append(f"fields scanned: {len(stats)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("box", help="box directory to inspect")
    parser.add_argument(
        "--schema-dir",
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "设计"
            / "抽取段-v6"
            / "schemas"
        ),
        help="directory holding the box schemas (default: 设计/抽取段-v6/schemas)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    box = Path(args.box)
    if not box.exists():
        print(f"box not found: {box}", file=sys.stderr)
        return 2
    if args.schema_dir:
        schema_dir = Path(args.schema_dir)
    else:
        # Resolve instead of hardcoding the sibling design tree; a published checkout
        # ships the schemas with the code.
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from cbb_stage1.schemas import schemas_dir

        schema_dir = schemas_dir()
    if not schema_dir.is_dir() or not any(schema_dir.glob("*.schema.json")):
        print(
            f"schema dir has no *.schema.json: {schema_dir}\n"
            "refusing to report a clean result from an empty scan",
            file=sys.stderr,
        )
        return 2

    stats, notes = check_box(box, schema_dir)
    if args.json:
        print(
            json.dumps(
                {
                    "defects": [
                        {
                            "table": stat.table,
                            "field": stat.field,
                            "rows": stat.rows,
                            "empty": stat.empty,
                        }
                        for stat in stats
                        if stat.is_defect
                    ],
                    "notes": notes,
                    "tables": len({s.table for s in stats}),
                    "fields": len(stats),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_report(stats, notes))
    return 1 if any(stat.is_defect for stat in stats) else 0


if __name__ == "__main__":
    raise SystemExit(main())
