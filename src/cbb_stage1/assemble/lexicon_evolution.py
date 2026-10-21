"""D5 — lexicon evolution: review proposals, approve one, re-scan affected units.

The inventory records D5 as "部分：只写增量闭环，无协议细节". Concretely: unregistered
predicates land in ``assemble/proposed_predicates.jsonl`` with ``status=proposed``
and then go nowhere — nothing can approve them, so the proposal table only ever
grows. Meanwhile ``PredicateRegistry.register()`` already refuses to run without
``approved_by``, i.e. the human gate exists but has no door.

This module is the door. It is deliberately small and deliberately boring:

  * ``list_proposals`` reads the table and reports what is pending;
  * ``approve_proposal`` writes the predicate into the lexicon file, recording who
    approved it and when, and flips the proposal row to ``approved``;
  * ``affected_units`` reports which units would need re-extraction, because a new
    predicate can only be emitted by re-running extraction, not by rewriting the
    assertions already in the work table.

Nothing here is automatic. A model cannot approve a predicate: the whole point of
I4 is that predicates are not grown from data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cbb_stage1.assemble.registry import (
    PredicateRegistry,
    PredicateSpec,
    RoleSpec,
)

__all__ = [
    "ApprovalError",
    "ApprovalResult",
    "affected_units",
    "approve_proposal",
    "list_proposals",
    "proposals_path",
]

#: Default location of the lexicon the registry is built from.
DEFAULT_LEXICON = Path("lexicon/predicates.json")


class ApprovalError(RuntimeError):
    """A proposal cannot be approved as asked."""


def proposals_path(work_root: str | Path) -> Path:
    return Path(work_root) / "assemble" / "proposed_predicates.jsonl"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def list_proposals(work_root: str | Path) -> list[dict[str, Any]]:
    """Every proposal row, newest state first, with its status."""
    rows = _read_rows(proposals_path(work_root))
    rows.sort(key=lambda row: (str(row.get("status") or ""), str(row.get("predicate") or "")))
    return rows


def _role_specs(examples: Sequence[Mapping[str, Any]]) -> list[RoleSpec]:
    """Infer a role schema from the proposal's observed role shapes.

    A predicate's roles are part of its meaning, so they are derived from what the
    extractor actually produced — but they are shown to the approver, not
    invented silently. When no example is usable, the approval is refused rather
    than guessing a schema.
    """
    if not examples:
        raise ApprovalError(
            "proposal carries no role_shape_examples; cannot infer a role schema. "
            "Register the predicate by hand in the lexicon instead."
        )
    first = examples[0]
    if not isinstance(first, Mapping) or not first:
        raise ApprovalError("role_shape_examples[0] is empty; nothing to infer from")

    specs: list[RoleSpec] = []
    for name, value in first.items():
        # An entity role is one shaped like "e:book:kind:hash"; anything else is a
        # literal. This mirrors how the assembler itself decides.
        is_entity = isinstance(value, str) and value.startswith("e:")
        if isinstance(value, (list, tuple)):
            is_entity = any(isinstance(item, str) and item.startswith("e:") for item in value)
        specs.append(
            RoleSpec(
                name=str(name),
                type="entity" if is_entity else "literal",
                required=True,
                repeatable=isinstance(value, (list, tuple)),
            )
        )
    return specs


@dataclass
class ApprovalResult:
    predicate: str
    library: str
    approved_by: str
    approved_at: str
    lexicon_path: str
    units_to_rescan: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "library": self.library,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "lexicon_path": self.lexicon_path,
            "units_to_rescan": list(self.units_to_rescan),
            "note": self.note,
        }


def ensure_lexicon_dir(work_root: str | Path) -> Path:
    """Make sure ``<work>/lexicon/`` exists, materialised from the project default.

    Why this has to happen here
    ---------------------------
    ``box/writer.py`` loads the registry from ``<work>/lexicon/`` and falls back to
    ``default_lexicon()`` when that directory is absent. A run that never had a
    registry therefore builds its box from the default — and an approval written
    anywhere else would be silently dropped on the next build.

    Measured: ``cbb_lexicon_approve`` refused with "lexicon not found at
    ``<work>/lexicon/predicates.json``" on a recorded run whose box had been built
    from the default. The refusal is right in isolation (a registry is the single
    source of truth) but it leaves the human gate unusable on the common path.

    So the registry is materialised from **this project's own** default: the files
    are what the build would have used anyway, plus the approved predicate. An
    existing registry is never touched — materialising is a creation, not a merge.
    """
    root = Path(work_root)
    directory = root / DEFAULT_LEXICON.parent
    predicates_path = directory / DEFAULT_LEXICON.name
    if predicates_path.exists():
        return predicates_path

    from cbb_stage1.box.lexicon import default_lexicon

    lexicon = default_lexicon()
    directory.mkdir(parents=True, exist_ok=True)

    genre = dict(lexicon.genre)
    genre.setdefault("id", lexicon.genre_id)
    (directory / "genre.json").write_text(
        json.dumps(genre, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "types.json").write_text(
        json.dumps({"types": lexicon.types}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / DEFAULT_LEXICON.name).write_text(
        json.dumps(
            {
                "contract": "cbb-lexicon/1",
                "predicates": [
                    {
                        "pred": spec.pred,
                        "library": spec.library,
                        "roles": [
                            {
                                "name": role.name,
                                "type": role.type,
                                "required": role.required,
                                "repeatable": role.repeatable,
                            }
                            for role in spec.roles
                        ],
                    }
                    for spec in lexicon.predicates.values()
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return predicates_path


def approve_proposal(
    work_root: str | Path,
    predicate: str,
    *,
    approved_by: str,
    library: str | None = None,
    lexicon_path: str | Path | None = None,
) -> ApprovalResult:
    """Approve one proposal: register it in the lexicon and mark the row approved.

    ``approved_by`` is mandatory and is recorded in the lexicon entry. This is the
    human gate I4 describes; passing an empty string is an error, not a default.
    """
    approver = str(approved_by or "").strip()
    if not approver:
        raise ApprovalError("approved_by is required: predicates are not grown from data")

    root = Path(work_root)
    path = proposals_path(root)
    rows = _read_rows(path)
    if not rows:
        raise ApprovalError(f"no proposals at {path}")

    target = str(predicate or "").strip()
    matching = [row for row in rows if str(row.get("predicate") or "") == target]
    if not matching:
        available = sorted({str(row.get("predicate") or "") for row in rows})
        raise ApprovalError(f"no proposal for {target!r}; available={available}")

    row = matching[0]
    if str(row.get("status") or "") == "approved":
        raise ApprovalError(f"proposal {target!r} is already approved")

    resolved_library = str(library or row.get("library") or "").strip()
    if not resolved_library:
        raise ApprovalError(
            f"proposal {target!r} has no library; pass --library explicitly"
        )

    spec = PredicateSpec(
        pred=target,
        library=resolved_library,  # type: ignore[arg-type]
        roles=_role_specs(list(row.get("role_shape_examples") or [])),
    )

    if lexicon_path:
        lex = Path(lexicon_path)
        if not lex.exists():
            raise ApprovalError(
                f"lexicon not found at {lex}; pass --lexicon. "
                "The registry is the single source of truth, so it must exist before "
                "a predicate can be added to it."
            )
    else:
        # Materialise the registry from the project default when the work tree has
        # none. Without this the approval would land in a file the box build does not
        # read, and would be lost on the next rebuild.
        lex = ensure_lexicon_dir(root)

    registry = PredicateRegistry.from_path(lex)
    registry.register(spec, approved_by=approver)

    approved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = registry.to_lexicon()
    payload["approved"] = payload.get("approved") or []
    payload["approved"].append(
        {"predicate": target, "approved_by": approver, "approved_at": approved_at}
    )
    lex.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Flip the proposal row so a second approval cannot happen by accident.
    for item in matching:
        item["status"] = "approved"
        item["decided_by"] = approver
        item["decided_at"] = approved_at
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
        encoding="utf-8",
    )

    units = affected_units(root, target)
    return ApprovalResult(
        predicate=target,
        library=resolved_library,
        approved_by=approver,
        approved_at=approved_at,
        lexicon_path=str(lex),
        units_to_rescan=units,
        note=(
            "注册表已更新。新谓词只能由重新抽取产出，已写入的断言不会被改写；"
            "受影响单元见 units_to_rescan。"
        ),
    )


def affected_units(work_root: str | Path, predicate: str) -> list[str]:
    """Units whose text mentions the predicate's observed spans.

    A newly registered predicate can only enter the box through re-extraction, so
    the useful answer is "which units should be re-run", derived from where the
    proposal was seen.
    """
    root = Path(work_root)
    rows = _read_rows(proposals_path(root))
    units: list[str] = []
    for row in rows:
        if str(row.get("predicate") or "") != predicate:
            continue
        for unit_id in row.get("seen_in") or []:
            value = str(unit_id)
            if value and value not in units:
                units.append(value)
    return units
