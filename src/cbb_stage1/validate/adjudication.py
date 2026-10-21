"""Human adjudication for conflicts and quarantine items.

The extraction stage had two human gates already — predicate approval (I4) and
supersede (D9) — and two decisions it had named but never given a door to:

  * ``validate/conflicts.py`` declares ``DECISIONS = ("left", "right", "merge",
    "disputed", "superseded")`` but implements one of them. The other four could not
    be recorded at all.
  * quarantine items had no adjudication. The projection stage has ``adjudicatedSeq``;
    this stage never got one, so a quarantined assertion sat there with no way for a
    person to say "that one is fine, drop it" or "that one was a false quarantine".

Two rules, both borrowed rather than invented:

  * **Nothing is deleted.** A decision appends a row to ``audit/decisions.jsonl`` and
    annotates the record. A superseded assertion is kept with ``status=superseded``
    (``apply_supersede``); a quarantined item is kept with an ``adjudication`` field.
    Deleting the thing being judged would destroy the evidence of the judgement.
  * **The human is named.** ``by`` is mandatory. An empty one is refused before any
    file is touched — the same gate ``cbb_lexicon_approve`` applies, and the same
    reason: a decision nobody owns is not a decision.

A decision row carries no material (``value``/``roles``/``quote``). The ledger records
that a decision was taken and why; the material stays in the table it came from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cbb_stage1.ids import stable_id

__all__ = [
    "CONFLICT_CHOICES",
    "QUARANTINE_DECISIONS",
    "adjudicate_quarantine",
    "resolve_conflict",
    "supersede_assertion",
]

#: The five choices ``validate/conflicts.py`` declares. All five are accepted here;
#: ``superseded`` additionally rewrites the assertions (it is the only one that does).
CONFLICT_CHOICES = ("left", "right", "merge", "disputed", "superseded")

#: Quarantine decisions. The three words differ in intent, not in effect: all three
#: record the decision and annotate the item. What "drop" should eventually mean for
#: the pipeline is a question for the first real adjudication, and guessing now would
#: mean inventing a semantic nobody agreed to.
QUARANTINE_DECISIONS = ("keep", "drop", "retain")

CONFLICTS_TABLE = "audit/conflicts.jsonl"
QUARANTINE_TABLE = "assemble/quarantine.jsonl"
DECISIONS_TABLE = "audit/decisions.jsonl"
ASSERTIONS_TABLE = "assemble/assertions.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(dict(row), ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _require_by(by: Any) -> str:
    who = str(by or "").strip()
    if not who:
        raise ValueError(
            "decided_by is required: a decision nobody owns is not a decision"
        )
    return who


def _append_decision(work_root: str | Path, decision: Mapping[str, Any]) -> Path:
    path = Path(work_root) / DECISIONS_TABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(decision), ensure_ascii=False) + "\n")
    return path


def resolve_conflict(
    work_root: str | Path,
    *,
    conflict_id: str,
    choice: str,
    by: str,
    reason: str,
    at: str | None = None,
) -> dict[str, Any]:
    """Record a human decision about one conflict.

    The conflict row is annotated, never deleted. For ``choice="superseded"`` the two
    assertions are also retconned through ``apply_supersede`` — that is the one choice
    with content consequences, and it reuses the existing implementation rather than
    growing a second one.
    """
    who = _require_by(by)
    picked = str(choice or "").strip()
    if picked not in CONFLICT_CHOICES:
        raise ValueError(
            f"choice {choice!r} is not one of {list(CONFLICT_CHOICES)}"
        )

    root = Path(work_root)
    conflicts_path = root / CONFLICTS_TABLE
    conflicts = _read_jsonl(conflicts_path)
    target = next(
        (row for row in conflicts if str(row.get("conflict_id") or "") == str(conflict_id)),
        None,
    )
    if target is None:
        raise KeyError(f"no such conflict: {conflict_id}")

    decision = {
        "decision_id": stable_id(
            "d:", {"kind": "conflict", "subject": str(conflict_id), "choice": picked}
        ),
        "kind": "conflict",
        "subject": str(conflict_id),
        "choice": picked,
        "decided_by": who,
        "reason": str(reason or ""),
        "at": at or "",
    }

    if picked == "superseded":
        # The only choice that rewrites content. Reuse the existing implementation.
        from cbb_stage1.validate.conflicts import apply_supersede

        assertions_path = root / ASSERTIONS_TABLE
        records = _read_jsonl(assertions_path)
        try:
            plan = apply_supersede(
                records,
                loser_id=str(target.get("left") or ""),
                winner_id=str(target.get("right") or ""),
                decided_by=who,
                reason=str(reason or ""),
                at=at,
            )
        except (KeyError, ValueError) as exc:
            # Name the conflict: "找不到 loser 记录" alone does not say which one.
            raise KeyError(
                f"conflict {conflict_id!r} (choice=superseded): {exc}"
            ) from exc
        _write_jsonl(assertions_path, plan.records)
        decision = dict(plan.decision)
        decision.setdefault("kind", "conflict")
        decision["conflict_id"] = str(conflict_id)

    target["status"] = picked
    target["adjudicated_by"] = who
    target["adjudication"] = {
        "choice": picked,
        "decided_by": who,
        "reason": str(reason or ""),
        "at": at or "",
    }
    _write_jsonl(conflicts_path, conflicts)
    _append_decision(root, decision)

    return {"ok": True, "choice": picked, "decision": decision}


def supersede_assertion(
    work_root: str | Path,
    *,
    loser: str,
    winner: str,
    by: str,
    reason: str,
    at: str | None = None,
) -> dict[str, Any]:
    """Mark one assertion superseded by another (D9 retcon).

    Both front-ends — the MCP tool and the CLI — call this. The CLI used to own the
    logic and the MCP tool shelled out to it; that made the MCP surface a wrapper over
    a development tool. The logic lives here, and each front-end is a thin adapter.

    The old record is kept with ``status=superseded``; nothing is deleted.
    """
    who = _require_by(by)
    root = Path(work_root)
    assertions_path = root / ASSERTIONS_TABLE
    records = _read_jsonl(assertions_path)
    if not records:
        raise KeyError(f"no assertion table at {assertions_path}")

    from cbb_stage1.validate.conflicts import apply_supersede

    plan = apply_supersede(
        records,
        loser_id=str(loser),
        winner_id=str(winner),
        decided_by=who,
        reason=str(reason or ""),
        at=at,
    )
    _write_jsonl(assertions_path, plan.records)
    _append_decision(root, plan.decision)
    return {
        "ok": True,
        "decision": plan.decision,
        "retcon": getattr(plan, "retcon", None),
    }


def adjudicate_quarantine(
    work_root: str | Path,
    *,
    item_id: str,
    decision: str,
    by: str,
    reason: str,
    at: str | None = None,
) -> dict[str, Any]:
    """Record a human decision about one quarantined item.

    The item is annotated, never deleted: the record of what was quarantined and why
    is the evidence that the quarantine happened, and a judgement that erases its own
    subject is not auditable.
    """
    who = _require_by(by)
    picked = str(decision or "").strip()
    if picked not in QUARANTINE_DECISIONS:
        raise ValueError(
            f"decision {decision!r} is not one of {list(QUARANTINE_DECISIONS)}"
        )

    root = Path(work_root)
    quarantine_path = root / QUARANTINE_TABLE
    items = _read_jsonl(quarantine_path)
    target = next(
        (row for row in items if str(row.get("item_id") or "") == str(item_id)), None
    )
    if target is None:
        raise KeyError(f"no such quarantine item: {item_id}")

    record = {
        "decision_id": stable_id(
            "d:", {"kind": "quarantine", "subject": str(item_id), "choice": picked}
        ),
        "kind": "quarantine",
        "subject": str(item_id),
        "choice": picked,
        "decided_by": who,
        "reason": str(reason or ""),
        "at": at or "",
    }

    target["adjudication"] = {
        "choice": picked,
        "decided_by": who,
        "reason": str(reason or ""),
        "at": at or "",
    }
    _write_jsonl(quarantine_path, items)
    _append_decision(root, record)

    return {"ok": True, "decision": record}
