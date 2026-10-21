"""Textual-research aggregate (v2.2 ``derived/research.jsonl``).

This is not a new fact source: each row points back to an existing cue,
hypothesis or conflict and carries its evidence/missing premises.  Projection can
render it as "考据 / 未解之谜 / 彩蛋" without treating hypotheses as facts.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cbb_stage1 import ids

__all__ = ["build_research"]

_KINDS = ("cue", "hypothesis", "conflict")


def _research_id(book_id: str, kind: str, source_id: str) -> str:
    return "rs:" + ids.short_hash(f"{book_id}|{kind}|{source_id}", 16)


def _evidence(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        nested = value.get("evidence")
        if isinstance(nested, (list, tuple)):
            return _evidence(nested)
        return [dict(value)]
    if isinstance(value, (list, tuple)):
        found: list[dict[str, Any]] = []
        for item in value:
            found.extend(_evidence(item))
        return found
    return []


def build_research(
    book_id: str,
    *,
    cues: Iterable[Mapping[str, Any]] = (),
    hypotheses: Iterable[Mapping[str, Any]] = (),
    conflicts: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue in cues:
        cue_id = str(cue.get("cue_id") or "")
        if not cue_id:
            continue
        evidence = _evidence(cue.get("setup"))
        evidence.extend(_evidence(cue.get("payoff")))
        evidence.extend(_evidence(cue.get("trigger")))
        rows.append(
            {
                "research_id": _research_id(book_id, "cue", cue_id),
                "book_id": book_id,
                "kind": "cue",
                "claim": str(cue.get("expected") or cue.get("subject") or cue_id),
                "status": str(cue.get("status") or "open"),
                "evidence": evidence,
                "source_ids": [cue_id],
                "missing_premises": [],
                "confidence": None,
                "produced_by": "derive:research",
            }
        )
    for hypothesis in hypotheses:
        hid = str(hypothesis.get("hypothesis_id") or "")
        if not hid:
            continue
        rows.append(
            {
                "research_id": _research_id(book_id, "hypothesis", hid),
                "book_id": book_id,
                "kind": "hypothesis",
                "claim": str(hypothesis.get("claim") or ""),
                "status": str(hypothesis.get("status") or "candidate"),
                "evidence": _evidence(hypothesis.get("support")),
                "source_ids": [hid],
                "missing_premises": [
                    str(item) for item in hypothesis.get("missing_premises") or ()
                ],
                "confidence": hypothesis.get("confidence"),
                "produced_by": "derive:research",
            }
        )
    for conflict in conflicts:
        cid = str(conflict.get("conflict_id") or "")
        if not cid:
            continue
        rows.append(
            {
                "research_id": _research_id(book_id, "conflict", cid),
                "book_id": book_id,
                "kind": "conflict",
                "claim": str(conflict.get("detail") or cid),
                "status": str(conflict.get("status") or "open"),
                "evidence": [],
                "source_ids": [cid],
                "missing_premises": [],
                "confidence": None,
                "produced_by": "derive:research",
            }
        )
    unique = {row["research_id"]: row for row in rows}
    return [unique[key] for key in sorted(unique)]


