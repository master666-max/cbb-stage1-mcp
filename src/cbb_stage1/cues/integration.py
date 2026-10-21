"""P0-C cues integration: work cues -> state machine -> ``work/assemble/cues.jsonl``.

Unified postprocess entry point::

    run(work_root, *, book_id, outputs, units, created_at, upstream=None) -> dict

Inputs (first existing wins):

* ``work/resolve/cues.jsonl`` -- the orchestrator collector table (cue
  candidates flattened from engine outputs); this is the normal input;
* ``outputs`` -- fallback for a standalone run without a work table.

The step runs the code state machine (:mod:`cbb_stage1.cues.state`), never an
LLM judgement:

* ``candidate --confirm--> active`` when setup evidence is present;
* ``active/dormant --trigger--> triggered`` when a trigger condition/evidence is
  present;
* ``triggered --payoff--> paid``; a payoff that arrives while only active/dormant
  (or before the trigger unit) sets ``premature_payoff=true`` and does **not**
  force the cue closed (I18/I24);
* ``--falsify--> falsified`` when ``falsified_reason`` is present.

Writes only ``work/assemble/cues.jsonl`` (all cue records, original evidence
kept intact for the box writer). Cue invariant violations are returned so the
validate step can single-write ``work/audit/quarantine.jsonl``; this step does
not touch audit tables.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..temporal.integration import (
    as_dict,
    first_table,
    from_outputs,
    outputs_list,
    unit_order_from,
    write_jsonl,
)
from .state import CUE_KINDS, CUE_STATES, apply_event, cue_id_of, validate_cue

STEP = "cues"

CUE_TABLES = ("resolve/cues.jsonl",)
CUES_TABLE = "assemble/cues.jsonl"


def _setup_unit(cue: Mapping[str, Any]) -> str | None:
    setup = cue.get("setup")
    if not isinstance(setup, Mapping):
        return None
    if isinstance(setup.get("unit_id"), str):
        return setup["unit_id"]
    for item in setup.get("evidence") or []:
        if isinstance(item, Mapping) and isinstance(item.get("unit_id"), str):
            return item["unit_id"]
    return None


def _trigger_unit(cue: Mapping[str, Any]) -> str | None:
    trigger = cue.get("trigger")
    if isinstance(trigger, Mapping) and isinstance(trigger.get("unit_id"), str):
        return trigger["unit_id"]
    if isinstance(trigger, Mapping):
        for item in trigger.get("evidence") or []:
            if isinstance(item, Mapping) and isinstance(item.get("unit_id"), str):
                return item["unit_id"]
    return None


def _payoff_unit(cue: Mapping[str, Any]) -> str | None:
    payoff = cue.get("payoff")
    if isinstance(payoff, Mapping) and isinstance(payoff.get("unit_id"), str):
        return payoff["unit_id"]
    if isinstance(payoff, Mapping):
        for item in payoff.get("evidence") or []:
            if isinstance(item, Mapping) and isinstance(item.get("unit_id"), str):
                return item["unit_id"]
    return None


def _has_setup_evidence(cue: Mapping[str, Any]) -> bool:
    setup = cue.get("setup")
    return isinstance(setup, Mapping) and bool(setup.get("evidence"))


def _has_trigger(cue: Mapping[str, Any]) -> bool:
    trigger = cue.get("trigger")
    if not isinstance(trigger, Mapping):
        return False
    condition = trigger.get("condition")
    return bool((isinstance(condition, str) and condition.strip()) or trigger.get("evidence"))


def _has_payoff_evidence(cue: Mapping[str, Any]) -> bool:
    payoff = cue.get("payoff")
    return isinstance(payoff, Mapping) and bool(payoff.get("evidence"))


def _emit_cue(raw: Any, *, book_id: str,
              unit_order: Mapping[str, int] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the code state machine for one cue candidate; return (record, violations)."""
    record = as_dict(raw)
    record.setdefault("book_id", book_id)
    if record.get("kind") not in CUE_KINDS:
        record["kind"] = "object"
    if record.get("status") not in CUE_STATES:
        record["status"] = "candidate"
    # 既有合法 cue_id 原样保留；缺号/非法才用统一 cue: 公式重算
    record["cue_id"] = cue_id_of(record)
    violations: list[dict[str, Any]] = []

    def _collect(transition: Any) -> None:
        for violation in transition.violations:
            violations.append({"cue_id": record["cue_id"], "code": violation.code,
                               "detail": violation.detail})

    # 1) candidate -> active (setup evidence is the guard)
    if record["status"] == "candidate" and _has_setup_evidence(record):
        transition = apply_event(record, "confirm", unit_id=_setup_unit(record))
        _collect(transition)
        record = dict(transition.cue)

    # 2) active/dormant -> triggered (condition or evidence is the guard)
    if record["status"] in ("active", "dormant") and _has_trigger(record):
        transition = apply_event(record, "trigger", unit_id=_trigger_unit(record))
        _collect(transition)
        record = dict(transition.cue)

    # 3) payoff: only after triggered is clean; earlier payoff is flagged.
    if _has_payoff_evidence(record) and record["status"] in (
            "active", "dormant", "triggered"):
        transition = apply_event(record, "payoff", unit_id=_payoff_unit(record),
                                 unit_order=dict(unit_order) if unit_order else None)
        _collect(transition)
        record = dict(transition.cue)

    # 4) falsified needs a reason; apply when the candidate carries one.
    reason = record.get("falsified_reason")
    if isinstance(reason, str) and reason.strip() and record["status"] in (
            "candidate", "active", "dormant", "triggered"):
        transition = apply_event(record, "falsify", reason=reason)
        _collect(transition)
        record = dict(transition.cue)

    # 5) derived bookkeeping only (never invent triggered_at / payoff evidence)
    if record["status"] in ("active", "triggered", "paid") and not record.get("activated_at"):
        setup_unit = _setup_unit(record)
        if setup_unit:
            record["activated_at"] = setup_unit

    violations.extend({"cue_id": record["cue_id"], "code": violation.code,
                      "detail": violation.detail}
                     for violation in validate_cue(record, unit_order=unit_order or None))
    return record, violations


def run(work_root: str | Path, *, book_id: str, outputs: Any, units: Any,
        created_at: str, upstream: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Advance cue states for this work root and write the cues work table."""
    root = Path(work_root)
    up = dict(upstream or {})
    output_rows = outputs_list(outputs)

    cues, cue_path = first_table(root, CUE_TABLES)
    if cue_path is None:
        cues = from_outputs(output_rows, "cues")
    unit_order = dict(up.get("unit_order") or {}) or unit_order_from(units)

    records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for raw in cues:
        record, cue_violations = _emit_cue(raw, book_id=book_id,
                                           unit_order=unit_order or None)
        records.append(record)
        violations.extend(cue_violations)
    records.sort(key=lambda row: str(row.get("cue_id")))
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in violations:
        unique.setdefault((str(item.get("cue_id")), str(item.get("code")),
                           str(item.get("detail"))), item)
    violations = list(unique.values())
    table_path = write_jsonl(root / CUES_TABLE, records)

    by_status = Counter(str(row.get("status")) for row in records)
    premature = sum(1 for row in records if row.get("premature_payoff"))
    notes: list[str] = []
    if not cues:
        notes.append("无 cue 候选输入：写出空 cues 表（不造伏笔）")
    if cue_path is not None:
        notes.append("读 cue 候选表 %s（%d 条）" % (cue_path.relative_to(root), len(cues)))
    if premature:
        notes.append("%d 条 premature_payoff=true（回收早于触发/未触发，交 validate 进隔离）"
                     % premature)
    if not unit_order:
        notes.append("无 unit order：payoff 单元先后无法复算（只用状态机守卫）")

    counts = {
        "candidates": len(cues),
        "written": len(records),
        "by_status": dict(sorted(by_status.items())),
        "premature_payoff": premature,
        "violations": len(violations),
    }
    return {
        "step": STEP,
        "book_id": book_id,
        "created_at": created_at,
        "counts": counts,
        "paths": {"cues": str(table_path)},
        "violations": violations,
        "notes": notes,
    }
