"""P2-B temporal integration: work events -> timeline solve -> work tables.

Unified postprocess entry point (same shape as ``resolve/assemble/claims``)::

    run(work_root, *, book_id, outputs, units, created_at, upstream=None) -> dict

Inputs (work layer, read-only, priority order):

* events -- ``work/assemble/events.jsonl`` (W3's producer, **primary**; when it
  exists it is authoritative even if empty and the step must not silently fall
  back), then ``work/resolve/events.jsonl`` (legacy collector table), then
  ``outputs[*]["events"]`` (standalone fallback only);
* assertions -- ``work/assemble/assertions.jsonl`` -> ``work/resolve/assertions.jsonl``
  -> ``outputs[*]["assertions"]``;
* timeline constraints (no upstream producer yet): ``upstream["timeline"]`` /
  ``upstream["assemble"]["timeline"]`` / ``outputs[*]["timeline"]``.

Outputs:

* ``work/assemble/timeline.jsonl`` -- accepted constraints only (box §5.8 shape);
* ``work/audit/quarantine.jsonl`` -- appended, never rewritten;
* the return dict carries timeline conflicts (cycles, during/overlap
  contradictions, NLI disagreements) so the validate step can write
  ``work/audit/conflicts.jsonl`` (single writer for that table).

D11 NLI: when the orchestrator injects a pairwise callable as
``upstream["nli"]`` (built by :func:`cbb_stage1.temporal.nli_client.build_nli`)
or a config as ``upstream["nli_cfg"]``, verification runs **by default** and each
accepted constraint gets ``verify={method,question,result,verifier,model}``.
``upstream["verify_timeline"]=False`` disables it explicitly. An ``unknown``
verdict is recorded as unknown -- never treated as confirmation. No injection ->
fully offline (no client, no network).

No fabrication: with no timeline evidence this step writes an **empty** table and
an empty quarantine file. It never derives story order from narrative order --
the design keeps ``narrative_order`` and ``story_order`` separate (design/01
§5.8, I20/I25).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .nli_client import build_nli
from .extract import annotate_event_times, build_timeline_from_events
from .solver import solve

STEP = "temporal"

#: work-table locations this step reads (first existing wins; ``assemble`` is
#: W3's producer and wins over the legacy collector and outputs fallbacks).
ASSERTION_TABLES = ("assemble/assertions.jsonl", "resolve/assertions.jsonl")
EVENT_TABLES = ("assemble/events.jsonl", "resolve/events.jsonl")

#: work-table locations this step writes.
TIMELINE_TABLE = "assemble/timeline.jsonl"
QUARANTINE_TABLE = "audit/quarantine.jsonl"


# ---------------------------------------------------------------------------
# small IO helpers (shared with cues/ and validate/ integration modules)
# ---------------------------------------------------------------------------
def as_dict(record: Any) -> dict[str, Any]:
    """Coerce a pydantic record or mapping into a plain dict (copy)."""
    if hasattr(record, "model_dump"):
        return dict(record.model_dump())
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("记录必须是 Mapping 或 pydantic 模型，现得 %r" % (type(record),))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL table; a missing file yields ``[]`` (never an error)."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically write a JSONL table (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
             for row in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]],
                 key_fields: Sequence[str] = ()) -> int:
    """Append rows to a JSONL table, optionally deduping by ``key_fields``.

    Returns the number of rows actually added. A missing file with no new rows
    is still created (an explicit empty table beats a silently absent one).
    """
    added = [dict(row) for row in rows]
    existing = read_jsonl(path)
    if not key_fields or not existing:
        merged = existing + added
    else:
        seen = {tuple(str(row.get(k, "")) for k in key_fields) for row in existing}
        merged = list(existing)
        for row in added:
            key = tuple(str(row.get(k, "")) for k in key_fields)
            if key not in seen:
                seen.add(key)
                merged.append(row)
    write_jsonl(path, merged)
    return len(merged) - len(existing)


def first_table(root: Path, relatives: Iterable[str]) -> tuple[list[dict[str, Any]], Path | None]:
    """Return ``(rows, path)`` for the first existing table among ``relatives``."""
    for rel in relatives:
        path = root / rel
        if path.is_file():
            return read_jsonl(path), path
    return [], None


def outputs_list(outputs: Any) -> list[dict[str, Any]]:
    """Normalize ``outputs`` (list of engine dicts or unit_id->EngineOutput map)."""
    if outputs is None:
        return []
    if isinstance(outputs, Mapping):
        rows: list[dict[str, Any]] = []
        for unit_id, value in outputs.items():
            record = as_dict(value)
            record.setdefault("unit_id", str(unit_id))
            rows.append(record)
        return rows
    return [as_dict(value) for value in outputs]


def from_outputs(outputs: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    """Flatten ``outputs[*][key]`` lists into one table (order preserved)."""
    rows: list[dict[str, Any]] = []
    for output in outputs:
        value = output.get(key)
        if isinstance(value, Mapping):
            rows.append(dict(value))
        elif isinstance(value, (list, tuple)):
            rows.extend(dict(item) for item in value if isinstance(item, Mapping))
    return rows


def unit_order_from(units: Any) -> dict[str, int]:
    """Map ``unit_id -> order`` from unit records (order missing -> list index)."""
    order: dict[str, int] = {}
    for index, raw in enumerate(units or []):
        record = as_dict(raw)
        unit_id = record.get("unit_id") or record.get("id")
        if not unit_id:
            continue
        value = record.get("order")
        order[str(unit_id)] = int(value) if isinstance(value, int) and \
            not isinstance(value, bool) else index
    return order


# ---------------------------------------------------------------------------
# input collection
# ---------------------------------------------------------------------------
def _timeline_candidates(outputs: Sequence[Mapping[str, Any]],
                         upstream: Mapping[str, Any], book_id: str) -> list[dict[str, Any]]:
    """Collect timeline constraint candidates from upstream/outputs (no work table)."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        if isinstance(value, Mapping):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = [item for item in value if isinstance(item, Mapping)]
        else:
            return
        for item in items:
            row = dict(item)
            row.setdefault("book_id", book_id)
            identity = str(row.get("constraint_id") or json.dumps(
                row, ensure_ascii=False, sort_keys=True, default=str))
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)

    _add(upstream.get("timeline"))
    _add(upstream.get("timeline_constraints"))
    assemble_up = upstream.get("assemble")
    if isinstance(assemble_up, Mapping):
        _add(assemble_up.get("timeline"))
    for output in outputs:
        _add(output.get("timeline"))
        _add(output.get("timeline_constraints"))
    return rows


def _known_events(events: Sequence[Mapping[str, Any]]) -> set[str]:
    """Event-id universe from a source table (may be empty: an empty source is
    still authoritative -- dangling references must be quarantined)."""
    return {str(event.get("event_id")) for event in events if event.get("event_id")}


def _resolve_nli(upstream: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    """Resolve the injected NLI verifier: callable first, config second."""
    injected = upstream.get("nli")
    if callable(injected):
        return injected, "upstream.nli"
    cfg = upstream.get("nli_cfg")
    if cfg is not None:
        return build_nli(cfg), "upstream.nli_cfg"
    return None, None


def _load_unit_texts(root: Path, units: Iterable[Any]) -> dict[str, str]:
    """unit_id -> text for explicit time-expression extraction."""
    texts: dict[str, str] = {}
    rows = list(units or ())
    if not rows and (root / "plan" / "units.jsonl").exists():
        rows = read_jsonl(root / "plan" / "units.jsonl")
    for unit in rows:
        record = as_dict(unit) if not isinstance(unit, Mapping) else dict(unit)
        unit_id = str(record.get("unit_id") or "")
        text_path = record.get("text_path")
        if not unit_id or not text_path:
            continue
        path = Path(text_path)
        if not path.is_absolute():
            path = root / path
        try:
            texts[unit_id] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return texts


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(work_root: str | Path, *, book_id: str, outputs: Any, units: Any,
        created_at: str, upstream: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Solve the timeline for this work root and write the temporal work tables."""
    root = Path(work_root)
    up = dict(upstream or {})
    output_rows = outputs_list(outputs)

    assertions, assertion_path = first_table(root, ASSERTION_TABLES)
    if assertion_path is None:
        assertions = from_outputs(output_rows, "assertions")
    events, event_path = first_table(root, EVENT_TABLES)
    if event_path is not None:
        events_source = ("assemble"
                         if event_path.relative_to(root).as_posix().startswith("assemble/")
                         else "resolve")
    elif any("events" in output for output in output_rows):
        events = from_outputs(output_rows, "events")
        events_source = "outputs"
    else:
        events_source = "none"
    # An existing work table (even empty) defines the event universe: dangling
    # references must be quarantined, not silently accepted. ``None`` means "no
    # source at all" (standalone run), where references cannot be checked.
    known_events = _known_events(events) if events_source != "none" else None
    unit_order = unit_order_from(units) or dict(up.get("unit_order") or {})

    timeline_records = _timeline_candidates(output_rows, up, book_id)
    generated: list[dict[str, Any]] = []
    if events:
        unit_texts = _load_unit_texts(root, units)
        if unit_texts:
            annotated_events = annotate_event_times(events, unit_texts)
            generated = build_timeline_from_events(book_id, annotated_events, unit_texts)
    if generated:
        by_id = {str(row.get("constraint_id")): row for row in timeline_records}
        for row in generated:
            by_id.setdefault(str(row.get("constraint_id")), row)
        timeline_records = list(by_id.values())
    nli, nli_source = _resolve_nli(up)
    verify_setting = up.get("verify_timeline")
    # Injected NLI verifies by default (D11); explicit False turns it off.
    verify = bool(verify_setting) if verify_setting is not None else nli is not None

    result = solve(
        timeline_records,
        known_events=known_events,
        known_units=set(unit_order) or None,
        nli=nli,
        verify=verify,
    )

    timeline_rows = sorted((constraint.as_record() for constraint in result.constraints),
                           key=lambda row: str(row.get("constraint_id")))
    timeline_path = write_jsonl(root / TIMELINE_TABLE, timeline_rows)

    quarantine_rows = sorted((item.as_record() for item in result.quarantine),
                             key=lambda row: (str(row.get("item_id")),
                                              str(row.get("reason"))))
    quarantine_path = root / QUARANTINE_TABLE
    quarantine_added = append_jsonl(
        quarantine_path, quarantine_rows,
        key_fields=("item_id", "kind", "reason", "detail"))

    conflicts = sorted((conflict.as_record() for conflict in result.conflicts),
                       key=lambda row: str(row.get("conflict_id")))
    notes: list[str] = []
    if not timeline_records:
        notes.append("无时间约束输入：写出空 timeline 表（不拿叙事序造假 story order）")
    if result.cycles:
        notes.append("时序环 %d 条：约束进隔离（timeline_cycle），conflicts 交 validate 落盘"
                     % len(result.cycles))
    if assertion_path is not None:
        notes.append("读断言表 %s（%d 条）" % (assertion_path.relative_to(root), len(assertions)))
    if event_path is not None:
        notes.append("读事件表 %s（%d 条；work 表优先，engine outputs 只 fallback）"
                     % (event_path.relative_to(root), len(events)))
    elif events_source == "outputs":
        notes.append("无 work events 表：fallback 到 engine outputs 的 %d 条事件" % len(events))
    else:
        notes.append("无任何事件来源：事件引用校验跳过（不许猜）")
    if nli is None:
        notes.append("未注入 nli：不执行 NLI 验证（保持离线，不留假 verify）")
    elif not verify:
        notes.append("nli 已注入（来源 %s）但 verify_timeline=false：本轮不验证" % nli_source)
    else:
        notes.append("nli 注入来源=%s：约束的 verify 落 method/result/verifier" % nli_source)

    counts = {
        "assertions": len(assertions),
        "events": len(events),
        "events_source": events_source,
        "timeline_input": len(timeline_records),
        "timeline_generated": len(generated),
        "timeline_written": len(timeline_rows),
        "quarantine": len(quarantine_rows),
        "quarantine_added": quarantine_added,
        "conflicts": len(conflicts),
        "cycles": len(result.cycles),
        "nli_injected": nli is not None,
        "verified": bool(verify and nli is not None),
    }
    return {
        "step": STEP,
        "book_id": book_id,
        "created_at": created_at,
        "counts": counts,
        "paths": {"timeline": str(timeline_path), "quarantine": str(quarantine_path)},
        "conflicts": conflicts,
        "quarantine": quarantine_rows,
        "notes": notes,
    }
