"""Conflict detection + ``superseded`` closure + retcon history (design/01 §5.17).

Three conflict classes (all land in ``audit/conflicts.jsonl``, never auto-fix):

* ``attribute`` -- same subject + predicate, different literal values;
* ``relation``  -- same participant pair, reversed role direction or an
  explicitly exclusive predicate pair (``师兄``/``师弟`` ...);
* ``timeline``  -- constraint cycles / contradictions (delegated to
  :mod:`cbb_stage1.temporal.solver`) and dead-entity participation.

Versioning: a losing record gets ``status=superseded`` + ``superseded_by``
pointing at the new record; the winner gets ``derived_from`` pointing back. The
chain must be closed and acyclic (I17). **Retcon is a feature**: the old value
must be preserved for the stage-2 "设定变更史" (old value + new value + reason +
chapter), which is what :func:`retcon_history` renders.

Pure functions: inputs never mutated; every function returns copied records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..ids import canonical_json, conflict_id as make_conflict_id, stable_id
from ..temporal.solver import SolveResult, solve
from .confidence import Violation

CONFLICT_KINDS = ("attribute", "relation", "timeline")
CONFLICT_STATUSES = ("open", "resolved")
DECISIONS = ("left", "right", "merge", "disputed", "superseded")

# §5.12 拒因闭集（本模块只引用，不新造）
QUARANTINE_REASONS = (
    "schema", "evidence_not_found", "role_mismatch", "ref_missing", "timeline_cycle",
    "contradiction", "low_confidence", "duplicate", "unresolved_quote",
    "claim_layer_conflict",
)

SUBJECT_ROLES = ("主体", "subject", "实体", "entity", "施事", "agent")
VALUE_ROLES = ("值", "value", "literal", "object", "受事", "patient", "结果")
RELATION_LIBRARIES = ("关系", "relation")
ENTITY_PREFIX = "e:"


def _dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return dict(record.model_dump())
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("记录必须是 Mapping 或 pydantic 模型，现得 %r" % (type(record),))


def record_id(record: Any) -> str:
    """Best-effort id for assertions/events/claims/cues/constraints."""
    rec = _dict(record)
    for key in ("assertion_id", "event_id", "claim_id", "cue_id", "constraint_id",
                "merge_id", "record_id", "id"):
        value = rec.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("记录没有可识别的 id 字段：%r" % (sorted(rec),))


def _canonical(value: Any) -> Any:
    """Comparable form of a role value; literals stay literals, entities stay ids."""
    if isinstance(value, Mapping):
        if "literal" in value:
            return ("literal", value.get("literal"))
        if "entity_id" in value:
            return ("entity", value.get("entity_id"))
        return ("dict", canonical_json(value))
    if isinstance(value, list):
        return ("list", tuple(_canonical(v) for v in value))
    return ("value", value)


def _role_items(assertion: Mapping[str, Any], names: Iterable[str]) -> list[tuple[str, Any]]:
    roles = assertion.get("roles")
    if not isinstance(roles, Mapping):
        return []
    wanted = set(names)
    return [(k, v) for k, v in roles.items() if k in wanted]


def _subject(assertion: Mapping[str, Any]) -> tuple[str, Any] | None:
    for k, v in _role_items(assertion, SUBJECT_ROLES):
        return k, _canonical(v)
    return None


def _value(assertion: Mapping[str, Any]) -> Any:
    items = _role_items(assertion, VALUE_ROLES)
    if items:
        vals = [_canonical(v) for _k, v in items]
        return tuple(sorted(vals, key=repr)) if len(vals) > 1 else vals[0]
    # fallback: remaining role values (excluding subject roles) — still a claim value
    roles = assertion.get("roles")
    if not isinstance(roles, Mapping):
        return None
    rest = [(k, _canonical(v)) for k, v in roles.items() if k not in SUBJECT_ROLES]
    if not rest:
        return None
    return tuple(sorted((c for _k, c in rest), key=repr)) if len(rest) > 1 else rest[0][1]


def _display(value: Any) -> str:
    """Human-readable form of a canonical role value (detail strings only)."""
    if isinstance(value, tuple) and len(value) == 2 and value[0] in ("literal", "entity", "value"):
        return str(value[1])
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "list":
        return "／".join(_display(v) for v in value[1])
    return json_text(value)


def _literal_value(assertion: Mapping[str, Any]) -> Any:
    """Attribute value: only a *literal* role value counts (entities are relations)."""
    items = _role_items(assertion, VALUE_ROLES)
    vals = [c for _k, v in items for c in [_canonical(v)]
            if isinstance(c, tuple) and c and c[0] == "literal"]
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else tuple(sorted(vals, key=repr))


def json_text(value: Any) -> str:
    try:
        return canonical_json(value)
    except TypeError:
        return str(value)


def make_conflict_record(book_id: str, kind: str, left: str, right: str, detail: str) -> dict[str, Any]:
    """Stable conflict record for ``audit/conflicts.jsonl`` (§5.12)."""
    if kind not in CONFLICT_KINDS:
        raise ValueError("conflict.kind 不属闭集 %s：%r" % (CONFLICT_KINDS, kind))
    cid = make_conflict_id(book_id, kind, left, right, detail)
    return {"conflict_id": cid, "book_id": book_id, "kind": kind, "left": left,
            "right": right, "detail": detail, "status": "open", "decision": None,
            "resolved_by": None, "at": None}


# ---------------------------------------------------------------------------
# attribute conflicts
# ---------------------------------------------------------------------------
def detect_attribute_conflicts(assertions: Iterable[Any], *,
                               include_layers: Iterable[str] | None = None) -> tuple[dict[str, Any], ...]:
    """Same subject + predicate + claim layer, different literal value.

    ``statement`` and ``fact`` are compared in separate groups: a character's
    claim is not a world fact (design/01 §5.16), so a statement never collides
    with a fact here. Only ``status=asserted`` records participate.
    """
    groups: dict[tuple[str, Any, str, str], dict[str, list[str]]] = {}
    display_of: dict[tuple[tuple[str, Any, str, str], str], str] = {}
    for raw in assertions:
        rec = _dict(raw)
        if rec.get("status", "asserted") != "asserted":
            continue
        if rec.get("library") in RELATION_LIBRARIES:
            continue
        if include_layers is not None and rec.get("claim_layer", "fact") not in include_layers:
            continue
        subject = _subject(rec)
        value = _literal_value(rec)
        if subject is None or value is None:
            continue
        key = (str(rec.get("book_id")), subject, str(rec.get("predicate")),
               str(rec.get("claim_layer", "fact")))
        value_key = json_text(value)
        display_of[(key, value_key)] = _display(value)
        groups.setdefault(key, {}).setdefault(value_key, []).append(record_id(rec))

    conflicts: list[dict[str, Any]] = []
    for (book_id, subject, predicate, _layer), by_value in sorted(groups.items(), key=repr):
        if len(by_value) < 2:
            continue
        displays = sorted(by_value)
        for i in range(len(displays)):
            for j in range(i + 1, len(displays)):
                left = sorted(by_value[displays[i]])[0]
                right = sorted(by_value[displays[j]])[0]
                detail = "同一主体『%s』的谓词『%s』两值：%s / %s" % (
                    _display(subject[1] if isinstance(subject, tuple) else subject),
                    predicate,
                    display_of.get(((book_id, subject, predicate, _layer), displays[i]), displays[i]),
                    display_of.get(((book_id, subject, predicate, _layer), displays[j]), displays[j]))
                conflicts.append(make_conflict_record(book_id, "attribute",
                                                      *sorted((left, right)), detail))
    return tuple(conflicts)


# ---------------------------------------------------------------------------
# relation conflicts
# ---------------------------------------------------------------------------
def _relation_participants(assertion: Mapping[str, Any]) -> tuple[frozenset[str], tuple[tuple[str, str], ...]] | None:
    roles = assertion.get("roles")
    if not isinstance(roles, Mapping):
        return None
    entity_roles = [(str(role), str(value)) for role, value in roles.items()
                    if isinstance(value, str) and value.startswith(ENTITY_PREFIX)]
    if len(entity_roles) < 2:
        return None
    participants = frozenset(value for _role, value in entity_roles)
    ordered = tuple(sorted(entity_roles))
    return participants, ordered


def _assign_text(ordered: tuple[tuple[str, str], ...]) -> str:
    return "，".join("%s=%s" % (role, value) for role, value in ordered)


def detect_relation_conflicts(assertions: Iterable[Any], *,
                              exclusive_predicates: Iterable[frozenset[str]] = ()) -> tuple[dict[str, Any], ...]:
    """Same participant pair with reversed role order, or an exclusive predicate pair.

    ``exclusive_predicates`` is caller-supplied (e.g. ``{frozenset({"师兄", "师弟"})}``)
    because exclusivity is a lexicon fact, not something this module can guess.
    """
    exclusive = [frozenset(p) for p in exclusive_predicates]
    groups: dict[tuple[str, frozenset[str]], list[dict[str, Any]]] = {}
    for raw in assertions:
        rec = _dict(raw)
        if rec.get("status", "asserted") != "asserted":
            continue
        if rec.get("library") not in RELATION_LIBRARIES and \
                rec.get("predicate") not in {p for pair in exclusive for p in pair}:
            continue
        parsed = _relation_participants(rec)
        if parsed is None:
            continue
        participants, ordered = parsed
        if len(participants) != 2:
            continue
        groups.setdefault((str(rec.get("book_id")), participants), []).append({
            "id": record_id(rec), "predicate": str(rec.get("predicate")), "ordered": ordered})

    conflicts: list[dict[str, Any]] = []
    for (book_id, _participants), entries in sorted(groups.items(), key=repr):
        # (a) same predicate, reversed role assignment
        by_pred: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            by_pred.setdefault(entry["predicate"], []).append(entry)
        for predicate, same in by_pred.items():
            if len(same) < 2:
                continue
            for i in range(len(same)):
                for j in range(i + 1, len(same)):
                    if same[i]["ordered"] != same[j]["ordered"]:
                        detail = "关系『%s』方向相反：%s / %s" % (
                            predicate, _assign_text(same[i]["ordered"]),
                            _assign_text(same[j]["ordered"]))
                        conflicts.append(make_conflict_record(
                            book_id, "relation", *sorted((same[i]["id"], same[j]["id"])), detail))
        # (b) mutually exclusive predicates on the same pair
        for pair in exclusive:
            present = [e for e in entries if e["predicate"] in pair]
            preds = sorted({e["predicate"] for e in present})
            if len(preds) >= 2:
                first = present[0]["id"]
                second = next(e["id"] for e in present if e["predicate"] != present[0]["predicate"])
                detail = "同一对参与者出现互斥关系：%s" % " / ".join(preds)
                conflicts.append(make_conflict_record(
                    book_id, "relation", *sorted((first, second)), detail))
    return tuple(conflicts)


# ---------------------------------------------------------------------------
# timeline conflicts
# ---------------------------------------------------------------------------
def _unit_index(unit: str, unit_order: Mapping[str, int] | None) -> int | None:
    if unit_order is not None:
        return unit_order.get(unit)
    return None


def _unit_after(left: str, right: str, unit_order: Mapping[str, int] | None) -> bool:
    """Strict story order: explicit unit_order when given, otherwise lexical unit ids."""
    if unit_order is not None:
        li, ri = _unit_index(left, unit_order), _unit_index(right, unit_order)
        if li is None or ri is None:
            return False
        return li > ri
    return left > right


def timeline_participation_violations(events: Iterable[Any], *,
                                      entity_lifetimes: Mapping[str, Any],
                                      unit_order: Mapping[str, int] | None = None) -> tuple[dict[str, Any], ...]:
    """Dead-entity participation (§5.17 timeline example: 死亡后活动)."""
    conflicts: list[dict[str, Any]] = []
    for raw in events:
        event = _dict(raw)
        unit = event.get("unit_id")
        if not isinstance(unit, str) or not unit:
            continue
        for participant in event.get("participants") or []:
            if not isinstance(participant, str):
                continue
            lifetime = entity_lifetimes.get(participant)
            if lifetime is None:
                continue
            died = lifetime.get("died_at_unit") if isinstance(lifetime, Mapping) else str(lifetime)
            if not isinstance(died, str) or not died:
                continue
            if _unit_after(unit, died, unit_order):
                detail = "『%s』在 %s 已死亡，却在 %s 的活动 %s 中出场" % (
                    participant, died, unit, record_id(event))
                conflicts.append(make_conflict_record(
                    str(event.get("book_id")), "timeline",
                    *sorted((record_id(event), participant)), detail))
    return tuple(conflicts)


@dataclass(frozen=True)
class TimelineAudit:
    result: SolveResult
    conflicts: tuple[dict[str, Any], ...]
    quarantine: tuple[dict[str, Any], ...]


def audit_timeline(timeline_records: Iterable[Any], *,
                   known_events: Iterable[str] | None = None,
                   known_units: Iterable[str] | None = None,
                   nli: Any = None, verify: bool | Iterable[str] = False,
                   events: Iterable[Any] | None = None,
                   entity_lifetimes: Mapping[str, Any] | None = None,
                   unit_order: Mapping[str, int] | None = None) -> TimelineAudit:
    result = solve(timeline_records, known_events=known_events, known_units=known_units,
                   nli=nli, verify=verify)
    conflicts = [c.as_record() for c in result.conflicts]
    if events is not None and entity_lifetimes is not None:
        conflicts.extend(timeline_participation_violations(
            events, entity_lifetimes=entity_lifetimes, unit_order=unit_order))
    deduped = tuple(sorted({c["conflict_id"]: c for c in conflicts}.values(),
                           key=lambda c: c["conflict_id"]))
    return TimelineAudit(result=result, conflicts=deduped,
                         quarantine=tuple(q.as_record() for q in result.quarantine))


def detect_timeline_conflicts(timeline_records: Iterable[Any], **kwargs: Any) -> tuple[dict[str, Any], ...]:
    return audit_timeline(timeline_records, **kwargs).conflicts


# ---------------------------------------------------------------------------
# superseded closure (I17)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SupersedeReport:
    errors: tuple[Violation, ...]
    chains: Mapping[str, tuple[str, ...]]
    cycles: tuple[tuple[str, ...], ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _superseded_by(record: Mapping[str, Any]) -> str | None:
    target = record.get("superseded_by")
    return target if isinstance(target, str) and target else None


def check_supersede_chain(records: Iterable[Any]) -> SupersedeReport:
    """Validate ``superseded_by``: resolvable, closed, acyclic, kept forever (I17)."""
    by_id: dict[str, dict[str, Any]] = {}
    for raw in records:
        rec = _dict(raw)
        by_id[record_id(rec)] = rec
    errors: list[Violation] = []
    chains: dict[str, tuple[str, ...]] = {}
    cycles_found: set[tuple[str, ...]] = set()

    for rid, rec in by_id.items():
        target = _superseded_by(rec)
        status = rec.get("status")
        if status == "superseded" and target is None:
            errors.append(Violation("superseded_without_pointer",
                                    "%s: status=superseded 但没有 superseded_by（I17）" % rid,
                                    "superseded_by"))
            continue
        if target is None:
            continue
        if status != "superseded":
            errors.append(Violation("superseded_by_on_live_record",
                                    "%s: 未 superseded 却带 superseded_by=%s" % (rid, target),
                                    "superseded_by"))
        if target == rid:
            errors.append(Violation("superseded_self_reference", "%s 指向自己" % rid,
                                    "superseded_by"))
            continue
        if target not in by_id:
            errors.append(Violation("superseded_target_missing",
                                    "%s: superseded_by=%s 不存在（指针不闭合）" % (rid, target),
                                    "superseded_by"))
            continue
        # walk the chain
        path = [rid]
        seen = {rid}
        cursor = target
        while True:
            if cursor in seen:
                cycle = tuple(path[path.index(cursor):])
                cycles_found.add(cycle)
                errors.append(Violation("superseded_cycle",
                                        "%s 的 superseded_by 链成环：%s" % (
                                            rid, " -> ".join(cycle + (cursor,))),
                                        "superseded_by"))
                break
            path.append(cursor)
            seen.add(cursor)
            nxt = _superseded_by(by_id[cursor])
            if nxt is None:
                break
            if nxt not in by_id:
                errors.append(Violation("superseded_target_missing",
                                        "%s: 链上 %s 的 superseded_by=%s 不存在" % (rid, cursor, nxt),
                                        "superseded_by"))
                break
            cursor = nxt
        chains[rid] = tuple(path)
    return SupersedeReport(errors=tuple(errors), chains=chains,
                           cycles=tuple(sorted(cycles_found)))


def supersede_chains(records: Iterable[Any]) -> dict[str, tuple[str, ...]]:
    return dict(check_supersede_chain(records).chains)


# ---------------------------------------------------------------------------
# applying a supersede (retcon as a feature, §5.17)
# ---------------------------------------------------------------------------
def _assertion_view(record: Mapping[str, Any]) -> dict[str, Any]:
    subject = _subject(record)
    value = _value(record)
    return {
        "subject": _display(subject[1]) if subject else None,
        "predicate": record.get("predicate"),
        "value": _display(value) if value is not None else None,
        "claim_layer": record.get("claim_layer"),
        "evidence_units": sorted({str(e.get("unit_id")) for e in (record.get("evidence") or [])
                                  if isinstance(e, Mapping) and e.get("unit_id")}),
    }


@dataclass(frozen=True)
class SupersedePlan:
    records: tuple[dict[str, Any], ...]
    decision: dict[str, Any]
    retcon: dict[str, Any]


def apply_supersede(records: Iterable[Any], *, loser_id: str, winner_id: str,
                    decided_by: str, reason: str, at: str | None = None) -> SupersedePlan:
    """Mark ``loser_id`` superseded by ``winner_id``; never delete the old record.

    Returns copied records plus a ``decisions.jsonl`` row and a retcon history
    row (old value + new value + reason) for stage-2 "设定变更史".
    """
    if loser_id == winner_id:
        raise ValueError("loser 与 winner 不能是同一记录：%s" % loser_id)
    copied: list[dict[str, Any]] = []
    loser: dict[str, Any] | None = None
    winner: dict[str, Any] | None = None
    for raw in records:
        rec = _dict(raw)
        rid = record_id(rec)
        if rid == loser_id:
            rec["status"] = "superseded"
            rec["superseded_by"] = winner_id
            loser = rec
        elif rid == winner_id:
            derived = list(rec.get("derived_from") or [])
            if loser_id not in derived:
                derived.append(loser_id)
            rec["derived_from"] = derived
            winner = rec
        copied.append(rec)
    if loser is None:
        raise KeyError("找不到 loser 记录：%s" % loser_id)
    if winner is None:
        raise KeyError("找不到 winner 记录：%s" % winner_id)
    decision = {
        "decision_id": stable_id("d:", {"kind": "supersede", "subject": loser_id,
                                        "choice": "superseded", "winner": winner_id}),
        "kind": "supersede", "subject": loser_id, "choice": "superseded",
        "reason": reason, "decided_by": decided_by, "at": at,
        "winner": winner_id,
    }
    retcon = {"old_id": loser_id, "new_id": winner_id,
              "old": _assertion_view(loser), "new": _assertion_view(winner),
              "reason": reason, "decided_by": decided_by, "at": at}
    return SupersedePlan(records=tuple(copied), decision=decision, retcon=retcon)


def retcon_history(records: Iterable[Any], *,
                   decisions: Iterable[Any] = ()) -> tuple[dict[str, Any], ...]:
    """Render the stage-2 "设定变更史": old value -> new value + decision reason."""
    by_id = {record_id(_dict(r)): _dict(r) for r in records}
    decision_by_subject: dict[str, dict[str, Any]] = {}
    for raw in decisions:
        dec = _dict(raw)
        subject = dec.get("subject")
        if isinstance(subject, str):
            decision_by_subject[subject] = dec
    rows: list[dict[str, Any]] = []
    for rid, rec in by_id.items():
        target = _superseded_by(rec)
        if not target:
            continue
        new = by_id.get(target)
        dec = decision_by_subject.get(rid) or {}
        rows.append({
            "old_id": rid, "new_id": target,
            "old": _assertion_view(rec), "new": _assertion_view(new) if new else None,
            "reason": dec.get("reason"), "decided_by": dec.get("decided_by"),
            "at": dec.get("at"),
        })
    return tuple(sorted(rows, key=lambda r: (r["old_id"], r["new_id"])))


# ---------------------------------------------------------------------------
# conflict record validation
# ---------------------------------------------------------------------------
def validate_conflict(record: Any) -> tuple[Violation, ...]:
    """Closed-set/shape check for one ``audit/conflicts.jsonl`` row (§5.12)."""
    rec = _dict(record)
    violations: list[Violation] = []
    if rec.get("kind") not in CONFLICT_KINDS:
        violations.append(Violation("conflict_kind_unknown",
                                    "kind 不属闭集 %s：%r" % (CONFLICT_KINDS, rec.get("kind")),
                                    "kind"))
    if rec.get("status") not in CONFLICT_STATUSES:
        violations.append(Violation("conflict_status_unknown",
                                    "status 不属闭集 %s：%r"
                                    % (CONFLICT_STATUSES, rec.get("status")), "status"))
    decision = rec.get("decision")
    if decision is not None and decision not in DECISIONS:
        violations.append(Violation("conflict_decision_unknown",
                                    "decision 不属闭集 %s：%r" % (DECISIONS, decision), "decision"))
    if rec.get("status") == "resolved" and not rec.get("resolved_by"):
        violations.append(Violation("resolved_without_resolved_by",
                                    "status=resolved 必须写 resolved_by", "resolved_by"))
    for key in ("left", "right"):
        if not (isinstance(rec.get(key), str) and rec.get(key)):
            violations.append(Violation("conflict_endpoint_missing",
                                        "%s 必须是非空记录 id" % key, key))
    return tuple(violations)
