"""Cue (foreshadowing) state machine -- pure functions, no LLM.

Contract: box ``knowledge/cues.jsonl`` (design/01 §5.9/§5.18) + invariants
**I18** (state guards), **I19** (confidence method) and **I24** (``paid`` must be
preceded by ``triggered``; a payoff earlier than the trigger sets
``premature_payoff=true`` and goes to quarantine/conflict).

State graph (v2.2)::

    candidate --confirm--> active
    active --dormant(N)--> dormant --reinforce--> active
    active/dormant --trigger--> triggered
    triggered --payoff--> paid            # I18/I24: paid 前必须 triggered
    active/dormant/triggered --falsify--> falsified
    active/dormant/triggered --close--> unresolved
    active/dormant --payoff(early)--> (stays) + premature_payoff=true

Deliberate boundaries:

* ``predicate``: the state machine never lets an LLM write ``status``; only this
  module does (design/01 §5.18 last line).
* A payoff that arrives before ``triggered`` does **not** force the cue closed:
  it is flagged ``premature_payoff`` and quarantined as ``contradiction`` for
  review, because "提前回收" is a CFPG defect, not a state.
* ``dormant`` needs ``dormant_after_units`` (N comes from L1/volume config);
  there is no hidden default here.
* Functions are pure: inputs are never mutated, the returned ``cue`` is a deep
  copy; ``cue:`` ids are the canonical formula from :mod:`cbb_stage1.ids`
  (``ids.cue_id``); an existing **valid** ``cue_id`` is preserved, never
  renumbered.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..ids import cue_id as make_cue_id
from ..validate.confidence import validate_confidence

CUE_STATES = ("candidate", "active", "triggered", "dormant", "paid", "falsified", "unresolved")
CUE_KINDS = ("object", "line", "anomaly", "prophecy", "relationship")
CUE_EVENTS = ("confirm", "reinforce", "trigger", "payoff", "falsify", "dormant", "close")

#: box §4 canonical cue id form (16-64 hex chars).
_CUE_ID_RE = re.compile(r"^cue:[0-9a-f]{16,64}$")

# violation code -> box quarantine reason 闭集（§5.12）
_REASON_BY_CODE = {
    "schema": "schema",
    "cue_active_without_setup": "schema",
    "cue_trigger_incomplete": "schema",
    "cue_paid_without_payoff_evidence": "schema",
    "cue_paid_without_trigger": "contradiction",
    "cue_premature_payoff": "contradiction",
    "cue_premature_flag_missing": "contradiction",
    "cue_falsified_without_reason": "schema",
    "cue_dormant_without_since": "schema",
    "cue_dormant_threshold_missing": "schema",
    "cue_transition_not_allowed": "schema",
    "cue_unit_order_missing": "schema",
}


def _dict(record: Any) -> dict[str, Any]:
    """Accept plain mappings or pydantic records; always return a deep copy."""
    if hasattr(record, "model_dump"):
        return copy.deepcopy(record.model_dump())
    if isinstance(record, Mapping):
        return copy.deepcopy(dict(record))
    raise TypeError("cue 记录必须是 Mapping 或 pydantic 模型，现得 %r" % (type(record),))


def _sub(record: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    return value if isinstance(value, dict) else {}


def _evidence(record: Mapping[str, Any], key: str) -> list:
    value = _sub(record, key).get("evidence")
    return value if isinstance(value, list) else []


def cue_id_of(cue: Mapping[str, Any]) -> str:
    """Return the canonical ``cue:`` id, preserving an existing valid id.

    旧合法 ID 保留: a record already carrying a schema-valid ``cue_id`` is never
    renumbered (id stability across reruns, I7). Only a missing/invalid id is
    rebuilt with the unified formula ``ids.cue_id(book_id, kind, subject,
    setup_unit)`` -- the same formula the box normalizer uses.
    """
    cid = cue.get("cue_id")
    if isinstance(cid, str) and _CUE_ID_RE.match(cid):
        return cid
    subject = cue.get("subject")
    setup = cue.get("setup")
    setup_unit = setup.get("unit_id") if isinstance(setup, Mapping) else None
    return make_cue_id(
        str(cue.get("book_id") or ""),
        str(cue.get("kind") or "object"),
        dict(subject) if isinstance(subject, Mapping) else {},
        str(setup_unit) if isinstance(setup_unit, str) else None,
    )


@dataclass(frozen=True)
class CueViolation:
    code: str
    detail: str
    field: str | None = None

    @property
    def quarantine_reason(self) -> str:
        return _REASON_BY_CODE.get(self.code, "schema")

    def as_quarantine(self, item_id: str) -> dict[str, Any]:
        return {"item_id": item_id, "kind": "cue", "reason": self.quarantine_reason,
                "detail": "%s：%s" % (self.code, self.detail), "source_run": None}


@dataclass(frozen=True)
class CueTransition:
    cue_id: str
    event: str
    from_state: str
    to_state: str
    allowed: bool
    cue: Mapping[str, Any]
    violations: tuple[CueViolation, ...] = ()
    premature_payoff: bool = False

    def quarantine_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(v.as_quarantine(self.cue_id) for v in self.violations if v.code != "schema")


def _transition(cue: Mapping[str, Any], event: str, from_state: str, to_state: str,
                allowed: bool, violations: Iterable[CueViolation] = (),
                premature_payoff: bool = False) -> CueTransition:
    return CueTransition(cue_id=cue_id_of(cue), event=event, from_state=from_state,
                         to_state=to_state, allowed=allowed, cue=cue,
                         violations=tuple(violations),
                         premature_payoff=bool(premature_payoff or cue.get("premature_payoff")))


def can_transition(cue: Any, event: str, **kwargs: Any) -> bool:
    return apply_event(cue, event, **kwargs).allowed


def apply_event(cue: Any, event: str, *, unit_id: str | None = None,
                unit_order: Mapping[str, int] | None = None,
                dormant_after_units: int | None = None,
                trigger_condition: str | None = None,
                reason: str | None = None,
                evidence: Iterable[Any] | None = None,
                last_seen_unit: str | None = None) -> CueTransition:
    """Apply one state-machine event; returns a transition + a deep-copied cue.

    ``allowed=False`` means the record must not be mutated; the caller should
    write ``transition.quarantine_records()`` to ``audit/quarantine.jsonl``.
    ``premature_payoff`` is reported on the cue even when the transition is
    rejected (I24).
    """
    if event not in CUE_EVENTS:
        raise ValueError("未知 cue 事件 %r（闭集 %s）" % (event, CUE_EVENTS))
    base = _dict(cue)
    base["cue_id"] = cue_id_of(base)   # 既有合法 id 原样保留；缺号/非法用统一公式
    state = base.get("status")
    if state not in CUE_STATES:
        v = CueViolation("schema", "status 不属闭集 %s：%r" % (CUE_STATES, state), "status")
        return _transition(base, event, str(state), str(state), False, [v])
    if base.get("kind") not in CUE_KINDS:
        v = CueViolation("schema", "kind 不属闭集 %s：%r" % (CUE_KINDS, base.get("kind")), "kind")
        return _transition(base, event, state, state, False, [v])

    new: dict[str, Any] = base
    violations: list[CueViolation] = []
    premature = bool(base.get("premature_payoff"))
    ev = list(evidence) if evidence is not None else None

    if event == "confirm":
        if state != "candidate":
            violations.append(CueViolation("cue_transition_not_allowed",
                                           "confirm 只能从 candidate 出发（现 %s）" % state))
        elif not _evidence(base, "setup"):
            violations.append(CueViolation("cue_active_without_setup",
                                           "active 需要非空 setup.evidence（I18）", "setup.evidence"))
        else:
            new["status"] = "active"
            new["activated_at"] = base.get("activated_at") or unit_id or \
                _sub(base, "setup").get("unit_id")

    elif event == "reinforce":
        reinforce = ev if ev is not None else list(base.get("reinforce") or [])
        if state == "dormant":
            if not reinforce:
                violations.append(CueViolation("cue_transition_not_allowed",
                                               "dormant→active 需要 reinforce 证据（§5.18）"))
            else:
                new["status"] = "active"
                new["reinforce"] = list(base.get("reinforce") or []) + (
                    [e for e in ev] if ev is not None else [])
                new["dormant_since"] = None
        elif state in ("active", "triggered"):
            if ev:
                new["reinforce"] = list(base.get("reinforce") or []) + ev
        else:
            violations.append(CueViolation("cue_transition_not_allowed",
                                           "reinforce 不能从 %s 出发（先 confirm）" % state))

    elif event == "trigger":
        trigger = _sub(base, "trigger")
        condition = trigger_condition if trigger_condition is not None else trigger.get("condition")
        trig_evidence = trigger.get("evidence") or []
        if state in ("active", "dormant"):
            if not (isinstance(condition, str) and condition.strip()) and not trig_evidence:
                violations.append(CueViolation(
                    "cue_trigger_incomplete",
                    "triggered 需要 trigger.condition 或 trigger.evidence（I18）", "trigger"))
            else:
                new["status"] = "triggered"
                new["triggered_at"] = base.get("triggered_at") or unit_id or trigger.get("unit_id")
                if isinstance(condition, str) and condition.strip():
                    new.setdefault("trigger", {})
                    if not isinstance(new.get("trigger"), dict):
                        new["trigger"] = {}
                    new["trigger"]["condition"] = condition
        elif state == "triggered":
            pass  # idempotent
        else:
            violations.append(CueViolation("cue_transition_not_allowed",
                                           "trigger 不能从 %s 出发" % state))

    elif event == "payoff":
        payoff = _sub(base, "payoff")
        payoff_evidence = ev if ev is not None else payoff.get("evidence")
        if not payoff_evidence:
            violations.append(CueViolation("cue_paid_without_payoff_evidence",
                                           "paid 需要非空 payoff.evidence（I18）", "payoff.evidence"))
        elif state != "triggered":
            if state in ("active", "dormant"):
                premature = True
                violations.append(CueViolation(
                    "cue_paid_without_trigger",
                    "payoff 到在 %s：paid 前必须 triggered（I18/I24）" % state, "status"))
                violations.append(CueViolation(
                    "cue_premature_payoff",
                    "回收早于触发：premature_payoff=true 并进隔离/复核（CFPG 提前回收）",
                    "premature_payoff"))
                new["premature_payoff"] = True
            else:
                violations.append(CueViolation("cue_transition_not_allowed",
                                               "payoff 不能从 %s 出发" % state))
        else:
            trigger_unit = _sub(base, "trigger").get("unit_id") or base.get("triggered_at")
            payoff_unit = unit_id or payoff.get("unit_id")
            if (
                unit_order
                and trigger_unit is not None
                and payoff_unit is not None
                and trigger_unit in unit_order
                and payoff_unit in unit_order
                and unit_order[str(payoff_unit)] < unit_order[str(trigger_unit)]
            ):
                premature = True
                violations.append(CueViolation(
                    "cue_premature_payoff",
                    "payoff 单元 %s 早于 trigger 单元 %s（I24）" % (payoff_unit, trigger_unit),
                    "premature_payoff"))
                new["premature_payoff"] = True
            else:
                new["status"] = "paid"

    elif event == "falsify":
        if state in ("active", "dormant", "triggered", "candidate"):
            if not (isinstance(reason, str) and reason.strip()):
                violations.append(CueViolation("cue_falsified_without_reason",
                                               "falsified 必须写 falsified_reason（I18）",
                                               "falsified_reason"))
            else:
                new["status"] = "falsified"
                new["falsified_reason"] = reason.strip()
        else:
            violations.append(CueViolation("cue_transition_not_allowed",
                                           "falsify 不能从 %s 出发" % state))

    elif event == "dormant":
        if state != "active":
            violations.append(CueViolation("cue_transition_not_allowed",
                                           "dormant 只能从 active 出发（现 %s）" % state))
        elif not isinstance(dormant_after_units, int) or dormant_after_units < 1:
            violations.append(CueViolation(
                "cue_dormant_threshold_missing",
                "dormant 需要 dormant_after_units（N 由 L1/卷配置给，本模块不设默认值）"))
        elif not unit_order or not unit_id or unit_id not in unit_order:
            violations.append(CueViolation("cue_unit_order_missing",
                                           "dormant 判定需要 unit_order 与当前 unit_id"))
        else:
            seen = last_seen_unit or base.get("activated_at") or base.get("triggered_at") \
                or _sub(base, "setup").get("unit_id")
            if seen not in unit_order:
                violations.append(CueViolation("cue_unit_order_missing",
                                               "上一出现单元 %r 不在 unit_order 里" % (seen,)))
            elif unit_order[str(unit_id)] - unit_order[str(seen)] >= dormant_after_units:
                new["status"] = "dormant"
                new["dormant_since"] = unit_id

    elif event == "close":
        if state in ("active", "dormant", "triggered", "candidate"):
            new["status"] = "unresolved"
        elif state in ("paid", "falsified", "unresolved"):
            pass  # idempotent; 不强行闭环

    if violations:
        new["premature_payoff"] = bool(premature or new.get("premature_payoff"))
        return _transition(new, event, state, str(new.get("status")), False, violations,
                           premature_payoff=premature)
    return _transition(new, event, state, str(new.get("status")), True,
                       premature_payoff=premature)


def validate_cue(cue: Any, *, unit_order: Mapping[str, int] | None = None) -> tuple[CueViolation, ...]:
    """Invariant check for one stored cue record (read-only; I18/I19/I24)."""
    record = _dict(cue)
    violations: list[CueViolation] = []
    state = record.get("status")
    if state not in CUE_STATES:
        violations.append(CueViolation("schema", "status 不属闭集 %s：%r" % (CUE_STATES, state),
                                       "status"))
        return tuple(violations)
    if record.get("kind") not in CUE_KINDS:
        violations.append(CueViolation("schema", "kind 不属闭集 %s：%r" % (CUE_KINDS,
                                                                         record.get("kind")), "kind"))
    trigger = _sub(record, "trigger")
    if state in ("active", "triggered", "paid"):
        if state == "active" and not _evidence(record, "setup"):
            violations.append(CueViolation("cue_active_without_setup",
                                           "active 但 setup.evidence 为空（I18）",
                                           "setup.evidence"))
        if state in ("triggered", "paid") and not (
                (isinstance(trigger.get("condition"), str) and trigger["condition"].strip())
                or trigger.get("evidence")):
            violations.append(CueViolation("cue_trigger_incomplete",
                                           "triggered/paid 但 trigger 既无 condition 也无 evidence"
                                           "（I18）", "trigger"))
    if state == "paid":
        if not _evidence(record, "payoff"):
            violations.append(CueViolation("cue_paid_without_payoff_evidence",
                                           "paid 但 payoff.evidence 为空（I18）",
                                           "payoff.evidence"))
        if record.get("triggered_at") is None:
            violations.append(CueViolation("cue_paid_without_trigger",
                                           "paid 但 triggered_at 为空＝未 triggered（I18/I24）",
                                           "triggered_at"))
    if state == "dormant" and not (
            isinstance(record.get("dormant_since"), str) and record["dormant_since"]):
        violations.append(CueViolation("cue_dormant_without_since",
                                       "dormant 但 dormant_since 为空（I18）", "dormant_since"))
    if state == "falsified" and not (
            isinstance(record.get("falsified_reason"), str) and record["falsified_reason"].strip()):
        violations.append(CueViolation("cue_falsified_without_reason",
                                       "falsified 但 falsified_reason 为空（I18）",
                                       "falsified_reason"))
    # I24: payoff evidence present but trigger missing/later => premature flag must be set
    payoff_evidence = _evidence(record, "payoff")
    payoff_unit = _sub(record, "payoff").get("unit_id")
    trigger_unit = trigger.get("unit_id") or record.get("triggered_at")
    if payoff_evidence:
        early = record.get("triggered_at") is None
        if (
            not early
            and unit_order
            and payoff_unit is not None
            and trigger_unit is not None
            and payoff_unit in unit_order
            and trigger_unit in unit_order
        ):
            early = unit_order[str(payoff_unit)] < unit_order[str(trigger_unit)]
        if early and not record.get("premature_payoff"):
            violations.append(CueViolation(
                "cue_premature_flag_missing",
                "payoff 早于 trigger 或未 triggered，但 premature_payoff 未标（I24）",
                "premature_payoff"))
    # I19 置信：只校验置信位，不产出置信
    for v in validate_confidence(record):
        violations.append(CueViolation(v.code, v.detail, v.field))
    return tuple(violations)


def transition_cue(cue: Any, event: str, **kwargs: Any) -> dict[str, Any]:
    """Convenience: return only the (possibly unchanged) cue copy."""
    return dict(apply_event(cue, event, **kwargs).cue)
