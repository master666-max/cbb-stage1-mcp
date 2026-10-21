"""P2-B validate integration: ``work/audit/{conflicts,decisions,quarantine}.jsonl``.

Unified postprocess entry point::

    run(work_root, *, book_id, outputs, units, created_at, upstream=None) -> dict

This is the **single writer** of the audit tables for P3. It reads the work
tables produced by the earlier postprocess steps, runs the pure validators, and
only writes audit artifacts (never the box). Events follow the same priority as
the temporal step: ``work/assemble/events.jsonl`` (W3's producer) wins, then
``work/resolve/events.jsonl``, and engine ``outputs`` are only a standalone
fallback -- an existing (even empty) assemble table must not be silently
bypassed.

* attribute / relation conflicts from ``work/assemble/assertions.jsonl``;
* timeline cycles / during-overlap contradictions from
  ``work/assemble/timeline.jsonl`` (via :func:`cbb_stage1.validate.conflicts.audit_timeline`)
  **plus** the timeline conflicts returned by the temporal step
  (``upstream["temporal"]["conflicts"]``) -- a cycle is quarantined out of the
  timeline table, so validate must take it from the upstream handoff;
* dead-entity participation when ``upstream["entity_lifetimes"]`` is supplied;
* confidence-method violations (I19) and cue guards/premature payoff (I18/I24);
* ``superseded_by`` closure/cycle errors over assertions + events (I17);
* ``audit/decisions.jsonl`` is preserved as-is (human/Arbiter decisions are not
  produced here); when absent an **empty explicit table** is written. No conflict
  is auto-resolved (design/01 §5.17: 不删除、不自动覆盖).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..temporal.integration import (
    append_jsonl,
    first_table,
    from_outputs,
    outputs_list,
    read_jsonl,
    unit_order_from,
    write_jsonl,
)
from ..ids import conflict_id as make_conflict_id
from ..ids import quarantine_id
from .confidence import (
    VERIFICATION_STRENGTH,
    inference_allowed,
    validate_assertion_record,
    validate_confidence,
    verification_level,
)
from cbb_stage1.confidence import score as confidence_score
from .speaker_verify import verify_speaker_quote
from .conflicts import (
    CONFLICT_KINDS,
    QUARANTINE_REASONS,
    audit_timeline,
    check_supersede_chain,
    detect_attribute_conflicts,
    detect_relation_conflicts,
    detect_timeline_conflicts,
    retcon_history,
    supersede_chains,
    record_id,
)

STEP = "validate"

ASSERTION_TABLES = ("assemble/assertions.jsonl", "resolve/assertions.jsonl")
EVENT_TABLES = ("assemble/events.jsonl", "resolve/events.jsonl")
TIMELINE_TABLES = ("assemble/timeline.jsonl",)
CUE_TABLES = ("assemble/cues.jsonl",)

CONFLICTS_TABLE = "audit/conflicts.jsonl"
DECISIONS_TABLE = "audit/decisions.jsonl"
QUARANTINE_TABLE = "audit/quarantine.jsonl"

_RID_RE = re.compile(r"^([a-z]{1,3}:[0-9a-f]{8,64})")

#: confidence violation -> box quarantine reason (§5.12 closed set)
_CONFIDENCE_REASON = {
    "confidence_method_missing": "low_confidence",
    "confidence_method_unknown": "low_confidence",
    "confidence_out_of_range": "low_confidence",
    "calibration_version_missing": "low_confidence",
}

#: superseded-chain violation -> box quarantine reason
_SUPERSEDE_REASON = {
    "superseded_without_pointer": "schema",
    "superseded_by_on_live_record": "schema",
    "superseded_self_reference": "schema",
    "superseded_target_missing": "ref_missing",
    "superseded_cycle": "contradiction",
}

#: cue violation code -> box quarantine reason (mirrors cues.state mapping)
_CUE_REASON = {
    "cue_paid_without_trigger": "contradiction",
    "cue_premature_payoff": "contradiction",
    "cue_premature_flag_missing": "contradiction",
}


def _record_id(record: Any) -> str:
    try:
        return record_id(record)
    except (TypeError, ValueError):
        # 无 id 记录不进冲突/超替图；隔离 id 用统一 q: 公式（旧 stable_id("q:", ...) 形）
        return quarantine_id({"record": record})


def _has_record_id(record: Any) -> bool:
    try:
        return bool(record_id(record))
    except (TypeError, ValueError):
        return False


def _rid_from_detail(detail: str) -> str | None:
    match = _RID_RE.match(str(detail or ""))
    return match.group(1) if match else None


def _qrow(item_id: str, kind: str, reason: str, detail: str) -> dict[str, Any]:
    reason = reason if reason in QUARANTINE_REASONS else "schema"
    return {"item_id": item_id, "kind": kind, "reason": reason,
            "detail": detail, "source_run": STEP}


def _cue_quarantine(cues: list[dict[str, Any]],
                    unit_order: Mapping[str, int] | None,
                    upstream_cues: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Cue guards + premature payoff; validate_cue is imported lazily (no cycle)."""
    from ..cues.state import validate_cue  # local import: cues -> validate.confidence

    rows: list[dict[str, Any]] = []
    for cue in cues:
        item_id = _record_id(cue)
        if cue.get("premature_payoff"):
            rows.append(_qrow(str(item_id), "cue", "contradiction",
                              "premature_payoff=true：回收早于触发/未 triggered（I24）"))
        for violation in validate_cue(cue, unit_order=unit_order or None):
            rows.append(_qrow(str(item_id), "cue", violation.quarantine_reason,
                              "%s：%s" % (violation.code, violation.detail)))
    upstream_violations = []
    if isinstance(upstream_cues, Mapping):
        value = upstream_cues.get("violations")
        if isinstance(value, (list, tuple)):
            upstream_violations = [item for item in value if isinstance(item, Mapping)]
    for violation in upstream_violations:
        code = str(violation.get("code") or "")
        reason = _CUE_REASON.get(code, "schema")
        rows.append(_qrow(str(violation.get("cue_id") or ""), "cue", reason,
                          "%s：%s" % (code, violation.get("detail") or "")))
    return rows


def _entity_tiers(root: Path) -> dict[str, str]:
    """entity_id -> tier, from the identity table.

    The tier decides how hard an assertion is verified (§5.15), so it has to come
    from the entity the assertion is about, not from the assertion itself.
    """
    path = root / "identity" / "entities.jsonl"
    tiers: dict[str, str] = {}
    if not path.exists():
        return tiers
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping):
            continue
        entity_id = str(row.get("entity_id") or "")
        tier = str(row.get("tier") or "minor")
        if entity_id:
            tiers[entity_id] = tier if tier in ("canonical", "major", "minor") else "minor"
    return tiers


def _assertion_tier(assertion: Mapping[str, Any], tiers: Mapping[str, str]) -> str:
    """The strictest tier among the entities the assertion refers to.

    Using the strictest is deliberate: an assertion about the protagonist and a
    walk-on character still concerns the protagonist, so it should not be verified
    at the walk-on's level.
    """
    ranks = {"minor": 0, "major": 1, "canonical": 2}
    best = "minor"
    roles = assertion.get("roles")
    values: list[Any] = []
    if isinstance(roles, Mapping):
        values = list(roles.values())
    elif isinstance(roles, (list, tuple)):
        values = list(roles)
    for value in values:
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for item in candidates:
            text = str(item or "")
            if not text.startswith("e:"):
                continue
            tier = tiers.get(text)
            if tier and ranks.get(tier, 0) > ranks.get(best, 0):
                best = tier
    return best


#: Default samples for semantic entropy when the policy does not say.
#:
#: Entropy needs >= 2 samples; the paper uses 5-10. 4 keeps a whole-book run
#: affordable while still measuring something, and it is the same default
#: ``ModelPolicy`` carries.
DEFAULT_CONFIDENCE_SAMPLES = 4


def _confidence_samples(up: Mapping[str, Any]) -> int:
    """How many samples the confidence methods may draw per assertion."""
    policy = up.get("policy")
    raw = None
    if isinstance(policy, Mapping):
        raw = policy.get("confidence_samples")
    if raw is None:
        config = up.get("config")
        if isinstance(config, Mapping):
            raw = config.get("confidence_samples")
        elif config is not None:
            raw = getattr(config, "confidence_samples", None)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE_SAMPLES
    return value if value >= 2 else DEFAULT_CONFIDENCE_SAMPLES


def _speaker_asker(up: Mapping[str, Any]) -> tuple[Any, bool]:
    """Return ``(llm_with_generate, same_model)`` for the speaker check.

    Two shapes arrive here and only one of them can answer a text question:

    * ``EndpointSubagent`` — has ``run(SubagentTask)``, not ``generate``. The codebase
      already ships the adapter for this (`agents.subagent.SubagentLLM`, whose own
      docstring says it exists so "existing generate/sample call sites use a
      subagent"); measured failure before using it: every check recorded
      ``AttributeError: 'EndpointSubagent' object has no attribute 'generate'``.
    * ``SyncLLM`` / ``VerifierLLM`` — already has ``generate``.

    Falls back to ``upstream["llm"]`` when the verifier cannot be adapted. Falling back
    means the check runs on the extractor's own model, so ``same_model`` is reported
    True and §5.15 independence is not claimed.
    """
    verifier = up.get("verifier_llm")
    if verifier is not None:
        if callable(getattr(verifier, "generate", None)):
            return verifier, False
        if callable(getattr(verifier, "run", None)):
            try:
                from cbb_stage1.agents.subagent import SubagentLLM

                return SubagentLLM(verifier), False
            except Exception:  # noqa: BLE001 - fall through to the aux model
                pass

    llm = up.get("llm")
    if llm is not None and callable(getattr(llm, "generate", None)):
        return llm, True
    return None, bool(up.get("verifier_same_model"))


def _unit_text_index(root: Path) -> dict[str, str]:
    """Unit id -> its text.

    The layout is by VOLUME and ORDER — ``plan/texts/v01/u001.txt`` — not by unit id, so
    the mapping has to come from ``plan/units.jsonl``. Globbing for ``*.txt`` finds
    nothing because the files sit one directory down, which is how the first version of
    this helper produced an empty context: measured, the prompt reached the model with
    ``原文上下文：`` followed by nothing, so the verifier was asked to attribute a quote
    it could not see the surroundings of.
    """
    index: dict[str, str] = {}
    units_path = root / "plan" / "units.jsonl"
    if not units_path.exists():
        return index
    texts_dir = root / "plan" / "texts"
    for line in units_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            unit = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(unit, Mapping):
            continue
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        try:
            volume = int(unit.get("volume") or 1)
            order = int(unit.get("order") or 0)
        except (TypeError, ValueError):
            continue
        path = texts_dir / f"v{volume:02d}" / f"u{order:03d}.txt"
        if not path.exists():
            continue
        try:
            index[unit_id] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return index


def _quote_context(text: str, quote: str, window: int = 200) -> str:
    """The quote plus a little surrounding text, so the verifier can see attribution."""
    if not text or not quote:
        return ""
    index = text.find(quote)
    if index < 0:
        return ""
    start = max(0, index - window)
    end = min(len(text), index + len(quote) + window)
    return text[start:end]


def _speaker_cast(root: Path) -> set[str]:
    """Every resolved entity name and alias — used only to short-circuit a check.

    Read inline rather than through a shared helper: the only `read_jsonl` in the tree
    lives in `temporal.integration`, and reaching into another stage's module for a
    four-line loop is how cross-stage coupling starts.
    """
    cast: set[str] = set()
    for relative in ("identity/entities.jsonl", "resolve/entity_candidates.jsonl"):
        path = root / relative
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("canonical_name") or "")
            if name:
                cast.add(name)
            for alias in row.get("aliases") or ():
                alias_text = str(alias or "")
                if alias_text:
                    cast.add(alias_text)
    return cast


def run(work_root: str | Path, *, book_id: str, outputs: Any, units: Any,
        created_at: str, upstream: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Detect conflicts and write the audit work tables."""
    root = Path(work_root)
    up = dict(upstream or {})
    output_rows = outputs_list(outputs)

    assertions, assertion_path = first_table(root, ASSERTION_TABLES)
    if assertion_path is None:
        assertions = from_outputs(output_rows, "assertions")
    events, event_path = first_table(root, EVENT_TABLES)
    if event_path is not None:
        try:
            events_source = ("assemble"
                             if event_path.relative_to(root).as_posix().startswith("assemble/")
                             else "resolve")
        except ValueError:  # pragma: no cover - defensive
            events_source = "work"
    elif any("events" in output for output in output_rows):
        events = from_outputs(output_rows, "events")
        events_source = "outputs"
    else:
        events_source = "none"
    timeline, timeline_path = first_table(root, TIMELINE_TABLES)
    cues, cue_path = first_table(root, CUE_TABLES)
    unit_order = dict(up.get("unit_order") or {}) or unit_order_from(units)

    # Postprocess normally runs after ``assemble``, so ids exist. A standalone
    # run over raw engine outputs may still carry id-less records; those cannot
    # participate in the conflict/supersede graph (``record_id`` raises), so they
    # are skipped there and only keep their confidence quarantine row.
    assertions_with_ids = [record for record in assertions if _has_record_id(record)]
    events_with_ids = [record for record in events if _has_record_id(record)]
    idless = (len(assertions) - len(assertions_with_ids)) + \
        (len(events) - len(events_with_ids))

    # ---- conflicts (three classes) --------------------------------------
    conflicts: list[dict[str, Any]] = []
    conflicts.extend(detect_attribute_conflicts(assertions_with_ids))
    # D9 requires three kinds of conflict audit: attribute, relation, TIMELINE.
    # Only the first two were wired, so ``conflicts_by_kind.timeline`` was always
    # 0 by construction — a green reading that measured nothing.
    if timeline:
        conflicts.extend(detect_timeline_conflicts(timeline))
    conflicts.extend(detect_relation_conflicts(
        assertions_with_ids, exclusive_predicates=up.get("exclusive_predicates") or ()))
    event_ids = {str(event.get("event_id")) for event in events_with_ids
                 if event.get("event_id")}
    # An existing work table (even empty) defines the event universe; ``None``
    # only when there is no source at all (standalone, refs not checkable).
    known_events = event_ids if events_source != "none" else None
    audit = audit_timeline(
        timeline,
        known_events=known_events,
        events=events_with_ids or None,
        entity_lifetimes=up.get("entity_lifetimes"),
        unit_order=unit_order or None,
    )
    conflicts.extend(audit.conflicts)
    temporal_up = up.get("temporal")
    if isinstance(temporal_up, Mapping):
        for row in temporal_up.get("conflicts") or []:
            if isinstance(row, Mapping):
                conflicts.append(dict(row))
    for row in up.get("timeline_conflicts") or []:
        if isinstance(row, Mapping):
            conflicts.append(dict(row))
    unique_conflicts: dict[str, dict[str, Any]] = {}
    for row in conflicts:
        if row.get("kind") not in CONFLICT_KINDS:
            continue
        key = str(row.get("conflict_id") or make_conflict_id(
            str(row.get("book_id") or ""),
            str(row.get("kind") or ""),
            str(row.get("left") or ""),
            str(row.get("right") or ""),
            str(row.get("detail") or "")))
        row.setdefault("conflict_id", key)
        unique_conflicts[key] = dict(row)
    conflict_rows = [unique_conflicts[key] for key in sorted(unique_conflicts)]

    # ---- quarantine (append; never rewrite) -----------------------------
    quarantine: list[dict[str, Any]] = [dict(row) for row in audit.quarantine]
    for assertion in assertions:
        item_id = _record_id(assertion)
        for violation in validate_confidence(assertion):
            quarantine.append(_qrow(str(item_id), "assertion",
                                    _CONFIDENCE_REASON.get(violation.code, "schema"),
                                    "%s：%s" % (violation.code, violation.detail)))
    quarantine.extend(_cue_quarantine(cues, unit_order or None,
                                      up.get("cues") if isinstance(up.get("cues"), Mapping) else None))
    # a record without an id (e.g. an engine event that was not assembled yet)
    # cannot participate in the supersede graph; it is not a validation error here.
    supersede_candidates = [record for record in assertions + events
                            if _has_record_id(record)]
    report = check_supersede_chain(supersede_candidates)
    for violation in report.errors:
        item_id = _rid_from_detail(violation.detail) or quarantine_id(
            {"code": violation.code, "detail": violation.detail})
        quarantine.append(_qrow(item_id, "assertion",
                                _SUPERSEDE_REASON.get(violation.code, "schema"),
                                "%s：%s" % (violation.code, violation.detail)))
    quarantine_added = append_jsonl(
        root / QUARANTINE_TABLE, quarantine,
        key_fields=("item_id", "kind", "reason", "detail"))
    quarantine_path = root / QUARANTINE_TABLE

    # ---- decisions: preserve, never auto-resolve ------------------------
    decisions_path = root / DECISIONS_TABLE
    decisions = read_jsonl(decisions_path)
    write_jsonl(decisions_path, decisions)   # explicit table even when empty

    conflicts_path = write_jsonl(root / CONFLICTS_TABLE, conflict_rows)

    by_kind = {kind: sum(1 for row in conflict_rows if row.get("kind") == kind)
               for kind in CONFLICT_KINDS}
    notes: list[str] = []
    if idless:
        notes.append("%d 条无 id 记录跳过冲突/超替图检测（装配后才有 id；置信项照查）" % idless)
    if not conflict_rows:
        notes.append("无冲突：attribute/relation/timeline 三类均 0（空表即真数）")
    if not timeline and timeline_path is None:
        notes.append("读不到 work/assemble/timeline.jsonl：timeline 类冲突只依赖上游交接")
    if event_path is not None:
        notes.append("读事件表 %s（%d 条；work 表优先，engine outputs 只 fallback）"
                     % (event_path.relative_to(root), len(events)))
    elif events:
        notes.append("无 work events 表：fallback 到 engine outputs 的 %d 条事件" % len(events))
    else:
        notes.append("无事件表：死亡后活动校验跳过（不许猜）")
    if not unit_order:
        notes.append("无 unit order：payoff/事件先后无法复算")
    if up.get("entity_lifetimes") is None:
        notes.append("上游未给 entity_lifetimes：死亡后活动校验未启用")
    notes.append("decisions 本轮不自动裁决（人工/Arbiter 决定后才写）")

    # Confidence must actually be computed, not inherited.
    #
    # The guard used to be "skip when confidence_method is already set", and the
    # normalizer always sets it to `engine_self` — so every assertion skipped the
    # scoring call and the whole B2/B4/B5 surface was dead code. Measured on a real
    # run: the validate receipt carried no `confidence_status` field at all,
    # which is only possible when the loop body never executed.
    #
    # `engine_self` is the extractor's *self-report*, explicitly the weakest
    # method (01-箱规格 §5.15). It is a fallback, not a result: a stronger method
    # should replace it whenever one is available. So the guard now only skips
    # methods that are already stronger than self-report.
    confidence_status: dict[str, int] = {}
    upgraded = 0
    for assertion in assertions:
        current = str(assertion.get("confidence_method") or "")
        if current and current != "engine_self":
            continue
        # ``confidence_method`` from upstream lets the caller pin the method
        # instead of relying on auto-selection. B4 (UQLM) has three classes and
        # only ``blackbox`` works without white-box access, so a run that wants
        # UQLM rather than semantic entropy must be able to say so; without this
        # the auto priority always lands on ``semantic_entropy``.
        # ``judge_llm`` is a separate model (B4 judge class). When present it is
        # preferred over the sampling methods because §5.15 wants the judgement to
        # come from a different model than the extractor — which is also what makes
        # ``model_decoupled`` satisfiable under the agent-driven transport.
        reading = confidence_score(
            assertion,
            llm=up.get("llm"),
            config=up.get("config"),
            method=up.get("confidence_method") or None,
            judge_llm=up.get("judge_llm"),
            # Sampling cost lives here. Every assertion costs this many model calls;
            # measured on a real run the default of 8 made restatement 60% of all
            # requests. ``confidence_samples`` comes from model-policy.json so the
            # cost is visible and adjustable instead of buried in a default.
            **({"samples": _confidence_samples(up)} if _confidence_samples(up) else {}),
        )
        confidence_status[reading.status] = confidence_status.get(reading.status, 0) + 1
        if reading.status == "ok" and reading.method and reading.method != "engine_self":
            assertion["confidence_method"] = reading.method
            if reading.confidence is not None:
                assertion["confidence"] = reading.confidence
            upgraded += 1
    # §5.15 verification matrix: ``verification.level`` is DERIVED from the
    # assertion subject's entity tier. The tables and the validator already
    # existed in ``validate.confidence`` but nothing called them, so every box row
    # carried ``verification = null`` — 7/7 measured — and the box could not say
    # how strongly any given assertion had been checked.
    tiers_by_entity = _entity_tiers(root)
    stamped = 0
    for assertion in assertions:
        tier = _assertion_tier(assertion, tiers_by_entity)
        level = verification_level(tier)
        if not level:
            continue
        current = assertion.get("verification")
        if isinstance(current, dict) and current.get("level") == level:
            continue
        record = dict(current) if isinstance(current, dict) else {}
        record["level"] = level
        record.setdefault("verdict", "none")
        record.setdefault("tier", tier)
        if record.get("verifier_model"):
            record.setdefault("verifier", "verifier")
        record["inference_scope"] = inference_allowed(tier) or "off"
        # What the level means in practice, kept beside it so a reader does not have
        # to look up the matrix. The level itself stays in the schema's closed set.
        record["strength"] = VERIFICATION_STRENGTH.get(tier, "schema_only")
        # §5.15 requires the verification to be auditable: who verified, with which
        # model. The per-record validator caught this field being absent right after
        # the stamping was wired, which is exactly what it is for.
        record.setdefault("verifier", "validate:level-stamp")
        record.setdefault(
            "model", str(up.get("verifier_model") or up.get("model_label") or "unrecorded")
        )
        assertion["verification"] = record
        stamped += 1
    if stamped:
        notes.append(
            "§5.15 验证分级：%d 条断言按 tier 标注 verification.level（level=tier，"
            "strength 见 verification.strength）" % stamped
        )
        if assertion_path is not None:
            write_jsonl(assertion_path, assertions)

    if upgraded:
        notes.append("confidence 升级：%d 条断言从 engine_self 换成可复算的方法" % upgraded)
        # Persist the upgrade. The computed score used to live only in this
        # function's local list: validate declared just three audit outputs, so
        # the strengthened ``confidence_method`` never reached the work table and
        # the box kept reporting ``engine_self`` even though semantic entropy had
        # actually run (measured: 59 sampling calls, box still engine_self).
        if assertion_path is not None:
            write_jsonl(assertion_path, assertions)
            notes.append("confidence 写回 %s" % assertion_path.name)
    if confidence_status:
        notes.append("confidence 接口：%s（无模型/白盒时显式 unavailable，不冒充分数）"
                     % ", ".join(f"{k}={v}" for k, v in sorted(confidence_status.items())))

    # AVeriTeC quote verification (H04): "was this quote actually said by the supposed
    # speaker". The record's own quotes are enough context for the verifier once the
    # surrounding text is attached.
    #
    # The verdict is RECORDED, never acted on: the assertion keeps its speaker and its
    # evidence. AVeriTeC's output is a judgement attached to a claim, not a filter — and
    # deleting an attribution on a `against` would be the same mistake the removed cast
    # check made, one layer up.
    speaker_status: dict[str, int] = {}
    speaker_checked = 0
    if up.get("verify_speaker"):
        speaker_cache: dict = {}
        unit_texts = _unit_text_index(root)
        speaker_llm, speaker_same_model = _speaker_asker(up)
        for assertion in assertions:
            if str(assertion.get("claim_layer") or "") != "statement":
                continue
            verification = assertion.get("verification")
            if not isinstance(verification, dict):
                continue
            if str(verification.get("level") or "none") == "none":
                continue
            speaker = assertion.get("speaker")
            name = str((speaker or {}).get("name") or "")
            quote = ""
            for evidence in assertion.get("evidence") or ():
                quote = str(evidence.get("quote") or "")
                if quote:
                    break
            if not quote:
                continue
            # An assertion can carry an empty `unit_id`; the evidence still knows which
            # unit it came from, so fall back to it rather than looking up "" and
            # silently producing no context.
            unit_id = str(assertion.get("unit_id") or "")
            if not unit_id:
                for evidence in assertion.get("evidence") or ():
                    unit_id = str(evidence.get("unit_id") or "")
                    if unit_id:
                        break
            verdict = verify_speaker_quote(
                quote=quote,
                speaker_name=name,
                context=_quote_context(unit_texts.get(unit_id, ""), quote),
                cast=_speaker_cast(root),
                llm=speaker_llm,
                cache=speaker_cache,
                same_model=speaker_same_model,
            )
            verification.update(verdict.as_dict())
            speaker_status[verdict.verdict] = speaker_status.get(verdict.verdict, 0) + 1
            speaker_checked += 1
        if speaker_checked:
            notes.append(
                "AVeriTeC 引文归属核验：%d 条 statement（%s）"
                % (
                    speaker_checked,
                    ", ".join(f"{k}={v}" for k, v in sorted(speaker_status.items())),
                )
            )
            if assertion_path is not None:
                write_jsonl(assertion_path, assertions)


    # I17 superseded closure: a ``superseded_by`` must resolve, must terminate, must
    # not cycle, and the replaced record must be kept. The validator existed but
    # nothing called it, so a dangling or cyclic supersede could reach the box.
    supersede_errors: list[str] = []
    if assertions_with_ids:
        try:
            report = check_supersede_chain(assertions_with_ids)
        except Exception as exc:  # noqa: BLE001 - a broken record must not abort
            supersede_errors.append(f"{type(exc).__name__}: {exc}")
        else:
            supersede_errors.extend(
                f"{violation.code}: {violation.detail}" for violation in report.errors
            )
    if supersede_errors:
        notes.append(
            "I17 superseded 校验：%d 处问题（%s）"
            % (len(supersede_errors), "；".join(supersede_errors[:3]))
        )

    # §5.15 per-record validation (I19 + tier + verification). The function existed
    # and nothing called it, so a record could carry a confidence without a method,
    # or a level that disagreed with its tier, and still reach the box.
    record_violations: list[str] = []
    for assertion in assertions_with_ids:
        tier = _assertion_tier(assertion, tiers_by_entity)
        for violation in validate_assertion_record(assertion, tier=tier):
            record_violations.append(
                f"{assertion.get('assertion_id')}:{violation.code}: {violation.detail}"
            )
    if record_violations:
        notes.append(
            "§5.15 记录校验：%d 处问题（%s）"
            % (len(record_violations), "；".join(record_violations[:3]))
        )

    # I17 supersede chains + the retcon history stage 2 renders as 设定变更史.
    chain_map: dict[str, list[str]] = {}
    retcon_rows: list[dict[str, Any]] = []
    if assertions_with_ids:
        try:
            chain_map = {
                key: list(value) for key, value in supersede_chains(assertions_with_ids).items()
            }
            retcon_rows = [dict(row) for row in retcon_history(assertions_with_ids)]
        except Exception as exc:  # noqa: BLE001 - a bad chain must not abort validate
            notes.append(f"supersede 链/变更史生成失败：{type(exc).__name__}: {exc}")
    if retcon_rows:
        notes.append("D9 设定变更史：%d 条（旧值→新值+理由）" % len(retcon_rows))

    counts = {
        "assertions": len(assertions),
        "events": len(events),
        "events_source": events_source,
        "timeline": len(timeline),
        "cues": len(cues),
        "conflicts": len(conflict_rows),
        "conflicts_by_kind": by_kind,
        "idless_skipped": idless,
        "quarantine_added": quarantine_added,
        "quarantine_total": len(read_jsonl(quarantine_path)),
        "decisions": len(decisions),
        "supersede_errors": len(supersede_errors),
        "supersede_chains": len(chain_map),
        "retcon_rows": len(retcon_rows),
        "record_violations": len(record_violations),
        "confidence_status": confidence_status,
    }
    return {
        "step": STEP,
        "book_id": book_id,
        "created_at": created_at,
        "counts": counts,
        "paths": {
            "conflicts": str(conflicts_path),
            "decisions": str(decisions_path),
            "quarantine": str(quarantine_path),
        },
        "conflicts": conflict_rows,
        "notes": notes,
    }
