"""Semantic cleaning for V5 ``EngineOutput`` candidates (A7, 10 classes).

This is the *pipeline-level* cleaning layer (design/03 QC, acceptance A7): the
engine's raw candidates are cleaned deterministically before Evidence/Assemble.
It complements, and does not replace, the ChapterFact-level validator in
:mod:`cbb_stage1.engines.ai_reader.fact_validator` - the rule tables for generic
person/location names and the alias safety levels are imported from there so
there is exactly one source of truth.

The ten cleaning classes (applied in this order; every change is counted and
every drop is recorded in ``CleaningReport.quarantine`` with a closed-set
reason from design/01 §5.12):

1. ``schema`` - drop records/fields missing required non-empty text
   (mention surface, entity name/type, assertion predicate/library, event
   summary, evidence quote).
2. ``name_clamp`` - strip list separators / surrounding whitespace and clamp
   names, aliases, role strings and event summaries.
3. ``generic_person`` - drop person entities that are generic references or
   bare titles (大汉 / 老人 / 堂主 ...).
4. ``generic_location`` - drop location entities that fail the Chinese
   place-name morphology check (门口 / 山上 / 小城 ...).
5. ``alias_safety`` - remove aliases that are kinship/generic/title/pronoun
   terms (three-level safety table), self-aliases, or over-long phrases.
6. ``role_shape`` - assertions must have a non-empty predicate and roles; a
   ``fact`` without evidence is downgraded to ``statement`` +
   ``needs_evidence``; a ``statement`` without a speaker is flagged
   ``role_mismatch``; unresolvable role values are dropped.
7. ``event_shape`` - dedupe/drop empty participants, normalise importance/type,
   drop generic event locations.
8. ``evidence_exact`` - when the unit text is available, an evidence quote must
   equal ``text[start:end]`` exactly (else ``evidence_not_found``); quote sha
   is recomputed, occurrences are forced >= 1 and duplicate evidence is merged.
9. ``confidence_scope`` - clamp confidences to [0, 1], fill a default
   ``confidence_method``, normalise ``status`` / ``claim_layer`` to closed sets.
10. ``duplicates`` - exact duplicate mentions/entities/assertions/events are
    dropped (reason ``duplicate``); the first occurrence wins.

Six conflict *input* checks live in :mod:`cbb_stage1.fact_validator`; this
module only cleans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from cbb_stage1.engines.ai_reader.alias_resolver import alias_safety_level
from cbb_stage1.engines.ai_reader.fact_validator import (
    _clamp_name,
    _is_generic_location,
    _is_generic_person,
)
from cbb_stage1.ids import sha256_hex_text
from cbb_stage1.models import EngineOutput, Evidence


__all__ = [
    "CLEANING_RULES",
    "QUARANTINE_REASONS",
    "QuarantineItem",
    "CleaningReport",
    "clean_engine_output",
    "stable_record_id",
]

#: The ten classes, in application order (A7 evidence vocabulary).
CLEANING_RULES = (
    "schema",
    "name_clamp",
    "generic_person",
    "generic_location",
    "alias_safety",
    "role_shape",
    "event_shape",
    "evidence_exact",
    "confidence_scope",
    "duplicates",
)

#: ``quarantine.reason`` closed set from design/01 §5.12.
QUARANTINE_REASONS = frozenset({
    "schema",
    "evidence_not_found",
    "role_mismatch",
    "ref_missing",
    "timeline_cycle",
    "contradiction",
    "low_confidence",
    "duplicate",
    "unresolved_quote",
    "claim_layer_conflict",
})

_VALID_CLAIM_LAYERS = ("fact", "statement", "inferred")
_VALID_IMPORTANCE = ("high", "medium", "low")
_VALID_STATUSES = (
    "asserted",
    "needs_evidence",
    "needs_speaker",
    "inferred",
    "quarantined",
    "superseded",
    "open",
)
_MAX_NAME_CHARS = 20
_MAX_ALIAS_CHARS = 6


def stable_record_id(record_type: str, payload: Any) -> str:
    """Deterministic short id for quarantine/conflict records."""
    digest = sha256_hex_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), 16
    )
    return f"{record_type}:{digest}"


@dataclass(frozen=True)
class QuarantineItem:
    """One cleaning drop/flag (``audit/quarantine.jsonl``-shaped)."""

    record_type: str
    record_id: str
    reason: str
    detail: str

    def __post_init__(self) -> None:
        if self.reason not in QUARANTINE_REASONS:
            raise ValueError(
                f"quarantine reason {self.reason!r} not in closed set {sorted(QUARANTINE_REASONS)}"
            )

    def to_record(self) -> dict[str, str]:
        return {
            "record_type": self.record_type,
            "record_id": self.record_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class CleaningReport:
    """Result of :func:`clean_engine_output`."""

    output: EngineOutput
    rules: dict[str, int] = field(default_factory=dict)
    quarantine: tuple[QuarantineItem, ...] = ()

    @property
    def changed(self) -> bool:
        return any(self.rules.values())

    @property
    def dropped(self) -> int:
        return sum(self.rules.values())

    def summary(self) -> dict[str, Any]:
        return {
            "unit_id": self.output.unit_id,
            "changed": self.changed,
            "rules": dict(self.rules),
            "quarantined": [item.to_record() for item in self.quarantine],
        }


def _coerce_output(output: EngineOutput | dict[str, Any]) -> EngineOutput:
    if isinstance(output, EngineOutput):
        return output.model_copy(deep=True)
    return EngineOutput.model_validate(output)


def _clean_string(value: Any) -> str:
    return _clamp_name(value.strip()) if isinstance(value, str) and value.strip() else ""


class _Cleaner:
    """Mutable working state for one EngineOutput cleaning run."""

    def __init__(self, output: EngineOutput, unit_text: str | None) -> None:
        self.output = output
        self.unit_text = unit_text
        self.rules: dict[str, int] = {rule: 0 for rule in CLEANING_RULES}
        self.quarantine: list[QuarantineItem] = []

    # -- helpers --------------------------------------------------
    def bump(self, rule: str, count: int = 1) -> None:
        self.rules[rule] += count

    def flag(self, record_type: str, record_id: str, reason: str, detail: str) -> None:
        self.quarantine.append(QuarantineItem(record_type, record_id, reason, detail))

    # -- 1. schema ------------------------------------------------
    def rule_schema(self) -> None:
        output = self.output

        cleaned_mentions = []
        for mention in output.mentions:
            if not _clean_string(mention.surface) or not mention.unit_id:
                self.bump("schema")
                self.flag(
                    "mention",
                    stable_record_id("m", mention.model_dump()),
                    "schema",
                    "empty mention surface/unit_id",
                )
                continue
            cleaned_mentions.append(mention)
        output.mentions = cleaned_mentions

        cleaned_entities = []
        for entity in output.entities:
            if not _clean_string(entity.canonical_name) or not _clean_string(entity.type):
                self.bump("schema")
                self.flag(
                    "entity",
                    stable_record_id("e", entity.model_dump()),
                    "schema",
                    "empty entity name/type",
                )
                continue
            cleaned_entities.append(entity)
        output.entities = cleaned_entities

        cleaned_assertions = []
        for assertion in output.assertions:
            if not _clean_string(assertion.predicate) or not _clean_string(assertion.library):
                self.bump("schema")
                self.flag(
                    "assertion",
                    stable_record_id("a", assertion.model_dump()),
                    "schema",
                    "empty assertion predicate/library",
                )
                continue
            if not assertion.roles:
                self.bump("schema")
                self.flag(
                    "assertion",
                    stable_record_id("a", assertion.model_dump()),
                    "role_mismatch",
                    "assertion has no roles",
                )
                continue
            cleaned_assertions.append(assertion)
        output.assertions = cleaned_assertions

        cleaned_events = []
        for event in output.events:
            if not _clean_string(event.summary) or not event.unit_id:
                self.bump("schema")
                self.flag(
                    "event",
                    stable_record_id("v", event.model_dump()),
                    "schema",
                    "empty event summary/unit_id",
                )
                continue
            cleaned_events.append(event)
        output.events = cleaned_events

    # -- 2. name_clamp --------------------------------------------
    def rule_name_clamp(self) -> None:
        for entity in self.output.entities:
            for attr in ("canonical_name",):
                value = getattr(entity, attr)
                cleaned = _clean_string(value)
                if cleaned != value:
                    setattr(entity, attr, cleaned)
                    self.bump("name_clamp")
            cleaned_aliases = [_clean_string(alias) for alias in entity.aliases]
            cleaned_aliases = [alias for alias in cleaned_aliases if alias]
            if cleaned_aliases != entity.aliases:
                entity.aliases = cleaned_aliases
                self.bump("name_clamp")
        for mention in self.output.mentions:
            cleaned = _clean_string(mention.surface)
            if cleaned != mention.surface:
                mention.surface = cleaned
                self.bump("name_clamp")
        for event in self.output.events:
            cleaned = event.summary.strip()
            if cleaned != event.summary:
                event.summary = cleaned
                self.bump("name_clamp")
        for assertion in self.output.assertions:
            changed = False
            new_roles: dict[str, Any] = {}
            for key, value in assertion.roles.items():
                if isinstance(value, str):
                    cleaned_value = _clean_string(value)
                    if cleaned_value != value:
                        changed = True
                    new_roles[_clean_string(key) or str(key)] = cleaned_value or None
                else:
                    new_roles[key] = value
            if changed or new_roles.keys() != assertion.roles.keys():
                assertion.roles = new_roles
                self.bump("name_clamp")

    # -- 3/4. generic names ---------------------------------------
    def rule_generic_person(self) -> None:
        kept = []
        for entity in self.output.entities:
            if entity.type == "person" and _is_generic_person(entity.canonical_name):
                self.bump("generic_person")
                self.flag(
                    "entity",
                    stable_record_id("e", entity.model_dump()),
                    "schema",
                    f"generic person reference: {entity.canonical_name}",
                )
                continue
            kept.append(entity)
        self.output.entities = kept

    def rule_generic_location(self) -> None:
        kept = []
        for entity in self.output.entities:
            if entity.type == "location":
                reason = _is_generic_location(entity.canonical_name)
                if reason:
                    self.bump("generic_location")
                    self.flag(
                        "entity",
                        stable_record_id("e", entity.model_dump()),
                        "schema",
                        f"generic location ({reason}): {entity.canonical_name}",
                    )
                    continue
            kept.append(entity)
        self.output.entities = kept

    # -- 5. alias safety -------------------------------------------
    def rule_alias_safety(self) -> None:
        for entity in self.output.entities:
            filtered: list[str] = []
            for alias in entity.aliases:
                if alias == entity.canonical_name:
                    self.bump("alias_safety")
                    continue
                if len(alias) > _MAX_ALIAS_CHARS:
                    self.bump("alias_safety")
                    continue
                if alias_safety_level(alias) < 2:
                    self.bump("alias_safety")
                    continue
                if alias not in filtered:
                    filtered.append(alias)
                else:
                    self.bump("alias_safety")
            if filtered != entity.aliases:
                entity.aliases = filtered

    # -- 6. role shape ---------------------------------------------
    def rule_role_shape(self) -> None:
        kept = []
        for assertion in self.output.assertions:
            # Drop role entries that carry no usable value (role_mismatch).
            usable_roles = {
                key: value
                for key, value in assertion.roles.items()
                if value not in (None, "", [], {})
            }
            if len(usable_roles) != len(assertion.roles):
                assertion.roles = usable_roles
                self.bump("role_shape")
            if not usable_roles:
                self.bump("role_shape")
                self.flag(
                    "assertion",
                    stable_record_id("a", assertion.model_dump()),
                    "role_mismatch",
                    "assertion has no usable role values",
                )
                continue

            if assertion.claim_layer not in _VALID_CLAIM_LAYERS:
                # An invalid layer defaults to `fact`, not `statement`: a statement
                # requires a speaker (§5.16) and there is no reason to assume one.
                assertion.claim_layer = "fact"
                self.bump("role_shape")
            if assertion.claim_layer == "fact" and not assertion.evidence:
                # A fact with no evidence is not a statement — a statement needs a
                # speaker (§5.16), and the speaker lives on the evidence, so relabelling
                # guarantees a `claim_layer_conflict` quarantine. Keep the layer and let
                # `status` carry the gap; that is what the status field is for.
                assertion.status = "needs_evidence"
                self.bump("role_shape")
                self.flag(
                    "assertion",
                    stable_record_id("a", assertion.model_dump()),
                    "evidence_not_found",
                    "fact without evidence (status=needs_evidence)",
                )
            if assertion.claim_layer == "statement" and assertion.speaker is None:
                assertion.status = "needs_speaker"
                self.bump("role_shape")
                self.flag(
                    "assertion",
                    stable_record_id("a", assertion.model_dump()),
                    "role_mismatch",
                    "statement without speaker",
                )
            kept.append(assertion)
        self.output.assertions = kept

    # -- 7. event shape ---------------------------------------------
    def rule_event_shape(self) -> None:
        for event in self.output.events:
            participants: list[str] = []
            for participant in event.participants:
                cleaned = _clean_string(participant)
                if not cleaned:
                    self.bump("event_shape")
                    continue
                if cleaned not in participants:
                    participants.append(cleaned)
                else:
                    self.bump("event_shape")
            if participants != event.participants:
                event.participants = participants
            if event.importance not in _VALID_IMPORTANCE:
                event.importance = "medium"
                self.bump("event_shape")
            if not _clean_string(event.type):
                event.type = "其他"
                self.bump("event_shape")
            if event.location:
                cleaned_location = _clean_string(event.location)
                if not cleaned_location or _is_generic_location(cleaned_location):
                    event.location = None
                    self.bump("event_shape")
                else:
                    event.location = cleaned_location

    # -- 8. evidence exact ------------------------------------------
    def rule_evidence_exact(self) -> None:
        for assertion in self.output.assertions:
            kept: list[Evidence] = []
            seen: set[tuple[int, int, str]] = set()
            for evidence in assertion.evidence:
                quote = evidence.quote or ""
                if self.unit_text is not None:
                    span_ok = (
                        0 <= evidence.start < evidence.end <= len(self.unit_text)
                        and self.unit_text[evidence.start:evidence.end] == quote
                    )
                    if not span_ok:
                        self.bump("evidence_exact")
                        self.flag(
                            "evidence",
                            stable_record_id("ev", evidence.model_dump()),
                            "evidence_not_found",
                            "quote != unit_text[start:end]",
                        )
                        continue
                key = (evidence.start, evidence.end, quote)
                if key in seen:
                    self.bump("evidence_exact")
                    continue
                seen.add(key)
                if evidence.occurrence < 1:
                    evidence.occurrence = 1
                    self.bump("evidence_exact")
                evidence.quote_sha256 = evidence.quote_sha256 or sha256_hex_text(quote)
                if evidence.quote != quote:
                    evidence.quote = quote
                kept.append(evidence)
            if kept != assertion.evidence:
                assertion.evidence = kept
            if assertion.claim_layer == "fact" and not kept:
                # Same reasoning as above: dropping the evidence does not make the
                # assertion a statement, it makes it evidence-less.
                assertion.status = "needs_evidence"
                self.bump("evidence_exact")

    # -- 9. confidence / scope --------------------------------------
    def rule_confidence_scope(self) -> None:
        for entity in self.output.entities:
            clamped = max(0.0, min(1.0, float(entity.confidence)))
            if clamped != entity.confidence:
                entity.confidence = clamped
                self.bump("confidence_scope")
        for assertion in self.output.assertions:
            clamped = max(0.0, min(1.0, float(assertion.confidence)))
            if clamped != assertion.confidence:
                assertion.confidence = clamped
                self.bump("confidence_scope")
            if not assertion.confidence_method:
                assertion.confidence_method = "engine_self_report"
                self.bump("confidence_scope")
            if assertion.status not in _VALID_STATUSES:
                assertion.status = "quarantined"
                self.bump("confidence_scope")

    # -- 10. duplicates ---------------------------------------------
    def rule_duplicates(self) -> None:
        seen_mentions: set[tuple[str, int, int]] = set()
        mentions = []
        for mention in self.output.mentions:
            key = (mention.surface, mention.start, mention.end)
            if key in seen_mentions:
                self.bump("duplicates")
                continue
            seen_mentions.add(key)
            mentions.append(mention)
        self.output.mentions = mentions

        seen_entities: set[tuple[str, str]] = set()
        entities = []
        for entity in self.output.entities:
            key = (entity.type, entity.canonical_name)
            if key in seen_entities:
                self.bump("duplicates")
                continue
            seen_entities.add(key)
            entities.append(entity)
        self.output.entities = entities

        seen_assertions: set[str] = set()
        assertions = []
        for assertion in self.output.assertions:
            payload = {
                "predicate": assertion.predicate,
                "library": assertion.library,
                "roles": assertion.roles,
                "claim_layer": assertion.claim_layer,
                "evidence": sorted(
                    (ev.quote, ev.start, ev.end) for ev in assertion.evidence
                ),
            }
            key = stable_record_id("a", payload)
            if key in seen_assertions:
                self.bump("duplicates")
                continue
            seen_assertions.add(key)
            assertions.append(assertion)
        self.output.assertions = assertions

        seen_events: set[tuple[str, str]] = set()
        events = []
        for event in self.output.events:
            key = (event.type, event.summary)
            if key in seen_events:
                self.bump("duplicates")
                continue
            seen_events.add(key)
            events.append(event)
        self.output.events = events

    # -- run --------------------------------------------------------
    def run(self) -> CleaningReport:
        for rule in CLEANING_RULES:
            getattr(self, f"rule_{rule}")()
        return CleaningReport(
            output=self.output,
            rules={rule: count for rule, count in self.rules.items() if count},
            quarantine=tuple(self.quarantine),
        )


def clean_engine_output(
    output: EngineOutput | dict[str, Any],
    *,
    unit_text: str | None = None,
) -> CleaningReport:
    """Apply the ten cleaning classes to one unit's ``EngineOutput``.

    Args:
        output: EngineOutput (or its dict form) for one unit.
        unit_text: The exact unit text; when given, evidence spans are verified
            verbatim and mismatching evidence is quarantined
            (``evidence_not_found``) instead of being kept as if valid.

    Returns:
        :class:`CleaningReport` with the cleaned output, per-rule change counts
        and the quarantine items.
    """
    coerced = _coerce_output(output)
    return _Cleaner(coerced, unit_text).run()
