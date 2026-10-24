"""Pipeline-level fact validation: 10 cleaning classes + 6 conflict input checks.

Acceptance A7 ("10 类语义清洗 + 6 类冲突检测", design/03 QC, design/01 §5.17).

Two layers:

* **cleaning** - :func:`clean_engine_output` (re-exported from
  :mod:`cbb_stage1.normalize.cleaning`) applies ten deterministic semantic
  cleaning classes to one unit's ``EngineOutput``; every drop lands in a
  ``quarantine`` record with a closed-set reason.
* **conflict input checks** - :func:`detect_input_conflicts` ports the six
  AI Reader conflict detectors (ability / relationship / location hierarchy /
  death continuity / direction / distance) onto V5 ``EngineOutput`` candidates.
  They *validate input* only: conflicts are reported, never auto-resolved and
  never overwritten (design/01 §5.17: "不删除；人工/Arbiter 裁决").  Each
  conflict is mapped to a v2.2 ``kind`` (``attribute`` / ``relation`` /
  ``timeline``) plus the source ``subtype`` so it can flow into
  ``knowledge/conflicts.jsonl``.

Nothing here calls a model; all rules are deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1.engines.ai_reader.location_names import is_homonym_prone
from cbb_stage1.models import Assertion, EngineOutput
from cbb_stage1.normalize.cleaning import (
    CLEANING_RULES,
    QUARANTINE_REASONS,
    CleaningReport,
    QuarantineItem,
    clean_engine_output,
    stable_record_id,
)

__all__ = [
    "CLEANING_RULES",
    "CONFLICT_KINDS",
    "CONFLICT_SUBTYPES",
    "QUARANTINE_REASONS",
    "CleaningReport",
    "InputConflict",
    "QuarantineItem",
    "ValidationReport",
    "clean_engine_output",
    "detect_input_conflicts",
    "validate_engine_output",
]

#: v2.2 conflicts.kind closed set (design/01 §5.17).
CONFLICT_KINDS = ("attribute", "relation", "timeline")
#: AI Reader conflict-detector subtypes (the six ported rules).
CONFLICT_SUBTYPES = (
    "ability",
    "relation",
    "location",
    "death",
    "direction",
    "distance",
)

_SUBTYPE_KIND = {
    "ability": "attribute",
    "relation": "relation",
    "location": "attribute",
    "death": "timeline",
    "direction": "relation",
    "distance": "relation",
}

_HOSTILE_TERMS = frozenset({"敌对", "仇人", "对手", "仇敌"})
_FAMILY_TERMS = frozenset({
    "亲属", "父子", "母子", "兄弟", "姐妹", "夫妻", "父女", "母女",
})
_OPPOSITE_DIRECTIONS = {
    "north_of": "south_of",
    "south_of": "north_of",
    "east_of": "west_of",
    "west_of": "east_of",
    "northeast_of": "southwest_of",
    "southwest_of": "northeast_of",
    "northwest_of": "southeast_of",
    "southeast_of": "northwest_of",
}
_DIR_LABEL = {
    "north_of": "北方", "south_of": "南方", "east_of": "东方", "west_of": "西方",
    "northeast_of": "东北方", "northwest_of": "西北方",
    "southeast_of": "东南方", "southwest_of": "西南方",
}
_DISTANCE_ORDINAL = {"near": 0, "medium": 1, "far": 2, "very_far": 3}
_DC_LABEL = {"near": "近", "medium": "中等", "far": "远", "very_far": "极远"}
_MIN_MINORITY_UNITS = 2
_MIN_DISTANCE_GAP = 2


@dataclass(frozen=True)
class InputConflict:
    """One detected input conflict (v2.2 record shape + AI Reader subtype)."""

    kind: str  # attribute | relation | timeline
    subtype: str  # ability | relation | location | death | direction | distance
    severity: str  # 严重 | 一般 | 提示
    description: str
    units: tuple[int, ...]
    entity: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in CONFLICT_KINDS:
            raise ValueError(f"conflict.kind {self.kind!r} not in {CONFLICT_KINDS}")
        if self.subtype not in CONFLICT_SUBTYPES:
            raise ValueError(f"conflict.subtype {self.subtype!r} not in {CONFLICT_SUBTYPES}")

    @property
    def conflict_id(self) -> str:
        return stable_record_id(
            "c",
            {
                "kind": self.kind,
                "subtype": self.subtype,
                "entity": self.entity,
                "units": list(self.units),
                "details": dict(self.details),
            },
        )

    def to_record(self, book_id: str) -> dict[str, Any]:
        """``knowledge/conflicts.jsonl``-shaped record (status stays ``open``)."""
        return {
            "conflict_id": self.conflict_id,
            "book_id": book_id,
            "kind": self.kind,
            "subtype": self.subtype,
            "severity": self.severity,
            "entity": self.entity,
            "units": list(self.units),
            "detail": self.description,
            "details": dict(self.details),
            "status": "open",
            "decision": None,
            "resolved_by": None,
            "at": None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "kind": self.kind,
            "subtype": self.subtype,
            "severity": self.severity,
            "description": self.description,
            "units": list(self.units),
            "entity": self.entity,
            "details": dict(self.details),
        }


@dataclass
class ValidationReport:
    """Combined cleaning + conflict-input validation for a set of units."""

    cleanings: list[CleaningReport] = field(default_factory=list)
    conflicts: tuple[InputConflict, ...] = ()
    outputs: list[EngineOutput] = field(default_factory=list)

    @property
    def quarantine(self) -> tuple[QuarantineItem, ...]:
        items: list[QuarantineItem] = []
        for cleaning in self.cleanings:
            items.extend(cleaning.quarantine)
        return tuple(items)

    def summary(self) -> dict[str, Any]:
        return {
            "units": len(self.cleanings),
            "changed_units": sum(1 for cleaning in self.cleanings if cleaning.changed),
            "quarantine": len(self.quarantine),
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "rules": {
                rule: sum(cleaning.rules.get(rule, 0) for cleaning in self.cleanings)
                for rule in CLEANING_RULES
            },
        }


# ── helpers ─────────────────────────────────────────────────────


def _resolve(name: Any, alias_map: Mapping[str, str] | None) -> str:
    if not isinstance(name, str):
        return ""
    canonical = alias_map.get(name, name) if alias_map else name
    return canonical.strip()


def _coerce_output(output: EngineOutput | dict[str, Any]) -> EngineOutput:
    return output if isinstance(output, EngineOutput) else EngineOutput.model_validate(output)


#: Detector key -> the names it may appear under, in the order they are tried.
#:
#: `roles` carries the registered role names the assembler produces (`主体`/`值`/`施事`), while
#: `normalize/predicate_map.py` puts everything the detector needs into `qualifiers`
#: (`dimension`, `ability_name`, `parent`, `relation_type`, ...). Before this table existed the
#: detectors asked only for `entity`/`dimension`/`name`/`parent` in `roles`, none of which the
#: normalizer writes, so every input conflict check returned nothing on real data.
_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "entity": ("entity", "主体", "施事", "subject"),
    "location": ("location", "entity", "主体"),
    "value": ("value", "值"),
    "name": ("name", "ability_name", "ability"),
    "ability": ("ability", "ability_name", "name"),
    "parent": ("parent",),
    "dimension": ("dimension",),
    "member": ("member", "成员", "entity", "主体"),
    "relation_type": ("relation_type", "关系类型"),
    "distance_class": ("distance_class",),
}


def _role(assertion: Assertion, *keys: str) -> str:
    """Read one role-ish value from `roles` or `qualifiers`, following the alias table.

    `roles` wins over `qualifiers` and an exact key wins over an alias, so an assertion that
    carries both shapes keeps the older, more specific answer.
    """
    roles = assertion.roles if isinstance(assertion.roles, dict) else {}
    qualifiers = assertion.qualifiers if isinstance(assertion.qualifiers, dict) else {}
    for key in keys:
        for candidate in _ROLE_ALIASES.get(key, (key,)):
            for source in (roles, qualifiers):
                value = source.get(candidate)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _iter_assertions(
    records: Sequence[tuple[int, EngineOutput | dict[str, Any]]]
) -> Iterable[tuple[int, EngineOutput, Assertion]]:
    for unit, output in sorted(records, key=lambda item: item[0]):
        coerced = _coerce_output(output)
        for assertion in coerced.assertions:
            yield unit, coerced, assertion


def _values_in_output(output: EngineOutput) -> set[str]:
    """All surface/name strings mentioned in an output (for death continuity)."""
    values: set[str] = set()
    for entity in output.entities:
        values.add(entity.canonical_name)
        values.update(entity.aliases)
    for mention in output.mentions:
        values.add(mention.surface)
    for event in output.events:
        values.update(p for p in event.participants if p)
    for assertion in output.assertions:
        for value in assertion.roles.values():
            if isinstance(value, str) and value:
                values.add(value)
    return values


# ── six conflict detectors ─────────────────────────────────────


def _detect_ability(
    records: Sequence[tuple[int, EngineOutput]], alias_map: Mapping[str, str] | None
) -> list[InputConflict]:
    """Ability dimension regression: A -> B -> A for the same character."""
    timeline: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for unit, _output, assertion in _iter_assertions(records):
        entity = _resolve(_role(assertion, "entity"), alias_map)
        dimension = _role(assertion, "dimension")
        ability = _role(assertion, "name", "ability")
        if not entity or not dimension or not ability:
            continue
        timeline.setdefault((entity, dimension), []).append((unit, ability))

    conflicts: list[InputConflict] = []
    for (entity, dimension), entries in timeline.items():
        if len(entries) < 3:
            continue
        for index in range(2, len(entries)):
            _, previous = entries[index - 1]
            _, current = entries[index]
            for back in range(index - 2, -1, -1):
                early_unit, early = entries[back]
                if current == early and previous != current:
                    middle_unit, middle = entries[index - 1]
                    conflicts.append(
                        InputConflict(
                            kind="attribute",
                            subtype="ability",
                            severity="一般",
                            description=(
                                f"{entity} 的{dimension}从「{early}」(unit {early_unit})"
                                f"变为「{middle}」(unit {middle_unit})又回到「{current}」"
                                f"(unit {entries[index][0]})，疑似回退"
                            ),
                            units=(early_unit, middle_unit, entries[index][0]),
                            entity=entity,
                            details={
                                "dimension": dimension,
                                "values": [early, middle, current],
                            },
                        )
                    )
                    break
            else:
                continue
            break
    return conflicts


def _detect_relation(
    records: Sequence[tuple[int, EngineOutput]], alias_map: Mapping[str, str] | None
) -> list[InputConflict]:
    """Hostile<->family transitions and A->B->A relation flip-flops."""
    timeline: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for unit, _output, assertion in _iter_assertions(records):
        person_a = _resolve(_role(assertion, "person_a", "subject"), alias_map)
        person_b = _resolve(_role(assertion, "person_b", "object"), alias_map)
        if not person_a or not person_b or person_a == person_b:
            continue
        key = (min(person_a, person_b), max(person_a, person_b))
        timeline.setdefault(key, []).append((unit, assertion.predicate))

    conflicts: list[InputConflict] = []
    for (person_a, person_b), entries in timeline.items():
        if len(entries) < 2:
            continue
        for index in range(1, len(entries)):
            previous_unit, previous_type = entries[index - 1]
            current_unit, current_type = entries[index]
            previous_hostile = any(term in previous_type for term in _HOSTILE_TERMS)
            current_family = any(term in current_type for term in _FAMILY_TERMS)
            previous_family = any(term in previous_type for term in _FAMILY_TERMS)
            current_hostile = any(term in current_type for term in _HOSTILE_TERMS)
            if previous_hostile and current_family:
                severity = "一般"
                note = "敌对→亲属转变异常"
            elif previous_family and current_hostile:
                severity = "提示"
                note = "亲属→敌对（可能是叛变剧情）"
            else:
                severity = ""
                note = ""
            if severity:
                conflicts.append(
                    InputConflict(
                        kind="relation",
                        subtype="relation",
                        severity=severity,
                        description=(
                            f"{person_a}与{person_b}的关系从「{previous_type}」"
                            f"(unit {previous_unit})变为「{current_type}」"
                            f"(unit {current_unit})，{note}"
                        ),
                        units=(previous_unit, current_unit),
                        entity=person_a,
                        details={"other": person_b, "from": previous_type, "to": current_type},
                    )
                )
        if len(entries) >= 3:
            for index in range(2, len(entries)):
                type0 = entries[index - 2][1]
                type1 = entries[index - 1][1]
                type2 = entries[index][1]
                if type0 == type2 and type1 != type0:
                    conflicts.append(
                        InputConflict(
                            kind="relation",
                            subtype="relation",
                            severity="提示",
                            description=(
                                f"{person_a}与{person_b}的关系反复："
                                f"「{type0}」→「{type1}」→「{type2}」"
                            ),
                            units=(
                                entries[index - 2][0],
                                entries[index - 1][0],
                                entries[index][0],
                            ),
                            entity=person_a,
                            details={"other": person_b, "pattern": [type0, type1, type2]},
                        )
                    )
    return conflicts


def _detect_location(
    records: Sequence[tuple[int, EngineOutput]],
) -> list[InputConflict]:
    """Same location with different parents across units (minority >= 2 units)."""
    timeline: dict[str, list[tuple[int, str]]] = {}
    for unit, _output, assertion in _iter_assertions(records):
        entity = _role(assertion, "entity", "location")
        parent = _role(assertion, "parent")
        if not entity or not parent:
            continue
        timeline.setdefault(entity, []).append((unit, parent))

    conflicts: list[InputConflict] = []
    for location, assignments in timeline.items():
        parents = {parent for _, parent in assignments}
        if len(parents) <= 1 or is_homonym_prone(location):
            continue
        by_parent: dict[str, list[int]] = {}
        for unit, parent in assignments:
            by_parent.setdefault(parent, []).append(unit)
        ordered = sorted(by_parent.items(), key=lambda item: len(item[1]), reverse=True)
        majority_parent = ordered[0][0]
        for parent, units in ordered[1:]:
            if len(units) < _MIN_MINORITY_UNITS:
                continue
            conflicts.append(
                InputConflict(
                    kind="attribute",
                    subtype="location",
                    severity="一般",
                    description=(
                        f"地点「{location}」的上级不一致：多数单元为「{majority_parent}」，"
                        f"但 unit {'/'.join(str(u) for u in units[:3])} 为「{parent}」"
                    ),
                    units=tuple(units[:3]),
                    entity=location,
                    details={
                        "majority_parent": majority_parent,
                        "conflict_parent": parent,
                    },
                )
            )
    return conflicts


def _detect_death(
    records: Sequence[tuple[int, EngineOutput]], alias_map: Mapping[str, str] | None
) -> list[InputConflict]:
    """A character marked 阵亡 (death) acting in a later unit."""
    death_unit: dict[str, int] = {}
    for unit, _output, assertion in _iter_assertions(records):
        action = _role(assertion, "action")
        member = _resolve(_role(assertion, "member", "entity"), alias_map)
        if member and action == "阵亡" and member not in death_unit:
            death_unit[member] = unit
    if not death_unit:
        return []

    conflicts: list[InputConflict] = []
    for unit, output in sorted(records, key=lambda item: item[0]):
        output = _coerce_output(output)
        mentioned = _values_in_output(output)
        for member, died_in in list(death_unit.items()):
            if unit > died_in and member in mentioned:
                conflicts.append(
                    InputConflict(
                        kind="timeline",
                        subtype="death",
                        severity="严重",
                        description=(
                            f"角色「{member}」在 unit {died_in} 阵亡，"
                            f"但在 unit {unit} 再次出现"
                        ),
                        units=(died_in, unit),
                        entity=member,
                        details={"death_unit": died_in, "reappear_unit": unit},
                    )
                )
                del death_unit[member]
                break
    return conflicts


def _detect_direction(
    records: Sequence[tuple[int, EngineOutput]], alias_map: Mapping[str, str] | None
) -> list[InputConflict]:
    """Contradictory direction claims for the same location pair."""
    pair_directions: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for unit, _output, assertion in _iter_assertions(records):
        if _role(assertion, "relation_type") != "direction":
            continue
        source = _resolve(_role(assertion, "source"), alias_map)
        target = _resolve(_role(assertion, "target"), alias_map)
        value = _role(assertion, "value")
        if not source or not target or not value:
            continue
        if source > target:
            source, target = target, source
            value = _OPPOSITE_DIRECTIONS.get(value, value)
        pair_directions.setdefault((source, target), []).append((unit, value))

    conflicts: list[InputConflict] = []
    for (source, target), entries in pair_directions.items():
        if len(entries) < 2:
            continue
        by_direction: dict[str, list[int]] = {}
        for unit, direction in entries:
            by_direction.setdefault(direction, []).append(unit)
        if len(by_direction) <= 1:
            continue
        reported: set[tuple[str, str]] = set()
        for direction_a, units_a in by_direction.items():
            opposite = _OPPOSITE_DIRECTIONS.get(direction_a)
            if not opposite or opposite not in by_direction:
                continue
            pair = tuple(sorted((direction_a, opposite)))
            if pair in reported:
                continue
            reported.add(pair)
            units_b = by_direction[opposite]
            conflicts.append(
                InputConflict(
                    kind="relation",
                    subtype="direction",
                    severity="一般",
                    description=(
                        f"「{source}」相对「{target}」的方向矛盾："
                        f"unit {'/'.join(str(u) for u in units_a[:3])} 为"
                        f"{_DIR_LABEL.get(direction_a, direction_a)}，"
                        f"unit {'/'.join(str(u) for u in units_b[:3])} 为"
                        f"{_DIR_LABEL.get(opposite, opposite)}"
                    ),
                    units=tuple(sorted(set(units_a[:2] + units_b[:2]))),
                    entity=source,
                    details={
                        "other": target,
                        "direction_a": direction_a,
                        "direction_b": opposite,
                        "units_a": units_a[:3],
                        "units_b": units_b[:3],
                    },
                )
            )
    return conflicts


def _detect_distance(
    records: Sequence[tuple[int, EngineOutput]], alias_map: Mapping[str, str] | None
) -> list[InputConflict]:
    """Contradictory distance classes (ordinal gap >= 2) for a location pair."""
    pair_distances: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for unit, _output, assertion in _iter_assertions(records):
        if _role(assertion, "relation_type") != "distance":
            continue
        source = _resolve(_role(assertion, "source"), alias_map)
        target = _resolve(_role(assertion, "target"), alias_map)
        distance_class = _role(assertion, "distance_class")
        if not source or not target or distance_class not in _DISTANCE_ORDINAL:
            continue
        key = (min(source, target), max(source, target))
        pair_distances.setdefault(key, []).append((unit, distance_class))

    conflicts: list[InputConflict] = []
    for (location_a, location_b), entries in pair_distances.items():
        if len(entries) < 2:
            continue
        by_class: dict[str, list[int]] = {}
        for unit, distance_class in entries:
            by_class.setdefault(distance_class, []).append(unit)
        if len(by_class) <= 1:
            continue
        classes = list(by_class)
        reported: set[tuple[str, str]] = set()
        for index, first in enumerate(classes):
            for second in classes[index + 1:]:
                gap = abs(_DISTANCE_ORDINAL[first] - _DISTANCE_ORDINAL[second])
                if gap < _MIN_DISTANCE_GAP:
                    continue
                pair = (min(first, second), max(first, second))
                if pair in reported:
                    continue
                reported.add(pair)
                units_a = by_class[first]
                units_b = by_class[second]
                conflicts.append(
                    InputConflict(
                        kind="relation",
                        subtype="distance",
                        severity="一般" if gap >= 3 else "提示",
                        description=(
                            f"「{location_a}」与「{location_b}」的距离矛盾："
                            f"unit {'/'.join(str(u) for u in units_a[:3])} 为"
                            f"{_DC_LABEL.get(first, first)}，"
                            f"unit {'/'.join(str(u) for u in units_b[:3])} 为"
                            f"{_DC_LABEL.get(second, second)}"
                        ),
                        units=tuple(sorted(set(units_a[:2] + units_b[:2]))),
                        entity=location_a,
                        details={
                            "other": location_b,
                            "distance_a": first,
                            "distance_b": second,
                            "gap": gap,
                        },
                    )
                )
    return conflicts


# ── public API ──────────────────────────────────────────────────

_DETECTORS = {
    "ability": _detect_ability,
    "relation": _detect_relation,
    "location": _detect_location,
    "death": _detect_death,
    "direction": _detect_direction,
    "distance": _detect_distance,
}


def detect_input_conflicts(
    records: Sequence[tuple[int, EngineOutput | dict[str, Any]]],
    *,
    alias_map: Mapping[str, str] | None = None,
) -> tuple[InputConflict, ...]:
    """Run the six conflict input checks over ``(unit_order, EngineOutput)``.

    Args:
        records: Per-unit engine outputs with their unit order (chapter/seq).
        alias_map: Optional alias -> canonical map so ``二愣子`` and ``韩立``
            are compared as one entity.

    Returns:
        Deterministically ordered conflicts (severity, then first unit).
    """
    normalized = [(unit, _coerce_output(output)) for unit, output in records]
    conflicts: list[InputConflict] = []
    for subtype in CONFLICT_SUBTYPES:
        detector = _DETECTORS[subtype]
        if subtype in ("location",):
            conflicts.extend(detector(normalized))  # type: ignore[arg-type]
        else:
            conflicts.extend(detector(normalized, alias_map))  # type: ignore[arg-type]

    severity_order = {"严重": 0, "一般": 1, "提示": 2}
    conflicts.sort(
        key=lambda conflict: (
            severity_order.get(conflict.severity, 3),
            conflict.units[0] if conflict.units else 0,
            conflict.subtype,
            conflict.entity,
        )
    )
    return tuple(conflicts)


def validate_engine_output(
    records: Sequence[tuple[int, EngineOutput | dict[str, Any]]],
    *,
    unit_texts: Mapping[int, str] | None = None,
    alias_map: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Clean every unit output, then run the six conflict input checks."""
    outputs: list[EngineOutput] = []
    cleanings: list[CleaningReport] = []
    for unit, output in records:
        coerced = _coerce_output(output)
        unit_text = unit_texts.get(unit) if unit_texts else None
        cleaning = clean_engine_output(coerced, unit_text=unit_text)
        cleanings.append(cleaning)
        outputs.append(cleaning.output)

    conflicts = detect_input_conflicts(
        list(zip([unit for unit, _ in records], outputs)), alias_map=alias_map
    )
    return ValidationReport(cleanings=cleanings, conflicts=conflicts, outputs=outputs)


def conflicts_as_jsonl(
    conflicts: Sequence[InputConflict], book_id: str
) -> str:
    """Render conflicts as stable one-JSON-per-line text (``conflicts.jsonl``)."""
    return "\n".join(
        json.dumps(conflict.to_record(book_id), ensure_ascii=False, sort_keys=True)
        for conflict in conflicts
    )
