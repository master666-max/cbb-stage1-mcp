"""Nine-class ``ChapterFact`` → L1 predicate mapping.

Why this module exists
----------------------
The AI Reader engine emits nine *record classes* (人物/关系/地点/空间/物品/组织/
事件/概念/世界声明).  Those class names are **not** predicates: the box contract
registers a closed L1 predicate set (``亲属`` / ``位于`` / ``别名`` / ``参与`` /
``性质`` / ``敌对`` …) and I4 says an unregistered predicate must go to
``proposed_predicates.jsonl``, never into ``assertions``.

Measured before this module existed: ``normalize`` emitted 59 assertions across
12 predicate names, of which only 2 (``敌对``) were registered — **57/59 = 96.6%
out-of-table**.  Nothing was lying: the box quarantined them, coverage fell, and
the knowledge layer stayed empty.

The tempting "fix" was to register the 12 class names as predicates.  That is
exactly what ``00-架构总设计`` forbids ("谓词不是从数据里长出来的", "谓词注册表是
唯一真源"), so the mapping belongs here instead:

    00-架构总设计 §: "把抽取的 relation/attribute/event 映射到谓词注册表：
                      predicate + roles（角色名来自注册表），不合法 → quarantine"
    02-引擎契约 §:    "关系类型词表作为 relation_type 映射到 L1 谓词；
                      未映射 → proposed_predicates"

Design rules
------------
1. **Never invent a predicate.** Every mapped target is an already-registered L1
   predicate. Anything with no honest mapping returns ``None`` and the caller
   emits a ``proposed_predicates`` row — the spec's designed path.
2. **Never drop information.** Values that do not fit a role go to
   ``qualifiers`` (which the box schema allows as free-form) instead of being
   silently discarded.
3. **Role names come from the registry**, never from the extractor. ``施事`` /
   ``受事`` / ``主体`` / ``值`` / ``事件`` / ``参与者`` only.
4. **Pure functions, no I/O**, so the mapping is testable without a box.

Mapping table (each row cites why)
----------------------------------
======================  ==========  ==================================================
extractor class         L1 pred     rationale
======================  ==========  ==================================================
relationship            relation_   registry 亲属/敌对 are relation predicates; the
                        type→亲属    concrete ``relation_type`` is kept in qualifiers
                        /敌对      so nothing is lost
外貌 (appearance)       性质        设定库的通用属性断言 (主体, 值)
能力获得 (ability)       性质        同上；维度/名称/描述进 qualifiers
地点属性 (location)      位于        关系库 (主体, 值)：地点是什么
概念定义 (concept)       性质        设定库通用属性
空间关系 (spatial)       位于        关系库 (主体, 值)：位置关系
世界观声明 (world)       性质        设定库通用属性
物品事件 (item)          参与        事件库：物品/动作进 qualifiers
组织事件 (org)           参与        事件库：组织/动作进 qualifiers
======================  ==========  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: L1 predicates this module is allowed to target. Every name here must exist in
#: the box predicate registry; the contract test asserts that.
MAPPED_PREDICATES: frozenset[str] = frozenset(
    {"亲属", "敌对", "位于", "性质", "参与", "别名"}
)

#: Relation predicates whose two roles are ``施事`` / ``受事``.
TWO_PARTY_PREDICATES: frozenset[str] = frozenset({"亲属", "敌对"})

#: Concrete ``relation_type`` values the AI Reader emits that map onto a
#: registered two-party predicate. Anything else keeps the extractor's own word
#: and becomes a proposal (02-引擎契约: 未映射 → proposed_predicates).
RELATION_TYPE_TO_L1: Mapping[str, str] = {
    "亲属": "亲属",
    "父子": "亲属",
    "母子": "亲属",
    "母女": "亲属",
    "兄弟": "亲属",
    "姐妹": "亲属",
    "夫妻": "亲属",
    "叔侄": "亲属",
    "师徒": "亲属",
    "主仆": "亲属",
    "敌对": "敌对",
    "仇敌": "敌对",
    "对手": "敌对",
}


@dataclass(frozen=True)
class MappedAssertion:
    """One assertion after mapping: registry predicate + registry roles."""

    predicate: str
    library: str
    roles: dict[str, Any] = field(default_factory=dict)
    qualifiers: dict[str, Any] = field(default_factory=dict)

    @property
    def is_registered(self) -> bool:
        return self.predicate in MAPPED_PREDICATES


def _clean(value: Any) -> Any:
    """Drop absent values: a ``None`` role is not a value (box rejects it)."""
    if value is None or value == "" or value == []:
        return None
    return value


def literal(value: Any) -> dict[str, str] | None:
    """Box-shaped literal value: `{"literal": str}`.

    The Assembler validates role values against the registry, whose `literal`
    branch expects this wrapper. Emitting a bare string here made the two ends
    of the same contract disagree and every setting/event assertion was
    quarantined (measured: 9 emitted -> 0 into the box). Producing the wrapper
    at the birth end means the value never has to be guessed downstream.
    """
    cleaned = _clean(value)
    if cleaned is None:
        return None
    if isinstance(cleaned, Mapping) and "literal" in cleaned:
        return dict(cleaned)
    if isinstance(cleaned, (list, tuple)):
        cleaned = "／".join(str(item) for item in cleaned)
    return {"literal": str(cleaned)}


def _literal_roles(predicate: str, subject: str, value: Any, **qualifiers: Any) -> MappedAssertion:
    """Build a ``(主体, 值)`` assertion — the shape of 性质/位于/别名."""
    roles: dict[str, Any] = {}
    subject_value = _clean(subject)
    if subject_value is not None:
        roles["主体"] = subject_value
    value_value = literal(value)
    if value_value is not None:
        roles["值"] = value_value
    return MappedAssertion(
        predicate=predicate,
        library="设定" if predicate == "性质" else "关系",
        roles=roles,
        qualifiers={key: item for key, item in qualifiers.items() if _clean(item) is not None},
    )


def map_relationship(relation_type: str, person_a: str, person_b: str) -> MappedAssertion | None:
    """Relation classes → a registered two-party predicate, or ``None``.

    ``None`` means "no honest mapping": the caller must emit a
    ``proposed_predicates`` row rather than registering the extractor's word.
    """
    target = RELATION_TYPE_TO_L1.get(str(relation_type or "").strip())
    if target is None:
        return None
    roles: dict[str, Any] = {}
    if _clean(person_a) is not None:
        roles["施事"] = person_a
    if _clean(person_b) is not None:
        roles["受事"] = person_b
    return MappedAssertion(
        predicate=target,
        library="关系",
        roles=roles,
        qualifiers={"relation_type": relation_type},
    )


def map_appearance(name: str, appearance: str) -> MappedAssertion:
    """外貌 → 性质（设定库通用属性）。"""
    return _literal_roles("性质", name, appearance)


def map_ability(name: str, dimension: str, ability_name: str, description: str) -> MappedAssertion:
    """能力获得 → 性质，维度/名称/描述进 qualifiers。"""
    return _literal_roles(
        "性质",
        name,
        ability_name or dimension,
        dimension=dimension,
        ability_name=ability_name,
        description=description,
    )


def map_location(name: str, location_type: str, parent: Any, peers: Any, description: Any) -> MappedAssertion:
    """地点属性 → 位于（关系库：地点是什么/在哪）。"""
    return _literal_roles(
        "位于",
        name,
        location_type,
        parent=parent,
        peers=peers,
        description=description,
    )


def map_concept(name: str, category: str, definition: Any, related: Any) -> MappedAssertion:
    """概念定义 → 性质（设定库通用属性）。"""
    return _literal_roles(
        "性质",
        name,
        definition or category,
        category=category,
        related=related,
    )


def map_spatial(source: str, target: str, relation_type: str, value: Any, **extra: Any) -> MappedAssertion:
    """空间关系 → 位于（关系库）。"""
    return _literal_roles(
        "位于",
        source,
        value or relation_type,
        target=target,
        relation_type=relation_type,
        **extra,
    )


def map_world_declaration(declaration_type: str, content: Any) -> MappedAssertion | None:
    """世界观声明 → `None` (no honest L1 mapping; the caller proposes it).

    A declaration has no subject *entity*: `declaration_type` is a category
    (`portal` / `region_division` …), not a thing in the world, and both
    `性质` and `位于` require an `主体` that must resolve to an `e:` id.
    Forcing the type into that slot produced `entity role expects an 'e:' id,
    got 'portal'` — an id that can never resolve, so the assertion was
    quarantined anyway, just later and with a misleading reason.

    Returning `None` is the honest answer: the spec routes an unmapped record
    to `proposed_predicates` rather than inventing a subject for it.
    """
    return None


def map_item_event(
    item_name: str, item_type: Any, action: str, actor: Any, recipient: Any, description: Any
) -> MappedAssertion:
    """物品事件 → 参与（事件库）。物品/动作进 qualifiers。"""
    roles: dict[str, Any] = {}
    participant = _clean(actor) or _clean(recipient)
    if participant is not None:
        roles["事件"] = participant
        roles["参与者"] = [participant]
    return MappedAssertion(
        predicate="参与",
        library="事件",
        roles=roles,
        qualifiers={
            key: item
            for key, item in {
                "item_name": item_name,
                "item_type": item_type,
                "action": action,
                "actor": actor,
                "recipient": recipient,
                "description": description,
            }.items()
            if _clean(item) is not None
        },
    )


def map_org_event(
    org_name: str, org_type: Any, member: Any, role: Any, action: str, description: Any
) -> MappedAssertion:
    """组织事件 → 参与（事件库）。组织/动作进 qualifiers。"""
    roles: dict[str, Any] = {}
    participant = _clean(member)
    if participant is not None:
        roles["事件"] = participant
        roles["参与者"] = [participant]
    return MappedAssertion(
        predicate="参与",
        library="事件",
        roles=roles,
        qualifiers={
            key: item
            for key, item in {
                "org_name": org_name,
                "org_type": org_type,
                "member": member,
                "role": role,
                "action": action,
                "description": description,
            }.items()
            if _clean(item) is not None
        },
    )
