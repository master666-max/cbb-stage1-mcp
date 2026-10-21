"""Alias resolution: AI Reader Union-Find + three-level safety table + fanout clean.

This module is a *thin* deterministic wrapper around the ported engine module
:mod:`cbb_stage1.engines.ai_reader.alias_resolver` -- it does not re-implement
the algorithm.  Two things are added on top of the engine port:

1. A pure entry point (:func:`build_alias_map`) that runs the engine's
   ``AliasResolver.build_from_sources`` directly on dictionaries/chapter facts,
   without any store or database.
2. The Lorebase fanout clean (``B07-lorebase/scripts/02b_merge_dict.py:73-121``,
   Apache-2.0), which the engine port does not have: a surface that hosts more
   than ``FANOUT`` distinct entities is a generic term (the upstream project
   measured "189 aliases absorbed by one generic bridge") and is removed
   globally before any merge.

Design references: ``00-架构总设计.md`` §4.8 (Resolver order: rule -> dictionary
+ fanout -> vector -> LLM -> human), ``01-箱规格-v2.md`` §5.5, and
``08-现成管线骨架调研-20261018.md`` §3 (``resolve/fanout.py``).

Safety invariant (from the engine port, kept verbatim): generic / kinship /
title terms can never become Union-Find keys, because the same surface may
refer to different entities in different chapters and would create false
bridges.  :data:`alias_safety_level` is re-exported so callers and tests use the
exact same table as the engine and the normalizer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from cbb_stage1.engines.ai_reader.alias_resolver import (
    AliasResolver as EngineAliasResolver,
    UnionFind,
    alias_safety_level,
)
from cbb_stage1.engines.ai_reader.schema import ChapterFact, EntityDictEntry
from cbb_stage1.engines.ai_reader.stores import InMemoryDictionaryStore, InMemoryFactStore

__all__ = [
    "AliasHost",
    "BLACKLIST",
    "DEFAULT_FANOUT",
    "FanoutFinding",
    "FanoutResult",
    "SAFETY_BLOCKED",
    "SAFETY_SAFE",
    "SAFETY_SUSPICIOUS",
    "UnionFind",
    "alias_is_safe",
    "alias_safety_level",
    "build_alias_map",
    "build_alias_map_from_entities",
    "canonical_surface",
    "fanout_clean",
    "filter_aliases",
    "surface_hosts",
]

# ── Three-level safety table (re-exported from the engine port) ───

SAFETY_BLOCKED = 0
SAFETY_SUSPICIOUS = 1
SAFETY_SAFE = 2


def alias_is_safe(alias: str) -> bool:
    """True when *alias* may be a Union-Find key (engine level 2 only)."""
    return alias_safety_level(alias) >= SAFETY_SAFE


def filter_aliases(
    aliases: Iterable[str],
    *,
    owner: str | None = None,
    min_level: int = SAFETY_SAFE,
) -> list[str]:
    """Strip, de-duplicate and safety-filter an alias list (order preserved)."""
    out: list[str] = []
    for raw in aliases:
        alias = (raw or "").strip()
        if not alias or alias == owner:
            continue
        if alias_safety_level(alias) < min_level:
            continue
        if alias not in out:
            out.append(alias)
    return out


# ── Pure alias map (no store / DB) ───────────────────────────────

# ``build_from_sources`` is pure, but ``AliasResolver`` asks for the two store
# protocols at construction.  One process-local stateless pair is enough.
_PURE_RESOLVER = EngineAliasResolver(InMemoryDictionaryStore(), InMemoryFactStore())


def build_alias_map(
    dictionary: Sequence[EntityDictEntry],
    facts: Sequence[ChapterFact] = (),
) -> dict[str, str]:
    """Run the engine's Union-Find over pre-scan entries + chapter facts.

    Returns ``alias -> canonical_name``.  Canonical names never map to
    themselves.  The three-level safety table and the "two established groups
    are never bridged" conflict guard are exactly the engine ones.
    """
    return _PURE_RESOLVER.build_from_sources(list(dictionary), list(facts))


def canonical_surface(alias_map: dict[str, str], surface: str) -> str:
    """Resolve one surface through an alias map (identity when unknown)."""
    return alias_map.get(surface, surface)


#: Tier weighting used only to break ties when the same alias network is given
#: as ``Entity`` records (the engine prefers high-frequency, 2-3 char names).
_TIER_WEIGHT = {"canonical": 3, "major": 2, "minor": 1}


def build_alias_map_from_entities(entities: Iterable[Any]) -> dict[str, str]:
    """Convenience wrapper: ``Entity`` records -> the engine's dictionary input.

    Accepts any object exposing ``canonical_name``/``aliases`` (``Entity``) or
    ``name``/``aliases`` (``EntityDictEntry``).  The engine's canonical-name
    scoring still decides the winner; tier only acts as a frequency prior.
    """
    entries: list[EntityDictEntry] = []
    for entity in entities:
        name = getattr(entity, "canonical_name", None) or getattr(entity, "name", None)
        if not name:
            continue
        etype = getattr(entity, "type", None) or getattr(entity, "entity_type", None) or "unknown"
        if etype == "unknown":
            continue
        tier = getattr(entity, "tier", None) or "minor"
        entries.append(
            EntityDictEntry(
                name=name,
                entity_type=etype,
                frequency=_TIER_WEIGHT.get(tier, 1),
                aliases=filter_aliases(getattr(entity, "aliases", ()) or (), owner=name),
                source="entity",
            )
        )
    return build_alias_map(entries)


# ── Lorebase fanout clean (B07 02b_merge_dict.py, Apache-2.0) ────

#: Upstream blacklist (B07 ``02b_merge_dict.py:26-31``): address/title terms the
#: dictionary extractor keeps proposing as names.  Data table, copied verbatim.
BLACKLIST: frozenset[str] = frozenset({
    "大人", "阁下", "陛下", "国王陛下", "神", "将军", "国王", "老弟", "大哥",
    "小姐姐", "小妹妹", "孩子", "小孩", "少年", "老者", "父亲", "父亲大人",
    "父王", "爸", "妈妈", "盟主", "舰长", "船长", "司令", "长官", "首领",
    "老师", "美男子", "赌徒", "骗子手", "天人", "外星人", "那孩子", "那个少年",
    "公主", "王子", "老女人", "骗子", "盟主陛下",
})

#: Upstream ``FANOUT = 3``; a surface hosted by **more than** 3 entities is
#: generic.  Kept as a parameter so a book/lexicon can raise it.
DEFAULT_FANOUT = 3


@runtime_checkable
class AliasHost(Protocol):
    """Minimal entity shape the fanout clean needs (``Entity`` / ``EntityDictEntry``)."""

    aliases: Sequence[str]


def _host_name(record: Any) -> str:
    name = getattr(record, "canonical_name", None) or getattr(record, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            "fanout entries must expose a non-empty canonical_name/name; "
            f"got {type(record).__name__}"
        )
    return name


def _with_aliases(record: Any, aliases: Sequence[str]) -> Any:
    """Return a copy of *record* with a cleaned alias list (input not mutated)."""
    if hasattr(record, "model_copy"):  # pydantic v2 models (Entity, EntityDictEntry)
        return record.model_copy(update={"aliases": list(aliases)})
    if is_dataclass(record):
        return replace(record, aliases=list(aliases))
    raise TypeError(f"unsupported fanout entry type: {type(record).__name__}")


@dataclass(frozen=True)
class FanoutFinding:
    """Why one surface was classified as generic."""

    surface: str
    host_count: int
    hosts: tuple[str, ...]
    reason: str  # "blacklist" | "unsafe_alias" | "fanout"


@dataclass(frozen=True)
class FanoutResult:
    """Outcome of :func:`fanout_clean` (pure; originals are untouched)."""

    kept: tuple[Any, ...]
    generic_names: frozenset[str]
    findings: tuple[FanoutFinding, ...]
    dropped_entities: tuple[str, ...]
    dropped_aliases: dict[str, tuple[str, ...]]
    threshold: int

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def dropped_entity_count(self) -> int:
        return len(self.dropped_entities)

    @property
    def dropped_alias_count(self) -> int:
        return sum(len(v) for v in self.dropped_aliases.values())


def surface_hosts(records: Iterable[Any]) -> dict[str, tuple[str, ...]]:
    """Map every surface (canonical name and aliases) to the entities hosting it."""
    hosts: dict[str, set[str]] = defaultdict(set)
    for record in records:
        name = _host_name(record)
        for surface in {name} | set(getattr(record, "aliases", ()) or ()):
            if surface:
                hosts[surface].add(name)
    return {surface: tuple(sorted(names)) for surface, names in hosts.items()}


def _generic_reason(
    surface: str,
    host_count: int,
    *,
    threshold: int,
    blacklist: frozenset[str],
    apply_safety_table: bool,
) -> str | None:
    if surface in blacklist:
        return "blacklist"
    if apply_safety_table and alias_safety_level(surface) < SAFETY_SAFE:
        return "unsafe_alias"
    if host_count > threshold:
        return "fanout"
    return None


def fanout_clean(
    records: Sequence[Any],
    *,
    fanout_threshold: int = DEFAULT_FANOUT,
    blacklist: Iterable[str] = BLACKLIST,
    apply_safety_table: bool = True,
) -> FanoutResult:
    """Remove generic surfaces globally (Lorebase fanout clean + safety table).

    * A surface is generic when it is blacklisted, fails the AI Reader safety
      table (level < 2) or is hosted by **more than** ``fanout_threshold``
      distinct entities.
    * Generic aliases are removed from every entity; generic canonical names
      remove the whole entry (the upstream "dropped generic-named entries").
    * The input records are not mutated; cleaned copies are returned in
      :attr:`FanoutResult.kept`.

    This is the interface the Resolver calls *before* :func:`build_alias_map`:
    clean the dictionary, then run Union-Find on the cleaned surfaces.
    """
    if fanout_threshold < 1:
        raise ValueError("fanout_threshold must be >= 1")
    blacklist_set = frozenset(blacklist)
    hosts = surface_hosts(records)

    generic: dict[str, FanoutFinding] = {}
    for surface, host_names in hosts.items():
        reason = _generic_reason(
            surface,
            len(host_names),
            threshold=fanout_threshold,
            blacklist=blacklist_set,
            apply_safety_table=apply_safety_table,
        )
        if reason is not None:
            generic[surface] = FanoutFinding(
                surface=surface,
                host_count=len(host_names),
                hosts=host_names,
                reason=reason,
            )

    kept: list[Any] = []
    dropped_entities: list[str] = []
    dropped_aliases: dict[str, tuple[str, ...]] = {}
    for record in records:
        name = _host_name(record)
        if name in generic:
            dropped_entities.append(name)
            continue
        original = list(getattr(record, "aliases", ()) or ())
        aliases = [
            alias
            for alias in dict.fromkeys(original)
            if alias and alias != name and alias not in generic
        ]
        dropped = [alias for alias in original if alias in generic]
        if dropped:
            dropped_aliases[name] = tuple(dict.fromkeys(dropped))
        kept.append(_with_aliases(record, aliases))

    return FanoutResult(
        kept=tuple(kept),
        generic_names=frozenset(generic),
        findings=tuple(generic[s] for s in sorted(generic)),
        dropped_entities=tuple(dropped_entities),
        dropped_aliases=dropped_aliases,
        threshold=fanout_threshold,
    )
