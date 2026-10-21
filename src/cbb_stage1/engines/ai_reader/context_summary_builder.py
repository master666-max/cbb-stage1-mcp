"""Build a rolling context summary from preceding ChapterFacts.

Ported from B06-AI-Reader-V2 ``extraction/context_summary_builder.py``.

Kept: preceding-fact aggregation, active window, scene focus, macro hubs,
hierarchy chains, location/item/relationship sections and the pre-scan
dictionary section (with the "aliases are hints only" disclaimer).

Dropped on purpose (design/08 §2.2: world structure services are not first
version): the ``WorldStructure`` section.  ``location_parents`` /
``location_tiers`` are still accepted so an orchestrator can pass an
authoritative hierarchy map later.

All store access is synchronous and injected; the builder itself does no I/O.
"""

from __future__ import annotations

import logging
from collections import Counter

from cbb_stage1.engines.ai_reader.budget import TokenBudget, compute_budget
from cbb_stage1.engines.ai_reader.schema import ChapterFact
from cbb_stage1.engines.ai_reader.stores import DictionaryStore, FactStore

logger = logging.getLogger(__name__)

# How many recent chapters to consider for "active" entities.
_ACTIVE_WINDOW = 20


class ContextSummaryBuilder:
    """Aggregate preceding ChapterFacts into a concise text summary."""

    def __init__(
        self,
        fact_store: FactStore,
        dictionary_store: DictionaryStore | None = None,
        budget: TokenBudget | None = None,
        *,
        include_dictionary: bool = True,
    ) -> None:
        self.fact_store = fact_store
        self.dictionary_store = dictionary_store
        self.budget = budget or compute_budget()
        # When the engine renders its own (injected) dictionary hints it disables
        # this section to avoid injecting the same pre-scan table twice.
        self.include_dictionary = include_dictionary

    def build(
        self,
        book_id: str,
        chapter_num: int,
        location_parents: dict[str, str] | None = None,
        location_tiers: dict[str, str] | None = None,
    ) -> str:
        """Build the context summary for ``chapter_num`` (1-indexed)."""
        sections: list[str] = []
        budget = self.budget

        chapter_facts: list[ChapterFact] = []
        if chapter_num > 1:
            all_facts = self.fact_store.list_facts(book_id)
            chapter_facts = [f for f in all_facts if f.chapter_id < chapter_num]

        if chapter_facts:
            recent_cutoff = chapter_num - _ACTIVE_WINDOW
            recent_facts = [f for f in chapter_facts if f.chapter_id >= recent_cutoff]
            if not recent_facts:
                recent_facts = chapter_facts[-5:]

            characters = self._aggregate_characters(chapter_facts, recent_facts)
            relationships = self._aggregate_relationships(chapter_facts, recent_facts)
            locations = self._aggregate_locations(chapter_facts, recent_facts)
            items = self._aggregate_items(chapter_facts, recent_facts)

            char_limit = budget.char_limit
            rel_limit = budget.rel_limit
            loc_limit = budget.loc_limit
            item_limit = budget.item_limit

            if characters:
                lines = ["### 已知人物"]
                for name, info in list(characters.items())[:char_limit]:
                    parts = [name]
                    if info.get("aliases"):
                        short_aliases = [a for a in info["aliases"] if len(a) <= 5][:3]
                        if short_aliases:
                            parts.append(f"(别名: {', '.join(short_aliases)})")
                    if info.get("abilities"):
                        parts.append(f"[{', '.join(info['abilities'][:3])}]")
                    lines.append("- " + " ".join(parts))
                sections.append("\n".join(lines))

            if relationships:
                lines = ["### 已知关系"]
                for rel in relationships[:rel_limit]:
                    lines.append(f"- {rel['a']} ↔ {rel['b']}: {rel['type']}")
                sections.append("\n".join(lines))

            focus_section = self._build_scene_focus_section(
                recent_facts, location_parents, locations
            )
            if focus_section:
                sections.append(focus_section)

            hub_section = self._build_macro_hub_section(location_parents, location_tiers)
            if hub_section:
                sections.append(hub_section)

            hierarchy_section = self._format_hierarchy_chains(locations, location_parents)
            if hierarchy_section:
                sections.append(hierarchy_section)

            if locations:
                lines = [
                    "### 已知地点",
                    '（如果本章出现"小城""那座山""此地"等泛称，'
                    "且上下文明确指代以下某个地名，请直接使用该地名，"
                    "不要将泛称作为独立地点提取）",
                ]
                for name, info in list(locations.items())[:loc_limit]:
                    desc = f"- {name} ({info['type']})"
                    parent = (
                        location_parents.get(name) if location_parents
                        else info.get("parent")
                    ) or info.get("parent")
                    if parent:
                        desc += f" ⊂ {parent}"
                    lines.append(desc)
                sections.append("\n".join(lines))

            if items:
                lines = ["### 已知物品"]
                for name, info in list(items.items())[:item_limit]:
                    holder = info.get("holder", "未知")
                    lines.append(f"- {name} ({info['type']}) — 持有: {holder}")
                sections.append("\n".join(lines))

        if self.include_dictionary:
            dict_section = self._build_dictionary_section(book_id)
            if dict_section:
                sections.append(dict_section)

        if not sections:
            return ""

        summary = "\n\n".join(sections)
        max_chars = budget.context_max_chars
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n...(已截断)"
        return summary

    # ── aggregators (unchanged algorithms) ───────────────────────

    def _aggregate_characters(
        self, all_facts: list[ChapterFact], recent_facts: list[ChapterFact]
    ) -> dict[str, dict]:
        chars: dict[str, dict] = {}
        for fact in all_facts:
            for ch in fact.characters:
                if ch.name not in chars:
                    chars[ch.name] = {"aliases": [], "abilities": []}
                entry = chars[ch.name]
                for alias in ch.new_aliases:
                    if alias not in entry["aliases"]:
                        entry["aliases"].append(alias)
                for ab in ch.abilities_gained:
                    label = f"{ab.dimension}:{ab.name}"
                    if label not in entry["abilities"]:
                        entry["abilities"].append(label)

        recent_names = {ch.name for fact in recent_facts for ch in fact.characters}
        return {name: info for name, info in chars.items() if name in recent_names}

    def _aggregate_relationships(
        self, all_facts: list[ChapterFact], recent_facts: list[ChapterFact]
    ) -> list[dict]:
        pair_map: dict[tuple[str, str], str] = {}
        for fact in all_facts:
            for rel in fact.relationships:
                first, second = sorted([rel.person_a, rel.person_b])
                key = (first, second)
                pair_map[key] = rel.relation_type

        recent_names = {ch.name for fact in recent_facts for ch in fact.characters}
        result = []
        for (a, b), rtype in pair_map.items():
            if a in recent_names or b in recent_names:
                result.append({"a": a, "b": b, "type": rtype})
        return result

    def _aggregate_locations(
        self, all_facts: list[ChapterFact], recent_facts: list[ChapterFact]
    ) -> dict[str, dict]:
        """Aggregate ALL known locations (needed for "小城" coreference)."""
        locs: dict[str, dict] = {}
        mention_counts: dict[str, int] = {}
        for fact in all_facts:
            for loc in fact.locations:
                mention_counts[loc.name] = mention_counts.get(loc.name, 0) + 1
                if loc.name not in locs:
                    locs[loc.name] = {"type": loc.type, "parent": loc.parent}
                elif loc.parent and not locs[loc.name].get("parent"):
                    locs[loc.name]["parent"] = loc.parent

        return dict(sorted(locs.items(), key=lambda x: mention_counts.get(x[0], 0), reverse=True))

    def _aggregate_items(
        self, all_facts: list[ChapterFact], recent_facts: list[ChapterFact]
    ) -> dict[str, dict]:
        items: dict[str, dict] = {}
        for fact in all_facts:
            for ie in fact.item_events:
                if ie.item_name not in items:
                    items[ie.item_name] = {"type": ie.item_type, "holder": ie.actor}
                if ie.action in ("获得", "赠予") and ie.recipient:
                    items[ie.item_name]["holder"] = ie.recipient
                elif ie.action == "获得":
                    items[ie.item_name]["holder"] = ie.actor
                elif ie.action in ("丢失", "损毁", "消耗"):
                    items[ie.item_name]["holder"] = "无"

        recent_item_names = {
            ie.item_name for fact in recent_facts for ie in fact.item_events
        }
        return {name: info for name, info in items.items() if name in recent_item_names}

    # ── scene focus / hubs / chains (unchanged) ──────────────────

    def _build_scene_focus_section(
        self,
        recent_facts: list[ChapterFact],
        location_parents: dict[str, str] | None,
        locations: dict[str, dict],
    ) -> str:
        if not recent_facts:
            return ""

        last_n = recent_facts[-3:]
        loc_freq: Counter = Counter()
        for fact in last_n:
            for char in fact.characters:
                for loc in char.locations_in_chapter:
                    if loc:
                        loc_freq[loc] += 1

        if not loc_freq:
            return ""

        focus_locs = [name for name, _ in loc_freq.most_common(3)]
        lines = [
            "### 当前场景焦点",
            "（以下是最近章节中角色最频繁出现的地点及其层级链。本章如出现新的建筑/房间名称，优先将其 parent 设为下方焦点地点）",
        ]
        for loc in focus_locs:
            chain = self._build_upward_chain(loc, location_parents)
            parts = []
            for name in chain:
                loc_type = locations.get(name, {}).get("type", "")
                parts.append(f"{name} ({loc_type})" if loc_type else name)
            lines.append(" > ".join(parts))
        return "\n".join(lines) if len(lines) > 2 else ""

    @staticmethod
    def _build_macro_hub_section(
        location_parents: dict[str, str] | None,
        location_tiers: dict[str, str] | None,
    ) -> str:
        if not location_parents:
            return ""

        children_set = set(location_parents.keys())
        parent_counts: Counter = Counter()
        for p in location_parents.values():
            if p not in children_set:
                parent_counts[p] += 1
        if not parent_counts:
            return ""
        uber_root = parent_counts.most_common(1)[0][0]

        children_map: dict[str, list[str]] = {}
        for child, parent in location_parents.items():
            children_map.setdefault(parent, []).append(child)

        def _count_descendants(node: str, visited: set) -> int:
            if node in visited:
                return 0
            visited.add(node)
            kids = children_map.get(node, [])
            total = len(kids)
            for k in kids:
                total += _count_descendants(k, visited)
            return total

        hubs: list[tuple[str, int]] = []
        for child in children_map.get(uber_root, []):
            desc_count = _count_descendants(child, set())
            if desc_count >= 3:
                hubs.append((child, desc_count))
        if not hubs:
            return ""

        hubs.sort(key=lambda x: -x[1])
        hubs = hubs[:8]

        lines = [
            "### 本小说主要宏观区域",
            f"（以下是「{uber_root}」下的主要区域。"
            "本章如出现新地点，请优先判断它属于哪个区域，"
            "并将该区域或其下属地点设为 parent）",
        ]
        for hub_name, desc_count in hubs:
            tier_str = ""
            if location_tiers:
                tier = location_tiers.get(hub_name, "")
                if tier:
                    tier_str = f" [{tier}]"
            sub_children = children_map.get(hub_name, [])
            if sub_children:
                shown = sub_children[:5]
                suffix = f" 等{len(sub_children)}处" if len(sub_children) > 5 else ""
                lines.append(
                    f"- **{hub_name}**{tier_str}（含 {desc_count} 处下属地点）→ "
                    f"{'、'.join(shown)}{suffix}"
                )
            else:
                lines.append(f"- **{hub_name}**{tier_str}（含 {desc_count} 处下属地点）")
        return "\n".join(lines)

    @staticmethod
    def _build_upward_chain(
        location: str,
        location_parents: dict[str, str] | None,
    ) -> list[str]:
        if not location_parents:
            return [location]
        chain = [location]
        visited = {location}
        current = location
        while current in location_parents:
            parent = location_parents[current]
            if parent in visited:
                break
            chain.append(parent)
            visited.add(parent)
            current = parent
        chain.reverse()
        return chain

    def _format_hierarchy_chains(
        self,
        locations: dict[str, dict],
        location_parents: dict[str, str] | None,
    ) -> str:
        if not location_parents:
            return ""

        children_map: dict[str, list[str]] = {}
        for child, parent in location_parents.items():
            children_map.setdefault(parent, []).append(child)

        all_children = set(location_parents.keys())
        all_parents = set(location_parents.values())
        roots = all_parents - all_children

        chains: list[list[str]] = []

        def _build_chain(node: str, current_chain: list[str]) -> None:
            current_chain.append(node)
            kids = children_map.get(node, [])
            if not kids:
                if len(current_chain) >= 2:
                    chains.append(list(current_chain))
            else:
                for kid in kids:
                    _build_chain(kid, current_chain)
            current_chain.pop()

        for root in roots:
            _build_chain(root, [])

        if not chains:
            return ""

        chains.sort(key=len, reverse=True)
        max_chains = self.budget.max_hierarchy_chains

        lines = [
            "### 已知地点层级",
            "（本章如出现新建筑/房间，请查看上述层级中是否有合适的 parent）",
        ]
        seen_chains: set[str] = set()
        for chain in chains[:max_chains]:
            parts = []
            for name in chain:
                loc_type = locations.get(name, {}).get("type", "")
                parts.append(f"{name} ({loc_type})" if loc_type else name)
            chain_str = " > ".join(parts)
            if chain_str not in seen_chains:
                seen_chains.add(chain_str)
                lines.append(chain_str)
        return "\n".join(lines) if len(lines) > 1 else ""

    # ── pre-scan dictionary section (unchanged policy) ───────────

    def _build_dictionary_section(self, book_id: str) -> str:
        if self.dictionary_store is None:
            return ""
        try:
            dictionary = self.dictionary_store.list_entries(book_id)
        except Exception:  # noqa: BLE001 - context is best-effort
            logger.warning("dictionary section failed for %s", book_id, exc_info=True)
            return ""
        if not dictionary:
            return ""

        naming_entries = [e for e in dictionary if e.source == "naming"]
        top = dictionary[:100]
        freq_entries = [e for e in top if e.name not in {n.name for n in naming_entries}]

        lines: list[str] = []
        if naming_entries:
            lines.append("### 文中明确命名的实体（必须使用完整名称提取）")
            lines.append(
                '以下名称在原文中通过"叫作/名叫/绰号"等方式明确引入，'
                "提取时务必使用完整名称，不要截断：",
            )
            for entry in naming_entries:
                line = f"- **{entry.name}**（{entry.entity_type}）"
                if entry.aliases:
                    safe = [a for a in entry.aliases if len(a) <= 4][:2]
                    if safe:
                        line += f" 可能别名：{'、'.join(safe)}（需文本确认）"
                lines.append(line)
            lines.append("")

        lines.append("### 本书高频实体参考")
        lines.append(
            "以下实体在全书中高频出现，提取时请特别注意不要遗漏"
            "（仅供参考，仍以原文为准）：",
        )
        lines.append(
            "⚠️ 下方「可能别名」仅为预扫描猜测，必须在本章原文中"
            "找到明确依据才能作为 new_aliases 输出。不同的人绝不能互为别名。",
        )
        for entry in freq_entries:
            line = f"- {entry.name}（{entry.entity_type}，出现{entry.frequency}次）"
            if entry.aliases:
                safe = [a for a in entry.aliases if len(a) <= 4][:2]
                if safe:
                    line += f" 可能别名：{'、'.join(safe)}（需文本确认）"
            lines.append(line)
        return "\n".join(lines)
