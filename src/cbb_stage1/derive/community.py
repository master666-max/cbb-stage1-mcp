"""Hierarchical structural community reports (GraphRAG shape, no LLM).

The full GraphRAG pass uses Leiden/Louvain + LLM community summaries.  Stage-1
has no embedding/LLM dependency here, so this module does deterministic work:

* **level 1** -- connected components over entity-role edges (a real structural
  community, not a summary);
* **level 2** -- one roll-up per volume, built from the level-1 communities whose
  members appear in that volume (requires an ``entity_volumes`` mapping, usually
  derived from resolve mentions + unit volumes). Level-1 rows get a
  ``parent_id`` pointing at their level-2 roll-up; level-2 rows have
  ``parent_id=None``.

Every row is emitted with ``title``/``summary=None``: structure is real, the
prose is left for a later synthesis pass and is never invented here.

Public API:

* :func:`build_community_reports` -- level 1, plus level 2 when a mapping is
  supplied (backwards compatible: no mapping => exactly the old level-1 output);
* :func:`rollup_communities` -- explicit level1 -> level2 roll-up;
* :func:`drilldown` -- level2 community id -> its level-1 children.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping
import json

from cbb_stage1 import ids

__all__ = [
    "build_community_reports",
    "rollup_communities",
    "drilldown",
    "validate_community_hierarchy",
]

_LEVEL1 = 1
_LEVEL2 = 2


def _community_id(book_id: str, level: int, members: Iterable[str], *, volume: int | None = None) -> str:
    """Wrapper over :func:`cbb_stage1.ids.community_id` (same bytes)."""
    return ids.community_id(book_id, level, members, volume=volume)


def _report_id(community_id: str) -> str:
    """Wrapper over :func:`cbb_stage1.ids.report_id` (same bytes)."""
    return ids.report_id(community_id)


def _entity_ids(roles: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for value in roles.values():
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, str) and item.startswith("e:"):
                out.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("entity"), str):
                out.append(item["entity"])
    return list(dict.fromkeys(out))


def _level1_components(assertions: Iterable[Mapping[str, Any]]) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for assertion in assertions:
        member_ids = _entity_ids(assertion.get("roles") or {})
        for i, left in enumerate(member_ids):
            for right in member_ids[i + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        stack = [start]
        members: list[str] = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            members.append(node)
            stack.extend(sorted(graph[node] - seen))
        # A one-entity "community" is not a community report.
        if len(members) >= 2:
            components.append(sorted(members))
    return sorted(components)


def _primary_volume(members: Iterable[str], entity_volumes: Mapping[str, Any]) -> int | None:
    """Dominant volume of a member set (highest member count, then lowest number)."""
    counts: dict[int, int] = defaultdict(int)
    for entity_id in members:
        volumes = entity_volumes.get(entity_id)
        if volumes is None:
            continue
        values = volumes if isinstance(volumes, (list, tuple, set, frozenset)) else [volumes]
        for value in values:
            try:
                counts[int(value)] += 1
            except (TypeError, ValueError):
                continue
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _level1_rows(
    book_id: str,
    assertions: Iterable[Mapping[str, Any]],
    *,
    produced_by: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for members in _level1_components(assertions):
        community_id = _community_id(book_id, _LEVEL1, members)
        rows.append(
            {
                "report_id": _report_id(community_id),
                "book_id": book_id,
                "level": _LEVEL1,
                "community_id": community_id,
                "parent_id": None,
                "member_entities": members,
                "title": None,
                "summary": None,
                "evidence": [],
                "stale": False,
                "produced_by": produced_by,
            }
        )
    return sorted(rows, key=lambda row: row["community_id"])


def rollup_communities(
    book_id: str,
    level1_rows: Iterable[Mapping[str, Any]],
    entity_volumes: Mapping[str, Any],
    *,
    produced_by: str = "derive:community_structural",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Roll level-1 communities up to one level-2 community per volume.

    Args:
        book_id: Book id used for content-addressed community ids.
        level1_rows: Level-1 rows (``member_entities`` mandatory).
        entity_volumes: entity id -> volume int (or iterable of volumes).
        produced_by: Provenance marker.

    Returns:
        ``(level1_with_parent, level2_rows)``. A level-1 community that spans
        volumes is attached to its dominant volume (deterministic; documented),
        and it is a member of every level-2 roll-up it touches.
    """
    rows = [dict(row) for row in level1_rows]
    by_volume: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for entity_id in row.get("member_entities") or ():
            volumes = entity_volumes.get(entity_id)
            if volumes is None:
                continue
            values = volumes if isinstance(volumes, (list, tuple, set, frozenset)) else [volumes]
            for value in values:
                try:
                    volume = int(value)
                except (TypeError, ValueError):
                    continue
                if row not in by_volume[volume]:
                    by_volume[volume].append(row)

    level2_rows: list[dict[str, Any]] = []
    parent_of: dict[int, str] = {}
    for volume in sorted(by_volume):
        members = sorted(
            {
                entity_id
                for row in by_volume[volume]
                for entity_id in row.get("member_entities") or ()
            }
        )
        if len(members) < 2:
            continue
        community_id = _community_id(book_id, _LEVEL2, members, volume=volume)
        parent_of[volume] = community_id
        level2_rows.append(
            {
                "report_id": _report_id(community_id),
                "book_id": book_id,
                "level": _LEVEL2,
                "volume": volume,
                "community_id": community_id,
                "parent_id": None,
                "child_community_ids": sorted(
                    str(row.get("community_id")) for row in by_volume[volume]
                ),
                "member_entities": members,
                "title": None,
                "summary": None,
                "evidence": [],
                "stale": False,
                "produced_by": produced_by,
            }
        )

    for row in rows:
        members = row.get("member_entities") or []
        volume = _primary_volume(members, entity_volumes)
        row["parent_id"] = parent_of.get(volume) if volume is not None else None
    return rows, sorted(level2_rows, key=lambda item: item["community_id"])


def build_community_reports(
    book_id: str,
    assertions: Iterable[Mapping[str, Any]],
    *,
    entities: Iterable[Mapping[str, Any]] = (),
    entity_volumes: Mapping[str, Any] | None = None,
    levels: Iterable[int] = (_LEVEL1, _LEVEL2),
    summarizer: Any = None,
) -> list[dict[str, Any]]:
    """Connected components (level 1) and optional volume roll-ups (level 2).

    Args:
        book_id: Book id.
        assertions: Assertions used for entity-role edges.
        entities: Optional entity records (reserved; isolated entities do not
            become community reports).
        entity_volumes: entity id -> volume int / volumes. When omitted only
            level 1 is emitted, preserving the pre-hierarchy behaviour.
        levels: Which levels to emit (default both, level 2 only with mapping).

    Returns:
        Rows sorted by ``(level, community_id)``.
    """
    del entities  # isolated entities are not communities
    assertion_rows = list(assertions)
    requested = {int(level) for level in levels}
    level1 = _level1_rows(book_id, assertion_rows, produced_by="derive:community_structural")
    level2: list[dict[str, Any]] = []
    if entity_volumes and _LEVEL2 in requested:
        level1, level2 = rollup_communities(book_id, level1, entity_volumes)
    result: list[dict[str, Any]] = []
    if _LEVEL1 in requested:
        result.extend(level1)
    if _LEVEL2 in requested:
        result.extend(level2)
    if summarizer is not None:
        related: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for assertion in assertion_rows:
            snippet = {
                "assertion_id": assertion.get("assertion_id"),
                "predicate": assertion.get("predicate"),
                "roles": dict(assertion.get("roles") or {}),
                "evidence": [
                    {"unit_id": ev.get("unit_id"), "quote": ev.get("quote")}
                    for ev in assertion.get("evidence") or ()
                    if isinstance(ev, Mapping) and ev.get("quote")
                ],
            }
            for entity_id in _entity_ids(assertion.get("roles") or {}):
                if len(related[entity_id]) < 8:
                    related[entity_id].append(snippet)
        for row in result:
            member_lines: list[str] = []
            source_ids: list[str] = []
            for entity_id in row.get("member_entities") or ():
                for snippet in related.get(entity_id, ())[:4]:
                    if snippet.get("assertion_id"):
                        source_ids.append(snippet["assertion_id"])
                    quotes = " / ".join(
                        str(item.get("quote")) for item in snippet.get("evidence") or ()
                    )
                    member_lines.append(f"- {snippet.get('predicate')}: {quotes}")
            prompt = (
                "用一句中文标题和一段客观摘要描述这个实体社区；"
                "只能使用下面的断言/引文，不得补充原文没有的事实。只输出 JSON: "
                '{"title":"...","summary":"..."}\n'
                f"成员：{', '.join(row.get('member_entities') or ())}\n"
                "相关证据：\n" + ("\n".join(member_lines) or "（无）")
            )
            raw = summarizer(prompt)
            if isinstance(raw, Mapping):
                row["title"] = raw.get("title")
                row["summary"] = raw.get("summary")
            elif isinstance(raw, str) and raw.strip():
                parsed = None
                try:
                    parsed = json.loads(raw.strip().strip("`"))
                except Exception:  # noqa: BLE001 - plain text is a valid fallback
                    parsed = None
                if isinstance(parsed, Mapping):
                    row["title"] = parsed.get("title")
                    row["summary"] = parsed.get("summary")
                else:
                    row["summary"] = raw.strip()
            row["source_assertion_ids"] = sorted(set(source_ids))
    return sorted(result, key=lambda row: (int(row["level"]), str(row["community_id"])))


def drilldown(
    reports: Iterable[Mapping[str, Any]],
    parent_community_id: str,
) -> list[dict[str, Any]]:
    """Return the level-1 children of a level-2 community id (sorted)."""
    return sorted(
        (
            dict(row)
            for row in reports
            if row.get("parent_id") == parent_community_id
            and int(row.get("level") or 0) < _LEVEL2
        ),
        key=lambda row: str(row["community_id"]),
    )


def validate_community_hierarchy(reports: Iterable[Mapping[str, Any]]) -> list[str]:
    """Check id uniqueness, parent closure, and level/parent ordering."""
    errors: list[str] = []
    rows = list(reports)
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        community_id = str(row.get("community_id") or "")
        if not community_id:
            errors.append(f"row {index}: missing community_id")
            continue
        if community_id in by_id:
            errors.append(f"row {index}: duplicate community_id {community_id}")
        by_id[community_id] = row
    for community_id, row in by_id.items():
        parent = row.get("parent_id")
        if parent is None:
            continue
        if parent not in by_id:
            errors.append(f"community {community_id}: parent missing {parent}")
            continue
        if int(by_id[parent].get("level") or 0) <= int(row.get("level") or 0):
            errors.append(f"community {community_id}: parent level must be higher")
    return errors
