"""Entity delta merging: GraphRAG's flow with this project's key.

The flow is borrowed from `库/code/B-项目案例/B02-graphrag/.../index/update/entities.py:26-60` — match, offset
ids, concat, group by the matched key, accumulate. The key is not borrowed: GraphRAG matches on `title`, and
A20 records that this project deliberately matches on content, because a chapter or entity retitled while its
text stays identical would otherwise look new.

The accumulation matters as much as the matching. GraphRAG collects descriptions into a **list** and
re-summarises later; this project's `describe` keeps whichever description is longest and throws the other
away, which loses information before any summariser can use it. A list defers that decision instead of making
it silently.

Idempotence is a requirement, not a nicety: an append path is run again whenever a book is reprocessed, and a
second identical delta must not double the table.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "merge_entities",
    "match_by_content",
]


def _desc_of(row: Mapping[str, Any]) -> str:
    for key in ("description", "desc"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _descriptions_of(row: Mapping[str, Any]) -> list[str]:
    stored = row.get("descriptions")
    if isinstance(stored, list):
        return [str(item) for item in stored if str(item).strip()]
    single = _desc_of(row)
    return [single] if single else []


def match_by_content(
    old_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """``{delta_entity_id: old_entity_id}`` for entities whose content hash matches.

    An entity with no hash is never matched: pairing on the name alone is what `title` keying does, and the
    reason this project moved away from it.
    """
    by_sha: dict[str, str] = {}
    for row in old_rows:
        sha = str(row.get("sha256") or "")
        entity_id = str(row.get("entity_id") or "")
        if sha and entity_id:
            by_sha.setdefault(sha, entity_id)

    mapping: dict[str, str] = {}
    for row in delta_rows:
        sha = str(row.get("sha256") or "")
        entity_id = str(row.get("entity_id") or "")
        if not sha or not entity_id:
            continue
        target = by_sha.get(sha)
        if target and target != entity_id:
            mapping[entity_id] = target
    return mapping


def merge_entities(
    old_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fold matched delta rows into their old counterparts, append the rest.

    Returns ``(merged_rows, report)``. The old rows are copied, never mutated: a caller that keeps its list
    must not find it rewritten underneath.
    """
    merged: list[dict[str, Any]] = []
    for row in old_rows:
        copy = dict(row)
        copy["descriptions"] = _descriptions_of(row)
        copy["aliases"] = list(dict.fromkeys(str(alias) for alias in row.get("aliases") or ()))
        merged.append(copy)

    mapping = match_by_content(old_rows, delta_rows)
    by_id = {str(row.get("entity_id")): row for row in merged}

    merged_count = 0
    appended_count = 0
    for row in delta_rows:
        delta_id = str(row.get("entity_id") or "")
        target_id = mapping.get(delta_id)
        if target_id and target_id in by_id:
            target = by_id[target_id]
            for description in _descriptions_of(row):
                if description not in target["descriptions"]:
                    target["descriptions"].append(description)
            for alias in list(row.get("aliases") or ()) + [str(row.get("canonical_name") or "")]:
                alias = str(alias).strip()
                if alias and alias != target.get("canonical_name") and alias not in target["aliases"]:
                    target["aliases"].append(alias)
            merged_count += 1
            continue

        # Unmatched: appended as a new row, unless it is already present (idempotence).
        if delta_id and delta_id in by_id:
            continue
        copy = dict(row)
        copy["descriptions"] = _descriptions_of(row)
        copy["aliases"] = list(dict.fromkeys(str(alias) for alias in row.get("aliases") or ()))
        merged.append(copy)
        if delta_id:
            by_id[delta_id] = copy
        appended_count += 1

    # Fresh human-readable ids for the rows that were appended, laid out after the old ones.
    existing = [int(row.get("human_readable_id") or 0) for row in old_rows]
    next_id = (max(existing) + 1) if existing else 0
    for row in merged[len(old_rows) :]:
        row["human_readable_id"] = next_id
        next_id += 1

    merged.sort(key=lambda row: str(row.get("entity_id")))
    return merged, {
        "merged": merged_count,
        "appended": appended_count,
        "mapping": mapping,
    }
