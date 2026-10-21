"""LLM summary synthesis for the derived layer (v2.2 §5.14).

The tree shape is deterministic; this module only fills ``text`` from the source
text.  It never invents citations, and no LLM/no output leaves ``text=None``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

__all__ = ["synthesize_unit_summaries", "synthesize_tree_texts"]

_MAX_SOURCE_CHARS = 12000


def _clean(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^```(?:text|markdown)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def synthesize_unit_summaries(
    units: Iterable[Mapping[str, Any]],
    unit_texts: Mapping[str, str],
    llm: Any,
) -> dict[str, str]:
    """unit_id -> 2–4 sentence Chinese summary; missing LLM/empty -> {}."""
    if llm is None:
        return {}
    out: dict[str, str] = {}
    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        text = (unit_texts.get(unit_id) or "").strip()
        if not unit_id or not text:
            continue
        prompt = (
            "用 2–4 句中文概括下面这章/节的内容，只写正文，不加标题、不加评价、"
            "不补充原文没有的事实。\n\n"
            f"{text[:_MAX_SOURCE_CHARS]}"
        )
        try:
            summary = _clean(llm.generate(prompt))
        except Exception:  # noqa: BLE001 - no fabricated summary on failure
            continue
        if summary:
            out[unit_id] = summary
    return out


def synthesize_tree_texts(
    rows: Iterable[Mapping[str, Any]],
    llm: Any,
) -> list[dict[str, Any]]:
    """Fill volume/book rows from their child summaries (bottom-up)."""
    if llm is None:
        return [dict(row) for row in rows]
    result = [dict(row) for row in rows]
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in result:
        parent = row.get("parent_summary_id")
        if parent:
            by_parent.setdefault(str(parent), []).append(row)
    # Children are chapters (level 2), then volumes (level 1), then book (0).
    for row in sorted(result, key=lambda item: -int(item.get("level") or 0)):
        if int(row.get("level") or 0) not in (0, 1):
            continue
        children = [c for c in by_parent.get(str(row.get("summary_id")), []) if c.get("text")]
        if not children:
            continue
        joined = "\n".join(f"- {c['text']}" for c in children)[:_MAX_SOURCE_CHARS]
        prompt = (
            "把下面各章摘要合并成一段 2–4 句中文总览，只写正文，不添加新事实。\n\n"
            f"{joined}"
        )
        try:
            text = _clean(llm.generate(prompt))
        except Exception:  # noqa: BLE001
            continue
        if text:
            row["text"] = text
    return result
