"""The reduce stage: a parent community summarises its children.

`derive/community.py` summarised every row from the raw assertions, so a level-2 community re-read evidence its
children had already condensed. That makes the tree decorative: nothing consumes it from the top, which is also
why D13's "global view" had no path.

GraphRAG's global search is map-reduce over community reports for this reason, and the same split applies here:

* **map** — a level-1 community reads evidence and produces a title/summary (unchanged);
* **reduce** — a level-2 community reads its children's summaries and produces its own.

Two rules that keep the reduce honest:

* the parent prompt must **not** carry the raw quotes; if it did, the children would be pointless and the
  result would be a re-derivation wearing a hierarchy's shape;
* when a child has no summary yet, the parent may fall back to evidence but must **say so in the prompt** — a
  combined summary and a re-derived one are different objects, and a reader cannot tell them apart otherwise.

Ordering is deterministic (children before parents, ids sorted) so rebuilding a box reproduces the same prompts.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "REDUCE_MARKER",
    "build_reduce_prompt",
    "summarise_tree",
]

#: Appears in a reduce prompt, so a caller can tell the two stages apart.
REDUCE_MARKER = "更高层社区"

_REDUCE_TEMPLATE = (
    "下面是若干下级社区的摘要。请写一句中文标题和一段客观摘要，描述它们合起来说明的更高层社区。\n"
    "只能使用下面给出的下级摘要，不得补充原文没有的事实；下级之间矛盾时如实并列，不要调和。\n"
    '只输出 JSON: {{"title":"...","summary":"..."}}\n'
    "下级社区：\n{children}"
)

_FALLBACK_NOTE = "（该下级暂无摘要，回退其证据引文如下）"


def _child_line(child: Mapping[str, Any]) -> str:
    title = str(child.get("title") or f"社区{child.get('community_id')}")
    summary = str(child.get("summary") or "").strip()
    if summary:
        return f"- {title}：{summary}"
    return f"- {title}：{_FALLBACK_NOTE}"


def _evidence_for(child: Mapping[str, Any], assertions: Mapping[str, Any]) -> list[str]:
    quotes: list[str] = []
    for assertion_id in child.get("relationship_ids") or ():
        payload = assertions.get(str(assertion_id))
        if isinstance(payload, Mapping):
            quote = payload.get("quote")
            if quote:
                quotes.append(str(quote))
    return quotes


def build_reduce_prompt(
    parent: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    assertions: Mapping[str, Any],
) -> str:
    """The reduce prompt: children's summaries, plus evidence only where a child has none."""
    lines: list[str] = []
    for child in children:
        line = _child_line(child)
        if _FALLBACK_NOTE in line:
            quotes = _evidence_for(child, assertions)
            if quotes:
                line = f"{line} {' / '.join(quotes)}"
        lines.append(line)
    return _REDUCE_TEMPLATE.format(children="\n".join(lines) or "（无）")


def summarise_tree(
    rows: Sequence[Mapping[str, Any]],
    *,
    summarizer: Callable[[str], Any],
    assertions: Mapping[str, Any],
) -> list[str]:
    """Summarise children first, then their parents; returns the prompts in the order they were sent.

    Rows are grouped by level and processed level by level, so a parent always sees finished children. A parent
    whose children are all missing is left alone rather than summarised from nothing.
    """
    # References, not copies: `_apply` writes the title/summary back onto these dicts and the caller
    # relies on it. Copying here silently discarded every summary.
    work: list[dict[str, Any]] = list(rows)  # type: ignore[arg-type]
    by_id = {str(row.get("community_id")): row for row in work}
    children_of: dict[str, list[dict[str, Any]]] = {}
    for row in work:
        parent_id = row.get("parent_id")
        if parent_id is not None and str(parent_id) in by_id:
            children_of.setdefault(str(parent_id), []).append(row)

    prompts: list[str] = []
    # Sort by level so children are always done before a parent is considered.
    for row in sorted(work, key=lambda item: (int(item.get("level") or 0), str(item.get("community_id")))):
        key = str(row.get("community_id"))
        children = children_of.get(key)
        if children:
            children = sorted(children, key=lambda item: str(item.get("community_id")))
            prompt = build_reduce_prompt(row, children, assertions)
        else:
            prompt = _map_prompt(row, assertions)
        prompts.append(prompt)
        response = summarizer(prompt)
        _apply(row, response)
    return prompts


def _map_prompt(row: Mapping[str, Any], assertions: Mapping[str, Any]) -> str:
    """The unchanged map stage: a level-1 community reads its evidence."""
    lines: list[str] = []
    for assertion_id in row.get("relationship_ids") or ():
        payload = assertions.get(str(assertion_id))
        if isinstance(payload, Mapping):
            quote = payload.get("quote")
            predicate = payload.get("predicate")
            lines.append(f"- {predicate}: {quote}")
    return (
        "用一句中文标题和一段客观摘要描述这个实体社区；只能使用下面的断言/引文，不得补充原文没有的事实。"
        '只输出 JSON: {"title":"...","summary":"..."}\n'
        f"成员：{', '.join(str(entity) for entity in row.get('member_entities') or ())}\n"
        "相关证据：\n" + ("\n".join(lines) or "（无）")
    )


def _apply(row: dict[str, Any], response: Any) -> None:
    """Write a title/summary back onto the row, ignoring anything unusable."""
    if isinstance(response, Mapping):
        row["title"] = response.get("title") or row.get("title")
        row["summary"] = response.get("summary") or row.get("summary")
        return
    text = str(response or "").strip()
    if not text:
        return
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return
    import json

    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return
    if isinstance(payload, Mapping):
        row["title"] = payload.get("title") or row.get("title")
        row["summary"] = payload.get("summary") or row.get("summary")
