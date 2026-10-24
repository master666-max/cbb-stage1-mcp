"""Batch entity merging: fan-out cleaning, then type-bucketed LLM merge.

Ported from ``库/code/B-项目案例/B07-lorebase/scripts/02b_merge_dict.py`` (MIT-era project of this series),
which carries reasoning the papers do not:

    "旧版 union-find 传递闭包会把泛称词(神/大人/舰长)当桥接, 造成巨型实体(实测吸收 189 别名)"

The order is therefore not a style choice. Exact-name merging plus transitive closure will absorb the whole
cast once one bridging word (大人/神/舰长) is shared, so **fan-out cleaning runs before any merging** and
the LLM is asked only about what survives.

Two measured facts are kept verbatim, because both are failure modes that come back silently:

* ``BATCH = 80`` — the source annotates "150 会让 M3 思考爆炸顶爆 max_tokens, 输出 0 组". A larger batch
  does not fail loudly; it returns zero groups.
* the merge prompt's four rules — at least two candidates, prefer missing a merge over a wrong one,
  different generations of one lineage are different entities, and no cross-type merges.

The LLM is injected as ``(prompt) -> str`` so nothing here needs a transport.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BLACKLIST",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_FANOUT",
    "MERGE_RULES",
    "assemble",
    "batch_merge",
    "build_batches",
    "fanout_clean",
    "load_name_merged",
    "merge_prompt",
    "normalize_name",
    "parse_merge_groups",
]

#: A word appearing in more entities than this is a generic term, not a name. The source's value, and its
#: comment records why it exists: one bridging word once absorbed 189 aliases into a single entity.
DEFAULT_FANOUT = 3

#: One LLM call's candidate count. The source measured that 150 overflows max_tokens and yields zero
#: groups — a silent failure, which is why the number is carried rather than chosen.
DEFAULT_BATCH_SIZE = 80

#: Generic honorifics and kinship terms, carried from the source. Fan-out catches the frequent ones; this
#: list catches a generic word that happens to be rare in a given book.
BLACKLIST: frozenset[str] = frozenset(
    {
        "大人", "阁下", "陛下", "国王陛下", "神", "将军", "国王", "老弟", "大哥",
        "小姐姐", "小妹妹", "孩子", "小孩", "少年", "老者", "父亲", "父亲大人",
        "父王", "爸", "妈妈", "盟主", "舰长", "船长", "司令", "长官", "首领",
        "老师", "美男子", "赌徒", "骗子手", "天人", "外星人", "那孩子", "那个少年",
        "公主", "王子", "老女人", "骗子", "盟主陛下",
    }
)

#: The four rules the merge prompt enforces. Kept as data so `parse_merge_groups` can be tested against the
#: same contract the prompt states, rather than the two drifting apart.
MERGE_RULES: tuple[str, ...] = (
    "只输出发生了合并的组（≥2 个候选）",
    "不确定是否同指的，宁可漏合不要错合",
    "同一轮回/世系下的不同代际个体是不同实体，不可合并；抽象概念与它的具体 incarnation 也应区分",
    "种族/组织/人物/地点是不同类型，不可跨类型合并",
)

_MAX_NAME_CHARS = 40


def normalize_name(value: Any) -> str:
    """Collapse whitespace; the source's ``norm``."""
    return re.sub(r"\s+", "", str(value or ""))


def load_name_merged(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge rows that share a name, accumulating frequency and aliases.

    A name longer than 40 characters is skipped: in this corpus those are sentence fragments captured as
    entities, and merging them by exact name produces nothing useful.
    """
    entries: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = normalize_name(row.get("name"))
        if not name or len(name) > _MAX_NAME_CHARS:
            continue
        aliases = {
            normalize_name(alias)
            for alias in row.get("aliases") or ()
            if 0 < len(normalize_name(alias)) <= _MAX_NAME_CHARS
        }
        entry = entries.setdefault(
            name,
            {"type": row.get("type") or "concept", "desc": "", "freq": 0, "aliases": set()},
        )
        entry["freq"] += int(row.get("freq") or 1)
        entry["aliases"] |= aliases
        desc = str(row.get("desc") or "")
        if len(desc) > len(entry["desc"]):
            entry["desc"] = desc
    return entries


def fanout_clean(
    rows: Iterable[Mapping[str, Any]], *, fanout: int = DEFAULT_FANOUT
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Drop generic terms, then strip them from the survivors' aliases.

    Returns ``(kept, generic)``. The generic set is returned rather than hidden: a word removed here is a
    judgement about this book, and a reader auditing a thin entity list needs to see it.
    """
    entries = load_name_merged(rows)

    hosts: dict[str, set[str]] = {}
    for name, entry in entries.items():
        for surface in {name} | set(entry["aliases"]):
            hosts.setdefault(surface, set()).add(name)

    generic = {surface for surface, owners in hosts.items() if len(owners) > int(fanout)}
    generic |= set(BLACKLIST)

    kept: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        if name in generic:
            continue
        entry = dict(entry)
        entry["aliases"] = {
            alias for alias in entry["aliases"] if alias not in generic and alias != name
        }
        kept[name] = entry
    return kept, generic


def build_batches(
    entries: Mapping[str, Mapping[str, Any]], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> list[dict[str, Any]]:
    """Bucket candidates by type (largest bucket first), then by frequency, then chunk.

    Type bucketing is not cosmetic: it lets the prompt's "no cross-type merges" rule be checked against the
    batch, instead of trusted to the model.
    """
    size = max(1, int(batch_size))
    by_type: dict[str, list[str]] = {}
    for name, entry in entries.items():
        by_type.setdefault(str(entry.get("type") or "concept"), []).append(name)

    batches: list[dict[str, Any]] = []
    for etype, names in sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        names.sort(key=lambda name: (-int(entries[name].get("freq") or 0), name))
        for start in range(0, len(names), size):
            batches.append({"type": etype, "names": names[start : start + size]})
    return batches


def merge_prompt(entries: Mapping[str, Mapping[str, Any]], batch_type: str) -> str:
    """The source's prompt, with its four rules stated in full."""
    lines: list[str] = []
    for name, entry in entries.items():
        aliases = ",".join(sorted(entry.get("aliases") or ())[:4])
        desc = str(entry.get("desc") or "")[:45]
        lines.append(f"{name}[{aliases}]: {desc} | {entry.get('type') or batch_type} | {entry.get('freq', 0)}")
    rules = "\n".join(f"- {index}. {rule}" for index, rule in enumerate(MERGE_RULES, 1))
    return (
        "任务: 实体归并。同一对象在不同章节可能以不同名字被抽出，"
        "请把下列候选中**确定指向同一对象**的归并成组。\n\n"
        f"规则(重要):\n{rules}\n\n"
        "canonical 选最有辨识度的通名（而非上下文短语如“某某的父亲”）。\n\n"
        "候选列表(主名[别名样本]: 描述 | 类型 | 出现次数):\n"
        + "\n".join(lines)
        + '\n\n只输出 JSON 数组: [{"canonical": "主名", "type": "类型", "desc": "一句话描述", '
        '"merged": ["被并入的主名", ...]}]'
    )


def parse_merge_groups(
    raw: Any,
    entries: Mapping[str, Mapping[str, Any]],
    batch_type: str,
) -> list[dict[str, Any]]:
    """Keep only groups the prompt's rules allow, and only candidates from this batch.

    A group is rejected when it is shorter than two, mixes types, or names something that was not offered —
    the last one matters because a model that invents a candidate would otherwise delete a real entity that
    merely shared a name.
    """
    text = str(raw or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []

    groups: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        canonical = normalize_name(item.get("canonical"))
        merged = [normalize_name(name) for name in item.get("merged") or () if normalize_name(name)]
        if not canonical or len(merged) < 2:
            continue
        members = list(dict.fromkeys([canonical, *merged]))
        if any(member not in entries for member in members):
            continue
        declared = str(item.get("type") or batch_type)
        if declared != batch_type:
            continue
        if any(str(entries[member].get("type") or "") != batch_type for member in members):
            continue
        groups.append(
            {
                "canonical": canonical,
                "type": batch_type,
                "desc": str(item.get("desc") or ""),
                "merged": members,
            }
        )
    return groups


def assemble(
    entries: Mapping[str, Mapping[str, Any]],
    merges: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply merge groups: accumulate frequency, union aliases, keep the longer description, delete the rest.

    A group whose canonical is not present falls back to its first surviving member rather than being
    dropped — losing a group loses more than a mislabelled canonical.
    """
    final: dict[str, dict[str, Any]] = {
        name: {
            "type": entry.get("type"),
            "desc": str(entry.get("desc") or ""),
            "freq": int(entry.get("freq") or 0),
            "aliases": set(entry.get("aliases") or ()),
        }
        for name, entry in entries.items()
    }
    for group in merges:
        members = [str(name) for name in group.get("merged") or ()]
        canonical = str(group.get("canonical") or "")
        if canonical not in final:
            canonical = next((name for name in members if name in final), "")
            if not canonical:
                continue
        target = final[canonical]
        for member in members:
            if member == canonical or member not in final:
                continue
            source = final[member]
            target["freq"] += source["freq"]
            target["aliases"] |= source["aliases"] | {member}
            if len(source["desc"]) > len(target["desc"]):
                target["desc"] = source["desc"]
            del final[member]
        if group.get("desc") and not target["desc"]:
            target["desc"] = str(group["desc"])
    return final


def batch_merge(
    rows: Sequence[Mapping[str, Any]],
    llm: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fanout: int = DEFAULT_FANOUT,
) -> dict[str, Any]:
    """Run the whole pass: fan-out clean, bucket, ask in batches, assemble.

    A failing batch is recorded and skipped: one bad call must not lose a dictionary that took the whole
    book to build.
    """
    kept, generic = fanout_clean(rows, fanout=fanout)
    batches = build_batches(kept, batch_size=batch_size)
    groups: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, batch in enumerate(batches):
        subset = {name: kept[name] for name in batch["names"]}
        response = ""
        try:
            response = str(llm(merge_prompt(subset, str(batch["type"]))) or "")
        except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the dictionary
            failures.append(f"batch {index} ({batch['type']}): {type(exc).__name__}: {exc}")
            continue
        groups.extend(parse_merge_groups(response, kept, str(batch["type"])))
    final = assemble(kept, groups)
    final_rows = [
        {
            "name": name,
            "type": entry["type"],
            "desc": entry["desc"],
            "aliases": sorted(entry["aliases"]),
            "freq": entry["freq"],
        }
        for name, entry in final.items()
    ]
    final_rows.sort(key=lambda row: (-row["freq"], row["name"]))
    return {
        "final": final_rows,
        "groups": groups,
        "generic": sorted(generic),
        "candidates": len(kept),
        "batches": len(batches),
        "failures": failures,
    }
