"""Motivation chains for the inference layer (§5.20 ``motivation_chains.jsonl``).

Method, from F01-MotiveBench-2025
---------------------------------
The paper's contribution that matters here is a taxonomy, not a model:

> Maslow's hierarchy of needs theory ... suggests that human actions are driven by
> various needs, which are divided into five levels: Physiological Needs, Safety Needs,
> Love and Belonging Needs, Esteem Needs, and Self-actualization Needs.

and it validated that taxonomy with heavy human annotation (five annotators per item,
93% agreement with the predefined answers). The closure is the point: a free-text
"deep need" cannot be checked by anyone, while "which of five levels" can be disagreed
with and adjudicated.

§5.20 adds two constraints of its own: the chain must carry an **evidence chain**, and
because the paper shows LLM motivational reasoning is weak, the result is an
**editorial recommendation**, never a fact.

Why this one is opt-in and guarded
----------------------------------
It is the only inference table that calls a model. §5.20 already says this is a
recommendation; the code says it again in ``note`` and refuses to emit anything when no
model is available rather than filling the table with guesses.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Mapping, Sequence

from cbb_stage1 import ids

__all__ = [
    "MASLOW_LEVELS",
    "METHOD",
    "build_motivation_chains",
    "parse_chain",
    "prompt_for_character",
]

logger = logging.getLogger(__name__)

#: Named so a reader can tell what produced the rows.
METHOD = "maslow_guided"

#: Closed set, translated from F01's five levels. Free text here would be unverifiable.
MASLOW_LEVELS = ("生理", "安全", "归属", "尊重", "自我实现")

PROMPT_TEMPLATE = """\
下面是一个小说人物的若干「行为」，每条都带原文引文。请为每条行为判断：

1. 直接动机（一句话，说明他为什么这样做）；
2. 这个动机对应马斯洛需求的哪一层，**只能从这五个里选**：{levels}。

规则：
- 只依据给出的引文判断，不要补充原文没有的信息；
- 如果某条行为的信息不足以判断动机，`needs_level` 填 null，不要猜；
- 只输出 JSON 数组，每项形如
  {{"behavior_index": 0, "motive": "……", "needs_level": "安全"}}

人物：{character}

行为：
{behaviors}
"""


def prompt_for_character(character: str, behaviors: Sequence[str]) -> str:
    lines = "\n".join(
        f"- index {index}: 「{behavior}」" for index, behavior in enumerate(behaviors)
    )
    return PROMPT_TEMPLATE.format(
        levels=" / ".join(MASLOW_LEVELS),
        character=character,
        behaviors=lines or "（无）",
    )


def parse_chain(raw: Any, behaviors: Sequence[str]) -> list[dict[str, Any]]:
    """Turn a model answer into links, dropping anything outside the closed set.

    A ``needs_level`` that is not one of the five is dropped rather than coerced: the
    whole reason the set is closed is that a near-miss label means the model did not
    follow the taxonomy, and quietly rounding it would hide that.
    """
    text = str(raw or "").strip()
    start = text.find("[")
    end = text.rfind("]")
    payload: Any = None
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start: end + 1])
        except (ValueError, TypeError):
            payload = None
    if not isinstance(payload, list):
        return []

    links: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        try:
            index = int(item.get("behavior_index"))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(behaviors) or index in seen:
            continue
        level = str(item.get("needs_level") or "").strip()
        if level not in MASLOW_LEVELS:
            continue
        motive = str(item.get("motive") or "").strip()
        if not motive:
            continue
        seen.add(index)
        links.append(
            {
                "behavior": behaviors[index],
                "motive": motive,
                "needs_level": level,
            }
        )
    return links


def _character_behaviors(
    assertions: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """character -> behaviours with their evidence quotes, keyed by the subject role.

    A behaviour is an assertion whose subject is that character. Only rows that already
    carry a verbatim quote are kept: an evidence chain is required by §5.20, and a link
    without a quote is not a chain.

    No speaker filter is needed, and that is worth stating because a `speakers: {}` dict
    briefly existed here for one. It was useless three times over: never assigned, so its
    predicate was always true; and even had it been filled, the two sides live in
    different key spaces — this function keys on the ``主体``/``施事`` role, which holds
    entity IDs (``e:spk13:concept:...``), while ``assertion.speaker.name`` holds display
    names (``阿尔法``). A character who only ever speaks never becomes a key here, because
    speaking is not a subject role, so "exclude pure speakers" already holds by
    construction rather than by filtering.
    """
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assertions:
        roles = row.get("roles") or {}
        subject = str(roles.get("主体") or roles.get("施事") or "").strip()
        if not subject:
            continue
        for evidence in row.get("evidence") or ():
            quote = str(evidence.get("quote") or "").strip()
            if not quote:
                continue
            by_character[subject].append(
                {
                    "unit_id": str(evidence.get("unit_id") or ""),
                    "quote": quote,
                    "quote_sha256": evidence.get("quote_sha256"),
                }
            )
            break
    # Only characters with at least one quoted behaviour can form a chain.
    return {name: rows for name, rows in by_character.items() if rows}


def build_motivation_chains(
    book_id: str,
    assertions: Sequence[Mapping[str, Any]],
    *,
    llm: Any = None,
    entity_names: Mapping[str, str] | None = None,
    max_characters: int = 10,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """One chain per character, or nothing when there is no model.

    ``llm`` must expose ``generate(prompt) -> str``. Without it the function returns
    ``[]`` rather than a table of empty chains — an empty table is honest, and a row
    whose ``links`` are invented would violate both §5.20 and the closed needs set.

    ``entity_names`` maps entity ID to canonical name. It is optional so the function
    stays pure, but a caller that omits it gets chains keyed by entity ID.
    """
    if llm is None:
        logger.info("motivation chains skipped: no model available")
        return []

    behaviours = _character_behaviors(assertions)
    # `roles.主体` holds an entity ID, not a name. Measured: a chain came out keyed by
    # "e:spk13:concept:03d9cce37634", which is unusable to a reader even though it
    # satisfies the schema's `minLength: 1`. Resolve when a map is supplied; when it is
    # not, the ID is kept rather than guessed, and `note` says the name was unresolved.
    names = entity_names or {}
    rows: list[dict[str, Any]] = []
    for subject in sorted(behaviours)[:max_characters]:
        character = names.get(subject) or subject
        resolved = character != subject
        evidence_rows = behaviours[subject]
        quotes = [row["quote"] for row in evidence_rows]
        prompt = prompt_for_character(character, quotes)
        try:
            answer = llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - one character must not abort the stage
            logger.warning("motivation chain failed for %s: %s", character, exc)
            continue
        links = parse_chain(answer, quotes)
        if not links:
            continue
        for link in links:
            matched = next(
                (row for row in evidence_rows if row["quote"] == link["behavior"]),
                evidence_rows[0],
            )
            link["evidence"] = [matched]
        rows.append(
            {
                "chain_id": ids.motivation_chain_id(book_id, character),
                "book_id": book_id,
                "character": character,
                "links": links,
                "method": METHOD,
                "note": (
                    "Maslow 五层闭集（F01 口径）；§5.20 定为编辑推荐，非事实；"
                    "needs_level 为闭集外的一律丢弃，不四舍五入"
                    + ("" if resolved else f"；主体名未解析，保留实体 ID {subject}")
                ),
                "produced_by": "derive:motivation",
                "created_at": created_at,
            }
        )
    return rows
