"""Prompt assets and prompt builders for the AI Reader engine.

Prompt text is copied verbatim from B06-AI-Reader-V2
(``backend/src/extraction/prompts/`` and ``prescan_prompts.py``); only the
loading path and the injection helpers are new.  The anti-fabrication and
alias-ban wording is intentionally kept in Chinese and unchanged (design/02
§4.1: "保持 AI Reader 的反编造/别名禁令 prompt 原文").
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from cbb_stage1.engines.ai_reader.schema import EntityDictEntry

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── Loaders ──────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """The extraction system prompt (``extraction_system.txt``, verbatim)."""
    return (_PROMPTS_DIR / "extraction_system.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_vot_guide() -> str:
    """Spatial-reasoning guide; empty string when the asset is missing."""
    path = _PROMPTS_DIR / "vot_spatial_guide.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


@lru_cache(maxsize=1)
def load_examples() -> list[dict]:
    """Few-shot examples. Callers must treat the list as read-only."""
    return json.loads((_PROMPTS_DIR / "extraction_examples.json").read_text(encoding="utf-8"))


# ── Extraction prompt builders (ported from chapter_fact_extractor) ──


def build_example_text(examples: list[dict], context_window: int) -> str:
    """Few-shot block; only 1 example on small (≤16K) context windows."""
    if not examples:
        return ""
    examples_to_show = [examples[0]]
    if len(examples) >= 4 and context_window > 16384:
        examples_to_show.append(examples[3])
    examples_json = json.dumps(examples_to_show, ensure_ascii=False, indent=2)
    return f"## 参考示例\n```json\n{examples_json}\n```\n\n"


def build_user_prompt(
    chapter_id: int,
    chapter_text: str,
    example_text: str,
    segment_hint: str = "",
) -> str:
    """User prompt for a chapter or one of its segments (source wording)."""
    return (
        f"{example_text}"
        f"## 第 {chapter_id} 章{segment_hint}\n\n{chapter_text}\n\n"
        "【关键要求】\n"
        "1. characters：宁多勿漏！包含所有有名字或固定称呼的人物。种族/物种名称作为称呼且有具体行为的角色也算（如赤尻马猴、通背猿猴）\n"
        "2. relationships：任何两个人物有互动或提及关系都必须提取，evidence 引用原文。命令/差遣/听令也是关系\n"
        "3. locations：宁多勿漏！所有具体地名都必须提取，即使只被简短提及也不可跳过\n"
        "4. events：每个事件的 participants 列出参与者姓名，location 填写地点，都不可为空\n"
        "5. spatial_relationships：提取地点间的方位(direction)、距离(distance)、包含(contains)、相邻(adjacent)、分隔(separated_by)、地形(terrain)、夹在中间(in_between)、移动路径(travel_path)、规模对比(relative_scale)、聚类(cluster)关系。可选填 distance_class(near/medium/far/very_far) 和 confidence_score(0.0-1.0)\n"
        "6. world_declarations：当文中有世界宏观结构描述时必须提取（区域划分region_division、区域方位region_position、空间层layer_exists如天界/地府/海底、传送通道portal），没有则输出空列表\n"
        "7. new_concepts：功法、丹药、修炼体系、世界观规则等首次出现或有详细介绍的概念，definition 必须详细（2-5句话）\n"
        "8. 只提取原文明确出现的内容，禁止编造\n"
        "9. **speaker（这条主张是谁说的）**：每条记录的 evidence 若来自对白（「」）"
        "或内心独白（（）），在 `evidence_speaker` 填说这句话的人。"
        "这是**这条主张的归属人**（claim 的 metadata），不是要求每条证据都填。"
        "原文没有给出归属时（连续对白只有引号、没有任何「XX说」）留空，"
        "不要猜——错误的归属比没有归属更有害。\n"
        "10. **evidence 必填**：`characters` / `locations` / `new_concepts` 的每条记录"
        "**也必须填 evidence**（原文逐字片段）。这三类最容易被漏填，但规范要求"
        "`asserted` 记录必须有至少一条精确 evidence；没有 evidence 的记录会被"
        "隔离为 `evidence_not_found`，等于白抽。\n"
    )


def build_cloud_schema_suffix(schema: dict) -> str:
    """Schema block appended to the system prompt for JSON-mode-only APIs."""
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "\n\n## 输出 JSON Schema\n"
        "你必须严格按照以下 JSON Schema 输出，不要输出多余字段或文本：\n"
        f"```json\n{schema_text}\n```"
    )


def build_known_entities_section(
    known_entities: list | None = None,
    glossary_entities: list | None = None,
) -> str:
    """Render the card glossary / known_entities as a context section.

    Only names, types and aliases are injected — never a claim.  Alias hints are
    capped and marked tentative, same policy as the source dictionary section.
    """
    lines: list[str] = []
    seen: set[str] = set()

    def _add(name: str, etype: str, aliases: list[str] | None) -> None:
        name = (name or "").strip()
        if not name or name in seen:
            return
        seen.add(name)
        line = f"- {name}（{etype or 'unknown'}）"
        safe_aliases = [a for a in (aliases or []) if len(a) <= 4][:2]
        if safe_aliases:
            line += f" 可能别名：{'、'.join(safe_aliases)}（需本章原文确认）"
        lines.append(line)

    for entry in glossary_entities or []:
        if isinstance(entry, dict):
            _add(entry.get("name", ""), entry.get("type", ""), entry.get("aliases"))
    for entity in known_entities or []:
        name = getattr(entity, "canonical_name", None) or getattr(entity, "name", "")
        etype = getattr(entity, "type", "")
        _add(str(name), str(etype), list(getattr(entity, "aliases", []) or []))

    if not lines:
        return ""
    header = [
        "### 本书已知实体（词典参考，仅作提示）",
        "⚠️ 以下仅为预扫描参考；别名必须在本章原文中找到明确依据才能输出，"
        "不同的人绝不能互为别名。",
    ]
    return "\n".join(header + lines)


# ── Gleaning (continue prompt) ───────────────────────────────────

# Mirrors GraphRAG's CONTINUE_PROMPT / LightRAG's gleaning instruction, in
# Chinese so it composes with the extraction system prompt.
CONTINUE_PROMPT = (
    "\n\n【续抽】以上是已经从本章抽取出的结果。请再仔细检查原文一遍，"
    "补充遗漏的实体（characters/locations/new_concepts）、关系（relationships）、"
    "事件（events/org_events/item_events）与空间关系（spatial_relationships）。\n"
    "只输出新增项，格式与之前完全相同的 JSON；如果确实没有遗漏，各数组输出空列表。\n"
    "不要重复已抽取的内容，不要编造。"
)


def build_gleaning_user_prompt(
    chapter_id: int,
    chapter_text: str,
    example_text: str,
    existing_json: str,
) -> str:
    """User prompt for a continuation round (initial prompt + existing output)."""
    return (
        build_user_prompt(chapter_id, chapter_text, example_text)
        + f"\n\n## 已抽取结果（JSON）\n```json\n{existing_json}\n```"
        + CONTINUE_PROMPT
    )


# ── Pre-scan Phase 2 classification (ported from prescan_prompts.py) ──

PRESCAN_SYSTEM_PROMPT = """\
你是一个小说实体分类专家。你的任务是对从一部小说中自动提取的高频专有名词候选列表进行分类。

## 任务
1. 对每个候选词判断其实体类型：person（人物）、location（地点）、item（物品/功法/武器）、org（组织/门派）、concept（概念/境界）
2. 识别可能的别名关系（同一实体的不同称呼）
3. 将明显不是实体的词标记为 rejected

## 输出格式
输出严格的 JSON，包含三个字段：
{
  "entities": [
    {"name": "韩立", "type": "person", "confidence": "high"},
    {"name": "七玄门", "type": "org", "confidence": "high"}
  ],
  "alias_groups": [
    ["孙悟空", "行者", "齐天大圣", "猴王"],
    ["唐僧", "三藏", "唐三藏", "御弟"]
  ],
  "rejected": ["然后", "不过", "自己"]
}

## 注意事项
- entities 中每个词只出现一次
- alias_groups 中的词必须是 entities 中已有的词
- 不确定的词宁可保留（标为 unknown），不要轻易 reject
- confidence: high（确定是实体）、medium（可能是实体）、low（不确定）
- 只输出 JSON，不要输出其他内容

## 别名分组注意
- 同姓不等于同人！"韩立"和"韩胖子"是不同角色（恰好同姓），绝对不要放在同一个 alias_group
- 只有明确是同一人的不同称呼才放一组（如"孙悟空/行者/齐天大圣"）
- 亲属关系（父子、叔侄）不是别名关系"""


def build_prescan_prompt(
    candidates: list[EntityDictEntry],
    max_candidates: int = 150,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for LLM entity classification."""
    top = candidates[:max_candidates]
    lines = ["以下是从小说中自动提取的高频专有名词候选列表，请进行分类：", ""]
    lines.append("| 词 | 出现频次 | 来源 | 上下文示例 |")
    lines.append("|---|---------|------|----------|")
    for entry in top:
        ctx = (entry.sample_context or "")[:50].replace("|", "\\|")
        lines.append(f"| {entry.name} | {entry.frequency} | {entry.source} | {ctx} |")
    return PRESCAN_SYSTEM_PROMPT, "\n".join(lines)
