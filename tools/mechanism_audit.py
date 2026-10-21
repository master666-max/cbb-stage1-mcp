"""Mechanism coverage audit — recompute the inventory's status from the tree.

The 2026-10-20 line-by-line check was written by hand and went stale within a day: it
recorded "语义熵拒答 ❌ 未实现" and "D15 动机/情绪/主题 ❌ 未实现" while the tree gained
both. A status table maintained by hand drifts; this script recomputes, so the table can
be regenerated instead of trusted.

How a row is judged, and what the verdict does NOT mean
--------------------------------------------------------
Each item names **evidence**: file globs plus a regex that must match inside them.

    PRESENT  every listed pattern matched — code exists and is reachable
    PARTIAL  some matched
    MISSING  none matched

That is a **presence** check. It answers "is there code for this", not "is this complete".
The first run of this script reported 81/81 PRESENT with zero gaps, and that number would
be a lie of the exact kind this project keeps getting burned by: most patterns are broad
(`def `), so any file answers them. Presence is necessary and nowhere near sufficient.

Two things keep the output honest:

* :attr:`Item.limited` — items known to be present-but-limited, with the reason (white-box
  signals the transport cannot supply, a proxy standing in for the paper's method). These
  render as ``PRESENT (limited: …)`` and are counted separately.
* :attr:`Item.excluded` — items the inventory itself rules out, with its own reason. They
  are not gaps and are not counted as coverage.

What remains unproven by this script is behaviour: whether the code is correct is what the
test suite answers, and the two are meant to be read together, not one instead of the
other.

Items that the inventory itself marks as deliberately excluded (段二 / 查无实据 / 不抄) are
carried as ``EXCLUDED`` with the inventory's own reason, so they are not counted as gaps.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["EXCLUDED", "ITEMS", "audit", "render"]

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src" / "cbb_stage1"


@dataclass(frozen=True)
class Item:
    ident: str
    name: str
    #: (relative glob, regex) pairs; every pattern must hit for OK
    evidence: tuple[tuple[str, str], ...] = ()
    #: set when the inventory itself excludes the item, with its reason
    excluded: str = ""
    #: set when the code exists but cannot reach its full capability here, with the reason
    limited: str = ""


ITEMS: tuple[Item, ...] = (
    # ── A 抽取管线 ──────────────────────────────────────────────────────────
    Item("A1", "两段式：先词典/别名，再带词典抽", (
        ("src/cbb_stage1/engines/ai_reader/entity_pre_scanner.py", r"pre_?scan"),
        ("src/cbb_stage1/orchestrator/pipeline.py", r"prescan|pre_scan"),
    )),
    Item("A2", "章级十类 schema", (("src/cbb_stage1/engines/ai_reader/schema.py", r"class \w+Fact|EVIDENCE_DESCRIPTION"),)),
    Item("A3", "Pydantic + minItems", (("src/cbb_stage1/engines/ai_reader/schema.py", r"Field\("),)),
    Item("A4", "反编造 + 别名禁令", (("src/cbb_stage1/engines/ai_reader/prompts.py", r"不要编造|不得编造|反编造"),)),
    Item("A5", "批处理 + 暂停/恢复", (("src/cbb_stage1/orchestrator/checkpoint.py", r"checkpoint|resume"),)),
    Item("A6", "AliasResolver Union-Find", (("src/cbb_stage1/resolve/", r"union|find\("),)),
    Item("A7", "10 类语义清洗 + 6 类冲突", (
        ("src/cbb_stage1/fact_validator.py", r"def "),
        ("src/cbb_stage1/validate/conflicts.py", r"detect_attribute_conflicts"),
        ("src/cbb_stage1/validate/conflicts.py", r"detect_relation_conflicts"),
        ("src/cbb_stage1/validate/conflicts.py", r"detect_timeline_conflicts"),
    )),
    Item("A8", "LPMM NER→RDF", (("src/cbb_stage1/engines/lpmm/", r"def "),)),
    Item("A9", "并发/重试/json_repair", (
        ("src/cbb_stage1/engines/ai_reader/openie.py", r"retry|repair"),
        ("src/cbb_stage1/engines/lpmm/adapter.py", r"retry|repair"),
    )),
    Item("A10", "批次导入/删除", (("src/cbb_stage1/incremental/", r"def "),)),
    Item("A11", "chunk 先大后小", (("src/cbb_stage1/ingest/chunk.py", r"strategy|chunk"),)),
    Item("A12", "semhash/KNN 去重", (("src/cbb_stage1/dedupe/", r"embedding|cosine|knn"),)),
    Item("A13", "结构优先切块", (("src/cbb_stage1/ingest/chunk.py", r"structure|heading|chapter"),)),
    Item("A14", "扇出清洗 + LLM 归并", (("src/cbb_stage1/dedupe/", r"merge"),)),
    Item("A15", "Book→Chunk→Connections", (("src/cbb_stage1/orchestrator/pipeline.py", r"def run_pipeline"),)),
    Item("A16", "checkpoint + sha 幂等", (("src/cbb_stage1/orchestrator/idempotency.py", r"hash"),)),
    Item("A17", "Gleaning 续抽", (("src/cbb_stage1/engines/ai_reader/", r"glean"),)),
    Item("A18", "描述聚合", (("src/cbb_stage1/dedupe/describe.py", r"def "),)),
    Item("A19", "社区检测", (("src/cbb_stage1/derive/community.py", r"community"),)),
    Item("A20", "增量更新工作流", (("src/cbb_stage1/incremental/", r"def "),)),
    Item("A21", "gleaning 参数", (("src/cbb_stage1/engines/", r"gleaning|max_gleaning"),)),
    Item("A22", "文档状态机", (("src/cbb_stage1/orchestrator/", r"state"),)),
    Item("A23", "四种 chunk 策略", (("src/cbb_stage1/ingest/chunk.py", r"STRATEG"),)),
    Item("A24", "描述精确去重 + map-reduce", (("src/cbb_stage1/dedupe/", r"dedupe|unique"),)),
    Item("A25", "RAPTOR", (("src/cbb_stage1/derive/raptor.py", r"def "),)),
    Item("A26", "OpenIE 两段式", (("src/cbb_stage1/engines/", r"extraction_mode|openie"),)),
    Item("A27", "逐字证据字段", (("src/cbb_stage1/evidence/anchor.py", r"def make_evidence"),)),
    # ── B 验证/置信/推断 ───────────────────────────────────────────────────
    Item("B1", "CoVe 四步闭环", (
        ("src/cbb_stage1/validate/cove.py", r"def cove_verify"),
        ("src/cbb_stage1/validate/cove.py", r"def plan_verifications"),
        ("src/cbb_stage1/validate/cove.py", r"def answer_independently"),
        ("src/cbb_stage1/validate/cove.py", r"def compare_with_claim"),
        ("src/cbb_stage1/claims/integration.py", r"cove_verify"),
    )),
    Item("B2", "语义熵", (("src/cbb_stage1/confidence/semantic_entropy.py", r"def "),)),
    Item("B3", "SEPs（白盒）", (("src/cbb_stage1/confidence/sep.py", r"def "),)),
    Item("B4", "UQLM 三类", (("src/cbb_stage1/confidence/uqlm.py", r"def "),)),
    Item("B5", "UQ Heads", (("src/cbb_stage1/confidence/uq_heads.py", r"def "),)),
    Item("B6", "AVeriTeC 四值 + 证据充分性", (
        ("src/cbb_stage1/claims/verdicts.py", r"def evidence_sufficiency"),
        ("src/cbb_stage1/claims/integration.py", r"evidence_sufficiency"),
    )),
    Item("B7", "AFEV 原子分解", (
        ("src/cbb_stage1/reasoning/atomic.py", r"def afev_verify"),
        ("src/cbb_stage1/claims/integration.py", r"afev_verify"),
    )),
    Item("B8", "Abductive RAG", (("src/cbb_stage1/derive/hypotheses.py", r"def "),)),
    Item("B9", "VeGraph 消歧", (("src/cbb_stage1/reasoning/vegraph.py", r"def vegraph_verify"),)),
    Item("B10", "临床可审计片段", (("src/cbb_stage1/evidence/", r"span|quote"),)),
    Item("B11", "独立标注/多标注者 κ", (
        ("src/cbb_stage1/qc/agreement.py", r"def independent_label"),
        ("src/cbb_stage1/qc/agreement.py", r"def cohen_kappa"),
        ("tools/mcp_pipeline_server.py", r"cbb_annotator_agreement"),
    )),
    Item("B12", "语义熵拒答阈值", (
        ("src/cbb_stage1/claims/verdicts.py", r"abstain_decision|DEFAULT_ABSTAIN_THRESHOLD"),
        ("src/cbb_stage1/claims/integration.py", r"abstain_threshold"),
    )),
    Item("B13", "AVeriTeC 引文归属核验（新增）", (
        ("src/cbb_stage1/validate/speaker_verify.py", r"def verify_speaker_quote"),
        ("src/cbb_stage1/validate/integration.py", r"verify_speaker_quote"),
    )),
    # ── C 执行模型 ─────────────────────────────────────────────────────────
    Item("C1", "三层分工", (("src/cbb_stage1/agents/", r"role"),)),
    Item("C2", "文件交接 + 回执", (("src/cbb_stage1/orchestrator/", r"receipt"),)),
    Item("C3", "Verifier 独立上下文", (("src/cbb_stage1/agents/subagent.py", r"ROLE_PROMPTS"),)),
    Item("C4", "五角色", (("src/cbb_stage1/agents/", r"ROLE_PROMPTS"),)),
    Item("C5", "Summarizer 树", (("src/cbb_stage1/derive/summaries.py", r"level"),)),
    Item("C6", "纯算法不做 agent", (("src/cbb_stage1/derive/", r"def "),)),
    Item("C7", "状态目录可重建", (("src/cbb_stage1/orchestrator/", r"state"),)),
    Item("C8", "并发上限", (("src/cbb_stage1/model_gateway.py", r"max_concurrency"),)),
    Item("C9", "单写者 + 原子写", (("src/cbb_stage1/box/atomic.py", r"def "),)),
    Item("C10", "不可信输入", (("src/cbb_stage1/preflight.py", r"def "),)),
    Item("C11", "治理控制台（新增）", (
        ("tools/gov_console.py", r"def main"),
        ("src/cbb_stage1/validate/adjudication.py", r"def resolve_conflict"),
    )),
    Item("C12", "MCP 进程内管辖面（新增）", (
        ("tools/mcp_pipeline_server.py", r"def cbb_start_run"),
        ("src/cbb_stage1/model_ops.py", r"def call_once"),
    )),
    # ── D 闭环/增量/叙述 ───────────────────────────────────────────────────
    Item("D1", "叙述层 claim_layer/speaker/mode", (("src/cbb_stage1/normalize/facts.py", r"claim_layer"),)),
    Item("D2", "不可靠叙述者降级", (("src/cbb_stage1/assemble/assertions.py", r"narrator_reliability"),)),
    Item("D3", "置信传播", (("src/cbb_stage1/confidence/", r"def "),)),
    Item("D4", "模型异档", (("src/cbb_stage1/model_gateway.py", r"judge|distinct"),)),
    Item("D5", "词典演化", (
        ("src/cbb_stage1/assemble/lexicon_evolution.py", r"def approve_proposal"),
        ("src/cbb_stage1/assemble/lexicon_evolution.py", r"units_to_rescan"),
    )),
    Item("D6", "O(新章) 增量", (("src/cbb_stage1/incremental/", r"def "),)),
    Item("D7", "cue 状态机", (("src/cbb_stage1/cues/", r"def "),)),
    Item("D8", "Trigger 提前回收", (("src/cbb_stage1/cues/", r"trigger"),)),
    Item("D9", "三类矛盾 + superseded", (("src/cbb_stage1/validate/conflicts.py", r"def apply_supersede"),)),
    Item("D10", "时间轴", (("src/cbb_stage1/temporal/", r"def "),)),
    Item("D11", "时序 NLI", (("src/cbb_stage1/temporal/", r"nli"),)),
    Item("D12", "clue 一等实体", (("src/cbb_stage1/box/lexicon.py", r"clue|埋线"),)),
    Item("D13", "RAPTOR 全局视野", (("src/cbb_stage1/derive/raptor.py", r"def build_raptor"),)),
    Item("D14", "100-Endings 原法", (("src/cbb_stage1/derive/forecasts.py", r"no_rate"),)),
    Item("D15", "动机/情绪/主题/风格", (
        ("src/cbb_stage1/derive/stylometry.py", r"def build_stylometry"),
        ("src/cbb_stage1/derive/themes.py", r"def build_themes"),
        ("src/cbb_stage1/derive/emotion.py", r"def build_emotion_curves"),
        ("src/cbb_stage1/derive/motivation.py", r"def build_motivation_chains"),
    )),
    # ── E 箱与契约 ─────────────────────────────────────────────────────────
    Item("E1", "逐字证据门", (("src/cbb_stage1/evidence/anchor.py", r"def verify_evidence|def make_evidence"),)),
    Item("E2", "候选契约", (("src/cbb_stage1/engines/ai_reader/schema.py", r"class "),)),
    Item("E3", "可撤销合并日志", (
        ("src/cbb_stage1/resolve/merges.py", r"def "),
        ("src/cbb_stage1/resolve/merges.py", r"merge"),
    )),
    Item("E4", "派工卡 + sha", (("src/cbb_stage1/box/writer.py", r"card"),)),
    Item("E5", "判卷/票数晋升", (("src/cbb_stage1/claims/", r"promot"),)),
    Item("E6", "唯一写口 + 审计", (("src/cbb_stage1/box/writer.py", r"def build_box"),)),
    Item("E7", "deps/dirty/replay", (("src/cbb_stage1/orchestrator/", r"replay|deps"),)),
    Item("E8", "pass 注册表/门禁", (("src/cbb_stage1/qc/acceptance.py", r"def verify"),)),
    Item("E9", "箱不可变/版本/parent", (("src/cbb_stage1/box/writer.py", r"parent|FileExistsError"),)),
    Item("E10", "分层箱", (("src/cbb_stage1/box/writer.py", r"identity|knowledge|derived"),)),
    Item("E11", "claims/QA、atomic_facts", (
        ("src/cbb_stage1/claims/integration.py", r"afev_verify"),
        ("src/cbb_stage1/box/writer.py", r"claims\.jsonl"),
    )),
    Item("E12", "三种 ID + 引用闭合", (("src/cbb_stage1/ids.py", r"def assertion_id"),)),
    # ── F 段二/未进箱（清单自带的排除项）──────────────────────────────────
    Item("F1", "小说 wiki 六大节", excluded="段二 IA，不属抽取段"),
    Item("F2", "角色页/卷级中间层", excluded="段二 IA，不属抽取段"),
    Item("F3", "进阶六能力", (
        ("src/cbb_stage1/derive/stylometry.py", r"def build_stylometry"),
        ("src/cbb_stage1/derive/themes.py", r"def build_themes"),
        ("src/cbb_stage1/derive/emotion.py", r"def build_emotion_curves"),
        ("src/cbb_stage1/derive/motivation.py", r"def build_motivation_chains"),
        ("src/cbb_stage1/derive/forecasts.py", r"def "),
    )),
    Item("F4", "渐进式披露", (("src/cbb_stage1/ingest/chunk.py", r"order"),)),
    Item("F5", "UQ Heads ETH 标注", excluded="已修正为共同署名，非机制"),
    Item("F6", "MDPI 2025 九模型基准", excluded="清单判：查无实据"),
    Item("F7", "医疗时间轴指标", excluded="清单判：查无实据"),
    Item("F8", "Underwood 风格干扰", excluded="清单判：不吸收"),
    Item("F9", "LightRAG 截断 source_id", excluded="清单判：不抄（与逐字 span 冲突）"),
    Item("F10", "GraphRAG 大写去重", excluded="清单判：不抄（中文误合）"),
    Item("F11", "V5 27 把硬序列", excluded="清单判：不搬（会造 invalid 箱）"),
)


#: Items whose code exists but whose full capability is out of reach here. Stated as data
#: rather than inferred, because a regex cannot tell a working method from a stub.
LIMITED: dict[str, str] = {
    "A1": "预扫描器与 pipeline 的 prescan 已接线；「先扫一遍再带词典抽」的显式两段编排仍不完整",
    "B3": "需要隐状态/logprobs；主推理走 host 传输只回文本，喂 logprobs 才测得到",
    "B4": "白盒分支同上；黑盒分支可用（实测 conf=1.0 但那是常数，不是真信号）",
    "B5": "需要 attention；本地可训（8 样本 → 0.886），主推理路径拿不到",
    "D14": "只有词法离散度代理（method=lexical_singleton_ratio）；E07 原法 18 万次调用不可行",
    "B11": "代码与 MCP 工具齐备；跑出 κ 需要模型，无模型时显式 unavailable",
}


@dataclass
class Verdict:
    ident: str
    name: str
    status: str  # OK | PARTIAL | MISSING | EXCLUDED
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    reason: str = ""


def _files(pattern: str):
    """Resolve a glob against the project root, so ``tools/`` and ``src/`` both work.

    A first version resolved package-relative patterns against ``src/cbb_stage1`` and used a
    ``../tools/...`` prefix for the rest. That prefix produced ``V5/tools`` and reported
    every tools/ item as missing — three false gaps in the first run of this very script.
    """
    for path in PROJECT.glob(pattern):
        if path.is_file():
            yield path
            continue
        for child in path.rglob("*.py"):
            if "__pycache__" not in child.parts:
                yield child


def _check(glob_pattern: str, regex: str) -> str | None:
    pattern = re.compile(regex)
    for path in _files(glob_pattern):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = pattern.search(text)
        if match:
            line = text[: match.start()].count("\n") + 1
            return f"{path.relative_to(PROJECT)}:{line}"
    return None


def audit() -> list[Verdict]:
    results: list[Verdict] = []
    for item in ITEMS:
        if item.excluded:
            results.append(Verdict(item.ident, item.name, "EXCLUDED", reason=item.excluded))
            continue
        hits: list[str] = []
        misses: list[str] = []
        for glob_pattern, regex in item.evidence:
            found = _check(glob_pattern, regex)
            if found:
                hits.append(found)
            else:
                misses.append(f"{glob_pattern} ~ /{regex}/")
        if not item.evidence:
            status = "MISSING"
        elif not misses:
            status = "PRESENT"
        elif hits:
            status = "PARTIAL"
        else:
            status = "MISSING"
        results.append(
            Verdict(
                item.ident,
                item.name,
                status,
                hits,
                misses,
                reason=LIMITED.get(item.ident, ""),
            )
        )
    return results


def render(results: list[Verdict], *, as_json: bool = False) -> str:
    from collections import Counter

    counts = Counter(v.status for v in results)
    limited = [v for v in results if v.status == "PRESENT" and v.reason]
    if as_json:
        return json.dumps(
            {
                "counts": dict(counts),
                "items": [
                    {
                        "id": v.ident,
                        "name": v.name,
                        "status": v.status,
                        "hits": v.hits,
                        "misses": v.misses,
                        "reason": v.reason,
                    }
                    for v in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "=" * 78,
        "MECHANISM COVERAGE — recomputed from the tree, not copied from a report",
        "=" * 78,
        "",
        "  PRESENT means: code exists and is reachable. It does NOT mean complete.",
        "",
        f"  items: {len(results)}",
    ]
    for key in ("PRESENT", "PARTIAL", "MISSING", "EXCLUDED"):
        lines.append(f"    {key:9s} {counts.get(key, 0)}")
    lines.append(f"    {'LIMITED':9s} {len(limited)}   (present, but cannot reach full capability)")
    lines.append("")

    if limited:
        lines.append("── LIMITED (present, capability bounded) " + "-" * 28)
        for v in limited:
            lines.append(f"  {v.ident:5s} {v.name[:30]:30s}")
            lines.append(f"        {v.reason}")
        lines.append("")

    for key in ("MISSING", "PARTIAL"):
        group = [v for v in results if v.status == key]
        if not group:
            continue
        lines.append(f"── {key} " + "─" * 60)
        for v in group:
            lines.append(f"  {v.ident:5s} {v.name[:40]:40s}")
            for miss in v.misses[:3]:
                lines.append(f"        miss: {miss}")
        lines.append("")

    excluded = [v for v in results if v.status == "EXCLUDED"]
    if excluded:
        lines.append("── EXCLUDED (the inventory's own reasons) " + "─" * 34)
        for v in excluded:
            lines.append(f"  {v.ident:5s} {v.name[:34]:34s} {v.reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute mechanism coverage from the tree")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 when anything is MISSING")
    ns = parser.parse_args(argv)

    results = audit()
    print(render(results, as_json=ns.json))

    if ns.strict:
        missing = [v.ident for v in results if v.status == "MISSING"]
        if missing:
            print(f"\nMISSING: {missing}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
