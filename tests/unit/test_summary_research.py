from __future__ import annotations

from cbb_stage1.derive.research import build_research
from cbb_stage1.derive.summarize import synthesize_tree_texts, synthesize_unit_summaries


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "合并" in prompt:
            return "卷级总览。"
        return "章级摘要。"


def test_summary_synthesis_fills_unit_and_ancestors():
    units = [
        {"unit_id": "u:1", "order": 1, "volume": 1},
        {"unit_id": "u:2", "order": 2, "volume": 1},
    ]
    llm = _FakeLLM()
    unit_summaries = synthesize_unit_summaries(
        units, {"u:1": "第一章原文。", "u:2": "第二章原文。"}, llm
    )
    assert unit_summaries == {"u:1": "章级摘要。", "u:2": "章级摘要。"}
    rows = [
        {"summary_id": "sm:book", "scope": "book", "level": 0, "parent_summary_id": None},
        {"summary_id": "sm:vol", "scope": "volume", "level": 1, "parent_summary_id": "sm:book"},
        {"summary_id": "sm:u1", "scope": "chapter", "level": 2, "parent_summary_id": "sm:vol", "text": "章级摘要。"},
        {"summary_id": "sm:u2", "scope": "chapter", "level": 2, "parent_summary_id": "sm:vol", "text": "章级摘要。"},
    ]
    filled = synthesize_tree_texts(rows, llm)
    by_id = {row["summary_id"]: row for row in filled}
    assert by_id["sm:vol"]["text"] == "卷级总览。"
    assert by_id["sm:book"]["text"] == "卷级总览。"


def test_summary_without_llm_stays_empty():
    units = [{"unit_id": "u:1", "order": 1, "volume": 1}]
    assert synthesize_unit_summaries(units, {"u:1": "原文"}, None) == {}


def test_research_aggregates_cue_hypothesis_conflict():
    rows = build_research(
        "b",
        cues=[{"cue_id": "cue:1", "expected": "影子来源", "status": "unresolved", "setup": {"unit_id": "u:1", "quote": "影子"}}],
        hypotheses=[{"hypothesis_id": "hy:1", "claim": "能力来自契约", "status": "candidate", "missing_premises": ["契约场景"], "support": []}],
        conflicts=[{"conflict_id": "c:1", "detail": "时间冲突", "status": "open"}],
    )
    assert len(rows) == 3
    assert {row["kind"] for row in rows} == {"cue", "hypothesis", "conflict"}
    assert all(row["research_id"].startswith("rs:") for row in rows)
