"""B4 judge + D4 decoupling: a second model is what makes §5.15 satisfiable.

Under the agent-driven transport the extractor and the inline verifier are the
same agent by construction, so ``model_decoupled`` could never become true no
matter how the code was written. A dedicated judge model is the only way out: the
judgement then comes from a different model even though extraction and
verification still share one.

The tests also pin the negative cases, because a decoupling flag that flips on
anything is worse than one that never flips.
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.box.writer import _measure_model_decoupling


def _calls(tmp_path: Path, rows: list[dict]) -> Path:
    work = tmp_path / "work"
    (work / "plan").mkdir(parents=True, exist_ok=True)
    (work / "plan" / "model-calls.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return work


EXTRACT = {"kind": "chat", "role": "extraction", "model": "deepseek-v4.1"}
VERIFY = {"kind": "chat", "role": "verifier", "model": "deepseek-v4.1"}


def test_no_judge_and_shared_model_is_false(tmp_path: Path) -> None:
    work = _calls(tmp_path, [EXTRACT, VERIFY])
    decoupled, note = _measure_model_decoupling(work)
    assert decoupled is False
    assert "同档" in note
    assert "judge" in note, "the note must say how to fix it"


def test_distinct_judge_flips_it_to_true(tmp_path: Path) -> None:
    work = _calls(
        tmp_path,
        [EXTRACT, VERIFY, {"kind": "chat", "role": "judge", "model": "kev-4b"}],
    )
    decoupled, note = _measure_model_decoupling(work)
    assert decoupled is True
    assert "kev-4b" in note
    assert "§5.15" in note


def test_judge_on_the_same_model_does_not_count(tmp_path: Path) -> None:
    """Configuring a judge that IS the extractor is not decoupling."""
    work = _calls(
        tmp_path,
        [EXTRACT, VERIFY, {"kind": "chat", "role": "judge", "model": "deepseek-v4.1"}],
    )
    decoupled, note = _measure_model_decoupling(work)
    assert decoupled is False
    assert "已配置 judge" in note


def test_no_verifier_is_not_measured_rather_than_false(tmp_path: Path) -> None:
    work = _calls(tmp_path, [EXTRACT])
    decoupled, note = _measure_model_decoupling(work)
    assert decoupled is None
    assert "not measured" in note


def test_judge_llm_returns_a_number_or_none() -> None:
    """A rambling judge must be recorded as 'no score', not rounded to 0.5."""
    from cbb_stage1.orchestrator.llm_bridge import build_judge_llm

    class Fixed:
        def __init__(self, answer: str) -> None:
            self.answer = answer

        def generate(self, prompt: str) -> str:
            return self.answer

    assert build_judge_llm(Fixed("0.8"))({"claim": "c", "evidence": "e"}) == 0.8
    assert build_judge_llm(Fixed("1"))({"claim": "c", "evidence": "e"}) == 1.0
    assert build_judge_llm(Fixed("很难说"))({"claim": "c", "evidence": "e"}) is None
    assert build_judge_llm(Fixed("7"))({"claim": "c", "evidence": "e"}) is None
    assert build_judge_llm(None) is None


def test_judge_accepts_a_record_with_normalized_claim() -> None:
    from cbb_stage1.orchestrator.llm_bridge import build_judge_llm

    class Fixed:
        def generate(self, prompt: str) -> str:
            assert "甲是铁匠" in prompt
            return "0.4"

    judge = build_judge_llm(Fixed())
    assert judge({"normalized_claim": "甲是铁匠"}) == 0.4
