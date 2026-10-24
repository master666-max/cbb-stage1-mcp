"""CoVe must actually be reachable from the claims stage, not just exist in a module.

The project has been bitten by "written but never called" often enough that a module
without a call site is treated as absent. These tests drive the real claims stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.claims import integration as claims_integration
from cbb_stage1.validate import cove


def _work(root: Path) -> None:
    unit = "u:t:v01:c001"
    assertion = {
        "assertion_id": "a:stmt0000000001",
        "book_id": "t",
        "unit_id": unit,
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "e:t:person:0001", "受事": "e:t:person:0002"},
        "claim_layer": "statement",
        "speaker": {"kind": "entity", "entity_id": "e:t:person:0001", "name": "阿尔法"},
        "evidence": [
            {
                "unit_id": unit,
                "start": 0,
                "end": 9,
                "quote": "「好。把它毁掉吧。」",
                "occurrence": 1,
                "mode": "dialogue",
            }
        ],
        "confidence": 0.9,
        "confidence_method": "semantic_entropy",
        "status": "asserted",
    }
    (root / "assemble").mkdir(parents=True, exist_ok=True)
    (root / "assemble" / "assertions.jsonl").write_text(
        json.dumps(assertion, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "plan").mkdir(parents=True, exist_ok=True)
    (root / "plan" / "units.jsonl").write_text(
        json.dumps(
            {
                "unit_id": unit,
                "book_id": "t",
                "order": 1,
                "volume": 1,
                "title": "第一章",
                "narrative_mode": "first_person",
                "narrator_reliability": "reliable",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "claims").mkdir(parents=True, exist_ok=True)
    (root / "claims" / "qa.jsonl").write_text(
        json.dumps(
            {
                "statement_assertion_id": "a:stmt0000000001",
                "questions": [{"qid": "q1", "question": "阿尔法要毁掉它吗？", "answer": "yes"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


class ScriptedLLM:
    """Answers each CoVe prompt deterministically and counts the calls."""

    def __init__(self, *, contradictions: str = "无矛盾", revision: str = "") -> None:
        self.prompts: list[str] = []
        self._contradictions = contradictions
        self._revision = revision

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "把断言拆成" in prompt or "原子事实" in prompt:
            return '["阿尔法决定毁掉它"]'
        if "最终版本" in prompt:
            return self._revision or "（修订稿）"
        if "相矛盾" in prompt:
            return self._contradictions
        if "验证问题" in prompt:
            return "- 谁要毁掉它？\n- 要毁掉的是什么？"
        if "证据" in prompt and "问题：" in prompt:
            return "证据里写着「把它毁掉吧」"
        if "核验原子事实" in prompt:
            return '{"verdict": "true", "rationale": "证据可见"}'
        if "复述" in prompt:
            return "阿尔法决定毁掉它。"
        return "{}"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_cove_runs_when_the_caller_opts_in(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root)
    llm = ScriptedLLM()
    result = claims_integration.run(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-21T00:00:00+08:00",
        upstream={"cove": True, "llm": llm, "reasoning_llm": llm},
    )
    rows = _rows(root / "assemble" / "claims.jsonl")
    assert rows, "the claim must still land"
    assert "cove" in rows[0], "CoVe must record its result on the claim"
    assert rows[0]["cove"]["method"] == cove.METHOD
    assert "cove_revised" in (result.get("counts") or {})
    assert any("验证问题" in prompt for prompt in llm.prompts), "step 2 must have run"


def test_cove_is_off_by_default(tmp_path: Path) -> None:
    """No flag means no cost — and the receipt says zero rather than pretending it ran."""
    root = tmp_path / "work"
    _work(root)
    llm = ScriptedLLM()
    result = claims_integration.run(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-21T00:00:00+08:00",
        upstream={"llm": llm, "reasoning_llm": llm},
    )
    rows = _rows(root / "assemble" / "claims.jsonl")
    assert "cove" not in rows[0]
    assert (result.get("counts") or {})["cove_revised"] == 0
    assert not any("验证问题" in prompt for prompt in llm.prompts)


def test_a_contradiction_is_counted_as_revised(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root)
    llm = ScriptedLLM(contradictions="Q1：回答与主张矛盾")
    result = claims_integration.run(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-21T00:00:00+08:00",
        upstream={"cove": True, "llm": llm, "reasoning_llm": llm},
    )
    rows = _rows(root / "assemble" / "claims.jsonl")
    assert rows[0]["cove"]["status"] == "revised"
    assert (result.get("counts") or {})["cove_revised"] >= 1


# ══ the parallel path must be reachable from here ═════════════════════
def test_the_cove_call_passes_a_worker_count() -> None:
    """`cove_verify(workers=...)` exists; a call site that omits it leaves the parallel path dead."""
    import inspect

    source = inspect.getsource(claims_integration)
    block = source.split("cove_verify(")[1][:400]
    assert "cove_workers" in block, (
        "the CoVe call passes no worker count, so answers stay serial — the paper says they can be "
        f"batched (A09 p.4). call site: {block[:200]!r}"
    )


def test_the_cove_worker_count_is_bounded_by_the_stage_budget() -> None:
    """Nested pools starve the gateway semaphore, so the inner count is capped."""
    import inspect

    source = inspect.getsource(claims_integration.cove_worker_count)
    assert "min(" in source, (
        "cove_worker_count does not cap its result; nested pools can exceed the gateway semaphore"
    )
    assert "budget" in source, "the cap must come from the stage budget"


def test_a_serial_stage_gives_cove_one_worker() -> None:
    from cbb_stage1.claims.integration import cove_worker_count

    assert cove_worker_count({"max_workers": 1}, 5) == 1


def test_the_count_never_exceeds_the_stage_budget() -> None:
    from cbb_stage1.claims.integration import cove_worker_count

    for budget in (1, 2, 4, 8, 32):
        value = cove_worker_count({"max_workers": budget}, 100)
        assert 1 <= value <= budget, f"budget={budget} produced {value}"


def test_a_raising_cove_does_not_abort_the_stage(tmp_path: Path) -> None:
    """Same rule as AFEV: one analysis failing must not kill the run."""
    root = tmp_path / "work"
    _work(root)

    class Boom(ScriptedLLM):
        def generate(self, prompt: str) -> str:
            if "验证问题" in prompt:
                raise RuntimeError("cove endpoint down")
            return super().generate(prompt)

    result = claims_integration.run(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-21T00:00:00+08:00",
        upstream={"cove": True, "llm": Boom(), "reasoning_llm": Boom()},
    )
    assert result.get("status") == "ok", "the stage must survive"
    rows = _rows(root / "assemble" / "claims.jsonl")
    assert rows, "and still produce the claim"
