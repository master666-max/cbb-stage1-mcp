"""Unit tests for :mod:`cbb_stage1.confidence` — absorption list 07 B2–B5.

The point of these tests is the **contract**, not happy-path arithmetic only:

* B2 semantic entropy: N samples → meaning clusters → entropy → confidence;
  no LLM / N<2 / no claim text ⇒ ``unavailable`` (never a fabricated score).
* B3 SEPs: hidden-state / log-prob white-box approximation; no white-box access
  ⇒ ``unavailable``.
* B4 UQLM: white-box / black-box / judge classes each return ``[0,1]``; a judge
  that does not return a parseable number ⇒ ``error`` (not rounded).
* B5 UQ Heads: no trained head ⇒ ``external`` (EXTERNAL) with no confidence;
  a head that returns an invalid number ⇒ ``error``.

The fake LLMs here are synchronous on purpose: the confidence adapters are
synchronous and record async clients as ``unavailable`` instead of blocking.
"""

from __future__ import annotations

import math

import pytest

from cbb_stage1.confidence import (
    CONFIDENCE_METHODS,
    IMPLEMENTED_METHODS,
    STATUS_ERROR,
    STATUS_EXTERNAL,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    ConfidenceResult,
    available_methods,
    score,
)
from cbb_stage1.confidence import semantic_entropy as se
from cbb_stage1.confidence import sep, uqlm, uq_heads
from cbb_stage1.models import Assertion

RECORD = {
    "normalized_claim": "甲是青牛镇的铁匠",
    "predicate": "职业",
    "roles": {"subject": "甲", "object": "铁匠"},
}

# Answers that must not cluster together (distinct meaning space).
DISTINCT = ["甲是青牛镇的铁匠", "乙去了北方的雪山", "丙在河边钓鱼", "丁读了一本古书"]


# ── fakes ──────────────────────────────────────────────────────────────────
class FakeSamplingLLM:
    """Cycles ``answers`` to return exactly ``n`` samples (deterministic)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def sample(self, prompt: str, n: int):
        self.calls.append((prompt, n))
        if not self.answers:
            return []
        return [self.answers[i % len(self.answers)] for i in range(n)]


class BlackboxGenerateLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate(self, prompt: str):
        self.calls.append(prompt)
        return self.answers[(len(self.calls) - 1) % len(self.answers)]


class AsyncGenerateLLM:
    async def generate(self, prompt: str):  # pragma: no cover - must not be driven
        return "甲是铁匠"


class WhiteboxLLM:
    whitebox = True

    def __init__(self, vectors=None, logprobs=None):
        self._vectors = vectors
        self._logprobs = logprobs

    def hidden_states(self, prompt: str):
        return self._vectors

    def token_logprobs(self, prompt: str):
        return self._logprobs


class JudgeLLM:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def judge(self, prompt: str):
        self.prompts.append(prompt)
        return self.result


class GenerateJudgeLLM:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt: str):
        return self.text


class HeadWithScore:
    def score(self, record: dict):
        return {"confidence": 0.6, "scope": "claim"}


# ── ConfidenceResult contract ──────────────────────────────────────────────
def test_ok_result_requires_enum_method_and_unit_interval():
    result = ConfidenceResult(method="semantic_entropy", confidence=0.5)
    assert result.status == STATUS_OK and result.available
    assert result.to_dict()["method"] == "semantic_entropy"
    with pytest.raises(ValueError):
        ConfidenceResult(method="made_up", confidence=0.5)
    with pytest.raises(ValueError):
        ConfidenceResult(method="semantic_entropy", confidence=1.5)
    with pytest.raises(ValueError):
        ConfidenceResult(method="semantic_entropy", confidence=True)


def test_non_ok_results_must_not_carry_a_score():
    unavailable = ConfidenceResult.unavailable("no llm", method="semantic_entropy")
    external = ConfidenceResult.external("needs checkpoint", method="uq_heads")
    error = ConfidenceResult.error("bad output", method="uq_heads")
    for result in (unavailable, external, error):
        assert result.confidence is None and not result.available
        assert result.reason
        assert result.status in (STATUS_UNAVAILABLE, STATUS_EXTERNAL, STATUS_ERROR)
    with pytest.raises(ValueError):
        ConfidenceResult(status=STATUS_UNAVAILABLE, method=None, confidence=0.3, reason="x")


def test_method_enum_matches_assemble():
    from cbb_stage1.assemble.assertions import CONFIDENCE_METHODS as ASSEMBLE_METHODS

    assert set(CONFIDENCE_METHODS) == set(ASSEMBLE_METHODS)
    assert set(IMPLEMENTED_METHODS) <= set(CONFIDENCE_METHODS)


# ── B2: semantic entropy ───────────────────────────────────────────────────
def test_semantic_entropy_all_identical_samples_is_confident():
    result = se.score(RECORD, FakeSamplingLLM(["甲是青牛镇的铁匠"] * 8))
    assert result.status == STATUS_OK
    assert result.method == "semantic_entropy"
    assert result.confidence == pytest.approx(1.0)
    assert result.details["cluster_count"] == 1
    assert result.details["samples"] == 8


def test_semantic_entropy_all_distinct_samples_is_uncertain():
    result = se.score(RECORD, FakeSamplingLLM(DISTINCT), samples=4)
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(0.0, abs=1e-6)
    assert result.details["cluster_count"] == 4
    assert result.details["entropy_normalized"] == pytest.approx(1.0)


def test_semantic_entropy_two_equal_clusters_gives_half():
    result = se.score(RECORD, FakeSamplingLLM(["甲是铁匠", "甲是铁匠", "乙是渔夫", "乙是渔夫"]), samples=4)
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(0.5, abs=1e-6)


def test_semantic_entropy_without_llm_is_unavailable():
    result = se.score(RECORD)
    assert result.status == STATUS_UNAVAILABLE
    assert result.method == "semantic_entropy"
    assert result.confidence is None
    assert "llm" in result.reason


def test_semantic_entropy_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        se.score(RECORD, FakeSamplingLLM(["甲是铁匠"]), samples=1)


def test_semantic_entropy_needs_claim_text():
    result = se.score({"book_id": "b"}, FakeSamplingLLM(["x"]))
    assert result.status == STATUS_UNAVAILABLE
    assert "主张文本" in result.reason


def test_semantic_entropy_llm_without_sampling_interface_is_unavailable():
    result = se.score(RECORD, object())
    assert result.status == STATUS_UNAVAILABLE
    assert "sample" in result.reason or "generate" in result.reason


def test_semantic_entropy_async_generate_is_unavailable_not_faked():
    result = se.score(RECORD, AsyncGenerateLLM())
    assert result.status == STATUS_UNAVAILABLE
    assert result.confidence is None
    assert "异步" in result.reason


def test_semantic_entropy_accepts_explicit_sampler():
    result = se.score(RECORD, sampler=lambda prompt, n: ["甲是铁匠"] * n)
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(1.0)


def test_semantic_entropy_uses_generate_fallback():
    result = se.score(RECORD, BlackboxGenerateLLM(["甲是铁匠"]), samples=3)
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(1.0)


def test_semantic_entropy_helper_normalises_punctuation():
    assert se.normalize_answer("甲是铁匠。  ") == "甲是铁匠"
    stats = se.semantic_entropy(["甲是铁匠。", "甲是铁匠！"])
    assert stats["usable"] and stats["confidence"] == pytest.approx(1.0)


# ── B3: SEPs (white-box) ───────────────────────────────────────────────────
def test_sep_hidden_states_same_vectors_is_confident():
    result = sep.score(RECORD, hidden_states=[[1.0, 0.0], [1.0, 0.0]])
    assert result.status == STATUS_OK
    assert result.method == "semantic_entropy"  # §5.15 keeps the SEP name in details
    assert result.details["approximation"] == "sep_hidden_states"
    assert result.confidence == pytest.approx(1.0)


def test_sep_hidden_states_orthogonal_vectors_is_uncertain():
    result = sep.score(RECORD, hidden_states=[[1.0, 0.0], [0.0, 1.0]])
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(0.0)
    assert result.details["cluster_count"] == 2


def test_sep_logprobs_geometric_mean():
    result = sep.score(RECORD, logprobs=[-0.1, -0.2, -0.3])
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(math.exp(-0.2), rel=1e-4)
    assert result.details["approximation"] == "sep_logprob_geom_mean"


def test_sep_positive_logprobs_are_rejected_as_invalid_input():
    with pytest.raises(ValueError):
        sep.score(RECORD, logprobs=[0.1, -0.2])


def test_sep_without_whitebox_is_unavailable():
    result = sep.score(RECORD)
    assert result.status == STATUS_UNAVAILABLE
    assert result.confidence is None
    assert "白盒" in result.reason


def test_sep_blackbox_llm_is_unavailable():
    result = sep.score(RECORD, FakeSamplingLLM(DISTINCT))
    assert result.status == STATUS_UNAVAILABLE
    assert "白盒" in result.reason or "hidden_states" in result.reason


def test_sep_pulls_hidden_states_from_whitebox_llm():
    result = sep.score(RECORD, WhiteboxLLM(vectors=[[1.0, 0.0], [1.0, 0.0]]))
    assert result.status == STATUS_OK
    assert result.details["source"] == "llm.hidden_states"


# ── B4: UQLM ───────────────────────────────────────────────────────────────
def test_uqlm_whitebox_from_logprobs():
    result = uqlm.score(RECORD, mode="whitebox", logprobs=[-0.05, -0.1])
    assert result.status == STATUS_OK
    assert result.method == "uqlm_whitebox"
    assert result.confidence == pytest.approx(math.exp(-0.075), rel=1e-4)
    assert result.details["uqlm_class"] == "whitebox"


def test_uqlm_whitebox_without_signal_is_unavailable():
    result = uqlm.score(RECORD, mode="whitebox")
    assert result.status == STATUS_UNAVAILABLE
    assert result.method == "uqlm_whitebox"


def test_uqlm_blackbox_consensus_is_confident():
    result = uqlm.score(RECORD, FakeSamplingLLM(["甲是铁匠"]), mode="blackbox", samples=5)
    assert result.status == STATUS_OK
    assert result.method == "uqlm_blackbox"
    assert result.confidence == pytest.approx(1.0)


def test_uqlm_blackbox_all_distinct_is_low():
    result = uqlm.score(RECORD, FakeSamplingLLM(DISTINCT), mode="blackbox", samples=4)
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(1.0 / 4.0)
    assert result.details["definition"].startswith("P(two sampled answers agree)")


def test_uqlm_judge_returns_claim_qa_with_uqlm_class():
    result = uqlm.score(RECORD, judge_llm=JudgeLLM(0.8), mode="judge")
    assert result.status == STATUS_OK
    assert result.method == "claim_qa"
    assert result.scope == "claim"
    assert result.confidence == pytest.approx(0.8)
    assert result.details["uqlm_class"] == "judge"


def test_uqlm_judge_parses_number_from_text():
    result = uqlm.score(RECORD, judge_llm=GenerateJudgeLLM("可信度：0.7"), mode="judge")
    assert result.status == STATUS_OK and result.confidence == pytest.approx(0.7)


def test_uqlm_judge_unparseable_output_is_error_not_a_score():
    result = uqlm.score(RECORD, judge_llm=JudgeLLM("完全无法判断"), mode="judge")
    assert result.status == STATUS_ERROR
    assert result.confidence is None
    assert result.method == "claim_qa"


def test_uqlm_judge_out_of_range_is_error():
    result = uqlm.score(RECORD, judge_llm=JudgeLLM(1.5), mode="judge")
    assert result.status == STATUS_ERROR and result.confidence is None


def test_uqlm_judge_without_judge_is_unavailable():
    result = uqlm.score(RECORD, mode="judge")
    assert result.status == STATUS_UNAVAILABLE
    assert "judge_llm" in result.reason


def test_uqlm_rejects_unknown_mode():
    with pytest.raises(ValueError):
        uqlm.score(RECORD, mode="magic")


def test_uqlm_auto_picks_blackbox_for_sampling_llm():
    result = uqlm.score(RECORD, FakeSamplingLLM(["甲是铁匠"]), samples=4)
    assert result.status == STATUS_OK and result.method == "uqlm_blackbox"


def test_uqlm_auto_picks_judge_when_only_judge_is_given():
    result = uqlm.score(RECORD, judge_llm=JudgeLLM(0.9))
    assert result.status == STATUS_OK and result.method == "claim_qa"


def test_uqlm_no_signals_is_unavailable():
    result = uqlm.score(RECORD)
    assert result.status == STATUS_UNAVAILABLE
    assert result.confidence is None


# ── B5: UQ Heads ───────────────────────────────────────────────────────────
def test_uq_heads_callable_head_scores():
    result = uq_heads.score(RECORD, head=lambda record: 0.75)
    assert result.status == STATUS_OK
    assert result.method == "uq_heads"
    assert result.scope == "claim"
    assert result.confidence == pytest.approx(0.75)


def test_uq_heads_object_head_with_score_method():
    result = uq_heads.score(RECORD, head=HeadWithScore())
    assert result.status == STATUS_OK
    assert result.confidence == pytest.approx(0.6)
    assert result.details["head_type"] == "HeadWithScore"


def test_uq_heads_missing_head_is_external_not_a_fake_score():
    result = uq_heads.score(RECORD)
    assert result.status == STATUS_EXTERNAL
    assert result.method == "uq_heads"
    assert result.confidence is None
    assert "EXTERNAL" in result.reason


def test_uq_heads_checkpoint_without_loader_is_external():
    result = uq_heads.score(RECORD, checkpoint="heads/uq.pt")
    assert result.status == STATUS_EXTERNAL
    assert result.confidence is None
    assert result.details["checkpoint"] == "heads/uq.pt"


def test_uq_heads_checkpoint_with_loader_returns_real_score():
    result = uq_heads.score(
        RECORD, checkpoint="heads/uq.pt", loader=lambda path: (lambda record: 0.9)
    )
    assert result.status == STATUS_OK and result.confidence == pytest.approx(0.9)


def test_uq_heads_loader_failure_is_external():
    def broken_loader(path):
        raise RuntimeError("no torch")

    result = uq_heads.score(RECORD, checkpoint="heads/uq.pt", loader=broken_loader)
    assert result.status == STATUS_EXTERNAL
    assert result.confidence is None


def test_uq_heads_out_of_range_is_error():
    result = uq_heads.score(RECORD, head=lambda record: 1.2)
    assert result.status == STATUS_ERROR and result.confidence is None


def test_uq_heads_exception_is_error():
    def doomed(record):
        raise RuntimeError("head exploded")

    result = uq_heads.score(RECORD, head=doomed)
    assert result.status == STATUS_ERROR and result.confidence is None


def test_uq_heads_available_helper():
    assert not uq_heads.available()
    assert uq_heads.available(head=object())
    assert not uq_heads.available(checkpoint="x.pt")


# ── unified dispatcher ─────────────────────────────────────────────────────
def test_score_dispatches_implemented_methods():
    assert score(RECORD, method="semantic_entropy", llm=FakeSamplingLLM(["甲是铁匠"] * 4)).confidence == 1.0
    assert score(RECORD, method="sep", hidden_states=[[1.0, 0.0], [1.0, 0.0]]).confidence == 1.0
    assert score(RECORD, method="uqlm_whitebox", logprobs=[-0.1, -0.1]).method == "uqlm_whitebox"
    assert score(RECORD, method="uqlm_blackbox", llm=FakeSamplingLLM(["甲是铁匠"])).method == "uqlm_blackbox"
    assert score(RECORD, method="uq_heads", head=lambda record: 0.5).method == "uq_heads"


def test_score_auto_picks_semantic_entropy_for_sampling_llm():
    result = score(RECORD, llm=FakeSamplingLLM(["甲是铁匠"] * 4))
    assert result.status == STATUS_OK and result.method == "semantic_entropy"


def test_score_auto_picks_uqlm_whitebox_for_logprobs():
    result = score(RECORD, logprobs=[-0.1, -0.2])
    assert result.status == STATUS_OK and result.method == "uqlm_whitebox"


def test_score_auto_picks_uq_heads_when_head_given():
    result = score(RECORD, head=lambda record: 0.4)
    assert result.status == STATUS_OK and result.method == "uq_heads"


def test_score_without_any_signal_is_unavailable():
    result = score(RECORD)
    assert result.status == STATUS_UNAVAILABLE
    assert result.method is None and result.confidence is None


def test_score_method_outside_package_scope_is_unavailable_with_reason():
    result = score(RECORD, method="evidence_span")
    assert result.status == STATUS_UNAVAILABLE
    assert result.method == "evidence_span"
    assert "B2" in result.reason


def test_score_accepts_pydantic_record():
    record = Assertion(book_id="b", predicate="职业", library="entity",
                       roles={"subject": "甲", "object": "铁匠"})
    result = score(record, llm=FakeSamplingLLM(["甲是铁匠"] * 4))
    assert result.status == STATUS_OK
    assert result.method == "semantic_entropy"


def test_score_rejects_unknown_b3_alias_conflict():
    with pytest.raises(ValueError):
        score(RECORD, method="uqlm_whitebox", mode="blackbox", logprobs=[-0.1])


def test_available_methods_is_input_driven():
    assert "semantic_entropy" in available_methods(FakeSamplingLLM(["甲是铁匠"]))
    assert "uq_heads" in available_methods(head=object())
    assert available_methods(object()) == ()


def test_claim_text_falls_back_to_evidence_quotes():
    assert se.claim_text({"evidence": [{"quote": "甲是铁匠"}]}) == "甲是铁匠"
