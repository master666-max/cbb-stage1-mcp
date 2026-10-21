"""Unit tests for :mod:`cbb_stage1.reasoning` (AFEV / Abductive RAG / VeGraph).

Acceptance (P1-C): the three adapters from the v6 design (B7/B8/B9) shape records
and enforce the design rules without a live model.  The mandated negative is the
same everywhere: **no LLM => no fabricated verdict**.  AFEV can only return
``unverifiable``; abduction can only return a ``candidate`` with no premises;
VeGraph can only return ``insufficient_evidence``.  Fakes stand in for the model
so every branch is exercised deterministically.
"""

from __future__ import annotations

import pytest

from cbb_stage1.models import Evidence
from cbb_stage1.reasoning import (
    DERIVED_TARGET,
    AbductionRuleError,
    AtomicRuleError,
    VeGraphRuleError,
    abduce,
    afev_verify,
    evidence_sufficiency,
    vegraph_verify,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _evidence(unit_id: str = "u:zj:v01:c001", quote: str = "拉法") -> Evidence:
    return Evidence(
        unit_id=unit_id,
        start=0,
        end=len(quote),
        quote=quote,
        mode="narration",
    )


class FakeAtomicLLM:
    """Decomposition rounds + a per-call verification plan."""

    def __init__(self, rounds, verdicts):
        self._rounds = [list(r) for r in rounds]
        self._verdicts = list(verdicts)
        self.contexts: list[str | None] = []
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def decompose(self, claim, context):
        self.contexts.append(context)
        return self._rounds.pop(0)

    def verify(self, fact, evidence):
        self.calls.append((fact, tuple(evidence)))
        return self._verdicts.pop(0)


class FakeAbductionLLM:
    def __init__(self, premises, consistency, plausibility):
        self._premises = list(premises)
        self._scores = {"consistency": consistency, "plausibility": plausibility}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def missing_premises(self, claim, evidence):
        self.calls.append((claim, tuple(evidence)))
        return list(self._premises)

    def score(self, claim, premises):
        return dict(self._scores)


class FakeVeGraphLLM:
    def __init__(self, triples, stances):
        self._triples = list(triples)
        self._stances = list(stances)
        self.triple_calls: list[str] = []
        self.verify_calls: list[dict] = []

    def triples(self, claim):
        self.triple_calls.append(claim)
        return [dict(t) for t in self._triples]

    def verify_subassertion(self, triple):
        self.verify_calls.append(dict(triple))
        return self._stances.pop(0)


def _vf(verdict: str, rationale: str | None = None) -> dict:
    return {"verdict": verdict, "rationale": rationale}


# ---------------------------------------------------------------------------
# AFEV (atomic.py)
# ---------------------------------------------------------------------------


def test_afev_without_llm_is_unverifiable_and_never_fabricates():
    result = afev_verify("黑之魔王的影子是苍蓝色的", book_id="zj")

    assert result.degenerate is True
    assert result.rounds == 0
    assert len(result.atomic_facts) == 1
    fact = result.atomic_facts[0]
    assert fact.verdict == "unverifiable"
    assert fact.evidence == []
    assert "no-llm" in (fact.rationale or "")
    assert result.verdict == "insufficient_evidence"
    assert result.aggregate == "insufficient_evidence"
    assert "no_llm=unverifiable" in result.notes
    record = result.atomic_records()[0]
    assert set(record) >= {"fact_id", "text", "verdict", "evidence", "rationale"}
    assert record["fact_id"].startswith("af:")


def test_afev_all_true_is_supported():
    llm = FakeAtomicLLM([["黑之魔王有影子", "影子是苍蓝色的"]], [_vf("true"), _vf("true")])

    result = afev_verify("黑之魔王的影子是苍蓝色的", book_id="zj", llm=llm)

    assert result.verdict == "supported"
    assert result.rounds == 1
    assert result.degenerate is False
    assert [f.text for f in result.atomic_facts] == ["黑之魔王有影子", "影子是苍蓝色的"]
    assert all(f.verdict == "true" for f in result.atomic_facts)


def test_afev_any_false_is_refuted():
    llm = FakeAtomicLLM([["A", "B"]], [_vf("true"), _vf("false", "正文否定")])

    result = afev_verify("断言", book_id="zj", llm=llm)

    assert result.verdict == "refuted"
    assert result.atomic_facts[1].rationale == "正文否定"


def test_afev_unverifiable_is_insufficient_evidence():
    llm = FakeAtomicLLM([["A", "B"]], [_vf("true"), _vf("unverifiable")])

    result = afev_verify("断言", book_id="zj", llm=llm, max_rounds=1)

    assert result.verdict == "insufficient_evidence"
    assert result.degenerate is False
    assert result.rounds == 1
    assert "unverifiable_after_max_rounds" in result.notes


def test_afev_iterates_with_previous_rationale_as_context():
    llm = FakeAtomicLLM(
        [["A"], ["A", "B"]],
        [_vf("unverifiable", "需要更多上下文"),
         _vf("true"), _vf("true")],
    )

    result = afev_verify("断言", book_id="zj", llm=llm)

    assert result.rounds == 2
    assert result.verdict == "supported"
    assert llm.contexts[0] is None
    assert "需要更多上下文" in (llm.contexts[1] or "")


def test_afev_empty_decomposition_raises():
    llm = FakeAtomicLLM([[]], [])

    with pytest.raises(AtomicRuleError):
        afev_verify("断言", llm=llm)


def test_afev_invalid_verdict_raises():
    llm = FakeAtomicLLM([["A"]], [_vf("maybe")])

    with pytest.raises(AtomicRuleError):
        afev_verify("断言", llm=llm)


def test_afev_retrieve_hook_feeds_each_fact():
    llm = FakeAtomicLLM([["A", "B"]], [_vf("true"), _vf("true")])

    afev_verify("断言", evidence=["claim-level"], llm=llm,
                retrieve=lambda fact: [f"{fact} 的原文"])

    assert llm.calls[0] == ("A", ("A 的原文",))
    assert llm.calls[1] == ("B", ("B 的原文",))


# ---------------------------------------------------------------------------
# Abductive RAG (abduction.py)
# ---------------------------------------------------------------------------


def test_sufficiency_counts_distinct_units():
    assert evidence_sufficiency([_evidence("u1"), _evidence("u2")]) == "sufficient"
    assert evidence_sufficiency([_evidence("u1"), _evidence("u1", "别的")]) == "insufficient"
    assert evidence_sufficiency([]) == "insufficient"


def test_abduce_sufficient_evidence_returns_no_hypothesis():
    result = abduce("甲是乙的老师", evidence=[_evidence("u1"), _evidence("u2")])

    assert result.evidence_sufficiency == "sufficient"
    assert result.hypothesis is None
    assert result.reason == "sufficient_evidence_use_induction"


def test_abduce_without_llm_is_candidate_only():
    result = abduce("灵石汇率约为 1:10:100", evidence=[_evidence("u1")], scope="setting")

    assert result.evidence_sufficiency == "insufficient"
    assert result.hypothesis is not None
    hyp = result.hypothesis
    assert hyp.status == "candidate"
    assert hyp.missing_premises == ()
    assert hyp.consistency is None and hyp.plausibility is None
    assert hyp.confidence == 0.0
    assert "需补证据" in hyp.claim
    assert hyp.target_file == DERIVED_TARGET
    assert result.derived is True
    record = hyp.to_record()
    assert record["evidence_sufficiency"] == "insufficient"
    assert record["derived_from"] == ["u1"]
    assert record["created_at"] is None  # no fabricated timestamp
    assert set(record) >= {
        "hypothesis_id", "book_id", "scope", "claim", "evidence_sufficiency",
        "missing_premises", "support", "against", "consistency", "plausibility",
        "status", "confidence", "method", "derived_from",
    }


def test_abduce_with_llm_can_verify_when_checks_pass():
    llm = FakeAbductionLLM(["多次交易样本", "不同人物报价"], 0.8, 0.9)

    result = abduce("灵石汇率约为 1:10:100", evidence=[_evidence("u1")], llm=llm)

    hyp = result.hypothesis
    assert hyp is not None
    assert hyp.status == "verified"
    assert hyp.missing_premises == ("多次交易样本", "不同人物报价")
    assert hyp.consistency == "pass"
    assert hyp.plausibility == "high"
    assert hyp.confidence == 0.8
    assert llm.calls[0][0] == "灵石汇率约为 1:10:100"


def test_abduce_low_scores_stay_candidate():
    llm = FakeAbductionLLM(["p"], 0.5, 0.4)

    result = abduce("假设", evidence=[_evidence("u1")], llm=llm)

    hyp = result.hypothesis
    assert hyp is not None
    assert hyp.status == "candidate"
    assert hyp.consistency == "fail"
    assert hyp.plausibility == "low"
    assert hyp.confidence == 0.4


def test_abduce_invalid_scope_raises():
    with pytest.raises(AbductionRuleError):
        abduce("x", scope="nonsense")


def test_abduce_out_of_range_score_raises():
    llm = FakeAbductionLLM(["p"], 1.5, 0.5)

    with pytest.raises(AbductionRuleError):
        abduce("假设", evidence=[_evidence("u1")], llm=llm)


def test_hypothesis_id_is_content_addressed():
    a = abduce("假设", evidence=[_evidence("u1")])
    b = abduce("假设", evidence=[_evidence("u1")])
    assert a.hypothesis is not None and b.hypothesis is not None
    assert a.hypothesis.hypothesis_id == b.hypothesis.hypothesis_id
    assert a.hypothesis.hypothesis_id.startswith("hy:")


# ---------------------------------------------------------------------------
# VeGraph (vegraph.py)
# ---------------------------------------------------------------------------


def test_vegraph_without_llm_is_insufficient_evidence():
    result = vegraph_verify("甲与乙敌对", name_to_id={"甲": "e:1", "乙": "e:2"})

    assert result.degenerate is True
    assert result.triples == ()
    assert result.sub_stances == ()
    assert result.verdict == "insufficient_evidence"
    assert "no_llm=insufficient_evidence" in result.notes


def test_unresolved_entity_blocks_stance_judgement():
    llm = FakeVeGraphLLM(
        [{"subject": "甲", "predicate": "敌对", "object": "丙"}],
        ["support"],
    )

    result = vegraph_verify("甲与丙敌对", name_to_id={"甲": "e:1"}, llm=llm)

    assert result.verdict == "insufficient_evidence"
    assert result.unresolved_entities == ("丙",)
    assert llm.verify_calls == []  # VeGraph order: never judge before resolving
    assert "unresolved_entities_block_stance" in result.notes


def test_resolved_triples_can_be_supported():
    llm = FakeVeGraphLLM(
        [{"subject": "甲", "predicate": "敌对", "object": "乙"}],
        ["support"],
    )

    result = vegraph_verify("甲与乙敌对", name_to_id={"甲": "e:1", "乙": "e:2"}, llm=llm)

    assert result.verdict == "supported"
    assert result.unresolved_entities == ()
    assert result.resolution.resolved == {"甲": "e:1", "乙": "e:2"}
    assert result.sub_assertion_records() == [
        {"subject": "甲", "predicate": "敌对", "object": "乙", "stance": "support"}
    ]


def test_mixed_support_and_refute_is_conflicting():
    llm = FakeVeGraphLLM(
        [{"subject": "甲", "predicate": "敌对", "object": "乙"},
         {"subject": "丙", "predicate": "效忠", "object": "丁"}],
        ["support", "refute"],
    )

    result = vegraph_verify("甲与乙敌对；丙效忠丁",
                           name_to_id={"甲": "e:1", "乙": "e:2", "丙": "e:3", "丁": "e:4"},
                           llm=llm)

    assert result.verdict == "conflicting"
    assert len(result.sub_stances) == 2


def test_only_refute_is_refuted():
    llm = FakeVeGraphLLM([{"subject": "甲", "predicate": "敌对", "object": "乙"}], ["refute"])

    result = vegraph_verify("甲与乙敌对", name_to_id={"甲": "e:1", "乙": "e:2"}, llm=llm)

    assert result.verdict == "refuted"


def test_invalid_stance_raises():
    llm = FakeVeGraphLLM([{"subject": "甲", "predicate": "敌对", "object": "乙"}], ["maybe"])

    with pytest.raises(VeGraphRuleError):
        vegraph_verify("甲与乙敌对", name_to_id={"甲": "e:1", "乙": "e:2"}, llm=llm)


def test_malformed_triple_raises():
    llm = FakeVeGraphLLM([{"subject": "甲", "predicate": "敌对"}], [])

    with pytest.raises(VeGraphRuleError):
        vegraph_verify("甲与乙敌对", name_to_id={"甲": "e:1"}, llm=llm)


def test_alias_map_resolves_surface_before_stance():
    llm = FakeVeGraphLLM(
        [{"subject": "甲哥", "predicate": "敌对", "object": "乙"}],
        ["support"],
    )

    result = vegraph_verify("甲哥与乙敌对", name_to_id={"甲": "e:1", "乙": "e:2"},
                            alias_map={"甲哥": "甲"}, llm=llm)

    assert result.verdict == "supported"
    assert result.resolution.resolved["甲哥"] == "e:1"


def test_no_triples_is_insufficient_without_fabricating():
    llm = FakeVeGraphLLM([], [])

    result = vegraph_verify("无法拆成三元组的断言", name_to_id={}, llm=llm)

    assert result.verdict == "insufficient_evidence"
    assert result.degenerate is False
    assert "no_triples" in result.notes
    assert llm.verify_calls == []
