"""Unit tests for :mod:`cbb_stage1.claims` (AVeriTeC verdicts + QA + promotion).

Coverage follows the P2 acceptance points: the four-valued verdict rule, atomic
facts aggregation consistency, and the fact-promotion gate.  The mandated
negative is ``insufficient_evidence``: it may never be promoted to ``fact``;
statements only promote with a supported claim and >= 2 independent sources.
"""

from __future__ import annotations

import pytest

from cbb_stage1.claims import (
    CLAIM_VERDICTS,
    ClaimBuildError,
    ClaimRuleError,
    PromotionRefused,
    aggregate_atomic_facts,
    build_claim,
    build_promoted_fact,
    claim_id_for,
    claim_verdict,
    independent_sources,
    make_answer,
    make_question,
    normalize_claim_text,
    promotion_decision,
    resolve_entities,
    support_answers,
    verdict_from_stances,
)
from cbb_stage1.evidence.anchor import sha256_text
from cbb_stage1.models import Assertion, AtomicFact, Evidence, Speaker

STATEMENT_ID = "a:0123456789abcdef"
SPEAKER = Speaker(kind="entity", entity_id="e:zj:p:1", name="甲")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _statement(**overrides: object) -> Assertion:
    defaults: dict[str, object] = {
        "assertion_id": STATEMENT_ID,
        "book_id": "zj",
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "e:zj:p:1", "受事": "e:zj:p:2"},
        "claim_layer": "statement",
        "speaker": SPEAKER,
        "evidence": [
            Evidence(
                unit_id="u:zj:v01:c001",
                start=0,
                end=2,
                quote="拉法",
                speaker=SPEAKER,
                mode="dialogue",
            )
        ],
        "confidence": 0.8,
        "confidence_method": "evidence_span",
        "status": "asserted",
    }
    defaults.update(overrides)
    return Assertion(**defaults)  # type: ignore[arg-type]


def _answer(
    unit_id: str = "u:zj:v01:c001",
    *,
    quote: str = "拉法",
    start: int = 0,
    text: str | None = None,
) -> dict[str, object]:
    return make_answer(text or quote, unit_id, start, start + len(quote), quote)


def _support_questions(units: tuple[str, ...] = ("u:zj:v01:c001", "u:zj:v01:c002")):
    return [
        make_question(f"q{index}", "甲与谁敌对？", answers=[_answer(unit)], stance="support")
        for index, unit in enumerate(units, start=1)
    ]


def _claim(statement: Assertion | None = None, units: tuple[str, ...] = ("u:zj:v01:c001", "u:zj:v01:c002")):
    statement = statement or _statement()
    return build_claim(
        statement,
        questions=_support_questions(units),
        normalized_claim="甲与乙敌对",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# verdict rules
# ---------------------------------------------------------------------------


def test_four_value_verdict_rule():
    assert verdict_from_stances(["support"]) == "supported"
    assert verdict_from_stances(["refute"]) == "refuted"
    assert verdict_from_stances(["support", "refute"]) == "conflicting"
    assert verdict_from_stances(["support", "irrelevant"]) == "supported"
    assert verdict_from_stances(["refute", "unanswerable"]) == "refuted"
    assert verdict_from_stances(["irrelevant", "unanswerable"]) == "insufficient_evidence"
    assert verdict_from_stances([]) == "insufficient_evidence"
    assert CLAIM_VERDICTS == (
        "supported", "refuted", "conflicting", "insufficient_evidence"
    )


def test_atomic_facts_aggregation():
    def fact(verdict: str) -> AtomicFact:
        return AtomicFact(fact_id="af:1", text="甲有影子", verdict=verdict)

    assert aggregate_atomic_facts([fact("true"), fact("true")]) == "supported"
    assert aggregate_atomic_facts([fact("true"), fact("false")]) == "refuted"
    assert aggregate_atomic_facts([fact("true"), fact("unverifiable")]) == "insufficient_evidence"
    assert aggregate_atomic_facts([fact("false"), fact("unverifiable")]) == "refuted"
    assert aggregate_atomic_facts([]) == "insufficient_evidence"
    with pytest.raises(ClaimRuleError):
        aggregate_atomic_facts([{"fact_id": "af:2", "text": "x", "verdict": "maybe"}])


def test_claim_verdict_requires_atomic_consistency():
    claim = _claim()
    consistent = claim.model_copy(
        update={"atomic_facts": [AtomicFact(fact_id="af:1", text="甲有影子", verdict="true")]}
    )
    assert claim_verdict(consistent) == "supported"

    inconsistent = claim.model_copy(
        update={"atomic_facts": [AtomicFact(fact_id="af:2", text="甲有影子", verdict="false")]}
    )
    with pytest.raises(ClaimRuleError):
        claim_verdict(inconsistent)


# ---------------------------------------------------------------------------
# answer / question builders
# ---------------------------------------------------------------------------


def test_make_answer_falls_back_to_span_and_hashes_quote():
    answer = make_answer("拉法", "u:zj:v01:c001", 2, 4, "拉法", unit_text="我见拉法")
    assert answer["start"] == 2 and answer["end"] == 4
    assert answer["quote_sha256"] == "sha256:" + sha256_text("拉法")
    assert answer["source"] == "narration"


def test_make_answer_rejects_bad_spans_and_quotes():
    with pytest.raises(ClaimBuildError):
        make_answer("拉法", "u:1", 0, 3, "拉法")  # span length != quote length
    with pytest.raises(ClaimBuildError):
        make_answer("拉法", "u:1", 0, 2, "拉法", unit_text="我见拉法")  # mismatch
    with pytest.raises(ClaimBuildError):
        make_answer("", "u:1", 0, 2, "拉法")
    with pytest.raises(ClaimBuildError):
        make_answer("拉法", "u:1", 0, 2, "拉法", quote_sha256="sha256:" + "0" * 64)
    with pytest.raises(ClaimBuildError):
        make_answer("拉法", "u:1", 0, 2, "拉法", source="dream")


def test_make_question_unanswerable_must_not_carry_answers():
    with pytest.raises(ClaimBuildError):
        make_question("q1", "谁？", answers=[_answer()], stance="unanswerable")
    with pytest.raises(ClaimBuildError):
        make_question("q1", "谁？", answers=[], stance="support")
    with pytest.raises(ClaimBuildError):
        make_question("q1", "谁？", stance="maybe")
    question = make_question("q1", "谁？", stance="unanswerable")
    assert question.answers == [] and question.stance == "unanswerable"


def test_normalize_claim_text_uses_literal_role_values():
    statement = _statement(roles={"主体": "e:zj:p:1", "值": {"literal": "苍蓝色"}})
    assert normalize_claim_text(statement) == "敌对：苍蓝色"


# ---------------------------------------------------------------------------
# build_claim happy paths
# ---------------------------------------------------------------------------


def test_build_claim_supported_statement():
    claim = _claim()
    assert claim.verdict == "supported"
    assert claim.status == "verified"
    assert claim.verdict_rule == "av-eritec"
    assert claim.confidence_scope == "claim"
    assert claim.confidence_method == "claim_qa"
    assert claim.claim_id == claim_id_for(STATEMENT_ID, "甲与乙敌对")
    assert claim.statement_assertion_id == STATEMENT_ID
    assert len(claim.questions) == 2
    assert claim.verified_by["context_mode"] == "isolated_source"


def test_build_claim_id_is_idempotent():
    assert _claim().claim_id == _claim().claim_id


def test_build_claim_unanswerable_is_insufficient_and_open():
    claim = build_claim(
        _statement(),
        questions=[make_question("q1", "谁与谁敌对？", stance="unanswerable")],
        normalized_claim="甲与乙敌对",
    )
    assert claim.verdict == "insufficient_evidence"
    assert claim.status == "open"


def test_build_claim_conflicting():
    questions = [
        make_question("q1", "谁？", answers=[_answer()], stance="support"),
        make_question("q2", "反证？", answers=[_answer("u:zj:v01:c003")], stance="refute"),
    ]
    claim = build_claim(_statement(), questions=questions, normalized_claim="甲与乙敌对")
    assert claim.verdict == "conflicting"
    assert claim.status == "verified"


def test_build_claim_accepts_consistent_atomic_facts():
    claim = build_claim(
        _statement(),
        questions=_support_questions(),
        normalized_claim="甲与乙敌对",
        atomic_facts=[AtomicFact(fact_id="af:1", text="甲与乙敌对", verdict="true")],
    )
    assert claim.verdict == "supported"


def test_build_claim_calibrated_requires_version():
    with pytest.raises(ClaimBuildError):
        build_claim(
            _statement(),
            questions=_support_questions(),
            normalized_claim="甲与乙敌对",
            confidence_method="calibrated",
            confidence=0.9,
        )
    claim = build_claim(
        _statement(),
        questions=_support_questions(),
        normalized_claim="甲与乙敌对",
        confidence_method="calibrated",
        confidence=0.9,
        calibration_version="cal-2026-10",
    )
    assert claim.verified_by["calibration_version"] == "cal-2026-10"


# ---------------------------------------------------------------------------
# build_claim negatives
# ---------------------------------------------------------------------------


def test_build_claim_rejects_non_statement_assertion():
    # negative: only claim_layer=statement can be claimed
    fact = _statement(claim_layer="fact", speaker=None)
    with pytest.raises(ClaimBuildError) as caught:
        build_claim(fact, questions=_support_questions(), normalized_claim="甲与乙敌对")
    assert caught.value.reason == "claim_layer_conflict"


def test_build_claim_rejects_missing_statement_id():
    statement = _statement(assertion_id=None)
    with pytest.raises(ClaimBuildError):
        build_claim(statement, questions=_support_questions(), normalized_claim="x")


def test_build_claim_rejects_no_questions():
    with pytest.raises(ClaimBuildError):
        build_claim(_statement(), questions=[], normalized_claim="x")


def test_build_claim_rejects_duplicate_qid():
    duplicate = [
        make_question("q1", "甲？", answers=[_answer()], stance="support"),
        make_question("q1", "乙？", answers=[_answer()], stance="support"),
    ]
    with pytest.raises(ClaimBuildError):
        build_claim(_statement(), questions=duplicate, normalized_claim="x")


def test_build_claim_verifies_answer_against_unit_text():
    bad_answer = make_answer("拉法", "u:zj:v01:c001", 0, 2, "拉法")
    question = make_question("q1", "甲？", answers=[bad_answer], stance="support")
    with pytest.raises(ClaimBuildError):
        build_claim(
            _statement(),
            questions=[question],
            normalized_claim="x",
            unit_texts={"u:zj:v01:c001": "不是这里"},
        )


def test_build_claim_rejects_inconsistent_atomic_facts():
    with pytest.raises(ClaimBuildError):
        build_claim(
            _statement(),
            questions=_support_questions(),
            normalized_claim="甲与乙敌对",
            atomic_facts=[AtomicFact(fact_id="af:1", text="甲与乙敌对", verdict="false")],
        )


def test_build_claim_rejects_bad_claim_type_and_context():
    for kwargs in (
        {"claim_type": "vibe"},
        {"context_mode": "saw_everything"},
        {"confidence_method": "engine_self"},
        {"confidence": 1.5},
    ):
        with pytest.raises(ClaimBuildError):
            build_claim(
                _statement(),
                questions=_support_questions(),
                normalized_claim="x",
                **kwargs,
            )


def test_unresolved_entities_block_stance_and_force_insufficient():
    # VeGraph order: resolve entities before judging a stance.
    with pytest.raises(ClaimBuildError) as caught:
        build_claim(
            _statement(),
            questions=_support_questions(),
            normalized_claim="甲与乙敌对",
            unresolved_entities=["神秘人"],
        )
    assert caught.value.reason == "entity_resolution_required"

    claim = build_claim(
        _statement(),
        questions=[make_question("q1", "谁？", stance="unanswerable")],
        normalized_claim="甲与乙敌对",
        unresolved_entities=["神秘人"],
    )
    assert claim.verdict == "insufficient_evidence"
    assert claim.verified_by["unresolved_entities"] == ["神秘人"]


def test_resolve_entities_uses_alias_map_before_stance():
    resolution = resolve_entities(
        ["心叶", "神秘人"],
        name_to_id={"贝塔心叶": "e:zj:p:1"},
        alias_map={"心叶": "贝塔心叶"},
    )
    assert resolution.resolved == {"心叶": "e:zj:p:1"}
    assert resolution.unresolved == ("神秘人",)
    assert resolution.complete is False


# ---------------------------------------------------------------------------
# promotion gate
# ---------------------------------------------------------------------------


def test_supported_claim_with_two_units_can_promote():
    claim = _claim()
    statement = _statement()
    decision = promotion_decision(claim, statement)
    assert decision.allowed is True
    assert decision.units == ("u:zj:v01:c001", "u:zj:v01:c002")
    assert decision.independent_source_count == 2
    assert support_answers(claim)


def test_insufficient_evidence_never_promotes():
    # negative: the mandated "insufficient must not become fact" case
    statement = _statement()
    claim = build_claim(
        statement,
        questions=[make_question("q1", "谁？", stance="unanswerable")],
        normalized_claim="甲与乙敌对",
    )
    assert claim.verdict == "insufficient_evidence"
    decision = promotion_decision(claim, statement)
    assert decision.allowed is False
    assert decision.reason == "insufficient_evidence_never_promotes"
    with pytest.raises(PromotionRefused) as caught:
        build_promoted_fact(claim, statement)
    assert caught.value.decision.reason == "insufficient_evidence_never_promotes"


def test_conflicting_and_refuted_claims_do_not_promote():
    statement = _statement()
    conflicting = build_claim(
        statement,
        questions=[
            make_question("q1", "甲？", answers=[_answer()], stance="support"),
            make_question("q2", "乙？", answers=[_answer("u:zj:v01:c003")], stance="refute"),
        ],
        normalized_claim="甲与乙敌对",
    )
    assert promotion_decision(conflicting, statement).reason == "verdict_conflicting_not_promotable"

    refuted = build_claim(
        statement,
        questions=[make_question("q1", "甲？", answers=[_answer()], stance="refute")],
        normalized_claim="甲与乙敌对",
    )
    assert refuted.verdict == "refuted"
    assert promotion_decision(refuted, statement).reason == "verdict_refuted_not_promotable"


def test_single_source_is_not_enough_but_two_speakers_are():
    statement = _statement()
    single = _claim(units=("u:zj:v01:c001",))
    decision = promotion_decision(single, statement)
    assert decision.allowed is False
    assert decision.reason == "needs_two_independent_sources"

    two_answers_one_unit = build_claim(
        statement,
        questions=[
            make_question(
                "q1",
                "甲与谁敌对？",
                answers=[_answer("u:zj:v01:c001"), _answer("u:zj:v01:c001", start=2)],
                stance="support",
            )
        ],
        normalized_claim="甲与乙敌对",
    )
    with_speakers = promotion_decision(
        two_answers_one_unit,
        statement,
        answer_speakers=["e:zj:p:1", "e:zj:p:2"],
    )
    assert with_speakers.allowed is True
    assert with_speakers.speakers == ("e:zj:p:1", "e:zj:p:2")


def test_promotion_rejects_stale_verdict_and_mismatched_statement():
    statement = _statement()
    insufficient = build_claim(
        statement,
        questions=[make_question("q1", "谁？", stance="unanswerable")],
        normalized_claim="甲与乙敌对",
    )
    stale = insufficient.model_copy(update={"verdict": "supported"})
    assert promotion_decision(stale, statement).reason == (
        "claim_verdict_stale_insufficient_evidence"
    )

    other = _statement(assertion_id="a:fedcba9876543210")
    assert promotion_decision(_claim(), other).reason == "claim_statement_mismatch"


def test_promotion_requires_statement_layer():
    statement = _statement()
    fact = statement.model_copy(update={"claim_layer": "fact"})
    assert promotion_decision(_claim(), fact).reason == "statement_assertion_not_statement"


def test_build_promoted_fact_creates_new_fact_assertion():
    statement = _statement()
    claim = _claim()
    fact = build_promoted_fact(claim, statement)

    assert fact.claim_layer == "fact"
    assert fact.assertion_id is not None and fact.assertion_id.startswith("a:")
    assert fact.assertion_id != statement.assertion_id
    assert fact.derived_from == [STATEMENT_ID]
    assert fact.status == "asserted"
    assert fact.confidence_method == "claim_qa"
    assert fact.confidence_scope == "assertion"
    assert [item.unit_id for item in fact.evidence] == [
        "u:zj:v01:c001",
        "u:zj:v01:c002",
    ]
    assert all(item.mode == "narration" for item in fact.evidence)
    # the statement is never mutated in place
    assert statement.claim_layer == "statement"
    assert claim.statement_assertion_id == STATEMENT_ID


def test_promotion_reasons_are_named():
    assert independent_sources([_answer()], answer_speakers=None) == (
        ("u:zj:v01:c001",),
        (),
    )


# ---------------------------------------------------------------------------
# cross-module chain: identity -> assembly -> claim -> promotion
# ---------------------------------------------------------------------------


def test_identity_to_claim_chain_composes():
    from cbb_stage1.assemble import Assembler, seed_registry
    from cbb_stage1.engines.ai_reader.schema import EntityDictEntry
    from cbb_stage1.resolve import build_alias_map

    alias_map = build_alias_map([
        EntityDictEntry(
            name="贝塔心叶", entity_type="person", frequency=10,
            aliases=["心叶"], source="freq",
        )
    ])
    resolution = resolve_entities(
        ["心叶"], name_to_id={"贝塔心叶": "e:zj:p:1"}, alias_map=alias_map
    )
    assert resolution.resolved["心叶"] == "e:zj:p:1"

    speaker = Speaker(kind="entity", entity_id="e:zj:p:1", name="贝塔心叶")
    raw = _statement(
        speaker=speaker,
        evidence=[
            Evidence(
                unit_id="u:zj:v01:c001", start=0, end=2, quote="拉法",
                speaker=speaker, mode="dialogue",
            )
        ],
    )
    assembled = Assembler(seed_registry(), book_id="zj").assemble([raw])
    statement = assembled.accepted[0]
    claim = build_claim(
        statement,
        questions=_support_questions(),
        normalized_claim="甲与乙敌对",
        confidence=0.9,
    )
    fact = build_promoted_fact(claim, statement)
    assert fact.claim_layer == "fact"
    assert fact.derived_from == [statement.assertion_id]
    assert fact.evidence and all(item.unit_id.startswith("u:") for item in fact.evidence)
