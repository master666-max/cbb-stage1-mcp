"""VeGraph's second stage must be an iterative clarification loop, not one dictionary hit.

Measured gap (B9)
-----------------
`reasoning/vegraph.py:40-47` declares two capabilities:

    triples(claim) -> [...]
    verify_subassertion(triple) -> stance

and `vegaph_verify` does exactly one pass over the triples. The paper's stage 2 is a loop: generate a
**clarifying question** for a sub-assertion that cannot be judged, retrieve/answer it, feed the
**rationale back**, update the graph across groups, and repeat up to k times. None of that exists, so
the mechanism reduces to "ask the model to judge each triple once" — which is a dictionary hit with a
model in the middle.

What is pinned
--------------
* a clarifying question is asked when a sub-assertion comes back unanswerable, and the answer's
  rationale is fed into the next round's judgement;
* the loop is bounded by k (the paper's 5) and **stops early when a round changes nothing** — the same
  discipline AFEV's loop has, and for the same reason;
* a sub-assertion that stays unanswerable is reported as such rather than being promoted by repetition;
* the number of rounds actually used is recorded, so a reader can tell a one-pass run from a loop.

The clarifying capability is **optional** on the protocol: a model without it keeps working and simply
cannot clarify, which is the honest degradation rather than a fabricated second stage.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cbb_stage1.reasoning.vegraph import DEFAULT_MAX_ROUNDS, vegraph_verify

NAME_TO_ID = {"言万": "e:t:person:aaaaaaaaaaaa", "拉法": "e:t:person:bbbbbbbbbbbb"}


class ClarifyingLLM:
    """Triples, per-triple stances, and clarifications that change a stance.

    `clarify_after` controls which round a stance flips on, so a test can distinguish "stopped because
    it became judgeable" from "ran out of rounds".
    """

    def __init__(self, *, clarify_after: int = 1, always_unanswerable: bool = False) -> None:
        self.clarify_after = clarify_after
        self.always_unanswerable = always_unanswerable
        self.clarify_calls: list[str] = []
        self.verify_calls: list[tuple[str, str]] = []

    def triples(self, claim: str) -> Sequence[Mapping[str, Any]]:
        return [
            {"subject": "言万", "predicate": "敌对", "object": "拉法"},
        ]

    def verify_subassertion(self, triple: Mapping[str, Any]) -> str:
        rationale = str(triple.get("rationale") or "")
        self.verify_calls.append((str(triple.get("subject")), rationale))
        if self.always_unanswerable:
            return "unanswerable"
        # Judgeable only once a rationale from a clarification has been fed back.
        return "support" if rationale else "unanswerable"

    def clarify(self, triple: Mapping[str, Any], question: str) -> str:
        self.clarify_calls.append(question)
        if len(self.clarify_calls) >= self.clarify_after:
            return "原文写明二人交手于集市"
        return ""


class NoClarifyLLM(ClarifyingLLM):
    """A model that lacks the optional capability entirely."""


def _verify(llm: Any, **kwargs: Any):
    return vegraph_verify(
        "言万与拉法敌对",
        book_id="t",
        surfaces=("言万", "拉法"),
        name_to_id=NAME_TO_ID,
        llm=llm,
        **kwargs,
    )


# ══ the loop exists ═══════════════════════════════════════════════════
def test_a_clarifying_question_is_asked() -> None:
    llm = ClarifyingLLM()
    _verify(llm)
    assert llm.clarify_calls, (
        "no clarifying question was asked; stage 2 is a single pass over the triples, which is the "
        "defect B9 names"
    )


def test_the_rationale_is_fed_back() -> None:
    llm = ClarifyingLLM()
    _verify(llm)
    rationales = [rationale for _subject, rationale in llm.verify_calls]
    assert any(rationale.strip() for rationale in rationales), (
        "the clarification's answer never reached the verifier, so it cannot change the judgement"
    )


def test_the_loop_records_how_many_rounds_ran() -> None:
    result = _verify(ClarifyingLLM())
    assert result.rounds >= 1, f"the result does not say how many rounds ran: {result.rounds}"


def test_the_default_budget_is_the_papers_five() -> None:
    assert DEFAULT_MAX_ROUNDS == 5


# ══ it stops when there is nothing more to learn ════════════════════
def test_an_unanswerable_sub_assertion_stops_the_loop() -> None:
    """Repeating the same question five times is cost without information."""
    llm = ClarifyingLLM(always_unanswerable=True, clarify_after=99)
    result = _verify(llm)
    assert result.rounds < DEFAULT_MAX_ROUNDS, (
        f"the loop ran {result.rounds} rounds on an unanswerable triple"
    )


def test_a_resolved_sub_assertion_ends_the_loop_early() -> None:
    llm = ClarifyingLLM(clarify_after=1)
    result = _verify(llm)
    assert result.verdict == "supported"
    assert result.rounds <= 2, f"kept looping after the triple became judgeable: {result.rounds}"


def test_an_unanswerable_verdict_is_never_promoted_by_repetition() -> None:
    result = _verify(ClarifyingLLM(always_unanswerable=True, clarify_after=99))
    assert result.verdict == "insufficient_evidence", (
        f"repetition turned an unanswerable triple into {result.verdict!r}"
    )


# ══ a model without the capability still works ══════════════════════
def test_a_model_without_clarify_still_produces_a_verdict() -> None:
    class Plain:
        def triples(self, claim: str) -> Sequence[Mapping[str, Any]]:
            return [{"subject": "言万", "predicate": "敌对", "object": "拉法"}]

        def verify_subassertion(self, triple: Mapping[str, Any]) -> str:
            return "support"

    result = vegraph_verify(
        "言万与拉法敌对",
        book_id="t",
        surfaces=("言万", "拉法"),
        name_to_id=NAME_TO_ID,
        llm=Plain(),
    )
    assert result.verdict == "supported"
    assert result.rounds == 1


def test_no_model_is_still_degenerate() -> None:
    result = vegraph_verify(
        "言万与拉法敌对", book_id="t", surfaces=("言万",), name_to_id=NAME_TO_ID
    )
    assert result.degenerate is True
    assert result.verdict == "insufficient_evidence"


def test_unresolved_entities_still_block_judgement() -> None:
    """The ordering rule must survive the loop."""
    result = vegraph_verify(
        "无名与拉法敌对",
        book_id="t",
        surfaces=("无名",),
        name_to_id=NAME_TO_ID,
        llm=ClarifyingLLM(),
    )
    assert result.unresolved_entities
    assert result.verdict == "insufficient_evidence"
