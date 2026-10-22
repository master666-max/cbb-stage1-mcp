"""AFEV must not burn a round re-verifying facts it already proved unverifiable.

What actually happened
----------------------
A 5000-character run stopped in the claims stage after 38 host-queue requests. The first
reading was "AFEV loops forever". Reading the code corrected that: ``afev_verify`` iterates
``for round_no in range(1, max_rounds + 1)`` with ``DEFAULT_MAX_ROUNDS = 3``, so it is
bounded. The same assertion being decomposed twice was rounds 1 and 2 of a design that
feeds the ``unverifiable`` rationale into the next round — working as documented, not a bug.

The real cost was volume: 19 assembled assertions x up to 3 rounds x (1 decompose + N verify)
is roughly 170 requests, and the driving agent was answering them one turn at a time.

The defect that *is* real
-------------------------
When a round's decomposition produces **exactly the facts the previous round produced**,
nothing has been learned. The evidence has not changed, the facts have not changed, so the
verdicts cannot change. Round 3 will decompose to the same texts and verify them to the same
``unverifiable``. That round is pure cost, and on the 5000-character run it is a third of the
AFEV requests for every claim whose evidence simply does not cover the claim.

So: stop when the decomposition fingerprint repeats. The verdict is unchanged, one round is
saved, and the result records why it stopped instead of leaving the reader to wonder.

This is an efficiency fix, not a behaviour change — the test below pins both halves: the
verdict must stay ``insufficient_evidence`` and the round count must drop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from cbb_stage1.reasoning import atomic
from cbb_stage1.reasoning.atomic import afev_verify


class RepeatingLLM:
    """Decomposes to the same two facts every round; verifies everything unverifiable.

    This is the shape the real run had: the claim is an appearance description
    ("小柴 / 有着藤紫色头发…"), the evidence is a line of dialogue that says nothing about
    appearance, so every atomic fact is honestly unverifiable — and every round reproduces
    the same two facts.
    """

    def __init__(self) -> None:
        self.decompose_calls = 0
        self.verify_calls = 0

    def decompose(self, claim: str, context: str | None) -> Sequence[str]:
        self.decompose_calls += 1
        return ["小柴有着藤紫色头发", "小柴像小动物一样"]

    def verify(self, fact: str, evidence: Sequence[str]) -> dict[str, Any]:
        self.verify_calls += 1
        return {"verdict": "unverifiable", "rationale": "证据未提及"}


class ChangingLLM:
    """Decomposes to different facts each round — the loop must keep going."""

    def __init__(self) -> None:
        self.decompose_calls = 0
        self.rounds: list[list[str]] = [
            ["事实甲"],
            ["事实甲", "事实乙"],
            ["事实甲", "事实乙", "事实丙"],
        ]

    def decompose(self, claim: str, context: str | None) -> Sequence[str]:
        index = min(self.decompose_calls, len(self.rounds) - 1)
        self.decompose_calls += 1
        return self.rounds[index]

    def verify(self, fact: str, evidence: Sequence[str]) -> dict[str, Any]:
        return {"verdict": "unverifiable", "rationale": f"{fact} 证据不足"}


# ── the fix ──────────────────────────────────────────────────────────────
def test_a_repeated_decomposition_stops_the_rounds() -> None:
    """Same facts twice means round 3 cannot learn anything. It must not run."""
    llm = RepeatingLLM()
    result = afev_verify("性质：小柴 / 有着藤紫色头发，像小动物一样的少女",
                         book_id="t", evidence=["「小柴喵呜，前来拜见！」"], llm=llm)
    assert llm.decompose_calls == 2, (
        f"decomposed {llm.decompose_calls} times; round 3 repeats round 2 exactly and "
        "cannot change the verdict"
    )
    assert result.rounds == 2


def test_a_repeated_decomposition_still_reports_insufficient_evidence() -> None:
    """The early stop must not change the verdict — only the cost."""
    llm = RepeatingLLM()
    result = afev_verify("性质：小柴 / 有着藤紫色头发", book_id="t",
                         evidence=["无关证据"], llm=llm)
    assert result.verdict == "insufficient_evidence"
    assert result.degenerate is False
    assert len(result.atomic_facts) == 2


def test_the_early_stop_is_recorded_rather_than_silent() -> None:
    """A reader must be able to tell 'stopped because nothing changed' from 'hit max_rounds'."""
    llm = RepeatingLLM()
    result = afev_verify("性质：小柴 / 有着藤紫色头发", book_id="t",
                         evidence=["无关证据"], llm=llm)
    joined = " ".join(result.notes)
    assert "repeat" in joined.lower(), f"notes do not explain the stop: {result.notes}"


# ── what must NOT change ────────────────────────────────────────────────
def test_changing_decompositions_still_use_every_round() -> None:
    """If the facts do change each round, the loop is doing real work and must continue."""
    llm = ChangingLLM()
    result = afev_verify("某断言", book_id="t", evidence=["证据"], llm=llm, max_rounds=3)
    assert llm.decompose_calls == 3
    assert result.rounds == 3


def test_a_verdict_of_true_still_breaks_immediately() -> None:
    """Success short-circuits before any repeat check is reached."""

    class FirstRoundTrue(RepeatingLLM):
        def verify(self, fact: str, evidence: Sequence[str]) -> dict[str, Any]:
            self.verify_calls += 1
            return {"verdict": "true", "rationale": "证据支持"}

    llm = FirstRoundTrue()
    result = afev_verify("某断言", book_id="t", evidence=["证据"], llm=llm)
    assert llm.decompose_calls == 1
    # aggregate_atomic_facts maps all-true to "supported", not the per-fact word.
    assert result.verdict == "supported"


def test_a_degenerate_no_model_run_is_untouched() -> None:
    """No llm means no rounds at all — the early stop must not disturb that path."""
    result = afev_verify("某断言", book_id="t", evidence=["证据"], llm=None)
    assert result.rounds == 0
    assert result.degenerate is True
    assert result.verdict == "insufficient_evidence"


def test_the_round_cap_still_binds() -> None:
    """One round is one round; the repeat check cannot make it exceed the cap."""
    llm = RepeatingLLM()
    result = afev_verify("某断言", book_id="t", evidence=["证据"], llm=llm, max_rounds=1)
    assert llm.decompose_calls == 1
    assert result.rounds == 1


def test_the_module_states_the_bound() -> None:
    """The bound must be visible where a reader looks, because I got it wrong once."""
    assert atomic.DEFAULT_MAX_ROUNDS == 3
