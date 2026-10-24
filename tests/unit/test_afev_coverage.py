"""AFEV's stopping rule must be the paper's, or say plainly that it is a bounded approximation.

Measured drift (B7)
-------------------
A13-AFEV-2025 p.9:

    "At each iteration t, the model determines whether the extraction process should terminate or
     continue ... If the extracted facts collectively provide sufficient coverage of the claim, the
     process terminates."

The implementation loops `for round_no in range(1, max_rounds + 1)` with `DEFAULT_MAX_ROUNDS = 3` and
no comment at the constant. It stops when (a) every fact is `true`/`false`, (b) the decomposition
repeats, or (c) the rounds run out. Coverage sufficiency — the paper's actual criterion — is never
asked of the model, and nothing says the round cap is an approximation.

Why this is a drift and not a substitute
----------------------------------------
`(a)` is "every fact resolved", which is *not* coverage: three facts can all be `true` while the claim
has a part nobody decomposed. So the implemented rule answers a different question than the paper's,
and no comment admits it.

What is pinned
--------------
1. the model gets asked whether coverage is sufficient, and its answer can stop the loop early;
2. `max_rounds` still bounds the loop — the approximation is kept, not removed;
3. the constant carries a comment saying it is a bound, with the paper's criterion named;
4. a run that stops on coverage is recorded distinctly from one that ran out of rounds.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

from cbb_stage1.reasoning import atomic
from cbb_stage1.reasoning.atomic import afev_verify


class CoverageLLM:
    """Decomposes, verifies, and reports whether coverage is sufficient.

    `sufficient_after` controls when `coverage_sufficient` starts returning True, so a test can tell
    "stopped because the model said so" from "stopped because rounds ran out".
    """

    def __init__(self, *, sufficient_after: int = 1, always_unverifiable: bool = False) -> None:
        self.sufficient_after = sufficient_after
        self.always_unverifiable = always_unverifiable
        self.coverage_calls = 0
        self.decompose_calls = 0

    def decompose(self, claim: str, context: str | None) -> Sequence[str]:
        self.decompose_calls += 1
        # A different decomposition each round, so the repeat-guard cannot end the loop.
        return [f"事实{self.decompose_calls}-甲", f"事实{self.decompose_calls}-乙"]

    def verify(self, fact: str, evidence: Sequence[str]) -> Mapping[str, Any]:
        if self.always_unverifiable:
            return {"verdict": "unverifiable", "rationale": "证据不足"}
        return {"verdict": "unverifiable", "rationale": "仍缺一环"}

    def coverage_sufficient(self, claim: str, facts: Sequence[str]) -> bool:
        self.coverage_calls += 1
        return self.coverage_calls >= self.sufficient_after


# ══ the paper's criterion is asked ════════════════════════════════════
def test_the_model_is_asked_whether_coverage_is_sufficient() -> None:
    llm = CoverageLLM(sufficient_after=99, always_unverifiable=True)
    afev_verify("某主张", book_id="t", llm=llm, max_rounds=3)
    assert llm.coverage_calls > 0, (
        "the model was never asked whether the facts cover the claim; the loop only checks whether "
        "every fact resolved, which is a different question"
    )


def test_a_sufficient_coverage_answer_stops_the_loop_early() -> None:
    llm = CoverageLLM(sufficient_after=1, always_unverifiable=True)
    result = afev_verify("某主张", book_id="t", llm=llm, max_rounds=5)
    assert result.rounds == 1, (
        f"the loop ran {result.rounds} rounds although coverage was sufficient after the first"
    )


def test_the_stop_reason_is_recorded() -> None:
    llm = CoverageLLM(sufficient_after=1, always_unverifiable=True)
    result = afev_verify("某主张", book_id="t", llm=llm, max_rounds=5)
    assert "stopped_on_sufficient_coverage" in result.notes, (
        f"a coverage stop is indistinguishable from a round-exhaustion stop: {result.notes}"
    )


# ══ the bound stays ══════════════════════════════════════════════════
def test_the_round_cap_still_bounds_the_loop() -> None:
    llm = CoverageLLM(sufficient_after=99, always_unverifiable=True)
    result = afev_verify("某主张", book_id="t", llm=llm, max_rounds=3)
    assert result.rounds <= 3, f"the cap was exceeded: {result.rounds}"
    assert "unverifiable_after_max_rounds" in result.notes


def test_a_llm_without_the_coverage_method_still_works() -> None:
    """Older/fake models must not break: absence of the method is not an error."""

    class Plain:
        def decompose(self, claim: str, context: str | None) -> Sequence[str]:
            return ["甲"]

        def verify(self, fact: str, evidence: Sequence[str]) -> Mapping[str, Any]:
            return {"verdict": "true", "rationale": "ok"}

    result = afev_verify("某主张", book_id="t", llm=Plain(), max_rounds=3)
    assert result.rounds == 1
    assert result.verdict == "supported"


# ══ the constant must admit it is a bound ════════════════════════════
def test_the_round_constant_documents_itself() -> None:
    """`DEFAULT_MAX_ROUNDS = 3` with no comment is how a drift stays invisible."""
    source = inspect.getsource(atomic)
    lines = source.splitlines()
    # Locate the assignment, not the `__all__` string that also names it.
    index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("DEFAULT_MAX_ROUNDS =")
    )
    context = "\n".join(lines[max(0, index - 12) : index + 1])
    assert "#" in context, (
        "DEFAULT_MAX_ROUNDS has no comment; the paper's criterion is model-judged coverage, so the "
        "cap must be labelled a bounded approximation"
    )
    lowered = context.lower()
    assert "coverage" in lowered or "覆盖" in lowered, (
        f"the comment does not name the paper's criterion: {context!r}"
    )
    assert "approximation" in lowered or "bound" in lowered or "近似" in lowered, (
        f"the comment does not say the cap is a bound: {context!r}"
    )
