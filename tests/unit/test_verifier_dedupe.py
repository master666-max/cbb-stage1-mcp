"""The verifier must not be asked the same question twice.

Measured on a 5000-character run: 50 verifier requests covering **34 distinct spans** — 16
re-dispatches, one span sent five times. The answer for a given (unit, span, quote) cannot
change between dispatches: the evidence is a fixed file and the quote is fixed. So a repeat
is pure cost, and on a long unit it is what keeps the queue from ever emptying: the run
reached 54 requests and was still in the verification stage.

Where the duplication comes from
--------------------------------
A claim can carry the same evidence more than once (the extractor emits overlapping quotes,
and gleaning re-emits evidence already present). `verify_evidence` is called per evidence
item, with no memory of what has already been asked. The fix is a per-unit memo keyed on the
thing that determines the answer:

    (unit_id, start, end, quote_expected)

Not on the claim id — two claims citing the same sentence get the same verdict, and asking
twice is the waste being removed.

What must not change
--------------------
Every distinct evidence item must still be verified, and a repeated ask must still return a
verdict (from the memo) rather than being dropped. The caller cannot tell the difference,
which is the point: same answer, one fewer request.
"""

from __future__ import annotations

from typing import Any

import pytest

from cbb_stage1.orchestrator.role_runtime import build_verifier_input


class CountingLLM:
    """Counts how many times it is asked, and answers `support` to everything."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, record: dict[str, Any]) -> str:
        self.calls.append(record)
        return "support"


# The quote must be the text that actually sits at the span, or the deterministic check
# decides `against` before the model is asked and these tests measure nothing.
UNIT = "".join(chr(0x4E00 + (index % 500)) for index in range(3000))
EVIDENCE = {"start": 1200, "end": 1210, "quote": UNIT[1200:1210]}


# ── the memo ────────────────────────────────────────────────────────────
def test_a_repeated_evidence_item_asks_the_model_once() -> None:
    """Same unit, same span, same quote: one request, two verdicts."""
    from cbb_stage1.orchestrator.role_runtime import VerifierCache, verify_evidence

    llm = CountingLLM()
    cache = VerifierCache()
    first = verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)
    second = verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)

    assert len(llm.calls) == 1, f"asked {len(llm.calls)} times for one span"
    assert first.verdict == second.verdict


def test_both_callers_still_get_a_verdict() -> None:
    """A cache hit must return the recorded verdict, not None or 'unsure'."""
    from cbb_stage1.orchestrator.role_runtime import VerifierCache, verify_evidence

    llm = CountingLLM()
    cache = VerifierCache()
    verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)
    second = verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)
    assert second.verdict == "support"


def test_a_different_span_is_a_different_question() -> None:
    from cbb_stage1.orchestrator.role_runtime import VerifierCache, verify_evidence

    llm = CountingLLM()
    cache = VerifierCache()
    verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)
    verify_evidence(
        UNIT, unit_id="u:1",
        evidence={"start": 1500, "end": 1510, "quote": UNIT[1500:1510]},
        llm=llm, cache=cache,
    )
    assert len(llm.calls) == 2


def test_a_different_quote_at_the_same_span_is_a_different_question() -> None:
    """The quote is part of the key: two asks about the same slice are not the same ask.

    The second quote deliberately does not match the span, so it is decided by the
    deterministic check and never reaches the model — which is correct behaviour, and the
    reason this test asserts on the memo rather than on `llm.calls`. Conflating the two was
    a mistake in the first draft of this test.
    """
    from cbb_stage1.orchestrator.role_runtime import VerifierCache, verify_evidence

    llm = CountingLLM()
    cache = VerifierCache()
    verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)
    other = verify_evidence(
        UNIT, unit_id="u:1",
        evidence={"start": 1200, "end": 1210, "quote": UNIT[1500:1510]},
        llm=llm, cache=cache,
    )
    assert len(cache) == 2, "two distinct (span, quote) pairs must memoise separately"
    assert other.verdict == "against", "a quote that is not at its span is a mismatch"
    assert len(llm.calls) == 1, "the mismatch never needs a model call"


def test_a_different_unit_is_a_different_question() -> None:
    from cbb_stage1.orchestrator.role_runtime import VerifierCache, verify_evidence

    llm = CountingLLM()
    cache = VerifierCache()
    verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)
    verify_evidence(UNIT, unit_id="u:2", evidence=EVIDENCE, llm=llm, cache=cache)
    assert len(llm.calls) == 2


def test_no_cache_still_verifies_every_time() -> None:
    """Without a cache the old behaviour is unchanged — this is opt-in."""
    from cbb_stage1.orchestrator.role_runtime import verify_evidence

    llm = CountingLLM()
    verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm)
    verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm)
    assert len(llm.calls) == 2


def test_the_cache_reports_its_hit_count() -> None:
    """A reader must be able to see how much was saved, not just trust that it was."""
    from cbb_stage1.orchestrator.role_runtime import VerifierCache, verify_evidence

    llm = CountingLLM()
    cache = VerifierCache()
    for _ in range(4):
        verify_evidence(UNIT, unit_id="u:1", evidence=EVIDENCE, llm=llm, cache=cache)
    assert len(llm.calls) == 1
    assert cache.hits == 3
    assert cache.misses == 1


def test_the_deterministic_path_is_also_cached() -> None:
    """A quote that does not match its span is decided without the model — twice is still waste."""
    from cbb_stage1.orchestrator.role_runtime import VerifierCache, verify_evidence

    llm = CountingLLM()
    cache = VerifierCache()
    bad = {"start": 1200, "end": 1210, "quote": "this text is not at that span"}
    first = verify_evidence(UNIT, unit_id="u:1", evidence=bad, llm=llm, cache=cache)
    second = verify_evidence(UNIT, unit_id="u:1", evidence=bad, llm=llm, cache=cache)
    assert first.verdict == second.verdict
    assert len(llm.calls) == 0, "a deterministically-decided item needs no model call at all"
    assert cache.hits == 1


# ── the request shape must not regress ─────────────────────────────────
def test_the_verifier_input_still_carries_the_unit_text() -> None:
    """The contract (design 04 §8) says the request carries the original text. Keep it."""
    record = build_verifier_input(UNIT, unit_id="u:1", evidence=EVIDENCE)
    assert record["text"] == UNIT
    assert record["contract"] == "cbb-verifier-input/1"
