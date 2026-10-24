"""The cache must be created by the real verification loop, not only by a test.

A `VerifierCache` that no production caller instantiates is a module-level object with a
green test suite — the exact failure this project keeps writing tests against. So this file
drives `verify_output` itself, with assertions that share one evidence item, and checks the
model is asked once.

That is the measured shape of the defect: on a 5000-character run, 50 verifier requests
covered only 34 distinct spans, one span sent five times.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.orchestrator import role_runtime

UNIT = "".join(chr(0x4E00 + (index % 500)) for index in range(2000))
UNIT_ID = "u:t:v01:c001"


class CountingLLM:
    """Records every prompt it receives, answers `support`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, record: dict[str, Any]) -> str:
        self.calls.append(record)
        return "support"


class Output:
    """Minimal stand-in for an EngineOutput: a unit id and a list of assertions."""

    def __init__(self, assertions: list[dict]) -> None:
        self.unit_id = UNIT_ID
        self.assertions = assertions


def evidence(start: int = 700, end: int = 712) -> dict:
    return {"start": start, "end": end, "quote": UNIT[start:end], "occurrence": 1}


def verify(assertions: list[dict], llm: CountingLLM):
    return role_runtime.verify_output(
        Output(assertions), UNIT, unit_id=UNIT_ID, llm=llm
    )


# ── the fix ─────────────────────────────────────────────────────────────
def test_the_same_evidence_inside_one_unit_is_asked_once() -> None:
    """Two assertions citing the same sentence: one model call, two verdicts."""
    llm = CountingLLM()
    verdicts, counts, _ = verify(
        [
            {"assertion_id": "a:1", "evidence": [evidence()]},
            {"assertion_id": "a:2", "evidence": [evidence()]},
        ],
        llm,
    )
    assert len(verdicts) == 2, "both assertions must still receive a verdict"
    assert len(llm.calls) == 1, f"asked {len(llm.calls)} times for one span"
    assert counts.get("deduped") == 1, f"hit count not reported in {counts}"


def test_five_copies_of_one_evidence_are_asked_once() -> None:
    """The real run had one span dispatched five times; the loop must collapse that."""
    llm = CountingLLM()
    verdicts, counts, _ = verify(
        [{"assertion_id": f"a:{i}", "evidence": [evidence()]} for i in range(5)],
        llm,
    )
    assert len(verdicts) == 5
    assert len(llm.calls) == 1
    assert counts.get("deduped") == 4


# ── what must not change ───────────────────────────────────────────────
def test_distinct_evidence_still_gets_distinct_calls() -> None:
    """Deduplication must not collapse different questions."""
    llm = CountingLLM()
    _verdicts, counts, _ = verify(
        [
            {"assertion_id": "a:1", "evidence": [evidence(700, 712)]},
            {"assertion_id": "a:2", "evidence": [evidence(900, 912)]},
        ],
        llm,
    )
    assert len(llm.calls) == 2
    assert counts.get("deduped", 0) == 0


def test_an_assertion_without_evidence_still_needs_no_model_call() -> None:
    llm = CountingLLM()
    verdicts, counts, _ = verify([{"assertion_id": "a:1", "evidence": []}], llm)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "unsure"
    assert len(llm.calls) == 0
    assert counts.get("unsure") == 1


def test_a_mismatched_quote_is_caught_without_a_model_call_and_memoised() -> None:
    """Two assertions with the same bad quote: one memo entry, zero model calls."""
    llm = CountingLLM()
    bad = {"start": 700, "end": 712, "quote": UNIT[900:912]}
    verdicts, _counts, _ = verify(
        [
            {"assertion_id": "a:1", "evidence": [bad]},
            {"assertion_id": "a:2", "evidence": [bad]},
        ],
        llm,
    )
    assert len(verdicts) == 2
    assert all(v.verdict == "against" for v in verdicts)
    assert len(llm.calls) == 0
