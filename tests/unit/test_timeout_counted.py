"""A host-queue timeout is not the model saying "I don't know".

`verify_evidence` catches every exception from the verifier and records `unsure`, which is the
right verdict — a check that could not run must not be reported as support. But `unsure` then
means two different things in `counts`:

* the model looked at the evidence and could not decide (a real judgement);
* nobody answered in 900 seconds (no judgement at all).

Measured on a real run: 4 of the 33 verifier calls timed out because the driving agent was slow,
and they were indistinguishable in the receipt from genuine `unsure` verdicts. That makes the
verify receipt unable to answer "was this run actually verified", which is the question a
reader of a box asks first.

The fix keeps the verdict (`unsure`) and separates the **cause**, so the count is honest:
`counts["timed_out"]` is reported next to `support/against/unsure`.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.agents.host_queue import HostQueueTimeout
from cbb_stage1.orchestrator import role_runtime

UNIT = "".join(chr(0x4E00 + (index % 500)) for index in range(1500))
UNIT_ID = "u:tst:v01:c001"


class TimeoutLLM:
    """Raises HostQueueTimeout for every call, like an unanswered queue request."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, record: dict[str, Any]) -> str:
        self.calls += 1
        raise HostQueueTimeout("host model did not reply to x within 900.0s")


class Output:
    def __init__(self, assertions: list[dict]) -> None:
        self.unit_id = UNIT_ID
        self.assertions = assertions


def _evidence() -> dict:
    return {"start": 100, "end": 112, "quote": UNIT[100:112], "occurrence": 1}


def test_a_timeout_is_still_unsure_not_fabricated() -> None:
    """The verdict must not change: an unanswered check is never support."""
    llm = TimeoutLLM()
    verdicts, _counts, _ = role_runtime.verify_output(
        Output([{"assertion_id": "a:1", "evidence": [_evidence()]}]),
        UNIT,
        unit_id=UNIT_ID,
        llm=llm,
    )
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "unsure"


def test_a_timeout_is_counted_separately() -> None:
    llm = TimeoutLLM()
    _verdicts, counts, _ = role_runtime.verify_output(
        Output([{"assertion_id": "a:1", "evidence": [_evidence()]}]),
        UNIT,
        unit_id=UNIT_ID,
        llm=llm,
    )
    assert counts.get("timed_out") == 1, f"timeout not separated from unsure: {counts}"
    assert counts["unsure"] == 1, "the verdict count itself must be unchanged"


def test_the_timeout_reason_is_named() -> None:
    """A reader of the verdict must see why it is `unsure`."""
    llm = TimeoutLLM()
    verdicts, _counts, _ = role_runtime.verify_output(
        Output([{"assertion_id": "a:1", "evidence": [_evidence()]}]),
        UNIT,
        unit_id=UNIT_ID,
        llm=llm,
    )
    assert "host_queue_timeout" in str(verdicts[0].reason), verdicts[0].reason


def test_a_plain_failure_is_not_counted_as_a_timeout() -> None:
    """Only the queue timeout is a transport problem; a model error is a model error."""

    class Boom:
        def __call__(self, record: dict[str, Any]) -> str:
            raise RuntimeError("model exploded")

    _verdicts, counts, _ = role_runtime.verify_output(
        Output([{"assertion_id": "a:1", "evidence": [_evidence()]}]),
        UNIT,
        unit_id=UNIT_ID,
        llm=Boom(),
    )
    assert counts.get("timed_out", 0) == 0
    assert counts["unsure"] == 1
