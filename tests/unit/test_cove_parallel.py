"""CoVe's independent answers must run concurrently, because the paper says they can.

Measured gap (B1)
-----------------
`validate/cove.py:311-316` answers the verification questions in a serial `for` loop. The paper is
explicit that these are the parallel part (A09, p.4):

    While this is potentially more computationally expensive, requiring the execution of many more
    LLM prompts, they can be run in parallel, and hence be batched.

So the factored variant is the right one and the execution model is not: N questions cost N
sequential round-trips, and the reason the paper chose factored CoVe (many independent prompts) is
exactly what the implementation fails to exploit.

What is pinned
--------------
* the questions are answered on more than one thread when `workers > 1`;
* the answers come back in question order, because the comparison step and the recorded
  `questions` tuple must not depend on scheduling;
* with `workers <= 1` the behaviour is unchanged (one thread), so callers that need determinism keep
  it;
* the resulting `CoVeResult` is the same value either way.

Parallelism is inferred from thread identities rather than elapsed time: a wall-clock assertion would
be flaky on a loaded machine and would prove nothing on a fast one.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

from cbb_stage1.validate.cove import cove_verify


class RecordingLLM:
    """Answers each prompt, recording which thread handled it.

    Prompts are routed by content so the test does not depend on prompt formatting: the planning
    prompt produces questions, each verification prompt produces an answer, and the comparison
    prompt reports no inconsistency.
    """

    def __init__(self) -> None:
        self.threads: list[int] = []
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        """`cove._ask` calls `.generate`; a model without it is treated as absent."""
        self.threads.append(threading.get_ident())
        self.calls.append(prompt)
        lowered = prompt.lower()
        if "问题" in prompt and "生成" in prompt or "plan" in lowered:
            return "1. 言万此时在哪？\n2. 拉法是否在场？\n3. 令牌是否出现？\n4. 时间是否吻合？"
        if "回答" in prompt or "answer" in lowered:
            return "无法从原文确定"
        if "矛盾" in prompt or "contradict" in lowered:
            return "无矛盾"
        return "无法确定"


def _run(workers: int) -> tuple[Any, RecordingLLM]:
    llm = RecordingLLM()
    result = cove_verify(
        "言万在集市上与拉法交手",
        llm=llm,
        evidence=(),
        max_questions=4,
        workers=workers,
    )
    return result, llm


# ══ concurrency ═══════════════════════════════════════════════════════
def test_questions_are_answered_on_more_than_one_thread() -> None:
    _result, llm = _run(workers=4)

    distinct = set(llm.threads)
    assert len(distinct) > 1, (
        "every model call ran on one thread: the independent answers are still serial. "
        f"threads={llm.threads}"
    )


def test_a_single_worker_stays_on_the_calling_thread() -> None:
    """`workers <= 1` must not create a pool; some callers rely on that."""
    _result, llm = _run(workers=1)
    assert len(set(llm.threads)) == 1, f"a pool was created for workers=1: {llm.threads}"
    assert llm.threads[0] == threading.get_ident()


# ══ order is preserved ════════════════════════════════════════════════
def test_answers_are_returned_in_question_order() -> None:
    parallel, _ = _run(workers=4)
    serial, _ = _run(workers=1)
    assert [q.qid for q in parallel.questions] == [q.qid for q in serial.questions]
    assert [q.text for q in parallel.questions] == [q.text for q in serial.questions]


def test_the_result_is_the_same_either_way() -> None:
    parallel, _ = _run(workers=4)
    serial, _ = _run(workers=1)
    assert parallel.status == serial.status
    assert parallel.revised == serial.revised
    assert len(parallel.questions) == len(serial.questions)


# ══ the serial default must not change ════════════════════════════════
def test_the_workers_parameter_defaults_to_serial() -> None:
    import inspect

    signature = inspect.signature(cove_verify)
    assert "workers" in signature.parameters, "cove_verify cannot be asked to run concurrently"
    assert signature.parameters["workers"].default == 1, (
        "the default must stay serial so existing callers keep identical behaviour"
    )
