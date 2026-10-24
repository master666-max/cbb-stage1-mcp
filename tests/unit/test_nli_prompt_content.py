"""The NLI verifier must see what the events say, not just their ids.

Measured degradation (D11)
--------------------------
`temporal/nli.py:96` calls the verifier as `impl(left_id, right_id, question)`, and
`question_for(kind, left_id, right_id)` fills a template with the two ids:

    "ev:3f2a before ev:91bc?"

So the model is asked to decide a temporal relation between two opaque strings. It cannot know whether
`ev:3f2a` happens before `ev:91bc`, because nothing tells it what either event is. The paper's premise
is "document context + hypothesis"; both are missing.

The consequence is worse than "no signal": any model that answers such a question is guessing, and a
guess is recorded as a `yes`/`no` verdict with a verifier name attached — a fabricated conflict with
paperwork. That is why this is graded a degradation and not a substitute.

What is pinned
--------------
* the prompt carries the events' text (and the chapter/unit they came from) when the caller has it;
* ids stay in the prompt too, so a verdict can still be traced back;
* when no text is available the question says so, rather than looking answerable.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.temporal.nli import DEFAULT_QUESTION_BY_KIND, question_for


# ══ the question must carry the event text ═══════════════════════════
def test_the_question_includes_the_event_text_when_given() -> None:
    question = question_for(
        "before",
        "ev:aaa",
        "ev:bbb",
        left_text="言万推开木门走进屋内",
        right_text="拉法在院中拔剑",
    )
    assert "言万推开木门走进屋内" in question, (
        "the left event's text is missing; a model asked only about ids cannot judge the relation"
    )
    assert "拉法在院中拔剑" in question, "the right event's text is missing"


def test_the_ids_are_still_present() -> None:
    """A verdict has to be traceable back to the events it judged."""
    question = question_for(
        "before", "ev:aaa", "ev:bbb", left_text="甲事", right_text="乙事"
    )
    assert "ev:aaa" in question and "ev:bbb" in question


def test_the_question_says_when_no_text_is_available() -> None:
    """An id-only question must look unanswerable, not answerable."""
    question = question_for("before", "ev:aaa", "ev:bbb")
    lowered = question.lower()
    assert "未提供" in question or "无法判断" in question or "unknown" in lowered, (
        f"an id-only question does not admit it lacks the event content: {question!r}"
    )


def test_every_kind_keeps_its_template() -> None:
    """The relation wording must still come from the fixed table (prompts stay comparable)."""
    for kind in DEFAULT_QUESTION_BY_KIND:
        question = question_for(kind, "ev:1", "ev:2", left_text="甲", right_text="乙")
        assert "甲" in question and "乙" in question, f"{kind} dropped the event text"


def test_an_over_long_text_is_truncated() -> None:
    """Event summaries can be long; the prompt must stay bounded."""
    long_text = "很" * 5000
    question = question_for(
        "before", "ev:1", "ev:2", left_text=long_text, right_text="乙"
    )
    assert len(question) < 2000, f"the question is {len(question)} characters"


# ══ the call site must pass the text through ════════════════════════
def test_nli_pairwise_passes_the_event_text() -> None:
    import inspect

    from cbb_stage1.temporal import nli

    source = inspect.getsource(nli.nli_pairwise)
    assert "text" in source, (
        "nli_pairwise does not forward event text to the verifier, so the prompt can only carry ids"
    )


def test_the_verifier_receives_text_not_only_ids() -> None:
    """Drive the real path: what the injected callable actually sees."""
    from cbb_stage1.temporal.nli import nli_pairwise

    seen: list[tuple[Any, ...]] = []

    def impl(left: Any, right: Any, question: str) -> str:
        seen.append((left, right, question))
        return "yes"

    class Node:
        def __init__(self, id_: str, text: str) -> None:
            self.id = id_
            self.text = text

    nli_pairwise(Node("ev:aaa", "甲事件正文"), Node("ev:bbb", "乙事件正文"), "谁先？", impl=impl)

    assert seen, "the verifier was never called"
    _left, _right, question = seen[0]
    assert "甲事件正文" in question, (
        "the verifier's prompt still contains no event text; it is being asked to guess"
    )
