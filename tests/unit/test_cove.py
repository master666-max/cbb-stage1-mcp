"""CoVe — Chain-of-Verification (A09, 2024).

The paper's four steps:

1. Generate Baseline Response
2. Plan Verifications — questions that could expose mistakes in the baseline
3. Execute Verifications — answer each, compare with the baseline
4. Generate Final Verified Response — revise if inconsistencies were found

and one sentence that is the whole point:

> The verification questions generated from planning are answered in the second step,
> where **crucially the context given to the LLM prompt only contains the questions, and
> not the original baseline response** and hence cannot repeat those answers directly.

That is what these tests pin. A CoVe implementation whose verification prompt still
carries the claim is just "ask the model the same thing twice" — it will agree with
itself and the check proves nothing.

The `factored` variant (each question answered as its own prompt, no shared context) is
used here: the paper notes it also removes interference *between* answer contexts, not
just from the baseline.
"""

from __future__ import annotations

import pytest

from cbb_stage1.validate import cove

CLAIM = "露娜在湖的浅滩发现一只六条腿的机械小狗，并认定它就是母体。"
EVIDENCE = (
    "湖的浅滩上，有什么小东西在滚动。",
    "它的体型比起别的，更接近小狗……它有六条腿。",
)


class RecordingLLM:
    """Records every prompt so the tests can inspect what the model was told.

    It recognises each of the four prompts by a phrase unique to it. An earlier version
    fell through to a default for the comparison prompt, so the tests passed without ever
    exercising step 3's comparison — green, and proving nothing.
    """

    def __init__(
        self,
        *,
        questions: list[str] | None = None,
        answers: list[str] | None = None,
        contradictions: str = "无矛盾",
        revision: str = "",
    ) -> None:
        self.prompts: list[str] = []
        self._questions = questions or [
            "露娜在哪里发现那只生物？",
            "那只生物有几条腿？",
            "露娜认定它是什么？",
            "那只生物是机械的吗？",
        ]
        self._answers = answers or ["湖的浅滩", "六条", "母体", "是"]
        self._contradictions = contradictions
        self._revision = revision
        self._index = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # 4. revision — its prompt names the inconsistencies
        if "最终版本" in prompt:
            return self._revision or CLAIM
        # 3. comparison — asks which answers are 相矛盾
        if "相矛盾" in prompt:
            return self._contradictions
        # 2. planning — asks for 验证问题
        if "验证问题" in prompt:
            return "\n".join(f"- {q}" for q in self._questions)
        # 1. answering — has 证据 + 问题 but no claim
        if "证据" in prompt and "问题：" in prompt:
            answer = self._answers[self._index % len(self._answers)]
            self._index += 1
            return answer
        raise AssertionError(f"unrecognised prompt: {prompt[:80]!r}")


# ── step 2: planning ─────────────────────────────────────────────────────────
def test_planning_generates_the_configured_number_of_questions() -> None:
    llm = RecordingLLM()
    questions = cove.plan_verifications(CLAIM, llm=llm, max_questions=4)
    assert len(questions) == 4, "the paper plans a list of questions, not one"


def test_planning_sees_the_baseline_because_it_must_target_it() -> None:
    """Step 2 DOES condition on the baseline — that is how it finds the weak spots."""
    llm = RecordingLLM()
    cove.plan_verifications(CLAIM, llm=llm)
    assert CLAIM in llm.prompts[0]


def test_planning_without_a_model_returns_nothing_rather_than_inventing() -> None:
    assert cove.plan_verifications(CLAIM, llm=None) == ()


# ── step 3: the discipline that makes CoVe work ─────────────────────────────
def test_the_answer_prompt_does_not_contain_the_claim() -> None:
    """The paper's central constraint.

    If the baseline response is in the answer prompt, the model re-reads its own claim
    and agrees with it. The check then measures agreement with itself.
    """
    llm = RecordingLLM()
    question = "那只生物有几条腿？"
    cove.answer_independently(question, llm=llm, evidence=EVIDENCE)
    prompt = llm.prompts[0]
    assert CLAIM not in prompt, "the claim must not be in the verification context"
    assert "六条腿的机械小狗" not in prompt, "no fragment of the claim either"
    assert question in prompt


def test_the_answer_prompt_does_not_carry_previous_answers() -> None:
    """Factored: no interference between answer contexts either."""
    llm = RecordingLLM()
    cove.answer_independently("第一问？", llm=llm, evidence=EVIDENCE)
    first_answer_area = llm.prompts[0]
    cove.answer_independently("第二问？", llm=llm, evidence=EVIDENCE)
    second_prompt = llm.prompts[1]
    assert "第一问" not in second_prompt, "questions must not see each other"
    assert first_answer_area != second_prompt


def test_an_answer_without_a_model_is_unsure_not_fabricated() -> None:
    assert cove.answer_independently("问？", llm=None, evidence=EVIDENCE) == ""


# ── step 4: comparison and revision ─────────────────────────────────────────
def test_a_consistent_claim_is_not_revised() -> None:
    llm = RecordingLLM(
        answers=["湖的浅滩", "六条", "母体", "是"],
        revision="",  # the model returns the claim unchanged
    )
    result = cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE)
    assert result.status in {"verified", "revised"}
    assert result.method == "cove_4step"
    assert len(result.questions) == 4


def test_an_answered_question_carries_its_answer() -> None:
    llm = RecordingLLM()
    result = cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE)
    assert all(item.answer for item in result.questions), "every question gets answered"


def test_a_reported_contradiction_triggers_a_revision() -> None:
    """Step 3 finds a contradiction -> step 4 runs and its text is the result."""
    llm = RecordingLLM(
        contradictions="Q2：答案说四条腿，主张说六条腿",
        revision="露娜在湖的浅滩发现一只四条腿的机械小狗。",
    )
    result = cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE)
    assert result.status == "revised"
    assert result.inconsistencies == ("q2",)
    assert "四条腿" in result.revised
    assert result.revised != CLAIM


def test_a_clean_comparison_keeps_the_original_and_spends_no_revision_call() -> None:
    llm = RecordingLLM(contradictions="无矛盾")
    result = cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE)
    assert result.status == "verified"
    assert result.inconsistencies == ()
    assert result.revised == CLAIM
    assert not any("最终版本" in prompt for prompt in llm.prompts), (
        "no contradiction means step 4 is not needed"
    )


def test_the_comparison_does_not_reuse_the_answer_prompts() -> None:
    """The comparison is its own call; it must not be skipped silently."""
    llm = RecordingLLM()
    cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE)
    comparison_prompts = [p for p in llm.prompts if "相矛盾" in p]
    assert len(comparison_prompts) == 1, "step 3's comparison must actually run"


def test_no_model_yields_unavailable_and_no_fabricated_verdict() -> None:
    result = cove.cove_verify(CLAIM, llm=None, evidence=EVIDENCE)
    assert result.status == "unavailable"
    assert result.method == "cove_4step"
    assert result.questions == ()
    assert result.revised == ""


def test_the_method_names_what_it_is() -> None:
    assert cove.METHOD == "cove_4step"


def test_questions_carry_stable_ids() -> None:
    llm = RecordingLLM()
    result = cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE)
    ids = [item.qid for item in result.questions]
    assert len(set(ids)) == len(ids), "question ids must be unique"
    assert all(ids), "and non-empty"


def test_the_result_is_json_serialisable() -> None:
    import json

    llm = RecordingLLM()
    result = cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE)
    payload = json.dumps(result.as_dict(), ensure_ascii=False)
    assert "cove_4step" in payload


# ── cost discipline ────────────────────────────────────────────────────────
def test_the_cost_is_one_planning_call_plus_one_call_per_question() -> None:
    """Documented cost: N+1 (plus a revision call only when it is needed).

    Written as a test because an opt-in analysis whose real cost is not pinned will
    quietly become an N*M surprise later.
    """
    llm = RecordingLLM(
        questions=["问一？", "问二？", "问三？"],
        answers=["湖的浅滩", "六条", "母体"],
        contradictions="无矛盾",
    )
    cove.cove_verify(CLAIM, llm=llm, evidence=EVIDENCE, max_questions=3)
    # 1 planning + 3 answers + 1 comparison = 5; +1 revision only if it found something
    assert len(llm.prompts) == 5
