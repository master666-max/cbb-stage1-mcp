"""Quote verification for speaker attribution (AVeriTeC, H04).

The behaviour that matters:

  * a speaker outside the roster yields `unsure`, never `against` — the roster is not
    authoritative for cross-unit attribution, and treating a miss as refutation deletes
    correct data (measured when the removed cast check did exactly that);
  * no verifier yields `unsure`, never a fabricated `support`;
  * a verdict is a recorded judgement — it never deletes a speaker or an assertion.
"""

from __future__ import annotations

from cbb_stage1.validate.speaker_verify import (
    SPEAKER_VERDICTS,
    build_speaker_prompt,
    parse_speaker_verdict,
    verify_speaker_quote,
)

QUOTE = "「这家伙…… 就是母体……──这个终末的源头。」"
CONTEXT = "「嗯？怎么了言万同学。你怕狗吗？」" + QUOTE


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:  # noqa: ANN001
        self.prompts.append(prompt)
        return self.answer


# ── short-circuits ───────────────────────────────────────────────────────────
def test_a_missing_speaker_is_unsure_without_calling_the_model() -> None:
    llm = FakeLLM("")
    verdict = verify_speaker_quote(quote=QUOTE, speaker_name="", cast={"言万"}, llm=llm)
    assert verdict.verdict == "unsure"
    assert verdict.method == "no_speaker"
    assert llm.prompts == [], "must not spend a call when there is nothing to check"


def test_a_cast_miss_is_unsure_not_against() -> None:
    """The roster does not cover cross-unit speakers, so a miss is not refutation."""
    llm = FakeLLM("")
    verdict = verify_speaker_quote(
        quote=QUOTE, speaker_name="第1章才出现的角色", cast={"言万", "露娜"}, llm=llm
    )
    assert verdict.verdict == "unsure"
    assert verdict.method == "cast_miss"
    assert llm.prompts == [], "a roster miss must not cost a call"


def test_no_verifier_is_unsure_not_support() -> None:
    verdict = verify_speaker_quote(quote=QUOTE, speaker_name="言万", cast={"言万"}, llm=None)
    assert verdict.verdict == "unsure"
    assert verdict.method == "no_verifier"


# ── model-backed path ────────────────────────────────────────────────────────
def test_a_supporting_answer_is_recorded() -> None:
    llm = FakeLLM('{"verdict": "support", "rationale": "上一句点名问了言万"}')
    verdict = verify_speaker_quote(
        quote=QUOTE,
        speaker_name="言万",
        context=CONTEXT,
        cast={"言万", "露娜"},
        llm=llm,
    )
    assert verdict.verdict == "support"
    assert "言万" in llm.prompts[0], "the prompt must name the speaker"


def test_a_refuting_answer_is_recorded_but_changes_nothing() -> None:
    llm = FakeLLM('{"verdict": "against", "rationale": "是露娜说的"}')
    verdict = verify_speaker_quote(
        quote=QUOTE, speaker_name="言万", context=CONTEXT, cast={"言万", "露娜"}, llm=llm
    )
    assert verdict.verdict == "against"
    # The verdict is a record; the caller keeps the speaker. Nothing to assert here
    # beyond the verdict itself — the point is that this function returns, not deletes.


def test_an_unparseable_answer_becomes_unsure() -> None:
    llm = FakeLLM("我想了想，大概是吧。")
    verdict = verify_speaker_quote(
        quote=QUOTE, speaker_name="言万", context=CONTEXT, cast={"言万"}, llm=llm
    )
    assert verdict.verdict == "unsure"
    assert verdict.method == "llm"


def test_a_failing_verifier_becomes_unsure_not_an_exception() -> None:
    class Boom:
        def generate(self, prompt: str) -> str:  # noqa: ANN001
            raise RuntimeError("endpoint down")

    verdict = verify_speaker_quote(
        quote=QUOTE, speaker_name="言万", context=CONTEXT, cast={"言万"}, llm=Boom()
    )
    assert verdict.verdict == "unsure"
    assert verdict.method == "llm_error"


def test_the_same_quote_and_speaker_are_only_checked_once() -> None:
    llm = FakeLLM('{"verdict": "support", "rationale": "ok"}')
    cache: dict = {}
    for _ in range(3):
        verify_speaker_quote(
            quote=QUOTE, speaker_name="言万", context=CONTEXT, cast={"言万"},
            llm=llm, cache=cache,
        )
    assert len(llm.prompts) == 1, "the same (quote, speaker) must not be re-asked"
    assert len(cache) == 1


# ── parsing and prompt ───────────────────────────────────────────────────────
def test_a_bare_word_is_accepted() -> None:
    assert parse_speaker_verdict("support")[0] == "support"
    assert parse_speaker_verdict("AGAINST")[0] == "against"
    assert parse_speaker_verdict("unsure")[0] == "unsure"


def test_an_unknown_verdict_word_becomes_unsure() -> None:
    assert parse_speaker_verdict('{"verdict": "probably"}')[0] == "unsure"


def test_the_verdict_set_matches_the_box_enum() -> None:
    assert set(SPEAKER_VERDICTS) == {"support", "against", "unsure"}


def test_the_prompt_states_the_unsure_rule() -> None:
    prompt = build_speaker_prompt(quote=QUOTE, speaker="言万", context=CONTEXT)
    assert "unsure" in prompt
    assert "言万" in prompt
    assert QUOTE in prompt


def test_same_model_is_carried_so_independence_can_be_reported() -> None:
    llm = FakeLLM('{"verdict": "support", "rationale": "ok"}')
    verdict = verify_speaker_quote(
        quote=QUOTE, speaker_name="言万", context=CONTEXT, cast={"言万"},
        llm=llm, same_model=True,
    )
    assert verdict.same_model is True
    assert verdict.as_dict()["speaker_check_same_model"] is True
