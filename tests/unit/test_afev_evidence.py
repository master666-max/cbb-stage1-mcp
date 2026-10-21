"""AFEV must receive evidence, or it can only answer `unverifiable`.

A real run (spk7) issued 7 fact-verification prompts whose evidence field was empty:

    事实：X
    证据：
    (nothing)

The verifier answered `unverifiable` for all of them — correctly, given the input — the
aggregate became `insufficient_evidence`, and the reasoner re-decomposed the same claim
for another round. A stage that burns calls while proving nothing looks exactly like a
stage that is simply slow, which is why this needs a test.
"""

from __future__ import annotations

from cbb_stage1.claims.integration import _statement_quotes
from cbb_stage1.models import Assertion, Evidence, Speaker


def _statement(*quotes: str) -> Assertion:
    return Assertion(
        book_id="t",
        predicate="敌对",
        library="关系",
        roles={"主体": "贝塔", "值": {"literal": "阿尔法"}},
        claim_layer="statement",
        speaker=Speaker(kind="entity", entity_id="e:t:person:1", name="阿尔法"),
        evidence=[
            Evidence(
                unit_id="u:t:v01:c001",
                start=index * 10,
                end=index * 10 + len(quote),
                quote=quote,
                occurrence=1,
                mode="dialogue",
            )
            for index, quote in enumerate(quotes)
        ],
    )


def test_statement_quotes_are_the_evidence() -> None:
    statement = _statement("「好。把它毁掉吧。」", "「阿尔法小姐，退后！」")
    assert _statement_quotes(statement) == ["「好。把它毁掉吧。」", "「阿尔法小姐，退后！」"]


def test_statement_quotes_are_deduplicated() -> None:
    statement = _statement("「好。」", "「好。」", "「退后！」")
    assert _statement_quotes(statement) == ["「好。」", "「退后！」"]


def test_statement_quotes_are_capped() -> None:
    statement = _statement(*[f"「第{index}句。」" for index in range(20)])
    assert len(_statement_quotes(statement)) == 8


def test_a_statement_without_quotes_yields_an_empty_list() -> None:
    """Empty is honest here — the caller then passes nothing and AFEV says unverifiable."""
    assert _statement_quotes(_statement()) == []


def test_afev_prompt_includes_the_evidence_it_is_given() -> None:
    """The prompt must not end at "证据：" with nothing after it."""
    from cbb_stage1.orchestrator.llm_bridge import ReasoningLLM

    captured: list[str] = []

    class _LLM:
        def generate(self, prompt: str, **kwargs):  # noqa: ANN003
            captured.append(prompt)
            return '{"verdict": "true", "rationale": "ok"}'

    reasoning = ReasoningLLM(_LLM())
    reasoning.verify("阿尔法与小兽敌对", ["「好。把它毁掉吧。」", "「阿尔法小姐，退后！」"])

    assert captured, "verify must call the model"
    prompt = captured[0]
    assert "证据：" in prompt
    # Every supplied quote must appear after the 证据 marker.
    tail = prompt.split("证据：", 1)[1]
    assert "「好。把它毁掉吧。」" in tail
    assert "「阿尔法小姐，退后！」" in tail
