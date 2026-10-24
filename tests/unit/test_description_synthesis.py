"""Description map-reduce synthesis tests (A18/A24).

Covers:
* no synthesiser -> unchanged ``observations_only`` behaviour (no model call);
* ``MapReduceSynthesizer`` really maps batches and reduces them (call counts,
  prompt shapes) over an injected ``Callable[[str], str]``;
* every synthesised sentence keeps an observation fallback, and
  ``inferred_ratio`` is a real fraction of sentences without any fallback;
* old observations are retained (observation list is never deleted);
* duplicates are removed *before* synthesis;
* no LLM / empty output / LLM failure -> explicit ``observations_only``
  degradation, never a fabricated description;
* a protocol-conforming custom synthesiser no longer raises
  ``NotImplementedError``.


A24 note: `describe` short-circuits a single deduplicated observation (it already is the
description), so the degradation tests below supply two observations to reach the synthesizer.
"""

from __future__ import annotations

import pytest

from cbb_stage1.dedupe.describe import (
    DescribeResult,
    MapReduceSynthesizer,
    Observation,
    SynthesisUnavailable,
    describe,
    split_description_sentences,
)


class FakeLLM:
    """Sync callable fake: returns queued outputs and records prompts."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("FakeLLM received more calls than expected")
        return self.outputs.pop(0)


# ── no synthesiser: unchanged behaviour ────────────────────────


def test_describe_without_synthesizer_is_observations_only() -> None:
    result = describe(
        "e1",
        [
            Observation("他笑了", unit_id="u1"),
            Observation(" 他笑了 ", unit_id="u2"),
            Observation("他是谁", unit_id="u3"),
            "",
        ],
    )
    assert isinstance(result, DescribeResult)
    assert result.mode == "observations_only"
    assert result.text is None
    assert result.inferred_ratio is None
    assert [obs.text for obs in result.observations] == ["他笑了", "他是谁"]
    assert result.sentences == ()
    assert len(result.fragment_dedupe.dropped) == 1


# ── map-reduce ─────────────────────────────────────────────────


def test_map_reduce_maps_batches_then_reduces() -> None:
    batch_outputs = ["韩立笑了。", "他是山村少年。", "他拜入七玄门。", "他怕水。"]
    final = "他从小在山村长大。他其实是一位隐藏的强者。"
    llm = FakeLLM(batch_outputs + [final])
    synthesizer = MapReduceSynthesizer(llm, map_batch_chars=10)

    observations = ["他笑了", "他从小在山村长大", "他拜入七玄门", "他怕水"]
    result = describe("e1", observations, synthesizer=synthesizer)

    assert result.mode == "synthesized"
    assert result.text == final
    # 4 map calls (one batch each at 10 chars) + 1 reduce call.
    assert len(llm.prompts) == 5
    assert synthesizer.stats() == {"map_calls": 4, "reduce_calls": 1, "batches": 4}
    assert any("观察：" in prompt for prompt in llm.prompts[:4])
    assert "片段：" in llm.prompts[-1]

    # Old observations are retained, not replaced by the synthesis.
    assert [obs.text for obs in result.observations] == observations

    # Sentence-level fallback: verbatim sentence supported, fabricated inferred.
    assert [s.text for s in result.sentences] == [
        "他从小在山村长大。",
        "他其实是一位隐藏的强者。",
    ]
    assert result.sentences[0].fallbacks == ("他从小在山村长大",)
    assert result.sentences[0].inferred is False
    assert result.sentences[1].fallbacks == ()
    assert result.sentences[1].inferred is True
    assert result.inferred_ratio == 0.5
    assert result.rollback(0) == (result.observations[1],)
    assert result.rollback(1) == ()
    assert any("1/2 sentence(s) inferred" in note for note in result.notes)


def test_single_batch_skips_reduce() -> None:
    llm = FakeLLM(["他笑了，他并不害怕。"])
    synthesizer = MapReduceSynthesizer(llm, map_batch_chars=10_000)

    result = describe("e1", ["他笑了", "他怕水"], synthesizer=synthesizer)

    assert result.mode == "synthesized"
    assert result.text == "他笑了，他并不害怕。"
    assert synthesizer.stats() == {"map_calls": 1, "reduce_calls": 0, "batches": 1}
    assert len(llm.prompts) == 1


def test_map_reduce_synthesizer_api_guards() -> None:
    with pytest.raises(SynthesisUnavailable):
        MapReduceSynthesizer(None).synthesize("e1", ["他笑了"])
    with pytest.raises(SynthesisUnavailable):
        MapReduceSynthesizer(FakeLLM([])).synthesize("e1", [])
    with pytest.raises(TypeError):
        MapReduceSynthesizer(FakeLLM(["x"])).synthesize(123, ["他笑了"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MapReduceSynthesizer(FakeLLM(["x"]), map_batch_chars=0)


# ── per-sentence fallback / inferred ratio ────────────────────


def test_paraphrased_sentence_is_inferred_not_fabricated_as_supported() -> None:
    class CustomSynthesizer:
        def synthesize(self, entity_id: str, fragments) -> str:
            return "他当时忍不住笑了。他笑了。"

    result = describe("e1", ["他笑了", "他怕水"], synthesizer=CustomSynthesizer())

    assert [s.inferred for s in result.sentences] == [True, False]
    assert result.sentences[1].fallbacks == ("他笑了",)
    assert result.inferred_ratio == 0.5


def test_custom_synthesizer_protocol_accepted_without_not_implemented() -> None:
    class CustomSynthesizer:
        def __init__(self) -> None:
            self.calls = []

        def synthesize(self, entity_id: str, fragments) -> str:
            self.calls.append((entity_id, tuple(fragments)))
            return "他笑了。"

    synthesizer = CustomSynthesizer()
    # Two *distinct* observations: `["他笑了", " 他笑了 "]` dedupes to one, which is short-circuited
    # before the synthesizer and would prove nothing about the protocol.
    result = describe("e1", ["他笑了", "他怕水"], synthesizer=synthesizer)

    assert result.mode == "synthesized"
    assert result.inferred_ratio == 0.0
    assert synthesizer.calls == [("e1", ("他笑了", "他怕水"))]


def test_duplicates_are_removed_before_synthesis() -> None:
    llm = FakeLLM(["他笑了。"])
    synthesizer = MapReduceSynthesizer(llm, map_batch_chars=10_000)

    result = describe("e1", ["他笑了", " 他笑了 ", "他是谁"], synthesizer=synthesizer)

    prompt = llm.prompts[0]
    assert prompt.count("他笑了") == 1  # exact duplicate not sent to the LLM
    assert "他是谁" in prompt
    assert [obs.text for obs in result.observations] == ["他笑了", "他是谁"]


# ── explicit degradation (no fabrication) ────────────────────


def test_no_llm_degrades_to_observations_only() -> None:
    result = describe("e1", ["他笑了", "他怕水"], synthesizer=MapReduceSynthesizer(None))

    assert result.mode == "observations_only"
    assert result.text is None
    assert result.inferred_ratio is None
    assert any("synthesis unavailable" in note for note in result.notes)
    assert [obs.text for obs in result.observations] == ["他笑了", "他怕水"]


def test_llm_failure_degrades_with_reason() -> None:
    class Boom:
        def __call__(self, prompt: str) -> str:
            raise RuntimeError("network down")

    result = describe("e1", ["他笑了", "他怕水"], synthesizer=MapReduceSynthesizer(Boom()))

    assert result.mode == "observations_only"
    assert result.text is None
    assert any("LLM call failed" in note and "network down" in note for note in result.notes)


def test_custom_synthesizer_crash_propagates() -> None:
    class Broken:
        def synthesize(self, entity_id: str, fragments) -> str:
            raise ValueError("bad implementation")

    with pytest.raises(ValueError):
        describe("e1", ["他笑了", "他怕水"], synthesizer=Broken())


def test_empty_output_and_empty_observations_degrade() -> None:
    class EmptySynthesizer:
        def synthesize(self, entity_id: str, fragments) -> str:
            return "   "

    empty_output = describe("e1", ["他笑了", "他怕水"], synthesizer=EmptySynthesizer())
    assert empty_output.mode == "observations_only"
    assert any("returned no text" in note for note in empty_output.notes)

    # A map-reduce LLM answering with whitespace is an unavailable LLM call.
    unavailable = describe("e1", ["他笑了", "他怕水"], synthesizer=MapReduceSynthesizer(FakeLLM(["  "])))
    assert unavailable.mode == "observations_only"
    assert any("empty description" in note for note in unavailable.notes)

    no_observations = describe("e1", [], synthesizer=MapReduceSynthesizer(FakeLLM([])))
    assert no_observations.mode == "observations_only"
    assert no_observations.text is None
    assert any("no observations" in note for note in no_observations.notes)


# ── helpers ───────────────────────────────────────────────────


def test_split_description_sentences_handles_mixed_punctuation() -> None:
    assert split_description_sentences("第一句。第二句！第三句？") == [
        "第一句。",
        "第二句！",
        "第三句？",
    ]
    assert split_description_sentences("") == []
    assert split_description_sentences("没有句号") == ["没有句号"]
