"""One observation needs no summarisation: it is already the description.

Measured gap (A24)
------------------
The source short-circuits at a single description (`describe` :409-410 returns the text verbatim).
`dedupe/describe.py:550-551` calls `synthesizer.synthesize(...)` unconditionally once a synthesizer is
wired, so a character mentioned in one scene costs a model call to "summarise" one sentence.

Why this is not a micro-optimisation
------------------------------------
Two reasons, both measurable in a real run:

* cost — most entities in a novel appear in exactly one unit, so this is the common case, not the edge;
* fidelity — a model asked to summarise a single sentence can paraphrase it, and a paraphrased
  description is no longer quotable. The source's short-circuit preserves the original wording, which is
  what a card should show.

The short-circuit must apply only when there is exactly one distinct observation. Two observations that
happen to be identical are still one, because dedupe already merged them — so the check is on the
deduplicated list, not the raw one.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cbb_stage1.dedupe.describe import Observation, describe


class RecordingSynthesizer:
    """Records every call; returns a recognisable summary."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def synthesize(self, entity_id: str, texts: Sequence[str]) -> str:
        self.calls.append((entity_id, tuple(texts)))
        return f"摘要（{len(texts)} 条）"


def _observation(text: str, unit: str, quote: str | None = None) -> Observation:
    return Observation(text=text, unit_id=unit, source=unit)


def _describe(observations: list[Observation]) -> Any:
    synthesizer = RecordingSynthesizer()
    result = describe(
        "e:t:person:aaaaaaaaaaaa",
        observations,
        synthesizer=synthesizer,
    )
    return result, synthesizer


# ══ one observation is returned verbatim ════════════════════════════
def test_a_single_observation_is_not_summarised() -> None:
    result, synthesizer = _describe([_observation("言万独自站在集市中央", "u:t:v01:c001")])
    assert not synthesizer.calls, (
        "one observation was sent to the model; the source short-circuits because the text already is "
        "the description"
    )


def test_the_single_observation_becomes_the_description() -> None:
    result, _ = _describe([_observation("言万独自站在集市中央", "u:t:v01:c001")])
    assert result.text == "言万独自站在集市中央", (
        f"the description is {result.text!r}, not the observation text"
    )


def test_the_short_circuit_is_recorded() -> None:
    """A reader must be able to tell a verbatim description from a synthesised one."""
    result, _ = _describe([_observation("言万独自站在集市中央", "u:t:v01:c001")])
    assert any("short" in note.lower() or "single" in note.lower() for note in result.notes), (
        f"the short-circuit left no trace: {result.notes}"
    )


# ══ duplicates of one observation are still one ═════════════════════
def test_an_observation_repeated_in_one_unit_is_still_one() -> None:
    """Dedupe merges them first, so the check is on the deduplicated list."""
    observations = [
        _observation("言万独自站在集市中央", "u:t:v01:c001"),
        _observation("言万独自站在集市中央", "u:t:v01:c001"),
    ]
    _result, synthesizer = _describe(observations)
    assert not synthesizer.calls, (
        "the same observation twice in one unit was treated as corroboration and summarised"
    )


# ══ two observations do need synthesis ══════════════════════════════
def test_two_observations_are_summarised() -> None:
    _result, synthesizer = _describe(
        [
            _observation("言万独自站在集市中央", "u:t:v01:c001"),
            _observation("言万在渡口遇见一名老者", "u:t:v01:c002"),
        ]
    )
    assert len(synthesizer.calls) == 1, "two observations did not reach the synthesizer"


def test_no_synthesizer_still_returns_observations() -> None:
    result = describe(
        "e:t:person:aaaaaaaaaaaa",
        [_observation("言万独自站在集市中央", "u:t:v01:c001")],
        synthesizer=None,
    )
    assert result.text is None or result.text == ""
    assert result.observations
