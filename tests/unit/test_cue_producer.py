"""Something must actually produce cues, or one of the six libraries is always empty.

Measured defect
---------------
`engines/ai_reader/openie.py:521` and `normalize/facts.py:674` both write `cues=[]` literally, and
`ChapterFact` has no cue field at all. So:

* the extraction prompt cannot ask for foreshadowing, because the schema has nowhere to put it;
* the converter has nothing to map;
* `knowledge/cues.jsonl` is empty in every box;
* the cue state machine (D7) and the early-payoff detection (D8) only ever ran on data injected by
  tests into `resolve/cues.jsonl`.

That is the whole 埋线 library — one of the six — standing on nothing.

What this pins
--------------
A cue that the extractor reports must travel: `ChapterFact.cues` → `EngineOutput.cues` → the box.
And a cue whose setup quote cannot be anchored in the unit text must be **dropped**, not given an
invented span: an unanchored cue is indistinguishable from a fabricated one.
"""

from __future__ import annotations

from cbb_stage1.engines.ai_reader.schema import ChapterFact
from cbb_stage1.normalize.facts import chapter_fact_to_engine_output

UNIT = "u:tst:v01:c001"
TEXT = (
    "老人从怀里摸出一枚黑色令牌，令牌背面刻着一个‘柒’字。"
    "他把令牌收好，什么也没说。"
    "第七章末尾，那枚黑色令牌在火光中裂成两半。"
)


def _chapter(cues: list[dict]) -> dict:
    return {
        "chapter_id": 1,
        "novel_id": "tst",
        "characters": [],
        "relationships": [],
        "locations": [],
        "spatial_relationships": [],
        "item_events": [],
        "org_events": [],
        "events": [],
        "new_concepts": [],
        "world_declarations": [],
        "cues": cues,
    }


def _cue(**overrides) -> dict:
    cue = {
        "kind": "object",
        "subject": "黑色令牌",
        "setup": "老人从怀里摸出一枚黑色令牌",
        "expected": "令牌在后面的章节点会再次出现并发挥作用",
        "payoff": "",
        "confidence": 0.7,
        "evidence": "老人从怀里摸出一枚黑色令牌",
    }
    cue.update(overrides)
    return cue


def _convert(chapter: dict):
    fact = ChapterFact.model_validate(chapter)
    return fact, chapter_fact_to_engine_output(
        fact, unit_id=UNIT, text=TEXT, book_id="tst"
    )


# ══ the schema must be able to express a cue ══════════════════════════
def test_chapter_fact_accepts_a_cue() -> None:
    fact = ChapterFact.model_validate(_chapter([_cue()]))
    assert len(fact.cues) == 1
    assert fact.cues[0].subject == "黑色令牌"


def test_a_cue_requires_a_subject_and_a_setup() -> None:
    import pytest

    with pytest.raises(Exception):
        ChapterFact.model_validate(_chapter([_cue(subject="")]))
    with pytest.raises(Exception):
        ChapterFact.model_validate(_chapter([_cue(setup="")]))


# ══ the converter must carry it ═══════════════════════════════════════
def test_the_converter_emits_a_cue() -> None:
    _fact, output = _convert(_chapter([_cue()]))
    assert output.cues, (
        "the converter produced no cue; `ChapterFact.cues` exists but nothing maps it"
    )
    assert output.cues[0].subject.get("literal") == "黑色令牌"


def test_the_cue_setup_is_anchored_in_the_unit() -> None:
    _fact, output = _convert(_chapter([_cue()]))
    evidence = output.cues[0].setup.get("evidence") or []
    assert evidence, "the cue carries no anchored setup evidence"
    first = evidence[0]
    assert first["unit_id"] == UNIT
    expected_start = TEXT.index("老人从怀里摸出一枚黑色令牌")
    assert first["start"] == expected_start
    assert first["end"] == expected_start + len("老人从怀里摸出一枚黑色令牌")


def test_an_unanchored_setup_produces_no_cue() -> None:
    """A quote that is not in the text must be dropped, never given an invented span."""
    _fact, output = _convert(_chapter([_cue(setup="这句话根本不在原文里")]))
    assert output.cues == [], "a cue with an unanchored quote was accepted"


def test_a_paid_off_cue_carries_the_payoff() -> None:
    _fact, output = _convert(
        _chapter([_cue(payoff="那枚黑色令牌在火光中裂成两半")])
    )
    payoff = output.cues[0].payoff or {}
    assert payoff.get("evidence"), "the payoff quote was not anchored"


def test_no_cues_means_no_cues() -> None:
    _fact, output = _convert(_chapter([]))
    assert output.cues == []


# ══ the prompt must ask for it ════════════════════════════════════════
def test_the_extraction_prompt_mentions_cues() -> None:
    from cbb_stage1.engines.ai_reader.prompts import load_system_prompt

    prompt = load_system_prompt()
    assert "cues" in prompt, (
        "the extraction prompt never mentions cues, so the model cannot emit them even though the "
        "schema now has a field"
    )
