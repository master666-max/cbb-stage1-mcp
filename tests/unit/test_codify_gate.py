"""D8: a codify gate decides whether to run the cue pass at all, and says why.

Measured gap
------------
The audit's D8: the trigger is a pass-through field — whatever upstream supplies is what gets recorded —
and "the paper's core codify gate does not exist". `cues/integration.py` runs the state machine over every
candidate unconditionally; nothing decides whether a chapter is worth processing.

The pattern is in the library, in a working project:
`库/code/B-项目案例/B06-AI-Reader-V2/backend/src/services/world_structure_agent.py:579-619`
`_should_trigger_llm` — five named conditions, each with a threshold:

    chapter_num <= 5                                  # bootstrap
    any region_division signal                        # event-driven
    layer_transition to a not-yet-existing layer      # structural novelty
    >= 2 new macro geography locations                # volume threshold
    chapter_num % 20 == 0                             # periodic safety net

That shape is better than the paper's prose: bootstrap + event + novelty + volume + periodic covers
"worth doing now" without a single heuristic score.

One improvement over the source, and it is the point of this batch: the source returns `bool`. That means
a run cannot explain afterwards **why** a chapter triggered, which is exactly the audit's complaint that
the F-T-P segments are not consumed by any decision. So this returns *which* condition fired.

Pinned here
-----------
* each of the five conditions fires on its own, and each has a boundary test (chapter 5 yes, 6 no);
* the first matching condition wins, in the source's order, so the reason is stable;
* when nothing matches the decision is `should_run=False` with a readable reason — a silent skip is the
  failure mode this exists to prevent;
* the decision is serialisable, because it has to be written to the gate table.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cbb_stage1.cues.codify import (
    BOOTSTRAP_CHAPTERS,
    PERIODIC_INTERVAL,
    CodifyDecision,
    codify_decision,
)

BOOK = "t"


def _unit(order: int, chapter: int, volume: int = 1) -> dict[str, Any]:
    return {
        "unit_id": f"u:t:v{volume:02d}:c{chapter:03d}",
        "book_id": BOOK,
        "volume": volume,
        "chapter": chapter,
        "order": order,
        "narrator_reliability": "reliable",
    }


def _cue(subject: str, kind: str = "object") -> dict[str, Any]:
    return {"kind": kind, "subject": {"entity_id": None, "literal": subject}, "setup": {}, "status": "candidate"}


def _decide(chapter: int, **kwargs: Any) -> CodifyDecision:
    return codify_decision(_unit(chapter, chapter), **kwargs)


# ══ the constants are the source's ══════════════════════════════════
def test_the_bootstrap_window_is_the_sources() -> None:
    assert BOOTSTRAP_CHAPTERS == 5


def test_the_periodic_interval_is_the_sources() -> None:
    assert PERIODIC_INTERVAL == 20


# ══ condition 1: bootstrap ══════════════════════════════════════════
def test_an_early_chapter_always_runs() -> None:
    assert _decide(3, new_cues=()).should_run is True


def test_the_bootstrap_boundary() -> None:
    assert _decide(5, new_cues=()).condition == "bootstrap"
    assert _decide(6, new_cues=()).condition != "bootstrap"


# ══ condition 2: an explicit trigger signal ═════════════════════════
def test_a_trigger_signal_runs_the_pass() -> None:
    decision = _decide(9, new_cues=(), signals=({"type": "trigger"},))
    assert decision.should_run is True
    assert decision.condition == "signal"


# ══ condition 3: a structurally new kind ════════════════════════════
def test_a_cue_kind_never_seen_before_runs_the_pass() -> None:
    decision = _decide(
        9, new_cues=(_cue("令牌", kind="prophecy"),), known_kinds={"object", "line"}
    )
    assert decision.should_run is True
    assert decision.condition == "new_kind"


def test_a_known_kind_alone_does_not() -> None:
    decision = _decide(
        9, new_cues=(_cue("令牌", kind="object"),), known_kinds={"object", "line"}
    )
    assert decision.condition != "new_kind"


# ══ condition 4: volume ═════════════════════════════════════════════
def test_two_new_cues_run_the_pass() -> None:
    decision = _decide(9, new_cues=(_cue("令牌"), _cue("石碑")))
    assert decision.should_run is True
    assert decision.condition == "volume"


def test_one_new_cue_alone_does_not() -> None:
    decision = _decide(9, new_cues=(_cue("令牌"),))
    assert decision.should_run is False


# ══ condition 5: periodic ═══════════════════════════════════════════
def test_the_periodic_chapter_runs() -> None:
    decision = _decide(20, new_cues=())
    assert decision.should_run is True
    assert decision.condition == "periodic"


def test_a_non_periodic_quiet_chapter_does_not_run() -> None:
    decision = _decide(21, new_cues=())
    assert decision.should_run is False


# ══ the reason is stable and readable ═══════════════════════════════
def test_the_first_matching_condition_wins() -> None:
    """Both bootstrap and volume could match; bootstrap is earlier, so it is the reported reason."""
    decision = _decide(2, new_cues=(_cue("甲"), _cue("乙")))
    assert decision.condition == "bootstrap"


def test_the_decision_is_serialisable() -> None:
    import json

    decision = _decide(9, new_cues=(_cue("令牌"), _cue("石碑")))
    payload = decision.as_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["should_run"] is True
    assert payload["condition"] == "volume"
    assert payload["unit_id"]


def test_a_skip_carries_a_reason() -> None:
    decision = _decide(21, new_cues=())
    payload = decision.as_dict()
    assert payload["should_run"] is False
    assert payload["reason"], "a skipped chapter with no reason is indistinguishable from a bug"


def test_the_condition_set_is_closed() -> None:
    from cbb_stage1.cues.codify import CODIFY_CONDITIONS

    assert set(CODIFY_CONDITIONS) == {
        "bootstrap",
        "signal",
        "new_kind",
        "volume",
        "periodic",
    }
