"""F4 steps 2–4: one gate, every exit, and aggregate text withheld whole.

Step 1 answered *when* a record may be shown. This is the gate that acts on the answer, plus the two decisions
the precedents force:

**Every exit must be wired, and `surface` must be mandatory.** Discourse's own bug report (Meta, 2026-09) is a
chat push-notification excerpt that "bypasses spoiler-alert's `:reduce_excerpt` completely" and printed the
spoiler verbatim. Making `surface` a required keyword means a new exit cannot be added without naming itself,
so the omission is a visible gap rather than a silent leak.

**Aggregate text is withheld whole, never trimmed.** Thulr's author names this as the unsolved part: a summary
can disclose something that no single sentence does. Cutting half a summary would leave a misleading fragment,
so the entire text is withheld and `reveal_at` tells the client when it becomes available.

Surface policies (deliberate, each with the reason):

    card      -> keep the record, mark the whole record withheld
    index     -> omit the row (a list is a place spoilers leak by being *counted*)
    alias     -> omit the entry (an alias reveals an identity, per Thulr's "identities not yet revealed")
    summary   -> keep the row, withhold the text
    search    -> omit (Without Spoilers: "characters you haven't met don't even appear in search")
    prompt    -> omit (the model must not be able to quote what the reader has not read)
    timeline  -> omit

Unknown `as_of` fails **closed**: a record we cannot date is withheld. Showing something too early cannot be
undone; withholding one paragraph can.
"""

from __future__ import annotations

from cbb_stage1.disclose.as_of import AsOf
from cbb_stage1.disclose.gate import (
    SURFACES,
    Disclosed,
    disclose,
    disclose_rows,
    horizon_from_unit,
)

ORDER = {"u:t:v01:c001": 1001, "u:t:v01:c002": 1002, "u:t:v01:c003": 1003}

CARD = {
    "entity_id": "e:1",
    "canonical_name": "言万",
    "first_unit": "u:t:v01:c003",
    "summary": "他是主角。",
}


# ══ the surface set is closed and complete ══════════════════════════
def test_every_exit_this_repository_has_is_named() -> None:
    """The seven exits: card, index, alias, summary, search, prompt, timeline."""
    assert set(SURFACES) == {
        "card",
        "index",
        "alias",
        "summary",
        "search",
        "prompt",
        "timeline",
    }


def test_surface_is_mandatory() -> None:
    """Discourse's bug was an exit nobody wired; a required keyword makes that visible."""
    import inspect

    parameters = inspect.signature(disclose).parameters
    surface = parameters.get("surface")
    assert surface is not None, "disclose() has no surface parameter"
    assert surface.default is inspect.Parameter.empty, (
        "surface has a default, so a new exit can silently reuse another one's policy"
    )


def test_an_unknown_surface_is_rejected() -> None:
    """Failing closed on an unknown exit beats guessing a policy for it."""
    import pytest

    with pytest.raises(ValueError, match="surface"):
        disclose(CARD, 9999, surface="unknown-exit")


# ══ within the horizon: nothing is touched ══════════════════════════
def test_a_record_inside_the_horizon_is_returned_unchanged() -> None:
    result = disclose(CARD, 1003, surface="card", unit_order=ORDER)
    assert result.withheld is False
    assert result.record["canonical_name"] == "言万"
    assert result.record["summary"] == "他是主角。"


def test_a_record_exactly_at_the_horizon_is_visible() -> None:
    """The reader has finished that chapter, so its content is theirs."""
    assert disclose(CARD, 1003, surface="card", unit_order=ORDER).withheld is False


# ══ beyond the horizon: per-surface policy ══════════════════════════
def test_a_card_beyond_the_horizon_is_withheld_but_kept() -> None:
    result = disclose(CARD, 1002, surface="card", unit_order=ORDER)
    assert result.withheld is True
    assert result.record is not None, "a card must stay so the client can render a placeholder"
    assert result.record["withheld"] is True
    assert result.record["reveal_at"] == 1003


def test_an_index_row_beyond_the_horizon_is_omitted() -> None:
    result = disclose(CARD, 1002, surface="index", unit_order=ORDER)
    assert result.withheld is True
    assert result.record is None, "an index still lists the row, which reveals its existence"


def test_an_alias_beyond_the_horizon_is_omitted() -> None:
    result = disclose(CARD, 1002, surface="alias", unit_order=ORDER)
    assert result.record is None


def test_a_search_hit_beyond_the_horizon_is_omitted() -> None:
    """Without Spoilers: characters you have not met do not appear in search at all."""
    result = disclose(CARD, 1002, surface="search", unit_order=ORDER)
    assert result.record is None


def test_a_summary_beyond_the_horizon_withholds_the_text_whole() -> None:
    row = {
        "summary_id": "s:1",
        "text": "言万最终成为宗主。",
        "evidence": [{"unit_id": "u:t:v01:c003"}],
    }
    result = disclose(row, 1002, surface="summary", unit_order=ORDER)
    assert result.withheld is True
    assert result.record["text"] is None, "a half summary would mislead; withhold the whole text"
    assert result.record["withheld"] is True
    assert result.record["reveal_at"] == 1003


def test_a_prompt_context_beyond_the_horizon_is_omitted() -> None:
    """The least obvious leak: whatever is handed to a model can be quoted back."""
    assert disclose(CARD, 1002, surface="prompt", unit_order=ORDER).record is None


def test_a_timeline_entry_beyond_the_horizon_is_omitted() -> None:
    assert disclose(CARD, 1002, surface="timeline", unit_order=ORDER).record is None


# ══ unknown fails closed ════════════════════════════════════════════
def test_an_undated_record_is_withheld_everywhere() -> None:
    undated = {"entity_id": "e:2", "canonical_name": "某甲"}
    for surface in SURFACES:
        result = disclose(undated, 9999, surface=surface, unit_order=ORDER)
        assert result.withheld is True, f"{surface} released a record with no date"
        assert result.reason, "a withheld record must say why"


def test_the_withheld_reason_is_recorded() -> None:
    result = disclose(CARD, 1002, surface="card", unit_order=ORDER)
    assert result.reason
    assert "1003" in result.reason or "horizon" in result.reason.lower()


# ══ list helpers ════════════════════════════════════════════════════
def test_disclose_rows_drops_what_the_surface_omits() -> None:
    rows = [
        {"entity_id": "e:1", "first_unit": "u:t:v01:c001"},
        {"entity_id": "e:2", "first_unit": "u:t:v01:c003"},
    ]
    kept = disclose_rows(rows, 1002, surface="index", unit_order=ORDER)
    assert [row["entity_id"] for row in kept] == ["e:1"]


def test_disclose_rows_marks_what_the_surface_keeps() -> None:
    rows = [
        {"entity_id": "e:1", "first_unit": "u:t:v01:c001"},
        {"entity_id": "e:2", "first_unit": "u:t:v01:c003"},
    ]
    kept = disclose_rows(rows, 1002, surface="card", unit_order=ORDER)
    assert len(kept) == 2
    assert kept[1]["withheld"] is True


def test_disclose_rows_never_reorders() -> None:
    rows = [{"entity_id": f"e:{index}", "first_unit": "u:t:v01:c001"} for index in range(5)]
    kept = disclose_rows(rows, 9999, surface="index", unit_order=ORDER)
    assert [row["entity_id"] for row in kept] == [f"e:{index}" for index in range(5)]


def test_accounting_says_how_many_were_withheld() -> None:
    rows = [
        {"entity_id": "e:1", "first_unit": "u:t:v01:c001"},
        {"entity_id": "e:2", "first_unit": "u:t:v01:c003"},
    ]
    report: dict = {}
    disclose_rows(rows, 1002, surface="index", unit_order=ORDER, report=report)
    assert report["withheld"] == 1
    assert report["released"] == 1


# ══ horizon arithmetic ══════════════════════════════════════════════
def test_a_horizon_can_be_given_as_a_unit_id() -> None:
    assert horizon_from_unit("u:t:v01:c002") == 1002
    assert horizon_from_unit("u:t:v01:c002", unit_order=ORDER) == 1002


def test_an_unresolvable_horizon_is_an_error_not_a_default() -> None:
    """A typo in the reader's position must not silently release the whole book."""
    import pytest

    with pytest.raises(ValueError, match="horizon"):
        horizon_from_unit("nonsense")


def test_a_released_result_reports_its_position() -> None:
    result = disclose(CARD, 1003, surface="card", unit_order=ORDER)
    assert isinstance(result.as_of, AsOf)
    assert result.as_of.order == 1003


def test_the_result_is_serialisable() -> None:
    import json

    payload = disclose(CARD, 1002, surface="card", unit_order=ORDER).as_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["withheld"] is True


def test_disclosed_is_the_documented_type() -> None:
    assert isinstance(disclose(CARD, 1003, surface="card", unit_order=ORDER), Disclosed)
