from cbb_stage1.temporal.extract import (
    annotate_event_times,
    build_timeline_from_events,
    extract_time_mentions,
)


def test_extract_time_mentions_relative_and_simple():
    mentions = extract_time_mentions("三日后，二人抵达。次日离开。")
    assert [m.text for m in mentions] == ["三日后", "次日"]


def test_timeline_is_built_only_from_explicit_event_time_mentions():
    events = [
        {
            "event_id": "ev:1",
            "unit_id": "u:1",
            "summary": "三日后，二人抵达",
            "evidence": [{"unit_id": "u:1", "quote": "三日后，二人抵达"}],
        },
        {
            "event_id": "ev:2",
            "unit_id": "u:1",
            "summary": "次日，二人离开",
            "evidence": [{"unit_id": "u:1", "quote": "次日，二人离开"}],
        },
    ]
    texts = {"u:1": "三日后，二人抵达。次日，二人离开。"}
    annotated = annotate_event_times(events, texts)
    assert annotated[0]["time"]["time_expr"] == "三日后"
    rows = build_timeline_from_events("book", annotated, texts)
    assert len(rows) == 1
    assert rows[0]["source"] == "inferred"
    assert rows[0]["left"]["id"] == "ev:1"
    assert rows[0]["right"]["id"] == "ev:2"


def test_event_without_time_mention_is_not_invented():
    events = [{"event_id": "ev:1", "unit_id": "u:1", "summary": "走进大厅", "evidence": []}]
    annotated = annotate_event_times(events, {"u:1": "走进大厅"})
    assert "time" not in annotated[0] or not annotated[0]["time"].get("time_expr")
    assert build_timeline_from_events("book", annotated, {"u:1": "走进大厅"}) == []
