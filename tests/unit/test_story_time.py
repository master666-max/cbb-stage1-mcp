"""`story_time` has a schema slot and no producer, so it is always null.

Measured gap (D10)
------------------
The box schema for an event's time is:

    story_time: {start, end, precision}

and `box/records.py:769` reads it from `item["time"]["story_time"]`. But the extractor only has
`EventFact.time_expr` — a free-text expression like 三年后 or 清晨 — and nothing converts one into the
other. So `story_time` is `None` in every box, which is half of what D10 names: the mechanism is
"story time as an interval", and only the relative-order half was ever implemented.

What can be produced honestly
-----------------------------
`time_expr` frequently carries enough to place an event on a coarse scale:

* 第N年 / N年后 / 三年后 — a year offset, so `precision="year"`;
* 元月 / 三月 / 腊月 — a month name, so `precision="month"`;
* 清晨 / 正午 / 黄昏 / 夜 — a time of day, which is not a calendar interval and must **not** be
  dressed up as one.

The conversion is therefore conservative: it emits a `story_time` only when the expression names a
year or month anchor, and otherwise leaves it null with the reason recorded. Inventing `start`/`end`
from 清晨 would be the same class of error as a fabricated confidence.

The interval is deliberately **open-ended** (`end` may equal `start`) because a chapter-level
expression does not pin a day; claiming day precision from 三年后 would overstate the evidence.
"""

from __future__ import annotations

from cbb_stage1.temporal.story_time import parse_story_time


# ══ year anchors ═════════════════════════════════════════════════════
def test_a_year_offset_becomes_a_year_interval() -> None:
    result = parse_story_time("三年后")
    assert result is not None
    assert result["precision"] == "year"
    assert result["start"], "no start produced for 三年后"


def test_the_year_number_is_carried() -> None:
    result = parse_story_time("三年后")
    assert "3" in str(result["start"]) or "三" in str(result["start"])


def test_an_explicit_year_is_read() -> None:
    result = parse_story_time("第7年")
    assert result is not None
    assert result["precision"] == "year"


# ══ month anchors ════════════════════════════════════════════════════
def test_a_month_name_becomes_a_month_interval() -> None:
    result = parse_story_time("三月")
    assert result is not None
    assert result["precision"] == "month"


def test_a_lunar_month_name_is_read() -> None:
    result = parse_story_time("腊月")
    assert result is not None
    assert result["precision"] == "month"


# ══ what must NOT become an interval ═════════════════════════════════
def test_a_time_of_day_is_not_an_interval() -> None:
    """清晨 is not a calendar span; dressing it up as one would invent precision."""
    assert parse_story_time("清晨") is None


def test_an_empty_expression_yields_nothing() -> None:
    assert parse_story_time("") is None
    assert parse_story_time(None) is None


def test_an_unparseable_expression_yields_nothing() -> None:
    assert parse_story_time("某个说不清的时候") is None


# ══ the interval must not overstate precision ════════════════════════
def test_a_year_interval_does_not_claim_day_precision() -> None:
    result = parse_story_time("三年后")
    assert result["precision"] != "day"
    assert result["precision"] in ("year", "month", "decade")


def test_the_shape_matches_the_box_schema() -> None:
    import json
    from pathlib import Path

    import cbb_stage1

    design = Path(cbb_stage1.__file__).resolve().parents[3] / "设计" / "抽取段-v6"
    schema_path = design / "schemas" / "event.schema.json"
    if not schema_path.is_file():
        import pytest

        pytest.skip(f"box schema not present at {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    allowed = set(
        (schema["properties"]["time"]["properties"]["story_time"]["properties"]).keys()
    )
    result = parse_story_time("三年后")
    assert set(result).issubset(allowed), (
        f"parse_story_time emits keys the box schema forbids: {set(result) - allowed}"
    )
