"""Stylometry must be computed, counted, copied, and non-empty.

Three "green-gate illusions" from the wiring audit, each with its own assertion here,
because each fooled a different gate before:

  * **空跑** — the code runs but produces nothing. `wiring` passes, tests pass, table is
    empty. Guard: assert rows > 0 on real text.
  * **静默丢弃** — computed, then dropped on the way to the box. Guard: assert the box
    table has the same row count as the work table.
  * **枚举错位** — the schema says one thing, the code emits another, and the writer
    nulls it. Guard: validate every row against the schema the spec defines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.derive.stylometry import METHOD, build_stylometry, measure_unit

#: Relative to this file, so the suite also runs on a machine that is not this one.
#: `parents[2]` is `cbb_stage1_v6`; the schemas sit beside it under `设计/抽取段-v6`.
from cbb_stage1.schemas import schemas_dir

SCHEMAS = schemas_dir(Path(__file__).resolve().parents[2])
UNIT = "u:t:v01:c001"
TEXT = (
    "「阿尔法小姐，那里是出口！」\n\n"
    "浑身是伤的男孩子拼命奔跑着，指向了光芒。\n\n"
    "（这孩子，要是能幸福就好了。）\n\n"
    "我怎么样都无所谓了。至少，这孩子要幸福。"
)


# ── the measurement itself ───────────────────────────────────────────────────
def test_a_non_empty_text_produces_metrics() -> None:
    row = measure_unit("t", UNIT, TEXT).to_row()
    metrics = row["metrics"]
    assert metrics["sentence_count"] > 0, "空跑: sentences must be counted"
    assert metrics["sentence_len_mean"] > 0
    assert 0.0 <= metrics["dialogue_ratio"] <= 1.0
    assert metrics["dialogue_ratio"] > 0, "the sample has quoted speech"
    assert 0.0 < metrics["type_token_ratio"] <= 1.0


def test_the_dialogue_ratio_only_counts_quoted_text() -> None:
    narration_only = measure_unit("t", UNIT, "他走进大厅，坐下。")
    with_quote = measure_unit("t", UNIT, "「你好。」")
    assert narration_only.dialogue_ratio == 0.0
    assert with_quote.dialogue_ratio > 0.0


def test_punctuation_density_is_a_fraction() -> None:
    row = measure_unit("t", UNIT, TEXT).to_row()
    assert 0.0 < row["metrics"]["punctuation_density"] < 1.0


def test_the_method_names_what_it_is() -> None:
    assert measure_unit("t", UNIT, TEXT).to_row()["method"] == METHOD
    assert METHOD == "char_statistics"


# ── no fabricated rows ───────────────────────────────────────────────────────
def test_a_unit_without_text_produces_no_row() -> None:
    """A zero-filled row would look like a measurement of zero."""
    rows = build_stylometry("t", [{"unit_id": UNIT}], texts={})
    assert rows == []


def test_one_row_per_unit_that_has_text() -> None:
    units = [{"unit_id": "u:t:v01:c001"}, {"unit_id": "u:t:v01:c002"}]
    rows = build_stylometry(
        "t", units, texts={"u:t:v01:c001": TEXT, "u:t:v01:c002": "短。"}
    )
    assert len(rows) == 2
    assert {row["unit_id"] for row in rows} == {"u:t:v01:c001", "u:t:v01:c002"}


# ── the schema is the reference, not the implementation ──────────────────────
def test_every_row_matches_the_spec_schema() -> None:
    schema = json.loads((SCHEMAS / "stylometry.schema.json").read_text(encoding="utf-8"))
    jsonschema = pytest.importorskip("jsonschema")

    rows = build_stylometry("t", [{"unit_id": UNIT}], texts={UNIT: TEXT})
    assert rows, "空跑: nothing to validate means the test proves nothing"
    for row in rows:
        jsonschema.validate(instance=row, schema=schema)


def test_required_fields_from_the_schema_are_all_present() -> None:
    """Structural check that runs even without jsonschema installed."""
    schema = json.loads((SCHEMAS / "stylometry.schema.json").read_text(encoding="utf-8"))
    row = measure_unit("t", UNIT, TEXT).to_row()

    assert [name for name in schema["required"] if name not in row] == []
    metric_required = schema["properties"]["metrics"]["required"]
    assert [name for name in metric_required if name not in row["metrics"]] == []
    assert [name for name in row if name not in schema["properties"]] == []


# ── silent-drop guard: the box table must match the work table ──────────────
def test_the_writer_copies_the_stylometry_table() -> None:
    """If the copy entry is missing, the table is computed and dropped."""
    writer = (
        Path(__file__).resolve().parents[2] / "src" / "cbb_stage1" / "box" / "writer.py"
    ).read_text(encoding="utf-8")
    assert '("derive/stylometry.jsonl", "derived/inference/stylometry.jsonl")' in writer


def test_the_derive_stage_counts_the_table() -> None:
    """A table nobody counts is a table nobody notices is empty."""
    integration = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "cbb_stage1"
        / "derive"
        / "integration.py"
    ).read_text(encoding="utf-8")
    assert '"stylometry": len(stylometry_rows)' in integration
    assert "stylometry_path" in integration


def test_the_schema_is_registered_with_the_field_coverage_gate() -> None:
    """Unregistered schema = the always-empty-field defect is invisible again."""
    tool = (
        Path(__file__).resolve().parents[2] / "tools" / "field_coverage.py"
    ).read_text(encoding="utf-8")
    assert '"stylometry.schema.json": "derived/inference/stylometry.jsonl"' in tool
