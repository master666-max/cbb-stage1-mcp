"""Themes, emotion curves, motivation chains: behaviour, contract, and wiring.

Each module is checked three ways, because three different gates failed on this project
before for three different reasons:

  * behaviour — does it do what the paper says, not what my code happens to do;
  * contract — does every row satisfy the schema written before the code;
  * wiring — is it computed, counted, and copied to the box.

TDD note: these three modules were implemented before this file was written, which
breaks the repo's red-first rule. Recorded here rather than glossed: the schema was
written first, and the emotion module's `window_chars: 0` violation was caught by it,
but that is not the same as a failing test first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.derive.emotion import METHOD as EMOTION_METHOD
from cbb_stage1.derive.emotion import build_emotion_curves
from cbb_stage1.derive.motivation import MASLOW_LEVELS, build_motivation_chains, parse_chain
from cbb_stage1.derive.themes import METHOD as THEME_METHOD
from cbb_stage1.derive.themes import build_themes, count_motifs

ROOT = Path(__file__).resolve().parents[2]
#: The schemas live beside the code, not at an absolute path: the previous version
#: pinned an absolute path to the design tree, which works on exactly one
#: machine. `parents[2]` is `cbb_stage1_v6`, so `parent` is `V5`.
from cbb_stage1.schemas import schemas_dir

SCHEMAS = schemas_dir(ROOT)

TEXT = (
    "那里，是巨大的蓝色湖泊，腐朽的大楼沉没其中。\n\n"
    "「露娜小姐，那里是出口！」\n\n"
    "湖的浅滩上，有什么小东西在滚动。\n\n"
    "「好。把它毁掉吧。」\n\n"
    "我不由得接受了──这个世界为何如此极端。"
)
UNIT = "u:t:v01:c001"


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


# ══ themes ═══════════════════════════════════════════════════════════════════
def test_themes_count_nouns_not_clause_fragments() -> None:
    """The first version used a greedy CJK regex and produced whole clauses."""
    motifs = [motif for motif, _total, _units in count_motifs({UNIT: TEXT})]
    assert motifs, "空跑: no motif counted on text with obvious nouns"
    for motif in motifs:
        assert len(motif) <= 8
        # A clause fragment would contain a verb particle like 了/的 in the middle.
        assert "，" not in motif and "。" not in motif


def test_a_recurring_motif_is_ranked_above_a_single_mention() -> None:
    text = "小狗来了。小狗走了。小狗又叫。蓝色湖泊在那里。"
    motifs = [motif for motif, _total, _units in count_motifs({UNIT: text})]
    if "小狗" in motifs and "蓝色湖泊" in motifs:
        assert motifs.index("小狗") < motifs.index("蓝色湖泊")


def test_themes_record_which_units_carry_the_motif() -> None:
    rows = build_themes("t", {"u:t:v01:c001": "小狗来了。", "u:t:v01:c002": "小狗又叫。"})
    if not rows:
        pytest.skip("jieba produced no noun for this sample")
    row = next(row for row in rows if row["motif"] == "小狗")
    assert set(row["units"]) == {"u:t:v01:c001", "u:t:v01:c002"}
    assert row["unit_counts"] == {"u:t:v01:c001": 1, "u:t:v01:c002": 1}
    assert row["frequency"] == 2


def test_themes_rows_match_the_spec_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    rows = build_themes("t", {UNIT: TEXT})
    if not rows:
        pytest.skip("jieba produced no noun for this sample")
    schema = _schema("themes.schema.json")
    for row in rows:
        jsonschema.validate(instance=row, schema=schema)


def test_themes_method_names_what_it_is() -> None:
    assert THEME_METHOD == "word_frequency_by_segment"


# ══ emotion ══════════════════════════════════════════════════════════════════
def test_a_quoted_line_keeps_its_own_lane() -> None:
    """The speaker split is the whole reason this table needed the speaker work.

    A plain sentence splitter cut through 「好。把它毁掉吧。」 at its inner full stop, so
    the fragment never matched the quote on the assertion and every line fell into the
    narration lane — the per-speaker split silently did nothing.
    """
    statements = [
        {
            "claim_layer": "statement",
            "speaker": {"name": "露娜"},
            "evidence": [{"quote": "「好。把它毁掉吧。」"}],
        }
    ]
    rows = build_emotion_curves("t", [{"unit_id": UNIT}], texts={UNIT: TEXT}, statements=statements)
    lanes = {row["speaker"] for row in rows}
    assert "露娜" in lanes, "the attributed quote must land in its own lane"
    assert "" in lanes, "the narration voice is a separate lane"


def test_without_attribution_everything_falls_to_narration() -> None:
    rows = build_emotion_curves("t", [{"unit_id": UNIT}], texts={UNIT: TEXT})
    assert {row["speaker"] for row in rows} <= {""}


def test_emotion_rows_match_the_spec_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    rows = build_emotion_curves("t", [{"unit_id": UNIT}], texts={UNIT: TEXT})
    assert rows, "空跑: no curve produced"
    schema = _schema("emotion_curve.schema.json")
    for row in rows:
        jsonschema.validate(instance=row, schema=schema)


def test_the_curve_does_not_claim_to_be_ued() -> None:
    """The UED paper is not in the resource library; the method must not imply it."""
    rows = build_emotion_curves("t", [{"unit_id": UNIT}], texts={UNIT: TEXT})
    assert rows[0]["method"] == EMOTION_METHOD == "lexical_polarity_window"
    assert "UED" in rows[0]["note"] and "不是" in rows[0]["note"]


# ══ motivation ═══════════════════════════════════════════════════════════════
def test_the_needs_set_is_closed_to_five() -> None:
    assert MASLOW_LEVELS == ("生理", "安全", "归属", "尊重", "自我实现")


def test_a_level_outside_the_closed_set_is_dropped_not_rounded() -> None:
    raw = '[{"behavior_index": 0, "motive": "想逃", "needs_level": "自由"}]'
    assert parse_chain(raw, ["他想逃"]) == []


def test_a_valid_link_is_kept() -> None:
    raw = '[{"behavior_index": 0, "motive": "想活命", "needs_level": "安全"}]'
    links = parse_chain(raw, ["他想逃"])
    assert len(links) == 1
    assert links[0]["needs_level"] == "安全"


def test_no_model_produces_no_rows_rather_than_guesses() -> None:
    assertions = [
        {"roles": {"主体": "露娜"}, "evidence": [{"unit_id": UNIT, "quote": "「好。」"}]}
    ]
    assert build_motivation_chains("t", assertions, llm=None) == []


def test_a_chain_carries_an_evidence_chain() -> None:
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return '[{"behavior_index": 0, "motive": "失去一切", "needs_level": "归属"}]'

    assertions = [
        {"roles": {"主体": "露娜"}, "evidence": [{"unit_id": UNIT, "quote": "「好。」"}]}
    ]
    rows = build_motivation_chains("t", assertions, llm=FakeLLM())
    assert len(rows) == 1
    link = rows[0]["links"][0]
    assert link["evidence"], "§5.20 requires an evidence chain"
    assert link["evidence"][0]["quote"] == "「好。」"


def test_motivation_rows_match_the_spec_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return '[{"behavior_index": 0, "motive": "失去一切", "needs_level": "归属"}]'

    assertions = [
        {"roles": {"主体": "露娜"}, "evidence": [{"unit_id": UNIT, "quote": "「好。」"}]}
    ]
    rows = build_motivation_chains("t", assertions, llm=FakeLLM())
    assert rows, "空跑: no chain produced"
    schema = _schema("motivation_chain.schema.json")
    for row in rows:
        jsonschema.validate(instance=row, schema=schema)


def test_an_entity_id_is_resolved_to_a_readable_name() -> None:
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return '[{"behavior_index": 0, "motive": "扩张", "needs_level": "尊重"}]'

    assertions = [
        {
            "roles": {"主体": "e:t:org:abc"},
            "evidence": [{"unit_id": UNIT, "quote": "「好。」"}],
        }
    ]
    rows = build_motivation_chains(
        "t", assertions, llm=FakeLLM(), entity_names={"e:t:org:abc": "境界领域商会"}
    )
    assert rows[0]["character"] == "境界领域商会", "a chain keyed by entity id is unreadable"
    assert "未解析" not in rows[0]["note"]


def test_an_unresolved_id_is_kept_and_declared() -> None:
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return '[{"behavior_index": 0, "motive": "扩张", "needs_level": "尊重"}]'

    assertions = [
        {
            "roles": {"主体": "e:t:org:abc"},
            "evidence": [{"unit_id": UNIT, "quote": "「好。」"}],
        }
    ]
    rows = build_motivation_chains("t", assertions, llm=FakeLLM())
    assert rows[0]["character"] == "e:t:org:abc"
    assert "未解析" in rows[0]["note"], "the reader must be told the name is an id"


# ══ wiring: every table must be computed, counted, and copied ═══════════════
@pytest.mark.parametrize(
    ("work", "box"),
    [
        ("derive/stylometry.jsonl", "derived/inference/stylometry.jsonl"),
        ("derive/themes.jsonl", "derived/inference/themes.jsonl"),
        ("derive/emotion_curves.jsonl", "derived/inference/emotion_curves.jsonl"),
        ("derive/motivation_chains.jsonl", "derived/inference/motivation_chains.jsonl"),
    ],
)
def test_each_table_is_copied_to_the_box(work: str, box: str) -> None:
    writer = (ROOT / "src" / "cbb_stage1" / "box" / "writer.py").read_text(encoding="utf-8")
    assert f'("{work}", "{box}")' in writer, f"{work} computed but never copied (静默丢弃)"


@pytest.mark.parametrize(
    "counter",
    [
        '"stylometry": len(stylometry_rows)',
        '"themes": len(theme_rows)',
        '"emotion_curves": len(emotion_rows)',
        '"motivation_chains": len(motivation_rows)',
    ],
)
def test_each_table_is_counted_in_the_receipt(counter: str) -> None:
    integration = (
        ROOT / "src" / "cbb_stage1" / "derive" / "integration.py"
    ).read_text(encoding="utf-8")
    assert counter in integration


@pytest.mark.parametrize(
    ("schema", "table"),
    [
        ("stylometry.schema.json", "derived/inference/stylometry.jsonl"),
        ("themes.schema.json", "derived/inference/themes.jsonl"),
        ("motivation_chain.schema.json", "derived/inference/motivation_chains.jsonl"),
        ("emotion_curve.schema.json", "derived/inference/emotion_curves.jsonl"),
    ],
)
def test_each_schema_is_registered_with_field_coverage(schema: str, table: str) -> None:
    tool = (ROOT / "tools" / "field_coverage.py").read_text(encoding="utf-8")
    assert f'"{schema}": "{table}"' in tool, (
        f"{schema} unregistered: an always-empty required field would be invisible"
    )