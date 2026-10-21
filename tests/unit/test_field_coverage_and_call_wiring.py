"""The two checks that would have caught the verification.level gap.

Three gates passed while ``verification.level`` was permanently null:

  * ``wiring`` proves every module is imported — ``records.py`` was;
  * ``pytest`` proves my own tests pass — they did;
  * ``counts`` only counts fields someone remembered.

Neither asked "does the schema's required field actually carry a value" nor "is
this public function ever called". These tests pin both questions, and pin the
specific defect that motivated them.
"""

from __future__ import annotations

import importlib.util
import tempfile
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
from cbb_stage1.schemas import schemas_dir

SCHEMAS = schemas_dir(ROOT)


def _load_tool(name: str):
    path = TOOLS / name
    spec = importlib.util.spec_from_file_location(f"cbb_{name.replace('.', '_')}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── field coverage ───────────────────────────────────────────────────────────
def test_nullable_required_field_is_suspicious_not_a_defect() -> None:
    """A schema that permits null has explicitly made the value optional."""
    module = _load_tool("field_coverage.py")
    stat = module.FieldStat(
        table="t", field="f", rows=5, empty=5, required=True, nullable=True
    )
    assert stat.is_defect is False
    assert stat.is_suspicious is True


def test_required_non_nullable_field_that_is_always_empty_is_a_defect() -> None:
    module = _load_tool("field_coverage.py")
    stat = module.FieldStat(
        table="t", field="f", rows=5, empty=5, required=True, nullable=False
    )
    assert stat.is_defect is True


def test_allowed_empty_needs_a_reason_and_suppresses_the_defect() -> None:
    module = _load_tool("field_coverage.py")
    stat = module.FieldStat(
        table="t", field="f", rows=5, empty=5, required=True, allowed_reason="by design"
    )
    assert stat.is_defect is False


def test_empty_values_are_detected_in_every_shape() -> None:
    module = _load_tool("field_coverage.py")
    for empty in (None, "", "   ", [], {}, ()):
        assert module._is_empty(empty) is True, empty
    for filled in (0, False, "x", [0], {"a": 1}):
        assert module._is_empty(filled) is False, filled


def test_nullability_is_read_from_the_schema() -> None:
    module = _load_tool("field_coverage.py")
    schema = {
        "properties": {
            "a": {"type": ["string", "null"]},
            "b": {"type": "string"},
            "c": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        }
    }
    assert module._allows_null(schema, "a") is True
    assert module._allows_null(schema, "b") is False
    assert module._allows_null(schema, "c") is True


def test_checker_reports_a_hollow_required_field() -> None:
    """The checker must flag a required, non-nullable field that is always empty.

    This is the defect shape that motivated the tool: `verification` present but
    hollow. The fixture is minimal on purpose — the point is the classification,
    not a full schema-conformant row.
    """
    if not SCHEMAS.exists():
        pytest.skip("schemas directory not available")
    module = _load_tool("field_coverage.py")
    schema = json.loads((SCHEMAS / "assertion.schema.json").read_text(encoding="utf-8"))
    # pick a required, non-nullable field the fixture deliberately omits
    required = [
        n for n in schema.get("required") or []
        if not module._allows_null(schema, n) and n != "assertion_id"
    ]
    assert required, "assertion schema should have another non-nullable required field"
    field = required[0]

    box = Path(tempfile.mkdtemp(prefix="cbb-cov-"))
    (box / "knowledge").mkdir(parents=True)
    (box / "knowledge" / "assertions.jsonl").write_text(
        json.dumps({"assertion_id": "a:1"}) + "\n", encoding="utf-8"
    )
    stats, _ = module.check_box(box, SCHEMAS)
    flagged = [stat for stat in stats if stat.field == field and stat.is_defect]
    assert flagged, f"expected {field} to be flagged as a defect"


# ── call-level wiring ────────────────────────────────────────────────────────
def test_call_wiring_finds_a_function_nobody_calls(tmp_path: Path) -> None:
    module = _load_tool("call_wiring.py")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text(
        "def used():\n    return 1\n\n\ndef forgotten():\n    return 2\n",
        encoding="utf-8",
    )
    (package / "caller.py").write_text(
        "from pkg.mod import used\n\n\ndef go():\n    return used()\n", encoding="utf-8"
    )
    findings = module.analyse(package)
    names = {item["function"] for item in findings}
    assert "forgotten" in names
    assert "used" not in names


def test_call_wiring_ignores_private_helpers(tmp_path: Path) -> None:
    module = _load_tool("call_wiring.py")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text("def _helper():\n    return 1\n", encoding="utf-8")
    assert module.analyse(package) == []


def test_verification_level_is_now_derived_from_tier() -> None:
    """The tables existed all along; what was missing was a caller."""
    from cbb_stage1.validate.confidence import inference_allowed, verification_level

    assert verification_level("canonical") == "canonical"
    assert verification_level("major") == "major"
    assert verification_level("minor") == "minor"
    assert inference_allowed("canonical") == "open"
    assert inference_allowed("major") == "motivation_chain"
    assert inference_allowed("minor") == "off"


def test_validate_stamps_verification_level_from_entity_tier(tmp_path: Path) -> None:
    """End to end: a canonical subject and a minor subject get different levels."""
    from cbb_stage1.validate.integration import run as validate_run

    work = tmp_path / "work"
    for sub in ("identity", "assemble", "sources", "plan/texts"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    (work / "plan" / "texts" / "u1.txt").write_text("主角很强。", encoding="utf-8")
    (work / "identity" / "entities.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"entity_id": "e:t:person:hero", "tier": "canonical"},
                {"entity_id": "e:t:person:extra", "tier": "minor"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    def assertion(aid: str, subject: str) -> dict:
        return {
            "assertion_id": aid,
            "predicate": "性质",
            "library": "设定",
            "roles": {"主体": subject, "值": {"literal": "强"}},
            "claim_layer": "fact",
            "status": "asserted",
            "confidence": 0.8,
            "confidence_method": "semantic_entropy",
            "confidence_scope": "assertion",
            "evidence": [
                {
                    "unit_id": "u:1",
                    "start": 0,
                    "end": 3,
                    "quote": "主角很",
                    "occurrence": 1,
                    "mode": "narration",
                }
            ],
        }

    rows = [assertion("a:hero", "e:t:person:hero"), assertion("a:extra", "e:t:person:extra")]
    (work / "assemble" / "assertions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    validate_run(
        work,
        book_id="t",
        outputs=[],
        units=[{"unit_id": "u:1", "order": 1, "text_path": "plan/texts/u1.txt"}],
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )

    written = [
        json.loads(line)
        for line in (work / "assemble" / "assertions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_id = {row["assertion_id"]: row.get("verification") or {} for row in written}
    assert by_id["a:hero"]["level"] == "canonical"
    assert by_id["a:hero"]["tier"] == "canonical"
    assert by_id["a:hero"]["inference_scope"] == "open"
    assert by_id["a:extra"]["level"] == "minor"
    assert by_id["a:extra"]["inference_scope"] == "off"


# ── event time ───────────────────────────────────────────────────────────────
def test_event_time_expression_survives_normalisation() -> None:
    """Without this the timeline could never be built, however explicit the prose."""
    from cbb_stage1.engines.ai_reader.schema import ChapterFact, EventFact
    from cbb_stage1.normalize.facts import chapter_fact_to_engine_output

    text = "老人把钥匙塞进他手里。三年后，他回到城里。"
    fact = ChapterFact(
        chapter_id=1,
        novel_id="t",
        events=[
            EventFact(
                summary="他回到城里",
                type="旅行",
                evidence="三年后，他回到城里",
                time_expr="三年后",
            ),
            EventFact(
                summary="老人交付钥匙",
                type="社交",
                evidence="老人把钥匙塞进他手里",
            ),
        ],
    )
    output = chapter_fact_to_engine_output(fact, unit_id="u:1", text=text, book_id="t")
    with_time = [e for e in output.events if (e.time or {}).get("time_expr")]
    assert len(with_time) == 1
    assert with_time[0].time["time_expr"] == "三年后"
    assert with_time[0].time["anchor_unit"] == "u:1"


def test_event_without_time_gets_an_empty_record_not_a_guess() -> None:
    from cbb_stage1.engines.ai_reader.schema import ChapterFact, EventFact
    from cbb_stage1.normalize.facts import chapter_fact_to_engine_output

    text = "老人把钥匙塞进他手里。"
    fact = ChapterFact(
        chapter_id=1,
        novel_id="t",
        events=[
            EventFact(
                summary="老人交付钥匙",
                type="社交",
                evidence="老人把钥匙塞进他手里",
            )
        ],
    )
    output = chapter_fact_to_engine_output(fact, unit_id="u:1", text=text, book_id="t")
    assert output.events[0].time == {}
