"""The newly wired paths must actually run, not merely be imported.

Every test here answers "is this mechanism invoked on the runtime path", which is
the question the three previous gates could not answer. Each one would have failed
before this change.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cbb_stage1.validate.integration import run as validate_run


def _work() -> Path:
    work = Path(tempfile.mkdtemp(prefix="cbb-wired-"))
    for sub in ("identity", "assemble", "sources", "audit", "plan/texts"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    (work / "plan" / "texts" / "u_1.txt").write_text("主角很强。", encoding="utf-8")
    (work / "identity" / "entities.jsonl").write_text(
        json.dumps({"entity_id": "e:t:person:hero", "tier": "canonical"}) + "\n",
        encoding="utf-8",
    )
    return work


def _assertion(aid: str, *, superseded_by: str | None = None) -> dict:
    return {
        "assertion_id": aid,
        "predicate": "性质",
        "library": "设定",
        "roles": {"主体": "e:t:person:hero", "值": {"literal": "强"}},
        "claim_layer": "fact",
        "status": "asserted",
        "confidence": 0.8,
        "confidence_method": "semantic_entropy",
        "confidence_scope": "assertion",
        "superseded_by": superseded_by,
        "evidence": [
            {
                "unit_id": "u:1",
                "start": 0,
                "end": 2,
                "quote": "主角",
                "occurrence": 1,
                "mode": "narration",
            }
        ],
    }


def test_validate_runs_per_record_section_5_15_checks() -> None:
    """validate_assertion_record existed and was never called."""
    work = _work()
    rows = [_assertion("a:1")]
    (work / "assemble" / "assertions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = validate_run(
        work,
        book_id="t",
        outputs=[],
        units=[{"unit_id": "u:1", "order": 1, "text_path": "plan/texts/u_1.txt"}],
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )
    counts = result["counts"]
    assert "record_violations" in counts, "per-record §5.15 validation did not run"
    assert counts["record_violations"] == 0
    # and the note proves the loop executed rather than short-circuiting
    notes = " ".join(result.get("notes") or [])
    assert "验证分级" in notes or "§5.15" in notes


def test_validate_reports_supersede_chains_and_retcon_history() -> None:
    """supersede_chains / retcon_history existed and were never called."""
    work = _work()
    rows = [_assertion("a:1"), _assertion("a:2")]
    (work / "assemble" / "assertions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = validate_run(
        work,
        book_id="t",
        outputs=[],
        units=[{"unit_id": "u:1", "order": 1, "text_path": "plan/texts/u_1.txt"}],
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )
    counts = result["counts"]
    assert "supersede_chains" in counts
    assert "retcon_rows" in counts


def test_dangling_supersede_is_reported() -> None:
    """A superseded_by pointing at a missing record must be caught."""
    work = _work()
    rows = [_assertion("a:1", superseded_by="a:missing")]
    (work / "assemble" / "assertions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = validate_run(
        work,
        book_id="t",
        outputs=[],
        units=[{"unit_id": "u:1", "order": 1, "text_path": "plan/texts/u_1.txt"}],
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )
    assert result["counts"]["supersede_errors"] >= 1


def test_timeline_conflicts_are_detected_not_skipped() -> None:
    """detect_timeline_conflicts existed; only attribute/relation were called."""
    work = _work()
    rows = [_assertion("a:1")]
    (work / "assemble" / "assertions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    # Two constraints forming a cycle: A before B and B before A.
    timeline = [
        {
            "constraint_id": "t:1",
            "book_id": "t",
            "kind": "before",
            "left": {"kind": "event", "id": "ev:1"},
            "right": {"kind": "event", "id": "ev:2"},
            "deps": [],
            "evidence": [],
        },
        {
            "constraint_id": "t:2",
            "book_id": "t",
            "kind": "before",
            "left": {"kind": "event", "id": "ev:2"},
            "right": {"kind": "event", "id": "ev:1"},
            "deps": [],
            "evidence": [],
        },
    ]
    (work / "assemble" / "timeline.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in timeline) + "\n",
        encoding="utf-8",
    )
    result = validate_run(
        work,
        book_id="t",
        outputs=[],
        units=[{"unit_id": "u:1", "order": 1, "text_path": "plan/texts/u_1.txt"}],
        created_at="2026-01-01T00:00:00",
        upstream={"unit_order": {"u:1": 1}},
    )
    # The detector reports a contradiction (or quarantines an unresolvable cycle);
    # the point is that it RAN. Before this change ``detect_timeline_conflicts`` was
    # never called, so ``conflicts_by_kind.timeline`` was 0 by construction.
    counts = result["counts"]
    assert counts["timeline"] == 2
    assert (
        counts.get("conflicts", 0)
        + counts.get("cycles", 0)
        + counts.get("quarantine_added", 0)
        >= 1
    ), counts


def test_build_box_records_work_table_problems() -> None:
    """check_work_tables existed; its finding now surfaces in qc.report."""
    from cbb_stage1.box.writer import _work_table_problems

    work = Path(tempfile.mkdtemp(prefix="cbb-wt-"))
    (work / "plan").mkdir(parents=True, exist_ok=True)
    # plan/wiring.json is the only GAP table; absent here.
    problems = _work_table_problems(work)
    assert any("wiring.json" in item for item in problems)


def test_call_wiring_reports_zero_after_the_wiring_pass() -> None:
    """The gate itself must be green, and it must be green for the right reason.

    Exemptions are recorded with a reason; anything left over is a real gap.
    """
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("cw", root / "tools" / "call_wiring.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cw"] = module
    spec.loader.exec_module(module)
    findings = module.analyse(root / "src" / "cbb_stage1")
    assert findings == [], [item["qualname"] for item in findings]
