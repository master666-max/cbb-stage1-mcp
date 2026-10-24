"""Regression tests for the "empty box is green" fix.

Two root causes are pinned here:

* ``tools/box_lint.py`` had only per-row checks (vacuously true on empty
  tables).  It now has table-level coverage invariants.
* ``writer.py`` wrote seven literal ``"pass"`` hard gates and fake ``0.0`` soft
  gates.  It now measures the canonical lint result.

A box with source units but no ``knowledge/events.jsonl`` / ``assertions.jsonl``
rows must FAIL, while the project's rich/example fixtures must keep passing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cbb_stage1.box import atomic
from cbb_stage1.box.gates import classify_error, measure_hard_gates
from cbb_stage1.box.lint import lint_box, run_box_lint
from cbb_stage1.box.writer import BuildError, build_box

FIXED = "2026-10-18T20:00:00+08:00"
BOOK = "t"
U1 = "u:t:v01:c001"
TEXT = "我走进房间。"
UNIT_TEXT_PATH = "plan/texts/v01/u001.txt"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _minimal_work(root: Path) -> Path:
    """One unit + one mention, but no assertions/events at all."""
    (root / "plan" / "texts" / "v01").mkdir(parents=True)
    (root / UNIT_TEXT_PATH).write_text(TEXT, encoding="utf-8", newline="\n")
    _write_jsonl(
        root / "plan" / "units.jsonl",
        [
            {
                "unit_id": U1,
                "book_id": BOOK,
                "volume": 1,
                "chapter": 1,
                "title": "第1章",
                "order": 1,
                "prev": None,
                "next": None,
                "text_path": UNIT_TEXT_PATH,
                "chars": len(TEXT),
                "sha256": atomic.sha256_text(TEXT),
                "narrative_mode": "present",
                "narrator": {"kind": "third_person", "entity_id": None, "name": "叙述者"},
                "narrator_reliability": "reliable",
                "source_locator": {"href": "chapter1.xhtml", "nav_index": 0},
                "scenes": [],
                "created_at": FIXED,
            }
        ],
    )
    _write_json(
        root / "plan" / "run.json",
        {
            "book_id": BOOK,
            "book_title": "空知识测试",
            "trace_id": "emptygates0001",
            "engine": "MockEngine:mock",
            "started_at": FIXED,
            "ended_at": FIXED,
        },
    )
    _write_json(
        root / "engine" / "u_t_v01_c001.json",
        {
            "unit_id": U1,
            "mentions": [
                {
                    "unit_id": U1,
                    "surface": "我",
                    "start": 0,
                    "end": 1,
                    "occurrence": 1,
                    "mention_type": "name",
                }
            ],
            "entities": [],
        },
    )
    return root


def _build_minimal(tmp_path: Path, name: str = "box", *, verify: bool = False):
    work = _minimal_work(tmp_path / "work")
    return build_box(work, tmp_path / name, created_at=FIXED, box_id="empty-gates", verify=verify)


def _rich_work(tmp_path: Path) -> Path:
    from test_box_writer import make_rich_work

    return make_rich_work(tmp_path)


# ---------------------------------------------------------------------------
# 1. the table-level invariant itself
# ---------------------------------------------------------------------------


def test_units_without_knowledge_fail_lint_and_name_the_tables(tmp_path):
    result = _build_minimal(tmp_path)
    errors = lint_box(result.box)
    joined = "\n".join(errors)

    assert "coverage invariant" in joined
    assert "knowledge/events.jsonl" in joined
    assert "knowledge/assertions.jsonl" in joined
    assert "有 1 个 unit" in joined
    # Empty quarantine: the message must still be explicit about the reason.
    assert "audit/quarantine.jsonl 无行" in joined

    qc = _read_json(result.box / "qc" / "report.json")
    assert qc["hard_gates"]["references"] == "fail"
    assert qc["status"] == "fail"
    assert qc["hard_gate_errors"]["references"]


def test_invariant_message_carries_top_quarantine_reason(tmp_path):
    work = _rich_work(tmp_path)
    complete = tmp_path / "complete"
    build_box(work, complete, created_at=FIXED, box_id="rich-copy")

    broken = tmp_path / "broken"
    import shutil

    shutil.copytree(complete, broken)
    # The published box is read-only (E9), so the deliberate corruption has to lift the marking first.
    # This is what the module is for: immutability you cannot lift is a bug, not a safeguard.
    from cbb_stage1.box.immutability import make_writable

    make_writable(broken)
    for relative in ("knowledge/events.jsonl", "knowledge/assertions.jsonl"):
        (broken / relative).write_text("", encoding="utf-8")

    reasons = Counter(
        row.get("reason") or "<missing>"
        for row in (
            json.loads(line)
            for line in (broken / "audit" / "quarantine.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    )
    assert reasons, "rich fixture should carry quarantine reasons"
    top_reason = reasons.most_common(1)[0][0]

    joined = "\n".join(lint_box(broken))
    assert "coverage invariant" in joined
    assert "有 2 个 unit" in joined
    assert top_reason in joined


# ---------------------------------------------------------------------------
# 2. writer: hard gates are measured, not literal
# ---------------------------------------------------------------------------


def test_writer_hard_gates_are_measured_and_manifest_syncs(tmp_path):
    result = _build_minimal(tmp_path, verify=False)

    qc = _read_json(result.box / "qc" / "report.json")
    assert qc["hard_gates"] != {gate: "pass" for gate in qc["hard_gates"]}
    assert "fail" in qc["hard_gates"].values()
    assert qc["status"] == "fail"

    manifest = _read_json(result.box / "manifest.json")
    assert manifest["qc"]["hard_gates"] == "fail"
    # The measured failure is the table invariant, not a vacuous pass.
    assert any(
        "coverage invariant" in message
        for message in qc["hard_gate_errors"].get("references", [])
    )

    # Default behaviour (verify=True) must refuse to publish the empty box.
    with pytest.raises(BuildError) as excinfo:
        build_box(
            tmp_path / "work",
            tmp_path / "box-verified",
            created_at=FIXED,
            box_id="empty-gates-verified",
        )
    assert "coverage invariant" in str(excinfo.value)
    assert not (tmp_path / "box-verified").exists()


def test_soft_gates_are_null_and_marked_not_measured(tmp_path):
    work = _rich_work(tmp_path)
    box = tmp_path / "box"
    build_box(work, box, created_at=FIXED, box_id="rich-soft")
    soft = _read_json(box / "qc" / "report.json")["soft_gates"]

    for key in ("unresolved_rate", "cost_usd", "model_decoupled"):
        assert soft[key] is None, key
        assert key in soft["notes"], key
        assert "not measured" in soft["notes"][key]


# ---------------------------------------------------------------------------
# 3. compliant boxes are not false-killed
# ---------------------------------------------------------------------------


def test_rich_box_still_passes_all_measured_gates(tmp_path):
    work = _rich_work(tmp_path)
    box = tmp_path / "box"
    result = build_box(work, box, created_at=FIXED, box_id="rich-pass")

    assert result.lint_errors == ()
    qc = _read_json(box / "qc" / "report.json")
    assert all(status == "pass" for status in qc["hard_gates"].values())
    assert _read_json(box / "manifest.json")["qc"]["hard_gates"] == "pass"
    assert lint_box(box) == []


# ---------------------------------------------------------------------------
# 4. the invariant has an explicit, loud escape hatch
# ---------------------------------------------------------------------------


def test_coverage_invariant_can_be_disabled_by_env(tmp_path, monkeypatch, capsys):
    result = _build_minimal(tmp_path)
    monkeypatch.setenv("CBB_BOX_LINT_DISABLE_COVERAGE_INVARIANT", "1")

    errors = lint_box(result.box)
    output = capsys.readouterr().out
    assert "coverage invariant DISABLED" in output
    assert not any("coverage invariant:" in error for error in errors)


def test_coverage_invariant_can_be_disabled_by_flag(tmp_path):
    result = _build_minimal(tmp_path)
    _rc, output = run_box_lint(result.box, "--no-coverage-invariant")
    assert "coverage invariant DISABLED" in output


# ---------------------------------------------------------------------------
# 5. classifier coverage (no failure is silently dropped)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "gate"),
    [
        ("[x] checksum: 文件缺失 knowledge/events.jsonl", "hashes"),
        ("[x] timeline t1: left 不存在", "timeline"),
        ("[x] assertion a1: 未声明角色 主体", "roles"),
        ("[x] assertion a1: quote 不在原文 span：'x' != 'y'", "evidence"),
        ("[x] graph edge g1: assertion 不存在", "references"),
        ("[x] event ev1: 重复", "ids"),
        ("[x] manifest: /counts: 'units' is a required property", "schema"),
        ("[x] coverage invariant: knowledge/events.jsonl 为 0 行", "references"),
    ],
)
def test_classify_error_maps_to_gate(message, gate):
    assert classify_error(message) == gate


def test_measure_hard_gates_never_returns_a_constant(tmp_path):
    result = _build_minimal(tmp_path)
    report = measure_hard_gates(result.box)
    assert not report.ok
    assert "fail" in report.gates.values()
    # Every reported error lands in exactly one gate bucket; nothing is dropped.
    assert sum(len(messages) for messages in report.by_gate.values()) == len(
        report.errors
    )
    assert set(report.gates) == {
        "schema",
        "evidence",
        "references",
        "roles",
        "timeline",
        "ids",
        "hashes",
    }
