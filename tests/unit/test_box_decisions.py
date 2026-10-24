"""Human decisions must reach the box, not be replaced by an empty list.

Measured defect
---------------
`box/writer.py:2289` writes the decisions table as a literal:

    "audit/decisions.jsonl": [],

while the validate stage does produce it — `validate/integration.py:483-484` reads and rewrites
`audit/decisions.jsonl` explicitly so the file exists even when empty, and `adjudication.py:92-97`
appends a row for every human decision. So a person resolves a conflict, the row is written, and
the box contains an empty table: §5.17's "keep the reason" never reaches the artefact.

There is also no `decision.schema.json`, so `TABLE_SCHEMAS` has no entry for the table and the box
lint validates nothing there — the table was declared, never populated, never checked.

These tests pin the pass-through and the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.box import writer as writer_module
from cbb_stage1.box.writer import TABLE_SCHEMAS, _load_work_decisions

CONFLICT_DECISION = {
    "decision_id": "d:aaaaaaaaaaaaaaaa",
    "kind": "conflict",
    "subject": "c:bbbbbbbbbbbbbbbb",
    "choice": "left",
    "decided_by": "lead",
    "reason": "左值为原文直述",
    "at": "2026-10-23T03:00:00+08:00",
}

SUPERSEDE_DECISION = {
    "decision_id": "d:cccccccccccccccc",
    "kind": "supersede",
    "subject": "a:1111111111111111",
    "winner": "a:2222222222222222",
    "choice": "superseded",
    "decided_by": "lead",
    "reason": "后者有第二处佐证",
    "at": "2026-10-23T03:05:00+08:00",
}


def _seed(work: Path, rows: list[dict]) -> None:
    (work / "audit").mkdir(parents=True, exist_ok=True)
    (work / "audit" / "decisions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


# ══ the loader exists and returns what the adjudicator wrote ═══════════
def test_the_loader_returns_the_decision_rows(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed(work, [CONFLICT_DECISION, SUPERSEDE_DECISION])
    rows = _load_work_decisions(work)
    assert len(rows) == 2
    assert [row["kind"] for row in rows] == ["conflict", "supersede"]


def test_an_empty_or_missing_table_yields_no_rows(tmp_path: Path) -> None:
    assert _load_work_decisions(tmp_path / "work") == []


def test_a_malformed_decision_is_rejected_rather_than_dropped(tmp_path: Path) -> None:
    """A decision without `decided_by` is not a human decision; say so at build time."""
    work = tmp_path / "work"
    broken = dict(CONFLICT_DECISION)
    del broken["decided_by"]
    _seed(work, [broken])
    with pytest.raises(writer_module.BuildError):
        _load_work_decisions(work)


def test_a_bad_choice_is_rejected(tmp_path: Path) -> None:
    work = tmp_path / "work"
    broken = dict(CONFLICT_DECISION)
    broken["choice"] = "随便"
    _seed(work, [broken])
    with pytest.raises(writer_module.BuildError):
        _load_work_decisions(work)


# ══ the table is declared, populated and validated ═════════════════════
def test_the_decisions_table_has_a_schema() -> None:
    assert "audit/decisions.jsonl" in TABLE_SCHEMAS, (
        "the box declares the table but no schema validates it"
    )


def test_the_schema_file_exists_and_accepts_a_real_row() -> None:
    from cbb_stage1 import schemas

    directory = next((p for p in schemas.candidate_dirs() if p.is_dir()), None)
    assert directory is not None, "no schema directory resolved"
    path = directory / TABLE_SCHEMAS["audit/decisions.jsonl"]
    assert path.is_file(), f"{path} is missing"

    import jsonschema

    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(CONFLICT_DECISION, schema)
    jsonschema.validate(SUPERSEDE_DECISION, schema)


def test_the_writer_no_longer_hardcodes_an_empty_table() -> None:
    """The literal `[]` is the defect; a source check keeps it from coming back."""
    source = Path(writer_module.__file__).read_text(encoding="utf-8")
    assert '"audit/decisions.jsonl": [],' not in source, (
        "the decisions table is hardcoded empty again"
    )
