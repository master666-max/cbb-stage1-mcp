"""A human conflict decision must land in the fields the box reads.

Measured defect
---------------
`validate/adjudication.py:170` writes the five-way choice into `status`:

    target["status"] = picked          # "left" / "right" / "merge" / "disputed" / "superseded"

but the contract is two closed sets (`validate/conflicts.py:30-31`):

    CONFLICT_STATUSES = ("open", "resolved")
    DECISIONS         = ("left", "right", "merge", "disputed", "superseded")

and the box loader enforces them (`box/writer.py:1074-1076`): a status outside the pair raises
`BuildError(f"{path}:{index}: bad conflict status {status!r}")`. So resolving a conflict made the
next `build_box` fail — the decision both corrupted the row and blocked the box, and `decision`
stayed `None` so the decision itself never reached the box.

These tests go through the real writer: `resolve_conflict` to record the decision, then
`_load_work_conflicts` to prove the box accepts what was written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.box.writer import BuildError, _load_work_conflicts
from cbb_stage1.validate.adjudication import resolve_conflict

LEFT = "a:1111111111111111"
RIGHT = "a:2222222222222222"
CONFLICT_ID = "c:aaaaaaaaaaaaaaaa"


def _seed(work: Path) -> None:
    """One open conflict row, shaped the way the detectors emit it."""
    (work / "audit").mkdir(parents=True, exist_ok=True)
    row = {
        "conflict_id": CONFLICT_ID,
        "kind": "attribute",
        "left": LEFT,
        "right": RIGHT,
        "detail": "属性冲突：言万 的眼睛颜色在两章不一致",
        "status": "open",
        "decision": None,
        "resolved_by": None,
        "at": None,
    }
    (work / "audit" / "conflicts.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read(work: Path) -> dict:
    path = work / "audit" / "conflicts.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[0])


# ══ the row must keep the contract's two closed sets ═══════════════════
def test_resolving_sets_status_resolved(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed(work)

    resolve_conflict(
        work, conflict_id=CONFLICT_ID, choice="left", by="lead", reason="左值为原文直述"
    )

    row = _read(work)
    assert row["status"] == "resolved", (
        f"status is {row['status']!r}; the contract is ('open', 'resolved') and the choice "
        "belongs in `decision`"
    )


def test_resolving_records_the_choice_in_decision(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed(work)
    resolve_conflict(
        work, conflict_id=CONFLICT_ID, choice="right", by="lead", reason="右值有第二处佐证"
    )
    row = _read(work)
    assert row["decision"] == "right", f"decision is {row['decision']!r}"


def test_resolving_records_who_and_when(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed(work)
    resolve_conflict(
        work,
        conflict_id=CONFLICT_ID,
        choice="disputed",
        by="lead",
        reason="两处都不可靠",
        at="2026-10-23T02:00:00+08:00",
    )
    row = _read(work)
    assert row["resolved_by"] == "lead"
    assert row["at"] == "2026-10-23T02:00:00+08:00"


# ══ the box must accept what the adjudicator wrote ═════════════════════
def test_the_box_loader_accepts_a_resolved_conflict(tmp_path: Path) -> None:
    """The regression that mattered: resolving a conflict used to break build_box."""
    work = tmp_path / "work"
    _seed(work)
    resolve_conflict(work, conflict_id=CONFLICT_ID, choice="left", by="lead", reason="ok")

    rows = _load_work_conflicts(
        work,
        book_id="tst",
        assertion_ids={LEFT, RIGHT},
        created_at="2026-10-23T02:00:00+08:00",
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "resolved"
    assert rows[0]["decision"] == "left"
    assert rows[0]["resolved_by"] == "lead"


def test_the_box_loader_still_sees_an_unresolved_conflict_as_open(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed(work)
    rows = _load_work_conflicts(
        work,
        book_id="tst",
        assertion_ids={LEFT, RIGHT},
        created_at="2026-10-23T02:00:00+08:00",
    )
    assert rows[0]["status"] == "open"
    assert rows[0]["decision"] is None


def test_a_bad_status_is_still_rejected(tmp_path: Path) -> None:
    """The loader's own validation must stay strict."""
    work = tmp_path / "work"
    _seed(work)
    path = work / "audit" / "conflicts.jsonl"
    row = _read(work)
    row["status"] = "left"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(BuildError, match="bad conflict status"):
        _load_work_conflicts(
            work,
            book_id="tst",
            assertion_ids={LEFT, RIGHT},
            created_at="2026-10-23T02:00:00+08:00",
        )
