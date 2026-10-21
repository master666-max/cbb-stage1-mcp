from pathlib import Path

import pytest

from cbb_stage1.orchestrator.roles import ROLES, Receipt, Task, Verdict, write_receipt, write_task, write_verdict


def test_roles_are_closed_and_verifier_needs_source():
    assert "verifier" in ROLES
    with pytest.raises(ValueError):
        Task(task_id="t1", role="verifier", unit_id="u:1", input_paths=[])
    Task(task_id="t1", role="verifier", unit_id="u:1", input_paths=["u:1.txt"])


def test_file_handoff_roundtrip(tmp_path: Path):
    work = tmp_path / "work"
    task = Task(task_id="t1", role="extractor", unit_id="u:1", output_path="out.json")
    write_task(work, task)
    write_receipt(work, Receipt(task_id="t1", status="ok", outputs=["out.json"], counts={"mentions": 1}))
    write_verdict(work, Verdict(task_id="t1", verdict="support", reason="quote matched"))
    assert (work / "plan" / "tasks" / "t1.json").exists()
    assert (work / "receipts" / "t1.json").exists()
    assert (work / "verify" / "t1.json").exists()
