"""An incremental plan must be applied or explicitly marked advisory, never silently ignored.

Measured defect
---------------
`incremental/runner.py` computes two things the pipeline never uses:

* ``cleanup_files`` — engine artifacts and old texts belonging to units that no longer exist;
* ``replace`` — ``{work table: record ids}`` to drop before recomputing.

`orchestrator/pipeline.py:750` only logs ``dirty_tables=len(inc_plan.replace)``. So a shrunk book
keeps the deleted units' engine outputs on disk, and the plan file claims a precision the pipeline
does not have.

What can be applied safely, and what cannot
-------------------------------------------
``replace`` **cannot** be consumed meaningfully today: every table it names
(`plan/units.jsonl`, `resolve/*.jsonl`, `assemble/*.jsonl`, `derive/hypotheses.jsonl`) is rewritten
whole by the postprocess stages, so dropping rows first changes nothing. Pretending otherwise would
be the same species of defect. It is recorded as advisory instead, with the reason.

``cleanup_files`` can be applied, and deleting a stale engine artifact is worth doing — with two
guards, because deleting the wrong file is not recoverable:

1. only paths under ``engine/`` or ``plan/texts/`` are eligible;
2. a file named by a *current* unit is never deleted, whatever the plan says.
"""

from __future__ import annotations

from pathlib import Path

from cbb_stage1.incremental.runner import IncrementalPlan, apply_cleanup_files

UNIT_A = "u:tst:v01:c001"
UNIT_B = "u:tst:v01:c002"


def _plan(cleanup: list[str]) -> IncrementalPlan:
    # The dataclass is frozen, so everything goes through the constructor.
    return IncrementalPlan(
        work_root=str(Path("work")),
        full_run=False,
        parent=None,
        previous_state_hash="sha256:" + "0" * 64,
        current_state_hash="sha256:" + "1" * 64,
        changed_units=(UNIT_B,),
        cleanup_files=tuple(cleanup),
        replace={"assemble/assertions.jsonl": ("a:1111111111111111",)},
    )


def _seed(work: Path, relative: str, body: str = "{}") -> Path:
    path = work / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ══ stale engine artifacts are removed ═════════════════════════════════
def test_a_stale_engine_artifact_is_deleted(tmp_path: Path) -> None:
    work = tmp_path / "work"
    stale = _seed(work, f"engine/{UNIT_A.replace(':', '_')}.json")
    assert stale.is_file()

    removed = apply_cleanup_files(work, _plan([f"engine/{UNIT_A.replace(':', '_')}.json"]), live_unit_ids={UNIT_B})

    assert not stale.exists(), "the stale engine artifact survived"
    assert removed == (f"engine/{UNIT_A.replace(':', '_')}.json",)


def test_a_file_belonging_to_a_live_unit_is_kept(tmp_path: Path) -> None:
    """The guard that matters: a wrong plan must not delete a unit that is still being extracted."""
    work = tmp_path / "work"
    live = _seed(work, f"engine/{UNIT_B.replace(':', '_')}.json")

    removed = apply_cleanup_files(work, _plan([f"engine/{UNIT_B.replace(':', '_')}.json"]), live_unit_ids={UNIT_B})

    assert live.exists(), "a live unit's engine artifact was deleted"
    assert removed == ()


def test_a_path_outside_the_allowed_directories_is_refused(tmp_path: Path) -> None:
    work = tmp_path / "work"
    keep = _seed(work, "assemble/assertions.jsonl", '{"assertion_id":"a:1"}\n')

    removed = apply_cleanup_files(work, _plan(["assemble/assertions.jsonl"]), live_unit_ids=set())

    assert keep.exists(), "a work table was deleted through cleanup_files"
    assert removed == ()


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "engine").mkdir(parents=True)
    removed = apply_cleanup_files(work, _plan(["engine/never_existed.json"]), live_unit_ids=set())
    assert removed == ()


# ══ the plan records what it could not apply ══════════════════════════
def test_the_plan_marks_replace_as_advisory() -> None:
    record = _plan([]).to_dict()
    assert record["replace"], "replace plan missing"
    advisory = record.get("replace_advisory")
    assert advisory, (
        "the plan lists rows to replace but never says they are not applied; a reader cannot tell "
        "an unapplied plan from an empty one"
    )
    assert "rewritten" in str(advisory).lower() or "full" in str(advisory).lower()


def test_cleanup_is_recorded_after_it_runs(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed(work, f"engine/{UNIT_A.replace(':', '_')}.json")
    plan = _plan([f"engine/{UNIT_A.replace(':', '_')}.json"])
    removed = apply_cleanup_files(work, plan, live_unit_ids={UNIT_B})
    assert removed
    assert plan.applied_cleanup == removed, "the plan does not record what it actually deleted"
