"""Use the established rich fixture instead of reinventing a work layer.

A hand-built minimal work layer failed the coverage gate (`qc/coverage.json` 53/60 cells absent), which
is the gate doing its job: a box is not valid without content. `make_rich_work` already builds a work
layer that passes every gate, so the replay test reuses it rather than duplicating a fixture that would
drift out of validity.

`tests/unit` is put on the path because the fixture lives there; other integration tests import unit
fixtures the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cbb_stage1.box.writer import build_box

_UNIT_TESTS = Path(__file__).resolve().parents[1] / "unit"
if str(_UNIT_TESTS) not in sys.path:
    sys.path.insert(0, str(_UNIT_TESTS))

FIXED = "2026-10-18T23:00:00+08:00"


@pytest.fixture
def rich_work(tmp_path: Path) -> Path:
    from test_box_writer import make_rich_work

    return make_rich_work(tmp_path)


def test_a_replay_against_a_parent_writes_a_report(rich_work: Path, tmp_path: Path) -> None:
    first = tmp_path / "box1"
    build_box(rich_work, first, created_at=FIXED, box_id="first")

    second = tmp_path / "box2"
    build_box(
        rich_work, second, created_at=FIXED, box_id="second", parent=str(first), replay=True
    )

    report_path = second / "qc" / "replay.json"
    assert report_path.is_file(), "replay=True produced no qc/replay.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["base"] == str(first), f"the report does not name its base: {report['base']!r}"
    assert report["status"] == "compared", f"a readable parent was reported as {report['status']!r}"
    assert report["counts"]["added"] == 0, (
        f"the same work layer produced added rows: {report['added']}"
    )
    assert report["counts"]["removed"] == 0, (
        f"the same work layer produced removed rows: {report['removed']}"
    )
    # The only legitimate difference is box identity: run records name the box they belong to. A replay
    # that hid this would be lying about reproducibility, and one that reported more would mean the
    # rebuild is not deterministic.
    for change in report["changed"]:
        assert change["table"] == "audit/runs.jsonl", (
            f"unexpected table in the diff: {change['table']}"
        )
        assert set(change["fields"]) == {"box_id"}, (
            f"a rebuild from identical inputs changed {list(change['fields'])}; only box identity may "
            "differ"
        )


def test_no_replay_flag_means_no_report(rich_work: Path, tmp_path: Path) -> None:
    first = tmp_path / "box1"
    build_box(rich_work, first, created_at=FIXED, box_id="first")
    second = tmp_path / "box2"
    build_box(rich_work, second, created_at=FIXED, box_id="second", parent=str(first))
    assert not (second / "qc" / "replay.json").exists(), (
        "a report was written without being asked for"
    )


def test_the_manifest_carries_the_parent_as_replay_base(
    rich_work: Path, tmp_path: Path
) -> None:
    first = tmp_path / "box1"
    build_box(rich_work, first, created_at=FIXED, box_id="first")
    second = tmp_path / "box2"
    build_box(rich_work, second, created_at=FIXED, box_id="second", parent=str(first))

    manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    incremental = manifest.get("incremental") or {}
    assert incremental.get("replay_base") == str(first), (
        f"replay_base is {incremental.get('replay_base')!r}, so a reader cannot tell what this build "
        "was derived from"
    )
