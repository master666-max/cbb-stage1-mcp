"""Two runs in one work directory must not see each other's queue.

Measured failure: a second run's `cbb_next_requests` returned the previous run's unanswered
requests (16 "pending", 14 already answered). The tools now read the queue path from the run
record, and a record without one falls back to the legacy shared directory so old work
directories keep working.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SERVER = PROJECT / "tools" / "mcp_pipeline_server.py"


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("cbb_mcp_queue", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_queue"] = module
    spec.loader.exec_module(module)
    return module


# ── _queue_dir ─────────────────────────────────────────────────────────
def test_a_run_scoped_queue_is_under_the_work_root(server, tmp_path: Path) -> None:
    work = tmp_path / "work"
    resolved = server._queue_dir(work, "plan/host-model/abc123")
    assert resolved == work / "plan" / "host-model" / "abc123"


def test_a_record_without_a_queue_falls_back_to_the_legacy_path(server, tmp_path: Path) -> None:
    """Old runs (no `queue` field) must keep working unchanged."""
    work = tmp_path / "work"
    assert server._queue_dir(work, None) == work / "plan" / "host-model"
    assert server._queue_dir(work, "") == work / "plan" / "host-model"


def test_the_legacy_constant_is_the_shared_directory(server) -> None:
    assert server.LEGACY_QUEUE_RELATIVE == "plan/host-model"


# ── isolation, end to end through the tools ────────────────────────────
def _write_request(queue: Path, request_id: str) -> None:
    (queue / "requests").mkdir(parents=True, exist_ok=True)
    (queue / "responses").mkdir(parents=True, exist_ok=True)
    (queue / "requests" / f"{request_id}.json").write_text(
        json.dumps({"request_id": request_id, "prompt": "x"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _record(work: Path, run_id: str, queue: str) -> None:
    runs = work / "plan" / "mcp-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "work": str(work),
                "out": str(work / "box"),
                "queue": queue,
                "pid": 1,
                "returncode": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_two_runs_in_one_work_directory_do_not_share_requests(server, tmp_path: Path) -> None:
    work = tmp_path / "work"
    _record(work, "runA", "plan/host-model/runA")
    _record(work, "runB", "plan/host-model/runB")

    _write_request(work / "plan" / "host-model" / "runA", "req-A")
    _write_request(work / "plan" / "host-model" / "runB", "req-B")

    result_a = server.cbb_next_requests({"work": str(work), "run_id": "runA"})
    result_b = server.cbb_next_requests({"work": str(work), "run_id": "runB"})

    ids_a = [row["request_id"] for row in result_a["requests"]]
    ids_b = [row["request_id"] for row in result_b["requests"]]

    assert ids_a == ["req-A"], f"runA saw {ids_a}"
    assert ids_b == ["req-B"], f"runB saw {ids_b}"


def test_a_legacy_record_still_finds_its_requests(server, tmp_path: Path) -> None:
    """No `queue` field: the shared directory is used, so pre-change runs keep working."""
    work = tmp_path / "work"
    runs = work / "plan" / "mcp-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "old.json").write_text(
        json.dumps(
            {"run_id": "old", "work": str(work), "out": str(work / "box"), "pid": 1,
             "returncode": None},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_request(work / "plan" / "host-model", "req-legacy")

    result = server.cbb_next_requests({"work": str(work), "run_id": "old"})
    ids = [row["request_id"] for row in result["requests"]]
    assert ids == ["req-legacy"]


def test_replying_into_a_run_scoped_queue_writes_there(server, tmp_path: Path) -> None:
    work = tmp_path / "work"
    _record(work, "runC", "plan/host-model/runC")
    _write_request(work / "plan" / "host-model" / "runC", "req-C")

    server.cbb_reply({"work": str(work), "run_id": "runC", "request_id": "req-C", "text": "ok"})

    written = work / "plan" / "host-model" / "runC" / "responses" / "req-C.json"
    assert written.is_file(), "reply did not land in the run's own queue"
    assert not (work / "plan" / "host-model" / "responses" / "req-C.json").exists()
