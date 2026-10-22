"""Two MCP tools that were lying by omission.

`cbb_build_box` — a failed build reported only the exception message
-------------------------------------------------------------------
`BuildError` carries the underlying gate errors as a second argument, and the tool caught
the exception and returned ``{"error": str(exc)}`` — dropping ``exc.args[1]``. A real failure
came back as:

    canonical box_lint gate failed
    box_lint rc=1

with no indication of which table or row was wrong, so the caller could not act on it. The
detail existed the whole time; it just never left the process.

`cbb_model_drain` — "pending" included requests that were already answered
--------------------------------------------------------------------------
`HostQueueClient.drain()` lists every file in ``requests/`` without checking ``responses/``.
On a real work directory it reported 16 pending requests of which 14 already had responses,
so an agent following it would answer the same requests twice. ``cbb_next_requests`` filters
correctly because it compares against ``responses/`` — the fix is to make `drain` do what its
name promises.
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
    spec = importlib.util.spec_from_file_location("cbb_mcp_errors", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_errors"] = module
    spec.loader.exec_module(module)
    return module


# ══ cbb_build_box — checked, and NOT a defect ═══════════════════════════
#
# I originally reported that `cbb_build_box` swallowed the gate detail because it returns
# `{"error": str(exc)}` and drops `exc.args[1]`. Reading `BuildError` shows the detail is
# already inlined into the message:
#
#     detail = message if not self.errors else f"{message}\n" + "\n".join(self.errors)
#     super().__init__(detail)
#
# so `str(exc)` is `message + "\n" + details`, and `args[1]` does not exist to drop. Three
# tests were written for this, all passed against unfixed code, and have been deleted: a
# test that passes before the fix is not a test.
#
# The message from the real run — "canonical box_lint gate failed / box_lint rc=1" — is
# raised elsewhere and its detail is the return code itself.


# ══ drain must not report answered requests ══════════════════════════════
def _queue(root: Path) -> Path:
    (root / "requests").mkdir(parents=True)
    (root / "responses").mkdir(parents=True)
    return root


def test_drain_skips_requests_that_already_have_responses(tmp_path: Path) -> None:
    from cbb_stage1.agents.host_queue import HostQueueClient

    root = _queue(tmp_path / "queue")
    for rid in ("r1", "r2", "r3"):
        (root / "requests" / f"{rid}.json").write_text(
            json.dumps({"request_id": rid, "prompt": "x"}), encoding="utf-8"
        )
    # r1 and r3 answered; r2 still waiting
    for rid in ("r1", "r3"):
        (root / "responses" / f"{rid}.json").write_text(
            json.dumps({"request_id": rid, "text": "ok"}), encoding="utf-8"
        )

    pending = list(HostQueueClient(root).drain())
    ids = [row["request_id"] for row in pending]
    assert ids == ["r2"], f"drain reported answered requests as pending: {ids}"


def test_drain_reports_everything_when_nothing_is_answered(tmp_path: Path) -> None:
    from cbb_stage1.agents.host_queue import HostQueueClient

    root = _queue(tmp_path / "queue")
    for rid in ("a", "b"):
        (root / "requests" / f"{rid}.json").write_text(
            json.dumps({"request_id": rid}), encoding="utf-8"
        )
    ids = sorted(row["request_id"] for row in HostQueueClient(root).drain())
    assert ids == ["a", "b"]


def test_drain_keeps_oldest_first(tmp_path: Path) -> None:
    """Ordering is part of the contract: the agent should answer in arrival order."""
    from cbb_stage1.agents.host_queue import HostQueueClient

    root = _queue(tmp_path / "queue")
    for index, rid in enumerate(("c", "a", "b")):
        (root / "requests" / f"{rid}.json").write_text(
            json.dumps({"request_id": rid}), encoding="utf-8"
        )
    ids = [row["request_id"] for row in HostQueueClient(root).drain()]
    assert ids == sorted(ids), f"not in file order: {ids}"
