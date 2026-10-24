"""Two consumers on one queue must not duplicate work, and the schema must be callable.

Measured with three subagents over MCP on a 12-request queue:

    requests 12, responses 12, consumer touches 15  ->  3 duplicated (25%)
    req-009/010/011 each carried TWO consumer traces
    both `cbb_reply` calls for the same request returned {"accepted": true}

Two independent defects:

1. **No claim.** `cbb_next_requests` lists whatever has no response file yet. Between the
   moment a consumer takes a request and the moment it replies, the request is still "pending"
   to everyone else, so a second consumer takes it too. On a *verifier* request — where the
   answer is a judgement rather than a derived string — the two consumers can answer
   differently and the last write wins silently. That is the failure this file prevents.

2. **Schema cannot be called.** `_resolve_run` requires `work`
   ("work is required (run state lives under <work>/plan/mcp-runs)"), but `cbb_status`,
   `cbb_next_requests` and `cbb_reply` do not declare `work` at all — a consumer that follows
   the published contract fails on its first call. One did, during the measurement.

The fix keeps `cbb_next_requests` non-destructive but makes the take exclusive: a returned
request is marked in `claims/`, skipped by other consumers until the claim expires (so a
crashed consumer does not strand it), and a reply still settles it for good. `cbb_reply` keeps
the **first** answer and reports the duplicate rather than overwriting: with two consumers in
play, "last write wins" makes the outcome depend on scheduling.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SERVER = PROJECT / "tools" / "mcp_pipeline_server.py"
RUN_ID = "claimtest1"


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("cbb_mcp_claims", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_claims"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def queue(tmp_path: Path, monkeypatch, server):
    """A throwaway work dir with a run record and `count` pending requests."""
    work = tmp_path / "work"
    (work / "plan" / "mcp-runs").mkdir(parents=True)
    (work / "plan" / "mcp-runs" / f"{RUN_ID}.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "work": str(work),
                "out": str(work / "box"),
                "pid": 1,
                "returncode": None,
            }
        ),
        encoding="utf-8",
    )
    requests = work / "plan" / "host-model" / "requests"
    responses = work / "plan" / "host-model" / "responses"
    requests.mkdir(parents=True)
    responses.mkdir(parents=True)

    for index in range(_REQUEST_COUNT):
        request_id = f"req-{index:03d}"
        (requests / f"{request_id}.json").write_text(
            json.dumps({"request_id": request_id, "prompt": f"q{index}"}),
            encoding="utf-8",
        )
    return work


_REQUEST_COUNT = 12


def _next(server, work: Path, limit: int = 100) -> list[str]:
    payload = server.cbb_next_requests({"work": str(work), "run_id": RUN_ID, "limit": limit})
    return [row["request_id"] for row in payload["requests"]]


def _reply(server, work: Path, request_id: str, text: str) -> dict:
    return server.cbb_reply(
        {"work": str(work), "run_id": RUN_ID, "request_id": request_id, "text": text}
    )


# ══ 1. the schema must describe what the handler requires ══════════════
@pytest.mark.parametrize("tool_name", ["cbb_status", "cbb_next_requests", "cbb_reply"])
def test_run_scoped_tools_declare_work(server, tool_name: str) -> None:
    tool = next(item for item in server.TOOLS if item["name"] == tool_name)
    schema = tool["inputSchema"]
    assert "work" in (schema.get("properties") or {}), f"{tool_name} does not document `work`"
    assert "work" in (schema.get("required") or []), f"{tool_name} does not require `work`"


def test_the_schema_matches_the_runtime_error(server, tmp_path: Path) -> None:
    """The message the consumer actually hit, kept as a contract test."""
    _record, error = server._resolve_run({"run_id": "x"})
    assert error is not None
    assert "work is required" in error["error"]


# ══ 2. taking a request is exclusive ═══════════════════════════════════
def test_two_consumers_get_disjoint_requests(server, queue: Path) -> None:
    first = _next(server, queue)
    second = _next(server, queue)
    assert first, "the first consumer got nothing"
    assert not set(first) & set(second), f"both consumers got {sorted(set(first) & set(second))}"


def test_a_claimed_request_comes_back_after_the_claim_expires(
    server, queue: Path, monkeypatch
) -> None:
    """A consumer that dies must not strand its requests forever."""
    first = _next(server, queue)
    assert first

    # pretend the claim was taken a long time ago
    claims = queue / "plan" / "host-model" / "claims"
    for path in claims.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["claimed_at"] = time.time() - 10_000
        path.write_text(json.dumps(row), encoding="utf-8")

    again = _next(server, queue)
    assert sorted(again) == sorted(first), f"expired claims did not return: {again}"


def test_an_answered_request_is_never_returned(server, queue: Path) -> None:
    first = _next(server, queue)
    target = first[0]
    _reply(server, queue, target, "ACK")
    again = _next(server, queue)
    assert target not in again


def test_claiming_is_atomic_under_threads(server, queue: Path) -> None:
    """The real case: several subagents polling at once must not get the same request."""
    results: list[list[str]] = []
    lock = threading.Lock()

    def consumer() -> None:
        got = _next(server, queue)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=consumer) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    flat = [request_id for group in results for request_id in group]
    duplicates = {rid for rid in flat if flat.count(rid) > 1}
    assert not duplicates, f"two consumers claimed the same request(s): {sorted(duplicates)}"
    assert len(flat) == _REQUEST_COUNT, f"expected every request claimed exactly once, got {len(flat)}"


# ══ 3. a duplicate reply must not silently overwrite ══════════════════
def test_a_duplicate_reply_keeps_the_first_answer(server, queue: Path) -> None:
    first = _next(server, queue)
    target = first[0]
    _reply(server, queue, target, "FIRST")

    second = _reply(server, queue, target, "SECOND")
    assert second.get("duplicate") is True, second

    stored = json.loads(
        (queue / "plan" / "host-model" / "responses" / f"{target}.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["text"] == "FIRST", "a second consumer overwrote the first answer"


def test_a_duplicate_reply_reports_both_texts(server, queue: Path) -> None:
    """A reader must see the conflict, not just that one was dropped."""
    first = _next(server, queue)
    target = first[0]
    _reply(server, queue, target, "FIRST")
    second = _reply(server, queue, target, "SECOND")
    assert second.get("kept") == "FIRST", second
    assert second.get("rejected") == "SECOND", second


def test_a_first_reply_is_not_flagged_as_duplicate(server, queue: Path) -> None:
    first = _next(server, queue)
    payload = _reply(server, queue, first[0], "ACK")
    assert payload.get("duplicate") in (None, False), payload


# ══ 4. the claim does not change the answered semantics ═══════════════
def test_drain_reports_only_unanswered(server, queue: Path, monkeypatch) -> None:
    first = _next(server, queue)
    _reply(server, queue, first[0], "ACK")
    drained = server.cbb_model_drain({"work": str(queue)})
    ids = [row["request_id"] for row in drained["data"]["pending"]]
    assert first[0] not in ids
