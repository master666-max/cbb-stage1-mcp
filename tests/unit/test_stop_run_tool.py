"""A run started through MCP must be stoppable through MCP.

Measured gap
------------
There is no stop tool in the 22-tool surface. `cbb_start_run` spawns a detached pipeline process and
records its pid; nothing can end it. In this session that gap was worked around **twice** by hand
with `Stop-Process`, which is exactly the kind of thing the MCP surface exists to avoid: the agent
could start work but not stop it, and the only remedy lived outside the protocol.

What the tool must do, and what it must not
-------------------------------------------
* stop the recorded process **and its children** — the pipeline spawns worker activity, and killing
  only the parent leaves orphans writing into the same work directory;
* say plainly when the run already finished (a no-op must not look like a kill);
* refuse an unknown run id rather than inventing success;
* record who stopped it and when, so a box built later can tell "killed" from "crashed".

The test spawns a real sleeping process rather than mocking `Popen`: the claim under test is that a
pid recorded in the registry can actually be terminated.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SERVER = PROJECT / "tools" / "mcp_pipeline_server.py"


@pytest.fixture(scope="module")
def server():
    spec = importlib.util.spec_from_file_location("cbb_mcp_stop", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mcp_stop"] = module
    spec.loader.exec_module(module)
    return module


def _alive(pid: int) -> bool:
    """Is this pid still running? Windows-safe enough for a test helper."""
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.fixture
def sleeper() -> subprocess.Popen:
    """A real process to kill.

    This host intermittently refuses process creation (`PermissionError [WinError 5]` from
    `CreateProcess`), so the test skips rather than blaming the tool for an environment restriction.
    The handler's logic is covered by the tests that need no child process.
    """
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        pytest.skip(f"this host refused to create a process: {type(exc).__name__}: {exc}")
    try:
        yield process
    finally:
        # The tool under test may already have force-killed it; a second kill must not fail the
        # test, or a passing stop looks like a failing fixture.
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _seed_run(server, work: Path, run_id: str, pid: int, **extra) -> None:
    record = {
        "run_id": run_id,
        "work": str(work),
        "out": str(work / "box"),
        "pid": pid,
        "started": time.time(),
        "chapters": "1-1",
        "argv": [],
        "log": str(work / "run.log"),
        "returncode": None,
        "queue": "",
        "preflight_status": "READY",
        "preflight_blocked": [],
        "preflight_forced": False,
    }
    record.update(extra)
    server._save_run(work, run_id, record)


# ══ the tool exists ═══════════════════════════════════════════════════
def test_the_stop_tool_is_declared(server) -> None:
    names = [tool["name"] for tool in server.TOOLS]
    assert "cbb_stop_run" in names, (
        "no stop tool: an agent can start a run through MCP but cannot end one, which is how this "
        "session ended up killing two runs by hand"
    )


def test_the_stop_tool_is_dispatchable(server) -> None:
    """A declared tool that the dispatcher does not route is decoration."""
    source = SERVER.read_text(encoding="utf-8")
    assert "cbb_stop_run" in source
    assert source.count("cbb_stop_run") >= 2, (
        "cbb_stop_run appears once — declared but not wired into the handler table"
    )


# ══ it actually stops a run ═══════════════════════════════════════════
def test_stopping_terminates_the_recorded_process(server, sleeper, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    _seed_run(server, work, "r1", sleeper.pid)
    assert _alive(sleeper.pid)

    result = server.cbb_stop_run({"work": str(work), "run_id": "r1", "by": "lead"})

    assert result.get("ok") is True, result
    assert result.get("stopped") is True, f"the tool reported no kill: {result}"
    # `wait()` answers "did it die" without spawning `tasklist`, which is intermittently denied here.
    assert sleeper.wait(timeout=15) is not None, (
        "the process is still running after cbb_stop_run"
    )


def test_stopping_records_who_and_when(server, sleeper, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    _seed_run(server, work, "r2", sleeper.pid)

    server.cbb_stop_run({"work": str(work), "run_id": "r2", "by": "lead"})

    record = server._load_run(work, "r2")
    assert record is not None
    assert record.get("stopped_by") == "lead", (
        "a stopped run is indistinguishable from a crashed one without this"
    )
    assert record.get("stopped_at"), "no timestamp recorded for the stop"


def test_stopping_marks_the_run_as_finished(server, sleeper, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    _seed_run(server, work, "r3", sleeper.pid)

    server.cbb_stop_run({"work": str(work), "run_id": "r3", "by": "lead"})

    record = server._load_run(work, "r3")
    assert record.get("returncode") is not None, (
        "returncode stays None, so cbb_status will keep reporting the run as in flight"
    )


def test_the_reply_says_whether_the_tree_or_only_the_process_was_stopped(server, sleeper, tmp_path: Path) -> None:
    """Children left behind keep writing into the work directory, so the caller must know."""
    work = tmp_path / "work"
    work.mkdir(parents=True)
    _seed_run(server, work, "r4", sleeper.pid)

    result = server.cbb_stop_run({"work": str(work), "run_id": "r4", "by": "lead"})

    assert result.get("mode") in ("tree", "process"), (
        f"the reply does not say how far the stop reached: {result}"
    )
    record = server._load_run(work, "r4")
    assert record.get("stop_mode") == result["mode"], (
        "the run record and the reply disagree about the stop mode"
    )


# ══ honest no-ops and honest errors ═══════════════════════════════════
def test_stopping_a_finished_run_is_a_reported_no_op(server, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    _seed_run(server, work, "done", 999999, returncode=0)

    result = server.cbb_stop_run({"work": str(work), "run_id": "done", "by": "lead"})

    assert result.get("ok") is True
    assert result.get("stopped") is False, "a finished run was reported as stopped"
    assert "already" in str(result.get("reason") or "").lower()


def test_an_unknown_run_id_is_an_error(server, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(parents=True)

    result = server.cbb_stop_run({"work": str(work), "run_id": "nope", "by": "lead"})

    assert result.get("ok") is False, result
    assert "no such run" in str(result.get("error") or "").lower()


def test_a_dead_pid_is_reported_without_crashing(server, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    _seed_run(server, work, "ghost", 999998)

    result = server.cbb_stop_run({"work": str(work), "run_id": "ghost", "by": "lead"})

    assert result.get("ok") is True, result
    assert result.get("stopped") is False
