"""The run shim must always leave an exit record, a stderr log, a pid, and a heartbeat.

Measured failure: a run died during `claims` leaving `plan/mcp-runs/<run_id>.json` with
`returncode: null`, a `.pid`, and **no `.rc`**. `cbb_status` reported exactly what that means —
"a missing file means the shim itself died" — and there was nothing else to go on: the log
stopped at a stage boundary and held no traceback.

Three defects, one of them the reason the diagnosis was impossible:

1. **`.rc` was written after `child.wait()`, outside any `finally`.** If the wait raised, or
   the shim died between the two points, the record vanished. It is now written from a
   `finally`, so every exit path that reaches Python records something.
2. **stderr shared the log with the pipeline's progress lines.** A traceback sits among
   thousands of progress lines and is effectively lost. It gets its own file.
3. **Nothing distinguished "the child crashed" from "the shim was killed".** A heartbeat does:
   no heartbeat and no `.rc` means the shim was killed from outside, which is a different
   problem with a different fix.

Scope of these tests, stated plainly
------------------------------------
They drive `main()` with `subprocess.Popen` replaced by a fake child, so they cover the shim's
own logic — the record, the stderr file, the pid, the heartbeat — and not real process death.
A real-spawn version was written first and passed in isolation, then failed with
`PermissionError: [WinError 5]` from `CreateProcess` once the suite had already spawned other
processes; the same shim spawned three times in a row outside pytest and wrote its `.rc` every
time, so that is a limit of this environment rather than of the shim. Simulating an external
kill is likewise out of scope: by design it is the one case that leaves no record, which is why
the heartbeat exists.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SHIM = PROJECT / "tools" / "_mcp_run_shim.py"


@pytest.fixture()
def shim():
    spec = importlib.util.spec_from_file_location("cbb_run_shim_under_test2", SHIM)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeChild:
    """Stands in for a pipeline process: records what it was given, returns a chosen code."""

    def __init__(self, returncode: int = 3) -> None:
        self.pid = 4321
        self.returncode = returncode
        self.stdout = None
        self.stderr = None

    def wait(self) -> int:
        return self.returncode


def _run(shim, monkeypatch, pidfile: Path, child: _FakeChild, argv_extra=None) -> int:
    captured: dict[str, object] = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        child.stdout = kwargs.get("stdout")
        child.stderr = kwargs.get("stderr")
        return child

    monkeypatch.setattr(shim.subprocess, "Popen", fake_popen)
    argv = ["shim", str(pidfile), "python", "-m", "cbb_stage1.cli"]
    if argv_extra:
        argv.extend(argv_extra)
    monkeypatch.setattr(sys, "argv", argv)
    code = shim.main()
    child.captured = captured  # type: ignore[attr-defined]
    return code


# ══ the exit record ════════════════════════════════════════════════════
def test_a_non_zero_exit_is_recorded_next_to_the_pid_file(shim, monkeypatch, tmp_path: Path) -> None:
    pidfile = tmp_path / "run.pid"
    assert _run(shim, monkeypatch, pidfile, _FakeChild(3)) == 3

    exit_file = pidfile.with_suffix(pidfile.suffix + ".rc")
    assert exit_file.is_file(), "no exit record for a failed run"
    record = json.loads(exit_file.read_text(encoding="utf-8"))
    assert record["returncode"] == 3, record
    assert record["pid"] == 4321, record
    assert record["elapsed_sec"] >= 0, record


def test_a_successful_exit_is_recorded_too(shim, monkeypatch, tmp_path: Path) -> None:
    pidfile = tmp_path / "run.pid"
    assert _run(shim, monkeypatch, pidfile, _FakeChild(0)) == 0
    record = json.loads(
        pidfile.with_suffix(pidfile.suffix + ".rc").read_text(encoding="utf-8")
    )
    assert record["returncode"] == 0


def test_the_record_is_written_even_when_wait_raises(shim, monkeypatch, tmp_path: Path) -> None:
    """The defect: a wait() that raised left no record at all and no way to tell why."""

    class ExplodingChild(_FakeChild):
        def wait(self) -> int:
            self.returncode = None
            raise KeyboardInterrupt("interrupted")

    pidfile = tmp_path / "run.pid"
    with pytest.raises(KeyboardInterrupt):
        _run(shim, monkeypatch, pidfile, ExplodingChild())

    exit_file = pidfile.with_suffix(pidfile.suffix + ".rc")
    assert exit_file.is_file(), "an interrupted wait left no record"
    assert json.loads(exit_file.read_text(encoding="utf-8"))["returncode"] is None


# ══ stderr gets its own file ═══════════════════════════════════════════
def test_stderr_is_redirected_to_its_own_file(shim, monkeypatch, tmp_path: Path) -> None:
    pidfile = tmp_path / "run.pid"
    child = _FakeChild(0)
    _run(shim, monkeypatch, pidfile, child)

    stderr_path = pidfile.with_suffix(pidfile.suffix + ".stderr.log")
    assert stderr_path.is_file(), "stderr has no file of its own"
    # the handle passed to the child must be that file, not the inherited stdout
    handle = getattr(child, "captured")["kwargs"]["stderr"]
    assert getattr(handle, "name", "") == str(stderr_path)


def test_stdout_stays_inherited(shim, monkeypatch, tmp_path: Path) -> None:
    """Progress must keep going to the caller's stdout, which is the run log."""
    pidfile = tmp_path / "run.pid"
    child = _FakeChild(0)
    _run(shim, monkeypatch, pidfile, child)
    assert getattr(child, "captured")["kwargs"]["stdout"] is sys.stdout


# ══ the pid and the heartbeat ══════════════════════════════════════════
def test_the_pid_file_records_the_child_pid(shim, monkeypatch, tmp_path: Path) -> None:
    pidfile = tmp_path / "run.pid"
    _run(shim, monkeypatch, pidfile, _FakeChild(0))
    assert pidfile.read_text(encoding="utf-8").strip() == "4321"


def test_a_heartbeat_is_written_before_waiting(shim, monkeypatch, tmp_path: Path) -> None:
    """An initial beat is what makes "no beat and no .rc" mean "killed", not "never started"."""
    pidfile = tmp_path / "run.pid"
    _run(shim, monkeypatch, pidfile, _FakeChild(0))
    heartbeat = pidfile.with_suffix(pidfile.suffix + ".heartbeat")
    assert heartbeat.is_file(), "no heartbeat written"
    float(heartbeat.read_text(encoding="utf-8").strip())  # parses as a timestamp


def test_the_heartbeat_interval_is_short_enough_to_matter(shim) -> None:
    """A beat every 5 seconds makes a killed shim obvious within seconds."""
    assert 0 < shim.HEARTBEAT_SECONDS <= 10


# ══ still a script ═════════════════════════════════════════════════════
def test_a_usage_error_is_reported(shim, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["shim"])
    assert shim.main() == 2
    assert "usage" in capsys.readouterr().err.lower()
