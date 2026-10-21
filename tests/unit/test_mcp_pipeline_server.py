from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "tools" / "mcp_pipeline_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("cbb_pipeline_mcp_under_test", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drive(module, requests):
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    real_in, real_out = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = stdin, stdout
    try:
        assert module.main() == 0
    finally:
        sys.stdin, sys.stdout = real_in, real_out
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_pipeline_server_exposes_agent_driving_tools():
    module = _load()
    replies = _drive(
        module,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
    )
    assert replies[0]["result"]["serverInfo"]["name"] == "cbb-pipeline-mcp"
    names = {tool["name"] for tool in replies[1]["result"]["tools"]}
    # The driving loop an agent needs: start -> service model calls -> build.
    assert {
        "cbb_start_run",
        "cbb_status",
        "cbb_next_requests",
        "cbb_reply",
        "cbb_build_box",
        "cbb_lint_box",
        "cbb_gate",
    } <= names


def test_pipeline_server_unknown_tool_is_an_error_not_a_fake_result():
    module = _load()
    replies = _drive(
        module,
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "cbb_nope", "arguments": {}},
            },
        ],
    )
    assert replies[1]["result"]["isError"] is True


def test_pipeline_server_start_run_writes_host_policy(tmp_path, monkeypatch):
    """Starting a run must declare host transport: the agent is the model host."""
    module = _load()

    launched: list[list[str]] = []

    class FakeStart:
        def __init__(self, argv, **kwargs):
            launched.append(list(argv))

    monkeypatch.setattr(module.subprocess, "Popen", FakeStart)
    monkeypatch.setattr(module, "_wait_for_pid_file", lambda path, timeout=15.0: 424242)
    work = tmp_path / "work"
    out = tmp_path / "box"
    book = tmp_path / "book.epub"
    book.write_bytes(b"PK\x03\x04 pretend epub payload")
    result = module.cbb_start_run(
        {
            "book": str(book),
            "book_id": "zj-v1-c01-03",
            "work": str(work),
            "out": str(out),
            "chapters": "1-3",
            # No model backend in a unit test; the preflight must be told to skip
            # it explicitly rather than silently passing.
            "preflight_skip": "embedding_backend,llm_endpoint,llm_credentials",
        }
    )
    assert result["run_id"]
    assert result["pid"] == 424242
    # Launched through the re-parenting shim, not as a direct child.
    assert launched and launched[0][:3] == ["cmd", "/c", "start"]
    policy = json.loads((work / "plan" / "model-policy.json").read_text(encoding="utf-8"))
    assert policy["transport"] == "host"
    assert policy["host_queue"] == "1"
    # Run state must persist on disk so a *new* server process can resume it.
    record = json.loads(
        (work / "plan" / "mcp-runs" / f"{result['run_id']}.json").read_text(encoding="utf-8")
    )
    assert record["pid"] == 424242
    assert record["chapters"] == "1-3"
    shim = work / "plan" / f"mcp-run-{result['run_id']}.cmd"
    assert shim.exists()
    # The shim is written in the OEM codepage (batch files are parsed that way).
    shim_text = shim.read_bytes().decode("mbcs")
    assert "CBB_HOST_MODEL_QUEUE" in shim_text
    # Unbuffered stdout: without -u the redirected log stayed 0 bytes for the
    # whole run, so cbb_status.log_tail was empty exactly when it mattered.
    assert "-u -m cbb_stage1.cli" in shim_text


def test_start_run_leaves_the_box_dir_absent_so_build_box_can_publish(
    tmp_path: Path, monkeypatch
) -> None:
    """A box is immutable, so ``build_box`` refuses an existing ``out``.

    ``cbb_start_run`` used to ``mkdir`` the out dir.  The pipeline never writes
    there (``cfg.out_dir`` is only read, for an incremental baseline), but the
    empty directory it left behind made every MCP-driven run die at the last
    step: ``FileExistsError: refusing to overwrite an existing box``.
    """
    module = _load()

    class FakeStart:
        def __init__(self, argv, **kwargs):
            pass

    monkeypatch.setattr(module.subprocess, "Popen", FakeStart)
    monkeypatch.setattr(module, "_wait_for_pid_file", lambda path, timeout=15.0: 424243)
    work = tmp_path / "work"
    out = tmp_path / "boxes" / "v1"
    book = tmp_path / "book.epub"
    book.write_bytes(b"PK\x03\x04 pretend epub payload")
    module.cbb_start_run(
        {
            "book": str(book),
            "book_id": "zj-v1-c01-03",
            "work": str(work),
            "out": str(out),
            "chapters": "1",
            "preflight_skip": "embedding_backend,llm_endpoint,llm_credentials",
        }
    )
    assert work.is_dir()
    # Parent exists (so the eventual build has somewhere to land) ...
    assert out.parent.is_dir()
    # ... but the box directory itself must not, or build_box can never publish.
    assert not out.exists()


def test_start_run_refuses_to_start_when_preflight_is_blocked(tmp_path, monkeypatch):
    """Evidence first: a blocked probe must stop the run, not surface mid-stage.

    Nothing used to check dependencies before a run — a dead embedding endpoint
    or an already-existing box directory only failed at the stage needing it.
    """
    module = _load()

    launched: list[list[str]] = []
    monkeypatch.setattr(module.subprocess, "Popen", lambda argv, **kw: launched.append(argv))
    monkeypatch.setattr(module, "_wait_for_pid_file", lambda path, timeout=15.0: 1)
    result = module.cbb_start_run(
        {
            "book": str(tmp_path / "missing.epub"),  # does not exist -> BLOCKED
            "book_id": "zj-v1-c01",
            "work": str(tmp_path / "work"),
            "out": str(tmp_path / "box"),
            "preflight_skip": "embedding_backend,llm_endpoint,llm_credentials",
        }
    )
    assert "run_id" not in result
    assert result["preflight"]["status"] == "BLOCKED"
    assert "book" in result["blocked"]
    assert launched == []  # nothing was spawned
    # The refusal must be inspectable, not a bare string.
    assert any(item["name"] == "book" for item in result["preflight"]["checks"])


def test_start_run_force_records_the_forced_preflight(tmp_path, monkeypatch):
    """A forced start is allowed, and the force is written down — not hidden."""
    module = _load()

    monkeypatch.setattr(module.subprocess, "Popen", lambda argv, **kw: None)
    monkeypatch.setattr(module, "_wait_for_pid_file", lambda path, timeout=15.0: 777)
    work = tmp_path / "work"
    out = tmp_path / "box"
    result = module.cbb_start_run(
        {
            "book": str(tmp_path / "missing.epub"),
            "book_id": "zj-v1-c01",
            "work": str(work),
            "out": str(out),
            "preflight_skip": "embedding_backend,llm_endpoint,llm_credentials",
            "force": True,
        }
    )
    assert result["run_id"]
    assert result["preflight"]["forced"] is True
    record = json.loads(
        (work / "plan" / "mcp-runs" / f"{result['run_id']}.json").read_text(encoding="utf-8")
    )
    assert record["preflight_forced"] is True
    assert record["preflight_blocked"] == ["book"]


def test_status_reads_the_exit_record_written_by_the_shim(tmp_path, monkeypatch):
    """The shim's exit code must survive, or every finished run reads 'unknown'."""
    module = _load()

    monkeypatch.setattr(module.subprocess, "Popen", lambda argv, **kw: None)
    monkeypatch.setattr(module, "_wait_for_pid_file", lambda path, timeout=15.0: 9001)
    book = tmp_path / "book.epub"
    book.write_bytes(b"payload")
    work = tmp_path / "work"
    out = tmp_path / "box"
    started = module.cbb_start_run(
        {
            "book": str(book),
            "book_id": "zj-v1-c01",
            "work": str(work),
            "out": str(out),
            "preflight_skip": "embedding_backend,llm_endpoint,llm_credentials",
        }
    )
    assert started["run_id"]
    # Simulate the shim finishing and dropping its <pidfile>.rc record.
    rc_path = work / "plan" / f"mcp-run-{started['run_id']}.pid.rc"
    rc_path.write_text(json.dumps({"returncode": 0, "elapsed_sec": 12.5}), encoding="utf-8")
    monkeypatch.setattr(module, "_pid_alive", lambda pid: False)
    status = module.cbb_status({"work": str(work), "run_id": started["run_id"]})
    assert status["returncode"] == 0
    assert status["exit_record"]["elapsed_sec"] == 12.5
    assert status["running"] is False


def test_status_aggregates_model_calls_and_log(tmp_path, monkeypatch):
    """cbb_status is the aggregation point: the call log is otherwise invisible."""
    module = _load()

    monkeypatch.setattr(module.subprocess, "Popen", lambda argv, **kw: None)
    monkeypatch.setattr(module, "_wait_for_pid_file", lambda path, timeout=15.0: 9002)
    book = tmp_path / "book.epub"
    book.write_bytes(b"payload")
    work = tmp_path / "work"
    out = tmp_path / "box"
    started = module.cbb_start_run(
        {
            "book": str(book),
            "book_id": "zj-v1-c01",
            "work": str(work),
            "out": str(out),
            "preflight_skip": "embedding_backend,llm_endpoint,llm_credentials",
        }
    )
    run_id = started["run_id"]
    calls = work / "plan" / "model-calls.jsonl"
    calls.write_text(
        "\n".join(
            [
                json.dumps({"kind": "chat", "role": "extraction", "status": "ok", "elapsed_ms": 10.0}),
                json.dumps({"kind": "chat", "role": "verifier", "status": "failed", "elapsed_ms": 90.0}),
                json.dumps({"kind": "embed", "role": "derive", "status": "ok", "elapsed_ms": 40000.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log = work / "plan" / f"mcp-run-{run_id}.log"
    log.write_text("[100.0%] Complete\n", encoding="utf-8")
    monkeypatch.setattr(module, "_pid_alive", lambda pid: True)
    status = module.cbb_status({"work": str(work), "run_id": run_id})
    summary = status["model_calls"]
    assert summary["total"] == 3
    assert summary["by_kind"] == {"chat": 2, "embed": 1}
    assert summary["failed"] == 1
    assert summary["elapsed_ms_total"] == 40100.0
    assert status["log"]["bytes"] > 0
    assert "Complete" in status["log_tail"]


def test_doctor_tool_is_registered_and_dispatched():
    """The preflight must be reachable through MCP, not only through the CLI."""
    module = _load()
    import inspect

    names = {tool["name"] for tool in module.TOOLS}
    assert "cbb_doctor" in names
    assert "cbb_doctor" in module._DISPATCH
    assert "force" in inspect.getsource(module.cbb_start_run)


def test_shim_hands_the_child_exit_code_back_next_to_the_pid_file(tmp_path, monkeypatch):
    """The shim used to keep ``child.wait()``'s code to itself and then exit.

    The exit code died with it, so every finished run read
    ``returncode: null`` and ``cbb_status`` always said "returncode unknown".
    """
    import importlib.util
    import sys

    shim_path = Path(__file__).resolve().parents[2] / "tools" / "_mcp_run_shim.py"
    spec = importlib.util.spec_from_file_location("cbb_run_shim_under_test", shim_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeChild:
        pid = 4321

        def wait(self) -> int:
            return 3

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FakeChild())
    pidfile = tmp_path / "run.pid"
    monkeypatch.setattr(sys, "argv", ["shim", str(pidfile), "python", "-m", "x"])
    assert module.main() == 3
    exit_file = pidfile.with_suffix(pidfile.suffix + ".rc")
    assert exit_file.exists()
    record = json.loads(exit_file.read_text(encoding="utf-8"))
    assert record["returncode"] == 3
    assert record["pid"] == 4321
    assert record["elapsed_sec"] >= 0
    assert pidfile.read_text(encoding="utf-8").strip() == "4321"

