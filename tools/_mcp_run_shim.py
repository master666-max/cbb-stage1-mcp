"""Launch shim for the MCP pipeline server.

The MCP client kills the *process tree* of the server when a tool call returns. A child
spawned directly by the server is still in that tree, so the long-running extraction gets
killed seconds after ``cbb_start_run`` returns. This shim is started through
``cmd /c start /b`` so it is re-parented away from the server's tree.

What this file is responsible for
---------------------------------
It is the only place that knows how a pipeline process ended, so it must survive the ways a
pipeline can end — including the ways it does not end cleanly:

* ``child.wait()`` returns the exit code to *this* process, and this process then exits, so the
  code used to vanish with the shim and ``plan/mcp-runs/<run_id>.json`` kept
  ``returncode: null`` forever. It is written to ``<pidfile>.rc``.
* That write is in a ``finally``. Measured failure: a run died during ``claims`` leaving a
  ``.pid`` and **no** ``.rc``, and the only conclusion available was "the shim itself died".
  Now a non-zero child exit always leaves a record; only an external kill can leave none.
* **stderr gets its own file** (``<pidfile>.stderr.log``). It used to share the progress log,
  where a traceback sits among thousands of progress lines and is effectively lost.
* **A heartbeat** (``<pidfile>.heartbeat``, written every few seconds while waiting) is what
  distinguishes "the child crashed" from "the shim was killed": a killed shim writes neither a
  heartbeat nor an ``.rc``, and that pair of facts is the diagnosis.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

#: How often the shim proves it is still alive while the pipeline runs.
HEARTBEAT_SECONDS = 5.0


def _companion(pidfile: Path, suffix: str) -> Path:
    return pidfile.with_suffix(pidfile.suffix + suffix)


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write the exit record atomically: a reader must never see a half-written one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: _mcp_run_shim.py <pidfile> <executable> [args...]", file=sys.stderr)
        return 2

    pidfile = Path(sys.argv[1])
    executable = sys.argv[2]
    args = [executable, *sys.argv[3:]]
    started = time.time()

    pidfile.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = _companion(pidfile, ".stderr.log")
    heartbeat_path = _companion(pidfile, ".heartbeat")
    exit_path = _companion(pidfile, ".rc")

    stop = threading.Event()

    def heartbeat() -> None:
        """Prove the shim is alive, so a missing `.rc` can be read as "killed"."""
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                heartbeat_path.write_text(f"{time.time():.3f}\n", encoding="utf-8")
            except OSError:
                return

    # stderr goes to a file of its own; stdout stays inherited (it is the progress log).
    with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_handle:
        child = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=stderr_handle,
            close_fds=True,
        )
        pidfile.write_text(str(child.pid), encoding="utf-8")
        try:
            heartbeat_path.write_text(f"{started:.3f}\n", encoding="utf-8")
        except OSError:
            pass
        beat = threading.Thread(target=heartbeat, name="cbb-shim-heartbeat", daemon=True)
        beat.start()

        # Bound before the try so the `finally` record can be written even when `wait()`
        # raises — referencing an unassigned name inside the handler would replace the real
        # interruption with an UnboundLocalError and lose the record again.
        returncode: int | None = None
        try:
            returncode = child.wait()
        finally:
            # Any path out of the wait still records an exit, including KeyboardInterrupt and
            # an exception raised by wait() itself. Only an external kill skips this, which is
            # exactly what the heartbeat is there to tell apart.
            stop.set()
            _write_json_atomic(
                exit_path,
                {
                    # `child.returncode` is preferred (it survives an exception from wait()),
                    # with the local as the fallback for objects that only expose wait().
                    "returncode": getattr(child, "returncode", returncode),
                    "pid": child.pid,
                    "started": started,
                    "finished": time.time(),
                    "elapsed_sec": round(time.time() - started, 1),
                },
            )

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
