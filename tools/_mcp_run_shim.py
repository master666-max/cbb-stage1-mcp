"""Launch shim for the MCP pipeline server.

The MCP client kills the *process tree* of the server when a tool call returns.
A child spawned directly by the server is still in that tree, so the long-running
extraction gets killed seconds after ``cbb_start_run`` returns.

This shim is started through ``cmd /c start /b`` so it is re-parented away from
the server's tree. It then spawns the real pipeline, records the **pipeline's**
PID (``os.execv`` is not a true exec on Windows — it creates a new process and
exits, so the PID would change), and waits for it.

Exit-code handoff
-----------------
``child.wait()`` returns the pipeline's exit code to *this* process, and this
process then exits — so the code used to vanish with the shim, leaving
``plan/mcp-runs/<run_id>.json`` with ``returncode: null`` forever.  Every caller
(``cbb_status``) then had to guess success from ``state.json`` and always
reported "process exited; returncode unknown".  The shim now writes the code to
``<pidfile>.rc`` as a tiny JSON record the status tool can read.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: _mcp_run_shim.py <pidfile> <executable> [args...]", file=sys.stderr)
        return 2
    pidfile = Path(sys.argv[1])
    executable = sys.argv[2]
    args = [executable, *sys.argv[3:]]
    started = time.time()
    child = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
        close_fds=True,
    )
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    with pidfile.open("w", encoding="utf-8") as handle:
        handle.write(str(child.pid))
        handle.flush()
    returncode = child.wait()
    # Hand the exit code back to whoever polls the run record; the shim's own
    # return value dies with this process.
    exit_file = pidfile.with_suffix(pidfile.suffix + ".rc")
    exit_file.write_text(
        json.dumps(
            {
                "returncode": returncode,
                "pid": child.pid,
                "started": started,
                "finished": time.time(),
                "elapsed_sec": round(time.time() - started, 1),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
