"""Thin in-process wrapper around the canonical box_lint.py.

The original wrapper used subprocess pipes; under pytest capture on Windows that
can hit WinError 5. Importing the module and redirecting stdout keeps the same
CLI contract without spawning a process.

Two entry points:

* :func:`lint_box` -- the historical contract (``[]`` on pass, a list of error
  strings on failure). Used by the CLI/MCP surface and the ``example`` test.
* :func:`canonical_lint_errors` -- same, kept here so the box writer's measured
  hard gates can call the canonical implementation without re-implementing it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path


def _load_box_lint(root: str | Path | None = None):
    """Import the canonical validator, resolved by chain.

    It used to be `parents[4] / 设计 / 抽取段-v6 / tools / box_lint.py` — a sibling of
    the project, which a published checkout does not have. The canonical lint is one of
    the box writer's hard gates, so that path made every build in a clone fail.
    """
    from cbb_stage1.schemas import find_script

    if root is None:
        root = Path(__file__).resolve().parents[3]
    found = find_script("box_lint.py", root)
    script = found if found is not None else (
        Path(__file__).resolve().parents[3]
        / "设计" / "抽取段-v6" / "tools" / "box_lint.py"
    )
    if not script.exists():
        return None, script
    spec = importlib.util.spec_from_file_location("_cbb_box_lint", script)
    if spec is None or spec.loader is None:
        return None, script
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, script


def _errors_from_output(output: str) -> list[str]:
    errors = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("[x]")
    ]
    # box_lint prints at most 80 details. When more were counted, surface that
    # so the writer's gate classification cannot silently miss a failure.
    match = re.search(r"FAIL：(\d+) 个错误", output)
    if match and int(match.group(1)) > len(errors):
        errors.append(
            f"box_lint: {int(match.group(1))} 个错误，只打印了 {len(errors)} 条"
        )
    return errors


def run_box_lint(box: Path, *extra_argv: str, quiet: bool = False) -> tuple[int, str]:
    """Run the canonical validator and return ``(returncode, captured stdout)``."""
    box = Path(box).resolve()
    module, script = _load_box_lint()
    if module is None:
        return 1, f"box_lint.py not found: {script}"
    old_argv = sys.argv
    buf = io.StringIO()
    try:
        sys.argv = [str(script), str(box), *extra_argv]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = int(module.main())
    finally:
        sys.argv = old_argv
    out = buf.getvalue()
    if not quiet:
        print(out, end="")
    return rc, out


def lint_box(box: Path) -> list[str]:
    rc, out = run_box_lint(box)
    if rc == 0:
        return []
    return _errors_from_output(out) or [f"box_lint rc={rc}"]


def canonical_lint_errors(box: Path, *, quiet: bool = False) -> list[str]:
    """Canonical validator errors (``[]`` when the box passes every check)."""
    rc, out = run_box_lint(box, quiet=quiet)
    if rc == 0:
        return []
    return _errors_from_output(out) or [f"box_lint rc={rc}"]
