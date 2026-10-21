"""Portability: the suite must pass on a Windows 11 machine that is not this one.

Everything here was found by scanning the tree, not by assuming. Each check encodes a
concrete way the project could fail to move:

  * an absolute path or a username baked into source;
  * a version gate that disagrees with what pip will install;
  * an endpoint with no environment override;
  * a test that reads a location only this machine has.

The checks are deliberately narrow: a broad "no drive letters anywhere" rule would flag
test data such as ``"C:\\\\a b\\\\python.exe"``, which is a parsing fixture, not a path.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # cbb_stage1_v6
SRC = ROOT / "src"

_SKIP = {"__pycache__", ".venv", ".git", ".pytest_cache", ".ruff_cache"}


def _python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in _SKIP for part in path.parts):
            continue
        yield path


# ── source: no machine-specific literals ────────────────────────────────────
def test_source_has_no_absolute_drive_paths() -> None:
    pattern = re.compile(r"""["'][A-Za-z]:[\\/]""")
    offenders = [
        f"{path.relative_to(ROOT)}:{line_no}"
        for path in _python_files(SRC)
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if pattern.search(line)
    ]
    assert offenders == [], f"absolute paths in source: {offenders}"


def test_source_has_no_hardcoded_user_profile() -> None:
    """The building machine's username must not leak into the package.

    The check reads the CURRENT user instead of hardcoding one. The first version
    pinned this machine's username into the test, which meant the test directory itself
    was one of the files that leaked it — caught by the publish rescan, not by the gate.
    """
    import os
    from pathlib import Path as _P

    names = {_P.home().name, os.environ.get("USERNAME") or "", os.environ.get("USER") or ""}
    names = {name for name in names if name and len(name) > 2}
    offenders = [
        f"{path.relative_to(ROOT)}:{line_no}"
        for path in _python_files(SRC)
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if any(name in line for name in names)
    ]
    assert offenders == [], f"hardcoded user profile in source: {offenders}"


def test_source_has_no_desktop_directory_references() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{line_no}"
        for path in _python_files(SRC)
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if re.search(r"Desktop|桌面", line)
    ]
    assert offenders == [], f"desktop references in source: {offenders}"


# ── version gate must agree with the installer ──────────────────────────────
def test_preflight_python_floor_matches_pyproject() -> None:
    """A preflight that accepts a version pip refuses reports READY for a broken env."""
    from cbb_stage1.preflight import MIN_PYTHON

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*">=\s*(\d+)\.(\d+)"', pyproject)
    assert match, "requires-python not declared in pyproject.toml"
    declared = (int(match.group(1)), int(match.group(2)))
    assert MIN_PYTHON == declared, (
        f"preflight says {MIN_PYTHON}, pyproject says {declared}"
    )


# ── endpoints must be overridable ───────────────────────────────────────────
def test_every_localhost_default_has_an_env_override() -> None:
    """A fresh machine may run the model server elsewhere.

    Two shapes are legitimate and both must pass:

      * the literal appears inline beside an ``os.environ`` read;
      * the literal is a module-level constant (``DEFAULT_EMBEDDING_BASE_URL``) that is
        consumed elsewhere as a fallback behind an env read. Checking only the second
        line's neighbourhood would flag the constant and miss the real question, so the
        constant's name is followed to its use sites.
    """
    offenders: list[str] = []
    for path in _python_files(SRC):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if not re.search(r"127\.0\.0\.1|localhost", line):
                continue
            window = " ".join(lines[max(0, index - 4): index + 2])
            if "os.environ" in window or "CBB_" in window or "base_url" in window:
                continue
            # module-level constant: follow it to where it is used
            constant = re.match(r"^([A-Z][A-Z0-9_]*)\s*=", line.strip())
            if constant:
                name = constant.group(1)
                uses = [
                    other
                    for other in lines
                    if name in other and not other.strip().startswith(name)
                ]
                if any("os.environ" in use or "CBB_" in use for use in uses):
                    continue
            offenders.append(f"{path.relative_to(ROOT)}:{index + 1}: {line.strip()[:80]}")
    assert offenders == [], f"localhost endpoints without an override: {offenders}"


def test_the_model_server_port_is_configured_not_assumed() -> None:
    from cbb_stage1 import preflight

    assert hasattr(preflight, "DEFAULT_EMBEDDING_BASE_URL")
    source = (SRC / "cbb_stage1" / "preflight.py").read_text(encoding="utf-8")
    assert "CBB_EMBEDDING_BASE_URL" in source, "the embedding endpoint must be overridable"


# ── tests: no absolute fixture paths ───────────────────────────────────────
def test_no_test_reads_an_absolute_path() -> None:
    """A test bound to `D:\\...` passes here and fails everywhere else.

    Only `Path("<drive>:...")` and `Path(r"<drive>:...")` count. A bare string literal
    is left alone: `test_mcp_server.py` uses ``"C:\\\\a b\\\\python.exe"`` as parse input,
    which is data rather than a location.
    """
    pattern = re.compile(r"""Path\(\s*r?["'][A-Za-z]:[\\/]""")
    offenders = [
        f"{path.relative_to(ROOT)}:{line_no}"
        for path in _python_files(ROOT / "tests")
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if pattern.search(line)
    ]
    assert offenders == [], f"tests bound to an absolute path: {offenders}"


def test_the_venv_is_not_committed() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".venv", "__pycache__", ".pytest_cache"):
        assert entry in gitignore, f"{entry} not ignored"


# ── Chinese paths and encoding ─────────────────────────────────────────────
#: Wrappers that take the text and pass an explicit encoding themselves. Verified by
#: reading them, not assumed: `box/atomic.py:write_text` calls
#: `path.write_text(text, encoding="utf-8", newline="\n")`.
_ENCODING_HELPERS = ("atomic.write_text", "atomic.write_json", "atomic.write_jsonl")


def test_the_encoding_helpers_really_supply_an_encoding() -> None:
    """If this ever stops being true, the next test's allow-list becomes a lie."""
    helper = (SRC / "cbb_stage1" / "box" / "atomic.py").read_text(encoding="utf-8")
    for name in ("write_text", "write_json", "write_jsonl"):
        assert f"def {name}(" in helper, f"{name} missing from atomic"
    assert 'encoding="utf-8"' in helper, "atomic no longer pins utf-8"


def _call_span(lines: list[str], index: int) -> str:
    """The whole call starting at ``index``, not a fixed four-line window.

    A ``write_text`` call can span twenty lines — the lexicon bootstrap writes a
    nested dict — and the ``encoding=`` argument lands on the last one. A fixed
    window made this gate report a false positive on code that does pin utf-8, and
    a gate that cries wolf gets ignored, which is worse than no gate.
    """
    indent = len(lines[index]) - len(lines[index].lstrip())
    span: list[str] = []
    for line in lines[index:]:
        span.append(line)
        stripped = line.strip()
        if stripped in {")", "),", ");"} and (len(line) - len(line.lstrip())) <= indent:
            break
    return " ".join(span)


def test_text_io_declares_utf8() -> None:
    """Windows defaults to the ANSI code page; an undeclared read mangles Chinese."""
    offenders: list[str] = []
    for path in _python_files(SRC):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not re.search(r"\.(write_text|read_text)\(|\bopen\(", line):
                continue
            if any(helper in line for helper in _ENCODING_HELPERS):
                continue
            if "encoding=" in _call_span(lines, index):
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{index + 1}: {line.strip()[:80]}")
    assert offenders == [], f"text I/O without an encoding: {offenders}"