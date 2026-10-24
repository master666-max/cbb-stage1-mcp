"""An "immutable" box that stays writable is only immutable by convention.

Measured gap (E9)
-----------------
`box/writer.py:27` says "Boxes are immutable — `build_box` refuses to overwrite an existing path", and
that refusal is real. But nothing makes the published tree read-only: no `chmod`, no read-only
attribute. So the guarantee is "our own code will not overwrite it", which is not what immutable means
once anything else — a later run, a script, a human — writes into it.

What the fix must do
--------------------
* after publishing, mark the tree read-only (on Windows: clear the write bit; elsewhere: drop the write
  permission);
* **verify** the marking worked, because on Windows the read-only attribute is advisory and a
  directory's own flag does not protect its children;
* offer an explicit way to make it writable again for the "I know what I'm doing" case — an
  immutability you cannot lift is a bug, not a safeguard;
* never fail the build because the OS refused the marking: report it, because a box that exists and is
  writable is better than no box, but it must not claim to be immutable.

The tests use a real temporary tree; no mocking of the filesystem, since the claim is about the
filesystem.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from cbb_stage1.box.immutability import make_read_only, make_writable, is_read_only


def _tree(root: Path) -> Path:
    (root / "knowledge").mkdir(parents=True)
    (root / "knowledge" / "assertions.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    (root / "manifest.json").write_text('{"b":2}\n', encoding="utf-8")
    return root


def _writable(path: Path) -> bool:
    return bool(os.stat(path).st_mode & stat.S_IWRITE)


# ══ marking ══════════════════════════════════════════════════════════
def test_files_become_read_only(tmp_path: Path) -> None:
    root = _tree(tmp_path / "box")
    report = make_read_only(root)
    assert report.marked >= 2, f"not every file was marked: {report.as_dict()}"
    assert not _writable(root / "knowledge" / "assertions.jsonl")
    assert not _writable(root / "manifest.json")


def test_directories_become_read_only_too(tmp_path: Path) -> None:
    """A writable directory lets anyone create files in an otherwise read-only tree."""
    root = _tree(tmp_path / "box")
    make_read_only(root)
    assert not _writable(root / "knowledge")


def test_the_report_says_what_was_marked(tmp_path: Path) -> None:
    root = _tree(tmp_path / "box")
    report = make_read_only(root)
    payload = report.as_dict()
    assert payload["marked"] >= 2
    assert payload["verified"] is True, f"the marking was not verified: {payload}"


# ══ verification is part of it ═══════════════════════════════════════
def test_is_read_only_agrees_with_the_marking(tmp_path: Path) -> None:
    root = _tree(tmp_path / "box")
    assert is_read_only(root) is False
    make_read_only(root)
    assert is_read_only(root) is True


def test_a_writable_child_is_detected(tmp_path: Path) -> None:
    root = _tree(tmp_path / "box")
    make_read_only(root)
    # Something re-enabled writes on one file.
    os.chmod(root / "manifest.json", stat.S_IWRITE | stat.S_IREAD)
    assert is_read_only(root) is False, (
        "a writable file inside the tree was not detected, so 'immutable' cannot be verified"
    )


# ══ it can be lifted on purpose ══════════════════════════════════════
def test_writes_are_possible_again_after_lifting(tmp_path: Path) -> None:
    root = _tree(tmp_path / "box")
    make_read_only(root)
    make_writable(root)
    (root / "manifest.json").write_text('{"c":3}\n', encoding="utf-8")
    assert (root / "manifest.json").read_text(encoding="utf-8") == '{"c":3}\n'


def test_lifting_restores_the_tree(tmp_path: Path) -> None:
    root = _tree(tmp_path / "box")
    make_read_only(root)
    make_writable(root)
    assert is_read_only(root) is False


# ══ honest failure ═══════════════════════════════════════════════════
def test_a_missing_tree_is_reported_not_raised(tmp_path: Path) -> None:
    report = make_read_only(tmp_path / "nope")
    assert report.marked == 0
    assert report.errors, "a missing tree produced no error report"


def test_the_report_is_json_serialisable(tmp_path: Path) -> None:
    import json

    root = _tree(tmp_path / "box")
    json.dumps(make_read_only(root).as_dict(), ensure_ascii=False)
