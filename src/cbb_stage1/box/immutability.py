"""Make a published box read-only, and be able to prove it.

"Immutable" meant only that `build_box` refuses to overwrite an existing path — a convention inside our
own code. Once anything else writes into the tree, the guarantee is gone with no way to tell.

Marking is done bottom-up (files first, then directories) because a read-only directory blocks the
chmod of its own children on some systems. Verification re-stats every entry: the Windows read-only
attribute is advisory, and a directory's flag does not protect what is inside it, so "we called chmod"
is not evidence.

Failure is reported, never raised. A box that exists and is writable is more useful than no box; what
must not happen is a box that calls itself immutable while being writable.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ImmutabilityReport",
    "is_read_only",
    "make_read_only",
    "make_writable",
]


@dataclass
class ImmutabilityReport:
    marked: int = 0
    unmarked: int = 0
    verified: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "marked": self.marked,
            "unmarked": self.unmarked,
            "verified": self.verified,
            "errors": list(self.errors),
        }


def _entries(root: Path) -> list[Path]:
    """Every entry under ``root``, files before their parent directories."""
    files: list[Path] = []
    dirs: list[Path] = []
    for current, dirnames, filenames in os.walk(root):
        for name in filenames:
            files.append(Path(current) / name)
        for name in dirnames:
            dirs.append(Path(current) / name)
    # Deepest directories last so a child is handled before its parent.
    dirs.sort(key=lambda path: len(path.parts), reverse=True)
    return files + dirs + [root]


def _set_writable(path: Path, writable: bool) -> None:
    """Clear or set the owner write bit, and the Windows read-only attribute."""
    mode = os.stat(path).st_mode
    if writable:
        os.chmod(path, mode | stat.S_IWRITE)
    else:
        os.chmod(path, mode & ~stat.S_IWRITE)


def make_read_only(root: str | Path) -> ImmutabilityReport:
    """Mark every entry under ``root`` read-only, then verify.

    Never raises: a refused chmod is recorded in ``errors`` and leaves ``verified`` False, because the
    caller has to be able to tell "marked" from "asked the OS to mark".
    """
    path_root = Path(root)
    report = ImmutabilityReport()
    if not path_root.exists():
        report.errors.append(f"no such path: {path_root}")
        return report

    for entry in _entries(path_root):
        try:
            _set_writable(entry, False)
            report.marked += 1
        except OSError as exc:
            report.errors.append(f"{entry}: {type(exc).__name__}: {exc}")

    report.verified = is_read_only(path_root)
    if not report.verified:
        for entry in _entries(path_root):
            try:
                if os.stat(entry).st_mode & stat.S_IWRITE:
                    report.unmarked += 1
            except OSError:
                continue
    return report


def make_writable(root: str | Path) -> None:
    """Lift the marking. An immutability that cannot be lifted is a bug, not a safeguard."""
    path_root = Path(root)
    if not path_root.exists():
        return
    for entry in _entries(path_root):
        try:
            _set_writable(entry, True)
        except OSError:
            continue


def is_read_only(root: str | Path) -> bool:
    """True when nothing under ``root`` is writable by its owner."""
    path_root = Path(root)
    if not path_root.exists():
        return False
    for entry in _entries(path_root):
        try:
            if os.stat(entry).st_mode & stat.S_IWRITE:
                return False
        except OSError:
            return False
    return True
