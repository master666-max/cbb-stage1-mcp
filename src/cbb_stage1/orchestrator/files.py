"""Filesystem helpers for the orchestration shell.

Work files are disposable. Every write that another stage (or a resumed run)
may read is atomic: temp file in the same directory + ``os.replace``. This
mirrors the crash-safety requirement in design ``04-执行模型-v6.1.md`` §4.3.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically (temp file + rename).

    Args:
        path: Destination file.
        text: UTF-8 text to persist.
        encoding: Text encoding (default UTF-8).

    Returns:
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def atomic_write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Serialize ``data`` to JSON and write it atomically."""
    payload = json.dumps(data, ensure_ascii=False, indent=indent, default=str) + "\n"
    return atomic_write_text(path, payload)


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` when missing or corrupt."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts; missing file yields ``[]``."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> Path:
    """Write ``records`` as JSONL atomically, one object per line."""
    lines = [json.dumps(record, ensure_ascii=False, default=str) for record in records]
    return atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def unit_file_stem(unit_id: str) -> str:
    """Return a Windows-safe file stem for a unit id.

    ``unit.schema.json`` ids look like ``u:zj:v01:c003``; ``:`` is illegal in
    Windows file names, so work artifacts use ``u_zj_v01_c003``.
    """
    return unit_id.replace(":", "_")
