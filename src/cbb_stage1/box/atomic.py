"""Filesystem + hashing helpers for the single-writer box builder.

The box is immutable: :func:`build_box` writes every file into a temporary
sibling directory and only renames it to the final path after all tables,
hashes and the canonical ``box_lint`` gate pass. A failed build leaves no
half-box behind (design ``01-箱规格-v2`` §7).

Hashes follow the spec: ``sha256:<64 hex>``; file paths are relative POSIX
paths; the merkle root is :func:`merkle_root` over ``path:sha`` lines sorted by
path (matches ``tools/box_lint.py``).
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from cbb_stage1.ids import merkle_root, sha256_bytes, sha256_text, short_hash

__all__ = [
    "sha256_bytes",
    "sha256_text",
    "short_hash",
    "merkle_root",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
    "write_text",
    "new_temp_dir",
    "publish_dir",
    "remove_tree",
]


# ``sha256_bytes`` / ``sha256_text`` / ``short_hash`` / ``merkle_root`` are
# re-exported from :mod:`cbb_stage1.ids` (single formula source; the names stay
# stable for existing callers and tests).


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file (``default`` when missing/corrupt)."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dicts; missing file yields ``[]``."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: JSONL rows must be objects")
        records.append(value)
    return records


def write_json(path: str | Path, data: Any) -> Path:
    """Write JSON with the box's canonical formatting (2 spaces, UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write JSONL with LF endings and no BOM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), ensure_ascii=False) for record in records]
    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_text(path: str | Path, text: str) -> Path:
    """Write UTF-8 text with LF newlines (used for copied source files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def new_temp_dir(out: Path) -> Path:
    """Create the hidden temp build directory next to ``out``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.parent / f".{out.name}.build-{uuid.uuid4().hex[:12]}"
    temp.mkdir(parents=True, exist_ok=False)
    return temp


def publish_dir(temp: Path, out: Path) -> None:
    """Atomically rename the finished temp dir to ``out``.

    Raises:
        FileExistsError: if ``out`` already exists — boxes are immutable.
    """
    temp = Path(temp)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing box: {out}")
    os.replace(temp, out)


def remove_tree(path: str | Path) -> None:
    """Best-effort recursive removal for failed builds."""
    shutil.rmtree(path, ignore_errors=True)
