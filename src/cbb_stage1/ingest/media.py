"""Real media extraction from EPUB/zip sources (v2.2 assets, not just regex refs).

``projection_tables.extract_assets`` handles markdown/HTML *references*; this
module extracts the actual image/audio/video bytes and records a work table the
box writer can copy into ``sources/assets/``.  No vision/OCR model is involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

from cbb_stage1 import ids

__all__ = ["MEDIA_EXTENSIONS", "extract_media_assets"]

MEDIA_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


def _digest(data: bytes) -> str:
    return ids.sha256_bytes(data)


def _asset_id(book_id: str, name: str, data: bytes) -> str:
    key = f"{book_id}|{name}|{ids.sha256_bytes(data)}"
    return "as:" + ids.short_hash(key, 16)


def extract_media_assets(
    book_path: str | Path,
    work_root: str | Path,
    book_id: str,
) -> list[dict[str, Any]]:
    """Extract image/audio/video members from an EPUB; return work rows.

    For non-EPUB sources returns ``[]`` (text novels have no packaged media).
    Rows are written by the caller to ``work/plan/media.jsonl``.
    """
    source = Path(book_path)
    if source.suffix.lower() != ".epub" or not source.is_file():
        return []
    root = Path(work_root)
    media_dir = root / "assets"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            suffix = Path(name).suffix.lower()
            media_type = MEDIA_EXTENSIONS.get(suffix)
            if media_type is None or name.endswith("/"):
                continue
            try:
                data = archive.read(name)
            except KeyError:
                continue
            if not data:
                continue
            asset_id = _asset_id(book_id, name, data)
            if asset_id in seen:
                continue
            seen.add(asset_id)
            file_name = asset_id.split(":", 1)[1] + suffix
            target = media_dir / file_name
            target.write_bytes(data)
            rows.append(
                {
                    "asset_id": asset_id,
                    "book_id": book_id,
                    "path": f"sources/assets/{file_name}",
                    "sha256": _digest(data),
                    "media_type": media_type,
                    "source_unit_id": None,
                    "alt": None,
                    "source_name": name,
                    "source_path": f"assets/{file_name}",
                }
            )
    return sorted(rows, key=lambda row: row["asset_id"])
