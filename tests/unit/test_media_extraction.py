from __future__ import annotations

import zipfile
from pathlib import Path

from cbb_stage1.ingest.media import extract_media_assets


def test_extract_media_assets_writes_real_bytes(tmp_path):
    epub = tmp_path / "book.epub"
    with zipfile.ZipFile(epub, "w") as archive:
        archive.writestr("OEBPS/images/cover.png", b"\x89PNG\r\n\x1a\nreal-bytes")
        archive.writestr("OEBPS/chapter.xhtml", "<html>text</html>")
    work = tmp_path / "work"
    rows = extract_media_assets(epub, work, "zj")
    assert len(rows) == 1
    row = rows[0]
    assert row["media_type"] == "image/png"
    assert row["path"].startswith("sources/assets/")
    target = work / row["source_path"]
    assert target.is_file()
    assert target.read_bytes().startswith(b"\x89PNG")
    assert row["sha256"].startswith("sha256:")
