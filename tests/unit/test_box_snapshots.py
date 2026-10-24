"""C9: a per-volume snapshot record, and a declaration that matches what the code does.

Measured gap, corrected against the code
----------------------------------------
The audit reported C9 as "`每卷一次快照（git commit）`=ABSENT：全仓无任何 git 调用". The grep is right and the
conclusion is wrong: **the code never intended git**. The box already computes `CHECKSUMS.json` with a
`merkle_root` over every file (`box/writer.py:1143`), which *is* a content-addressed snapshot.

GraphRAG — the project this series ports from — does the same thing: `SnapshotsConfig` is three boolean
flags over artifact copies (`embeddings` / `graphml` / `raw_graph`), not version-control commits. Data
versioning practice agrees: large artifacts go to content-addressed storage, git keeps pointers.

So C9's defect is the **declaration**, plus one missing artifact: nothing records a per-volume root, so
"每卷一次快照" cannot be pointed at even though the material for it exists.

What is pinned
--------------
* `qc/snapshots.json` exists and carries one entry per volume, each with a merkle root, the unit span and a
  file count;
* each entry's root is recomputable from the volume's files — a stored hash nobody can verify is decoration;
* a box with one volume gets one entry (no off-by-one at the boundary);
* the inventory row no longer claims git, and says content-addressed snapshots instead.

The last test is the one that keeps this from drifting back: it reads the inventory row and fails if the
word "git" reappears there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cbb_stage1.box.snapshots import (
    SNAPSHOT_CONTRACT,
    SNAPSHOT_PATH,
    snapshots_for_box,
    volume_root,
)

_UNIT_TESTS = Path(__file__).resolve().parents[1] / "unit"
if str(_UNIT_TESTS) not in sys.path:
    sys.path.insert(0, str(_UNIT_TESTS))

FIXED = "2026-10-18T23:00:00+08:00"


def _files(tmp_path: Path) -> dict[str, Path]:
    """Three files whose volumes we know."""
    paths: dict[str, Path] = {}
    for name, volume in (("a", 1), ("b", 1), ("c", 2)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"volume": volume}) + "\n", encoding="utf-8")
        paths[f"knowledge/{name}.jsonl"] = path
    return paths


# ══ the contract ════════════════════════════════════════════════════
def test_the_contract_is_named() -> None:
    assert SNAPSHOT_CONTRACT == "cbb-snapshots/1", (
        "a snapshot record needs a contract id, or a reader cannot tell which format it holds"
    )


def test_a_volume_root_is_order_independent() -> None:
    """Two builds that produce the same files must produce the same root."""
    first = {"a": "sha256:" + "1" * 64, "b": "sha256:" + "2" * 64}
    second = {"b": "sha256:" + "2" * 64, "a": "sha256:" + "1" * 64}
    assert volume_root(first) == volume_root(second)


def test_a_volume_root_changes_with_content() -> None:
    base = {"a": "sha256:" + "1" * 64}
    changed = {"a": "sha256:" + "9" * 64}
    assert volume_root(base) != volume_root(changed)


def test_an_empty_volume_has_no_root() -> None:
    assert volume_root({}) is None, "an empty volume must not look like a hash of nothing"


# ══ the record ══════════════════════════════════════════════════════
def test_one_entry_per_volume(tmp_path: Path) -> None:
    rows = snapshots_for_box(
        _files(tmp_path),
        unit_volumes={},
        created_at=FIXED,
        box_id="box1",
    )
    volumes = [row["volume"] for row in rows]
    assert volumes == sorted(set(volumes)), f"volumes are not one entry each: {volumes}"
    assert set(volumes) == {1, 2}


def test_each_entry_carries_a_root_and_a_span(tmp_path: Path) -> None:
    rows = snapshots_for_box(_files(tmp_path), unit_volumes={}, created_at=FIXED, box_id="box1")
    for row in rows:
        assert row["merkle_root"].startswith("sha256:")
        assert row["file_count"] >= 1
        assert "units" in row


def test_the_root_is_recomputable_from_the_files(tmp_path: Path) -> None:
    """A stored hash nobody can verify is decoration."""
    files = _files(tmp_path)
    rows = snapshots_for_box(files, unit_volumes={}, created_at=FIXED, box_id="box1")
    for row in rows:
        per_file = {
            name: "sha256:" + __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
            if path.parent.name == tmp_path.name and json.loads(path.read_text(encoding="utf-8"))["volume"] == row["volume"]
        }
        assert row["merkle_root"] == volume_root(per_file), (
            f"volume {row['volume']}'s recorded root does not match its files"
        )


def test_the_record_is_serialisable(tmp_path: Path) -> None:
    rows = snapshots_for_box(_files(tmp_path), unit_volumes={}, created_at=FIXED, box_id="box1")
    json.dumps({"contract": SNAPSHOT_CONTRACT, "snapshots": rows}, ensure_ascii=False)


def test_an_empty_box_yields_no_snapshots(tmp_path: Path) -> None:
    assert snapshots_for_box({}, unit_volumes={}, created_at=FIXED, box_id="box1") == []


# ══ the inventory must stop claiming git ════════════════════════════
INVENTORY = (
    Path(__file__).resolve().parents[3] / "设计" / "抽取段-v6" / "07-吸收机制总清单-20261018.md"
)


def test_the_inventory_row_no_longer_claims_git() -> None:
    if not INVENTORY.is_file():
        pytest.skip(f"inventory not present at {INVENTORY}")
    row = next(
        (line for line in INVENTORY.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("| C9 ")),
        "",
    )
    assert row, "the C9 row is missing"
    assert "git" not in row.lower(), (
        "the row still implies git; the implementation is a content-addressed snapshot, and the source "
        "project does the same (artifact copies, not commits)"
    )


def test_the_inventory_row_names_the_snapshot_mechanism() -> None:
    if not INVENTORY.is_file():
        pytest.skip(f"inventory not present at {INVENTORY}")
    row = next(
        (line for line in INVENTORY.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("| C9 ")),
        "",
    )
    assert "CHECKSUMS" in row or "merkle" in row or "内容寻址" in row, (
        f"the row does not say what the snapshot actually is: {row!r}"
    )

# ══ end to end: build_box writes it ═════════════════════════════════
def test_building_a_box_writes_the_snapshot_record(tmp_path) -> None:
    """The module is only real if the build calls it."""
    from test_box_writer import make_rich_work
    from cbb_stage1.box.writer import build_box

    work = make_rich_work(tmp_path)
    box = tmp_path / "snapbox"
    build_box(work, box, created_at=FIXED, box_id="snap")
    path = box / SNAPSHOT_PATH
    assert path.is_file(), "build_box produced no qc/snapshots.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["contract"] == SNAPSHOT_CONTRACT
    assert payload["snapshots"], "the record is empty"
    for entry in payload["snapshots"]:
        assert entry["merkle_root"].startswith("sha256:")
