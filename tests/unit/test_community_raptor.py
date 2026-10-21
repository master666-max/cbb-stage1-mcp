"""P1-B tests: hierarchical community, recursive RAPTOR, box events/assets.

Covers:

* community level 1 (connected components) and level 2 (volume roll-up) with
  ``parent_id`` / ``child_community_ids``, rollup + drilldown;
* RAPTOR recursive cluster trees: structural token-Jaccard fallback, optional
  embedder, ``parent_cluster_id`` closure, ``text=None`` (no fabricated prose);
* the box writer preferring ``work/assemble/events.jsonl`` over legacy engine
  events (fallback always warns), asset extraction from markdown/HTML media
  references into ``sources/assets.jsonl``, and verbatim passthrough of the
  derive ``raptor``/``community`` tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.box import atomic
from cbb_stage1.box.projection_tables import extract_assets
from cbb_stage1.box.writer import build_box, canonical_lint_errors
from cbb_stage1.derive.community import (
    build_community_reports,
    drilldown,
    rollup_communities,
    validate_community_hierarchy,
)
from cbb_stage1.derive.integration import run as run_derive
from cbb_stage1.derive.raptor import (
    EMBEDDING_METHOD,
    STRUCTURAL_METHOD,
    build_raptor_clusters,
    validate_raptor_tree,
)

FIXED = "2026-10-19T01:30:00+08:00"
BOOK = "book"
U1 = "u:book:v01:c001"
U2 = "u:book:v01:c002"
U3 = "u:book:v02:c003"
U4 = "u:book:v02:c004"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unit(uid: str, text: str, order: int, volume: int = 1) -> dict:
    rel = f"plan/texts/v{volume:02d}/u{order:03d}.txt"
    return {
        "unit_id": uid,
        "book_id": BOOK,
        "volume": volume,
        "chapter": order,
        "title": f"第{order}章",
        "order": order,
        "text_path": rel,
        "chars": len(text),
        "sha256": atomic.sha256_text(text),
        "narrative_mode": "present",
        "narrator_reliability": "reliable",
        "source_locator": {"href": f"ch{order}.xhtml", "nav_index": order - 1},
        "scenes": [],
        "created_at": FIXED,
    }


def _write_unit_texts(work: Path, texts: dict[str, tuple[str, int, int]]) -> list[dict]:
    units = []
    for uid, (text, order, volume) in texts.items():
        unit = _unit(uid, text, order, volume)
        path = work / unit["text_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        units.append(unit)
    _write_jsonl(work / "plan" / "units.jsonl", units)
    return units


# ---------------------------------------------------------------------------
# community hierarchy
# ---------------------------------------------------------------------------


def test_community_reports_are_structural_and_not_summaries():
    rows = build_community_reports(
        "book",
        [
            {"assertion_id": "a:1", "roles": {"主体": "e:1", "客体": "e:2"}},
            {"assertion_id": "a:2", "roles": {"主体": "e:2", "客体": "e:3"}},
        ],
    )
    assert len(rows) == 1
    assert rows[0]["member_entities"] == ["e:1", "e:2", "e:3"]
    assert rows[0]["summary"] is None
    assert rows[0]["title"] is None
    assert rows[0]["parent_id"] is None
    assert rows[0]["produced_by"] == "derive:community_structural"


def test_community_hierarchy_rolls_components_up_by_volume():
    assertions = [
        {"roles": {"主体": "e:1", "客体": "e:2"}},   # volume 1
        {"roles": {"主体": "e:3", "客体": "e:4"}},   # volume 2
    ]
    entity_volumes = {"e:1": 1, "e:2": 1, "e:3": 2, "e:4": 2}
    rows = build_community_reports("book", assertions, entity_volumes=entity_volumes)
    level1 = [row for row in rows if row["level"] == 1]
    level2 = [row for row in rows if row["level"] == 2]
    assert len(level1) == 2
    assert len(level2) == 2
    assert validate_community_hierarchy(rows) == []

    by_volume = {row["volume"]: row for row in level2}
    assert set(by_volume) == {1, 2}
    for volume, parent in by_volume.items():
        children = drilldown(rows, parent["community_id"])
        assert len(children) == 1
        assert children[0]["member_entities"] == sorted(
            [entity for entity, vol in entity_volumes.items() if vol == volume]
        )
        assert children[0]["parent_id"] == parent["community_id"]
        assert parent["parent_id"] is None
        assert parent["child_community_ids"] == [children[0]["community_id"]]
        assert parent["summary"] is None and parent["title"] is None

    # Explicit rollup is the same structure as the inline hierarchy.
    rolled_level1, rolled_level2 = rollup_communities(
        "book", level1, entity_volumes
    )
    assert rolled_level1 == level1
    assert rolled_level2 == level2


def test_community_levels_filter_and_no_mapping():
    assertions = [{"roles": {"主体": "e:1", "客体": "e:2"}}]
    only_level1 = build_community_reports(
        "book",
        assertions,
        entity_volumes={"e:1": 1, "e:2": 1},
        levels=(1,),
    )
    assert [row["level"] for row in only_level1] == [1]
    assert only_level1[0]["parent_id"] is None

    without_mapping = build_community_reports("book", assertions)
    assert [row["level"] for row in without_mapping] == [1]


def test_community_multi_volume_component_uses_dominant_volume_parent():
    assertions = [{"roles": {"主体": "e:1", "客体": "e:2"}}]
    entity_volumes = {"e:1": [1, 2], "e:2": [1]}
    rows = build_community_reports("book", assertions, entity_volumes=entity_volumes)
    level1 = next(row for row in rows if row["level"] == 1)
    level2 = {row["volume"]: row for row in rows if row["level"] == 2}
    # dominant volume = 1 (e:1 appears in both volumes but tie-breaks to lowest)
    assert level1["parent_id"] == level2[1]["community_id"]
    assert set(level2) == {1, 2}
    assert validate_community_hierarchy(rows) == []


# ---------------------------------------------------------------------------
# RAPTOR recursive tree
# ---------------------------------------------------------------------------


def _structural_units() -> tuple[list[dict], dict[str, str]]:
    units = [
        {"unit_id": "u:1", "order": 1, "volume": 1},
        {"unit_id": "u:2", "order": 2, "volume": 1},
        {"unit_id": "u:3", "order": 3, "volume": 1},
        {"unit_id": "u:4", "order": 4, "volume": 1},
    ]
    texts = {
        "u:1": "甲 乙 丙 丁",
        "u:2": "甲 乙 丙 丁",
        "u:3": "甲 乙 丙 戊",
        "u:4": "甲 乙 丙 戊",
    }
    return units, texts


def test_raptor_tree_is_recursive_and_parent_closed():
    units, texts = _structural_units()
    rows = build_raptor_clusters(
        "book",
        units,
        texts,
        structural_threshold=0.8,
        threshold_decay=0.6,
    )
    assert rows == build_raptor_clusters(
        "book", units, texts, structural_threshold=0.8, threshold_decay=0.6
    )
    levels = {row["level"] for row in rows}
    assert levels == {1, 2}
    level1 = [row for row in rows if row["level"] == 1]
    level2 = [row for row in rows if row["level"] == 2]
    assert len(level1) == 2 and len(level2) == 1
    assert level2[0]["member_units"] == ["u:1", "u:2", "u:3", "u:4"]
    assert level2[0]["parent_cluster_id"] is None
    assert sorted(level2[0]["child_cluster_ids"]) == sorted(
        row["cluster_id"] for row in level1
    )
    for child in level1:
        assert child["parent_cluster_id"] == level2[0]["cluster_id"]
        assert child["text"] is None
        assert child["method"] == STRUCTURAL_METHOD
    assert validate_raptor_tree(rows) == []


def test_raptor_uses_embedder_and_marks_method():
    units, _ = _structural_units()
    target = {
        "u:1": [1.0, 0.0],
        "u:2": [0.9, 0.1],
        "u:3": [0.4, 0.4],
        "u:4": [0.35, 0.45],
    }

    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts):
            self.calls.append(list(texts))
            return [target[text] for text in texts]

    texts = {unit_id: unit_id for unit_id in target}
    embedder = FakeEmbedder()
    rows = build_raptor_clusters(
        "book",
        units,
        texts,
        embedder=embedder,
        embedding_threshold=0.8,
        threshold_decay=0.6,
    )
    assert len(embedder.calls) == 1  # unit texts embedded exactly once
    assert rows
    assert all(row["method"] == EMBEDDING_METHOD for row in rows)
    assert {row["level"] for row in rows} == {1, 2}
    assert validate_raptor_tree(rows) == []


def test_raptor_max_levels_and_embedder_validation():
    units, texts = _structural_units()
    rows = build_raptor_clusters(
        "book", units, texts, structural_threshold=0.8, max_levels=1
    )
    assert {row["level"] for row in rows} == {1}

    class BrokenEmbedder:
        def encode(self, texts):
            return [[1.0, 0.0]]  # wrong count

    with pytest.raises(ValueError):
        build_raptor_clusters("book", units, texts, embedder=BrokenEmbedder())
    with pytest.raises(ValueError):
        build_raptor_clusters("book", units, texts, max_levels=0)
    with pytest.raises(TypeError):
        build_raptor_clusters("book", units, texts, embedder="nope")


# ---------------------------------------------------------------------------
# asset extraction
# ---------------------------------------------------------------------------


def test_extract_assets_markdown_and_html(tmp_path):
    text = (
        "正文。\n"
        '![祭坛](assets/altar.png "祭坛图")\n'
        '<img src="assets/map.png" alt="地图">\n'
        '<audio src="assets/voice.mp3"></audio>\n'
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "altar.png").write_bytes(b"png-bytes")
    units = [{"unit_id": U1, "order": 1}]
    rows = extract_assets(
        BOOK, units, {U1: text}, media_roots=[tmp_path]
    )
    assert len(rows) == 3
    by_path = {row["path"]: row for row in rows}
    assert set(by_path) == {"assets/altar.png", "assets/map.png", "assets/voice.mp3"}
    assert by_path["assets/altar.png"]["media_type"] == "image/png"
    assert by_path["assets/altar.png"]["alt"] == "祭坛"
    assert by_path["assets/altar.png"]["sha256"] == atomic.sha256_bytes(b"png-bytes")
    assert by_path["assets/map.png"]["alt"] == "地图"
    assert by_path["assets/map.png"]["media_type"] == "image/png"
    assert by_path["assets/map.png"]["sha256"] == atomic.sha256_text("assets/map.png")
    assert by_path["assets/voice.mp3"]["media_type"] == "audio/mpeg"
    assert all(row["asset_id"].startswith("as:") for row in rows)
    assert all(row["source_unit_id"] == U1 for row in rows)


def test_extract_assets_empty_is_valid():
    assert extract_assets(BOOK, [{"unit_id": U1, "order": 1}], {U1: "纯文本。"}) == []


# ---------------------------------------------------------------------------
# box: events priority, assets table, derive passthrough
# ---------------------------------------------------------------------------


def _make_box_work(tmp_path: Path, text: str) -> Path:
    work = tmp_path / "work"
    (work / "plan").mkdir(parents=True)
    units = _write_unit_texts(work, {U1: (text, 1, 1)})
    _write_json(
        work / "plan" / "run.json",
        {
            "book_id": BOOK,
            "book_title": "P1-B",
            "trace_id": "p1b0001",
            "engine": "MockEngine:mock",
            "started_at": FIXED,
            "ended_at": FIXED,
        },
    )
    engine_event = "ev:" + "1" * 16
    _write_json(
        work / "engine" / "u_book_v01_c001.json",
        {
            "unit_id": U1,
            "mentions": [
                {"unit_id": U1, "surface": "我", "start": 0, "end": 1, "occurrence": 1, "mention_type": "name"}
            ],
            "events": [
                {
                    "event_id": engine_event,
                    "book_id": BOOK,
                    "type": "登场",
                    "summary": "engine 事件",
                    "participants": ["我"],
                    "unit_id": U1,
                    "importance": "medium",
                    "evidence": [{"unit_id": U1, "start": 0, "end": 1, "quote": "我"}],
                    "status": "asserted",
                }
            ],
        },
    )
    return work


def test_box_prefers_work_events_and_warns_on_legacy_fallback(tmp_path):
    work = _make_box_work(tmp_path, "我走进大厅。")
    legacy = build_box(work, tmp_path / "box-legacy", created_at=FIXED, box_id="p1b-events-legacy")
    events = _read_jsonl(legacy.box / "knowledge" / "events.jsonl")
    assert [row["summary"] for row in events] == ["engine 事件"]
    assert any("legacy engine-output" in warning for warning in legacy.warnings)
    assert canonical_lint_errors(legacy.box) == []

    work_event = "ev:" + "9" * 16
    _write_jsonl(
        work / "assemble" / "events.jsonl",
        [
            {
                "event_id": work_event,
                "book_id": BOOK,
                "type": "事件",
                "summary": "work 事件",
                "participants": ["我"],
                "unit_id": U1,
                "importance": "high",
                "evidence": [{"unit_id": U1, "start": 0, "end": 1, "quote": "我"}],
                "status": "asserted",
            }
        ],
    )
    preferred = build_box(work, tmp_path / "box-work", created_at=FIXED, box_id="p1b-events-work")
    events = _read_jsonl(preferred.box / "knowledge" / "events.jsonl")
    assert [row["summary"] for row in events] == ["work 事件"]
    assert not any("legacy engine-output" in warning for warning in preferred.warnings)
    assert canonical_lint_errors(preferred.box) == []

    # An existing (empty) work table is authoritative: no silent fallback.
    # The resulting box has no knowledge rows at all, so the measured
    # coverage invariant rejects it; publish with verify=False to inspect it.
    _write_jsonl(work / "assemble" / "events.jsonl", [])
    empty = build_box(
        work, tmp_path / "box-empty", created_at=FIXED,
        box_id="p1b-events-empty", verify=False,
    )
    assert _read_jsonl(empty.box / "knowledge" / "events.jsonl") == []
    assert any(
        "coverage invariant" in error
        for error in canonical_lint_errors(empty.box)
    )


def test_box_writes_extracted_assets(tmp_path):
    text = '我走进大厅。\n![祭坛](assets/altar.png "祭坛图")\n<img src="assets/map.png" alt="地图">'
    work = _make_box_work(tmp_path, text)
    (work / "assets").mkdir(exist_ok=True)
    (work / "assets" / "altar.png").write_bytes(b"altar")

    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="p1b-assets")
    assets = _read_jsonl(result.box / "sources" / "assets.jsonl")
    assert len(assets) == 2
    assert result.counts["assets"] == 2
    by_path = {row["path"]: row for row in assets}
    assert by_path["assets/altar.png"]["sha256"] == atomic.sha256_bytes(b"altar")
    assert by_path["assets/map.png"]["sha256"] == atomic.sha256_text("assets/map.png")
    assert canonical_lint_errors(result.box) == []

    plain_work = _make_box_work(tmp_path / "plain", "没有媒体。")
    # The engine's mention/event quotes do not exist in this text, so the built
    # box has units but no knowledge rows: the measured invariant rejects it.
    plain = build_box(
        plain_work, tmp_path / "box-plain", created_at=FIXED,
        box_id="p1b-noassets", verify=False,
    )
    assert _read_jsonl(plain.box / "sources" / "assets.jsonl") == []
    assert plain.counts["assets"] == 0


def test_box_passes_through_raptor_and_community_tables(tmp_path):
    work = _make_box_work(tmp_path, "我走进大厅。")
    raptor_rows = build_raptor_clusters(
        BOOK,
        [{"unit_id": U1, "order": 1, "volume": 1}],
        {U1: "我走进大厅。"},
    )
    community_rows = build_community_reports(
        BOOK, [{"roles": {"主体": "e:1", "客体": "e:2"}}]
    )
    _write_jsonl(work / "derive" / "raptor.jsonl", raptor_rows)
    _write_jsonl(work / "derive" / "community.jsonl", community_rows)

    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="p1b-passthrough")
    assert _read_jsonl(result.box / "derived" / "raptor.jsonl") == raptor_rows
    assert (
        _read_jsonl(result.box / "derived" / "community_reports.jsonl")
        == community_rows
    )


def test_derive_integration_writes_hierarchical_raptor_and_community(tmp_path):
    work = tmp_path / "work"
    texts = {
        U1: ("甲 乙 丙 丁 共同", 1, 1),
        U2: ("甲 乙 丙 丁 共同", 2, 1),
        U3: ("甲 乙 丙 戊 共同", 3, 2),
        U4: ("甲 乙 丙 戊 共同", 4, 2),
    }
    units = _write_unit_texts(work, texts)
    _write_jsonl(
        work / "resolve" / "mentions.jsonl",
        [
            {"unit_id": U1, "surface": "甲", "start": 0, "end": 1, "entity_id": "e:1"},
            {"unit_id": U3, "surface": "乙", "start": 2, "end": 3, "entity_id": "e:2"},
        ],
    )
    _write_jsonl(
        work / "assemble" / "assertions.jsonl",
        [{"assertion_id": "a:1", "roles": {"主体": "e:1", "客体": "e:2"}}],
    )

    result = run_derive(
        work,
        book_id=BOOK,
        outputs=[],
        units=units,
        created_at=FIXED,
        upstream={},
    )
    raptor_rows = _read_jsonl(work / "derive" / "raptor.jsonl")
    community_rows = _read_jsonl(work / "derive" / "community.jsonl")
    assert result["counts"]["raptor"] == len(raptor_rows)
    assert validate_raptor_tree(raptor_rows) == []
    assert validate_community_hierarchy(community_rows) == []
    assert {row["level"] for row in community_rows} == {1, 2}
    assert result["counts"]["community_levels"] == 2
