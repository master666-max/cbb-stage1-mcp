"""Unit tests for the P0-04 orchestration shell and ingest layer.

Covers the preserved B09 shape (9 stages / checkpoint / idempotency / progress /
tracer), the cbb_stage1 bindings (Config / paths / Engine protocol / models),
EPUB and text ingestion, big-first chunking with overlap, and the
narrator / narrator_reliability unit field slots.

Run from ``V5/cbb_stage1_v6``::

    .venv\\Scripts\\python.exe -m pytest tests/unit/test_orchestrator.py -q
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from cbb_stage1.config import Config
from cbb_stage1.engines.mock import MockEngine
from cbb_stage1.ingest.chunk import (
    chunk_text,
    normalize_paragraphs,
    split_long_paragraph,
)
from cbb_stage1.ingest.epub import html_to_text, parse_epub
from cbb_stage1.ingest.loader import load_book
from cbb_stage1.ingest.units import (
    NARRATOR_KINDS,
    NARRATOR_RELIABILITIES,
    NARRATIVE_MODES,
    UNIT_ID_RE,
    build_units,
    detect_narrator,
    make_unit_id,
    select_chapters,
)
from cbb_stage1.orchestrator.idempotency import (
    IdempotencyStore,
    generate_idempotency_key,
)
from cbb_stage1.orchestrator.pipeline import run_pipeline
from cbb_stage1.orchestrator.progress import (
    STAGE_ORDER,
    ProgressStage,
    add_progress_listener,
    remove_progress_listener,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "orchestrator"
BOOK_TXT = FIXTURES / "book.txt"

REQUIRED_STAGE_NAMES = [
    "LOADING_BOOK",
    "ANALYZING_STRUCTURE",
    "EXTRACTING_CHARACTERS_AND_ENTITIES",
    "RESOLVING_DUPLICATE_NAMES",
    "GENERATING_DESCRIPTIONS",
    "BUILDING_CONNECTIONS",
    "WRITING_OUTPUT",
    "SAVING_STATE",
    "COMPLETE",
]


# =============================================================================
# Helpers / fakes
# =============================================================================

def make_config(tmp_path: Path, book: Path = BOOK_TXT, book_id: str = "testbook") -> Config:
    return Config(
        book_id=book_id,
        book_path=Path(book),
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        genre="light-novel-zh",
        model="mock",
        provider="mock",
    )


class SpyEngine(MockEngine):
    """MockEngine that records which units were sent to the engine."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, unit_id, text, card, known_entities):
        self.calls.append(unit_id)
        return super().extract(unit_id, text, card, known_entities)


class FailingOnCall(SpyEngine):
    """SpyEngine that raises exactly on the n-th call (simulated crash)."""

    def __init__(self, fail_at: int = 2) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.raised = False

    def extract(self, unit_id, text, card, known_entities):
        self.calls.append(unit_id)
        if len(self.calls) == self.fail_at:
            self.raised = True
            raise RuntimeError(f"simulated engine crash on {unit_id}")
        return MockEngine.extract(self, unit_id, text, card, known_entities)


def build_minimal_epub(path: Path, chapters: list[tuple[str, str]]) -> Path:
    """Write a minimal but spec-valid EPUB3 with one XHTML document per chapter."""
    manifest = []
    spine = []
    documents: dict[str, str] = {}
    for index, (title, body_html) in enumerate(chapters, 1):
        manifest.append(
            f'<item id="ch{index}" href="ch{index}.xhtml" '
            'media-type="application/xhtml+xml"/>'
        )
        spine.append(f'<itemref idref="ch{index}"/>')
        documents[f"OEBPS/ch{index}.xhtml"] = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            f"<head><title>{title}</title></head>"
            f"<body><h1>{title}</h1>{body_html}</body></html>"
        )

    opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        'unique-identifier="bookid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:identifier id="bookid">urn:uuid:test-epub-0001</dc:identifier>'
        "<dc:title>测试书</dc:title>"
        "<dc:creator>测试作者</dc:creator>"
        "<dc:language>zh</dc:language>"
        "</metadata>"
        f"<manifest>{''.join(manifest)}</manifest>"
        f"<spine>{''.join(spine)}</spine>"
        "</package>"
    )
    container = (
        '<?xml version="1.0"?>'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )

    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("mimetype")
        archive.writestr(info, "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        for name, content in documents.items():
            archive.writestr(name, content)
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# =============================================================================
# 1. 9-stage contract
# =============================================================================

def test_stage_contract_is_exactly_nine_stages_in_order():
    assert len(STAGE_ORDER) == 9
    assert [stage.name for stage in STAGE_ORDER] == REQUIRED_STAGE_NAMES
    assert STAGE_ORDER[0] is ProgressStage.LOADING_BOOK
    assert STAGE_ORDER[-1] is ProgressStage.COMPLETE


def test_pipeline_emits_all_nine_stages_in_order(tmp_path):
    cfg = make_config(tmp_path)
    events = []
    add_progress_listener(events.append)
    try:
        result = run_pipeline(cfg, chapters="all", engine=SpyEngine())
    finally:
        remove_progress_listener(events.append)

    assert result.status == "complete"
    first_seen: list[ProgressStage] = []
    for event in events:
        if not first_seen or first_seen[-1] is not event.stage:
            first_seen.append(event.stage)
    assert first_seen == list(STAGE_ORDER)


# =============================================================================
# 2. Ingest: EPUB / text -> units
# =============================================================================

def test_parse_text_book_into_chapters_and_units(tmp_path):
    book = load_book(BOOK_TXT, book_id="testbook")
    assert book.format == "text"
    assert [chapter.title for chapter in book.chapters] == ["第一章 低语", "第二章 上船"]

    units = build_units(book, tmp_path / "work", book_id="testbook", chapters="all")
    assert [unit.chapter for unit in units] == [1, 2]
    assert [unit.unit_id for unit in units] == [
        "u:testbook:v01:c001",
        "u:testbook:v01:c002",
    ]
    assert units[0].prev is None
    assert units[0].next == units[1].unit_id
    assert units[1].prev == units[0].unit_id
    assert units[1].next is None

    for unit in units:
        record = unit.to_record()
        # schema v2.2 §5.2 shape (narrator slots included, per the updated schema)
        for key in (
            "unit_id",
            "book_id",
            "volume",
            "chapter",
            "title",
            "order",
            "prev",
            "next",
            "text_path",
            "chars",
            "sha256",
            "narrative_mode",
            "narrator",
            "narrator_reliability",
            "source_locator",
            "scenes",
            "created_at",
        ):
            assert key in record, key
        assert UNIT_ID_RE.match(record["unit_id"])
        assert record["narrator"]["kind"] in NARRATOR_KINDS
        assert record["narrator_reliability"] in NARRATOR_RELIABILITIES
        assert record["narrative_mode"] in NARRATIVE_MODES
        assert record["sha256"].startswith("sha256:") and len(record["sha256"]) == 71
        assert record["chars"] == len(unit.text)
        assert record["source_locator"]["href"] == "book.txt"
        # text file on disk is exactly the unit text
        assert (tmp_path / "work" / record["text_path"]).read_text(encoding="utf-8") == unit.text


def test_detect_narrator_heuristic_and_unknown_default():
    assert detect_narrator("我" * 100)["kind"] == "first_person"
    assert detect_narrator("他走进房间，看了一眼窗外。")["kind"] == "third_person"
    assert detect_narrator("他走进房间，看了一眼窗外。")["name"] == "叙述者"
    assert detect_narrator("")["kind"] == "unknown"


def test_unit_id_validation_rejects_out_of_schema_parts():
    assert make_unit_id("ok_book", 1, 3) == "u:ok_book:v01:c003"
    assert make_unit_id("ok_book", 1, 3, 2) == "u:ok_book:v01:c003:s02"
    with pytest.raises(ValueError):
        make_unit_id("坏ID", 1, 1)
    with pytest.raises(ValueError):
        make_unit_id("ok", 0, 1)
    with pytest.raises(ValueError):
        make_unit_id("ok", 1, 1000)


def test_parse_minimal_epub(tmp_path):
    epub_path = build_minimal_epub(
        tmp_path / "mini.epub",
        [
            (
                "第一章 心叶",
                "<div><p>我的名字叫贝塔心叶。他人的思想不由分说地在脑中响起，"
                "他人的内心是毒药。</p><p>我抬起头，看着窗外的雨，房间里很安静，"
                "只有钟表走动的声音。我知道自己不该听见这些声音，可它们从来没有"
                "停过。</p><p>于是我决定不把这件事告诉任何人。</p></div>",
            ),
            (
                "第二章 甲板",
                "<p>拉斐尔·加西亚站在甲板上。他四十多岁，肌肉发达，"
                "是个高大的男人。</p><p>「你来了啊。」黑之魔王露出笑容，"
                "她的影子是苍蓝色的，像海面上的光。</p><p>整艘船已是一片惨状，"
                "宛如无差别屠杀的处刑场。海风很冷，他却一点也不在意。船身在风浪里"
                "轻轻摇晃，远处传来海鸟的叫声。</p>",
            ),
        ],
    )
    book = parse_epub(epub_path)
    assert book.format == "epub"
    assert book.title == "测试书"
    assert book.author == "测试作者"
    assert len(book.chapters) == 2
    assert book.chapters[0].href == "ch1.xhtml"
    first = book.chapters[0].content
    # nested div/p text must not be duplicated by the parser
    assert first.count("我的名字叫贝塔心叶") == 1
    assert "贝塔心叶" in first and "苍蓝色" in book.chapters[1].content
    # deterministic content hash
    assert parse_epub(epub_path).content_hash == book.content_hash


def test_html_to_text_does_not_duplicate_nested_blocks():
    html = "<html><body><div><p>第一段</p><p>第二段</p></div></body></html>"
    assert html_to_text(html) == "第一段\n\n第二段"


# =============================================================================
# 3. Chunking: big first, then split with overlap
# =============================================================================

def test_chunk_big_first_keeps_short_chapter_whole():
    text = "短章内容。" * 20
    chunks = chunk_text(text, target_chars=100, overlap_chars=10, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].split_reason == "big_first"
    assert chunks[0].overlap_chars == 0
    assert chunks[0].body_text == normalize_paragraphs(text)


def test_chunk_split_has_overlap_and_reconstructs_exactly():
    paragraphs = [f"第{i}段。" + "内容" * 40 for i in range(12)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, target_chars=200, overlap_chars=40, max_chars=320)

    assert len(chunks) > 1
    # splitting happens on paragraph boundaries => re-joining with "\n\n" is exact
    assert "\n\n".join(chunk.body_text for chunk in chunks) == normalize_paragraphs(text)
    assert chunks[0].overlap_chars == 0
    for chunk in chunks:
        assert chunk.body_chars <= 320
        assert chunk.split_reason == "structural_aggregate"
    for previous, current in zip(chunks, chunks[1:]):
        assert current.overlap_chars > 0
        assert current.overlap_chars <= 40
        tail = previous.body_text[-current.overlap_chars :]
        assert current.text.startswith(tail)
        assert current.text[current.overlap_chars :].lstrip("\n") == current.body_text


def test_chunk_sentence_fallback_for_overlong_paragraph():
    long_paragraph = "字" * 1000
    chunks = chunk_text(
        long_paragraph, target_chars=100, overlap_chars=20, max_chars=100
    )
    assert len(chunks) == 10
    assert all(chunk.body_chars <= 100 for chunk in chunks)
    assert all(chunk.split_reason == "sentence_fallback" for chunk in chunks)
    assert "".join(chunk.body_text for chunk in chunks) == long_paragraph


def test_split_long_paragraph_prefers_sentence_boundaries():
    piece = "第一句。第二句！第三句？"
    assert split_long_paragraph(piece, max_chars=1000) == [piece]
    parts = split_long_paragraph("甲" * 50 + "。" + "乙" * 50 + "。", max_chars=60)
    assert "".join(parts) == "甲" * 50 + "。" + "乙" * 50 + "。"


# =============================================================================
# 4. Chapter selection
# =============================================================================

def test_select_chapters_forms():
    assert select_chapters("all", [1, 2, 3]) == [1, 2, 3]
    assert select_chapters(None, [1, 2, 3]) == [1, 2, 3]
    assert select_chapters("2", [1, 2, 3]) == [2]
    assert select_chapters("1,3", [1, 2, 3]) == [1, 3]
    assert select_chapters("1-3", [1, 2]) == [1, 2]
    assert select_chapters("1-3,5", [1, 2, 3, 5]) == [1, 2, 3, 5]
    with pytest.raises(ValueError):
        select_chapters("5", [1, 2])
    with pytest.raises(ValueError):
        select_chapters("x", [1, 2])


def test_pipeline_honours_chapter_selector(tmp_path):
    cfg = make_config(tmp_path)
    result = run_pipeline(cfg, chapters="2", engine=SpyEngine())
    assert result.units_total == 1
    records = read_jsonl(tmp_path / "work" / "plan" / "units.jsonl")
    assert [record["chapter"] for record in records] == [2]

    with pytest.raises(ValueError):
        run_pipeline(cfg, chapters="9", engine=SpyEngine())


# =============================================================================
# 5. Idempotency
# =============================================================================

def test_idempotency_store_staleness_and_stats(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.json")
    key = generate_idempotency_key("book", "chapter", 0, content_hash="sha256:abc")
    assert not store.is_processed(key)
    assert store.is_stale(key, "m1", "p1")  # not cached => stale

    store.mark_complete(key, result={"x": 1}, model_name="m1", prompt_hash="p1")
    assert store.is_processed(key)
    assert store.get_cached_result(key) == {"x": 1}
    assert not store.is_stale(key, "m1", "p1")
    assert store.is_stale(key, "m2", "p1")
    assert store.is_stale(key, "m1", "p2")

    stats = store.get_stats()
    assert stats["total_processed"] == 1
    assert stats["untracked_entries"] == 0
    assert store.invalidate_for_model("m1") == 1
    assert not store.is_processed(key)


def test_second_pipeline_run_reuses_cache_without_engine_calls(tmp_path):
    cfg = make_config(tmp_path)
    first_engine = SpyEngine()
    first = run_pipeline(cfg, chapters="all", engine=first_engine)
    assert first.status == "complete"
    assert first.units_processed == 2
    assert first.units_cached == 0
    assert len(first_engine.calls) == 2

    engine_dir = tmp_path / "work" / "engine"
    hashes_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in engine_dir.glob("*.json")
    }
    assert len(hashes_before) == 2

    second_engine = SpyEngine()
    second = run_pipeline(cfg, chapters="all", engine=second_engine)
    assert second_engine.calls == []
    assert second.units_processed == 0
    assert second.units_cached == 2
    assert second.status == "complete"

    hashes_after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in engine_dir.glob("*.json")
    }
    assert hashes_after == hashes_before

    tasks = read_jsonl(tmp_path / "work" / "plan" / "tasks.jsonl")
    assert len(tasks) == 2
    assert all(task["cached"] for task in tasks)


# =============================================================================
# 6. Checkpoint / resume
# =============================================================================

def test_checkpoint_resume_skips_completed_units(tmp_path):
    cfg = make_config(tmp_path)
    failing_engine = FailingOnCall(fail_at=2)
    with pytest.raises(RuntimeError, match="simulated engine crash"):
        run_pipeline(
            cfg,
            chapters="all",
            engine=failing_engine,
            fail_fast=True,
            checkpoint_interval=1,
        )
    assert failing_engine.raised
    assert len(failing_engine.calls) == 2

    checkpoint_path = tmp_path / "work" / "plan" / "checkpoint.json"
    assert checkpoint_path.exists()
    saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert saved["completed_units"] == ["u:testbook:v01:c001"]

    resume_engine = SpyEngine()
    result = run_pipeline(
        cfg, chapters="all", engine=resume_engine, checkpoint_interval=1
    )
    assert result.status == "complete"
    assert resume_engine.calls == ["u:testbook:v01:c002"]
    assert result.units_cached == 1  # c001 came from the checkpoint, not the engine
    assert result.units_processed == 1
    # success clears the checkpoint
    assert not checkpoint_path.exists()


def test_per_unit_failure_isolation_without_fail_fast(tmp_path):
    cfg = make_config(tmp_path)
    result = run_pipeline(cfg, chapters="all", engine=FailingOnCall(fail_at=2))
    assert result.status == "partial"
    assert result.units_failed == 1
    assert result.units_processed == 1

    tasks = read_jsonl(tmp_path / "work" / "plan" / "tasks.jsonl")
    failed = [task for task in tasks if task["state"] == "failed"]
    extracted = [task for task in tasks if task["state"] == "extracted"]
    assert len(failed) == 1 and failed[0]["error"]
    assert len(extracted) == 1
    # failed unit has no engine artifact
    assert not (tmp_path / "work" / "engine" / "u_testbook_v01_c002.json").exists()


# =============================================================================
# 7. Pipeline end-to-end on EPUB + work artifacts + tracer
# =============================================================================

def test_pipeline_runs_end_to_end_on_epub(tmp_path):
    epub_path = build_minimal_epub(
        tmp_path / "mini.epub",
        [
            (
                "第一章 心叶",
                "<p>我的名字叫贝塔心叶。他人的思想不由分说地在脑中响起，"
                "他人的内心是毒药。</p><p>我抬起头，看着窗外的雨，房间里很安静，"
                "只有钟表走动的声音。我知道自己不该听见这些声音，可它们从来没有停过。"
                "于是我决定不把这件事告诉任何人。</p>",
            ),
            (
                "第二章 甲板",
                "<p>拉斐尔·加西亚站在甲板上。他四十多岁，肌肉发达，"
                "是个高大的男人。</p><p>「你来了啊。」黑之魔王露出笑容，"
                "她的影子是苍蓝色的，像海面上的光。</p><p>整艘船已是一片惨状，"
                "宛如无差别屠杀的处刑场。海风很冷，他却一点也不在意。船身在风浪里"
                "轻轻摇晃，远处传来海鸟的叫声。</p>",
            ),
        ],
    )
    cfg = make_config(tmp_path, book=epub_path)
    result = run_pipeline(cfg, chapters="all", engine=SpyEngine())

    assert result.status == "complete"
    assert result.book_format == "epub"
    assert result.units_total == 2
    units = read_jsonl(tmp_path / "work" / "plan" / "units.jsonl")
    assert [unit["source_locator"]["href"] for unit in units] == [
        "ch1.xhtml",
        "ch2.xhtml",
    ]


def test_unit_records_validate_against_design_schema(tmp_path):
    """Unit records must satisfy the box ``unit.schema.json`` (read-only check)."""
    jsonschema = pytest.importorskip("jsonschema")
    from cbb_stage1.schemas import schemas_dir

    try:
        schema_path = schemas_dir(Path(__file__).resolve().parents[2]) / "unit.schema.json"
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    cfg = make_config(tmp_path)
    run_pipeline(cfg, chapters="all", engine=SpyEngine())
    records = read_jsonl(tmp_path / "work" / "plan" / "units.jsonl")
    assert records
    for record in records:
        jsonschema.validate(record, schema)


def test_work_artifacts_trace_and_no_box_write(tmp_path):
    cfg = make_config(tmp_path)
    result = run_pipeline(cfg, chapters="all", engine=SpyEngine())
    work = tmp_path / "work"

    for relative in (
        "plan/units.jsonl",
        "plan/tasks.jsonl",
        "plan/state.json",
        "plan/run.json",
        "plan/ingestion_log.json",
        "plan/idempotency.json",
        "engine/index.jsonl",
        "resolve/mentions.jsonl",
        "resolve/entity_candidates.jsonl",
        "resolve/assertions.jsonl",
        "resolve/events.jsonl",
        "resolve/cues.jsonl",
        "assemble/entity_cards.jsonl",
        "assemble/connections.jsonl",
    ):
        assert (work / relative).exists(), relative

    # no box: output writing belongs to the v2.2 box-builder task
    assert not cfg.out_dir.exists()

    state = json.loads((work / "plan" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["units_total"] == 2
    assert state["trace_id"] == result.trace_id

    run_summary = json.loads((work / "plan" / "run.json").read_text(encoding="utf-8"))
    assert run_summary["status"] == "complete"
    assert run_summary["stages"] == REQUIRED_STAGE_NAMES

    tasks = read_jsonl(work / "plan" / "tasks.jsonl")
    assert {task["state"] for task in tasks} == {"extracted"}
    assert all(task["output_sha256"] for task in tasks)

    # tracer: one trace, one stage span per stage, decisions recorded
    trace_files = list((work / "plan" / "traces").glob("*.json"))
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace["trace_id"] == result.trace_id
    stage_spans = {
        span["name"] for span in trace["spans"] if span["name"].startswith("stage:")
    }
    assert stage_spans == {f"stage:{name}" for name in REQUIRED_STAGE_NAMES}
    assert trace["metadata"]["result"]["status"] == "complete"
    categories = {decision["category"] for decision in trace["decisions"]}
    assert {"load_book", "build_units", "engine_call", "resolve", "connections"} <= categories

    # progress log contains all 9 stages
    progress_files = list((work / "plan" / "progress").glob("*.json"))
    assert len(progress_files) == 1
    progress = json.loads(progress_files[0].read_text(encoding="utf-8"))
    logged_stages = {event["stage"] for event in progress["events"]}
    assert logged_stages == {stage.value for stage in STAGE_ORDER}

    telemetry_files = list((work / "plan" / "telemetry").glob("*.json"))
    assert len(telemetry_files) == 1
    telemetry = json.loads(telemetry_files[0].read_text(encoding="utf-8"))
    assert telemetry["counters"]["units_processed"] == 2


def test_smoke_entry_point_offline(tmp_path):
    from cbb_stage1.orchestrator.smoke import smoke

    result = smoke(tmp_path / "smoke")
    assert result.status == "complete"
    assert result.units_total == 2
    assert (tmp_path / "smoke" / "plan" / "run.json").exists()
