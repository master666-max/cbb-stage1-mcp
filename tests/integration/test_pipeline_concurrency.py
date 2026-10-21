"""C8 concurrency integration tests for ``run_pipeline``.

Verifies the extraction thread pool contract:

* ``ThreadPoolExecutor(max_workers=cfg.max_workers)`` overlaps ``engine.extract``
  calls for batches >= the parallel threshold;
* the **main thread** persists engine outputs / idempotency / checkpoints /
  task records in unit order (single writer);
* ``max_workers=1`` (and fail-fast) is equivalent to serial extraction;
* per-unit failure isolation still works under parallelism.

Run::

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_pipeline_concurrency.py -q
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from cbb_stage1.config import Config
from cbb_stage1.engines.base import Engine
from cbb_stage1.models import EngineOutput, Mention
from cbb_stage1.orchestrator.pipeline import run_pipeline
from cbb_stage1.orchestrator.progress import (
    ProgressStage,
    add_progress_listener,
    remove_progress_listener,
)

CHAPTER_COUNT = 6


class TrackingEngine(Engine):
    """Engine that records concurrency and optionally fails one unit."""

    def __init__(self, *, delay: float = 0.05, fail_unit_id: str | None = None) -> None:
        self.delay = delay
        self.fail_unit_id = fail_unit_id
        self.calls: list[str] = []
        self.threads: list[str] = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def extract(self, unit_id, text, card, known_entities):
        with self._lock:
            self.calls.append(unit_id)
            self.threads.append(threading.current_thread().name)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail_unit_id is not None and unit_id == self.fail_unit_id:
                raise RuntimeError(f"boom on {unit_id}")
            start = text.find("言")
            start = start if start >= 0 else 0
            surface = text[start : start + 1] or "?"
            return EngineOutput(
                unit_id=unit_id,
                mentions=[
                    Mention(
                        unit_id=unit_id,
                        surface=surface,
                        start=start,
                        end=start + len(surface),
                    )
                ],
            )
        finally:
            with self._lock:
                self._active -= 1


def _write_book(tmp_path: Path) -> Path:
    chapters = []
    for index in range(1, CHAPTER_COUNT + 1):
        chapters.append(
            f"# 第{index}章\n\n贝塔心叶与黑之魔王在第{index}处相遇，"
            f"她的影子是苍蓝色的。这是第{index}章用于并发测试的正文。"
        )
    book = tmp_path / "book.txt"
    book.write_text("\n\n".join(chapters) + "\n", encoding="utf-8")
    return book


def _make_config(tmp_path: Path, *, max_workers: int) -> Config:
    return Config(
        book_id="t",
        book_path=_write_book(tmp_path),
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        model="mock",
        provider="mock",
        max_workers=max_workers,
    )


def _expected_unit_ids() -> list[str]:
    return [f"u:t:v01:c{index:03d}" for index in range(1, CHAPTER_COUNT + 1)]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trace_decisions(tmp_path: Path) -> list[dict]:
    trace_file = next((tmp_path / "work" / "plan" / "traces").glob("*.json"))
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    return trace["decisions"]


# =============================================================================
# Parallel path
# =============================================================================

def test_parallel_extraction_overlaps_and_persists_in_unit_order(tmp_path):
    cfg = _make_config(tmp_path, max_workers=3)
    engine = TrackingEngine(delay=0.05)

    events: list = []
    add_progress_listener(events.append)
    try:
        result = run_pipeline(cfg, chapters="all", engine=engine)
    finally:
        remove_progress_listener(events.append)

    assert result.status == "complete"
    assert result.units_total == CHAPTER_COUNT
    assert result.units_processed == CHAPTER_COUNT

    # engine calls: all units, genuinely overlapping, within max_workers
    assert sorted(engine.calls) == _expected_unit_ids()
    assert engine.max_active >= 2, f"no overlap observed: max_active={engine.max_active}"
    assert engine.max_active <= 3
    worker_threads = {name for name in engine.threads if name.startswith("cbb-extract")}
    assert len(worker_threads) >= 2
    assert all(name.startswith("cbb-extract") for name in engine.threads)

    # main thread persisted in unit order
    tasks = _read_jsonl(tmp_path / "work" / "plan" / "tasks.jsonl")
    assert [task["unit_id"] for task in tasks] == _expected_unit_ids()
    assert [task["state"] for task in tasks] == ["extracted"] * CHAPTER_COUNT
    engine_files = sorted((tmp_path / "work" / "engine").glob("u_t_v01_c*.json"))
    assert len(engine_files) == CHAPTER_COUNT

    # extraction progress is emitted in unit order
    extraction_subs = [
        event.sub_current
        for event in events
        if event.stage is ProgressStage.EXTRACTING_CHARACTERS_AND_ENTITIES
        and event.sub_current > 0
        and event.message
        and event.message.startswith("unit ")
    ]
    assert extraction_subs == list(range(1, CHAPTER_COUNT + 1))

    decisions = _trace_decisions(tmp_path)
    concurrency = [d for d in decisions if d["category"] == "concurrency"]
    assert concurrency and concurrency[-1]["option_chosen"] == "parallel"


# =============================================================================
# Serial-equivalent paths
# =============================================================================

def test_max_workers_one_is_serial(tmp_path):
    cfg = _make_config(tmp_path, max_workers=1)
    engine = TrackingEngine(delay=0.02)
    result = run_pipeline(cfg, chapters="all", engine=engine)

    assert result.status == "complete"
    assert engine.calls == _expected_unit_ids()  # submission order preserved
    assert engine.max_active == 1
    assert set(engine.threads) == {"MainThread"}  # no worker pool at all


def test_small_batch_stays_serial(tmp_path):
    cfg = _make_config(tmp_path, max_workers=4)
    cfg.book_path = tmp_path / "book2.txt"
    cfg.book_path.write_text(
        "# 第1章\n\n贝塔心叶登场。\n\n# 第2章\n\n黑之魔王登场。\n", encoding="utf-8"
    )
    engine = TrackingEngine(delay=0.02)
    result = run_pipeline(cfg, chapters="all", engine=engine)

    assert result.status == "complete"
    assert result.units_total == 2
    assert engine.max_active == 1
    decisions = _trace_decisions(tmp_path)
    concurrency = [d for d in decisions if d["category"] == "concurrency"]
    assert concurrency and concurrency[-1]["option_chosen"] == "serial"


def test_fail_fast_is_serial_and_stops_at_first_error(tmp_path):
    cfg = _make_config(tmp_path, max_workers=4)
    failing_unit = "u:t:v01:c002"
    engine = TrackingEngine(delay=0.0, fail_unit_id=failing_unit)

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(
            cfg,
            chapters="all",
            engine=engine,
            fail_fast=True,
            checkpoint_interval=1,
        )

    # deterministic serial semantics: no unit after the failing one is called
    assert engine.calls == ["u:t:v01:c001", failing_unit]
    checkpoint = json.loads(
        (tmp_path / "work" / "plan" / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["completed_units"] == ["u:t:v01:c001"]
    run_summary = json.loads(
        (tmp_path / "work" / "plan" / "run.json").read_text(encoding="utf-8")
    )
    assert run_summary["status"] == "failed"


# =============================================================================
# Failure isolation under parallelism + idempotency
# =============================================================================

def test_failure_isolation_under_parallelism(tmp_path):
    cfg = _make_config(tmp_path, max_workers=3)
    failing_unit = "u:t:v01:c003"
    engine = TrackingEngine(delay=0.02, fail_unit_id=failing_unit)

    result = run_pipeline(cfg, chapters="all", engine=engine)
    assert result.status == "partial"
    assert result.units_failed == 1
    assert result.units_processed == CHAPTER_COUNT - 1

    tasks = _read_jsonl(tmp_path / "work" / "plan" / "tasks.jsonl")
    failed = [task for task in tasks if task["state"] == "failed"]
    extracted = [task for task in tasks if task["state"] == "extracted"]
    assert len(failed) == 1 and failed[0]["unit_id"] == failing_unit
    assert "boom" in failed[0]["error"]
    assert len(extracted) == CHAPTER_COUNT - 1
    assert not (tmp_path / "work" / "engine" / "u_t_v01_c003.json").exists()


def test_second_run_with_parallel_config_uses_cache(tmp_path):
    cfg = _make_config(tmp_path, max_workers=3)
    first_engine = TrackingEngine(delay=0.0)
    first = run_pipeline(cfg, chapters="all", engine=first_engine)
    assert first.status == "complete"
    assert first.units_processed == CHAPTER_COUNT

    second_engine = TrackingEngine(delay=0.0)
    second = run_pipeline(cfg, chapters="all", engine=second_engine)
    assert second.status == "complete"
    assert second_engine.calls == []
    assert second.units_cached == CHAPTER_COUNT
    assert second.units_processed == 0
