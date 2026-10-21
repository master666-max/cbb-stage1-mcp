from __future__ import annotations

from pathlib import Path

from cbb_stage1.config import Config
from cbb_stage1.engines.mock import MockEngine
from cbb_stage1.orchestrator.pipeline import run_pipeline


def test_three_chapter_pipeline_with_mock_engine(tmp_path: Path):
    book = tmp_path / "book.txt"
    book.write_text(
        "# 第1章 起\n\n贝塔心叶走进大厅，看见黑之魔王。\n\n"
        "# 第2章 承\n\n拉法把剑放在桌上，说这是证据。\n\n"
        "# 第3章 转\n\n苍蓝色的影子从墙后出现。\n",
        encoding="utf-8",
    )
    cfg = Config(
        book_id="t",
        book_path=book,
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        model="mock",
        provider="mock",
    )
    result = run_pipeline(cfg, chapters="all", engine=MockEngine())
    assert result.status == "complete"
    assert result.units_total == 3
    assert result.units_processed == 3
    assert (tmp_path / "work" / "plan" / "run.json").exists()
    units = (tmp_path / "work" / "plan" / "units.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(units) == 3
    assert "narrator_reliability" in units[0]
