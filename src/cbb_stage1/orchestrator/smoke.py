"""Offline mock smoke run (``cbb-stage1 smoke``).

Writes a tiny two-chapter book into the given work root, runs the 9-stage
pipeline with ``MockEngine``, and returns the result. No network, no box write:
this is the "EPUB/text -> units -> engine -> work artifacts" dry run from design
``08`` step D1.
"""

from __future__ import annotations

from pathlib import Path

from cbb_stage1.config import Config

from .pipeline import PipelineResult, run_pipeline

__all__ = ["smoke", "SMOKE_TEXT"]

SMOKE_TEXT = """# 第一章 烟测

我的名字叫烟测甲。这是一段用于离线冒烟的最小中文正文，不含任何真实作品内容。
他走进房间，看了一眼窗外的雨。

# 第二章 烟测续

烟测乙说：「今天的烟测应该通过。」房间里的灯亮着，雨还在下。
"""


def smoke(work_dir: str | Path) -> PipelineResult:
    """Run the offline mock pipeline inside ``work_dir``."""
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "_smoke_book.txt"
    source.write_text(SMOKE_TEXT, encoding="utf-8", newline="\n")

    cfg = Config(
        book_id="smoke",
        book_path=source,
        out_dir=root / "box",
        work_dir=root,
        provider="mock",
        model="mock",
    )
    result = run_pipeline(cfg, chapters="all")
    print(
        f"[smoke] status={result.status} units={result.units_total} "
        f"processed={result.units_processed} cached={result.units_cached} "
        f"mentions={result.mentions} trace={result.trace_id}"
    )
    return result
