"""The speaker check needs the surrounding text, and the layout is not what it looks like.

Measured on a replay of spk10: the prompt reached the model with

    原文上下文：
    (nothing)

Two independent causes, both fixed here:

  * `plan/texts` stores by VOLUME and ORDER (`v01/u001.txt`), so globbing `*.txt`
    found nothing — the files are one directory down;
  * an assertion record can carry an empty `unit_id`, while its evidence still knows
    which unit it came from.

Without the context the verifier is asked to attribute a quote it cannot see the
surroundings of, which matters because the whole judgement rests on the neighbouring
question-and-answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.validate.integration import _quote_context, _unit_text_index

UNIT = "u:t:v01:c001"
TEXT = "第一段。\n\n「嗯？怎么了言万同学。你怕狗吗？」\n\n「这家伙…… 就是母体……──这个终末的源头。」\n\n（这么小只。）"


def _work(root: Path, *, volume: int = 1, order: int = 1) -> None:
    plan = root / "plan"
    (plan / "texts" / f"v{volume:02d}").mkdir(parents=True, exist_ok=True)
    (plan / "texts" / f"v{volume:02d}" / f"u{order:03d}.txt").write_text(
        TEXT, encoding="utf-8"
    )
    (plan / "units.jsonl").write_text(
        json.dumps(
            {"unit_id": UNIT, "book_id": "t", "order": order, "volume": volume},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_the_index_follows_the_volume_and_order_layout(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root)

    index = _unit_text_index(root)

    assert index == {UNIT: TEXT}, "must map by units.jsonl, not by filename stem"


def test_the_index_handles_a_second_volume(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root, volume=2, order=7)

    index = _unit_text_index(root)

    assert index == {UNIT: TEXT}


def test_a_missing_text_file_is_skipped_not_faked(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root)
    (root / "plan" / "texts" / "v01" / "u001.txt").unlink()

    assert _unit_text_index(root) == {}


def test_the_context_includes_the_neighbouring_lines(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root)
    text = _unit_text_index(root)[UNIT]
    quote = "「这家伙…… 就是母体……──这个终末的源头。」"

    context = _quote_context(text, quote)

    assert quote in context
    assert "你怕狗吗" in context, "the preceding question is the evidence for attribution"
    assert "这么小只" in context, "the following monologue corroborates it"


def test_the_context_is_empty_only_when_it_must_be(tmp_path: Path) -> None:
    assert _quote_context("", "「x」") == ""
    assert _quote_context("没有任何引文", "「x」") == "", "a quote absent from the text"