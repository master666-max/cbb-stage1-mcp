from __future__ import annotations

import json

import pytest

from cbb_stage1.engines.ai_reader.chapter_fact_extractor import (
    ExtractionError,
    _parse_json_text,
)


def test_parses_plain_json_text():
    assert _parse_json_text('{"a": 1}') == {"a": 1}


def test_parses_fenced_json_text():
    assert _parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}


def test_non_json_text_is_an_explicit_error():
    with pytest.raises(ExtractionError):
        _parse_json_text("抱歉，我无法完成这个任务。")
