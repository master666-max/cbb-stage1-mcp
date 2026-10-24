"""The forecast is a proxy, and must say so.

History matters here. The module was deleted once as a "fake implementation"; that
was wrong — ``01-箱规格`` §5.20 asks for a chapter-scoped auxiliary reading marked
theoretical, which is exactly what it provides. It was then restored with ``method``
added so the number cannot be mistaken for the paper's metric.

These tests pin the honest labelling rather than the arithmetic.
"""

from __future__ import annotations

from cbb_stage1.derive.forecasts import METHOD, forecast_chapter


def test_forecast_returns_a_chapter_scoped_theory_only_reading() -> None:
    forecast = forecast_chapter("言万心叶走进大厅。黑之魔王在大厅。大厅很暗。")
    assert forecast.scope == "chapter"
    assert forecast.theory is True
    assert 0.0 <= forecast.no_rate <= 1.0
    assert forecast.top_terms


def test_forecast_is_deterministic() -> None:
    text = "言万心叶走进大厅。黑之魔王在大厅。大厅很暗。"
    assert forecast_chapter(text) == forecast_chapter(text)


def test_the_method_is_named_so_it_cannot_be_mistaken_for_the_paper_metric() -> None:
    """``no_rate`` here is lexical dispersion, not the fraction of divergent endings."""
    forecast = forecast_chapter("任意文本内容用于测试。")
    assert forecast.method == METHOD
    assert METHOD == "lexical_singleton_ratio"
    assert forecast.to_dict()["method"] == METHOD


def test_the_note_states_the_validation_limit() -> None:
    forecast = forecast_chapter("任意文本。")
    assert "短篇" in forecast.note
    assert "理论" in forecast.note


def test_empty_text_returns_the_neutral_reading() -> None:
    forecast = forecast_chapter("")
    assert forecast.no_rate == 1.0
    assert forecast.top_terms == []
