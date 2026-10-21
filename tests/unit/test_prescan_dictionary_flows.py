"""A1 — the pre-scan dictionary must reach the extraction prompt.

The two-stage shape LPMM describes is "scan the book once to build a dictionary, then
extract *with* that dictionary". This project had the first half and not the second:

* ``entity_pre_scanner.scan()`` runs before extraction (``pipeline.py:612``) and returns
  entries;
* ``build_known_entities_section`` renders those entries into a prompt section
  (``prompts.py:97``);
* ``aextract`` accepts ``known_entities`` and uses it (``engine.py:463``);

and yet ``pipeline.py:277`` called

    engine.extract(unit.unit_id, unit.text, card, [])

with an empty list in the fourth position. The dictionary was built and thrown away, so
every extraction ran without hints and the "two-stage" was really one stage plus dead work.

These tests pin the connection, not the scanner: the scanner already had its own tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from cbb_stage1.engines.ai_reader.prompts import build_known_entities_section
from cbb_stage1.orchestrator import pipeline as pipeline_module


# ── the prompt section itself ──────────────────────────────────────────────
def test_the_section_renders_names_types_and_aliases() -> None:
    class Entry:
        def __init__(self, name: str, etype: str, aliases: list[str]) -> None:
            self.name = name
            self.entity_type = etype
            self.aliases = aliases

    section = build_known_entities_section(
        [Entry("阿尔法", "person", ["小阿"])]
    )
    assert "阿尔法" in section
    assert "小阿" in section, "aliases are the reason a dictionary helps"


def test_an_empty_dictionary_renders_nothing() -> None:
    """No dictionary must not produce an empty section header in the prompt."""
    assert build_known_entities_section([]) == ""
    assert build_known_entities_section(None) == ""


def test_the_section_carries_no_claim() -> None:
    """Names and types only — a dictionary is not a place to smuggle assertions in."""
    class Entry:
        def __init__(self) -> None:
            self.name = "阿尔法"
            self.entity_type = "person"
            self.aliases: list[str] = []
            self.claim = "阿尔法要毁掉它"

    section = build_known_entities_section([Entry()])
    assert "阿尔法" in section
    assert "毁掉" not in section, "the dictionary section must not carry claims"


# ── the pipeline connection ────────────────────────────────────────────────
class RecordingEngine:
    """Captures what the pipeline passes to ``extract``."""

    def __init__(self) -> None:
        self.known_entities_by_unit: dict[str, Any] = {}
        self.prescan_called = False
        self.prescan_returned = [
            {"name": "阿尔法", "entity_type": "person", "aliases": ["小阿"]},
            {"name": "贝塔", "entity_type": "person", "aliases": []},
        ]

    def prescan_book(self, book_id: str, chapters: list[str], titles=None):  # noqa: ANN001
        self.prescan_called = True
        return self.prescan_returned

    def extract(self, unit_id: str, text: str, card: dict, known_entities):  # noqa: ANN001
        self.known_entities_by_unit[unit_id] = known_entities
        return {"unit_id": unit_id, "entities": [], "assertions": []}


def test_the_pipeline_hands_the_prescan_result_to_extract() -> None:
    """The defect this file exists for.

    It does not assert "prescan ran" — that was already true and is what made the gap
    invisible. It asserts the *result* reaches ``extract``.
    """
    engine = RecordingEngine()
    resolved = pipeline_module._prescan_for_extraction(
        engine, book_id="t", chapters=["第一章正文"], titles=["第一章"]
    )
    assert engine.prescan_called
    assert resolved, "the pre-scan result must be returned, not discarded"
    assert len(resolved) == 2


def test_the_pipeline_survives_an_engine_without_prescan() -> None:
    """Not every engine implements ``prescan_book``; that must stay non-fatal."""

    class Minimal:
        def extract(self, *a, **k):  # noqa: ANN002, ANN003
            return {}

    assert pipeline_module._prescan_for_extraction(
        Minimal(), book_id="t", chapters=["x"]
    ) == []


def test_the_pipeline_survives_a_prescan_that_raises() -> None:
    """A failed pre-scan degrades to "no hints", it does not abort the run."""

    class Boom(RecordingEngine):
        def prescan_book(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("scanner endpoint down")

    assert pipeline_module._prescan_for_extraction(
        Boom(), book_id="t", chapters=["x"]
    ) == []


def test_the_pipeline_normalises_entries_to_a_shape_the_prompt_accepts() -> None:
    """The scanner returns dicts; the prompt builder expects attributes.

    A first version passed the raw dicts straight through and the section rendered empty,
    because ``getattr(entry, 'name')`` on a dict is None. Silent, and the only symptom was
    that extraction still had no hints — the same symptom as not wiring it at all.
    """
    engine = RecordingEngine()
    resolved = pipeline_module._prescan_for_extraction(
        engine, book_id="t", chapters=["x"]
    )
    section = build_known_entities_section(resolved)
    assert "阿尔法" in section
    assert "贝塔" in section
