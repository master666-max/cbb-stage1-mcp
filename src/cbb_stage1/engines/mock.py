"""Deterministic mock engine for tests and offline smoke runs."""

from __future__ import annotations

from cbb_stage1.models import EngineOutput, Entity, Mention


class MockEngine:
    """Returns one mention + one assertion per unit, enough to exercise the spine."""

    def extract(self, unit_id: str, text: str, card: dict, known_entities: list[Entity]) -> EngineOutput:
        # find first non-space run of CJK/letters as a stable surface
        surface = ""
        start = 0
        for i, ch in enumerate(text):
            if ch.strip() and (ch.isalnum() or "\u4e00" <= ch <= "\u9fff"):
                surface = ch
                start = i
                break
        end = start + len(surface)
        return EngineOutput(
            unit_id=unit_id,
            mentions=[Mention(unit_id=unit_id, surface=surface, start=start, end=end)],
            assertions=[],
            notes=["mock engine: no assertions; wire ai_reader for real extraction"],
        )
