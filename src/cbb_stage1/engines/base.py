"""Engine interface. Engines only produce candidates; they never write the box."""

from __future__ import annotations

from typing import Protocol

from cbb_stage1.models import EngineOutput, Entity


class Engine(Protocol):
    def extract(self, unit_id: str, text: str, card: dict, known_entities: list[Entity]) -> EngineOutput:
        ...


class EngineError(RuntimeError):
    pass
