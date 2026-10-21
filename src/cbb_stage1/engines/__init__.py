"""Extraction engines (model side of the line: they only discover candidates).

``cbb_stage1.engines.base`` defines the Engine Protocol; this package also hosts
the AI Reader adapter, the LPMM/OpenIE two-stage adapters and the offline mock.
``build_engine`` is the single provider -> engine wiring point.
"""

from __future__ import annotations

__all__ = ["AiReaderEngine", "LpmmEngine", "MockEngine", "build_engine"]


def __getattr__(name: str):  # lazy imports keep `engines.base` import light
    if name == "build_engine":
        from cbb_stage1.engines.factory import build_engine

        return build_engine
    if name == "MockEngine":
        from cbb_stage1.engines.mock import MockEngine

        return MockEngine
    if name == "AiReaderEngine":
        from cbb_stage1.engines.ai_reader import AiReaderEngine

        return AiReaderEngine
    if name == "LpmmEngine":
        from cbb_stage1.engines.lpmm import LpmmEngine

        return LpmmEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
