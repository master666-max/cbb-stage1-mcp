"""Engine factory for the V5 extraction stage.

Single wiring point so ``provider`` in the runtime config maps to a concrete
:class:`cbb_stage1.engines.base.Engine`:

===========================  ==================================================
provider                     engine
===========================  ==================================================
``mock``                     :class:`cbb_stage1.engines.mock.MockEngine`
``openai`` / ``ollama`` /    :class:`cbb_stage1.engines.ai_reader.AiReaderEngine`
``anthropic`` / ``ai_reader``
``openie``                   AI Reader with ``extraction_mode="openie"``
``lpmm``                     :class:`cbb_stage1.engines.lpmm.LpmmEngine`
===========================  ==================================================

The factory never falls back to ``mock`` for a real provider: an unknown
provider or a missing adapter raises :class:`EngineError` (a degraded run must
be visible, not silently green).
"""

from __future__ import annotations

import logging
from pathlib import Path

from cbb_stage1.engines.base import Engine, EngineError
from cbb_stage1.engines.mock import MockEngine

logger = logging.getLogger(__name__)

__all__ = ["ENGINE_PROVIDERS", "build_engine"]

ENGINE_PROVIDERS = (
    "mock",
    "ai_reader",
    "openai",
    "ollama",
    "anthropic",
    "openie",
    "lpmm",
)

_AI_READER_PROVIDERS = frozenset({"ai_reader", "ai-reader", "openai", "ollama", "anthropic"})
_MOCK_PROVIDERS = frozenset({"mock", ""})


def build_engine(
    provider: str = "mock",
    *,
    llm=None,
    model: str | None = None,
    max_gleanings: int = 0,
    extraction_mode: str = "chapter_fact",
    lpmm_settings=None,
    checkpoint_dir: str | Path | None = None,
) -> Engine:
    """Build the engine for *provider* (explicit, no silent fallback)."""
    name = (provider or "mock").strip().lower()

    if name in _MOCK_PROVIDERS:
        return MockEngine()

    if name == "lpmm":
        from cbb_stage1.engines.lpmm import LpmmEngine, LpmmSettings

        settings = lpmm_settings or LpmmSettings()
        if checkpoint_dir is not None:
            settings.checkpoint_dir = Path(checkpoint_dir)
        return LpmmEngine(llm=llm, settings=settings)

    if name == "openie":
        from cbb_stage1.engines.ai_reader import AiReaderEngine

        return AiReaderEngine(
            llm=llm,
            model=model,
            max_gleanings=max_gleanings,
            extraction_mode="openie",
        )

    if name in _AI_READER_PROVIDERS:
        from cbb_stage1.engines.ai_reader import AiReaderEngine
        from cbb_stage1.engines.ai_reader.stores import (
            InMemoryDictionaryStore,
            InMemoryFactStore,
        )

        return AiReaderEngine(
            llm=llm,
            model=model,
            max_gleanings=max_gleanings,
            extraction_mode=extraction_mode,
            dictionary_store=InMemoryDictionaryStore(),
            fact_store=InMemoryFactStore(),
        )

    raise EngineError(
        f"unknown engine provider {provider!r}; expected one of {sorted(ENGINE_PROVIDERS)}"
    )
