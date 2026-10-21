"""AI Reader extraction kernel (ported from B06-AI-Reader-V2, AGPL-3.0).

Public entry point is :class:`AiReaderEngine`; the rest of the package is the
ported kernel (extractor, validator, pre-scanner, context builder, alias
resolver, LLM clients, prompts).  No FastAPI, no SQLite, no Chroma: persistence
goes through the :mod:`cbb_stage1.engines.ai_reader.stores` protocols.
"""

from cbb_stage1.engines.ai_reader.alias_resolver import (
    AliasResolver,
    UnionFind,
    build_alias_map,
    invalidate_alias_cache,
)
from cbb_stage1.engines.ai_reader.budget import TokenBudget, compute_budget, get_budget
from cbb_stage1.engines.ai_reader.chapter_fact_extractor import (
    ChapterFactExtractor,
    ExtractionError,
    ExtractionMeta,
)
from cbb_stage1.engines.ai_reader.context_summary_builder import ContextSummaryBuilder
from cbb_stage1.engines.ai_reader.engine import EXTRACTION_MODES, AiReaderEngine
from cbb_stage1.engines.ai_reader.entity_pre_scanner import EntityPreScanner
from cbb_stage1.engines.ai_reader.fact_validator import FactValidator
from cbb_stage1.engines.ai_reader.llm_client import (
    AnthropicClient,
    LLMClient,
    LLMError,
    LLMSettings,
    LLMTimeoutError,
    LlmUsage,
    OpenAICompatibleClient,
    get_llm_client,
    load_settings_from_env,
)
from cbb_stage1.engines.ai_reader.openie import (
    OpenIEDoc,
    OpenIEError,
    OpenIEExtractor,
    OpenIEResult,
    openie_result_to_engine_output,
)
from cbb_stage1.engines.ai_reader.schema import (
    ChapterFact,
    CharacterFact,
    ConceptFact,
    EntityDictEntry,
    EventFact,
    ItemEventFact,
    LocationFact,
    OrgEventFact,
    RelationshipFact,
    SpatialRelationship,
    WorldDeclaration,
)
from cbb_stage1.engines.ai_reader.stores import (
    DictionaryStore,
    FactStore,
    InMemoryDictionaryStore,
    InMemoryFactStore,
    JsonlDictionaryStore,
    JsonlFactStore,
)

__all__ = [
    "AiReaderEngine",
    "AliasResolver",
    "AnthropicClient",
    "ChapterFact",
    "ChapterFactExtractor",
    "CharacterFact",
    "ConceptFact",
    "ContextSummaryBuilder",
    "DictionaryStore",
    "EXTRACTION_MODES",
    "EntityDictEntry",
    "EntityPreScanner",
    "EventFact",
    "ExtractionError",
    "ExtractionMeta",
    "FactStore",
    "FactValidator",
    "InMemoryDictionaryStore",
    "InMemoryFactStore",
    "ItemEventFact",
    "JsonlDictionaryStore",
    "JsonlFactStore",
    "LLMClient",
    "LLMError",
    "LLMSettings",
    "LLMTimeoutError",
    "LlmUsage",
    "LocationFact",
    "OpenAICompatibleClient",
    "OpenIEDoc",
    "OpenIEError",
    "OpenIEExtractor",
    "OpenIEResult",
    "OrgEventFact",
    "RelationshipFact",
    "SpatialRelationship",
    "TokenBudget",
    "UnionFind",
    "WorldDeclaration",
    "build_alias_map",
    "compute_budget",
    "get_budget",
    "get_llm_client",
    "invalidate_alias_cache",
    "load_settings_from_env",
    "openie_result_to_engine_output",
]
