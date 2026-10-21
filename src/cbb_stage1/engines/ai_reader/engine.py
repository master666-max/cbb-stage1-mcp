"""AiReaderEngine: the B06 AI Reader extraction kernel behind the V5 Engine contract.

    engine = AiReaderEngine(llm=<async generate() client>)
    out: EngineOutput = engine.extract(unit_id, text, card, known_entities)

The engine only *discovers*: it returns mentions/entities/assertions/events as
candidates (plus notes), never box records, ids or merges.  The sync ``extract``
is the Engine Protocol method; ``aextract`` is the native coroutine for callers
that already run an event loop.

Wiring (task P1-A):
* **Pre-scan** — ``prescan_book(book_id, chapters, titles)`` runs the
  :class:`EntityPreScanner` once per book and persists candidates through the
  dictionary store.  The pipeline may also inject an already-built dictionary
  with ``set_dictionary(book_id, entries)``.  Both feed a "pre-scan dictionary"
  hint section; when neither exists the engine **explicitly degrades** and
  records it in ``EngineOutput.notes`` (never silently pretends hints exist).
* **Aliases** — an :class:`AliasResolver` is instantiated when both stores are
  available; its alias→canonical map is injected as a hint section and exposed
  through ``alias_map(book_id)``.  It is a hint only: identity resolution stays
  in the Resolve stage.
* **Mode switch** — ``extraction_mode="openie"`` swaps the nine-class
  ChapterFact path for the two-stage NER→RDF path (see
  :mod:`cbb_stage1.engines.ai_reader.openie`).

Gleaning (design/02 §10.1) applies to the chapter-fact mode and is off by
default: ``extract(..., gleaning_round=0, max_gleanings=None)``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from cbb_stage1.engines._async import run_coroutine_sync
from cbb_stage1.engines.ai_reader.alias_resolver import AliasResolver
from cbb_stage1.engines.ai_reader.budget import TokenBudget, compute_budget
from cbb_stage1.engines.ai_reader.chapter_fact_extractor import (
    ChapterFactExtractor,
    _merge_chapter_facts,
)
from cbb_stage1.engines.ai_reader.context_summary_builder import ContextSummaryBuilder
from cbb_stage1.engines.ai_reader.entity_pre_scanner import EntityPreScanner
from cbb_stage1.engines.ai_reader.fact_validator import FactValidator
from cbb_stage1.engines.ai_reader.llm_client import (
    LLMSettings,
    load_settings_from_env,
)
from cbb_stage1.engines.ai_reader.openie import (
    OpenIEExtractor,
    OpenIEResult,
    openie_result_to_engine_output,
)
from cbb_stage1.engines.ai_reader.prompts import build_known_entities_section
from cbb_stage1.engines.ai_reader.schema import EntityDictEntry
from cbb_stage1.engines.ai_reader.stores import (
    InMemoryDictionaryStore,
    InMemoryFactStore,
)
from cbb_stage1.engines.base import EngineError
from cbb_stage1.models import EngineOutput, Entity
from cbb_stage1.normalize.facts import chapter_fact_to_engine_output

logger = logging.getLogger(__name__)

EXTRACTION_MODES = ("chapter_fact", "openie")


def _usage_note(usage: object | None) -> str:
    """Token note that tolerates transports without usage accounting.

    Host-queue and MCP replies carry no usage object; dereferencing it used to
    raise AttributeError *after* a perfectly good extraction, so every
    host-served unit was recorded as failed.
    """
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and completion is None:
        return "tokens: unavailable (transport did not report usage)"
    return f"tokens: prompt={prompt or 0} completion={completion or 0}"


def ids_from_card(card: dict | None, unit_id: str) -> tuple[str, int]:
    """Best-effort ``(book_id, chapter_no)`` from a Card v2 (or a plain dict)."""
    book_id = ""
    chapter: int | None = None
    if isinstance(card, dict):
        book = card.get("book")
        if isinstance(book, dict):
            book_id = str(book.get("id") or "")
        book_id = book_id or str(card.get("book_id") or "")
        unit = card.get("unit")
        if isinstance(unit, dict):
            raw_chapter = unit.get("chapter")
            if isinstance(raw_chapter, int):
                chapter = raw_chapter
            elif isinstance(raw_chapter, str) and raw_chapter.isdigit():
                chapter = int(raw_chapter)
        if chapter is None:
            raw = card.get("chapter") or card.get("chapter_id")
            if isinstance(raw, int):
                chapter = raw
            elif isinstance(raw, str) and raw.isdigit():
                chapter = int(raw)
    if chapter is None:
        match = re.search(r"c(\d+)$", unit_id)
        chapter = int(match.group(1)) if match else 0
    return book_id, chapter


class AiReaderEngine:
    """Engine Protocol implementation backed by the ported AI Reader kernel."""

    name = "ai-reader"
    version = "0.1.0"

    def __init__(
        self,
        llm=None,
        *,
        settings: LLMSettings | None = None,
        budget: TokenBudget | None = None,
        dictionary_store=None,
        fact_store=None,
        max_gleanings: int = 0,
        extractor: ChapterFactExtractor | None = None,
        validator: FactValidator | None = None,
        context_builder: ContextSummaryBuilder | None = None,
        prescanner: EntityPreScanner | None = None,
        alias_resolver: AliasResolver | None = None,
        dictionary: dict[str, list[EntityDictEntry]] | None = None,
        extraction_mode: str = "chapter_fact",
        openie: OpenIEExtractor | None = None,
        model: str | None = None,
    ) -> None:
        if extraction_mode not in EXTRACTION_MODES:
            raise ValueError(
                f"unknown extraction_mode {extraction_mode!r}; expected one of {EXTRACTION_MODES}"
            )
        self.extraction_mode = extraction_mode
        if extraction_mode == "openie":
            self.name = "ai-reader-openie"

        self.settings = settings or load_settings_from_env()
        if model:
            self.settings = replace(self.settings, model=model)
        self.budget = budget or compute_budget(self.settings.context_window)

        if extractor is None:
            extractor = ChapterFactExtractor(
                llm=llm, budget=self.budget, settings=self.settings
            )
        self.llm = llm if llm is not None else extractor.llm
        self.extractor = extractor
        self.validator = validator or FactValidator()

        self.dictionary_store = dictionary_store
        self.fact_store = fact_store
        self._dictionary: dict[str, list[EntityDictEntry]] = {
            book_id: list(entries) for book_id, entries in (dictionary or {}).items()
        }
        self._alias_cache: dict[str, dict[str, str]] = {}
        self._prescan_status: dict[str, str] = {}

        if prescanner is None and dictionary_store is not None:
            prescanner = EntityPreScanner(dictionary_store=dictionary_store, llm=self.llm)
        self.prescanner = prescanner

        if alias_resolver is None and dictionary_store is not None and fact_store is not None:
            alias_resolver = AliasResolver(dictionary_store, fact_store)
        self.alias_resolver = alias_resolver

        self.max_gleanings = max(0, int(max_gleanings))

        if extraction_mode == "openie" and openie is None:
            openie = OpenIEExtractor(llm=self.llm)
        self.openie = openie

        if context_builder is None and (dictionary_store is not None or fact_store is not None):
            # The engine renders pre-scan dictionary hints itself (including
            # injected dictionaries); disable the builder's duplicate section.
            context_builder = ContextSummaryBuilder(
                fact_store=fact_store or InMemoryFactStore(),
                dictionary_store=dictionary_store,
                budget=self.budget,
                include_dictionary=False,
            )
        self.context_builder = context_builder

    # ── Engine Protocol (sync) ───────────────────────────────────

    def extract(
        self,
        unit_id: str,
        text: str,
        card: dict,
        known_entities: list[Entity],
        *,
        gleaning_round: int = 0,
        max_gleanings: int | None = None,
    ) -> EngineOutput:
        """Extract one unit. Returns candidates only; no persistence."""
        return run_coroutine_sync(
            self.aextract(
                unit_id,
                text,
                card,
                known_entities,
                gleaning_round=gleaning_round,
                max_gleanings=max_gleanings,
            )
        )

    # ── pre-scan / dictionary / alias wiring ────────────────────

    def set_dictionary(self, book_id: str, entries: list[EntityDictEntry]) -> None:
        """Inject a dictionary produced by an out-of-process pre-scan."""
        self._dictionary[book_id] = list(entries)
        self._alias_cache.pop(book_id, None)
        if self.dictionary_store is not None:
            try:
                self.dictionary_store.replace_entries(book_id, list(entries))
                self.dictionary_store.set_status(book_id, "completed")
            except Exception:  # noqa: BLE001 - hints are best-effort
                logger.warning("dictionary persist failed for %s", book_id, exc_info=True)
        if self.alias_resolver is not None:
            self.alias_resolver.invalidate(book_id)
        logger.info("Injected dictionary for %s: %d entries", book_id, len(entries))

    def dictionary_for(self, book_id: str) -> list[EntityDictEntry]:
        """Current dictionary: explicit injection first, then the store."""
        if book_id in self._dictionary:
            return list(self._dictionary[book_id])
        if self.dictionary_store is not None:
            try:
                return list(self.dictionary_store.list_entries(book_id))
            except Exception:  # noqa: BLE001 - hints are best-effort
                logger.warning("dictionary store read failed for %s", book_id, exc_info=True)
        return []

    def dictionary_status(self, book_id: str) -> str:
        """``completed`` / ``injected`` / store status / ``missing``."""
        if self._prescan_status.get(book_id) == "completed":
            return "completed"
        if book_id in self._dictionary:
            return "injected"
        if self.dictionary_store is not None:
            try:
                return self.dictionary_store.get_status(book_id)
            except Exception:  # noqa: BLE001
                return "unknown"
        return "missing"

    def alias_map(self, book_id: str) -> dict[str, str]:
        """Alias -> canonical map (hint layer; identity resolution is Resolve's job)."""
        if self.alias_resolver is not None:
            return self.alias_resolver.build_alias_map(book_id)
        if book_id in self._alias_cache:
            return self._alias_cache[book_id]
        entries = self._dictionary.get(book_id)
        if not entries:
            return {}
        # No resolver stores configured: build from the injected dictionary only.
        ephemeral = AliasResolver(InMemoryDictionaryStore(), InMemoryFactStore())
        alias_map = ephemeral.build_from_sources(entries, [])
        self._alias_cache[book_id] = alias_map
        return alias_map

    async def aprescan_book(
        self,
        book_id: str,
        chapters: list[str],
        titles: list[str] | None = None,
    ) -> list[EntityDictEntry]:
        """Run the pre-scan once per book and persist the dictionary."""
        if self.prescanner is None:
            raise EngineError(
                f"pre-scan unavailable for {book_id!r}: no EntityPreScanner configured"
            )
        entries = await self.prescanner.scan(book_id, chapters, titles)
        entries = list(entries or [])
        if self.dictionary_store is not None:
            # The scanner persists itself when given a store; make the guarantee
            # unconditional so an injected scanner without a store still lands.
            try:
                self.dictionary_store.replace_entries(book_id, entries)
                self.dictionary_store.set_status(book_id, "completed")
            except Exception:  # noqa: BLE001
                logger.warning("dictionary persist failed for %s", book_id, exc_info=True)
        else:
            self._dictionary[book_id] = entries
        self._prescan_status[book_id] = "completed"
        self._alias_cache.pop(book_id, None)
        if self.alias_resolver is not None:
            self.alias_resolver.invalidate(book_id)
        logger.info("pre-scan completed for %s: %d candidates", book_id, len(entries))
        return entries

    def prescan_book(
        self,
        book_id: str,
        chapters: list[str],
        titles: list[str] | None = None,
    ) -> list[EntityDictEntry]:
        """Sync wrapper around :meth:`aprescan_book`."""
        return run_coroutine_sync(self.aprescan_book(book_id, chapters, titles))

    # ── native async entry ───────────────────────────────────────

    async def aextract(
        self,
        unit_id: str,
        text: str,
        card: dict,
        known_entities: list[Entity],
        *,
        gleaning_round: int = 0,
        max_gleanings: int | None = None,
    ) -> EngineOutput:
        if not text or not text.strip():
            return EngineOutput(
                unit_id=unit_id,
                abstain=True,
                notes=["ai-reader: empty unit text (abstain=no_text)"],
            )

        book_id, chapter_no = ids_from_card(card, unit_id)
        context_summary = self._build_context_summary(
            book_id, chapter_no, card, known_entities
        )
        prescan_notes = self._prescan_notes(book_id, card, known_entities)

        if self.extraction_mode == "openie":
            return await self._aextract_openie(
                unit_id,
                text,
                book_id,
                context_summary,
                prescan_notes,
                gleaning_round=gleaning_round,
                max_gleanings=max_gleanings,
            )

        novel_id = book_id or "book"
        notes: list[str] = list(prescan_notes)
        try:
            fact, usage, meta = await self.extractor.extract(
                novel_id, chapter_no, text, context_summary
            )
        except Exception as exc:  # noqa: BLE001 - engine failure => abstain, never crash the line
            logger.warning("ai-reader extraction failed for %s: %s", unit_id, exc)
            return EngineOutput(
                unit_id=unit_id,
                abstain=True,
                notes=notes + [f"ai-reader: extraction failed: {type(exc).__name__}: {exc}"],
            )

        if meta.segment_count > 1:
            notes.append(f"segmented extraction: {meta.segment_count} segments")
        if meta.is_truncated:
            notes.append(
                f"chapter truncated: {meta.truncated_len}/{meta.original_len} chars"
            )
        notes.append(_usage_note(usage))

        fact = self.validator.validate(fact)
        used_rounds = 0
        max_rounds = self.max_gleanings if max_gleanings is None else max(0, int(max_gleanings))
        base_round = max(0, int(gleaning_round))
        while used_rounds < max_rounds:
            used_rounds += 1
            try:
                extra, extra_usage = await self.extractor.extract_continue(
                    novel_id, chapter_no, text, context_summary, fact
                )
            except Exception as exc:  # noqa: BLE001 - gleaning is best-effort
                notes.append(
                    f"gleaning round {used_rounds} failed: {type(exc).__name__}: {exc}"
                )
                break
            extra = self.validator.validate(extra)
            merged = _merge_chapter_facts([fact, extra], novel_id, chapter_no)
            if merged.model_dump() == fact.model_dump():
                notes.append(f"gleaning round {used_rounds}: no new candidates")
                break
            fact = merged
            notes.append(
                f"gleaning round {used_rounds}: merged (+{len(extra.characters)} chars, "
                f"+{len(extra.relationships)} rels, +{len(extra.events)} events candidates)"
            )

        return chapter_fact_to_engine_output(
            fact,
            unit_id=unit_id,
            text=text,
            book_id=book_id,
            gleaning_round=base_round + used_rounds,
            extra_notes=notes,
        )

    async def _aextract_openie(
        self,
        unit_id: str,
        text: str,
        book_id: str,
        context_summary: str,
        prescan_notes: list[str],
        *,
        gleaning_round: int,
        max_gleanings: int | None,
    ) -> EngineOutput:
        if self.openie is None:  # pragma: no cover - guarded by __init__
            raise EngineError("openie extraction mode has no OpenIEExtractor")
        notes = list(prescan_notes)
        max_rounds = self.max_gleanings if max_gleanings is None else max(0, int(max_gleanings))
        if max_rounds > 0:
            notes.append("openie mode: gleaning rounds are not supported (single two-stage pass)")
        try:
            result: OpenIEResult = await self.openie.extract(
                text, context_hint=context_summary
            )
        except Exception as exc:  # noqa: BLE001 - abstain, never crash the line
            logger.warning("openie extraction failed for %s: %s", unit_id, exc)
            return EngineOutput(
                unit_id=unit_id,
                abstain=True,
                notes=notes + [f"openie: extraction failed: {type(exc).__name__}: {exc}"],
            )

        notes.append(_usage_note(result.usage))
        output = openie_result_to_engine_output(
            result,
            unit_id=unit_id,
            text=text,
            book_id=book_id,
            extra_notes=notes,
        )
        if gleaning_round:
            output.notes.append(f"gleaning_round={gleaning_round} (unsupported in openie mode)")
        return output

    # ── context assembly ────────────────────────────────────────

    def _build_context_summary(
        self,
        book_id: str,
        chapter_no: int,
        card: dict,
        known_entities: list[Entity],
    ) -> str:
        sections: list[str] = []

        if isinstance(card, dict):
            rolling = card.get("rolling")
            if isinstance(rolling, dict) and rolling.get("summary"):
                sections.append(f"### 前情摘要\n{rolling['summary']}")

            glossary = card.get("glossary")
            glossary_entities = None
            if isinstance(glossary, dict):
                glossary_entities = glossary.get("entities")
            known_section = build_known_entities_section(known_entities, glossary_entities)
            if known_section:
                sections.append(known_section)

        if book_id:
            dict_section = self._dictionary_hints_section(book_id)
            if dict_section:
                sections.append(dict_section)
            alias_section = self._alias_hints_section(book_id)
            if alias_section:
                sections.append(alias_section)

        if self.context_builder is not None and book_id:
            try:
                store_section = self.context_builder.build(book_id, chapter_no)
            except Exception:  # noqa: BLE001 - context is best-effort
                logger.warning("context summary failed for %s/%s", book_id, chapter_no, exc_info=True)
                store_section = ""
            if store_section:
                sections.append(store_section)

        return "\n\n".join(sections)

    def _dictionary_hints_section(self, book_id: str, *, max_entries: int = 60) -> str:
        entries = self.dictionary_for(book_id)
        if not entries:
            return ""
        naming = [e for e in entries if e.source == "naming"][:20]
        naming_names = {e.name for e in naming}
        ordered = naming + [e for e in entries if e.name not in naming_names]
        ordered = ordered[:max_entries]

        lines = [
            "### 预扫词典（提示，非事实）",
            "以下为预扫描候选，仅作提取提示；别名必须在本章原文中找到明确依据，"
            "不同的人绝不能互为别名。",
        ]
        for entry in ordered:
            line = f"- {entry.name}（{entry.entity_type or 'unknown'}，出现{entry.frequency}次）"
            safe_aliases = [a for a in entry.aliases if len(a) <= 4][:2]
            if safe_aliases:
                line += f" 可能别名：{'、'.join(safe_aliases)}（需原文确认）"
            lines.append(line)
        return "\n".join(lines)

    def _alias_hints_section(self, book_id: str, *, max_entries: int = 40) -> str:
        try:
            mapping = self.alias_map(book_id)
        except Exception:  # noqa: BLE001 - hints are best-effort
            logger.warning("alias map failed for %s", book_id, exc_info=True)
            return ""
        if not mapping:
            return ""
        items = sorted(mapping.items())[:max_entries]
        lines = [
            "### 别名映射（预扫/历史提示，仅作提示）",
            "以下是已知的「别名 → 规范名」提示；如本章原文另有明确依据，以原文为准。",
        ]
        lines.extend(f"- {alias} → {canonical}" for alias, canonical in items)
        return "\n".join(lines)

    def _prescan_notes(
        self,
        book_id: str,
        card: dict,
        known_entities: list[Entity],
    ) -> list[str]:
        entries = self.dictionary_for(book_id) if book_id else []
        status = self.dictionary_status(book_id) if book_id else "missing"
        glossary_entities = []
        if isinstance(card, dict) and isinstance(card.get("glossary"), dict):
            glossary_entities = card["glossary"].get("entities") or []

        notes: list[str] = []
        if entries:
            notes.append(
                f"prescan: dictionary hints={len(entries)} (status={status})"
            )
        elif glossary_entities or known_entities:
            notes.append(
                "prescan: no pre-scan dictionary; using card glossary/known_entities hints only"
            )
        else:
            # Explicit degradation: the engine must never pretend hints exist.
            notes.append(
                "prescan: degraded — no dictionary available (no pre-scan, no card "
                "glossary, no known_entities); extraction runs without dictionary hints"
            )
        if book_id:
            try:
                alias_count = len(self.alias_map(book_id))
            except Exception:  # noqa: BLE001
                alias_count = 0
            if alias_count:
                notes.append(f"prescan: alias map={alias_count} entries")
        return notes

    # ── introspection ───────────────────────────────────────────

    @property
    def engine_info(self) -> dict[str, str]:
        model = getattr(self.llm, "model", "") or getattr(self.extractor.llm, "model", "")
        return {
            "name": self.name,
            "version": self.version,
            "model": str(model),
            "mode": self.extraction_mode,
        }
