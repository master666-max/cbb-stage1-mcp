"""LPMM adapter: B05T two-stage NER -> RDF extraction behind the Engine Protocol.

Ported behaviour from B05T-MaiBot-0.12.0:
* ``src/chat/knowledge/ie_process.py`` — two-stage ``info_extract_from_str``
  (entity list -> RDF triples), tolerant JSON extraction, per-stage retry;
* ``scripts/info_extraction.py`` — paragraph thread pool, per-paragraph
  checkpoint file (skip already extracted, delete corrupt cache), failure
  collection (failed paragraph hashes), deterministic output ordering;
* ``src/chat/knowledge/open_ie.py`` — entity/triple filtering + de-duplication;
* ``src/chat/knowledge/utils/json_fix.py`` — broken-JSON repair.  The external
  ``json_repair`` package is not a dependency, so the same semantics are
  implemented locally in :mod:`cbb_stage1.engines.ai_reader.openie`.

V5 wiring:
* implements :class:`cbb_stage1.engines.base.Engine`;
* failure is explicit: if every paragraph fails the unit abstains (never a
  silent mock result);
* triples carry the paragraph span+hash for the Evidence stage to backfill
  quotes (LPMM has no quote field, so assertions are ``needs_evidence``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from cbb_stage1.engines._async import run_coroutine_sync
from cbb_stage1.engines._textsplit import Paragraph, split_paragraphs
from cbb_stage1.engines.ai_reader.llm_client import (
    LLMSettings,
    LlmUsage,
    get_llm_client,
    load_settings_from_env,
)
from cbb_stage1.engines.ai_reader.openie import (
    OpenIEDoc,
    OpenIEError,
    OpenIEResult,
    filter_entities,
    filter_triples,
    openie_result_to_engine_output,
    parse_json_payload,
)
from cbb_stage1.engines.lpmm.prompts import (
    ENTITY_EXTRACT_SYSTEM_PROMPT,
    RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT,
    build_entity_prompt,
    build_rdf_prompt,
)
from cbb_stage1.ids import sha256_hex_text
from cbb_stage1.models import EngineOutput, Entity

logger = logging.getLogger(__name__)

__all__ = ["LpmmEngine", "LpmmSettings", "LpmmAdapter"]


@dataclass
class LpmmSettings:
    """Tuning for the LPMM adapter (all values override-able in tests)."""

    max_workers: int = 4
    retries: int = 3  # source: 3 attempts per stage
    retry_delay: float = 0.0  # source sleeps 5s; default 0 for CI, set in live runs
    target_chars: int = 800
    min_chars: int = 10
    max_entities: int = 100
    max_triples: int = 200
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 120
    checkpoint_dir: Path | None = None


class LpmmEngine:
    """Engine Protocol implementation for the LPMM/OpenIE two-stage pipeline."""

    name = "lpmm"
    version = "0.1.0"

    def __init__(
        self,
        llm=None,
        *,
        settings: LpmmSettings | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.settings = settings or LpmmSettings()
        if checkpoint_dir is not None:
            self.settings.checkpoint_dir = Path(checkpoint_dir)
        if llm is None:
            llm_settings: LLMSettings = load_settings_from_env()
            llm = get_llm_client(llm_settings)
        self.llm = llm

    # ── Engine Protocol ─────────────────────────────────────────

    def extract(
        self,
        unit_id: str,
        text: str,
        card: dict,
        known_entities: list[Entity],
    ) -> EngineOutput:
        """Extract one unit (sync Engine Protocol method)."""
        return run_coroutine_sync(self.aextract(unit_id, text, card, known_entities))

    async def aextract(
        self,
        unit_id: str,
        text: str,
        card: dict,
        known_entities: list[Entity],
    ) -> EngineOutput:
        if not text or not text.strip():
            return EngineOutput(
                unit_id=unit_id,
                abstain=True,
                notes=["lpmm: empty unit text (abstain=no_text)"],
            )

        book_id = _book_id_from_card(card)
        paragraphs = split_paragraphs(
            text,
            target_chars=self.settings.target_chars,
            min_chars=self.settings.min_chars,
        )
        if not paragraphs:
            return EngineOutput(
                unit_id=unit_id,
                abstain=True,
                notes=["lpmm: no extractable paragraphs (abstain=no_text)"],
            )

        docs_with_usage, failed = await asyncio.to_thread(
            self._run_thread_pool, paragraphs
        )

        if not docs_with_usage:
            failed_hashes = [p.sha256[:12] for p in failed[:10]]
            return EngineOutput(
                unit_id=unit_id,
                abstain=True,
                notes=[
                    f"lpmm: all {len(paragraphs)} paragraphs failed two-stage extraction",
                    f"lpmm: failed paragraph hashes={failed_hashes}",
                ],
            )

        result = OpenIEResult()
        for doc, doc_usage in docs_with_usage:
            result.docs.append(doc)
            result.usage = result.usage + doc_usage
        result.failed = failed
        usage = result.usage

        notes = [
            f"lpmm: {len(result.docs)}/{len(paragraphs)} paragraphs, "
            f"{len(result.entities)} entities, {len(result.triples)} triples, "
            f"{len(failed)} failed",
            "lpmm: triples carry paragraph span/hash; Evidence stage must backfill quotes",
            f"tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}",
        ]
        if failed:
            notes.append(f"lpmm: failed paragraph hashes={[p.sha256[:12] for p in failed[:10]]}")

        return openie_result_to_engine_output(
            result,
            unit_id=unit_id,
            text=text,
            book_id=book_id,
            extra_notes=notes,
        )

    # ── thread-pool batch (ported from scripts/info_extraction.py) ──

    def _run_thread_pool(
        self, paragraphs: list[Paragraph]
    ) -> tuple[list[tuple[OpenIEDoc, LlmUsage]], list[Paragraph]]:
        docs: list[tuple[int, OpenIEDoc, LlmUsage]] = []
        failed: list[Paragraph] = []
        workers = max(1, int(self.settings.max_workers))

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lpmm") as executor:
            future_map = {
                executor.submit(self._process_paragraph_sync, paragraph): paragraph
                for paragraph in paragraphs
            }
            for future in as_completed(future_map):
                paragraph = future_map[future]
                try:
                    doc, usage = future.result()
                except Exception as exc:  # noqa: BLE001 - per-paragraph isolation
                    logger.warning(
                        "lpmm paragraph %s failed: %s", paragraph.sha256[:12], exc
                    )
                    failed.append(paragraph)
                    continue
                docs.append((paragraph.start, doc, usage))

        docs.sort(key=lambda item: item[0])
        failed.sort(key=lambda p: p.start)
        return [(doc, usage) for _start, doc, usage in docs], failed

    def _process_paragraph_sync(
        self, paragraph: Paragraph
    ) -> tuple[OpenIEDoc, LlmUsage]:
        """Thread-pool worker: checkpoint lookup, then one async two-stage call."""
        cached = self._load_checkpoint(paragraph)
        if cached is not None:
            logger.debug("lpmm checkpoint hit: %s", paragraph.sha256[:12])
            return cached, LlmUsage()
        doc, usage = run_coroutine_sync(self._extract_paragraph_async(paragraph))
        self._save_checkpoint(doc)
        return doc, usage

    # ── checkpoint (port of temp/{sha}.json cache) ──────────────

    def _checkpoint_path(self, paragraph: Paragraph) -> Path | None:
        if self.settings.checkpoint_dir is None:
            return None
        return Path(self.settings.checkpoint_dir) / f"{paragraph.sha256}.json"

    def _load_checkpoint(self, paragraph: Paragraph) -> OpenIEDoc | None:
        path = self._checkpoint_path(paragraph)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Source behaviour: delete a corrupt cache and recompute.
            logger.warning("lpmm corrupt checkpoint removed: %s (%s)", path.name, exc)
            try:
                path.unlink()
            except OSError:
                pass
            return None

        if payload.get("passage") != paragraph.text:
            return None
        try:
            start, end = payload["span"]
            entities = filter_entities(payload.get("extracted_entities", []))
            triples = filter_triples(payload.get("extracted_triples", []))
        except (KeyError, TypeError, OpenIEError):
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return OpenIEDoc(
            passage=paragraph.text,
            start=int(start),
            end=int(end),
            sha256=paragraph.sha256,
            entities=entities,
            triples=triples,
        )

    def _save_checkpoint(self, doc: OpenIEDoc) -> None:
        path = self._checkpoint_path(Paragraph(doc.passage, doc.start, doc.end, doc.sha256))
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "idx": doc.sha256,
            "passage": doc.passage,
            "span": [doc.start, doc.end],
            "extracted_entities": doc.entities,
            "extracted_triples": doc.triples,
            "engine": f"lpmm/{self.version}",
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    # ── two-stage extraction with retry ────────────────────────

    async def _extract_paragraph_async(
        self, paragraph: Paragraph
    ) -> tuple[OpenIEDoc, LlmUsage]:
        entities, usage_ner = await self._stage_with_retry(
            ENTITY_EXTRACT_SYSTEM_PROMPT,
            build_entity_prompt(paragraph.text),
            list_keys=("entities", "result", "data", "items"),
            stage="ner",
        )
        entities = filter_entities(entities, max_entities=self.settings.max_entities)
        if not entities:
            raise OpenIEError("entity stage returned no valid entities")

        triples_raw, usage_rdf = await self._stage_with_retry(
            RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT,
            build_rdf_prompt(paragraph.text, entities),
            list_keys=("triples", "result", "data", "items"),
            stage="rdf",
        )
        triples = filter_triples(triples_raw, max_triples=self.settings.max_triples)

        return (
            OpenIEDoc(
                passage=paragraph.text,
                start=paragraph.start,
                end=paragraph.end,
                sha256=paragraph.sha256,
                entities=entities,
                triples=triples,
            ),
            usage_ner + usage_rdf,
        )

    async def _stage_with_retry(
        self,
        system: str,
        prompt: str,
        *,
        list_keys: tuple[str, ...],
        stage: str,
    ) -> tuple[list, LlmUsage]:
        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                content, usage = await self.llm.generate(
                    system=system,
                    prompt=prompt,
                    format=None,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    timeout=self.settings.timeout,
                )
                if not isinstance(content, str) or not content.strip():
                    raise OpenIEError(f"{stage}: empty text response")
                payload = parse_json_payload(content, list_keys)
                return payload, usage
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last_error = exc
                if attempt < self.settings.retries:
                    logger.warning(
                        "lpmm %s failed (attempt %d/%d): %s",
                        stage, attempt + 1, self.settings.retries + 1, exc,
                    )
                    if self.settings.retry_delay:
                        await asyncio.sleep(self.settings.retry_delay)
        raise OpenIEError(
            f"{stage} failed after {self.settings.retries + 1} attempts: {last_error}"
        )

    # ── introspection ───────────────────────────────────────────

    @property
    def prompt_sha256(self) -> str:
        """Stable prompt hash for the orchestrator's idempotency fingerprint."""
        payload = ENTITY_EXTRACT_SYSTEM_PROMPT + RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT
        return sha256_hex_text(payload, 16)

    @property
    def model(self) -> str:
        return str(getattr(self.llm, "model", "") or "")

    @property
    def engine_info(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "model": self.model}


def _book_id_from_card(card: dict | None) -> str:
    if not isinstance(card, dict):
        return ""
    book = card.get("book")
    if isinstance(book, dict) and book.get("id"):
        return str(book["id"])
    return str(card.get("book_id") or "")


LpmmAdapter = LpmmEngine
