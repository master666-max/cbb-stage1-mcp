"""ChapterFact extractor: chapter text -> LLM -> validated nine-class fact.

Ported from B06-AI-Reader-V2 ``extraction/chapter_fact_extractor.py``.  Kept:
single/segmented extraction, per-segment retry, aggressive-truncation retry,
VoT spatial-guide injection, cloud schema embedding, minItems schema, and the
``_merge_chapter_facts`` segment merger.  Removed: ``src.infra.config`` /
``get_llm_client()`` globals (settings are injected) and DB persistence.

Also adds the **gleaning continuation** entry point required by design/02 §10.1:
``extract()`` is round 0, ``extract_continue()`` is one more round; the engine
decides how many rounds a unit gets.

C10 (design 07, untrusted-source guard): the chapter text is novel source, never
instructions.  Every prompt that quotes it wraps it via
``cbb_stage1.orchestrator.security.wrap_untrusted`` and logs a warning when the
injection scan flags it.  Evidence ``quote``/``span`` still refer to the
**original** text: ``_truncate``/segmentation run before wrapping and
``normalize.facts.chapter_fact_to_engine_output`` anchors quotes against the raw
unit text, so the wrapper prefix never shifts an offset.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from cbb_stage1.engines.ai_reader.budget import TokenBudget, compute_budget
from cbb_stage1.engines.ai_reader.llm_client import (
    LLMSettings,
    LlmUsage,
    load_settings_from_env,
)
from cbb_stage1.engines.ai_reader.prompts import (
    build_cloud_schema_suffix,
    build_example_text,
    build_gleaning_user_prompt,
    build_user_prompt,
    load_examples,
    load_system_prompt,
    load_vot_guide,
)
from cbb_stage1.engines.ai_reader.schema import (
    ChapterFact,
    CharacterFact,
    build_extraction_schema,
)
from cbb_stage1.normalize.fields import normalize_llm_payload
from cbb_stage1.orchestrator.security import scan, wrap_untrusted

logger = logging.getLogger(__name__)


#: Record classes that carry ``evidence`` + ``evidence_speaker``, in the order the
#: model sees them. Kept as data so the backfill loop needs no per-class branches.
# Segment splitting thresholds (chars). Only used when budget.segment_enabled.
_SEGMENT_THRESHOLD_2 = 7000
_SEGMENT_THRESHOLD_3 = 12000


def _parse_json_text(text: str) -> object:
    """Parse a JSON answer that arrived as text.

    Endpoint transports may return a parsed object, but host-queue/MCP transports
    only ever return text. Strip the usual code fence and parse; a non-JSON
    answer stays an ExtractionError rather than being silently coerced.
    """
    import re

    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"structured output was text but not valid JSON: {exc}"
        ) from exc


def _wrap_source(text: str, chapter_id: int | None = None) -> str:
    """Wrap novel source for a prompt; scan first and warn on injection shapes.

    The returned string is prompt-only. Evidence offsets must always be computed
    against ``text`` (the original), never against the wrapped prompt.
    """
    result = scan(text)
    if result.suspicious:
        logger.warning(
            "Chapter %s contains injection-shaped phrases %s; quoting it as "
            "untrusted novel source (never instructions)",
            chapter_id if chapter_id is not None else "?",
            result.flags,
        )
    return wrap_untrusted(text)


@dataclass
class ExtractionMeta:
    """Quality metadata about one extraction call."""

    is_truncated: bool = False
    original_len: int = 0
    truncated_len: int = 0
    segment_count: int = 1


class ExtractionError(Exception):
    """Raised when chapter fact extraction fails after retries."""


def _split_chapter_text(text: str) -> list[str]:
    """Split long chapter text into 1–3 segments at paragraph boundaries."""
    text_len = len(text)
    if text_len <= _SEGMENT_THRESHOLD_2:
        return [text]

    num_parts = 3 if text_len > _SEGMENT_THRESHOLD_3 else 2

    breaks: list[int] = [i for i, ch in enumerate(text) if ch == "\n" and i > 0]

    if not breaks:
        seg_len = text_len // num_parts
        return [
            text[i * seg_len: (i + 1) * seg_len if i < num_parts - 1 else text_len]
            for i in range(num_parts)
        ]

    segments: list[str] = []
    prev = 0
    for part_idx in range(1, num_parts):
        ideal = text_len * part_idx // num_parts
        best = min(breaks, key=lambda b: abs(b - ideal))
        segments.append(text[prev:best].strip())
        prev = best
    segments.append(text[prev:].strip())

    return [s for s in segments if s]


def _merge_chapter_facts(
    facts: list[ChapterFact],
    novel_id: str,
    chapter_id: int,
) -> ChapterFact:
    """Merge segment (or gleaning-round) ChapterFacts, de-duplicating entries."""
    if len(facts) == 1:
        return facts[0]

    char_map: dict[str, CharacterFact] = {}
    for fact in facts:
        for ch in fact.characters:
            if ch.name in char_map:
                existing = char_map[ch.name]
                char_map[ch.name] = CharacterFact(
                    name=ch.name,
                    new_aliases=list(dict.fromkeys(existing.new_aliases + ch.new_aliases)),
                    appearance=existing.appearance or ch.appearance,
                    abilities_gained=existing.abilities_gained + ch.abilities_gained,
                    locations_in_chapter=list(
                        dict.fromkeys(existing.locations_in_chapter + ch.locations_in_chapter)
                    ),
                )
            else:
                char_map[ch.name] = ch

    rel_seen: set[tuple[str, str, str]] = set()
    relationships = []
    for fact in facts:
        for rel in fact.relationships:
            key = (rel.person_a, rel.person_b, rel.relation_type)
            if key not in rel_seen:
                rel_seen.add(key)
                relationships.append(rel)

    loc_map: dict[str, object] = {}
    for fact in facts:
        for loc in fact.locations:
            if loc.name not in loc_map:
                loc_map[loc.name] = loc
            elif loc.description and not loc_map[loc.name].description:  # type: ignore[attr-defined]
                loc_map[loc.name] = loc

    sp_seen: set[tuple[str, str, str]] = set()
    spatial_relationships = []
    for fact in facts:
        for sr in fact.spatial_relationships:
            key = (sr.source, sr.target, sr.relation_type)
            if key not in sp_seen:
                sp_seen.add(key)
                spatial_relationships.append(sr)

    event_seen: set[str] = set()
    events = []
    for fact in facts:
        for ev in fact.events:
            if ev.summary not in event_seen:
                event_seen.add(ev.summary)
                events.append(ev)

    item_events = []
    for fact in facts:
        item_events.extend(fact.item_events)
    org_events = []
    for fact in facts:
        org_events.extend(fact.org_events)

    concept_map: dict[str, object] = {}
    for fact in facts:
        for c in fact.new_concepts:
            if c.name not in concept_map or (
                c.definition and len(c.definition) > len(concept_map[c.name].definition or "")  # type: ignore[attr-defined]
            ):
                concept_map[c.name] = c

    wd_seen: set[str] = set()
    world_declarations = []
    for fact in facts:
        for wd in fact.world_declarations:
            key = f"{wd.declaration_type}:{json.dumps(wd.content, sort_keys=True, ensure_ascii=False)}"
            if key not in wd_seen:
                wd_seen.add(key)
                world_declarations.append(wd)

    return ChapterFact(
        chapter_id=chapter_id,
        novel_id=novel_id,
        characters=list(char_map.values()),
        relationships=relationships,
        locations=list(loc_map.values()),  # type: ignore[arg-type]
        spatial_relationships=spatial_relationships,
        item_events=item_events,
        org_events=org_events,
        events=events,
        new_concepts=list(concept_map.values()),  # type: ignore[arg-type]
        world_declarations=world_declarations,
    )


class ChapterFactExtractor:
    """Extract a structured :class:`ChapterFact` from one chapter via an LLM."""

    def __init__(
        self,
        llm=None,
        *,
        budget: TokenBudget | None = None,
        settings: LLMSettings | None = None,
    ):
        self.settings = settings or load_settings_from_env()
        if llm is None:
            from cbb_stage1.engines.ai_reader.llm_client import get_llm_client

            llm = get_llm_client(self.settings)
        self.llm = llm
        self.budget = budget or compute_budget(self.settings.context_window)
        self.system_template = load_system_prompt()
        self._vot_guide = load_vot_guide()
        self.examples = load_examples()
        self.schema = build_extraction_schema()
        self._is_cloud = bool(getattr(llm, "is_cloud", False))

    # ── prompt/system assembly ────────────────────────────────────

    def _build_system(self, context_summary: str) -> tuple[str, bool]:
        """Return (system prompt, vot_injected) for this call."""
        system = self.system_template.replace(
            "{context}", context_summary or "（无前序上下文）"
        )
        vot_injected = False
        if not self.settings.vot_spatial_enabled or not self._vot_guide:
            return system, vot_injected
        if self.budget.context_window <= 8192:
            logger.info("VoT spatial guide skipped: context window %d <= 8192", self.budget.context_window)
            return system, vot_injected
        marker = "## 空间关系提取规则"
        if marker in system:
            system = system.replace(marker, self._vot_guide + "\n\n" + marker)
            vot_injected = True
        else:
            logger.warning("VoT injection skipped: marker %r not found in system prompt", marker)
        return system, vot_injected

    def _effective_max_len(self, vot_injected: bool) -> int:
        if vot_injected:
            return max(
                self.budget.max_chapter_len - len(self._vot_guide),
                self.budget.max_chapter_len // 2,
            )
        return self.budget.max_chapter_len

    def _truncate(self, text: str, vot_injected: bool) -> tuple[str, bool]:
        """Truncate to the effective chapter budget; return (text, was_truncated)."""
        max_len = self._effective_max_len(vot_injected)
        if len(text) > max_len:
            return text[:max_len], True
        return text, False

    # ── public API ───────────────────────────────────────────────

    async def extract(
        self,
        novel_id: str,
        chapter_id: int,
        chapter_text: str,
        context_summary: str = "",
    ) -> tuple[ChapterFact, LlmUsage, ExtractionMeta]:
        """Extract round 0. Long chapters are segmented and merged."""
        system, vot_injected = self._build_system(context_summary)
        original_len = len(chapter_text)
        chapter_text, truncated = self._truncate(chapter_text, vot_injected)
        meta = ExtractionMeta(
            is_truncated=truncated,
            original_len=original_len,
            truncated_len=len(chapter_text) if truncated else original_len,
        )

        segments = _split_chapter_text(chapter_text) if self.budget.segment_enabled else [chapter_text]
        meta.segment_count = len(segments)

        if len(segments) > 1:
            logger.info(
                "Chapter %d: splitting %d chars into %d segments (%s)",
                chapter_id, len(chapter_text), len(segments),
                ", ".join(f"{len(s)}c" for s in segments),
            )
            fact, usage = await self._extract_segmented(system, novel_id, chapter_id, segments)
            return fact, usage, meta

        fact, usage = await self._extract_single(system, novel_id, chapter_id, chapter_text)
        return fact, usage, meta

    async def extract_continue(
        self,
        novel_id: str,
        chapter_id: int,
        chapter_text: str,
        context_summary: str,
        previous: ChapterFact,
    ) -> tuple[ChapterFact, LlmUsage]:
        """One gleaning round: ask for missed entities/relations/events only."""
        system, vot_injected = self._build_system(context_summary)
        chapter_text, _ = self._truncate(chapter_text, vot_injected)
        example_text = build_example_text(self.examples, self.budget.context_window)
        existing_json = json.dumps(previous.model_dump(), ensure_ascii=False)
        prompt = build_gleaning_user_prompt(
            chapter_id, _wrap_source(chapter_text, chapter_id), example_text, existing_json
        )
        return await self._call_and_parse(system, prompt, novel_id, chapter_id)

    # ── internals ────────────────────────────────────────────────

    async def _extract_single(
        self,
        system: str,
        novel_id: str,
        chapter_id: int,
        chapter_text: str,
    ) -> tuple[ChapterFact, LlmUsage]:
        example_text = build_example_text(self.examples, self.budget.context_window)
        user_prompt = build_user_prompt(
            chapter_id,
            _wrap_source(chapter_text, chapter_id),
            example_text,
        )

        try:
            return await self._call_and_parse(system, user_prompt, novel_id, chapter_id)
        except Exception as first_err:  # noqa: BLE001 - retry path, then re-raise
            logger.warning("First extraction attempt failed for chapter %d: %s", chapter_id, first_err)

        retry_len = self.budget.retry_len
        truncated = chapter_text[:retry_len] if len(chapter_text) > retry_len else chapter_text
        retry_prompt = build_user_prompt(
            chapter_id,
            _wrap_source(truncated, chapter_id),
            example_text,
        )
        retry_prompt += "【重要】请输出严格的 JSON，不要输出多余文本。"
        try:
            return await self._call_and_parse(system, retry_prompt, novel_id, chapter_id)
        except Exception as second_err:  # noqa: BLE001 - wrapped as ExtractionError
            raise ExtractionError(
                f"Extraction failed for chapter {chapter_id} after 2 attempts: {second_err}"
            ) from second_err

    async def _extract_segmented(
        self,
        system: str,
        novel_id: str,
        chapter_id: int,
        segments: list[str],
    ) -> tuple[ChapterFact, LlmUsage]:
        example_text = build_example_text(self.examples, self.budget.context_window)
        segment_facts: list[ChapterFact] = []
        total_usage = LlmUsage()

        for idx, seg_text in enumerate(segments):
            seg_label = f"（第 {idx + 1}/{len(segments)} 部分）"
            logger.info(
                "Chapter %d segment %d/%d: %d chars",
                chapter_id, idx + 1, len(segments), len(seg_text),
            )
            user_prompt = build_user_prompt(
                chapter_id,
                _wrap_source(seg_text, chapter_id),
                example_text,
                segment_hint=seg_label,
            )
            try:
                fact, seg_usage = await self._call_and_parse(system, user_prompt, novel_id, chapter_id)
            except Exception as err:  # noqa: BLE001 - per-segment retry
                logger.warning(
                    "Chapter %d segment %d/%d failed: %s — retrying",
                    chapter_id, idx + 1, len(segments), err,
                )
                try:
                    retry_prompt = build_user_prompt(
                        chapter_id,
                        seg_text,
                        example_text,
                        segment_hint=seg_label,
                    )
                    retry_prompt += "【重要】请输出严格的 JSON，不要输出多余文本。"
                    fact, seg_usage = await self._call_and_parse(
                        system, retry_prompt, novel_id, chapter_id
                    )
                except Exception as retry_err:  # noqa: BLE001 - partial data is kept
                    logger.error(
                        "Chapter %d segment %d/%d failed after retry: %s",
                        chapter_id, idx + 1, len(segments), retry_err,
                    )
                    continue
            segment_facts.append(fact)
            total_usage = total_usage + seg_usage

        if not segment_facts:
            raise ExtractionError(
                f"All {len(segments)} segments failed for chapter {chapter_id}"
            )

        merged = _merge_chapter_facts(segment_facts, novel_id, chapter_id)
        logger.info(
            "Chapter %d: merged %d segments → %d chars, %d locs, %d events",
            chapter_id, len(segment_facts),
            len(merged.characters), len(merged.locations), len(merged.events),
        )
        return merged, total_usage

    async def _call_and_parse(
        self,
        system: str,
        prompt: str,
        novel_id: str,
        chapter_id: int,
        timeout: int = 600,
    ) -> tuple[ChapterFact, LlmUsage]:
        """Call the LLM and parse the response into a validated ChapterFact."""
        effective_system = system
        if self._is_cloud:
            effective_system += build_cloud_schema_suffix(self.schema)

        max_out = self.settings.max_tokens if self._is_cloud else 8192
        result, usage = await self.llm.generate(
            system=effective_system,
            prompt=prompt,
            format=self.schema,
            temperature=0.1,
            max_tokens=max_out,
            timeout=timeout,
            num_ctx=self.budget.extraction_num_ctx,
        )

        # Transports differ: an endpoint may hand back a parsed object, while the
        # host-queue / MCP transports only ever return text. Refusing str here
        # made every host-served chapter fail no matter how good the answer was.
        if isinstance(result, str):
            result = _parse_json_text(result)

        try:
            payload = normalize_llm_payload(result)
            payload["novel_id"] = novel_id
            payload["chapter_id"] = chapter_id
            return ChapterFact.model_validate(payload), usage
        except (ValidationError, ValueError) as exc:
            raise ExtractionError(
                f"Schema validation failed for chapter {chapter_id}: {exc}"
            ) from exc
