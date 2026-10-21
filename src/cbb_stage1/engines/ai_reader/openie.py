"""Two-stage OpenIE (NER -> RDF triples) as a switchable AI Reader mode.

Ported from B05T-MaiBot ``chat/knowledge/ie_process.py`` + ``prompt_template.py``
(the LPMM two-stage shape): extract entities from a paragraph, then ask the model
for ``[subject, relation, object]`` triples constrained to that entity list.

This module is deliberately provider-agnostic: it only needs an object with the
async ``generate()`` signature used by :mod:`cbb_stage1.engines.ai_reader.llm_client`.
It is exposed as ``AiReaderEngine(extraction_mode="openie")`` and produces the V5
``EngineOutput`` contract through :func:`openie_result_to_engine_output`.

Evidence policy: LPMM/OpenIE has **no quote field** (only a passage).  Triples
therefore carry the paragraph span/hash for quote-backfill and are emitted with
``status="needs_evidence"``; a quote is never fabricated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cbb_stage1.engines._textsplit import Paragraph, split_paragraphs
from cbb_stage1.engines.ai_reader.llm_client import (
    LLMError,
    LlmUsage,
    _extract_json,
    _repair_truncated_json,
    _strip_thinking,
)
from cbb_stage1.evidence.anchor import find_occurrences
from cbb_stage1.models import Assertion, EngineOutput, Entity, Mention

logger = logging.getLogger(__name__)

__all__ = [
    "INVALID_ENTITIES",
    "OpenIEDoc",
    "OpenIEError",
    "OpenIEExtractor",
    "OpenIEResult",
    "build_entity_extract_prompt",
    "build_rdf_triple_prompt",
    "filter_entities",
    "filter_triples",
    "openie_result_to_engine_output",
    "parse_json_payload",
]

# Source INVALID_ENTITY list (B05T knowledge/__init__.py), kept verbatim.
INVALID_ENTITIES = frozenset({"", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们"})

# ── Prompts (verbatim from B05T prompt_template.py) ──────────────

ENTITY_EXTRACT_SYSTEM_PROMPT = """你是一个性能优异的实体提取系统。请从段落中提取出所有实体，并以JSON列表的形式输出。

输出格式示例：
[ "实体A", "实体B", "实体C" ]

请注意以下要求：
- 将代词（如“你”、“我”、“他”、“她”、“它”等）转化为对应的实体命名，以避免指代不清。
- 尽可能多的提取出段落中的全部实体；
"""

RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT = """你是一个性能优异的RDF（资源描述框架，由节点和边组成，节点表示实体/资源、属性，边则表示了实体和实体之间的关系以及实体和属性的关系。）构造系统。你的任务是根据给定的段落和实体列表构建RDF图。

请使用JSON回复，使用三元组的JSON列表输出RDF图中的关系（每个三元组代表一个关系）。

输出格式示例：
[
        ["某实体","关系","某属性"],
        ["某实体","关系","某实体"],
        ["某资源","关系","某属性"]
]

请注意以下要求：
- 每个三元组应包含每个段落的实体命名列表中的至少一个命名实体，但最好是两个。
- 将代词（如“你”、“我”、“他”、“她”、“它”等）转化为对应的实体命名，以避免指代不清。
"""


def build_entity_extract_prompt(paragraph: str, context_hint: str = "") -> str:
    """User prompt for stage 1 (NER), source layout preserved."""
    hint = f"\n\n已有上下文（仅作参考）：\n{context_hint}" if context_hint else ""
    return f"""段落：
```
{paragraph}
```{hint}"""


def build_rdf_triple_prompt(paragraph: str, entities: list[str], context_hint: str = "") -> str:
    """User prompt for stage 2 (RDF triples), source layout preserved."""
    entities_json = json.dumps(entities, ensure_ascii=False)
    hint = f"\n\n已有上下文（仅作参考）：\n{context_hint}" if context_hint else ""
    return f"""段落：
```
{paragraph}
```

实体列表：
```
{entities_json}
```{hint}"""


# ── Errors / result records ──────────────────────────────────────


class OpenIEError(RuntimeError):
    """Raised when a two-stage extraction cannot produce a usable result."""


@dataclass
class OpenIEDoc:
    """One paragraph's two-stage output (OpenIE doc item from B05T)."""

    passage: str
    start: int
    end: int
    sha256: str
    entities: list[str] = field(default_factory=list)
    triples: list[list[str]] = field(default_factory=list)


@dataclass
class OpenIEResult:
    """Aggregated two-stage result for one unit."""

    docs: list[OpenIEDoc] = field(default_factory=list)
    usage: LlmUsage = field(default_factory=LlmUsage)
    failed: list[Paragraph] = field(default_factory=list)

    @property
    def entities(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for doc in self.docs:
            for entity in doc.entities:
                if entity not in seen:
                    seen.add(entity)
                    result.append(entity)
        return result

    @property
    def triples(self) -> list[tuple[list[str], OpenIEDoc]]:
        return [(triple, doc) for doc in self.docs for triple in doc.triples]


# ── JSON parsing / filtering (json_repair semantics, dependency-free) ──


def _find_unclosed_brackets(json_str: str) -> list[str]:
    """Unclosed ``{``/``[`` in opening order (port of B05T json_fix)."""
    unclosed: list[str] = []
    inside_string = False
    escape_next = False
    for char in json_str:
        if inside_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                inside_string = False
        else:
            if char == '"':
                inside_string = True
            elif char in "{[":
                unclosed.append(char)
            elif char in "}]":
                if unclosed and (
                    (char == "}" and unclosed[-1] == "{")
                    or (char == "]" and unclosed[-1] == "[")
                ):
                    unclosed.pop()
    return unclosed


def _fix_broken_json(json_str: str) -> str:
    """Close truncated JSON by trimming an incomplete tail and appending closers.

    Port of B05T ``utils/json_fix.fix_broken_generated_json`` semantics (the
    ``json_repair`` dependency is not installed).  Works for both top-level
    arrays (LPMM NER/RDF responses) and objects: a trailing unterminated string
    is dropped at the last comma, then unclosed ``{``/``[`` are closed.
    """
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    text = json_str.rstrip()
    # Odd number of unescaped quotes => the last string literal is incomplete.
    if (len(re.findall(r'(?<!\\)"', text))) % 2 == 1:
        last_comma = text.rfind(",")
        if last_comma > 0:
            text = text[:last_comma]
    text = text.rstrip().rstrip(",")
    closers = "".join("]" if opener == "[" else "}" for opener in reversed(_find_unclosed_brackets(text)))
    return text + closers


def _loads_tolerant(text: str) -> Any:
    """Parse model output that may be fenced, prefixed, or truncated.

    Implements the tolerant path of LPMM's ``json_repair`` usage without adding
    the external dependency: strict parse -> strip thinking/fences -> first JSON
    array/object -> broken-JSON repair (array-friendly, unlike the object-only
    repair in ``llm_client``).
    """
    if text is None:
        raise OpenIEError("empty LLM response")
    cleaned = _strip_thinking(text)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        match = re.search(pattern, cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    for repaired in (_fix_broken_json(cleaned), _repair_truncated_json(cleaned)):
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    try:
        return _extract_json(cleaned)
    except LLMError as exc:
        raise OpenIEError(f"unparseable JSON: {exc}") from exc


def parse_json_payload(text: str, list_keys: tuple[str, ...]) -> Any:
    """Parse a response expected to contain a JSON list (or a known wrapper key)."""
    payload = _loads_tolerant(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in list_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if len(payload) == 1:
            value = next(iter(payload.values()))
            if isinstance(value, list):
                return value
    raise OpenIEError(f"expected JSON list, got {type(payload).__name__}")


def filter_entities(raw_entities: Any, *, max_entities: int = 100) -> list[str]:
    """Filter invalid/duplicate entity strings (port of OpenIE filters)."""
    if not isinstance(raw_entities, list):
        raise OpenIEError("entity stage did not return a list")
    entities: list[str] = []
    seen: set[str] = set()
    for entity in raw_entities:
        if not isinstance(entity, str):
            continue
        entity = entity.strip()
        if not entity or entity in INVALID_ENTITIES or entity in seen:
            continue
        if len(entity) > 40:
            continue
        seen.add(entity)
        entities.append(entity)
        if len(entities) >= max_entities:
            break
    return entities


def filter_triples(raw_triples: Any, *, max_triples: int = 200) -> list[list[str]]:
    """Validate/dedupe ``[subject, relation, object]`` triples."""
    if not isinstance(raw_triples, list):
        raise OpenIEError("triple stage did not return a list")
    triples: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in raw_triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        if any(not isinstance(item, str) or not item.strip() for item in triple):
            continue
        clean = [item.strip() for item in triple]
        key = (clean[0], clean[1], clean[2])
        if key in seen:
            continue
        seen.add(key)
        triples.append(clean)
        if len(triples) >= max_triples:
            break
    return triples


# ── Two-stage extractor ─────────────────────────────────────────


class OpenIEExtractor:
    """Two-stage NER -> RDF extractor over paragraph chunks."""

    name = "openie"

    def __init__(
        self,
        llm,
        *,
        retries: int = 2,
        retry_delay: float = 0.0,
        target_chars: int = 800,
        min_chars: int = 10,
        max_entities: int = 100,
        max_triples: int = 200,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int = 120,
    ) -> None:
        if llm is None:
            raise ValueError("OpenIEExtractor requires an LLM client")
        self.llm = llm
        self.retries = max(0, int(retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.max_entities = max_entities
        self.max_triples = max_triples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def extract(self, text: str, *, context_hint: str = "") -> OpenIEResult:
        """Run NER->RDF over every paragraph; raise if nothing succeeded."""
        paragraphs = split_paragraphs(
            text, target_chars=self.target_chars, min_chars=self.min_chars
        )
        if not paragraphs:
            raise OpenIEError("no_text")
        return await self.extract_paragraphs(paragraphs, context_hint=context_hint)

    async def extract_paragraphs(
        self,
        paragraphs: list[Paragraph],
        *,
        context_hint: str = "",
    ) -> OpenIEResult:
        result = OpenIEResult()
        for paragraph in paragraphs:
            try:
                doc, usage = await self.extract_one(paragraph, context_hint=context_hint)
            except (OpenIEError, LLMError, ValueError) as exc:
                logger.warning(
                    "openie paragraph %s failed: %s", paragraph.sha256[:12], exc
                )
                result.failed.append(paragraph)
                continue
            result.docs.append(doc)
            result.usage = result.usage + usage
        if not result.docs:
            raise OpenIEError(
                f"all {len(paragraphs)} paragraphs failed two-stage extraction"
            )
        return result

    async def extract_one(
        self, paragraph: Paragraph, *, context_hint: str = ""
    ) -> tuple[OpenIEDoc, LlmUsage]:
        entities, usage = await self._ner(paragraph.text, context_hint)
        triples, usage2 = await self._rdf(paragraph.text, entities, context_hint)
        doc = OpenIEDoc(
            passage=paragraph.text,
            start=paragraph.start,
            end=paragraph.end,
            sha256=paragraph.sha256,
            entities=entities,
            triples=triples,
        )
        return doc, usage + usage2

    async def _call_with_retry(self, system: str, prompt: str) -> tuple[str, LlmUsage]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                content, usage = await self.llm.generate(
                    system=system,
                    prompt=prompt,
                    format=None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                if not isinstance(content, str) or not content.strip():
                    raise OpenIEError("empty text response")
                return content, usage
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised
                last_error = exc
                if attempt < self.retries:
                    logger.warning(
                        "openie LLM call failed (attempt %d/%d): %s",
                        attempt + 1, self.retries + 1, exc,
                    )
                    if self.retry_delay:
                        await asyncio.sleep(self.retry_delay)
        raise OpenIEError(f"LLM call failed after {self.retries + 1} attempts: {last_error}")

    async def _ner(self, paragraph: str, context_hint: str) -> tuple[list[str], LlmUsage]:
        content, usage = await self._call_with_retry(
            ENTITY_EXTRACT_SYSTEM_PROMPT,
            build_entity_extract_prompt(paragraph, context_hint),
        )
        raw = parse_json_payload(content, ("entities", "result", "data", "items"))
        entities = filter_entities(raw, max_entities=self.max_entities)
        if not entities:
            raise OpenIEError("entity stage returned no valid entities")
        return entities, usage

    async def _rdf(
        self, paragraph: str, entities: list[str], context_hint: str
    ) -> tuple[list[list[str]], LlmUsage]:
        content, usage = await self._call_with_retry(
            RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT,
            build_rdf_triple_prompt(paragraph, entities, context_hint),
        )
        raw = parse_json_payload(content, ("triples", "result", "data", "items"))
        triples = filter_triples(raw, max_triples=self.max_triples)
        return triples, usage


# ── EngineOutput mapping ─────────────────────────────────────────


def openie_result_to_engine_output(
    result: OpenIEResult,
    *,
    unit_id: str,
    text: str,
    book_id: str = "",
    extra_notes: list[str] | None = None,
) -> EngineOutput:
    """Map an :class:`OpenIEResult` to the V5 ``EngineOutput`` contract."""
    entities: list[Entity] = []
    mentions: list[Mention] = []
    assertions: list[Assertion] = []

    mention_keys: set[tuple[str, int]] = set()
    for surface in result.entities:
        entities.append(
            Entity(
                book_id=book_id,
                type="unknown",
                canonical_name=surface,
                aliases=[],
                tier="minor",
                status="active",
                confidence=0.5,
            )
        )
        for idx, start in enumerate(find_occurrences(text, surface)):
            if idx >= 20:
                break
            key = (surface, start)
            if key in mention_keys:
                continue
            mention_keys.add(key)
            mentions.append(
                Mention(
                    unit_id=unit_id,
                    surface=surface,
                    start=start,
                    end=start + len(surface),
                    occurrence=idx + 1,
                    mention_type="name",
                    confidence=0.5,
                )
            )

    for triple, doc in result.triples:
        subject, relation, obj = triple
        assertions.append(
            Assertion(
                book_id=book_id,
                predicate=relation,
                library="relation",
                roles={
                    "subject": subject,
                    "object": obj,
                    # backfill anchors for the Evidence stage; no quote is claimed
                    "paragraph_span": [doc.start, doc.end],
                    "paragraph_sha256": doc.sha256,
                },
                claim_layer="statement",
                evidence=[],
                confidence=0.45,
                confidence_method="engine_self_report",
                status="needs_evidence",
            )
        )

    notes: list[str] = list(extra_notes or [])
    notes.append(
        f"openie: {len(result.docs)} paragraphs, {len(entities)} entities, "
        f"{len(assertions)} triples, {len(result.failed)} failed"
    )
    notes.append("openie: triples carry paragraph span/hash; Evidence stage must backfill quotes")

    mentions.sort(key=lambda m: (m.start, m.end))
    return EngineOutput(
        unit_id=unit_id,
        mentions=mentions,
        entities=entities,
        assertions=assertions,
        events=[],
        cues=[],
        summary=None,
        abstain=False,
        notes=notes,
        gleaning_round=0,
    )
