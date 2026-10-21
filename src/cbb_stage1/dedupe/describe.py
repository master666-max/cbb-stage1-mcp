"""Description fragments: exact dedupe + map-reduce synthesis (sentence-anchored).

Ported intent from the v6 dedupe/enrich contract (``设计/抽取段-v6/00`` §4.12,
``02`` §4.3): the same entity/relation can arrive with several description
observations, and the pipeline must (a) drop *exact* duplicates before they
inflate anything, and (b) synthesise a description **without deleting the old
observations** ("旧观察不删，合成句逐句可回落").

Three layers:

* :func:`normalize_fragment` / :func:`dedupe_fragments` /
  :func:`dedupe_description_fragments` -- exact (character-identical modulo
  whitespace/NFKC) fragment dedupe with a reversible mapping;
* :func:`merge_observations` -- dedupe :class:`Observation` records while
  keeping the first occurrence and every provenance field intact;
* :func:`describe` -- the synthesis entry point.  With no synthesiser (or a
  synthesiser with no LLM available) it returns the deduplicated observation
  list (``mode="observations_only"``); with one it runs the synthesiser and
  keeps a **per-sentence fallback** to the observations plus a real
  ``inferred_ratio`` (fraction of sentences that cannot be traced back to any
  observation).

:class:`MapReduceSynthesizer` implements the design's map-reduce shape over any
``Callable[[str], str]`` LLM (map = per-batch compress, reduce = merge batch
outputs).  Everything here is pure apart from the injected LLM callable; input
order is preserved and no observation is deleted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from cbb_stage1.ids import sha256_hex_text

__all__ = [
    "FRAGMENT_HASH_PREFIX",
    "DEFAULT_MAP_BATCH_CHARS",
    "Observation",
    "FragmentDedupeResult",
    "DescribeResult",
    "SynthesizedSentence",
    "DescriptionSynthesizer",
    "MapReduceSynthesizer",
    "SynthesisUnavailable",
    "normalize_fragment",
    "fragment_key",
    "dedupe_fragments",
    "dedupe_description_fragments",
    "merge_observations",
    "describe",
    "render_observations",
    "split_description_sentences",
]

#: Version prefix for fragment content hashes (see ``semhash.SEMHASH_PREFIX``).
FRAGMENT_HASH_PREFIX = "df1:"

#: Default character budget for one map batch.
DEFAULT_MAP_BATCH_CHARS = 2000

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;…])")

#: Map prompt: compress one observation batch, no invention.
_MAP_PROMPT = """你是中文小说知识库的描述合成器。请只依据下面的观察，把它们合并成 1-2 句简洁的中文描述。
要求：不得编造观察中没有的信息；不要添加评价或推测；不要输出观察清单。
实体：{entity_id}
观察：
{observations}
只输出描述文本。"""

#: Reduce prompt: merge batch outputs into one description.
_REDUCE_PROMPT = """下面是同一实体的若干段合成描述，请合并成一段 2-4 句的中文描述。
要求：去重、不重复、不编造、不添加观察之外的信息。
实体：{entity_id}
片段：
{fragments}
只输出最终描述文本。"""


# ── fragment normalization / exact dedupe (unchanged) ───────────


def normalize_fragment(text: str) -> str:
    """NFKC-normalise and collapse whitespace; leave case/punctuation alone.

    Exact dedupe is *exact*: ``"他笑了"`` and ``"他笑了。"`` stay distinct, while
    ``" 他笑了 "`` / ``"他笑了\\n"`` / full-width-space variants collapse to one
    fragment (``"exact"`` here means "same characters modulo whitespace").
    """
    if not isinstance(text, str):
        raise TypeError(f"fragment must be str, got {type(text).__name__}")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def fragment_key(text: str) -> str:
    """Content hash of the normalized fragment (``df1:`` + sha256 hex)."""
    normalized = normalize_fragment(text)
    digest = sha256_hex_text(normalized)
    return f"{FRAGMENT_HASH_PREFIX}{digest}"


@dataclass(frozen=True)
class FragmentDedupeResult:
    """Result of :func:`dedupe_fragments`.

    Attributes:
        kept: First occurrence of each distinct normalized fragment.
        dropped: Duplicate occurrences (already represented in ``kept``).
        ignored: Empty/whitespace-only occurrences (no content to keep).
        mapping: Original fragment -> kept representative (reversible merge).
        clusters: Per kept representative, all original occurrences in order;
            ``clusters[i][0] == kept[i]`` (repeats of the same surface appear
            more than once here, but ``kept`` stays unique).
        total_items: Input occurrences (kept + dropped + ignored).
    """

    kept: tuple[str, ...]
    dropped: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()
    mapping: dict[str, str] = field(default_factory=dict)
    clusters: tuple[tuple[str, ...], ...] = ()
    total_items: int = 0

    @property
    def deduplicated_items(self) -> int:
        return len(self.kept)

    @property
    def duplicate_items(self) -> int:
        return len(self.dropped)

    @property
    def reduction(self) -> float:
        """Percentage of input occurrences removed (duplicates + ignored)."""
        if self.total_items == 0:
            return 0.0
        return (len(self.dropped) + len(self.ignored)) / self.total_items * 100.0

    @property
    def duplicates(self) -> dict[str, str]:
        """Only the non-identity entries of :attr:`mapping`."""
        return {item: rep for item, rep in self.mapping.items() if item != rep}

    def canonical_of(self, fragment: str) -> str:
        """Representative for ``fragment``, or the fragment itself if unknown."""
        return self.mapping.get(fragment, fragment)

    def rollback(self, representative: str) -> tuple[str, ...]:
        """Original occurrences merged into ``representative`` (input order)."""
        for cluster in self.clusters:
            if cluster and cluster[0] == representative:
                return cluster
        return ()


def dedupe_fragments(fragments: Iterable[str]) -> FragmentDedupeResult:
    """Drop exact-duplicate fragments (first occurrence wins)."""
    kept: list[str] = []
    dropped: list[str] = []
    ignored: list[str] = []
    mapping: dict[str, str] = {}
    clusters: list[list[str]] = []
    index_by_key: dict[str, int] = {}
    total = 0

    for fragment in fragments:
        if not isinstance(fragment, str):
            raise TypeError(f"fragments must be str, got {type(fragment).__name__}")
        total += 1
        normalized = normalize_fragment(fragment)
        if not normalized:
            ignored.append(fragment)
            continue
        key = fragment_key(normalized)
        index = index_by_key.get(key)
        if index is None:
            index = len(kept)
            index_by_key[key] = index
            kept.append(fragment)
            clusters.append([fragment])
            mapping[fragment] = fragment
        else:
            representative = kept[index]
            mapping.setdefault(fragment, representative)
            dropped.append(fragment)
            clusters[index].append(fragment)

    return FragmentDedupeResult(
        kept=tuple(kept),
        dropped=tuple(dropped),
        ignored=tuple(ignored),
        mapping=mapping,
        clusters=tuple(tuple(cluster) for cluster in clusters),
        total_items=total,
    )


def dedupe_description_fragments(fragments: Iterable[str]) -> FragmentDedupeResult:
    """Alias of :func:`dedupe_fragments` with the v6 entity-description naming."""
    return dedupe_fragments(fragments)


@dataclass(frozen=True)
class Observation:
    """One description observation about an entity/relation.

    Provenance fields are opaque here and are never rewritten: the synthesiser
    must be able to fall each output sentence back to one of these.
    """

    text: str
    unit_id: str | None = None
    source: str | None = None
    evidence: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"Observation.text must be str, got {type(self.text).__name__}")


def _coerce_observation(item: Observation | str) -> Observation:
    if isinstance(item, Observation):
        return item
    if isinstance(item, str):
        return Observation(text=item)
    raise TypeError(f"expected Observation or str, got {type(item).__name__}")


def merge_observations(
    observations: Iterable[Observation | str],
) -> tuple[Observation, ...]:
    """Dedupe observations by exact fragment; keep first occurrence and fields."""
    coerced = tuple(_coerce_observation(item) for item in observations)
    replaced: dict[str, str] = {}
    kept: list[Observation] = []
    for obs in coerced:
        normalized = normalize_fragment(obs.text)
        if not normalized:
            continue
        key = fragment_key(normalized)
        if key in replaced:
            continue
        replaced[key] = obs.text
        kept.append(obs)
    return tuple(kept)


# ── synthesis ───────────────────────────────────────────────────


class SynthesisUnavailable(RuntimeError):
    """Synthesis cannot run (no LLM / empty output); callers degrade explicitly."""


class DescriptionSynthesizer(Protocol):
    """Map-reduce synthesiser interface (sentence fallback is applied by describe)."""

    def synthesize(self, entity_id: str, fragments: Sequence[str]) -> str: ...


@dataclass(frozen=True)
class SynthesizedSentence:
    """One sentence of the synthesised description with its observation fallback.

    Attributes:
        text: Sentence text as produced by the synthesiser (trimmed).
        fallbacks: Observation texts that verbatim support this sentence; empty
            means the sentence is inferred (not traceable to any observation).
        inferred: ``not fallbacks`` (kept as a field for easy filtering).
    """

    text: str
    fallbacks: tuple[str, ...] = ()
    inferred: bool = False


def split_description_sentences(text: str) -> list[str]:
    """Split a synthesised description into trimmed sentences (order kept)."""
    if not isinstance(text, str):
        return []
    sentences: list[str] = []
    for line in text.splitlines():
        for part in _SENTENCE_END.split(line):
            sentence = part.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def _sentence_fallbacks(
    sentence: str,
    observation_texts: Sequence[str],
    *,
    max_fallbacks: int = 3,
) -> tuple[str, ...]:
    """Which observations verbatim support *sentence* (longest-common-substring proof)."""
    normalized_sentence = normalize_fragment(sentence)
    if len(normalized_sentence) < 2:
        return ()
    matches: list[str] = []
    for observation in observation_texts:
        normalized_observation = normalize_fragment(observation)
        if not normalized_observation:
            continue
        if normalized_observation in normalized_sentence or normalized_sentence in normalized_observation:
            matches.append(observation)
        else:
            match = SequenceMatcher(
                None, normalized_sentence, normalized_observation, autojunk=False
            ).find_longest_match(
                0, len(normalized_sentence), 0, len(normalized_observation)
            )
            # A sentence must share a verbatim 6-char run (or the whole shorter
            # observation) to count as supported; paraphrases stay inferred.
            needed = 6 if len(normalized_observation) >= 6 else len(normalized_observation)
            if match.size >= needed:
                matches.append(observation)
        if len(matches) >= max_fallbacks:
            break
    return tuple(matches)


class MapReduceSynthesizer:
    """Map-reduce description synthesiser over an injected ``Callable[[str], str]``.

    Map: observations are partitioned into character-budget batches; each batch
    is compressed independently.  Reduce: when more than one batch was mapped,
    the batch outputs are merged into the final description.  The synthesiser
    itself never invents provenance -- :func:`describe` attributes each output
    sentence back to observations afterwards.
    """

    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        *,
        map_batch_chars: int = DEFAULT_MAP_BATCH_CHARS,
    ) -> None:
        if map_batch_chars < 1:
            raise ValueError("map_batch_chars must be >= 1")
        self.llm = llm
        self.map_batch_chars = map_batch_chars
        #: Call counters of the *last* :meth:`synthesize` call (for QC/notes).
        self.map_calls = 0
        self.reduce_calls = 0
        self.batches = 0

    # -- DescriptionSynthesizer ----------------------------------

    def synthesize(self, entity_id: str, fragments: Sequence[str]) -> str:
        if not isinstance(entity_id, str):
            raise TypeError(f"entity_id must be str, got {type(entity_id).__name__}")
        self.map_calls = 0
        self.reduce_calls = 0
        self.batches = 0

        cleaned = [normalize_fragment(fragment) for fragment in fragments]
        cleaned = [fragment for fragment in cleaned if fragment]
        if not cleaned:
            raise SynthesisUnavailable("no observations to synthesise")
        if self.llm is None:
            raise SynthesisUnavailable("no LLM callable wired (observations_only)")

        batches = self._batch(cleaned)
        self.batches = len(batches)
        mapped: list[str] = []
        for batch in batches:
            prompt = _MAP_PROMPT.format(
                entity_id=entity_id, observations=self._numbered(batch)
            )
            mapped.append(self._call(prompt))
            self.map_calls += 1

        if len(mapped) == 1:
            return mapped[0]

        reduce_prompt = _REDUCE_PROMPT.format(
            entity_id=entity_id, fragments=self._numbered(mapped)
        )
        reduced = self._call(reduce_prompt)
        self.reduce_calls += 1
        return reduced

    # -- helpers -------------------------------------------------

    def _batch(self, fragments: Sequence[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        size = 0
        for fragment in fragments:
            extra = len(fragment) + (2 if current else 0)
            if current and size + extra > self.map_batch_chars:
                batches.append(current)
                current, size = [], 0
                extra = len(fragment)
            current.append(fragment)
            size += extra
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _numbered(items: Sequence[str]) -> str:
        return "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1))

    def _call(self, prompt: str) -> str:
        if self.llm is None:  # pragma: no cover - guarded in synthesize
            raise SynthesisUnavailable("no LLM callable wired (observations_only)")
        try:
            output = self.llm(prompt)
        except Exception as exc:  # noqa: BLE001 - surfaced as explicit unavailability
            raise SynthesisUnavailable(
                f"LLM call failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(output, str) or not output.strip():
            raise SynthesisUnavailable("LLM returned an empty description")
        return output.strip()

    def stats(self) -> dict[str, int]:
        """Call statistics of the last synthesis (map/reduce/batches)."""
        return {
            "map_calls": self.map_calls,
            "reduce_calls": self.reduce_calls,
            "batches": self.batches,
        }


# ── describe ────────────────────────────────────────────────────


@dataclass(frozen=True)
class DescribeResult:
    """Output of :func:`describe`.

    Attributes:
        entity_id: Entity the observations belong to.
        mode: ``observations_only`` (no synthesis ran) or ``synthesized``.
        observations: Deduplicated observations (first occurrence order).
        text: Synthesised description; ``None`` in ``observations_only`` mode.
        fragment_dedupe: Exact-dedupe result over the observation texts.
        inferred_ratio: Fraction of synthesised sentences without any
            observation fallback; ``None`` until synthesis runs.
        notes: Human-readable caveats/counts.
        sentences: Per-sentence text + observation fallbacks (empty until
            synthesis runs).  ``inferred`` sentences are the ones counted by
            :attr:`inferred_ratio`.
    """

    entity_id: str
    mode: str
    observations: tuple[Observation, ...]
    text: str | None
    fragment_dedupe: FragmentDedupeResult
    inferred_ratio: float | None = None
    notes: tuple[str, ...] = ()
    sentences: tuple[SynthesizedSentence, ...] = ()

    def rollback(self, sentence_index: int) -> tuple[Observation, ...]:
        """Observations supporting sentence ``sentence_index`` (旧观察不删的回落口)."""
        sentence = self.sentences[sentence_index]
        fallbacks = set(sentence.fallbacks)
        return tuple(obs for obs in self.observations if obs.text in fallbacks)


def _dedupe_observations(
    observations: Iterable[Observation | str],
) -> tuple[tuple[Observation, ...], FragmentDedupeResult]:
    coerced = tuple(_coerce_observation(item) for item in observations)
    dedupe = dedupe_fragments(tuple(obs.text for obs in coerced))
    kept: list[Observation] = []
    seen: set[str] = set()
    for obs in coerced:
        normalized = normalize_fragment(obs.text)
        if not normalized:
            continue
        key = fragment_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        kept.append(obs)
    return tuple(kept), dedupe


def _observations_only_result(
    entity_id: str,
    observations: tuple[Observation, ...],
    dedupe: FragmentDedupeResult,
    *,
    notes: tuple[str, ...],
) -> DescribeResult:
    return DescribeResult(
        entity_id=entity_id,
        mode="observations_only",
        observations=observations,
        text=None,
        fragment_dedupe=dedupe,
        inferred_ratio=None,
        notes=notes,
        sentences=(),
    )


def describe(
    entity_id: str,
    observations: Iterable[Observation | str],
    *,
    synthesizer: DescriptionSynthesizer | None = None,
) -> DescribeResult:
    """Dedupe observations and optionally synthesise a description.

    Without a synthesiser (or when the synthesiser reports no LLM / returns
    nothing) the result is ``mode="observations_only"``: no model is called and
    no text is fabricated.  With a synthesiser the returned ``text`` is the
    synthesised description, ``sentences`` carries a per-sentence observation
    fallback, and ``inferred_ratio`` is computed from the sentences that could
    **not** be traced back to any observation.
    """
    if not isinstance(entity_id, str):
        raise TypeError(f"entity_id must be str, got {type(entity_id).__name__}")
    kept, dedupe = _dedupe_observations(observations)
    dedupe_note = (
        f"dropped {len(dedupe.dropped)} duplicate fragment(s), "
        f"ignored {len(dedupe.ignored)} empty fragment(s)"
    )

    if synthesizer is None:
        return _observations_only_result(
            entity_id,
            kept,
            dedupe,
            notes=(
                "no synthesizer wired; returning deduplicated observations only",
                dedupe_note,
            ),
        )

    if not kept:
        return _observations_only_result(
            entity_id,
            kept,
            dedupe,
            notes=("no observations to synthesise; returning empty observation list", dedupe_note),
        )

    try:
        text = synthesizer.synthesize(entity_id, tuple(obs.text for obs in kept))
    except SynthesisUnavailable as exc:
        return _observations_only_result(
            entity_id,
            kept,
            dedupe,
            notes=(
                f"synthesis unavailable: {exc}; returning deduplicated observations only",
                dedupe_note,
            ),
        )

    if not isinstance(text, str) or not text.strip():
        return _observations_only_result(
            entity_id,
            kept,
            dedupe,
            notes=(
                "synthesizer returned no text; returning deduplicated observations only",
                dedupe_note,
            ),
        )

    observation_texts = tuple(obs.text for obs in kept)
    sentence_records: list[SynthesizedSentence] = []
    for sentence in split_description_sentences(text):
        fallbacks = _sentence_fallbacks(sentence, observation_texts)
        sentence_records.append(
            SynthesizedSentence(
                text=sentence, fallbacks=fallbacks, inferred=not fallbacks
            )
        )
    if not sentence_records:
        return _observations_only_result(
            entity_id,
            kept,
            dedupe,
            notes=(
                "synthesizer returned no sentences; returning deduplicated observations only",
                dedupe_note,
            ),
        )

    inferred = sum(1 for sentence in sentence_records if sentence.inferred)
    inferred_ratio = round(inferred / len(sentence_records), 4)

    notes = [
        f"map-reduce synthesis over {len(kept)} observation(s)",
        f"{inferred}/{len(sentence_records)} sentence(s) inferred "
        f"(no observation fallback) = {inferred_ratio:.1%}",
        dedupe_note,
    ]
    stats = getattr(synthesizer, "stats", None)
    if callable(stats):
        try:
            counters = stats()
        except Exception:  # noqa: BLE001 - notes only
            counters = None
        if isinstance(counters, Mapping):
            notes.append(
                "synthesizer calls: "
                + ", ".join(f"{key}={value}" for key, value in counters.items())
            )

    return DescribeResult(
        entity_id=entity_id,
        mode="synthesized",
        observations=kept,
        text=text.strip(),
        fragment_dedupe=dedupe,
        inferred_ratio=inferred_ratio,
        notes=tuple(notes),
        sentences=tuple(sentence_records),
    )


def render_observations(
    observations: Iterable[Observation | str], *, bullet: str = "- "
) -> str:
    """Render observations as one bullet per line (debug/readability helper)."""
    lines = []
    for item in observations:
        obs = _coerce_observation(item)
        text = normalize_fragment(obs.text)
        if text:
            lines.append(f"{bullet}{text}")
    return "\n".join(lines)
