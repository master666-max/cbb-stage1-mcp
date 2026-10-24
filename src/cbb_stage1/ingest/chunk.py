"""Chapter chunking - four strategies (structural / fixed / title / semantic).

Adapted from two read-only references:

* ``B09-BookContentGenerator`` ``CHUNK_SIZE`` / ``CHUNK_OVERLAP`` continuous
  chunking (the shell we are porting);
* ``B07-lorebase/scripts/01_chunking.py`` (structure-first boundaries, greedy
  paragraph aggregation, sentence fallback for long paragraphs, previous-tail
  overlap).

Strategies (``chunk_text(..., strategy=...)`` / ``chunk_chapter``):

``structural`` (default, v6 rule ``02`` §10.3)
    **Try the whole structural chapter first** (big-first); only split when it
    exceeds the engine window, aggregate paragraphs toward ``target``, split
    over-long paragraphs on sentence boundaries, carry the previous tail as
    overlap. Reconstruction: ``"\\n\\n".join(body_text) == normalize(text)``.

``fixed``
    Fixed windows of ``target_chars`` with ``overlap_chars`` step overlap, no
    structure awareness (hard cuts allowed). Reconstruction: ``"".join(...)``.

``title``
    Split at *heading boundaries* inside the chapter (``## …``, ``第X节 ``,
    ``一、 …``, ``序章/楔子/尾声/后记/番外`` and ``***`` separator lines), then
    pack whole sections toward ``target``; a heading is never cut from its
    section. Falls back to ``structural`` when the chapter has no headings.
    Reconstruction: ``"".join(...)``.

``semantic``
    **Lexical-semantic** clustering (no embeddings): paragraphs are grouped by
    adjacent character-ngram Jaccard similarity, so topically related
    paragraphs land in one chunk and topic shifts start a new one. The label is
    deliberately "lexical-semantic" - this is a word-surface proxy, not a
    vector semantic split. Reconstruction: ``"\\n\\n".join(...)``.

All sizes are characters; token values are converted with a documented
approximation (CJK ≈ 1.5 chars/token for chunk sizing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_TARGET_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "DEFAULT_MAX_CHARS",
    "CHUNK_STRATEGIES",
    "DEFAULT_STRATEGY",
    "DEFAULT_SEMANTIC_THRESHOLD",
    "Chunk",
    "tokens_to_chars",
    "chars_to_tokens_approx",
    "normalize_paragraphs",
    "split_long_paragraph",
    "chunk_text",
    "chunk_chapter",
]

#: Conservative conversion for chunk sizing (Chinese prose ≈ 1-1.5 chars/token).
CHARS_PER_TOKEN = 1.5

#: Approximation of ``Config.chunk_target_tokens`` (1200 tokens).
DEFAULT_TARGET_CHARS = 1800
#: Approximation of ``Config.chunk_overlap_tokens`` (100 tokens).
DEFAULT_OVERLAP_CHARS = 150
#: Engine-window guard: above this the chapter is split (≈8000 tokens).
DEFAULT_MAX_CHARS = 12000

#: Supported strategies (A23: four chunk strategies).
CHUNK_STRATEGIES = ("structural", "fixed", "title", "semantic")
DEFAULT_STRATEGY = "structural"

#: Adjacent-paragraph lexical similarity needed to stay in one semantic chunk.
DEFAULT_SEMANTIC_THRESHOLD = 0.12

#: Split points *after* sentence-final punctuation (keeps the punctuation).
_SENT_SPLIT = re.compile(r"(?<=[。！？…!?；;])|(?<=[」』”’])")
#: First sentence boundary inside an overlap window.
_SENT_END = re.compile(r"[。！？…!?；;」』”’]\s*")

#: Heading lines that start a new title-aware section.
_HEADING_LINE = re.compile(
    r"^[ \t]*(?:"
    r"#{1,6}[ \t]+\S[^\n]*"
    r"|第[0-9一二三四五六七八九十百千零两]+[章节回卷篇节][^\n]*"
    r"|[一二三四五六七八九十]+[、.．][^\n]+"
    r"|(?:序章|楔子|尾声|后记|番外)[^\n]*"
    r")[ \t]*$",
    re.MULTILINE,
)
#: Decorative separator lines; keep them at the end of the previous section.
_SEPARATOR_LINE = re.compile(
    r"^[ \t]*(?:[*＊]{3,}|[-—_＝=]{3,}|[·•]{3,})[ \t]*$", re.MULTILINE
)


def tokens_to_chars(tokens: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Convert a token budget to a character budget (round, at least 1)."""
    return max(1, int(round(tokens * chars_per_token)))


def chars_to_tokens_approx(chars: int, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Inverse approximation of :func:`tokens_to_chars`."""
    return max(1, int(round(chars / chars_per_token)))


def validate_chunk_strategy(strategy: str) -> str:
    """Return *strategy* if supported, else raise ``ValueError``."""
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(
            f"unknown chunk strategy {strategy!r}; expected one of {CHUNK_STRATEGIES}"
        )
    return strategy


def normalize_paragraphs(text: str) -> str:
    """Normalize paragraph separators to exactly ``\\n\\n``.

    Chunking invariants: structural/semantic blocks are assembled from whole
    paragraphs, so ``"\\n\\n".join(chunk.body_text for chunk in chunks) ==
    normalize_paragraphs(text)``; when an over-long paragraph had to be split,
    its sentence pieces are contiguous, so ``"".join(...)`` reconstructs it.
    ``fixed``/``title`` strategies reconstruct with ``"".join(...)``.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text)]
    return "\n\n".join(part for part in paragraphs if part)


@dataclass(frozen=True)
class Chunk:
    """One chunk of a chapter.

    Attributes:
        body_text: Chunk text without the overlap prefix (used for
            reconstruction and for byte-accurate unit text when no split).
        text: Chunk text as the engine sees it (overlap prefix + body).
        seq: Zero-based chunk index within the chapter.
        overlap_chars: Length of the overlap prefix in ``text``.
        split_reason: Why this chunk exists: ``big_first`` |
            ``structural_aggregate`` | ``sentence_fallback`` | ``fixed_window`` |
            ``title_section`` | ``semantic_cluster``.
        strategy: Chunking strategy that produced this chunk
            (``structural`` / ``fixed`` / ``title`` / ``semantic``).
    """

    body_text: str
    text: str
    seq: int
    overlap_chars: int
    split_reason: str
    strategy: str = DEFAULT_STRATEGY

    @property
    def char_len(self) -> int:
        return len(self.text)

    @property
    def body_chars(self) -> int:
        return len(self.body_text)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "text": self.text,
            "body_text": self.body_text,
            "char_len": self.char_len,
            "body_chars": self.body_chars,
            "overlap_chars": self.overlap_chars,
            "split_reason": self.split_reason,
            "strategy": self.strategy,
        }


def split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph through a separator cascade.

    Levels, coarsest first — the same order the source's recursive splitter uses:

    1. ``\\n`` — keep whole lines and pack them under the limit. Dialogue and scene lists are
       paragraphs of single-newline lines, and cutting them at a ``。`` inside a line throws the
       structure away (measured: ``他抬头，看着窗外的雨。2`` became ``…。`` plus a stray ``2``);
    2. sentence ends — for a line that is itself too long;
    3. a hard cut at ``max_chars`` — only when a single sentence exceeds the limit, because the engine
       window is a hard bound rather than a preference.
    """
    if len(para) <= max_chars:
        return [para]

    if "\n" in para:
        packed = _pack_lines(para, max_chars)
        # A line can still be too long: hand just those to the sentence level.
        result: list[str] = []
        for piece in packed:
            if len(piece) <= max_chars:
                result.append(piece)
            else:
                result.extend(_split_sentences(piece, max_chars))
        return result or [para]

    return _split_sentences(para, max_chars)


def _pack_lines(para: str, max_chars: int) -> list[str]:
    """Pack whole lines into pieces under ``max_chars``, never cutting a line."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in para.split("\n"):
        addition = len(line) + (1 if current else 0)
        if current and size + addition > max_chars:
            pieces.append("\n".join(current))
            current, size = [line], len(line)
        else:
            current.append(line)
            size += addition
    if current:
        pieces.append("\n".join(current))
    return pieces


def _split_sentences(para: str, max_chars: int) -> list[str]:
    """Sentence-level split; hard-cuts a single sentence that exceeds the limit."""
    sentences = [part for part in _SENT_SPLIT.split(para) if part]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces or [para]


def _tail_window(body: str, overlap_chars: int) -> str:
    """Return up to ``overlap_chars`` chars from the end of ``body``.

    Starts at a sentence boundary inside the window when one exists, so the
    overlap does not begin mid-sentence.
    """
    if overlap_chars <= 0 or not body:
        return ""
    if len(body) <= overlap_chars:
        return body
    candidate = body[-overlap_chars:]
    match = _SENT_END.search(candidate)
    if match and match.end() < len(candidate):
        aligned = candidate[match.end() :]
        if aligned.strip():
            return aligned
    return candidate


# ── shared assembly ─────────────────────────────────────────────


def _finalize_chunks(
    parts: list[tuple[str, str]],
    *,
    overlap_chars: int,
    strategy: str,
) -> list[Chunk]:
    """Turn ``(body, split_reason)`` parts into ordered chunks with overlap."""
    chunks: list[Chunk] = []
    previous_body = ""
    for seq, (body, reason) in enumerate(parts):
        if seq == 0 or overlap_chars <= 0:
            text_out, overlap_len = body, 0
        else:
            tail = _tail_window(previous_body, overlap_chars)
            overlap_len = len(tail)
            text_out = f"{tail}\n\n{body}" if tail else body
        chunks.append(
            Chunk(
                body_text=body,
                text=text_out,
                seq=seq,
                overlap_chars=overlap_len,
                split_reason=reason,
                strategy=strategy,
            )
        )
        previous_body = body
    return chunks


def _merge_tiny_tail(
    parts: list[tuple[str, str]],
    *,
    max_chars: int,
    min_chars: int,
    joiner: str,
) -> list[tuple[str, str]]:
    """Merge a tiny final body into the previous one when it still fits.

    ``min_chars <= 0`` disables merging (used by the boundary-preserving
    ``title`` / ``semantic`` strategies unless the caller asks for it).
    """
    if min_chars <= 0 or len(parts) <= 1:
        return parts
    last_body = parts[-1][0]
    if len(last_body) >= min_chars:
        return parts
    merged_body = joiner.join((parts[-2][0], last_body))
    if len(merged_body) <= max_chars:
        merged_reason = parts[-1][1]
        merged = parts[:-2] + [(merged_body, merged_reason)]
        return merged
    return parts


# ── structural (default; behaviour preserved from v6 shell) ─────


def _structural_parts(
    normalized: str,
    *,
    target_chars: int,
    max_chars: int,
    min_chars: int | None,
) -> list[tuple[str, str]]:
    if len(normalized) <= max_chars:
        return [(normalized, "big_first")]

    paragraphs = normalized.split("\n\n")
    pieces: list[str] = []
    sentence_fallback = False
    for para in paragraphs:
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            sentence_fallback = True
            pieces.extend(split_long_paragraph(para, max_chars))

    blocks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        # Keep the assembled block under the engine window.
        if current and size + len(piece) + 2 > max_chars:
            blocks.append(current)
            current, size = [], 0
        size += len(piece) + (2 if current else 0)
        current.append(piece)
        if size >= target_chars:
            blocks.append(current)
            current, size = [], 0
    if current:
        blocks.append(current)

    # Merge a tiny tail into the previous block when it still fits the window.
    if len(blocks) > 1:
        merged_min = min_chars if min_chars is not None else max(1, target_chars // 3)
        if sum(len(part) for part in blocks[-1]) + 2 * (len(blocks[-1]) - 1) < merged_min:
            merged = blocks[-2] + blocks[-1]
            merged_len = sum(len(part) for part in merged) + 2 * (len(merged) - 1)
            if merged_len <= max_chars:
                blocks[-2].extend(blocks.pop())

    blocks = _attach_orphan_headings(blocks)

    reason = (
        "big_first"
        if len(blocks) == 1
        else ("sentence_fallback" if sentence_fallback else "structural_aggregate")
    )
    return [("\n\n".join(block), reason) for block in blocks]


def _is_heading_only(parts: list[str]) -> bool:
    """True when a block's entire content is one heading line."""
    joined = "\n\n".join(part.strip() for part in parts if part.strip())
    if not joined or "\n" in joined:
        return False
    return bool(_HEADING_LINE.match(joined))


def _attach_orphan_headings(blocks: list[list[str]]) -> list[list[str]]:
    """Move a heading-only block onto the front of the block that follows it.

    A13 ②: with a heading followed by an oversized paragraph, the paragraph is split away and the heading
    is left as a chunk of its own — no text to extract from, and the body chunk loses its section. The
    structural boundaries are otherwise untouched, because the big-first rule is a v6 decision.

    A trailing heading with nothing after it stays put: there is nowhere to move it, and dropping it
    would lose content.
    """
    if len(blocks) < 2:
        return blocks
    result: list[list[str]] = []
    carry: list[str] = []
    for block in blocks:
        block = carry + block
        carry = []
        if _is_heading_only(block):
            carry = block
            continue
        result.append(block)
    if carry:
        # A heading at the very end: keep it rather than lose it.
        if result:
            result[-1] = result[-1] + carry
        else:
            result.append(carry)
    return result


# ── fixed window ────────────────────────────────────────────────


def _fixed_parts(
    normalized: str,
    *,
    target_chars: int,
    overlap_chars: int,
    max_chars: int,
    min_chars: int | None,
) -> list[tuple[str, str]]:
    window = max(1, min(target_chars, max_chars))
    effective_overlap = max(0, min(overlap_chars, window - 1))
    step = max(1, window - effective_overlap)
    min_tail = min_chars if min_chars is not None else max(1, window // 3)

    length = len(normalized)
    starts = [0]
    position = 0
    while position + window < length:
        position += step
        starts.append(position)

    # Merge a tiny tail window into its predecessor when it still fits.
    while len(starts) > 1 and length - starts[-1] < min_tail:
        if length - starts[-2] > max_chars:
            break
        starts.pop()

    parts: list[tuple[str, str]] = []
    for index, start in enumerate(starts):
        end = length if index == len(starts) - 1 else start + window
        parts.append((normalized[start:end], "fixed_window"))
    return parts


# ── title boundary ──────────────────────────────────────────────


def _title_sections(normalized: str) -> list[tuple[int, int]]:
    """Return section spans split at heading / separator boundaries."""
    boundaries: set[int] = set()
    for match in _HEADING_LINE.finditer(normalized):
        if match.start() > 0:
            boundaries.add(match.start())
    for match in _SEPARATOR_LINE.finditer(normalized):
        if 0 < match.end() < len(normalized):
            boundaries.add(match.end())

    starts = [0] + sorted(boundaries)
    sections: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(normalized)
        if end > start:
            sections.append((start, end))
    return sections


def _title_parts(
    normalized: str,
    *,
    target_chars: int,
    max_chars: int,
    min_chars: int | None,
) -> list[tuple[str, str]] | None:
    sections = _title_sections(normalized)
    if len(sections) <= 1:
        return None  # no usable headings -> caller falls back to structural

    # Split over-long sections but keep every span contiguous.
    spans: list[tuple[int, int]] = []
    for start, end in sections:
        section = normalized[start:end]
        if len(section) <= max_chars:
            spans.append((start, end))
            continue
        offset = start
        for piece in split_long_paragraph(section, max_chars):
            spans.append((offset, offset + len(piece)))
            offset += len(piece)

    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    size = 0
    for span in spans:
        span_len = span[1] - span[0]
        if current and size + span_len > target_chars:
            groups.append(current)
            current, size = [], 0
        current.append(span)
        size += span_len
    if current:
        groups.append(current)

    parts = [
        (normalized[group[0][0]: group[-1][1]], "title_section") for group in groups
    ]
    # Heading boundaries are semantic; only merge a tiny tail when explicitly
    # requested (min_chars), never by default.
    tiny = min_chars if min_chars is not None else 0
    return _merge_tiny_tail(parts, max_chars=max_chars, min_chars=tiny, joiner="")


# ── semantic (lexical) clustering ──────────────────────────────


def _lexical_tokens(paragraph: str) -> frozenset[str]:
    """Character n-gram set used as the lexical-semantic similarity proxy."""
    compact = re.sub(r"\s+", "", paragraph)
    if len(compact) < 3:
        return frozenset(compact) if compact else frozenset()
    return frozenset(compact[i:i + 3] for i in range(len(compact) - 2))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _semantic_parts(
    normalized: str,
    *,
    target_chars: int,
    max_chars: int,
    min_chars: int | None,
    threshold: float,
) -> list[tuple[str, str]] | None:
    paragraphs = normalized.split("\n\n")
    if len(paragraphs) <= 1:
        return None  # single paragraph -> structural fallback

    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(split_long_paragraph(paragraph, max_chars))

    groups: list[list[str]] = []
    current = [pieces[0]]
    current_len = len(pieces[0])
    previous_tokens = _lexical_tokens(pieces[0])
    for piece in pieces[1:]:
        tokens = _lexical_tokens(piece)
        similarity = _jaccard(previous_tokens, tokens)
        fits = current_len + len(piece) + 2 <= max_chars
        if similarity >= threshold and fits:
            current.append(piece)
            current_len += len(piece) + 2
        else:
            groups.append(current)
            current = [piece]
            current_len = len(piece)
        previous_tokens = tokens
    groups.append(current)

    parts: list[tuple[str, str]] = []
    for group in groups:
        body = "\n\n".join(group)
        if len(body) > max_chars:  # safety: should not happen, see `fits`
            for piece in split_long_paragraph(body, max_chars):
                parts.append((piece, "semantic_cluster"))
        else:
            parts.append((body, "semantic_cluster"))

    tiny = min_chars if min_chars is not None else 0
    return _merge_tiny_tail(parts, max_chars=max_chars, min_chars=tiny, joiner="\n\n")


# ── public API ──────────────────────────────────────────────────


def chunk_text(
    text: str,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int | None = None,
    strategy: str = DEFAULT_STRATEGY,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> list[Chunk]:
    """Chunk one chapter with the requested strategy.

    Args:
        text: Chapter text (normalized internally with
            :func:`normalize_paragraphs`).
        target_chars: Target size of one chunk.
        overlap_chars: Previous-tail overlap carried into the next chunk.
        max_chars: Engine-window guard; no body may exceed it.
        min_chars: Tail-merge threshold (defaults to ``target_chars // 3``).
        strategy: ``structural`` (default) / ``fixed`` / ``title`` / ``semantic``.
        semantic_threshold: Adjacent-paragraph lexical Jaccard needed by the
            ``semantic`` strategy.

    Returns:
        Ordered chunks; ``[]`` for empty text.

    Raises:
        ValueError: On an unknown strategy.
    """
    validate_chunk_strategy(strategy)
    normalized = normalize_paragraphs(text)
    if not normalized:
        return []

    parts: list[tuple[str, str]] | None
    if strategy == "structural":
        parts = _structural_parts(
            normalized,
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
        )
    elif strategy == "fixed":
        parts = _fixed_parts(
            normalized,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            max_chars=max_chars,
            min_chars=min_chars,
        )
    elif strategy == "title":
        parts = _title_parts(
            normalized,
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        if parts is None:
            parts = _structural_parts(
                normalized,
                target_chars=target_chars,
                max_chars=max_chars,
                min_chars=min_chars,
            )
    else:  # semantic
        parts = _semantic_parts(
            normalized,
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
            threshold=semantic_threshold,
        )
        if parts is None:
            parts = _structural_parts(
                normalized,
                target_chars=target_chars,
                max_chars=max_chars,
                min_chars=min_chars,
            )

    return _finalize_chunks(parts, overlap_chars=overlap_chars, strategy=strategy)


def chunk_chapter(
    text: str,
    *,
    target_tokens: int = 1200,
    overlap_tokens: int = 100,
    max_tokens: int = 8000,
    chars_per_token: float = CHARS_PER_TOKEN,
    strategy: str = DEFAULT_STRATEGY,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> list[Chunk]:
    """Chunk a chapter using token budgets (``Config``) converted to chars."""
    return chunk_text(
        text,
        target_chars=tokens_to_chars(target_tokens, chars_per_token),
        overlap_chars=tokens_to_chars(overlap_tokens, chars_per_token),
        max_chars=tokens_to_chars(max_tokens, chars_per_token),
        strategy=strategy,
        semantic_threshold=semantic_threshold,
    )
