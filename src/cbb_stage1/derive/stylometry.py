"""Stylometry for the inference layer (§5.20 ``derived/inference/stylometry.jsonl``).

Why this one is first
---------------------
§5.20 lists five optional inference files and marks this one "**低幻觉，最先做**". It is
the only one of the five that needs no model: every number here is a count over the
unit's own characters, so there is nothing to hallucinate. The other four either call a
model (motivation) or depend on a method the resource library does not carry (emotion).

What is measured, and why these measures
----------------------------------------
Chinese prose has no spaces, so "word length" is only meaningful against the token
regex the rest of this codebase already uses (``[\\u4e00-\\u9fff]{1,4}|[A-Za-z]{2,}``,
the same one in ``derive/forecasts.py``). Everything else is character- or
sentence-level, where the numbers are stable:

* sentence length mean / p50 / p90 — rhythm, and the tail catches the long expository
  stretches that averages hide;
* punctuation density — how clipped the prose is;
* dialogue ratio — share of characters inside 「」/『』, which separates a talky chapter
  from a descriptive one;
* type-token ratio over tokens — vocabulary richness. Comparable only within similar
  lengths, so ``sample_chars`` travels with it.

Statistical basis, not evidence
-------------------------------
I27 requires every ``derived/inference/*`` row to carry evidence **or a statistical
basis**. These rows carry ``method`` + ``sample_chars`` + the raw counts, and no quote:
there is no single span that "is" a sentence-length distribution, and inventing one
would be fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from cbb_stage1 import ids

__all__ = ["METHOD", "Stylometry", "measure_unit", "build_stylometry"]

#: Named so a reader can tell what produced the numbers.
METHOD = "char_statistics"

#: Same token convention as ``derive/forecasts.py``: CJK runs up to 4 chars, or latin
#: words of 2+. Consistency matters more than linguistic purity here — two different
#: tokenisers in one project produce two numbers that cannot be compared.
_TOKEN = re.compile(r"[\u4e00-\u9fff]{1,4}|[A-Za-z]{2,}")

#: Sentence terminators for Chinese prose. Newlines count as boundaries too, because a
#: dialogue line usually ends without a full stop.
_SENTENCE_END = re.compile(r"[。！？…；!?;]|\n")

_PUNCT = re.compile(r"[，。！？…；：、『』「」（）《》〈〉—\-·,.!?;:\"'()\[\]{}]")

_QUOTE_SPAN = re.compile(r"[「『][^」』]*[」』]")


def _percentile(values: list[int], fraction: float) -> float:
    """Linear-interpolated percentile; returns 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in _SENTENCE_END.split(text)]
    return [piece for piece in pieces if piece]


@dataclass(frozen=True)
class Stylometry:
    """One unit's statistics; ``to_row`` shapes them for the contract."""

    book_id: str
    unit_id: str
    sample_chars: int
    sentence_lengths: list[int] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list)
    unique_tokens: int = 0
    punctuation_count: int = 0
    dialogue_chars: int = 0

    @property
    def token_count(self) -> int:
        return len(self.token_lengths)

    @property
    def sentence_count(self) -> int:
        return len(self.sentence_lengths)

    @property
    def dialogue_ratio(self) -> float:
        if not self.sample_chars:
            return 0.0
        return round(self.dialogue_chars / self.sample_chars, 6)

    @property
    def type_token_ratio(self) -> float:
        if not self.token_count:
            return 0.0
        return round(self.unique_tokens / self.token_count, 6)

    @property
    def punctuation_density(self) -> float:
        if not self.sample_chars:
            return 0.0
        return round(self.punctuation_count / self.sample_chars, 6)

    def metrics(self) -> dict[str, Any]:
        lengths = self.sentence_lengths
        token_lengths = self.token_lengths
        return {
            "sentence_len_mean": (
                round(sum(lengths) / len(lengths), 6) if lengths else 0.0
            ),
            "sentence_len_p50": round(_percentile(lengths, 0.5), 6),
            "sentence_len_p90": round(_percentile(lengths, 0.9), 6),
            "word_len_mean": (
                round(sum(token_lengths) / len(token_lengths), 6) if token_lengths else 0.0
            ),
            "punctuation_density": self.punctuation_density,
            "dialogue_ratio": self.dialogue_ratio,
            "type_token_ratio": self.type_token_ratio,
            "sentence_count": self.sentence_count,
        }

    def to_row(self, *, created_at: str | None = None) -> dict[str, Any]:
        return {
            "style_id": ids.style_id(self.book_id, self.unit_id),
            "book_id": self.book_id,
            "unit_id": self.unit_id,
            "method": METHOD,
            "sample_chars": self.sample_chars,
            "metrics": self.metrics(),
            "note": (
                "字符级统计；无模型参与，无证据引文（I27 允许统计口径替代 evidence）"
            ),
            "produced_by": "derive:stylometry",
            "created_at": created_at,
        }


def measure_unit(book_id: str, unit_id: str, text: str) -> Stylometry:
    """Measure one unit. Pure function of the text; no model, no I/O."""
    body = text or ""
    tokens = _TOKEN.findall(body)
    token_lengths = [len(token) for token in tokens]
    dialogue_chars = sum(len(match.group(0)) for match in _QUOTE_SPAN.finditer(body))
    return Stylometry(
        book_id=book_id,
        unit_id=unit_id,
        sample_chars=len(body),
        sentence_lengths=[len(sentence) for sentence in _sentences(body)],
        token_lengths=token_lengths,
        unique_tokens=len(set(tokens)),
        punctuation_count=len(_PUNCT.findall(body)),
        dialogue_chars=dialogue_chars,
    )


def build_stylometry(
    book_id: str,
    units: Iterable[Mapping[str, Any]],
    *,
    texts: Mapping[str, str] | None = None,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """One row per unit that has text.

    ``units`` are the ``plan/units.jsonl`` records; ``texts`` maps unit id to its body.
    A unit without text yields no row rather than a row of zeros — a zero-filled row
    would look like a measurement.
    """
    lookup = texts or {}
    rows: list[dict[str, Any]] = []
    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        text = lookup.get(unit_id)
        if not text:
            continue
        rows.append(measure_unit(book_id, unit_id, text).to_row(created_at=created_at))
    rows.sort(key=lambda row: row["style_id"])
    return rows
