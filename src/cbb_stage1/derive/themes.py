"""Theme / motif counting for the inference layer (§5.20 ``themes.jsonl``).

Method, from F02-MotifDistantReading-2026
-----------------------------------------
The paper's distant-reading approach is deliberately plain:

> a computational analysis of the statistics of word occurrences based on a digitized
> literary corpus
> To extract motifs, a search will be conducted for the five most frequently occurring
> words across all periods, including their frequency of occurrence within each period.

Count word occurrences, group by period (here: unit), keep what recurs. No model, no
semantic clustering, no synthesis — which is why this file's hallucination risk is zero.

Why jieba and not a regex
-------------------------
The first version tokenised with ``[\\u4e00-\\u9fff]{2,8}``. On real text the top
"motifs" came out as clause fragments —

    保护他才对 / 他一定很痛苦 / 而我甚至连这种事

— because in Chinese a greedy CJK run is a clause, not a word. The project already
solves this in ``engines/ai_reader/entity_pre_scanner`` with jieba POS tagging, keeping
nouns of 2-8 characters; this module borrows the same approach and the same stopword
list rather than inventing a second one.

Statistical basis, not evidence
-------------------------------
I27 allows a statistical basis in place of an evidence quote. A frequency count over a
whole unit has no single span that "is" the count, so rows carry ``method`` plus per-unit
counts; inventing a quote would be fabrication.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Mapping

from cbb_stage1 import ids

__all__ = ["DEFAULT_TOP_N", "METHOD", "build_themes", "count_motifs"]

logger = logging.getLogger(__name__)

#: Named so a reader can tell what produced the numbers.
METHOD = "word_frequency_by_segment"

#: How many motifs to keep per book. F02 reports the top five; this is a summary, not
#: an index, and each row still carries its own ``frequency``.
DEFAULT_TOP_N = 20

#: Up to this many characters are tokenised; beyond it jieba's cost grows with no gain
#: in motif quality. Same cap the pre-scanner uses.
_TEXT_CAP = 1_000_000


def _stopwords() -> set[str]:
    """The project's stopword list, borrowed rather than duplicated.

    It lives beside the entity pre-scanner because that is where the need first
    appeared. Importing it keeps one source of truth; a second list would eventually
    disagree with this one about the same word.
    """
    from cbb_stage1.engines.ai_reader.entity_pre_scanner import _STOPWORDS

    return set(_STOPWORDS)


def _tokens(text: str) -> Counter:
    """Nouns of 2-8 characters, via jieba POS tagging.

    Returns an empty Counter when jieba is unavailable. That is deliberate: a regex
    fallback would silently produce clause fragments again, and a caller cannot tell
    the two apart from the output.
    """
    try:
        import jieba.posseg as pseg
    except ImportError:
        logger.warning("jieba unavailable: themes table left empty rather than guessed")
        return Counter()

    stop = _stopwords()
    counter: Counter = Counter()
    for word, flag in pseg.cut(text[:_TEXT_CAP]):
        if len(word) < 2 or len(word) > 8:
            continue
        if word in stop:
            continue
        if flag.startswith(("nr", "ns", "nz", "n")):
            counter[word] += 1
    return counter


def count_motifs(
    texts: Mapping[str, str],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> list[tuple[str, int, dict[str, int]]]:
    """Return ``[(motif, total, {unit_id: count}), ...]`` sorted by total desc.

    Counts are per unit so a reader can see whether a motif recurs across the book or
    is confined to one chapter — the point of grouping by period.
    """
    per_unit: dict[str, Counter] = {}
    totals: Counter = Counter()

    for unit_id, text in texts.items():
        if not text:
            continue
        counter = _tokens(text)
        if counter:
            per_unit[unit_id] = counter
            totals.update(counter)

    ordered: list[tuple[str, int, dict[str, int]]] = []
    for motif, total in totals.most_common(top_n):
        by_unit = {
            unit_id: counter[motif]
            for unit_id, counter in sorted(per_unit.items())
            if counter.get(motif)
        }
        ordered.append((motif, int(total), by_unit))
    return ordered


def build_themes(
    book_id: str,
    texts: Mapping[str, str],
    *,
    top_n: int = DEFAULT_TOP_N,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """One row per motif, carrying its per-unit distribution."""
    rows: list[dict[str, Any]] = []
    for motif, total, by_unit in count_motifs(texts, top_n=top_n):
        rows.append(
            {
                "theme_id": ids.theme_id(book_id, motif),
                "book_id": book_id,
                "motif": motif,
                "frequency": total,
                "units": list(by_unit),
                "unit_counts": by_unit,
                "method": METHOD,
                "note": (
                    "jieba 词性过滤后的名词词频（F02 distant reading 口径）；"
                    "跨单元分布见 unit_counts，无模型参与、无语义归纳"
                ),
                "produced_by": "derive:themes",
                "created_at": created_at,
            }
        )
    return rows
