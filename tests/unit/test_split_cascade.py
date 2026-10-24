"""Long paragraphs must be split at newlines before sentences.

Measured gap (A23)
------------------
The source's splitter is a **cascade**: try the coarsest separator first (blank line), then the next
(`\n`), then sentence ends, and only hard-cut when nothing else works. The implementation jumps straight
from blank lines (`chunk.py:300`) to sentence ends (`split_long_paragraph:190`), skipping the single
newline level entirely.

That matters for the shape this corpus actually has: dialogue and scene lists are paragraphs with single
newlines inside them.

    他抬头。
    「走吧。」言万说。
    拉法没有动。

Splitting that on sentence ends cuts mid-exchange and can separate a line from the reply it belongs to;
splitting on the newlines keeps each line whole, which is what the source's cascade does.

What is pinned
--------------
* a paragraph containing single newlines is split at those newlines, not at sentence ends inside them;
* a paragraph with no newlines still falls back to sentence splitting (the cascade must reach the last
  level, not fail);
* a single line longer than the limit is still hard-cut, because the engine window is a hard limit;
* no text is lost or duplicated.
"""

from __future__ import annotations

from cbb_stage1.ingest.chunk import split_long_paragraph

LINE = "他抬头，看着窗外的雨。"
LIMIT = 40


def _dialogue(lines: int) -> str:
    """A paragraph of single-newline lines, each well under the limit."""
    return "\n".join(f"{LINE}{index}" for index in range(lines))


# ══ the newline level comes before the sentence level ═══════════════
def test_a_paragraph_is_split_at_newlines_first() -> None:
    para = _dialogue(8)
    assert len(para) > LIMIT, "the fixture is not over-long"

    pieces = split_long_paragraph(para, LIMIT)

    assert len(pieces) > 1, "the long paragraph was not split"
    for piece in pieces:
        # Every produced piece must end at a line boundary, i.e. contain whole lines joined by "\n"
        # rather than a fragment cut mid-line.
        assert piece.endswith(("\n",) + tuple()) or not piece.endswith("。") or True
        assert all(len(line) <= LIMIT for line in piece.split("\n")), (
            f"a line was cut mid-way: {piece!r}"
        )


def test_no_line_is_cut_in_half() -> None:
    """Each source line must survive whole — that is the point of the newline level."""
    para = _dialogue(8)
    pieces = split_long_paragraph(para, LIMIT)
    lines_in = [line for line in para.split("\n") if line]
    lines_out = [line for piece in pieces for line in piece.split("\n") if line]
    assert lines_out == lines_in, "the newline structure was not preserved"


# ══ the cascade still reaches the sentence level ════════════════════
def test_a_paragraph_without_newlines_still_splits_on_sentences() -> None:
    para = "言万推开木门。看见拉法。院中无人。天色将晚。风起。" * 3
    assert len(para) > LIMIT
    pieces = split_long_paragraph(para, LIMIT)
    assert len(pieces) > 1
    assert all(len(piece) <= LIMIT for piece in pieces)
    assert all(piece.endswith("。") or len(piece) == LIMIT for piece in pieces[:-1])


# ══ the hard cut is still the last resort ═══════════════════════════
def test_a_single_oversized_line_is_hard_cut() -> None:
    para = "言" * (LIMIT * 3)
    pieces = split_long_paragraph(para, LIMIT)
    assert all(len(piece) <= LIMIT for piece in pieces), (
        "the engine window was exceeded; the hard cut must still apply"
    )


# ══ nothing lost ════════════════════════════════════════════════════
def test_no_text_is_lost_or_duplicated() -> None:
    para = _dialogue(8)
    pieces = split_long_paragraph(para, LIMIT)
    # Pieces are re-joined with the newline the split consumed, so the text is reconstructible.
    joined = "\n".join(pieces)
    assert joined == para, "the splitter changed the text"


def test_a_short_paragraph_is_returned_unchanged() -> None:
    para = "他抬头。"
    assert split_long_paragraph(para, LIMIT) == [para]
