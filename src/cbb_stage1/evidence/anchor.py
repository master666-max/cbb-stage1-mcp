"""Exact quote -> span anchoring.

Contract (design/02 §4.7):
- `quote == text[start:end]` must hold; a fuzzy match is NOT evidence.
- if the quote repeats, pick the occurrence nearest to the provided anchor
  (`near`, usually a mention offset); record which occurrence it was (1-based).
- `quote_sha256 = sha256(quote.encode("utf-8"))`.
- a quote that cannot be located returns None; callers must degrade the record
  (needs_evidence / inferred), never fabricate an offset.
"""

from __future__ import annotations

from dataclasses import dataclass

from cbb_stage1.ids import sha256_text as _ids_sha256_text
from cbb_stage1.models import Evidence, Speaker


def sha256_text(text: str) -> str:
    """Bare sha256 hex of the UTF-8 bytes of *text* (compatibility wrapper).

    Delegates to :func:`cbb_stage1.ids.sha256_text` - the single formula source.
    The ID module returns the box-shaped ``sha256:<hex>`` string, while
    ``Evidence.quote_sha256`` (and its consumers in assemble/claims/reasoning)
    store the bare hex and add the ``sha256:`` prefix at the box boundary, so
    the prefix is stripped here to keep the existing byte contract.
    """
    return _ids_sha256_text(text).removeprefix("sha256:")


def find_occurrences(text: str, needle: str) -> list[int]:
    """All start offsets where *needle* occurs in *text* (overlapping included)."""
    if not needle:
        return []
    starts: list[int] = []
    start = text.find(needle)
    while start >= 0:
        starts.append(start)
        start = text.find(needle, start + 1)
    return starts


def choose_occurrence(
    text: str,
    quote: str,
    *,
    near: int | None = None,
) -> tuple[int, int, int] | None:
    """Locate *quote* and return ``(start, end, occurrence)``.

    ``occurrence`` is 1-based.  When *near* is given, the occurrence whose
    start offset is closest to it wins; otherwise the first one wins.
    Exact quote first, then the stripped quote (LLMs often pad quotes);
    the returned span always matches the strings actually found.
    """
    if not quote:
        return None

    candidates = [quote]
    stripped = quote.strip()
    if stripped and stripped != quote:
        candidates.append(stripped)

    for candidate in candidates:
        starts = find_occurrences(text, candidate)
        if not starts:
            continue
        if near is None:
            index = 0
        else:
            index = min(range(len(starts)), key=lambda i: abs(starts[i] - near))
        start = starts[index]
        return start, start + len(candidate), index + 1
    return None


@dataclass(frozen=True)
class AnchoredQuote:
    """A quote that has been proven to exist in the unit text."""

    quote: str
    start: int
    end: int
    occurrence: int
    quote_sha256: str

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


def anchor_quote(
    text: str,
    quote: str,
    *,
    near: int | None = None,
) -> AnchoredQuote | None:
    """Anchor *quote* in *text*; return None when it is not verbatim present."""
    hit = choose_occurrence(text, quote, near=near)
    if hit is None:
        return None
    start, end, occurrence = hit
    matched = text[start:end]
    return AnchoredQuote(
        quote=matched,
        start=start,
        end=end,
        occurrence=occurrence,
        quote_sha256=sha256_text(matched),
    )


def make_evidence(
    unit_id: str,
    text: str,
    quote: str,
    *,
    near: int | None = None,
    speaker: Speaker | None = None,
    mode: str = "narration",
) -> Evidence | None:
    """Build a fully anchored :class:`Evidence`, or None if quote not found."""
    anchored = anchor_quote(text, quote, near=near)
    if anchored is None:
        return None
    return Evidence(
        unit_id=unit_id,
        start=anchored.start,
        end=anchored.end,
        quote=anchored.quote,
        quote_sha256=anchored.quote_sha256,
        occurrence=anchored.occurrence,
        speaker=speaker,
        mode=mode,  # type: ignore[arg-type]
    )


def locate_surface(
    text: str,
    surface: str,
    *,
    near: int | None = None,
) -> tuple[int, int, int] | None:
    """Locate an entity surface form; same rules as :func:`choose_occurrence`."""
    return choose_occurrence(text, surface, near=near)


def verify_evidence(text: str, evidence: Evidence) -> bool:
    """Recompute an :class:`Evidence` against the unit text.

    True only when all four hold:

    * ``0 <= start < end <= len(text)``;
    * the quote matches ``text[start:end]`` exactly;
    * the recorded sha matches the quote;
    * ``occurrence`` is the occurrence the span actually points at.

    The fourth check is new. ``occurrence`` was recorded at creation
    (:func:`choose_occurrence`) and never verified, so a row could claim to be the second time a
    sentence appears while pointing at the first — and two events quoting the same sentence became
    indistinguishable, which is the one thing the field exists to prevent.
    """
    if not (0 <= evidence.start < evidence.end <= len(text)):
        return False
    if text[evidence.start:evidence.end] != evidence.quote:
        return False
    if evidence.quote_sha256 and evidence.quote_sha256 != sha256_text(evidence.quote):
        return False
    occurrence = int(evidence.occurrence or 0)
    if occurrence < 1:
        return False
    starts = find_occurrences(text, evidence.quote)
    if len(starts) < occurrence:
        return False
    return starts[occurrence - 1] == evidence.start
