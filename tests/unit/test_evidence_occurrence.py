"""`occurrence` is part of the evidence contract, so verification must check it.

Measured gap
------------
`evidence/anchor.py:149-161` recomputes `start`/`end`, the quote and the quote's sha — and stops
there. `occurrence` (1-based, "this is the n-th time the quote appears") is written when the
evidence is created and never re-checked. So an evidence row can claim to be the second
occurrence of a sentence while pointing at the first, and every gate in the pipeline passes it.

`anchor.py`'s own module docstring says what it is for:

    "if the quote repeats, pick the occurrence nearest to the provided anchor ... record which
     occurrence it was (1-based)"

Recording it and never verifying it is the same shape as the other drifts in this audit: the
field exists so the artifact looks complete.

Why it matters
--------------
Two different events often quote the same sentence. If both rows claim `occurrence=1` and point at
the same offset, they are indistinguishable — one of the two events silently becomes a duplicate of
the other, and the box cannot tell them apart.
"""

from __future__ import annotations

import pytest

from cbb_stage1.evidence.anchor import sha256_text, verify_evidence
from cbb_stage1.models import Evidence

TEXT = "甲乙丙丁。甲乙丙丁。后面还有别的句子。"
QUOTE = "甲乙丙丁"


def _evidence(start: int, occurrence: int, quote: str = QUOTE) -> Evidence:
    return Evidence(
        unit_id="u:tst:v01:c001",
        start=start,
        end=start + len(quote),
        quote=quote,
        occurrence=occurrence,
        quote_sha256=sha256_text(quote),
    )


# ══ the first occurrence ═══════════════════════════════════════════════
def test_the_first_occurrence_verifies() -> None:
    assert verify_evidence(TEXT, _evidence(0, 1))


def test_the_second_occurrence_verifies() -> None:
    assert verify_evidence(TEXT, _evidence(5, 2))


# ══ a wrong occurrence must be caught ══════════════════════════════════
def test_claiming_the_second_occurrence_at_the_first_position_fails() -> None:
    """The defect: span and quote match, so this used to pass."""
    assert not verify_evidence(TEXT, _evidence(0, 2)), (
        "an evidence row pointing at the first occurrence while claiming the second was accepted"
    )


def test_claiming_the_first_occurrence_at_the_second_position_fails() -> None:
    assert not verify_evidence(TEXT, _evidence(5, 1))


def test_an_occurrence_beyond_the_number_of_real_occurrences_fails() -> None:
    assert not verify_evidence(TEXT, _evidence(0, 3))


def test_a_zero_or_negative_occurrence_fails() -> None:
    for bad in (0, -1):
        assert not verify_evidence(TEXT, _evidence(0, bad)), f"occurrence={bad} was accepted"


# ══ the existing checks must keep working ═════════════════════════════
def test_a_span_that_does_not_match_the_quote_still_fails() -> None:
    broken = Evidence(
        unit_id="u:tst:v01:c001",
        start=0,
        end=4,
        quote="完全不同的引文",
        occurrence=1,
    )
    assert not verify_evidence(TEXT, broken)


def test_a_span_out_of_range_still_fails() -> None:
    assert not verify_evidence(TEXT, _evidence(999, 1))


def test_a_wrong_sha_still_fails() -> None:
    bad = Evidence(
        unit_id="u:tst:v01:c001",
        start=0,
        end=4,
        quote=QUOTE,
        occurrence=1,
        quote_sha256="0" * 64,
    )
    assert not verify_evidence(TEXT, bad)


@pytest.mark.parametrize("start,occurrence", [(0, 1), (5, 2)])
def test_the_happy_paths_are_unaffected(start: int, occurrence: int) -> None:
    assert verify_evidence(TEXT, _evidence(start, occurrence))
