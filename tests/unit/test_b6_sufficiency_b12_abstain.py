"""B6 evidence sufficiency and B12 abstain — the two halves that were missing.

AVeriTeC asks two questions before a verdict: is there enough evidence, and does
it agree? Only the second was implemented. The consequence was that
``insufficient_evidence`` meant "the answers happened to be irrelevant" rather
than "we cannot tell", and a claim could be promoted while its confidence was
low, because nothing downstream read the semantic-entropy score.

These tests pin both halves, including the distinctions that matter:

* no evidence vs. low confidence are *different* reasons;
* an absent score is not a zero score;
* the threshold is a real number that changes the outcome.
"""

from __future__ import annotations

from cbb_stage1.claims.verdicts import (
    DEFAULT_ABSTAIN_THRESHOLD,
    abstain_decision,
    evidence_sufficiency,
)
from cbb_stage1.models import Evidence


def _ev(unit: str = "u:1", start: int = 0) -> Evidence:
    return Evidence(unit_id=unit, start=start, end=start + 5, quote="abcde", occurrence=1)


# ── B6 ────────────────────────────────────────────────────────────────────────
def test_no_evidence_is_insufficient() -> None:
    decision = evidence_sufficiency([])
    assert decision.sufficient is False
    assert decision.reason == "no_evidence"
    assert decision.status == "insufficient"


def test_single_item_is_enough_but_reports_its_source_count() -> None:
    decision = evidence_sufficiency([_ev()])
    assert decision.sufficient is True
    assert decision.evidence_count == 1
    assert decision.distinct_units == 1


def test_distinct_units_are_counted_not_duplicated() -> None:
    decision = evidence_sufficiency([_ev("u:1"), _ev("u:1", 10), _ev("u:2")])
    assert decision.evidence_count == 3
    assert decision.distinct_units == 2


def test_sufficiency_accepts_mappings_too() -> None:
    """The claims stage may hold dicts rather than Evidence models."""
    decision = evidence_sufficiency([{"unit_id": "u:9"}])
    assert decision.sufficient is True
    assert decision.distinct_units == 1


# ── B12 ───────────────────────────────────────────────────────────────────────
def test_low_confidence_abstains() -> None:
    decision = abstain_decision(0.2)
    assert decision.abstain is True
    assert decision.reason == "below_abstain_threshold"
    assert decision.confidence == 0.2
    assert decision.threshold == DEFAULT_ABSTAIN_THRESHOLD
    assert decision.verdict == "insufficient_evidence"


def test_high_confidence_does_not_abstain() -> None:
    decision = abstain_decision(0.9)
    assert decision.abstain is False
    assert decision.reason == "above_abstain_threshold"


def test_absent_score_is_not_a_zero_score() -> None:
    """'no measurement' and 'measured 0.0' must not collapse into one reason."""
    decision = abstain_decision(None)
    assert decision.abstain is True
    assert decision.reason == "no_confidence_score"
    assert decision.confidence is None


def test_insufficient_evidence_wins_over_high_confidence() -> None:
    """A confident claim with no evidence still cannot be judged."""
    sufficiency = evidence_sufficiency([])
    decision = abstain_decision(0.99, sufficiency=sufficiency)
    assert decision.abstain is True
    assert decision.reason.startswith("insufficient_evidence:")


def test_threshold_actually_changes_the_outcome() -> None:
    assert abstain_decision(0.45, threshold=0.3).abstain is False
    assert abstain_decision(0.45, threshold=0.6).abstain is True


def test_threshold_out_of_range_is_rejected() -> None:
    import pytest

    with pytest.raises(Exception):
        abstain_decision(0.5, threshold=1.5)


def test_decision_records_the_threshold_used() -> None:
    """A reader must be able to tell 'abstained at 0.31 < 0.5' from a bare flag."""
    decision = abstain_decision(0.31, threshold=0.4)
    assert decision.threshold == 0.4
    assert decision.confidence == 0.31
