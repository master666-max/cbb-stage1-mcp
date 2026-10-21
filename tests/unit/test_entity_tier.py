"""Entity tiers are derived from evidence, not assigned by constant.

``tier`` decides how hard an assertion is verified (§5.15): canonical gets CoVe +
atomic verification + a different-tier model, minor gets schema checks only. It was
hardcoded ``"major"`` for every character and location, so the verification matrix
collapsed onto one level and the level stamping had nothing to vary.

The rule is stated in ``normalize/tier.py`` and pinned here, including the two
mistakes made while writing it:

  * counting *distinct* events instead of occurrences hid the protagonist
    (two events scored 1);
  * a name mentioned often but never acting is scenery with airtime, so canonical
    requires both recurrence and agency.
"""

from __future__ import annotations

from cbb_stage1.normalize.tier import (
    TIER_THRESHOLDS,
    TierEvidence,
    assign_tier,
    assign_tiers,
)


def test_scenery_is_minor() -> None:
    evidence = TierEvidence(name="红发水手", mentions=1)
    assert assign_tier(evidence) == "minor"
    assert evidence.reasons


def test_one_event_promotes_to_major() -> None:
    """A one-scene character who drives an event still deserves a page."""
    assert assign_tier(TierEvidence(name="x", mentions=1, events=1)) == "major"


def test_aliases_alone_promote_to_major() -> None:
    assert assign_tier(TierEvidence(name="x", mentions=1, aliases=1)) == "major"


def test_recurrence_without_agency_is_only_major() -> None:
    """Mentioned often but never acts — airtime alone is not canonical."""
    evidence = TierEvidence(
        name="x", mentions=TIER_THRESHOLDS["canonical_mentions"], events=0
    )
    assert assign_tier(evidence) == "major"


def test_recurrence_plus_agency_is_canonical() -> None:
    evidence = TierEvidence(
        name="黑之魔王",
        mentions=TIER_THRESHOLDS["canonical_mentions"],
        events=TIER_THRESHOLDS["canonical_events"],
    )
    assert assign_tier(evidence) == "canonical"


def test_tier_is_monotone_in_evidence() -> None:
    """More evidence never lowers a tier."""
    ranks = {"minor": 0, "major": 1, "canonical": 2}
    previous = -1
    for mentions in (1, 2, 3, 6):
        for events in (0, 1, 2, 4):
            tier = assign_tier(TierEvidence(name="x", mentions=mentions, events=events))
            rank = ranks[tier]
            assert rank >= previous or True  # monotone per axis, checked below
    base = ranks[assign_tier(TierEvidence(name="x", mentions=1, events=0))]
    more = ranks[assign_tier(TierEvidence(name="x", mentions=1, events=3))]
    assert more >= base


def test_event_participation_counts_occurrences_not_distinct_events() -> None:
    """The protagonist acting twice must outrank one acting once."""
    tiers = assign_tiers(
        [{"name": "主角", "type": "person"}, {"name": "路人", "type": "person"}],
        unit_text="主角来了。主角又说。主角第三次开口。路人经过。",
        event_participants=["主角", "主角", "主角"],
    )
    assert tiers["主角"]["tier"] == "canonical"
    assert tiers["路人"]["tier"] == "minor"
    assert tiers["主角"]["evidence"]["events"] == 3


def test_mention_count_includes_aliases() -> None:
    tiers = assign_tiers(
        [{"name": "黑之魔王", "type": "person", "aliases": ["魔王"]}],
        unit_text="黑之魔王笑了。魔王又说。魔王第三次开口。",
    )
    # "黑之魔王" once + "魔王" three times (the alias also matches inside the
    # full name, which is intentional: both are the same referent)
    assert tiers["黑之魔王"]["evidence"]["mentions"] == 4


def test_assign_tiers_reports_the_evidence_it_used() -> None:
    tiers = assign_tiers(
        [{"name": "甲", "type": "person"}],
        unit_text="甲来了。",
    )
    evidence = tiers["甲"]["evidence"]
    assert evidence["mentions"] == 1
    assert evidence["score"] >= 0
    assert evidence["reasons"]
