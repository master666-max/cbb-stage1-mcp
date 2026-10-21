"""Assign an entity tier from evidence, not from a constant.

``tier`` decides how hard an assertion is verified (§5.15): canonical gets CoVe +
atomic verification + a different-tier model, minor gets schema checks only. It was
hardcoded ``"major"`` for every character and location, which made the whole
verification matrix collapse into one tier — the tables existed, the levels were
derivable, and every assertion still landed on ``evidence_span``.

What the spec says about the *meaning* (``01-箱规格`` §5.4): ``canonical`` gets its own
wiki page, ``major`` is merged into a canonical page, ``minor`` is shown inline. It
does not give a numeric rule, so the rule has to be defined here — and it has to be
recomputable from the box, or two runs of the same book would disagree.

The signals available at normalise time, in order of reliability:

  * ``mentions``   — how often the surface appears in the text;
  * ``aliases``    — an entity with aliases is referred to in several ways, which
                     only happens for entities the narration keeps returning to;
  * ``events``     — participation in an event means the entity *does* something,
                     which is what makes a page worth having;
  * ``relations``  — appearing in a relationship means it is part of the social
                     graph rather than scenery.

A single-unit run cannot see cross-chapter recurrence, so the thresholds are
deliberately low and the rule is stated rather than tuned: **a tier is a promise
about how much verification effort this entity will receive**, and over-promising
wastes model calls on scenery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "TIER_THRESHOLDS",
    "TierEvidence",
    "assign_tier",
    "assign_tiers",
]

#: Thresholds for promoting an entity out of ``minor``.
#:
#: ``canonical`` requires BOTH recurrence and agency: a name mentioned often but
#: never doing anything is scenery with a lot of airtime. ``major`` needs only one
#: of the two, because a one-scene character who drives an event still deserves a
#: page.
TIER_THRESHOLDS: dict[str, int] = {
    "canonical_mentions": 3,
    "canonical_events": 2,
    "major_mentions": 2,
    "major_events": 1,
    "major_aliases": 1,
}


@dataclass
class TierEvidence:
    """Everything the tier decision is allowed to look at."""

    name: str
    entity_type: str = "unknown"
    mentions: int = 0
    aliases: int = 0
    events: int = 0
    relations: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        """A single comparable number, for logging and for tie-breaking."""
        return (
            self.mentions * 2
            + self.aliases
            + self.events * 3
            + self.relations * 2
        )


def _count_mentions(names: Iterable[str], text: str) -> int:
    total = 0
    for name in names:
        if name and name in text:
            total += text.count(name)
    return total


def assign_tier(evidence: TierEvidence) -> str:
    """Map evidence to ``canonical`` / ``major`` / ``minor``, recording why.

    Deterministic and monotone: more evidence never lowers a tier.
    """
    reasons: list[str] = []

    canonical = (
        evidence.mentions >= TIER_THRESHOLDS["canonical_mentions"]
        and evidence.events >= TIER_THRESHOLDS["canonical_events"]
    )
    if canonical:
        reasons.append(
            f"mentions={evidence.mentions}>={TIER_THRESHOLDS['canonical_mentions']} "
            f"and events={evidence.events}>={TIER_THRESHOLDS['canonical_events']}"
        )
        evidence.reasons = reasons
        return "canonical"

    major = (
        evidence.mentions >= TIER_THRESHOLDS["major_mentions"]
        or evidence.events >= TIER_THRESHOLDS["major_events"]
        or evidence.aliases >= TIER_THRESHOLDS["major_aliases"]
        or evidence.relations >= TIER_THRESHOLDS["major_events"]
    )
    if major:
        if evidence.mentions >= TIER_THRESHOLDS["major_mentions"]:
            reasons.append(f"mentions={evidence.mentions}")
        if evidence.events >= TIER_THRESHOLDS["major_events"]:
            reasons.append(f"events={evidence.events}")
        if evidence.aliases >= TIER_THRESHOLDS["major_aliases"]:
            reasons.append(f"aliases={evidence.aliases}")
        if evidence.relations >= TIER_THRESHOLDS["major_events"]:
            reasons.append(f"relations={evidence.relations}")
        evidence.reasons = reasons
        return "major"

    reasons.append(
        f"only mentions={evidence.mentions}, events={evidence.events}, "
        f"aliases={evidence.aliases}"
    )
    evidence.reasons = reasons
    return "minor"


def assign_tiers(
    entities: Sequence[Mapping[str, Any]],
    *,
    unit_text: str,
    event_participants: Iterable[str] = (),
    relation_names: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Return ``{canonical_name: {tier, evidence}}`` for a chapter's entities.

    ``unit_text`` is the chapter body: mention counts are measured against the text
    the entity was extracted from, not against the book, because a normalise pass
    only ever sees one unit.
    """
    # Count occurrences, not distinct events: an entity that acts in three events
    # has more claim to a page than one that acts in one, and collapsing to a set
    # hid exactly that signal (the protagonist with two events scored 1).
    from collections import Counter as _Counter

    participant_counts = _Counter(str(n) for n in event_participants if n)
    related_counts = _Counter(str(n) for n in relation_names if n)

    out: dict[str, dict[str, Any]] = {}
    for entity in entities:
        name = str(entity.get("name") or entity.get("canonical_name") or "")
        if not name:
            continue
        aliases = [str(item) for item in (entity.get("aliases") or []) if str(item)]
        evidence = TierEvidence(
            name=name,
            entity_type=str(entity.get("type") or "unknown"),
            mentions=_count_mentions([name, *aliases], unit_text),
            aliases=len(aliases),
            events=participant_counts.get(name, 0),
            relations=related_counts.get(name, 0),
        )
        tier = assign_tier(evidence)
        out[name] = {
            "tier": tier,
            "evidence": {
                "mentions": evidence.mentions,
                "aliases": evidence.aliases,
                "events": evidence.events,
                "relations": evidence.relations,
                "score": evidence.score,
                "reasons": list(evidence.reasons),
            },
        }
    return out
