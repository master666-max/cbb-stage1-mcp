"""Downgrading narration to `statement` must not lose it.

Measured failure (D2)
---------------------
`assemble/assertions.py:148` downgrades narration to `claim_layer=statement` when the narrator is
unreliable, and `:430-449` quarantines any `statement` without a speaker. Nothing in the codebase ever
builds a `Speaker(kind="narrator")` — the only two construction sites are `kind="entity"` — so every
downgraded narration assertion has no speaker and is quarantined.

Net effect of `--narrator-reliability unreliable`: the run loses its narration instead of lowering its
claim layer. That is worse than not having the mechanism: the flag silently deletes evidence.

What the fix must and must not do
---------------------------------
An unreliable narrator is still a *speaker*: "the narrator claims X" is exactly the claim a reader
needs when the narrator may be lying. So the downgrade attaches a narrator speaker.

It must not become a loophole: a `fact`-layer assertion still cannot acquire a narrator speaker to
dodge the `_statement_derivation_violation` check, and a `statement` that names an entity speaker keeps
that speaker.
"""

from __future__ import annotations

from cbb_stage1.assemble.assertions import Assembler
from cbb_stage1.assemble.registry import seed_registry
from cbb_stage1.models import Assertion

REGISTRY = seed_registry()
UNIT = "u:t:v01:c001"


def _assertion(**overrides) -> dict:
    base = {
        "assertion_id": "a:1111111111111111",
        "book_id": "t",
        "predicate": "敌对",
        "library": "关系",
        # Entity roles take resolved ids, not surface names (role_mismatch otherwise).
        "roles": {"施事": "e:t:person:aaaaaaaaaaaa", "受事": "e:t:person:bbbbbbbbbbbb"},
        "qualifiers": {},
        "claim_layer": "fact",
        "speaker": None,
        "evidence": [
            {
                "unit_id": UNIT,
                "start": 0,
                "end": 12,
                "quote": "言万独自站在集市中央",
                "occurrence": 1,
                "mode": "narration",
            }
        ],
        "confidence": 0.9,
        "confidence_method": "evidence_span",
    }
    base.update(overrides)
    return base


def _assemble(assertions: list[dict], *, reliability: str):
    assembler = Assembler(REGISTRY, book_id="t", narrator_reliability=reliability)
    return assembler.assemble([Assertion.model_validate(item) for item in assertions])


# ══ an unreliable narrator keeps the narration ═══════════════════════
def test_unreliable_narration_is_accepted_not_quarantined() -> None:
    result = _assemble([_assertion()], reliability="unreliable")

    assert not result.quarantined, (
        "the downgraded narration was quarantined: "
        f"{[(q.reason, q.detail) for q in result.quarantined]}"
    )
    assert len(result.accepted) == 1, "the narration assertion was lost"


def test_the_accepted_assertion_is_statement_layer() -> None:
    result = _assemble([_assertion()], reliability="unreliable")
    assert result.accepted
    assert result.accepted[0].claim_layer == "statement"


def test_the_downgraded_assertion_names_the_narrator_as_speaker() -> None:
    result = _assemble([_assertion()], reliability="unreliable")
    speaker = result.accepted[0].speaker
    assert speaker is not None, "a statement without a speaker is exactly what gets quarantined"
    assert speaker.kind == "narrator", (
        f"the downgrade attached a {speaker.kind!r} speaker; an unreliable narrator is still a "
        "speaker, and naming them is the whole point of the downgrade"
    )


def test_the_downgrade_is_still_recorded() -> None:
    result = _assemble([_assertion()], reliability="unreliable")
    assert result.downgrades, "the downgrade event disappeared"


# ══ a reliable narrator is unchanged ═════════════════════════════════
def test_reliable_narration_stays_a_fact() -> None:
    result = _assemble([_assertion()], reliability="reliable")
    assert result.accepted
    assert result.accepted[0].claim_layer == "fact"
    assert not result.downgrades


def test_reliable_narration_gets_no_narrator_speaker() -> None:
    """A fact does not need a speaker; attaching one would imply it is hearsay."""
    result = _assemble([_assertion()], reliability="reliable")
    assert result.accepted[0].speaker is None


# ══ the loophole must stay closed ════════════════════════════════════
def test_an_entity_speaker_is_not_replaced() -> None:
    result = _assemble(
        [_assertion(speaker={"kind": "entity", "name": "言万"})], reliability="unreliable"
    )
    assert result.accepted
    assert result.accepted[0].speaker.kind == "entity", (
        "the narrator overwrote a real speaker"
    )


def test_dialogue_evidence_still_requires_its_speaker() -> None:
    """The quarantine must still fire for a `statement` with no speaker available at all."""
    assertion = _assertion(
        evidence=[
            {
                "unit_id": UNIT,
                "start": 0,
                "end": 8,
                "quote": "把它毁掉吧",
                "occurrence": 1,
                "mode": "dialogue",
            }
        ]
    )
    result = _assemble([assertion], reliability="unreliable")
    assert result.quarantined, (
        "dialogue with no attribution was accepted; the narrator must not be used as the speaker of "
        "someone else's line"
    )
