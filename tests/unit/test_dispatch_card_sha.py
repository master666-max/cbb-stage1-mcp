"""A dispatch card must be identifiable after the fact.

Measured gap
------------
`orchestrator/pipeline.py:441-473` builds the card that is handed to the engine for one unit, then
passes it straight in. Nothing records it: no `card_sha256`, no file on disk, nothing in the task
record (`produced_by` carries only `prompt_sha256`). So the design's promise — "the card proves
which inputs the candidate was extracted from" — cannot be checked; a reader cannot even tell two
different cards apart.

These tests pin the two things that make a card auditable:

1. the card carries a `card_sha256` that is recomputable from the card's own content, and does not
   include itself in that hash;
2. two cards that differ in any meaningful field hash differently, and the same card hashes the
   same on every run (the builder is documented as deterministic and timestamp-free).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cbb_stage1.ids import sha256_text
from cbb_stage1.orchestrator.pipeline import _card_for, _card_sha256


def _unit(**overrides) -> SimpleNamespace:
    unit = SimpleNamespace(
        unit_id="u:tst:v01:c001",
        volume=1,
        chapter=1,
        title="出发，去集市",
        order=1,
        narrative_mode="first_person",
        narrator="言万",
        narrator_reliability="reliable",
        text_path="plan/texts/v01/u001.txt",
        sha256="sha256:" + "a" * 64,
        chars=1234,
    )
    for key, value in overrides.items():
        setattr(unit, key, value)
    return unit


def _cfg(**overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(book_id="tst", genre="fantasy")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ══ the hash exists and is recomputable ═══════════════════════════════
def test_the_card_carries_a_sha() -> None:
    card = _card_for(_unit(), _cfg())
    assert "card_sha256" in card, "the dispatch card has no hash; it cannot be identified"
    assert str(card["card_sha256"]).startswith("sha256:")


def test_the_sha_is_recomputable_from_the_card() -> None:
    card = _card_for(_unit(), _cfg())
    recomputed = _card_sha256(card)
    assert card["card_sha256"] == recomputed, (
        "the recorded hash does not match the card's own content; _card_sha256 must ignore the "
        "hash field itself, or the value can never be verified"
    )


def test_the_same_unit_and_config_hash_the_same() -> None:
    first = _card_for(_unit(), _cfg())
    second = _card_for(_unit(), _cfg())
    assert first["card_sha256"] == second["card_sha256"], (
        "the card is documented as deterministic; two builds disagree"
    )


# ══ any meaningful change changes the hash ════════════════════════════
def test_a_changed_text_identity_changes_the_hash() -> None:
    base = _card_for(_unit(), _cfg())
    changed = _card_for(_unit(sha256="sha256:" + "b" * 64), _cfg())
    assert base["card_sha256"] != changed["card_sha256"]


def test_a_changed_narrator_changes_the_hash() -> None:
    base = _card_for(_unit(), _cfg())
    changed = _card_for(_unit(narrator="露娜"), _cfg())
    assert base["card_sha256"] != changed["card_sha256"]


def test_a_changed_genre_changes_the_hash() -> None:
    base = _card_for(_unit(), _cfg())
    changed = _card_for(_unit(), _cfg(genre="wuxia"))
    assert base["card_sha256"] != changed["card_sha256"]


# ══ the hash covers the content, not the field order ══════════════════
def test_the_hash_ignores_key_order() -> None:
    card = _card_for(_unit(), _cfg())
    recorded = card["card_sha256"]
    reordered = dict(reversed(list(card.items())))
    assert _card_sha256(reordered) == recorded, (
        "the hash depends on key order, so it is not a content hash"
    )


def test_the_card_serialises_to_json() -> None:
    """It has to survive being written to disk, which is the point of having a hash."""
    card = _card_for(_unit(), _cfg())
    text = json.dumps(card, ensure_ascii=False, sort_keys=True)
    assert json.loads(text)["card_sha256"] == card["card_sha256"]
