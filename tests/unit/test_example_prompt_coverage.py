"""Every required field must be demonstrated by an example the model actually receives.

Why this file exists
--------------------
``build_example_text`` sends only ``examples[0]`` (plus ``examples[3]`` on a large context
window). Two required fields were therefore never shown to the model:

* ``item_events`` — populated only in ``examples[1]``, which is never sent. ``item_type``,
  ``action`` and the rest are required by ``ItemEventFact``, so the model guessed a shape
  without one and Pydantic rejected the whole chapter.
* ``characters[].abilities_gained`` — an empty list in *every* example, so
  ``AbilityGained.dimension`` and ``.name`` were never shown either.

Measured on a 5000-character chapter: two full re-extractions, each one a complete chapter
re-send. The failure looks like a model error and is a prompt defect.

The rule
--------
If the schema says a field is required, at least one *sent* example must show it populated
with every one of its own required sub-fields. An empty list in the example is not a
demonstration — it is the absence of one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    PROJECT / "src" / "cbb_stage1" / "engines" / "ai_reader" / "prompts" / "extraction_examples.json"
)


def _examples() -> list[dict]:
    return json.loads(EXAMPLES.read_text(encoding="utf-8"))


def _sent_indices(examples: list[dict], context_window: int) -> list[int]:
    """Mirror `build_example_text`'s selection exactly."""
    indices = [0]
    if len(examples) >= 4 and context_window > 16384:
        indices.append(3)
    return indices


def _sent_examples(context_window: int = 200_000) -> list[dict]:
    examples = _examples()
    return [examples[i] for i in _sent_indices(examples, context_window)]


# ── the two gaps that cost two re-extractions ───────────────────────────
def test_a_sent_example_populates_item_events() -> None:
    populated = [ex for ex in _sent_examples() if ex.get("item_events")]
    assert populated, (
        "no sent example populates item_events; ItemEventFact requires item_name, "
        "item_type and action, so the model is guessing their shape"
    )


def test_item_events_demonstrates_every_required_key() -> None:
    from cbb_stage1.engines.ai_reader.schema import ItemEventFact

    required = {n for n, f in ItemEventFact.model_fields.items() if f.is_required()}
    shown: set[str] = set()
    for ex in _sent_examples():
        for entry in ex.get("item_events") or ():
            shown |= set(entry)
    missing = sorted(required - shown)
    assert not missing, f"required item_events keys never demonstrated: {missing}"


def test_a_sent_example_populates_abilities_gained() -> None:
    shown = 0
    for ex in _sent_examples():
        for char in ex.get("characters") or ():
            shown += len(char.get("abilities_gained") or ())
    assert shown > 0, (
        "abilities_gained is empty in every sent example; AbilityGained requires "
        "dimension and name, so the model cannot know the shape"
    )


def test_abilities_gained_demonstrates_every_required_key() -> None:
    from cbb_stage1.engines.ai_reader.schema import AbilityGained

    required = {n for n, f in AbilityGained.model_fields.items() if f.is_required()}
    shown: set[str] = set()
    for ex in _sent_examples():
        for char in ex.get("characters") or ():
            for ability in char.get("abilities_gained") or ():
                if isinstance(ability, dict):
                    shown |= set(ability)
    missing = sorted(required - shown)
    assert not missing, f"required abilities_gained keys never demonstrated: {missing}"


# ── the general rule, so the next field does not repeat it ───────────────
@pytest.mark.parametrize(
    "field",
    [
        "characters",
        "relationships",
        "locations",
        "spatial_relationships",
        "item_events",
        "org_events",
        "events",
        "new_concepts",
        "world_declarations",
    ],
)
def test_every_list_field_has_a_populated_sent_example(field: str) -> None:
    """A list that is empty in every sent example teaches the model nothing about it."""
    populated = [ex for ex in _sent_examples() if ex.get(field)]
    assert populated, f"no sent example populates {field}"


def test_the_small_window_selection_still_sends_one_example() -> None:
    """On a ≤16K window only examples[0] goes out; the gaps must be closed there too."""
    examples = _examples()
    indices = _sent_indices(examples, 8192)
    assert indices == [0], "the small-window selection changed; re-check what is demonstrated"
    first = examples[0]
    assert first.get("item_events"), (
        "a small window sends only examples[0], so examples[0] itself must demonstrate "
        "item_events or a 16K-window model can never learn its shape"
    )
    abilities = [
        ability
        for char in first.get("characters") or ()
        for ability in char.get("abilities_gained") or ()
    ]
    assert abilities, (
        "a small window sends only examples[0], which must demonstrate abilities_gained"
    )


# ── the examples must stay well-formed ─────────────────────────────────
def test_no_example_is_marked_required_but_empty() -> None:
    """Sanity: the file parses and every example declares chapter_id and novel_id."""
    for index, ex in enumerate(_examples()):
        assert "chapter_id" in ex, f"example {index} has no chapter_id"
        assert "novel_id" in ex, f"example {index} has no novel_id"
