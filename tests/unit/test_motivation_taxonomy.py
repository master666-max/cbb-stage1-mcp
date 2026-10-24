"""The motivation taxonomy must carry the paper's two-level structure, not five flattened labels.

Measured gap (D15)
------------------
`derive/motivation.py:50` declares:

    MASLOW_LEVELS = ("生理", "安全", "归属", "尊重", "自我实现")

and the prompt's closed set is exactly those five. The paper's taxonomy is two-level: five Maslow needs
**and** sixteen Reiss desires as sub-levels, and it names them. The audit counted 5/21 implemented.

Why the flattening matters
--------------------------
Maslow's five are coarse enough that most behaviours collapse into one bucket: a character who wants
revenge, one who wants status and one who wants their family safe can all land on "安全" or "尊重". The
sub-level is what makes the label informative — "复仇" and "安宁" are different motives that the coarse
level cannot distinguish.

It also matters for verification: with 21 categories the model's agreement with a second labeller is a
real measurement (κ), and with 5 it is nearly free.

What is pinned
--------------
* the closed set is the paper's 21 (5 levels × their sub-desires), and each sub-level declares its
  parent, so a consumer can roll up;
* the prompt asks for the sub-level, and lists them (a closed set the model is not shown is not a closed
  set);
* `parse_chain` accepts a sub-level and drops an unknown one — same discipline as before, just a larger
  set;
* a row carries both the sub-level and its rolled-up level, so old readers that only understand the five
  keep working.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.derive.motivation import (
    MASLOW_LEVELS,
    REISS_DESIRES,
    needs_level_for,
    parse_chain,
    prompt_for_character,
)

#: The paper's sixteen Reiss desires.
EXPECTED_REISS = {
    "权力", "独立", "好奇", "接纳", "秩序", "荣誉", "理想", "社交",
    "家庭", "地位", "复仇", "浪漫", "食欲", "身体活动", "安宁", "储蓄",
}


# ══ the two-level taxonomy ═══════════════════════════════════════════
def test_the_five_levels_are_unchanged() -> None:
    """The coarse level is what older readers understand; do not move it."""
    assert MASLOW_LEVELS == ("生理", "安全", "归属", "尊重", "自我实现")


def test_the_sixteen_desires_are_present() -> None:
    assert set(REISS_DESIRES) == EXPECTED_REISS, (
        "the sub-level set is not the paper's sixteen: "
        f"missing {sorted(EXPECTED_REISS - set(REISS_DESIRES))}, "
        f"extra {sorted(set(REISS_DESIRES) - EXPECTED_REISS)}"
    )


def test_every_desire_rolls_up_to_one_of_the_five() -> None:
    for desire in REISS_DESIRES:
        level = needs_level_for(desire)
        assert level in MASLOW_LEVELS, f"{desire} rolls up to {level!r}, which is not a Maslow level"


def test_the_rollup_covers_every_desire() -> None:
    """A desire with no parent would be unreachable through the rollup."""
    for desire in REISS_DESIRES:
        assert needs_level_for(desire) is not None


# ══ the prompt shows the closed set ══════════════════════════════════
def test_the_prompt_lists_the_sub_levels() -> None:
    """A closed set the model is never shown is not a closed set."""
    prompt = prompt_for_character("言万", ["推开木门"])
    missing = [desire for desire in REISS_DESIRES if desire not in prompt]
    assert not missing, f"the prompt does not offer these sub-levels: {missing}"


# ══ parsing accepts a sub-level and still rejects an unknown one ═════
def test_a_sub_level_is_accepted() -> None:
    links = parse_chain(
        '[{"behavior_index": 0, "motive": "想夺回失物", "needs_level": "复仇"}]',
        ["推开木门"],
    )
    assert links
    assert links[0]["desire"] == "复仇"
    assert links[0]["needs_level"] == "安全", (
        f"the sub-level did not roll up: {links[0].get('needs_level')!r}"
    )


def test_an_unknown_label_is_still_dropped() -> None:
    links = parse_chain(
        '[{"behavior_index": 0, "motive": "说不清", "needs_level": "随便什么"}]',
        ["推开木门"],
    )
    assert links == [] or links[0].get("needs_level") is None, (
        "an unknown label was coerced instead of dropped"
    )


def test_a_coarse_level_is_still_accepted() -> None:
    """A model that answers with a Maslow level has not left the taxonomy."""
    links = parse_chain(
        '[{"behavior_index": 0, "motive": "躲开危险", "needs_level": "安全"}]',
        ["推开木门"],
    )
    assert links
    assert links[0]["needs_level"] == "安全"


def test_the_row_carries_both_levels() -> None:
    links = parse_chain(
        '[{"behavior_index": 0, "motive": "想夺回失物", "needs_level": "复仇"}]',
        ["推开木门"],
    )
    row: dict[str, Any] = links[0]
    assert row["desire"] == "复仇"
    assert row["needs_level"] in MASLOW_LEVELS
