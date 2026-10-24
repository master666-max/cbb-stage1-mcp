"""D5: entity delta merging — borrow the flow, keep our key.

Measured gap and the correction
-------------------------------
The audit's D5 asked for "新实体/别名增量 + 该实体出场集重扫". GraphRAG ships the first half in
`index/update/entities.py:26-60`:

1. match old and delta entities to get `{delta_id: old_id}`;
2. offset `human_readable_id` from `old_max + 1`;
3. concat;
4. **`groupby` the matched key**, taking the first id/type and **collecting descriptions into a list**;
5. re-summarise afterwards.

Step 1's key is `title` — exactly the fragile choice A20 records this project deliberately avoided, because a
chapter retitled with identical text would look like a new document. So the flow is borrowed and the key
stays `sha256` plus `_detect_renames`.

Step 4 is the part worth copying that we do **not** do today: our `describe` keeps "the longer description"
and throws the other away. A list preserves both until a summariser sees them, which is what makes the
re-summarise step possible at all.

What is pinned
--------------
* identical content under a new id is matched (the reason for the key);
* a changed title with identical text still matches — the case `title` keying would lose;
* descriptions accumulate into a list instead of one being dropped;
* aliases union, and the merged entity's own name becomes an alias;
* appending is idempotent: the same delta twice does not duplicate a row.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.incremental.entities_append import (
    merge_entities,
    match_by_content,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _entity(
    entity_id: str,
    name: str,
    sha: str,
    *,
    desc: str = "",
    aliases: list[str] | None = None,
    human_readable_id: int = 0,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "canonical_name": name,
        "sha256": sha,
        "description": desc,
        "aliases": list(aliases or []),
        "human_readable_id": human_readable_id,
    }


# ══ matching by content ═════════════════════════════════════════════
def test_identical_content_under_a_new_id_matches() -> None:
    old = [_entity("e:1", "言万", SHA_A)]
    delta = [_entity("e:2", "言万", SHA_A)]
    assert match_by_content(old, delta) == {"e:2": "e:1"}


def test_a_retitled_entity_with_identical_text_still_matches() -> None:
    """This is why the key is content and not title."""
    old = [_entity("e:1", "青牛镇", SHA_A)]
    delta = [_entity("e:2", " 青牛镇 ", SHA_A)]
    assert match_by_content(old, delta) == {"e:2": "e:1"}


def test_different_content_does_not_match() -> None:
    old = [_entity("e:1", "言万", SHA_A)]
    delta = [_entity("e:2", "言万", SHA_B)]
    assert match_by_content(old, delta) == {}


def test_an_entity_without_a_sha_is_not_matched_by_guess() -> None:
    old = [_entity("e:1", "言万", "")]
    delta = [_entity("e:2", "言万", "")]
    assert match_by_content(old, delta) == {}, (
        "two entities with no content hash were paired on their name alone"
    )


# ══ merging ═════════════════════════════════════════════════════════
def test_a_matched_delta_folds_into_the_old_row() -> None:
    old = [_entity("e:1", "言万", SHA_A, desc="第一个描述")]
    delta = [_entity("e:2", "言万", SHA_A, desc="第二个描述")]
    merged, report = merge_entities(old, delta)
    assert len(merged) == 1, f"the matched delta was not folded in: {[r['entity_id'] for r in merged]}"
    assert report["merged"] == 1


def test_descriptions_accumulate_into_a_list() -> None:
    """Our `describe` keeps the longer one and drops the other; a list keeps both."""
    old = [_entity("e:1", "言万", SHA_A, desc="短的")]
    delta = [_entity("e:2", "言万", SHA_A, desc="更长的一段描述")]
    merged, _report = merge_entities(old, delta)
    descriptions = merged[0]["descriptions"]
    assert isinstance(descriptions, list)
    assert "短的" in descriptions and "更长的一段描述" in descriptions, (
        f"a description was dropped: {descriptions}"
    )


def test_aliases_union_and_include_the_absorbed_name() -> None:
    old = [_entity("e:1", "言万", SHA_A, aliases=["心叶"])]
    delta = [_entity("e:2", "韩立", SHA_A, aliases=["二愣子"])]
    merged, _report = merge_entities(old, delta)
    aliases = merged[0]["aliases"]
    assert "心叶" in aliases and "二愣子" in aliases
    assert "韩立" in aliases, "the absorbed entity's own name did not become an alias"


def test_an_unmatched_delta_is_appended() -> None:
    old = [_entity("e:1", "言万", SHA_A)]
    delta = [_entity("e:9", "拉法", SHA_B)]
    merged, report = merge_entities(old, delta)
    assert len(merged) == 2
    assert report["appended"] == 1


def test_human_readable_ids_are_offset_for_new_rows() -> None:
    old = [_entity("e:1", "言万", SHA_A, human_readable_id=0)]
    delta = [_entity("e:9", "拉法", SHA_B, human_readable_id=0)]
    merged, _report = merge_entities(old, delta)
    ids = sorted(row["human_readable_id"] for row in merged)
    assert ids == [0, 1], f"new rows did not get fresh human-readable ids: {ids}"


# ══ idempotence ═════════════════════════════════════════════════════
def test_applying_the_same_delta_twice_does_not_duplicate() -> None:
    old = [_entity("e:1", "言万", SHA_A)]
    delta = [_entity("e:9", "拉法", SHA_B)]
    once, _ = merge_entities(old, delta)
    twice, _ = merge_entities(once, delta)
    assert len(twice) == len(once), f"the second append duplicated rows: {len(twice)} vs {len(once)}"


def test_an_empty_delta_returns_the_old_rows() -> None:
    old = [_entity("e:1", "言万", SHA_A)]
    merged, report = merge_entities(old, [])
    assert [row["entity_id"] for row in merged] == ["e:1"]
    assert report["merged"] == 0 and report["appended"] == 0


def test_the_old_rows_are_not_mutated() -> None:
    old = [_entity("e:1", "言万", SHA_A, desc="原描述")]
    merge_entities(old, [_entity("e:2", "言万", SHA_A, desc="新描述")])
    assert old[0]["description"] == "原描述", "the caller's list was modified in place"
