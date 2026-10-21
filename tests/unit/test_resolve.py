"""Unit tests for :mod:`cbb_stage1.resolve` (aliases, fanout, merge ledger).

Coverage follows the P2 acceptance points: the AI Reader three-level safety
table is reused verbatim, generic / kinship / title surfaces can never bridge
two entity groups (alias false-merge negative), the Lorebase fanout clean drops
surfaces hosted by too many entities, and merges are reversible, cycle-safe and
idempotent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cbb_stage1.engines.ai_reader.alias_resolver import UnionFind
from cbb_stage1.engines.ai_reader.schema import ChapterFact, CharacterFact, EntityDictEntry
from cbb_stage1.models import Entity, Evidence
from cbb_stage1.resolve import (
    BLACKLIST,
    SAFETY_BLOCKED,
    SAFETY_SAFE,
    SAFETY_SUSPICIOUS,
    MergeError,
    MergeLedger,
    MergeRecord,
    alias_is_safe,
    alias_safety_level,
    build_alias_map,
    build_alias_map_from_entities,
    canonical_surface,
    fanout_clean,
    filter_aliases,
    make_merge,
    merge_id_for,
    surface_hosts,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dict_entry(
    name: str,
    aliases: tuple[str, ...] = (),
    *,
    frequency: int = 1,
    entity_type: str = "person",
) -> EntityDictEntry:
    return EntityDictEntry(
        name=name,
        entity_type=entity_type,
        frequency=frequency,
        aliases=list(aliases),
        source="freq",
    )


def _chapter(characters: list[CharacterFact], chapter_id: int = 1) -> ChapterFact:
    return ChapterFact(chapter_id=chapter_id, novel_id="zj", characters=characters)


def _entity(name: str, aliases: tuple[str, ...] = (), *, etype: str = "person") -> Entity:
    return Entity(book_id="zj", type=etype, canonical_name=name, aliases=list(aliases))


# ---------------------------------------------------------------------------
# three-level safety table (reused from the engine port)
# ---------------------------------------------------------------------------


def test_safety_levels_match_ai_reader_table():
    assert alias_safety_level("大哥") == SAFETY_BLOCKED
    assert alias_safety_level("老人") == SAFETY_BLOCKED
    assert alias_safety_level("大人") == SAFETY_BLOCKED
    assert alias_safety_level("我的父亲") == SAFETY_BLOCKED
    assert alias_safety_level("剑") == SAFETY_SUSPICIOUS
    assert alias_safety_level("贝塔心叶") == SAFETY_SAFE
    assert alias_safety_level("心叶") == SAFETY_SAFE


def test_alias_is_safe_only_accepts_level_two():
    assert alias_is_safe("拉法") is True
    assert alias_is_safe("大哥") is False
    assert alias_is_safe("剑") is False


def test_filter_aliases_drops_unsafe_and_owner_and_duplicates():
    kept = filter_aliases(["心叶", "大哥", "心叶", "贝塔心叶", "剑", "  "], owner="贝塔心叶")
    assert kept == ["心叶"]


# ---------------------------------------------------------------------------
# Union-Find reuse (engine class, not re-implemented)
# ---------------------------------------------------------------------------


def test_union_find_returns_engine_groups():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    groups = uf.groups()
    assert groups[uf.find("a")] == ["a", "b", "c"]
    assert uf.group_size("c") == 3


# ---------------------------------------------------------------------------
# build_alias_map: explicit aliases merge, canonical name chosen by engine
# ---------------------------------------------------------------------------


def test_build_alias_map_merges_explicit_aliases():
    alias_map = build_alias_map(
        [
            _dict_entry("贝塔心叶", ("心叶",), frequency=10),
            _dict_entry("拉法", ("拉法尔",), frequency=8),
        ],
        [_chapter([CharacterFact(name="贝塔心叶", new_aliases=["叶子"])])],
    )
    assert alias_map == {
        "心叶": "贝塔心叶",
        "叶子": "贝塔心叶",
        "拉法尔": "拉法",
    }
    assert "贝塔心叶" not in alias_map  # canonical never maps to itself


def test_build_alias_map_from_entities_uses_tier_prior():
    alias_map = build_alias_map_from_entities(
        [
            _entity("贝塔心叶", ("心叶",)),
            _entity("另一个角色", ("心叶",)),
        ]
    )
    assert alias_map.get("心叶") in {"贝塔心叶", "另一个角色"}


def test_canonical_surface_is_identity_for_unknown():
    assert canonical_surface({"心叶": "贝塔心叶"}, "心叶") == "贝塔心叶"
    assert canonical_surface({}, "未知") == "未知"


# ---------------------------------------------------------------------------
# negative: forbidden aliases must never become Union-Find keys
# ---------------------------------------------------------------------------


def test_generic_kinship_alias_is_not_a_union_find_key():
    alias_map = build_alias_map([_dict_entry("贝塔心叶", ("大哥", "心叶"), frequency=5)])
    assert "大哥" not in alias_map
    assert alias_map["心叶"] == "贝塔心叶"


def test_unsafe_character_name_is_skipped_entirely():
    # "老人" is a generic surface: its alias assignment must not create a key.
    alias_map = build_alias_map(
        [], [_chapter([CharacterFact(name="老人", new_aliases=["贝塔心叶"])])]
    )
    assert alias_map == {}


def test_two_established_groups_are_never_bridged():
    # Group A and group B both have 3 members; a late alias that would bridge
    # them is refused by the engine's conflict-protected union (negative:
    # alias false-merge).
    alias_map = build_alias_map([
        _dict_entry("林惊羽", ("惊羽", "羽哥"), frequency=10),
        _dict_entry("伽马莉莎", ("伽马", "莉莎", "惊羽"), frequency=8),
    ])
    assert alias_map["惊羽"] == "林惊羽"
    assert alias_map["伽马"] == "伽马莉莎"
    assert alias_map["惊羽"] != "伽马莉莎"
    assert "林惊羽" not in alias_map and "伽马莉莎" not in alias_map


# ---------------------------------------------------------------------------
# fanout clean (Lorebase 02b_merge_dict.py)
# ---------------------------------------------------------------------------


def test_surface_hosts_counts_canonical_and_alias_hosts():
    hosts = surface_hosts(
        [_entity("甲", ("苍蓝",)), _entity("乙", ("苍蓝",)), _entity("丙", ("苍蓝",))]
    )
    assert hosts["苍蓝"] == ("丙", "乙", "甲")


def test_fanout_drops_alias_hosted_by_more_than_threshold():
    entities = [_entity(f"角色{i}", ("苍蓝",)) for i in range(4)]
    result = fanout_clean(entities)
    assert result.generic_names == frozenset({"苍蓝"})
    assert result.kept_count == 4
    assert all(entity.aliases == [] for entity in result.kept)
    assert result.dropped_alias_count == 4
    finding = result.findings[0]
    assert finding.reason == "fanout"
    assert finding.host_count == 4


def test_fanout_keeps_surface_at_threshold_boundary():
    entities = [_entity(f"角色{i}", ("苍蓝",)) for i in range(3)]
    result = fanout_clean(entities)  # 3 hosts is NOT > 3
    assert result.generic_names == frozenset()
    assert [entity.aliases for entity in result.kept] == [["苍蓝"]] * 3


def test_fanout_blacklist_drops_even_single_host():
    result = fanout_clean([_entity("甲角色", ("神",))])
    assert "神" in result.generic_names
    assert result.findings[0].reason == "blacklist"
    assert "神" in BLACKLIST
    assert result.kept[0].aliases == []


def test_fanout_safety_table_drops_unsafe_aliases_and_names():
    result = fanout_clean([
        _entity("甲角色", ("大哥", "剑")),
        _entity("剑形", ("某剑",)),
    ])
    assert {"大哥", "剑"}.issubset(result.generic_names)
    assert result.dropped_entities == ()
    assert [entity.canonical_name for entity in result.kept] == ["甲角色", "剑形"]
    assert result.kept[0].aliases == []  # "大哥" and "剑" both failed the table
    assert result.kept[1].aliases == ["某剑"]  # a safe alias survives


def test_fanout_accepts_entity_dict_entries_and_does_not_mutate_inputs():
    entry = _dict_entry("甲角色", ("苍蓝",))
    result = fanout_clean([entry])
    assert result.kept[0] is not entry
    assert entry.aliases == ["苍蓝"]
    assert isinstance(result.kept[0], EntityDictEntry)


def test_fanout_custom_threshold_and_safety_switch():
    entities = [_entity("甲角色", ("苍蓝",)), _entity("乙角色", ("苍蓝",))]
    assert fanout_clean(entities, fanout_threshold=1).generic_names == frozenset({"苍蓝"})
    # "伯伯" is kinship (level 0) but not in the upstream blacklist, so the
    # safety switch is observable.
    assert fanout_clean([_entity("甲角色", ("伯伯",))]).generic_names == frozenset({"伯伯"})
    relaxed = fanout_clean([_entity("甲角色", ("伯伯",))], apply_safety_table=False)
    assert relaxed.generic_names == frozenset()


def test_fanout_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        fanout_clean([_entity("甲")], fanout_threshold=0)


# ---------------------------------------------------------------------------
# merge records: reversible, content-addressed, schema shape
# ---------------------------------------------------------------------------


def _evidence() -> Evidence:
    return Evidence(unit_id="u:zj:v01:c001", start=0, end=2, quote="拉法", occurrence=1)


def test_merge_id_is_content_addressed_and_short_hex():
    first = merge_id_for("zj", "alias", ["e:zj:p:b", "e:zj:p:a"], "e:zj:p:c")
    second = merge_id_for("zj", "alias", ["e:zj:p:a", "e:zj:p:b"], "e:zj:p:c")
    assert first == second
    assert first.startswith("mg:")
    assert len(first) == len("mg:") + 16


def test_make_merge_has_box_shape_and_undo_payload():
    record = make_merge(
        "zj", ["e:zj:p:a"], "e:zj:p:b", kind="coref", decided_by="llm",
        confidence=0.9, evidence=[_evidence()], at="2026-10-18T20:30:00+08:00",
    )
    payload = record.box_dict()
    assert payload["from"] == ["e:zj:p:a"]
    assert payload["to"] == "e:zj:p:b"
    assert payload["reversible"] is True
    assert payload["undo"] == {"from": ["e:zj:p:a"], "to": "e:zj:p:b"}
    assert payload["evidence"][0]["quote"] == "拉法"
    assert record.merge_id == merge_id_for("zj", "coref", ["e:zj:p:a"], "e:zj:p:b")


def test_make_merge_rejects_self_merge():
    with pytest.raises(MergeError):
        make_merge("zj", ["e:zj:p:a"], "e:zj:p:a")


def test_merge_record_rejects_non_entity_ids():
    with pytest.raises(ValidationError):
        make_merge("zj", ["person-a"], "e:zj:p:b")
    with pytest.raises(ValidationError):
        make_merge("zj", ["e:zj:p:a"], "person-b")


# ---------------------------------------------------------------------------
# merge ledger: apply / resolve / undo
# ---------------------------------------------------------------------------


def test_ledger_resolves_merge_chains():
    ledger = MergeLedger(book_id="zj")
    ledger.add(["e:zj:p:a"], "e:zj:p:b")
    ledger.add(["e:zj:p:b"], "e:zj:p:c")
    assert ledger.resolve("e:zj:p:a") == "e:zj:p:c"
    assert ledger.resolve("e:zj:p:c") == "e:zj:p:c"
    assert ledger.resolve("e:zj:p:unknown") == "e:zj:p:unknown"
    assert ledger.merged_into_map() == {
        "e:zj:p:a": "e:zj:p:c",
        "e:zj:p:b": "e:zj:p:c",
    }
    assert ledger.merge_groups() == {
        "e:zj:p:c": ["e:zj:p:a", "e:zj:p:b", "e:zj:p:c"]
    }


def test_ledger_undo_restores_identity():
    ledger = MergeLedger(book_id="zj")
    record = ledger.add(["e:zj:p:a"], "e:zj:p:b")
    assert ledger.resolve("e:zj:p:a") == "e:zj:p:b"
    ledger.undo(record.merge_id)
    assert ledger.resolve("e:zj:p:a") == "e:zj:p:a"
    assert len(ledger) == 1  # the record is kept (append-only audit)
    assert ledger.get(record.merge_id) is record
    assert ledger.undone_ids == frozenset({record.merge_id})
    assert record.undo == {"from": ["e:zj:p:a"], "to": "e:zj:p:b"}


def test_ledger_undo_twice_and_unknown_undo_raise():
    ledger = MergeLedger(book_id="zj")
    record = ledger.add(["e:zj:p:a"], "e:zj:p:b")
    ledger.undo(record.merge_id)
    with pytest.raises(MergeError):
        ledger.undo(record.merge_id)
    with pytest.raises(MergeError):
        ledger.undo("mg:0000000000000000")


def test_ledger_rejects_merge_cycle():
    ledger = MergeLedger(book_id="zj")
    ledger.add(["e:zj:p:a"], "e:zj:p:b")
    with pytest.raises(MergeError):
        ledger.add(["e:zj:p:b"], "e:zj:p:a")


def test_ledger_rejects_remerging_an_old_id():
    ledger = MergeLedger(book_id="zj")
    ledger.add(["e:zj:p:a"], "e:zj:p:b")
    with pytest.raises(MergeError):
        ledger.add(["e:zj:p:a"], "e:zj:p:c")


def test_ledger_apply_is_idempotent_by_merge_id():
    ledger = MergeLedger(book_id="zj")
    record = make_merge(
        "zj", ["e:zj:p:a"], "e:zj:p:b", at="2026-10-18T20:00:00+08:00"
    )
    ledger.apply(record)
    # a rerun of the same decision may carry a new timestamp; the stored record wins
    rerun = make_merge(
        "zj", ["e:zj:p:a"], "e:zj:p:b", at="2026-10-18T21:00:00+08:00"
    )
    assert rerun.merge_id == record.merge_id
    ledger.apply(rerun)
    assert len(ledger) == 1
    assert ledger.resolve("e:zj:p:a") == "e:zj:p:b"
    assert ledger.get(record.merge_id) is record


def test_ledger_rejects_reused_id_with_different_decision():
    ledger = MergeLedger(book_id="zj")
    record = make_merge("zj", ["e:zj:p:a"], "e:zj:p:b", kind="alias")
    ledger.apply(record)
    forged = MergeRecord.from_box_dict(
        {**record.box_dict(), "kind": "coref"}
    )
    with pytest.raises(MergeError):
        ledger.apply(forged)


def test_ledger_rejects_foreign_book():
    ledger = MergeLedger(book_id="zj")
    with pytest.raises(MergeError):
        ledger.add(["e:other:p:a"], "e:other:p:b", book_id="other")


def test_ledger_history_contains_active_and_undone_records():
    ledger = MergeLedger(book_id="zj")
    record = ledger.add(["e:zj:p:a"], "e:zj:p:b")
    ledger.undo(record.merge_id)
    assert ledger.history("e:zj:p:a") == [record]
    assert ledger.history("e:zj:p:b") == []


def test_ledger_jsonl_roundtrip(tmp_path):
    ledger = MergeLedger(book_id="zj")
    ledger.add(["e:zj:p:a"], "e:zj:p:b", kind="alias", decided_by="rule")
    ledger.add(["e:zj:p:b"], "e:zj:p:c", kind="coref", decided_by="human", confidence=0.8)
    path = ledger.to_jsonl(tmp_path / "identity" / "merges.jsonl")

    loaded = MergeLedger.from_jsonl(path, book_id="zj")
    assert len(loaded) == 2
    assert loaded.resolve("e:zj:p:a") == "e:zj:p:c"
    assert [record.kind for record in loaded.records] == ["alias", "coref"]
    assert path.read_text(encoding="utf-8").count("\n") == 2
