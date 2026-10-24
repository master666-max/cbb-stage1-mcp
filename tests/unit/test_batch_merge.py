"""A14 batch entity merging, ported from the library's working implementation.

Source: `库/code/B-项目案例/B07-lorebase/scripts/02b_merge_dict.py` (171 lines, complete).

The source's own header explains why the naive approach fails, and that is the part worth porting
faithfully:

    "旧版 union-find 传递闭包会把泛称词(神/大人/舰长)当桥接, 造成巨型实体(实测吸收 189 别名)"

So the order is: merge exact names, **remove generic terms by fan-out**, then ask the model in batches.
The fan-out step cannot be dropped — transitive closure swallows everything through a bridging word.

`BATCH = 80` carries a measured reason (`150 会让 M3 思考爆炸顶爆 max_tokens, 输出 0 组`), and the merge
prompt carries four rules, including "different generations of one lineage are different entities".

What is pinned here is the mechanism, not the source's print statements: the four stages, the four prompt
rules' *effect* on parsing, and the accumulate-and-delete assembly.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from cbb_stage1.dedupe.batch_merge import (
    BLACKLIST,
    load_name_merged,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FANOUT,
    assemble,
    batch_merge,
    build_batches,
    fanout_clean,
    merge_prompt,
    parse_merge_groups,
)

BOOK = "t"


def _entry(name: str, etype: str = "person", desc: str = "", **kw: Any) -> dict[str, Any]:
    return {
        "name": name,
        "type": etype,
        "desc": desc,
        "aliases": list(kw.get("aliases", [])),
        "freq": int(kw.get("freq", 1)),
    }


# ══ the constants carry the source's measured values ════════════════
def test_the_fanout_default_is_the_sources() -> None:
    assert DEFAULT_FANOUT == 3


def test_the_batch_default_is_the_sources_measured_value() -> None:
    assert DEFAULT_BATCH_SIZE == 80, (
        "the source annotates 80 with a measurement (150 blows max_tokens and returns 0 groups); "
        "changing it without re-measuring is how that failure comes back silently"
    )


def test_the_blacklist_is_carried() -> None:
    """Generic honorifics are the bridging words that caused the 189-alias entity."""
    assert "大人" in BLACKLIST
    assert "神" in BLACKLIST


# ══ stage 1: fan-out removes generic terms ══════════════════════════
def test_a_name_in_many_hosts_aliases_is_generic() -> None:
    rows = [
        _entry("甲", aliases=["大人"]),
        _entry("乙", aliases=["大人"]),
        _entry("丙", aliases=["大人"]),
        _entry("丁", aliases=["大人"]),
    ]
    kept, generic = fanout_clean(rows, fanout=3)
    assert "大人" in generic, "a word in four entities' aliases was not treated as generic"
    for entry in kept.values():
        assert "大人" not in entry["aliases"], "the generic alias survived on a real entity"


def test_a_blacklisted_word_is_generic_even_when_rare() -> None:
    kept, generic = fanout_clean([_entry("甲", aliases=["神"])])
    assert "神" in generic
    assert "神" not in kept["甲"]["aliases"]


def test_a_rare_alias_is_kept() -> None:
    rows = [_entry("言万", aliases=["心叶"]), _entry("拉法")]
    kept, generic = fanout_clean(rows, fanout=3)
    assert "心叶" not in generic
    assert "心叶" in kept["言万"]["aliases"]


def test_a_generic_named_entity_is_dropped() -> None:
    """If the name itself is generic there is no entity to keep."""
    rows = [_entry("大人"), _entry("甲", aliases=["大人"]), _entry("乙", aliases=["大人"]),
            _entry("丙", aliases=["大人"])]
    kept, _generic = fanout_clean(rows, fanout=3)
    assert "大人" not in kept


# ══ stage 2: batching by type, largest bucket first ═════════════════
def test_batches_are_grouped_by_type() -> None:
    rows = [_entry("a1"), _entry("a2"), _entry("p1", etype="place")]
    batches = build_batches(load_name_merged(rows), batch_size=80)
    types = [batch["type"] for batch in batches]
    assert types == sorted(types, key=lambda name: -sum(1 for r in rows if r["type"] == name))
    for batch in batches:
        assert all(rows_name_type(row_name, rows) == batch["type"] for row_name in batch["names"])


def rows_name_type(name: str, rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        if row["name"] == name:
            return str(row["type"])
    raise KeyError(name)


def test_names_within_a_batch_are_frequency_ordered() -> None:
    rows = [_entry("low", freq=1), _entry("high", freq=9), _entry("mid", freq=4)]
    batches = build_batches(load_name_merged(rows), batch_size=80)
    assert batches[0]["names"] == ["high", "mid", "low"]


def test_a_bucket_larger_than_the_batch_size_is_split() -> None:
    rows = [_entry(f"n{index:03d}") for index in range(5)]
    batches = build_batches(load_name_merged(rows), batch_size=2)
    assert sum(len(batch["names"]) for batch in batches) == 5
    assert all(len(batch["names"]) <= 2 for batch in batches)


# ══ stage 3: the prompt carries the four rules ══════════════════════
def test_the_prompt_carries_the_four_rules() -> None:
    text = merge_prompt(
        {"甲": {"type": "person", "desc": "", "freq": 2, "aliases": set()}}, "person"
    )
    assert "宁可漏合" in text, "the caution against wrong merges is missing"
    assert "代" in text, "the different-generation rule is missing"
    assert "跨类型" in text or "同类型" in text, "the cross-type rule is missing"
    assert "≥2" in text or ">=2" in text, "the at-least-two rule is missing"


# ══ stage 4: parsing rejects what the prompt forbids ════════════════
def test_a_group_of_two_is_accepted() -> None:
    rows = {"甲": _entry("甲"), "乙": _entry("乙")}
    groups = parse_merge_groups(
        '[{"canonical":"甲","type":"person","desc":"d","merged":["甲","乙"]}]', rows, "person"
    )
    assert len(groups) == 1
    assert set(groups[0]["merged"]) == {"甲", "乙"}


def test_a_single_item_group_is_rejected() -> None:
    rows = {"甲": _entry("甲"), "乙": _entry("乙")}
    assert parse_merge_groups(
        '[{"canonical":"甲","type":"person","merged":["甲"]}]', rows, "person"
    ) == []


def test_a_cross_type_group_is_rejected() -> None:
    rows = {"甲": _entry("甲", "person"), "某地": _entry("某地", "place")}
    assert parse_merge_groups(
        '[{"canonical":"甲","type":"place","merged":["甲","某地"]}]', rows, "person"
    ) == [], "a group mixing a person and a place was accepted"


def test_a_group_naming_an_unknown_candidate_is_rejected() -> None:
    rows = {"甲": _entry("甲")}
    assert parse_merge_groups(
        '[{"canonical":"甲","type":"person","merged":["甲","不存在"]}]', rows, "person"
    ) == []


def test_malformed_output_yields_no_groups() -> None:
    assert parse_merge_groups("not json", {}, "person") == []


# ══ stage 5: assembly accumulates and deletes ═══════════════════════
def test_frequency_accumulates() -> None:
    entries = {"甲": {"type": "person", "desc": "", "freq": 3, "aliases": set()},
               "乙": {"type": "person", "desc": "", "freq": 5, "aliases": set()}}
    final = assemble(entries, [{"canonical": "甲", "merged": ["甲", "乙"]}])
    assert final["甲"]["freq"] == 8


def test_merged_names_become_aliases() -> None:
    entries = {"甲": {"type": "person", "desc": "", "freq": 1, "aliases": {"心叶"}},
               "乙": {"type": "person", "desc": "", "freq": 1, "aliases": {"二愣子"}}}
    final = assemble(entries, [{"canonical": "甲", "merged": ["甲", "乙"]}])
    assert "乙" in final["甲"]["aliases"], "the merged name itself did not become an alias"
    assert "二愣子" in final["甲"]["aliases"], "the merged entity's aliases were lost"


def test_the_longer_description_wins() -> None:
    entries = {"甲": {"type": "person", "desc": "短", "freq": 1, "aliases": set()},
               "乙": {"type": "person", "desc": "更长的一段描述", "freq": 1, "aliases": set()}}
    final = assemble(entries, [{"canonical": "甲", "merged": ["甲", "乙"]}])
    assert final["甲"]["desc"] == "更长的一段描述"


def test_the_merged_entry_is_deleted() -> None:
    entries = {"甲": {"type": "person", "desc": "", "freq": 1, "aliases": set()},
               "乙": {"type": "person", "desc": "", "freq": 1, "aliases": set()}}
    final = assemble(entries, [{"canonical": "甲", "merged": ["甲", "乙"]}])
    assert "乙" not in final


def test_a_canonical_missing_from_the_batch_falls_back_to_a_member() -> None:
    entries = {"乙": {"type": "person", "desc": "", "freq": 1, "aliases": set()},
               "丙": {"type": "person", "desc": "", "freq": 1, "aliases": set()}}
    final = assemble(entries, [{"canonical": "不存在", "merged": ["乙", "丙"]}])
    assert "乙" in final, "a group whose canonical does not exist was dropped entirely"


# ══ the driver ═════════════════════════════════════════════════════
class RecordingLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "[]"


def test_batch_merge_runs_one_call_per_batch() -> None:
    rows = [_entry(f"n{index:03d}") for index in range(5)]
    llm = RecordingLLM("[]", "[]", "[]")
    result = batch_merge(rows, llm, batch_size=2)
    assert len(llm.prompts) == 3, f"expected 3 batches, saw {len(llm.prompts)}"
    assert result["batches"] == 3


def test_a_failing_batch_does_not_abort_the_merge() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("model down")

    rows = [_entry("甲"), _entry("乙")]
    result = batch_merge(rows, boom, batch_size=80)
    assert result["final"], "one failed batch lost the whole dictionary"


def test_the_merge_groups_are_reported_for_audit() -> None:
    rows = [_entry("甲", freq=3), _entry("乙", freq=1)]
    llm = RecordingLLM(json.dumps([{"canonical": "甲", "type": "person", "merged": ["甲", "乙"]}]))
    result = batch_merge(rows, llm)
    assert result["groups"], "the audit's merge_groups is empty"
    assert result["groups"][0]["canonical"] == "甲"


def test_the_result_is_deterministic() -> None:
    rows = [_entry("甲"), _entry("乙")]
    first = batch_merge(rows, RecordingLLM("[]"))
    second = batch_merge(rows, RecordingLLM("[]"))
    assert [row["name"] for row in first["final"]] == [row["name"] for row in second["final"]]
