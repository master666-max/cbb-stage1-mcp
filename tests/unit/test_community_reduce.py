"""C5/D13: a parent community summarises its children, not the raw assertions again.

Measured gap, corrected against the code
----------------------------------------
The audit said C5's summarizer had "no role runtime". It **is** wired (`derive/integration.py:370-375` passes
`aux_llm.generate`), so that part of the report is wrong. What is actually missing is the **reduce** stage:
`community.py:309` loops `for row in result` and builds every row's prompt from the raw assertions, so a
level-2 community re-reads the evidence its children already summarised.

That is the difference between a summariser and a hierarchy. GraphRAG's global search is map-reduce over
community reports precisely because the reduce step is what makes a whole-book answer possible from
community-sized inputs — and it is also D13's missing "global view" path: without it, nothing consumes the
tree from the top.

What is pinned
--------------
* a level-2 prompt contains its **children's summaries**, not the underlying assertion quotes;
* a level-1 prompt still uses assertions (the map stage is unchanged);
* when a child has no summary yet the parent falls back to assertions and **says so**, rather than silently
  degrading to an evidence-free summary;
* the order is deterministic, so a rebuilt box produces the same prompts;
* the reduce prompt forbids inventing facts, because a summary of summaries is where drift compounds.
"""

from __future__ import annotations

from typing import Any

from cbb_stage1.derive.reduce import build_reduce_prompt, summarise_tree


def _tree() -> list[dict[str, Any]]:
    """A two-level community tree: children 1 and 2, parent 10."""
    return [
        {
            "community_id": 1,
            "level": 1,
            "parent_id": 10,
            "member_entities": ["e:a", "e:b"],
            "title": "青牛镇",
            "summary": "言万与拉法在青牛镇初次交手。",
            "relationship_ids": ["a:1"],
        },
        {
            "community_id": 2,
            "level": 1,
            "parent_id": 10,
            "member_entities": ["e:c"],
            "title": "药园",
            "summary": "药园里藏着丹药线索。",
            "relationship_ids": ["a:2"],
        },
        {
            "community_id": 10,
            "level": 2,
            "parent_id": None,
            "member_entities": ["e:a", "e:b", "e:c"],
            "title": None,
            "summary": None,
            "relationship_ids": ["a:1", "a:2"],
        },
    ]


# ══ the reduce prompt consumes children ═════════════════════════════
def test_a_parent_prompt_carries_its_childrens_summaries() -> None:
    prompt = build_reduce_prompt(_tree()[2], _tree()[:2], {})
    assert "青牛镇" in prompt and "言万与拉法在青牛镇初次交手。" in prompt
    assert "药园里藏着丹药线索。" in prompt


def test_a_parent_prompt_does_not_re_read_the_raw_quotes() -> None:
    """The children already read the evidence; the parent's job is to combine, not to re-derive."""
    assertions = {"a:1": {"quote": "原始引文甲"}, "a:2": {"quote": "原始引文乙"}}
    prompt = build_reduce_prompt(_tree()[2], _tree()[:2], assertions)
    assert "原始引文甲" not in prompt, (
        "the parent prompt re-read the raw evidence its children already summarised"
    )


def test_the_reduce_prompt_forbids_invention() -> None:
    prompt = build_reduce_prompt(_tree()[2], _tree()[:2], {})
    assert "不得补充" in prompt or "只能使用" in prompt, (
        "a summary of summaries is where drift compounds; it must forbid adding facts"
    )


# ══ the fallback is explicit ════════════════════════════════════════
def test_a_child_without_a_summary_falls_back_to_evidence_and_records_it() -> None:
    children = [dict(_tree()[0], summary=None), _tree()[1]]
    prompt = build_reduce_prompt(_tree()[2], children, {"a:1": {"quote": "原始引文甲"}})
    assert "原始引文甲" in prompt, "the fallback did not include the evidence"
    assert "无摘要" in prompt or "回退" in prompt, (
        "the fallback happened silently; a reader cannot tell a combined summary from a re-derived one"
    )


# ══ the map stage is unchanged ══════════════════════════════════════
def test_a_level_one_prompt_still_uses_assertions() -> None:
    prompts = summarise_tree(
        _tree(),
        summarizer=lambda prompt: '{"title":"x","summary":"y"}',
        assertions={"a:1": {"quote": "原始引文甲", "predicate": "敌对"}},
    )
    # Row order is level-then-id, so prompts[0] is community 1 (which references a:1).
    assert "原始引文甲" in prompts[0], "the map stage stopped using evidence"


# ══ determinism ═════════════════════════════════════════════════════
def test_the_prompts_are_deterministic() -> None:
    def run() -> list[str]:
        return summarise_tree(
            _tree(), summarizer=lambda prompt: "{}", assertions={}
        )

    assert run() == run()


# ══ the reduce actually runs, in order ══════════════════════════════
def test_children_are_summarised_before_their_parent() -> None:
    order: list[str] = []

    def recorder(prompt: str) -> str:
        order.append("parent" if "更高层社区" in prompt else "child")
        return '{"title":"t","summary":"s"}'

    summarise_tree(_tree(), summarizer=recorder, assertions={})
    assert order[-1] == "parent", f"the parent was summarised before its children: {order}"
    assert order.count("parent") == 1


def test_a_parent_without_children_is_left_alone() -> None:
    orphan = [dict(_tree()[2], member_entities=["e:z"])]
    prompts = summarise_tree(orphan, summarizer=lambda p: "{}", assertions={})
    assert len(prompts) == len(orphan)

# ══ the real entry point uses the reduce path ═══════════════════════
def test_build_community_reports_summarises_parents_from_children() -> None:
    """The unit tests prove the reduce logic; this proves the stage actually calls it."""
    from cbb_stage1.derive.community import build_community_reports

    BOOK = "t"
    A = "e:t:person:aaaaaaaaaaaa"
    B = "e:t:person:bbbbbbbbbbbb"
    C = "e:t:person:cccccccccccc"
    assertions = [
        {
            "assertion_id": "a:1",
            "book_id": BOOK,
            "predicate": "敌对",
            "library": "关系",
            "roles": {"施事": A, "受事": B},
            "claim_layer": "fact",
            "confidence": 0.9,
            "confidence_method": "evidence_span",
            "status": "asserted",
            "evidence": [{"unit_id": "u:t:v01:c001", "start": 0, "end": 4, "quote": "对阵甲", "occurrence": 1}],
        },
        {
            "assertion_id": "a:2",
            "book_id": BOOK,
            "predicate": "结盟",
            "library": "关系",
            "roles": {"施事": B, "受事": C},
            "claim_layer": "fact",
            "confidence": 0.9,
            "confidence_method": "evidence_span",
            "status": "asserted",
            "evidence": [{"unit_id": "u:t:v01:c002", "start": 0, "end": 4, "quote": "结盟乙", "occurrence": 1}],
        },
    ]
    seen: list[str] = []

    def summarizer(prompt: str) -> str:
        seen.append("reduce" if "更高层社区" in prompt else "map")
        return '{"title":"t","summary":"s"}'

    rows = build_community_reports(
        BOOK,
        assertions,
        entity_volumes={A: 1, B: 1, C: 1},
        summarizer=summarizer,
    )
    assert rows
    assert "map" in seen, "no community was summarised at all"
    # With a single volume the level-2 parent exists, so the reduce stage must have run.
    level2 = [row for row in rows if int(row.get("level") or 0) == 2]
    assert level2, "the fixture produced no level-2 community, so the reduce path was not exercised"
    assert "reduce" in seen, f"a level-2 community exists but no reduce prompt was sent: {seen}"
    # The reduce must actually land: a summariser whose output is discarded is the defect this found.
    assert all(row.get("summary") == "s" for row in rows), (
        "the summaries were not written back: " + str([(r.get("community_id"), r.get("summary")) for r in rows])
    )
