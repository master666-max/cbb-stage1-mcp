"""RAPTOR's recursive summarisation loop does not exist, and the code says it does.

Measured drift (A25)
--------------------
`derive/raptor.py:11` claims:

    * ``text`` is always ``None`` here -- a later synthesis pass writes summaries;

and `derive/integration.py:248` does call `synthesize_tree_texts(...)` — but on
`build_summary_tree(...)`'s output, a **different** tree. The RAPTOR rows never get text.

So the loop the paper is built on is absent: summarise a cluster, embed the summary, cluster again on
the summaries, repeat. What runs is greedy clustering over unit text, with `text` pinned to `None` at
`_emit_level_rows:267`. The `max_levels=4` + `threshold_decay=0.6` pair stands in for "until
infeasible", which is a different rule.

Why the comment matters
-----------------------
A reader who trusts it concludes the summaries are written elsewhere. They are not — for this tree.
That is the same shape as A16's `min_weight` comment: a deviation presented as fidelity.

What is pinned
--------------
* when a summariser is supplied, a cluster row carries the summary text and says who wrote it;
* the next level clusters on those summaries, not on the original unit bags — that is the feedback
  loop, and without it the tree is just a hierarchy of the same vectors;
* without a summariser the behaviour is unchanged (`text` stays None) and the rows say so.
"""

from __future__ import annotations

from typing import Any, Sequence

from cbb_stage1.derive.raptor import build_raptor_clusters

UNITS = [
    {"unit_id": f"u:t:v01:c{index:03d}", "volume": 1, "order": index}
    for index in range(1, 9)
]
TEXTS = {
    f"u:t:v01:c{index:03d}": text
    for index, text in enumerate(
        [
            "言万在集市上与拉法交手，两人都没有占到便宜",
            "言万离开集市，拉法留在原地",
            "拉法回到宗门复命，宗主脸色阴沉",
            "宗主命拉法去查那枚黑色令牌的来历",
            "言万在渡口遇见一名老者，老者提到令牌",
            "老者说令牌来自北方的旧战场",
            "拉法查到令牌曾属于一个消失的宗门",
            "两条线索在令牌上汇合",
        ],
        1,
    )
}


class Summariser:
    """Records what it was asked to summarise, and returns a recognisable summary."""

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []

    def __call__(self, texts: Sequence[str]) -> str:
        self.calls.append(tuple(texts))
        return f"摘要{len(self.calls)}：共{len(texts)}段"


def _rows(**kwargs: Any) -> list[dict]:
    return build_raptor_clusters("t", UNITS, TEXTS, **kwargs)


# ══ summaries reach the rows ═════════════════════════════════════════
def test_a_summariser_produces_cluster_text() -> None:
    summariser = Summariser()
    rows = _rows(summariser=summariser)
    with_text = [row for row in rows if row.get("text")]
    assert with_text, (
        "no cluster row carries text even though a summariser was supplied; the recursive "
        "summarisation loop is absent and `text` is pinned to None"
    )


def test_the_summary_names_its_producer() -> None:
    rows = _rows(summariser=Summariser())
    with_text = [row for row in rows if row.get("text")]
    assert with_text
    assert all("summar" in str(row.get("produced_by") or "") for row in with_text), (
        f"a summarised row does not say a summariser wrote it: {with_text[0].get('produced_by')!r}"
    )


def test_the_summariser_receives_the_member_texts() -> None:
    summariser = Summariser()
    _rows(summariser=summariser)
    assert summariser.calls, "the summariser was never called"
    first = summariser.calls[0]
    assert first, "the summariser was called with no texts"
    assert any(text in TEXTS.values() for text in first), (
        f"the summariser did not receive unit text: {first!r}"
    )


# ══ the feedback loop: level 2 clusters the summaries ════════════════
def test_the_next_level_clusters_the_summaries() -> None:
    """This is the loop the paper is built on, and the reason `text` exists at all."""
    summariser = Summariser()
    rows = _rows(summariser=summariser, max_levels=3)

    level_one = [row for row in rows if int(row["level"]) == 1]
    assert level_one, "no level-1 rows"
    # At least one higher level means a second clustering pass ran; it must have been fed summaries.
    higher = [row for row in rows if int(row["level"]) > 1]
    if higher:
        assert len(summariser.calls) >= len(level_one), (
            "a higher level exists but the summariser was not called for each level-1 cluster, so "
            "the next level clustered raw units rather than summaries"
        )


# ══ without a summariser nothing changes ═════════════════════════════
def test_without_a_summariser_text_stays_none() -> None:
    rows = _rows()
    assert rows
    assert all(row.get("text") is None for row in rows), (
        "text appeared without a summariser, which means it was fabricated"
    )


def test_the_rows_say_they_are_unsummarised() -> None:
    rows = _rows()
    assert all("no-summary" in str(row.get("produced_by") or "") or "structural" in str(
        row.get("produced_by") or ""
    ) for row in rows), (
        "an unsummarised row must say so rather than looking like a summarised one"
    )


def test_the_tree_still_validates() -> None:
    from cbb_stage1.derive.raptor import validate_raptor_tree

    assert validate_raptor_tree(_rows(summariser=Summariser())) == []
    assert validate_raptor_tree(_rows()) == []
