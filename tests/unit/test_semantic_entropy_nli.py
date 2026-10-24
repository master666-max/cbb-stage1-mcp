"""Semantic entropy's clustering must be able to use a real NLI, and must say which it used.

Measured substitution (B2)
--------------------------
`confidence/semantic_entropy.py:128` `similar()` is documented as a "meaning-space similarity proxy"
and decides cluster membership from character-bigram Jaccard plus `SequenceMatcher` ratio. The paper's
clustering is by *meaning*: two answers that say the same thing in different words belong together,
and two that differ in one character but mean opposite things do not.

Character similarity fails both directions — "他活着" / "他死了" share most bigrams, and
"言万离开了" / "言万走了" share few. So the entropy estimate is over spelling, not meaning, and
`confidence` derived from it is not the paper's quantity.

The module documents itself as a proxy, which is why this is a SUBSTITUTE rather than a DRIFT. What
was missing is that nothing could replace it: the repo already has an NLI channel
(`temporal/nli_client.build_nli`), and `confidence/` had zero references to it.

What is pinned
--------------
* a pairwise similarity callable can be injected, and when present it decides clustering;
* the returned record says which method produced it, so a reader can tell a meaning-based entropy
  from a spelling-based one;
* with nothing injected the bigram path still works and is labelled as the proxy it is.
"""

from __future__ import annotations

from typing import Sequence

from cbb_stage1.confidence import semantic_entropy as se

SAME = "言万离开了集市"
PARAPHRASE = "言万从集市上走了"   # same meaning, few shared characters
OPPOSITE = "言万没有离开集市"     # opposite meaning, many shared characters


# ══ an injected judge decides clustering ══════════════════════════════
def test_an_injected_similarity_decides_the_clusters() -> None:
    """A judge that knows the two phrasings mean the same thing must merge them."""
    def judge(a: str, b: str) -> bool:
        return {a, b} <= {SAME, PARAPHRASE} or a == b

    result = se.semantic_entropy([SAME, PARAPHRASE, SAME], judge=judge)
    assert result["usable"] is True
    assert len(result["clusters"]) == 1, (
        f"the injected judge said the answers are equivalent, but they were split: {result['clusters']}"
    )


def test_the_injected_judge_separates_opposites() -> None:
    def judge(a: str, b: str) -> bool:
        return a == b

    result = se.semantic_entropy([SAME, OPPOSITE, SAME], judge=judge)
    assert len(result["clusters"]) == 2, (
        f"opposite answers were merged: {result['clusters']}"
    )


# ══ the record says which method was used ═════════════════════════════
def test_the_result_records_the_method_when_a_judge_is_injected() -> None:
    result = se.semantic_entropy([SAME, PARAPHRASE], judge=lambda a, b: a == b)
    assert result.get("similarity_method") == "injected", (
        f"the result does not say an external judge produced it: {result.get('similarity_method')!r}"
    )


def test_the_result_records_the_proxy_when_nothing_is_injected() -> None:
    result = se.semantic_entropy([SAME, PARAPHRASE, SAME])
    assert result.get("similarity_method") == "char_bigram_proxy", (
        "the spelling-based path must label itself; otherwise a proxy reads as the paper's method"
    )


# ══ the proxy path is unchanged ══════════════════════════════════════
def test_the_proxy_still_clusters_identical_answers() -> None:
    result = se.semantic_entropy(["同样的话", "同样的话", "同样的话"])
    assert len(result["clusters"]) == 1
    assert result["confidence"] == 1.0


def test_the_proxy_still_splits_clearly_different_answers() -> None:
    result = se.semantic_entropy(["完全不同的甲", "毫不相干的乙"])
    assert len(result["clusters"]) == 2


def test_a_judge_that_raises_falls_back_rather_than_aborting() -> None:
    """One broken pair must not kill the confidence estimate.

    The two strings must differ: identical answers short-circuit without consulting the judge, so the
    judge would never fail and the test would prove nothing about the fallback.
    """
    def judge(a: str, b: str) -> bool:
        raise RuntimeError("nli down")

    result = se.semantic_entropy([SAME, PARAPHRASE], judge=judge)
    assert result["usable"] is True
    assert result.get("similarity_method") == "char_bigram_proxy", (
        "after the judge failed the record must not still claim an injected judge"
    )


def test_the_judge_is_not_consulted_for_identical_strings() -> None:
    """Cheap short-circuit: equality needs no model call."""
    calls: list[tuple[str, str]] = []

    def judge(a: str, b: str) -> bool:
        calls.append((a, b))
        return False

    se.semantic_entropy([SAME, SAME], judge=judge)
    assert not calls, f"the judge was asked about two identical strings: {calls}"
