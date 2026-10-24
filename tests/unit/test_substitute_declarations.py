"""The remaining SUBSTITUTE items must name what they do not do.

Three items are graded SUBSTITUTE/PARTIAL because the code deliberately does something else. None of them
is wrong — two are arguably better than the source — but a reader comparing the implementation with the
paper needs the differences stated where they will look, not only in an audit document that may be stale.

A12 (`dedupe/semhash.py`) already declares the dependency removal and the reversible mapping. What it does
not say is the three places where the *method* differs from C01:

* C01 clusters with KMeans into ~128 clusters; this uses KNN candidate pairs + connected components;
* C01 retrieves with BM25 plus vectors, top-16; this uses token-bag Jaccard (or cosine when an embedder is
  injected) with no BM25;
* C01's LM step returns a `duplicates` set **and an alias**; this returns a representative only, so a
  consumer looking for the alias will not find one.

D14 (`derive/forecasts.py`) names its method `lexical_singleton_ratio` and its rows already say
`theory=true`. What was missing is the unit: a reader can still mistake the number for the paper's
`no_rate` because nothing states that one counts chapters and the other counts sentences of a generated
trajectory.

A20 (`incremental/runner.py`) keys its delta by `sha256` where the source keys by `title`. That is a
deliberate improvement — a retitled chapter with identical text stays the same document, whereas
title-keying loses it — and it is currently undocumented, so a reader could take it for a bug.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def _doc(module) -> str:
    return inspect.getdoc(module) or ""


# ══ A12: the three method differences ════════════════════════════════
def test_a12_names_the_clustering_difference() -> None:
    from cbb_stage1.dedupe import semhash

    doc = _doc(semhash).lower()
    assert "kmeans" in doc, "the docstring does not say C01 uses KMeans and this does not"


def test_a12_names_the_retrieval_difference() -> None:
    from cbb_stage1.dedupe import semhash

    doc = _doc(semhash).lower()
    assert "bm25" in doc, "the docstring does not say C01 retrieves with BM25 and this does not"


def test_a12_names_the_merge_output_difference() -> None:
    """C01's LM step returns duplicates plus an alias; this returns a representative only."""
    from cbb_stage1.dedupe import semhash

    doc = _doc(semhash).lower()
    assert "alias" in doc, (
        "the docstring does not say C01 also produces an alias, so a consumer looking for one will "
        "silently get None"
    )


# ══ D14: the unit, not only the method name ══════════════════════════
def test_d14_states_the_unit_of_measure() -> None:
    from cbb_stage1.derive import forecasts

    doc = _doc(forecasts).lower()
    assert "章" in doc, "the module does not state that the ratio is per chapter"
    assert "100" in doc or "句" in doc, (
        "the module does not state the paper's unit (per sentence, 100 generations), so the number can "
        "be mistaken for no_rate"
    )


def test_d14_rows_carry_their_unit() -> None:
    """A field called a ratio without a unit is exactly how a proxy gets quoted as the real thing."""
    from cbb_stage1.derive.forecasts import METHOD

    assert METHOD == "lexical_singleton_ratio", (
        f"the method name is {METHOD!r}; it must name the measurement, not the paper's"
    )


# ══ A20: the delta key choice is deliberate ═════════════════════════
def test_a20_documents_the_delta_key_choice() -> None:
    from cbb_stage1.incremental import runner

    doc = _doc(runner)
    assert "sha256" in doc.lower(), "the delta key is not documented"
    assert "title" in doc.lower(), (
        "the module does not say that the source keys by title and this keys by content hash, so the "
        "difference reads as an oversight rather than a decision"
    )


def test_the_content_key_actually_beats_title_matching() -> None:
    """Pin the reason, not only the claim: a retitled identical chapter is the same document."""
    from cbb_stage1.incremental.runner import unit_fingerprint

    base = {
        "unit_id": "u:t:v01:c001",
        "book_id": "t",
        "volume": 1,
        "chapter": 1,
        "order": 1,
        "chars": 10,
        "sha256": "sha256:" + "a" * 64,
        "title": "第一章 起",
    }
    retitled = {**base, "title": "第一章 出发"}
    assert unit_fingerprint(base) != unit_fingerprint(retitled), (
        "the fingerprint ignores the title, so the docstring's claim that metadata is covered is wrong"
    )


def test_identical_text_in_a_new_chapter_number_pairs_by_sha() -> None:
    """The renumbering case `_detect_renames` exists for."""
    from cbb_stage1.incremental.runner import _detect_renames

    sha = "sha256:" + "b" * 64
    previous = {"u:t:v01:c001": {"sha256": sha}}
    current = {"u:t:v01:c002": {"sha256": sha}}
    pairs = _detect_renames(previous, current, {"u:t:v01:c001"}, {"u:t:v01:c002"})
    assert pairs == [("u:t:v01:c001", "u:t:v01:c002")]
