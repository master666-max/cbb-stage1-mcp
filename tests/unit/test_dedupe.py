"""Unit tests for :mod:`cbb_stage1.dedupe` (semhash + description fragments).

Ported behaviour is checked against the upstream expectations in
``C01-kg-gen`` (NFKC/singularize/threshold/reversible mapping) and the v6
contract (exact description-fragment dedupe, observations are never deleted,
no LLM call in the stage-1 skeleton).
"""

from __future__ import annotations

import pytest

import cbb_stage1.dedupe as dedupe_pkg
from cbb_stage1.dedupe import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DescribeResult,
    DedupeMethod,
    DedupeResult,
    FragmentDedupeResult,
    Observation,
    dedupe_description_fragments,
    dedupe_fragments,
    dedupe_graph,
    dedupe_items,
    dedupe_relations,
    describe,
    fragment_key,
    merge_observations,
    normalize_fragment,
    normalize_text,
    render_observations,
    semhash_key,
    similarity,
    singularize,
    tokenize,
)

# ---------------------------------------------------------------------------
# semhash: normalisation chain
# ---------------------------------------------------------------------------


def test_normalize_text_nfkc_casefold_and_whitespace():
    assert normalize_text("  Ｈａｒｒｙ　Ｐｏｔｔｅｒ  ") == "harry potter"
    assert normalize_text("A\n\tB") == "a b"
    assert normalize_text("  哈利·波特  ") == "哈利·波特"


def test_normalize_text_rejects_non_str():
    with pytest.raises(TypeError):
        normalize_text(42)  # type: ignore[arg-type]


def test_singularize_common_english_plurals():
    assert singularize("stories boxes wolves children people") == (
        "story box wolf child person"
    )
    assert singularize("knives leaves cities boys") == "knife leaf city boy"


def test_singularize_leaves_invariants_and_cjk_alone():
    assert singularize("news status analysis this always") == (
        "news status analysis this always"
    )
    assert singularize("哈利波特 斯内普") == "哈利波特 斯内普"


def test_tokenize_keeps_order_and_handles_cjk_and_possessive():
    assert tokenize("New York") == ("new", "york")
    assert tokenize("哈利·波特") == ("哈", "利", "波", "特")
    assert tokenize("Harry's wand") == ("harry", "wand")


# ---------------------------------------------------------------------------
# semhash: key / similarity
# ---------------------------------------------------------------------------


def test_semhash_key_is_order_insensitive_and_versioned():
    key = semhash_key("New York")
    assert key.startswith("sh1:")
    assert len(key) == len("sh1:") + 64
    assert semhash_key("york   NEW") == key
    assert semhash_key("New Jersey") != key


def test_semhash_key_collapses_plural_and_punctuation_variants():
    assert semhash_key("wolves") == semhash_key("wolf")
    assert semhash_key("哈利·波特") == semhash_key("哈利波特")
    assert semhash_key("Harry Potter.") == semhash_key("harry potter")


def test_similarity_bounds_and_known_values():
    assert similarity("New York", "new york") == 1.0
    assert similarity("New York", "New Jersey") == pytest.approx(1 / 3)
    assert similarity("哈利", "哈利波特") == pytest.approx(0.5)
    assert similarity("", "") == 1.0
    assert similarity("", "x") == 0.0


# ---------------------------------------------------------------------------
# semhash: dedupe_items mapping / stats
# ---------------------------------------------------------------------------


def test_dedupe_items_first_wins_with_reversible_mapping():
    result = dedupe_items(["A", "a", "A "])
    assert isinstance(result, DedupeResult)
    assert result.selected == ("A",)
    assert result.total_items == 3
    assert result.deduplicated_items == 1
    assert result.duplicate_items == 2
    assert result.reduction == pytest.approx(200 / 3)
    assert result.mapping == {"A": "A", "a": "A", "A ": "A"}
    assert result.duplicates == {"a": "A", "A ": "A"}
    assert result.canonical_of("a") == "A"
    assert result.canonical_of("unknown") == "unknown"
    assert result.rollback("A") == ("A", "a", "A ")
    assert result.clusters == (("A", "a", "A "),)


def test_dedupe_items_preserves_input_order_of_representatives():
    result = dedupe_items(["b", "a", "B"])
    assert result.selected == ("b", "a")
    assert result.mapping["B"] == "b"


def test_dedupe_items_near_duplicate_threshold_is_tunable():
    default = dedupe_items(["New York", "New York City"])
    assert default.selected == ("New York", "New York City")
    loose = dedupe_items(["New York", "New York City"], threshold=0.6)
    assert loose.selected == ("New York",)
    assert loose.mapping["New York City"] == "New York"


def test_dedupe_items_threshold_one_still_groups_token_bag_equivalents():
    result = dedupe_items(["New York", "York  New"], threshold=1.0)
    assert result.selected == ("New York",)


def test_dedupe_items_does_not_transitively_union():
    # a~b and b~c at 0.65, but a and c only reach 0.6: each item is compared to
    # the current representative only, so c starts its own group (no union).
    result = dedupe_items(
        ["t1 t2 t3 t4", "t1 t2 t3", "t1 t2 t3 t5"], threshold=0.65
    )
    assert result.selected == ("t1 t2 t3 t4", "t1 t2 t3 t5")
    assert result.mapping["t1 t2 t3"] == "t1 t2 t3 t4"


def test_dedupe_items_empty_and_stats():
    empty = dedupe_items([])
    assert empty.selected == ()
    assert empty.total_items == 0
    assert empty.reduction == 0.0
    result = dedupe_items(["x", "x", "y"])
    assert "Reduction: 33.3%" in result.stats()


def test_dedupe_items_validates_threshold_and_item_type():
    with pytest.raises(ValueError):
        dedupe_items(["x"], threshold=0.0)
    with pytest.raises(ValueError):
        dedupe_items(["x"], threshold=1.5)
    with pytest.raises(TypeError):
        dedupe_items(["x"], threshold="0.9")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dedupe_items(["x", 1])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# semhash: relations / graph
# ---------------------------------------------------------------------------


def test_dedupe_relations_rewrites_and_collapses():
    entities = dedupe_items(["Jon Snow", "Jon Snow ", "Arya Stark"])
    edges = dedupe_items(["knows", "Knows"])
    relations = [
        ("Jon Snow", "knows", "Arya Stark"),
        ("Jon Snow ", "Knows", "Arya Stark"),
        ("Arya Stark", "knows", "Jon Snow"),
    ]
    assert dedupe_relations(relations, entities, edges) == (
        ("Jon Snow", "knows", "Arya Stark"),
        ("Arya Stark", "knows", "Jon Snow"),
    )


def test_dedupe_relations_accepts_plain_mappings_and_identity():
    assert dedupe_relations(
        [("a", "p", "b"), ("a", "p", "b")],
        {"a": "A", "b": "B"},
        {"p": "P"},
    ) == (("A", "P", "B"),)
    assert dedupe_relations([["x", "y", "z"]]) == (("x", "y", "z"),)


def test_dedupe_relations_rejects_bad_arity():
    with pytest.raises(ValueError):
        dedupe_relations([("a", "b")])


def test_dedupe_graph_semhash_end_to_end():
    result = dedupe_graph(
        ["Arya", "Arya ", "Jon"],
        ["knows", "Knows"],
        [("Arya", "knows", "Jon"), ("Arya ", "Knows", "Jon")],
    )
    assert result.entities == ("Arya", "Jon")
    assert result.edges == ("knows",)
    assert result.relations == (("Arya", "knows", "Jon"),)
    assert result.entity_clusters == (("Arya", "Arya "), ("Jon",))
    assert result.edge_clusters == (("knows", "Knows"),)
    assert result.entity_map["Arya "] == "Arya"
    assert result.edge_map["Knows"] == "knows"


def test_dedupe_graph_unimplemented_methods_raise_instead_of_degrading():
    assert DedupeMethod.SEMHASH.value == "semhash"
    for method in (DedupeMethod.LM_BASED, DedupeMethod.FULL, "full", "lm_based"):
        with pytest.raises(NotImplementedError):
            dedupe_graph([], [], [], method=method)
    with pytest.raises(ValueError):
        dedupe_graph([], [], [], method="bogus")


def test_default_threshold_matches_upstream():
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.95


# ---------------------------------------------------------------------------
# describe: fragment normalisation / exact dedupe
# ---------------------------------------------------------------------------


def test_normalize_fragment_collapses_width_and_whitespace_only():
    assert normalize_fragment("　他　笑了  ") == "他 笑了"
    assert normalize_fragment("ＡＢ　Ｃ") == "AB C"
    # Exact dedupe keeps case and punctuation semantics.
    assert normalize_fragment("He smiled.") == "He smiled."
    assert normalize_fragment("he smiled.") == "he smiled."


def test_normalize_fragment_rejects_non_str():
    with pytest.raises(TypeError):
        normalize_fragment(b"bytes")  # type: ignore[arg-type]


def test_fragment_key_is_normalized_and_versioned():
    key = fragment_key("他笑了")
    assert key.startswith("df1:")
    assert fragment_key("  他笑了 ") == key
    assert fragment_key("他笑了。") != key
    assert fragment_key("他笑了") != fragment_key("他笑 了")


def test_dedupe_fragments_exact_first_wins():
    result = dedupe_fragments(["他笑了", " 他笑了 ", "他是谁", "他笑了。", ""])
    assert isinstance(result, FragmentDedupeResult)
    assert result.kept == ("他笑了", "他是谁", "他笑了。")
    assert result.dropped == (" 他笑了 ",)
    assert result.ignored == ("",)
    assert result.total_items == 5
    assert result.deduplicated_items == 3
    assert result.duplicate_items == 1
    assert result.reduction == pytest.approx(40.0)
    assert result.mapping[" 他笑了 "] == "他笑了"
    assert result.canonical_of(" 他笑了 ") == "他笑了"
    assert result.rollback("他笑了") == ("他笑了", " 他笑了 ")
    assert result.duplicates == {" 他笑了 ": "他笑了"}


def test_dedupe_fragments_repeated_identical_surface_counts_occurrences():
    result = dedupe_fragments(["a", "a"])
    assert result.kept == ("a",)
    assert result.dropped == ("a",)
    assert result.total_items == 2
    assert result.clusters == (("a", "a"),)
    assert result.mapping == {"a": "a"}
    assert result.duplicates == {}


def test_dedupe_fragments_empty_input_and_type_guard():
    empty = dedupe_fragments([])
    assert empty.kept == ()
    assert empty.total_items == 0
    assert empty.reduction == 0.0
    with pytest.raises(TypeError):
        dedupe_fragments([None])  # type: ignore[list-item]


def test_dedupe_description_fragments_matches_generic_alias():
    assert dedupe_description_fragments(["x", " x "]) == dedupe_fragments(["x", " x "])


# ---------------------------------------------------------------------------
# describe: observations + interface skeleton
# ---------------------------------------------------------------------------


def test_merge_observations_keeps_first_and_all_provenance():
    observations = (
        Observation("他笑了", unit_id="u1", source="ch1"),
        Observation(" 他笑了 ", unit_id="u2", source="ch2"),
        Observation("他是谁", unit_id="u3", source="ch3"),
        Observation("   ", unit_id="u4"),
    )
    kept = merge_observations(observations)
    assert [obs.text for obs in kept] == ["他笑了", "他是谁"]
    assert [obs.unit_id for obs in kept] == ["u1", "u3"]
    assert [obs.source for obs in kept] == ["ch1", "ch3"]
    # Old observations are not deleted from the caller's data.
    assert len(observations) == 4
    assert observations[1].unit_id == "u2"


def test_merge_observations_accepts_plain_strings():
    assert [obs.text for obs in merge_observations(["a", " a ", ""])] == ["a"]


def test_merge_observations_type_guard():
    with pytest.raises(TypeError):
        merge_observations([1])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        Observation(text=1)  # type: ignore[arg-type]


def test_describe_without_synthesizer_returns_observations_only():
    result = describe(
        "e1",
        [
            Observation("他笑了", unit_id="u1"),
            Observation(" 他笑了 ", unit_id="u2"),
            Observation("他是谁", unit_id="u3"),
            "",
        ],
    )
    assert isinstance(result, DescribeResult)
    assert result.entity_id == "e1"
    assert result.mode == "observations_only"
    assert result.text is None
    assert result.inferred_ratio is None
    assert [obs.text for obs in result.observations] == ["他笑了", "他是谁"]
    assert [obs.unit_id for obs in result.observations] == ["u1", "u3"]
    assert result.fragment_dedupe.total_items == 4
    assert len(result.fragment_dedupe.dropped) == 1
    assert len(result.fragment_dedupe.ignored) == 1
    assert len(result.notes) == 2


def test_describe_with_synthesizer_runs_and_attributes_sentences():
    """P3-A: a synthesiser is no longer NotImplementedError; it runs and the
    synthesised sentences keep an observation fallback."""

    class FakeSynthesizer:
        def synthesize(self, entity_id: str, fragments: tuple[str, ...]) -> str:
            assert entity_id == "e1"
            assert fragments == ("他笑了",)
            return "他笑了。他是谁"

    result = describe("e1", ["他笑了"], synthesizer=FakeSynthesizer())

    assert result.mode == "synthesized"
    assert result.text == "他笑了。他是谁"
    assert [sentence.text for sentence in result.sentences] == ["他笑了。", "他是谁"]
    assert result.sentences[0].fallbacks == ("他笑了",)
    assert result.sentences[0].inferred is False
    assert result.sentences[1].inferred is True
    assert result.inferred_ratio == 0.5
    assert [obs.text for obs in result.observations] == ["他笑了"]


def test_describe_type_guard_and_empty_input():
    with pytest.raises(TypeError):
        describe(123, ["x"])  # type: ignore[arg-type]
    result = describe("e1", [])
    assert result.observations == ()
    assert result.mode == "observations_only"


def test_render_observations_skips_blank_and_honours_bullet():
    rendered = render_observations(
        [Observation("a"), Observation("  "), "b"], bullet="* "
    )
    assert rendered == "* a\n* b"


# ---------------------------------------------------------------------------
# package surface
# ---------------------------------------------------------------------------


def test_package_exports_are_consistent():
    assert len(dedupe_pkg.__all__) == len(set(dedupe_pkg.__all__))
    for name in dedupe_pkg.__all__:
        assert hasattr(dedupe_pkg, name), name
