"""A speaker on the evidence is what unlocks the claims path (B6/B7).

``01-箱规格`` §5.16 says an assertion whose evidence is ``dialogue|inner`` defaults
to ``claim_layer=statement``, and a statement must carry a speaker. The extractor
had no way to supply one, so:

* every assertion stayed ``fact``;
* the claims stage (which only consumes ``statement``) produced nothing;
* and the narrative layer was structurally unimplemented.

Measured on a real chapter before this change: 7 assertions, all ``fact``, claims
0 — not because the claims code was broken, but because no statement could ever
exist.

These tests pin the whole chain: schema field -> normalized evidence speaker ->
claim layer -> a statement the box will accept.
"""

from __future__ import annotations

from cbb_stage1.engines.ai_reader.schema import (
    ChapterFact,
    CharacterFact,
    EventFact,
    ItemEventFact,
    LocationFact,
    RelationshipFact,
)
from cbb_stage1.models import Evidence, Speaker
from cbb_stage1.normalize.facts import _claim_layer, chapter_fact_to_engine_output, infer_mode

TEXT = (
    "「──毁灭宇宙！」\n\n"
    "她笑了。\n\n"
    "「但是，也只能努力撑下去了吧？」\n"
)


def _evidence(mode: str, speaker: Speaker | None = None) -> Evidence:
    return Evidence(
        unit_id="u:1", start=0, end=5, quote="abcde", occurrence=1, mode=mode, speaker=speaker
    )


def test_schema_exposes_a_speaker_field_on_every_evidence_bearing_class() -> None:
    """Without the field the extractor literally cannot attribute a line."""
    for cls in (
        CharacterFact,
        LocationFact,
        RelationshipFact,
        EventFact,
        ItemEventFact,
    ):
        assert "evidence_speaker" in cls.model_fields, f"{cls.__name__} has no evidence_speaker"
        field = cls.model_fields["evidence_speaker"]
        assert field.description, f"{cls.__name__}.evidence_speaker has no description"


def test_dialogue_without_a_speaker_stays_fact() -> None:
    """A statement with no speaker is quarantined, so it must not be emitted."""
    assert _claim_layer([_evidence("dialogue")]) == "fact"
    assert _claim_layer([_evidence("inner")]) == "fact"


def test_dialogue_with_a_speaker_becomes_statement() -> None:
    speaker = Speaker(kind="entity", entity_id=None, name="黑之魔王")
    assert _claim_layer([_evidence("dialogue", speaker)]) == "statement"
    assert _claim_layer([_evidence("inner", speaker)]) == "statement"


def test_narration_stays_fact_even_with_a_speaker() -> None:
    speaker = Speaker(kind="narrator", entity_id=None, name="叙述者")
    assert _claim_layer([_evidence("narration", speaker)]) == "fact"


def test_empty_evidence_is_fact() -> None:
    assert _claim_layer([]) == "fact"


def test_extractor_speaker_reaches_the_normalized_evidence() -> None:
    """The extractor's attribution must survive into the evidence record."""
    quote = "「──毁灭宇宙！」"
    fact = ChapterFact(
        chapter_id=1,
        novel_id="t",
        relationships=[
            RelationshipFact(
                person_a="黑之魔王",
                person_b="贝塔心叶",
                relation_type="敌对",
                evidence=quote,
                evidence_speaker="黑之魔王",
            )
        ],
    )
    output = chapter_fact_to_engine_output(fact, unit_id="u:1", text=TEXT, book_id="t")
    relation = next(a for a in output.assertions if a.predicate == "敌对")
    assert relation.evidence, "the quote was not anchored"
    evidence = relation.evidence[0]
    assert evidence.mode == "dialogue"
    assert evidence.speaker is not None
    assert evidence.speaker.name == "黑之魔王"
    assert relation.claim_layer == "statement", (
        "dialogue evidence with a speaker must produce a statement, "
        "otherwise the claims stage has nothing to consume"
    )


def test_missing_speaker_keeps_the_assertion_as_a_fact() -> None:
    """No attribution => no statement, and the fact is not lost."""
    quote = "「──毁灭宇宙！」"
    fact = ChapterFact(
        chapter_id=1,
        novel_id="t",
        relationships=[
            RelationshipFact(
                person_a="黑之魔王",
                person_b="贝塔心叶",
                relation_type="敌对",
                evidence=quote,
            )
        ],
    )
    output = chapter_fact_to_engine_output(fact, unit_id="u:1", text=TEXT, book_id="t")
    relation = next(a for a in output.assertions if a.predicate == "敌对")
    assert relation.evidence[0].mode == "dialogue"
    assert relation.evidence[0].speaker is None
    assert relation.claim_layer == "fact"
    assert relation.status == "asserted", "the fact itself must still be usable"


def test_mode_inference_reads_punctuation_only() -> None:
    assert infer_mode("「你好。」") == "dialogue"
    assert infer_mode("『内心』") == "dialogue"
    assert infer_mode("（我想逃。）") == "inner"
    assert infer_mode("(旁白)") == "inner"
    assert infer_mode("她笑了。") == "narration"
    # A quote that continues the narrator's sentence is narration, not dialogue.
    assert infer_mode("「Susurrador」，意思是「低语者」。") == "narration"
