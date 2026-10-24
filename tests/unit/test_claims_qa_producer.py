"""Something must answer the claim questions, or the four-value rule only ever uses its fourth value.

Measured gap (B6)
-----------------
`claims/integration.py:194` reads `upstream["claims_qa"]` and nothing in the pipeline ever supplies
it — only tests inject it. So every claim is built from a single `unanswerable` question, the
verdict is always `insufficient_evidence`, and nothing is ever promoted. The design's four-value
rule (supported / refuted / insufficient / unanswerable) collapses to one value in every real run.
The stage is honest about it (it appends a note saying exactly this), but honest emptiness is still
emptiness.

What the producer may and may not claim
---------------------------------------
A local producer can ground a statement in its own evidence: are the quotes really in the unit, and
do they actually mention the entities the assertion is about? That justifies `support` — the text
does back the claim.

It can **not** justify `refute`. Refutation needs contradiction analysis, which is conflicts/CoVe's
job. So this producer emits either `support` with answers, or nothing at all (leaving the existing
`unanswerable` fallback). It never invents a refutation, and it never emits answers on an
`unanswerable` question — `qa.py:183` rejects that.
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.claims.integration import build_claims_qa
from cbb_stage1.claims import run as claims_run
from cbb_stage1.assemble import run as assemble_run
from cbb_stage1.resolve import run as resolve_run

BOOK = "tst"
CREATED = "2026-10-18T23:00:00+08:00"
UNIT_A = "u:tst:v01:c001"
UNIT_B = "u:tst:v01:c002"
QUOTE_A = "言万在集市上与拉法交手"
QUOTE_B = "拉法与言万在渡口再次交手"
TEXT_A = f"天刚亮，{QUOTE_A}，两人都没占到便宜。"
TEXT_B = f"三日后，{QUOTE_B}，围观的人挤满了河岸。"


def _statement(unit_id: str, quote: str, start: int = 0) -> dict:
    return {
        "assertion_id": f"a:{unit_id[-4:]}00000000000000",
        "book_id": BOOK,
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "言万", "受事": "拉法"},
        "qualifiers": {},
        "claim_layer": "statement",
        "speaker": {"kind": "entity", "entity_id": None, "name": "言万"},
        "evidence": [
            {
                "unit_id": unit_id,
                "start": start,
                "end": start + len(quote),
                "quote": quote,
                "occurrence": 1,
                "mode": "dialogue",
                "speaker": {"kind": "entity", "entity_id": None, "name": "言万"},
            }
        ],
        "confidence": 0.9,
        "confidence_method": "evidence_span",
    }


def _texts() -> dict[str, str]:
    return {UNIT_A: TEXT_A, UNIT_B: TEXT_B}


# ══ the grounding check ════════════════════════════════════════════════
def test_a_grounded_statement_gets_a_support_question() -> None:
    statement = _statement(UNIT_A, QUOTE_A, TEXT_A.index(QUOTE_A))
    qa = build_claims_qa([statement], _texts())
    assert qa.get(statement["assertion_id"]), (
        "a statement whose quotes are in the unit and whose roles appear in those quotes was not "
        "answered; the four-value rule would fall back to unanswerable again"
    )
    question = qa[statement["assertion_id"]][0]
    assert question["stance"] == "support"
    assert question["answers"], "a support question must carry answers"


def test_a_quote_missing_from_the_unit_is_not_answered() -> None:
    statement = _statement(UNIT_A, "这句话根本不在原文里")
    assert build_claims_qa([statement], _texts()) == {}


def test_a_statement_with_no_unit_text_is_not_answered() -> None:
    statement = _statement(UNIT_A, QUOTE_A)
    assert build_claims_qa([statement], {}) == {}


def test_a_role_missing_from_the_quote_is_not_answered() -> None:
    """The quote must actually be about the entities the assertion claims."""
    statement = _statement(UNIT_A, "集市上人很多", TEXT_A.index("集市"))
    assert build_claims_qa([statement], _texts()) == {}


def test_an_unanswered_statement_gets_no_entry_at_all() -> None:
    """No entry means the existing unanswerable fallback applies; an empty entry would not."""
    statement = _statement(UNIT_A, "无关的句子")
    qa = build_claims_qa([statement], _texts())
    assert statement["assertion_id"] not in qa


def test_unanswerable_questions_never_carry_answers() -> None:
    """`qa.py:183` rejects answers on an unanswerable stance; the producer must not trip it."""
    statement = _statement(UNIT_A, QUOTE_A, TEXT_A.index(QUOTE_A))
    for questions in build_claims_qa([statement], _texts()).values():
        for question in questions:
            if question["stance"] == "unanswerable":
                assert not question.get("answers")


# ══ end to end: a grounded claim can be promoted ═══════════════════════
def _run_chain(work: Path, unit_texts: dict[str, str]) -> tuple[list[dict], dict]:
    units = [
        {
            "unit_id": unit_id,
            "book_id": BOOK,
            "volume": 1,
            "chapter": index,
            "order": index,
            "narrator_reliability": "reliable",
            "text": text,
        }
        for index, (unit_id, text) in enumerate(sorted(unit_texts.items()), 1)
    ]
    outputs = [
        {
            "unit_id": unit_id,
            "entities": [
                {"canonical_name": "言万", "type": "person", "aliases": [], "tier": "canonical", "confidence": 0.9},
                {"canonical_name": "拉法", "type": "person", "aliases": [], "tier": "major", "confidence": 0.8},
            ],
            "assertions": [
                _statement(unit_id, quote, text.index(quote))
            ],
        }
        for unit_id, text, quote in (
            (UNIT_A, TEXT_A, QUOTE_A),
            (UNIT_B, TEXT_B, QUOTE_B),
        )
    ]
    kwargs = dict(book_id=BOOK, outputs=outputs, units=units, created_at=CREATED)
    resolve_run(work, upstream={}, **kwargs)
    assemble_run(work, upstream={}, **kwargs)
    result = claims_run(work, upstream={}, **kwargs)
    path = work / "assemble" / "claims.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows, result


def test_the_pipeline_uses_the_local_producer(tmp_path: Path) -> None:
    """Without an upstream `claims_qa`, the stage must answer what it can."""
    rows, _result = _run_chain(tmp_path / "work", _texts())
    assert rows, "no claims produced"
    supported = [row for row in rows if row.get("verdict") == "supported"]
    assert supported, (
        "no claim reached `supported`; every claim still reads insufficient_evidence, so the "
        f"four-value rule is still only using one value. verdicts={[r.get('verdict') for r in rows]}"
    )


def test_a_supported_claim_with_two_units_is_promoted(tmp_path: Path) -> None:
    _rows, result = _run_chain(tmp_path / "work", _texts())
    assert result["counts"]["promotions_allowed"] >= 1, (
        f"nothing was promoted; counts={result['counts']}"
    )


def test_without_unit_texts_nothing_changes(tmp_path: Path) -> None:
    """The producer must be a no-op when it cannot check anything."""
    rows, result = _run_chain(tmp_path / "work", {})
    assert rows, "no claims produced"
    assert all(row.get("verdict") == "insufficient_evidence" for row in rows)
    assert result["counts"]["promotions_allowed"] == 0
