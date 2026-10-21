"""Integration tests for the P0-B resolve -> assemble -> claims wiring.

Synthetic engine outputs flow through the three ``integration.run`` stages and
the work tables are asserted on disk:

* ``resolve/{alias_map.json,entity_candidates.jsonl,merges.jsonl}`` -- candidates,
  fanout cleaning, reversible/acyclic merges;
* ``assemble/assertions.jsonl`` -- registry/claim-layer/speaker/roles rules;
* ``assemble/claims.jsonl`` + ``assemble/promoted_facts.jsonl`` -- unanswerable
  claims are ``insufficient_evidence`` and never promote; supported QA promotes
  a *derived* fact row without mutating ``assertions.jsonl``.

The stages are deterministic: a second run with the same inputs produces
byte-identical work tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.assemble import run as assemble_run
from cbb_stage1.claims import PromotionRefused, build_promoted_fact, run as claims_run
from cbb_stage1.claims.qa import make_answer, make_question
from cbb_stage1.models import Assertion, Claim
from cbb_stage1.resolve import run as resolve_run

BOOK = "tst"
CREATED = "2026-10-18T23:00:00+08:00"
UNIT1 = "u:tst:v01:c001"
UNIT2 = "u:tst:v01:c002"


# ---------------------------------------------------------------------------
# synthetic pipeline input
# ---------------------------------------------------------------------------


def _units() -> list[dict]:
    return [
        {
            "unit_id": UNIT1,
            "book_id": BOOK,
            "volume": 1,
            "chapter": 1,
            "order": 1,
            "narrator_reliability": "reliable",
        },
        {
            "unit_id": UNIT2,
            "book_id": BOOK,
            "volume": 1,
            "chapter": 2,
            "order": 2,
            "narrator_reliability": "reliable",
        },
    ]


def _evidence(
    quote: str,
    *,
    unit_id: str = UNIT1,
    mode: str = "narration",
    speaker: dict | None = None,
) -> dict:
    return {
        "unit_id": unit_id,
        "start": 0,
        "end": len(quote),
        "quote": quote,
        "occurrence": 1,
        "mode": mode,
        "speaker": speaker,
    }


def _outputs() -> list[dict]:
    speaker = {"kind": "entity", "entity_id": None, "name": "贝塔心叶"}
    return [
        {
            "unit_id": UNIT1,
            "entities": [
                {
                    "canonical_name": "贝塔心叶",
                    "type": "person",
                    "aliases": ["心叶"],
                    "tier": "canonical",
                    "confidence": 0.9,
                },
                {
                    "canonical_name": "拉法",
                    "type": "person",
                    "aliases": ["拉法尔"],
                    "tier": "major",
                    "confidence": 0.8,
                },
                # generic-named entity + generic alias -> fanout clean drops both
                {
                    "canonical_name": "老人",
                    "type": "person",
                    "aliases": ["大哥"],
                    "confidence": 0.4,
                },
            ],
            "assertions": [
                {
                    "predicate": "敌对",
                    "library": "关系",
                    "roles": {"施事": "贝塔心叶", "受事": "拉法"},
                    "claim_layer": "fact",
                    "evidence": [_evidence("贝塔心叶遇见拉法")],
                    "confidence": 0.9,
                    "confidence_method": "evidence_span",
                },
                # negative: role names do not match the registry
                {
                    "predicate": "敌对",
                    "library": "关系",
                    "roles": {"主体": "贝塔心叶", "对象": "拉法"},
                    "claim_layer": "fact",
                    "evidence": [_evidence("贝塔心叶遇见拉法")],
                    "confidence": 0.8,
                    "confidence_method": "evidence_span",
                },
                # negative: unregistered predicate -> proposal, not an assertion
                {
                    "predicate": "赐予",
                    "library": "关系",
                    "roles": {"施事": "贝塔心叶", "受事": "拉法"},
                    "claim_layer": "fact",
                    "evidence": [_evidence("贝塔心叶遇见拉法")],
                    "confidence": 0.5,
                    "confidence_method": "engine_self",
                },
            ],
        },
        {
            "unit_id": UNIT2,
            "entities": [
                # canonical name equals the alias of another entity -> merge
                {
                    "canonical_name": "心叶",
                    "type": "person",
                    "aliases": [],
                    "tier": "minor",
                    "confidence": 0.7,
                }
            ],
            "assertions": [
                {
                    "predicate": "敌对",
                    "library": "关系",
                    "roles": {"施事": "心叶", "受事": "拉法"},
                    "claim_layer": "statement",
                    "speaker": speaker,
                    "evidence": [
                        _evidence(
                            "「心叶说」",
                            unit_id=UNIT2,
                            mode="dialogue",
                            speaker=speaker,
                        )
                    ],
                    "confidence": 0.7,
                    "confidence_method": "evidence_span",
                },
                # negative: statement without a speaker
                {
                    "predicate": "敌对",
                    "library": "关系",
                    "roles": {"施事": "心叶", "受事": "拉法"},
                    "claim_layer": "statement",
                    "speaker": None,
                    "evidence": [_evidence("「心叶说」", unit_id=UNIT2, mode="dialogue")],
                    "confidence": 0.7,
                    "confidence_method": "evidence_span",
                },
            ],
        },
    ]


def _run_all(work: Path, outputs: list[dict], units: list[dict], *, upstream: dict | None = None):
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream=upstream or {}
    )
    resolved = resolve_run(work, **kwargs)
    assembled = assemble_run(work, **kwargs)
    claimed = claims_run(work, **kwargs)
    return resolved, assembled, claimed


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture()
def wired(tmp_path: Path):
    work = tmp_path / "work"
    outputs = _outputs()
    units = _units()
    resolved, assembled, claimed = _run_all(work, outputs, units)
    return {
        "work": work,
        "outputs": outputs,
        "units": units,
        "resolved": resolved,
        "assembled": assembled,
        "claimed": claimed,
    }


# ---------------------------------------------------------------------------
# resolve stage
# ---------------------------------------------------------------------------


def test_resolve_writes_identity_tables(wired):
    work = wired["work"]
    resolved = wired["resolved"]
    root = work / "resolve"

    assert (root / "alias_map.json").exists()
    assert (root / "entity_candidates.jsonl").exists()
    assert (root / "merges.jsonl").exists()
    assert resolved["stage"] == "resolve"
    assert resolved["counts"]["candidates"] == 3  # 老人 dropped by fanout clean
    assert resolved["paths"]["merges"] == "resolve/merges.jsonl"

    candidates = _read_jsonl(root / "entity_candidates.jsonl")
    names = {row["canonical_name"] for row in candidates}
    assert names == {"贝塔心叶", "拉法", "心叶"}
    assert "老人" not in names
    assert all("大哥" not in row["aliases"] for row in candidates)

    alias_payload = json.loads((root / "alias_map.json").read_text(encoding="utf-8"))
    assert alias_payload["contract"] == "cbb-alias-map/1"
    assert alias_payload["alias_map"]["心叶"] == "贝塔心叶"
    assert alias_payload["alias_map"]["拉法尔"] == "拉法"
    assert alias_payload["fanout"]["dropped_entities"] == ["老人"]

    by_name = {row["canonical_name"]: row for row in candidates}
    assert by_name["贝塔心叶"]["units"] == [UNIT1]
    assert by_name["心叶"]["units"] == [UNIT2]
    assert by_name["贝塔心叶"]["entity_id"].startswith("e:tst:person:")


def test_candidate_entity_ids_match_the_box_identity_formula(wired):
    from cbb_stage1.box.records import make_entity_id

    candidates = _read_jsonl(wired["work"] / "resolve" / "entity_candidates.jsonl")
    for candidate in candidates:
        assert candidate["entity_id"] == make_entity_id(
            BOOK, candidate["type"], candidate["canonical_name"]
        )


def test_resolve_merges_are_reversible_and_acyclic(wired):
    from cbb_stage1.resolve import MergeLedger

    root = wired["work"] / "resolve"
    candidates = _read_jsonl(root / "entity_candidates.jsonl")
    candidate_ids = {row["entity_id"] for row in candidates}
    merges = _read_jsonl(root / "merges.jsonl")

    assert len(merges) == 1
    row = merges[0]
    assert row["reversible"] is True
    assert row["undo"] == {"from": row["from"], "to": row["to"]}
    assert set(row["from"]) <= candidate_ids
    assert row["to"] in candidate_ids
    assert row["from"] != [row["to"]]

    ledger = MergeLedger.from_jsonl(root / "merges.jsonl", book_id=BOOK)
    for source in row["from"]:
        assert ledger.resolve(source) == row["to"]
    # no cycle: resolving every candidate terminates at a candidate id
    for entity_id in candidate_ids:
        assert ledger.resolve(entity_id) in candidate_ids

    by_name = {candidate["canonical_name"]: candidate for candidate in candidates}
    assert row["to"] == by_name["贝塔心叶"]["entity_id"]
    assert row["from"] == [by_name["心叶"]["entity_id"]]


# ---------------------------------------------------------------------------
# assemble stage
# ---------------------------------------------------------------------------


def test_assemble_writes_validated_assertions(wired):
    work = wired["work"]
    assembled = wired["assembled"]
    rows = _read_jsonl(work / "assemble" / "assertions.jsonl")

    assert assembled["stage"] == "assemble"
    assert assembled["paths"]["assertions"] == "assemble/assertions.jsonl"
    assert len(rows) == 2
    assert {row["claim_layer"] for row in rows} == {"fact", "statement"}
    assert all(row["library"] == "关系" for row in rows)
    assert all(row["assertion_id"].startswith("a:") for row in rows)
    assert assembled["registry"]["source"] == "builtin:SEED_LEXICON"
    assert assembled["registry"]["digest"].startswith("sha256:")

    candidates = _read_jsonl(work / "resolve" / "entity_candidates.jsonl")
    candidate_ids = {row["entity_id"] for row in candidates}
    by_name = {row["canonical_name"]: row["entity_id"] for row in candidates}

    fact = next(row for row in rows if row["claim_layer"] == "fact")
    assert fact["roles"] == {"施事": by_name["贝塔心叶"], "受事": by_name["拉法"]}

    statement = next(row for row in rows if row["claim_layer"] == "statement")
    # name -> entity id mapping + merge redirect landed on the surviving entity
    assert statement["roles"]["施事"] == by_name["贝塔心叶"]
    assert statement["speaker"]["entity_id"] == by_name["贝塔心叶"]
    assert statement["evidence"][0]["mode"] == "dialogue"
    assert statement["evidence"][0]["speaker"]["entity_id"] == by_name["贝塔心叶"]
    for row in rows:
        for value in row["roles"].values():
            assert value in candidate_ids


def test_assemble_enforces_role_and_predicate_rules(wired):
    assembled = wired["assembled"]
    reasons = [item["reason"] for item in assembled["quarantine"]]
    details = " | ".join(item["detail"] for item in assembled["quarantine"])

    assert "role_mismatch" in reasons
    assert "claim_layer_conflict" in reasons
    assert "schema" in reasons  # unregistered predicate -> proposed
    assert "speaker" in details
    assert [proposal["predicate"] for proposal in assembled["proposals"]] == ["赐予"]
    assert "赐予" not in {
        row["predicate"]
        for row in _read_jsonl(wired["work"] / "assemble" / "assertions.jsonl")
    }


def test_assemble_without_resolve_tables_quarantines_name_roles(tmp_path):
    work = tmp_path / "work"
    assembled = assemble_run(
        work,
        book_id=BOOK,
        outputs=_outputs(),
        units=_units(),
        created_at=CREATED,
        upstream={},
    )
    assert assembled["resolution"]["source"] == "none"
    assert any("no resolve tables" in note for note in assembled["notes"])
    assert assembled["counts"]["accepted"] == 0
    assert "role_mismatch" in [item["reason"] for item in assembled["quarantine"]]


def test_assemble_redirects_merged_entity_ids(tmp_path):
    work = tmp_path / "work"
    outputs = _outputs()
    units = _units()
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    resolve_run(work, **kwargs)
    candidates = {
        row["canonical_name"]: row["entity_id"]
        for row in _read_jsonl(work / "resolve" / "entity_candidates.jsonl")
    }
    # feed the statement with the merged-away id explicitly; assembler must
    # redirect it to the surviving entity
    outputs[1]["assertions"][0]["roles"]["施事"] = candidates["心叶"]
    assembled = assemble_run(work, **kwargs)
    rows = _read_jsonl(work / "assemble" / "assertions.jsonl")
    statement = next(row for row in rows if row["claim_layer"] == "statement")
    assert statement["roles"]["施事"] == candidates["贝塔心叶"]
    assert assembled["resolution"]["merge_redirects"] == 1


def test_assertion_ids_use_the_box_content_key(wired):
    """Work assertion ids must equal the ids the box builder will compute.

    The box key is ``sha16(book|predicate|json(roles)|evidence[0].quote_sha256)``
    (``box/records.py``); a mismatch would make every claim point at a missing
    statement and the box claims table would silently stay empty.
    """
    import hashlib

    rows = _read_jsonl(wired["work"] / "assemble" / "assertions.jsonl")
    assert rows
    for row in rows:
        quote_sha = row["evidence"][0]["quote_sha256"]
        key = "|".join(
            [
                BOOK,
                row["predicate"],
                json.dumps(row["roles"], ensure_ascii=False, sort_keys=True),
                quote_sha,
            ]
        )
        expected = "a:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        assert row["assertion_id"] == expected
        assert quote_sha.startswith("sha256:") and len(quote_sha) == len("sha256:") + 64


def test_resolve_mention_fallback_for_mock_outputs(tmp_path):
    """Units with no extracted entities get explicit low-confidence candidates.

    The mock/spine engines only emit mentions; the resolve stage derives
    ``candidate_source=mention`` candidates (type ``unknown``) and records a
    warning note -- never a silent fallback.
    """
    work = tmp_path / "work"
    units = [
        {"unit_id": UNIT1, "book_id": BOOK, "order": 1},
        {"unit_id": UNIT2, "book_id": BOOK, "order": 2},
    ]
    outputs = [
        {
            "unit_id": UNIT1,
            "entities": [],
            "mentions": [
                {"unit_id": UNIT1, "surface": "我", "start": 0, "end": 1},
                {"unit_id": UNIT1, "surface": "你", "start": 3, "end": 4,
                 "mention_type": "pronoun"},
            ],
        },
        {
            "unit_id": UNIT2,
            "entities": [],
            "mentions": [{"unit_id": UNIT2, "surface": "烟", "start": 0, "end": 1}],
        },
    ]
    resolved = resolve_run(
        work,
        book_id=BOOK,
        outputs=outputs,
        units=units,
        created_at=CREATED,
        upstream={},
    )
    candidates = _read_jsonl(work / "resolve" / "entity_candidates.jsonl")
    assert {(row["canonical_name"], row["candidate_source"]) for row in candidates} == {
        ("我", "mention"),
        ("烟", "mention"),
    }
    assert all(row["type"] == "unknown" and row["tier"] == "minor" for row in candidates)
    assert resolved["counts"]["mention_candidates"] == 2
    assert any("candidate_source=mention" in note for note in resolved["notes"])
    # pronoun mentions are not turned into entities
    assert "你" not in {row["canonical_name"] for row in candidates}


# ---------------------------------------------------------------------------
# claims stage
# ---------------------------------------------------------------------------


def test_claims_insufficient_never_promotes(wired):
    work = wired["work"]
    claimed = wired["claimed"]
    rows = _read_jsonl(work / "assemble" / "claims.jsonl")
    promoted = _read_jsonl(work / "assemble" / "promoted_facts.jsonl")

    assert claimed["stage"] == "claims"
    assert claimed["paths"]["claims"] == "assemble/claims.jsonl"
    assert len(rows) == 1
    claim_row = rows[0]
    assert claim_row["verdict"] == "insufficient_evidence"
    assert claim_row["status"] == "open"
    assert claim_row["questions"][0]["stance"] == "unanswerable"
    assert claim_row["questions"][0]["answers"] == []
    assert claim_row["created_at"] == CREATED
    assert claim_row["statement_assertion_id"].startswith("a:")

    statements = [
        row
        for row in _read_jsonl(work / "assemble" / "assertions.jsonl")
        if row["claim_layer"] == "statement"
    ]
    assert claim_row["statement_assertion_id"] == statements[0]["assertion_id"]

    assert promoted == []
    assert claimed["counts"]["promoted_facts"] == 0
    assert claimed["promotions"][0]["allowed"] is False
    assert claimed["promotions"][0]["reason"] == "insufficient_evidence_never_promotes"

    claim = Claim.model_validate(claim_row)
    statement = Assertion.model_validate(statements[0])
    with pytest.raises(PromotionRefused):
        build_promoted_fact(claim, statement)


def test_claims_never_mutate_the_assertions_table(wired):
    work = wired["work"]
    before = (work / "assemble" / "assertions.jsonl").read_bytes()
    claims_run(
        work,
        book_id=BOOK,
        outputs=wired["outputs"],
        units=wired["units"],
        created_at=CREATED,
        upstream={},
    )
    after = (work / "assemble" / "assertions.jsonl").read_bytes()
    assert before == after


def test_supported_qa_promotes_a_derived_fact_only(tmp_path):
    work = tmp_path / "work"
    outputs = _outputs()
    units = _units()
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    resolve_run(work, **kwargs)
    assemble_run(work, **kwargs)
    statement_row = next(
        row
        for row in _read_jsonl(work / "assemble" / "assertions.jsonl")
        if row["claim_layer"] == "statement"
    )
    statement_id = statement_row["assertion_id"]
    questions = [
        make_question(
            "q1",
            "贝塔心叶与谁敌对？",
            answers=[make_answer("拉法", UNIT1, 0, 2, "拉法")],
            stance="support",
        ).model_dump(),
        make_question(
            "q2",
            "贝塔心叶与谁敌对？",
            answers=[make_answer("拉法", UNIT2, 0, 2, "拉法")],
            stance="support",
        ).model_dump(),
    ]
    claimed = claims_run(
        work,
        book_id=BOOK,
        outputs=outputs,
        units=units,
        created_at=CREATED,
        upstream={"claims_qa": {statement_id: questions}},
    )
    claims = _read_jsonl(work / "assemble" / "claims.jsonl")
    promoted = _read_jsonl(work / "assemble" / "promoted_facts.jsonl")

    assert claims[0]["verdict"] == "supported"
    assert claimed["counts"]["promotions_allowed"] == 1
    assert len(promoted) == 1
    fact = promoted[0]
    assert fact["claim_layer"] == "fact"
    assert fact["derived_from"] == [statement_id]
    assert fact["status"] == "asserted"
    assert fact["claim_layer"] != "statement"
    # the statement table still contains exactly the assembled rows
    assert len(_read_jsonl(work / "assemble" / "assertions.jsonl")) == 2


def test_claims_without_assertions_writes_empty_tables(tmp_path):
    work = tmp_path / "work"
    claimed = claims_run(
        work,
        book_id=BOOK,
        outputs=[],
        units=[],
        created_at=CREATED,
        upstream={},
    )
    assert claimed["counts"]["statements"] == 0
    assert claimed["counts"]["claims"] == 0
    assert (work / "assemble" / "claims.jsonl").exists()
    assert (work / "assemble" / "claims.jsonl").read_text(encoding="utf-8") == ""
    assert any("run assemble first" in note for note in claimed["notes"])


def test_invalid_qa_answer_is_refused_not_fabricated(tmp_path):
    work = tmp_path / "work"
    outputs = _outputs()
    units = _units()
    kwargs = dict(
        book_id=BOOK, outputs=outputs, units=units, created_at=CREATED, upstream={}
    )
    resolve_run(work, **kwargs)
    assemble_run(work, **kwargs)
    statement_id = next(
        row["assertion_id"]
        for row in _read_jsonl(work / "assemble" / "assertions.jsonl")
        if row["claim_layer"] == "statement"
    )
    bad_question = {
        "qid": "q1",
        "question": "贝塔心叶与谁敌对？",
        "stance": "support",
        "answers": [
            {
                "text": "拉法",
                "unit_id": UNIT1,
                "start": 0,
                "end": 2,
                "quote": "拉法",
                "quote_sha256": "sha256:" + "0" * 64,
                "source": "narration",
            }
        ],
    }
    claimed = claims_run(
        work,
        book_id=BOOK,
        outputs=outputs,
        units=units,
        created_at=CREATED,
        upstream={"claims_qa": {statement_id: [bad_question]}},
    )
    assert claimed["counts"]["claims"] == 0
    assert claimed["refused"][0]["reason"] == "bad_answer"
    assert (work / "assemble" / "claims.jsonl").read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_three_stages_are_byte_idempotent(wired):
    work = wired["work"]
    paths = [
        work / "resolve" / "alias_map.json",
        work / "resolve" / "entity_candidates.jsonl",
        work / "resolve" / "merges.jsonl",
        work / "assemble" / "assertions.jsonl",
        work / "assemble" / "claims.jsonl",
        work / "assemble" / "promoted_facts.jsonl",
    ]
    before = {path: path.read_bytes() for path in paths}
    _run_all(work, wired["outputs"], wired["units"])
    after = {path: path.read_bytes() for path in paths}
    assert before == after
