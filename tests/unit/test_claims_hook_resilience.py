"""A claim that cannot be atomised must not abort the whole run.

Measured: spk9 died with

    PostprocessError: postprocess stage 'claims' failed:
    AtomicRuleError: decompose() returned no atomic facts (no fabrication fallback)

`afev_verify` raises when `decompose()` returns an empty list — correct at the module
level, because inventing atomic facts would be fabrication. But the claims stage called
it unguarded, and `claims` sits BEFORE `validate`, so one uncooperative claim killed
everything downstream and left the box empty.

`promotion_decision` in the same loop was already guarded. These tests make the two
hooks match it, and check that the failure is recorded rather than swallowed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.claims import integration as claims_integration


def _work(root: Path) -> None:
    """A work tree with one statement assertion and one claim QA row."""
    unit = "u:t:v01:c001"
    assertion = {
        "assertion_id": "a:stmt0000000001",
        "book_id": "t",
        "unit_id": unit,
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "e:t:person:0001", "受事": "e:t:person:0002"},
        "claim_layer": "statement",
        "speaker": {"kind": "entity", "entity_id": "e:t:person:0001", "name": "阿尔法"},
        "evidence": [
            {
                "unit_id": unit,
                "start": 0,
                "end": 9,
                "quote": "「好。把它毁掉吧。」",
                "occurrence": 1,
                "mode": "dialogue",
            }
        ],
        "confidence": 0.9,
        "confidence_method": "semantic_entropy",
        "status": "asserted",
    }
    assemble = root / "assemble"
    assemble.mkdir(parents=True, exist_ok=True)
    (assemble / "assertions.jsonl").write_text(
        json.dumps(assertion, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plan = root / "plan"
    plan.mkdir(parents=True, exist_ok=True)
    (plan / "units.jsonl").write_text(
        json.dumps(
            {
                "unit_id": unit,
                "book_id": "t",
                "order": 1,
                "volume": 1,
                "title": "第一章",
                "narrative_mode": "first_person",
                "narrator_reliability": "reliable",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    claims_dir = root / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    (claims_dir / "qa.jsonl").write_text(
        json.dumps(
            {
                "statement_assertion_id": "a:stmt0000000001",
                "questions": [
                    {"qid": "q1", "question": "阿尔法要毁掉它吗？", "answer": "yes"}
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _run(root: Path) -> dict:
    return claims_integration.run(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-20T00:00:00+08:00",
        upstream={"reasoning_llm": _ExplodingLLM()},
    )


class _ExplodingLLM:
    """A reasoner whose decompose answers ``[]`` — the measured failure."""

    def decompose(self, claim: str, context: str | None = None):  # noqa: ANN001
        return []

    def verify(self, fact: str, evidence) -> dict:  # noqa: ANN001
        return {"verdict": "unverifiable", "rationale": "no evidence"}

    def verify_subassertion(self, triple: dict) -> str:
        return "unsure"


def test_an_unatomisable_claim_does_not_abort_the_stage(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root)

    result = _run(root)  # must not raise

    assert result["stage"] == "claims"
    assert result["status"] == "ok"


def test_the_failure_is_recorded_not_swallowed(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _work(root)

    result = _run(root)

    counts = result["counts"]
    assert counts["afev_errors"] >= 1, "the raising hook must be counted"
    rows = result["afev_error_rows"]
    assert rows, "the error rows must reach the receipt"
    assert rows[0]["stage"] == "afev"
    assert "AtomicRuleError" in rows[0]["error"]


def test_the_claim_still_lands(tmp_path: Path) -> None:
    """The point of guarding: the claim is still written, with no fabricated verdict."""
    root = tmp_path / "work"
    _work(root)

    _run(root)

    claims_file = root / "assemble" / "claims.jsonl"
    assert claims_file.exists()
    rows = [
        json.loads(line)
        for line in claims_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "the claim must still be produced"
