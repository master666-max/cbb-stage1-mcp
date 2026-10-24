"""AFEV's atomic facts must be attached to the claim, not just counted.

Measured defect
---------------
`claims/integration.py` runs `afev_verify` (which decomposes the claim and verifies each atomic
fact), then reads **only** `afev.verdict` to increment a degenerate counter. The `atomic_facts`
list is discarded, so `claims.atomic_facts` is `[]` in every box — while `claims/qa.py:339` accepts
an `atomic_facts` argument and `models.Claim` has the field.

The ordering is why: the claim is built and dumped *before* `afev_verify` runs, so the facts do not
exist yet when `build_claim` is called. `row["cove"]` is already patched in place after its own
call, so the same pattern applies.

Why it matters: AFEV is the mechanism that turns "X is Y" into checkable pieces. Keeping the pieces
only in a counter means the box cannot show what was actually verified — the expensive part is
invisible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cbb_stage1.assemble import run as assemble_run
from cbb_stage1.claims import run as claims_run
from cbb_stage1.resolve import run as resolve_run

BOOK = "tst"
CREATED = "2026-10-18T23:00:00+08:00"
UNIT = "u:tst:v01:c001"
SPEAKER = {"kind": "entity", "entity_id": None, "name": "言万"}


class AtomicLLM:
    """Implements AtomicLLM and VeGraphLLM with recognisable, checkable output."""

    def decompose(self, claim: str, context: str | None) -> Sequence[str]:
        return ["言万 与 拉法 敌对", "该敌对关系发生在集市"]

    def verify(self, fact: str, evidence: Sequence[str]) -> Mapping[str, Any]:
        # All-unverifiable aggregates to insufficient_evidence, which is what a statement with no
        # QA produces. A 	rue here would make the row contradict its own verdict, and the
        # consistency gate (correctly) refuses that.
        return {"verdict": "unverifiable", "rationale": "证据不足"}

    def triples(self, claim: str) -> Sequence[Mapping[str, Any]]:
        return [{"subject": "言万", "predicate": "敌对", "object": "拉法"}]

    def verify_subassertion(self, triple: Mapping[str, Any]) -> str:
        return "support"


class Output:
    def __init__(self, assertions: list[dict]) -> None:
        self.unit_id = UNIT
        self.assertions = assertions


def _statement(index: int) -> dict:
    quote = f"言万在第{index}次交锋中与拉法对阵"
    start = index * 100
    return {
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "言万", "受事": "拉法"},
        "claim_layer": "statement",
        "speaker": SPEAKER,
        "evidence": [
            {
                "unit_id": UNIT,
                "start": start,
                "end": start + len(quote),
                "quote": quote,
                "occurrence": 1,
                "mode": "dialogue",
                "speaker": SPEAKER,
            }
        ],
        "confidence": 0.9,
        "confidence_method": "evidence_span",
    }


def _run(tmp_path: Path) -> tuple[list[dict], dict]:
    work = tmp_path / "work"
    kwargs = dict(
        book_id=BOOK,
        outputs=[
            {
                "unit_id": UNIT,
                "entities": [
                    {"canonical_name": "言万", "type": "person", "aliases": [], "tier": "canonical", "confidence": 0.9},
                    {"canonical_name": "拉法", "type": "person", "aliases": [], "tier": "major", "confidence": 0.8},
                ],
                "assertions": [_statement(0), _statement(1)],
            }
        ],
        units=[
            {
                "unit_id": UNIT,
                "book_id": BOOK,
                "volume": 1,
                "chapter": 1,
                "order": 1,
                "narrator_reliability": "reliable",
            }
        ],
        created_at=CREATED,
    )
    resolve_run(work, upstream={}, **kwargs)
    assemble_run(work, upstream={}, **kwargs)
    result = claims_run(
        work, upstream={"reasoning_llm": AtomicLLM(), "max_workers": 1}, **kwargs
    )
    path = work / "assemble" / "claims.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, result


# ══ the facts must reach the claim rows ════════════════════════════════
def test_claims_carry_atomic_facts(tmp_path: Path) -> None:
    rows, _result = _run(tmp_path)
    assert rows, "no claims produced"
    populated = [row for row in rows if row.get("atomic_facts")]
    assert populated, (
        "every claim row has atomic_facts == []; AFEV ran and its decomposition was discarded "
        f"(rows={len(rows)})"
    )


def test_the_facts_are_the_ones_afev_produced(tmp_path: Path) -> None:
    rows, _result = _run(tmp_path)
    texts = [item.get("text") for row in rows for item in (row.get("atomic_facts") or [])]
    assert any("敌对" in str(text) for text in texts), f"unexpected fact texts: {texts[:4]}"


def test_each_fact_carries_its_verdict(tmp_path: Path) -> None:
    rows, _result = _run(tmp_path)
    facts = [item for row in rows for item in (row.get("atomic_facts") or [])]
    assert facts, "no atomic facts"
    assert all(item.get("verdict") for item in facts), (
        f"an atomic fact without a verdict: {facts[:2]}"
    )


def test_the_stage_still_reports_ok(tmp_path: Path) -> None:
    _rows, result = _run(tmp_path)
    assert result["status"] == "ok"
    assert result["counts"]["claims"] == 2
