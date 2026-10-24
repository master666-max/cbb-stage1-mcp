"""Abductive RAG's dual check must reach the record, not stop at the function boundary.

Measured defect
---------------
`derive/integration.py:155-177` calls ``reasoning.abduction.abduce`` — which computes
``consistency``, ``plausibility`` and ``confidence`` from the model's dual check — and then reads
only ``hypothesis.missing_premises`` and ``hypothesis.claim``. The three scores are dropped on the
floor, and the record is built with ``derive.hypotheses.abduce``'s defaults:

    consistency="unknown", plausibility="unknown", confidence=0.0

So every ``derived/hypotheses.jsonl`` row in every box says "the dual check found nothing", no
matter what the model actually judged. Abductive RAG's whole point is that dual check (A18's paper:
consistency = does the hypothesis contradict the text, plausibility = is it a reasonable inference);
dropping it leaves an abduction step that is expensive and unfalsifiable.

`derive.hypotheses.abduce` already accepts all three — the caller simply never passed them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from cbb_stage1.derive.integration import _conflict_hypotheses

CONFLICT_ID = "c:aaaaaaaaaaaaaaaa"


class DualCheckLLM:
    """Implements ``AbductionLLM``: propose premises, then score them."""

    def __init__(self, *, consistency: float = 0.9, plausibility: float = 0.8) -> None:
        self.consistency = consistency
        self.plausibility = plausibility
        self.calls: list[str] = []

    def missing_premises(self, claim: str, evidence: Sequence[str]) -> Sequence[str]:
        self.calls.append("missing_premises")
        return ["冲突双方中一方缺少后文佐证"]

    def score(self, claim: str, premises: Sequence[str]) -> Mapping[str, Any]:
        self.calls.append("score")
        return {
            "consistency": self.consistency,
            "plausibility": self.plausibility,
            "confidence": min(self.consistency, self.plausibility),
        }


def _seed_conflict(work: Path) -> None:
    (work / "audit").mkdir(parents=True, exist_ok=True)
    row = {
        "conflict_id": CONFLICT_ID,
        "kind": "attribute",
        "left": "a:1111111111111111",
        "right": "a:2222222222222222",
        "detail": "言万 的眼睛颜色在两章不一致",
        "status": "open",
        "decision": None,
    }
    (work / "audit" / "conflicts.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _rows(work: Path, llm: Any) -> list[dict]:
    rows, _warnings = _conflict_hypotheses(
        work, "tst", "2026-10-23T03:00:00+08:00", llm
    )
    return rows


# ══ the dual check must survive the call boundary ══════════════════════
def test_the_dual_check_is_actually_asked(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed_conflict(work)
    llm = DualCheckLLM()
    _rows(work, llm)
    assert "score" in llm.calls, (
        "the abduction model was never asked to score the hypothesis; only missing_premises ran"
    )


def test_consistency_reaches_the_record(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed_conflict(work)
    rows = _rows(work, DualCheckLLM(consistency=0.9, plausibility=0.8))
    assert rows, "no hypothesis produced"
    assert rows[0]["consistency"] != "unknown", (
        f"consistency is {rows[0]['consistency']!r}; the model's dual check was discarded"
    )


def test_plausibility_reaches_the_record(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed_conflict(work)
    rows = _rows(work, DualCheckLLM(consistency=0.9, plausibility=0.8))
    assert rows[0]["plausibility"] != "unknown", (
        f"plausibility is {rows[0]['plausibility']!r}"
    )


def test_confidence_reaches_the_record(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed_conflict(work)
    rows = _rows(work, DualCheckLLM(consistency=0.9, plausibility=0.8))
    assert float(rows[0]["confidence"]) > 0.0, (
        f"confidence is {rows[0]['confidence']}; a scored hypothesis must not read 0.0"
    )


# ══ the values must be the contract's labels ═══════════════════════════
def test_the_scores_land_as_contract_labels(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed_conflict(work)
    rows = _rows(work, DualCheckLLM(consistency=0.9, plausibility=0.8))
    assert rows[0]["consistency"] in ("pass", "fail", "unknown")
    assert rows[0]["plausibility"] in ("high", "medium", "low", "unknown")


def test_a_failing_dual_check_is_recorded_as_fail(tmp_path: Path) -> None:
    """Low consistency must not be silently upgraded."""
    work = tmp_path / "work"
    _seed_conflict(work)
    rows = _rows(work, DualCheckLLM(consistency=0.1, plausibility=0.2))
    assert rows[0]["consistency"] == "fail", (
        f"consistency is {rows[0]['consistency']!r}; a low score must read as fail"
    )


# ══ without a model the row must stay honestly unknown ═════════════════
def test_without_a_model_the_scores_stay_unknown(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _seed_conflict(work)
    rows = _rows(work, None)
    assert rows, "a conflict without an llm should still produce a candidate hypothesis"
    assert rows[0]["consistency"] == "unknown"
    assert float(rows[0]["confidence"]) == 0.0 or rows[0]["confidence"] == 0
