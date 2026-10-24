"""The wiring probe must report the shortfalls that are real, and only those.

Two problems, one probe:

**It lists a gap that is closed.** `WIRE-04 description synthesizer not injected into dedupe` is
stale: `dedupe/integration.py:403-414` builds its own `MapReduceSynthesizer` from
`upstream["config"].description_synthesis` and `aux_llm`. The probe was looking for a pipeline-side
injection that was never needed. A probe that reports a fixed gap teaches its reader to ignore it.

**It cannot see the receipt gap.** `orchestrator/roles.py:receipt_contract_gaps()` computes the
shortfall between `schemas/exec/receipt.schema.json` (14 required fields) and `Receipt` (5), but
nothing called it — `call_wiring` correctly flagged it as written-but-never-called. This probe is
where an agent looks for exactly that kind of information, so it belongs in the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.orchestrator.roles import Receipt, receipt_contract_gaps
from cbb_stage1.preflight import _probe_wiring


def _probe() -> dict:
    return _probe_wiring()


# ══ the receipt shortfall must be visible ═════════════════════════════
def test_the_probe_reports_the_receipt_shortfall() -> None:
    evidence = _probe().get("evidence") or {}
    assert "receipt_contract_gaps" in evidence, (
        "the doctor does not report the receipt contract shortfall; an agent cannot learn that the "
        "receipt does not satisfy its own schema"
    )
    assert evidence["receipt_contract_gaps"], (
        "the shortfall is reported as empty while Receipt carries 5 of 14 required fields"
    )


def test_the_reported_shortfall_matches_the_computed_one() -> None:
    evidence = _probe().get("evidence") or {}
    assert list(evidence["receipt_contract_gaps"]) == list(receipt_contract_gaps())


def test_the_probe_still_reports_real_gaps() -> None:
    evidence = _probe().get("evidence") or {}
    assert evidence.get("gaps"), "the probe reports no injection gaps at all"


# ══ the stale gap must be gone ════════════════════════════════════════
def test_the_closed_wire_04_gap_is_not_reported() -> None:
    gaps = _probe().get("evidence", {}).get("gaps") or []
    joined = " ".join(gaps)
    assert "WIRE-04" not in joined, (
        f"the probe still reports WIRE-04, but dedupe builds its own synthesizer: {gaps}"
    )


def test_the_probe_still_names_the_gaps_that_are_open() -> None:
    gaps = _probe().get("evidence", {}).get("gaps") or []
    joined = " ".join(gaps)
    assert "WIRE-01" in joined or "WIRE-06" in joined, (
        f"the probe dropped every real gap as well: {gaps}"
    )


# ══ the probe stays a probe ═══════════════════════════════════════════
def test_the_probe_reports_observed_not_ready() -> None:
    """WIRING is never READY while anything is only wired — that was the point of the check."""
    assert _probe().get("status") == "OBSERVED"


def test_the_evidence_is_json_serialisable() -> None:
    json.dumps(_probe(), ensure_ascii=False)


def test_the_receipt_still_carries_five_fields() -> None:
    """The probe must describe today's receipt, not a hoped-for one."""
    assert len(Receipt.__dataclass_fields__) == 5
