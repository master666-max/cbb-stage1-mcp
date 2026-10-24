"""The declaration says "no promotion gate"; the code has one. They are talking about different gates.

Measured conflict (E5)
----------------------
`00-架构总设计.md:276` and `03-质量与实施.md:100` both declare:

    `judge-panel/plant-gate/promote-vote` | 降为 QC 抽样/校准，不做晋升门

while `claims/verdicts.py` implements `promotion_decision` (the AVeriTeC §5.19 rule-4 fact
promotion gate) and `claims/integration.py` applies it, with the promoted row reaching the box.

Both statements are true of different things, and the declaration does not say which:

* the **vote trio** — judge panel, plant gate, promote-vote — is genuinely absent;
* the **fact promotion gate** — a `statement` becomes a `fact` only when supported, with ≥2
  independent sources and no `against` evidence — is implemented and wired.

A reader who checks the declaration concludes there is no promotion logic at all, and stops looking.
That is the failure mode: not a missing feature, a false statement about a present one.

`plant-gate` (植株: plant a known-wrong claim and check it is caught) really is absent, and this test
does not claim otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import cbb_stage1

#: Derived from the package so the test is not bound to this machine's layout.
DESIGN = Path(cbb_stage1.__file__).resolve().parents[3] / "设计" / "抽取段-v6"
DOCS = ("00-架构总设计.md", "03-质量与实施.md")

pytestmark = pytest.mark.skipif(
    not all((DESIGN / name).is_file() for name in DOCS),
    reason=f"design documents not present under {DESIGN}",
)
DECLARED_ROW = "judge-panel/plant-gate/promote-vote"


def _row(name: str) -> str:
    """The declaration table row, not every line that happens to mention the same words."""
    for line in (DESIGN / name).read_text(encoding="utf-8").splitlines():
        if DECLARED_ROW in line and line.lstrip().startswith("|"):
            return line
    raise AssertionError(f"{name} no longer has a table row for {DECLARED_ROW}")


# ══ the code really does have the gate ════════════════════════════════
def test_the_fact_promotion_gate_exists() -> None:
    from cbb_stage1.claims.verdicts import promotion_decision

    assert callable(promotion_decision)


def test_the_gate_is_applied_by_the_claims_stage() -> None:
    import inspect

    from cbb_stage1.claims import integration

    source = inspect.getsource(integration)
    assert "promotion_decision(" in source, "the claims stage does not call the promotion gate"


# ══ the declaration must distinguish the two gates ════════════════════
def test_the_declaration_does_not_say_there_is_no_promotion_gate() -> None:
    for name in DOCS:
        row = _row(name)
        assert "不做晋升门" not in row, (
            f"{name} still says '不做晋升门' without qualification, while promotion_decision exists "
            f"and is applied. row={row!r}"
        )


def test_the_declaration_names_what_is_actually_absent() -> None:
    """The vote trio and the plant gate are gone; the row must say so specifically."""
    for name in DOCS:
        row = _row(name)
        assert "票数" in row or "植株" in row or "面板" in row, (
            f"{name} does not name the absent mechanisms: {row!r}"
        )


def test_the_declaration_records_that_the_fact_gate_is_implemented() -> None:
    for name in DOCS:
        row = _row(name)
        assert "事实晋升" in row or "promotion_decision" in row, (
            f"{name} does not say the fact promotion gate exists: {row!r}"
        )


def test_the_row_still_marks_the_vote_gate_as_not_in_the_pipeline() -> None:
    """Correcting the record must not accidentally promise the vote trio."""
    for name in DOCS:
        row = _row(name)
        assert "降为" in row or "不进入" in row or "不做" in row, (
            f"{name} no longer records that the vote gate is out of the production chain: {row!r}"
        )
