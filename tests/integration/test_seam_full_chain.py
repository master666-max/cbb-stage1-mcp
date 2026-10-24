"""Full-chain seam test: normalize -> resolve -> assemble.

This is the mechanical gate the audit asked for (B-P2-1). It exists because the
same class of defect — the emitting end and the receiving end disagreeing about
predicate names, role names, or value shapes — has now been found six times in a
row, and each time it was caught by a live run instead of by CI.

Why the earlier contract test was not enough
--------------------------------------------
``test_predicate_mapping.py`` only checks that emitted *predicate names* are
registered. It never feeds the assertions through the Assembler, so it stayed
green while the box received nothing. Measured on the same fixture:
9 assertions emitted, 0 into the box. A test that does not exercise the seam
cannot protect the seam.

What this test asserts
----------------------
* every assertion whose predicate maps onto a registered L1 predicate reaches
  the box (accepted == mapped count);
* nothing is quarantined for a reason other than the one *intended* path
  (an unmapped record becoming a proposal, per I4);
* the per-predicate reconciliation is exact: emitted == accepted for every
  mapped predicate, so a partial loss cannot hide behind a green total.

It runs the real order (resolve before assemble) because ``map_surface`` needs
the entity table: a probe that skips resolve reports every surface as
unresolved, which is a property of the probe, not of the pipeline.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cbb_stage1.assemble.integration import run as assemble_run
from cbb_stage1.engines.ai_reader.schema import (
    AbilityGained,
    ChapterFact,
    CharacterFact,
    ConceptFact,
    EventFact,
    ItemEventFact,
    LocationFact,
    OrgEventFact,
    RelationshipFact,
    SpatialRelationship,
    WorldDeclaration,
)
from cbb_stage1.normalize.facts import chapter_fact_to_engine_output
from cbb_stage1.resolve.integration import run as resolve_run

TEXT = (
    "言万心叶与拉斐尔是朋友。言万心叶穿着黑色夹克。"
    "神殿在太平洋之上。低语者是一种能力。黑手党把言万心叶逐出。"
)
UNIT = "u:seam:1"


def _fact() -> ChapterFact:
    """One fact touching every record class that produces an assertion."""
    return ChapterFact(
        chapter_id=1,
        novel_id="seam",
        characters=[
            CharacterFact(
                name="言万心叶",
                appearance="穿着黑色夹克",
                abilities_gained=[
                    AbilityGained(
                        dimension="技能",
                        name="低语者",
                        description="读心",
                        evidence="低语者是一种能力。",
                    )
                ],
                evidence="言万心叶穿着黑色夹克。",
            ),
            CharacterFact(name="拉斐尔", evidence="言万心叶与拉斐尔是朋友。"),
        ],
        relationships=[
            RelationshipFact(
                person_a="言万心叶",
                person_b="拉斐尔",
                relation_type="师徒",
                evidence="言万心叶与拉斐尔是朋友。",
            )
        ],
        locations=[
            LocationFact(
                name="太平洋", type="海域", parent="异世界", evidence="神殿在太平洋之上。"
            ),
            LocationFact(name="神殿", type="建筑", evidence="神殿在太平洋之上。"),
        ],
        spatial_relationships=[
            SpatialRelationship(
                source="神殿",
                target="太平洋",
                relation_type="direction",
                value="north_of",
                narrative_evidence="神殿在太平洋之上。",
            )
        ],
        new_concepts=[
            ConceptFact(
                name="低语者", category="能力", definition="读心", evidence="低语者是一种能力。"
            )
        ],
        world_declarations=[
            WorldDeclaration(
                declaration_type="portal",
                content={"name": "门"},
                narrative_evidence="神殿在太平洋之上。",
            )
        ],
        item_events=[
            ItemEventFact(
                item_name="糖",
                item_type="食物",
                action="赠予",
                actor="言万心叶",
                evidence="言万心叶穿着黑色夹克。",
            )
        ],
        org_events=[
            OrgEventFact(
                org_name="黑手党",
                org_type="组织",
                member="言万心叶",
                action="逐出",
                evidence="黑手党把言万心叶逐出。",
            )
        ],
        events=[
            EventFact(
                summary="两人交谈",
                type="社交",
                participants=["言万心叶"],
                evidence="言万心叶与拉斐尔是朋友。",
            )
        ],
    )


def _run_chain(tmp_path: Path):
    output = chapter_fact_to_engine_output(
        _fact(), unit_id=UNIT, text=TEXT, book_id="seam"
    )
    (tmp_path / "plan" / "texts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "plan" / "texts" / "u1.txt").write_text(TEXT, encoding="utf-8")
    units = [
        {
            "unit_id": UNIT,
            "order": 1,
            "volume": 1,
            "chapter": 1,
            "text_path": "plan/texts/u1.txt",
        }
    ]
    resolve_run(
        tmp_path,
        book_id="seam",
        outputs=[output.model_dump()],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={},
    )
    result = assemble_run(
        tmp_path,
        book_id="seam",
        outputs=[output.model_dump()],
        units=units,
        created_at="2026-01-01T00:00:00",
        upstream={},
    )
    accepted_path = tmp_path / "assemble" / "assertions.jsonl"
    accepted = (
        [
            json.loads(line)
            for line in accepted_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if accepted_path.exists()
        else []
    )
    return output, result, accepted


def test_every_mapped_assertion_reaches_the_box(tmp_path: Path) -> None:
    """The seam that was open: emitted N, box got 1.

    If this fails, do not "fix" it by registering a new predicate — the spec
    (00-架构总设计) forbids deriving the registry from data. Fix whichever end
    disagrees: predicate name, role name, or value shape.
    """
    output, result, accepted = _run_chain(tmp_path)

    unmapped = {a.predicate for a in output.assertions if a.qualifiers.get("unmapped")}
    mapped_count = len(output.assertions) - len(
        [a for a in output.assertions if a.qualifiers.get("unmapped")]
    )

    assert accepted, "the box received no assertions at all"
    assert len(accepted) == mapped_count, (
        f"emitted {len(output.assertions)} assertions "
        f"({mapped_count} mapped, {len(unmapped)} unmapped), "
        f"box accepted {len(accepted)}; "
        f"quarantined={result['counts'].get('quarantined')}"
    )


def test_per_predicate_reconciliation_is_exact(tmp_path: Path) -> None:
    """A partial loss must not hide behind a green total.

    The audit's B-P2-3: a box with 1 of 9 assertions would pass the table-level
    coverage invariant. Comparing counts per predicate catches the shape of loss
    that a single total hides.
    """
    output, _result, accepted = _run_chain(tmp_path)
    emitted = Counter(
        a.predicate for a in output.assertions if not a.qualifiers.get("unmapped")
    )
    into_box = Counter(row["predicate"] for row in accepted)
    assert into_box == emitted, (
        "per-predicate loss:\n"
        + "\n".join(
            f"  {predicate}: emitted {emitted.get(predicate, 0)} "
            f"accepted {into_box.get(predicate, 0)}"
            for predicate in sorted(set(emitted) | set(into_box))
            if emitted.get(predicate, 0) != into_box.get(predicate, 0)
        )
    )


def test_only_intended_quarantine_paths_are_used(tmp_path: Path) -> None:
    """Quarantine must be explainable by the spec, not by a contract mismatch.

    ``schema`` for an unregistered predicate is the designed I4 path. Anything
    else (``role_mismatch``, ``evidence_not_found``, ``claim_layer_conflict``)
    means the two ends still disagree.
    """
    _output, result, _accepted = _run_chain(tmp_path)
    reasons = Counter(item.get("reason") for item in result.get("quarantine") or [])
    unexpected = {
        reason: count
        for reason, count in reasons.items()
        if reason not in {"schema"}
    }
    assert not unexpected, f"unintended quarantine reasons: {unexpected}"


def test_unmapped_record_becomes_a_proposal_not_an_assertion(tmp_path: Path) -> None:
    """I4: no honest mapping -> proposed_predicates, never an invented predicate."""
    _output, result, accepted = _run_chain(tmp_path)
    assert result["counts"].get("proposals", 0) >= 1
    proposal_path = tmp_path / "assemble" / "proposed_predicates.jsonl"
    assert proposal_path.exists()
    predicates = {
        json.loads(line)["predicate"]
        for line in proposal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert predicates
    assert not (predicates & {row["predicate"] for row in accepted})


def test_chain_is_deterministic(tmp_path: Path) -> None:
    """Same input, same box rows — the seam fix must not introduce ordering drift."""
    first = Path(tmp_path) / "a"
    second = Path(tmp_path) / "b"
    first.mkdir()
    second.mkdir()
    _o1, _r1, accepted1 = _run_chain(first)
    _o2, _r2, accepted2 = _run_chain(second)
    assert [row["assertion_id"] for row in accepted1] == [
        row["assertion_id"] for row in accepted2
    ]
    assert accepted1 == accepted2
