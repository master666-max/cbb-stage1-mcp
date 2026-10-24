"""Adjudication: the two decisions the extraction stage had no door for.

The stage already had two human gates — ``cbb_lexicon_approve`` (I4: a predicate is
never grown from data) and ``cbb_supersede`` (D9: retcon). It declared two more
without a writer:

  * ``validate/conflicts.py`` names five choices in ``DECISIONS`` but implements one
    (``superseded``); the other four had no way to be recorded.
  * quarantine items had no adjudication at all — the projection stage has
    ``adjudicatedSeq``, this stage never got one.

Two rules shape the implementation:

  * **nothing is deleted.** A decision is appended to ``audit/decisions.jsonl`` and the
    record is annotated. The old record stays, the same way a superseded assertion
    stays with ``status=superseded``.
  * **the human is named.** Every decision carries ``by``; an empty one is refused
    before anything is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.validate import adjudication

CONFLICT = {
    "conflict_id": "cf:1",
    "book_id": "t",
    "kind": "attribute",
    "left": "a:aaa1",
    "right": "a:bbb2",
    "detail": "两值互斥",
    "status": "open",
}

QUARANTINE = {
    "item_id": "assertion:u:t:v01:c001",
    "kind": "assertion",
    "reason": "ref_missing",
    "detail": "entity name not resolved: 母体",
    "source_run": "run:1",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    _write_jsonl(root / "audit" / "conflicts.jsonl", [dict(CONFLICT)])
    _write_jsonl(root / "assemble" / "quarantine.jsonl", [dict(QUARANTINE)])
    # The two assertions the conflict points at. `superseded` retcons them, so they
    # have to exist — a conflict naming assertions that are not there is refused, and
    # that refusal is the point of the test below.
    _write_jsonl(
        root / "assemble" / "assertions.jsonl",
        [
            {"assertion_id": "a:aaa1", "predicate": "位于", "status": "asserted"},
            {"assertion_id": "a:bbb2", "predicate": "敌对", "status": "asserted"},
        ],
    )
    return root


# ── conflict resolution ──────────────────────────────────────────────────────
def test_a_conflict_resolution_writes_a_decision_row(work: Path) -> None:
    result = adjudication.resolve_conflict(
        work, conflict_id="cf:1", choice="left", by="张三", reason="左边更准"
    )
    assert result["ok"] is True
    decisions = _read_jsonl(work / "audit" / "decisions.jsonl")
    assert len(decisions) == 1
    row = decisions[0]
    assert row["kind"] == "conflict"
    assert row["subject"] == "cf:1"
    assert row["choice"] == "left"
    assert row["decided_by"] == "张三"
    assert row["reason"] == "左边更准"
    assert row["decision_id"].startswith("d:")


def test_the_conflict_itself_is_closed_with_the_choice(work: Path) -> None:
    """闭合用 status=resolved，五选一进 decision。

    这条测试原先断言 ``status == "right"``，那是把缺陷当成了期望：``status`` 的闭集只有
    ``("open", "resolved")``，五选一属于 ``decision``；写进 ``status`` 会让下次 ``build_box``
    抛 ``bad conflict status 'right'``。
    """
    adjudication.resolve_conflict(
        work, conflict_id="cf:1", choice="right", by="张三", reason="r"
    )
    conflicts = _read_jsonl(work / "audit" / "conflicts.jsonl")
    assert len(conflicts) == 1, "冲突记录不许被删"
    assert conflicts[0]["status"] == "resolved", "status 只能取 open/resolved"
    assert conflicts[0]["decision"] == "right", "五选一必须落在 decision"
    assert conflicts[0]["resolved_by"] == "张三"
    assert conflicts[0]["adjudicated_by"] == "张三"


@pytest.mark.parametrize(
    "choice", ["left", "right", "merge", "disputed", "superseded"]
)
def test_every_declared_choice_is_accepted(work: Path, choice: str) -> None:
    """``DECISIONS`` 的五个选项一个都不许是死的。"""
    result = adjudication.resolve_conflict(
        work, conflict_id="cf:1", choice=choice, by="张三", reason="r"
    )
    assert result["ok"] is True
    assert result["choice"] == choice


def test_supersede_whose_assertions_are_missing_is_refused(work: Path) -> None:
    """冲突指向的断言不在表里 ⇒ 拒绝，不许写半个决定。"""
    (work / "assemble" / "assertions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(KeyError):
        adjudication.resolve_conflict(
            work, conflict_id="cf:1", choice="superseded", by="张三", reason="r"
        )
    assert _read_jsonl(work / "audit" / "decisions.jsonl") == []


def test_a_choice_outside_the_declared_set_is_refused(work: Path) -> None:
    with pytest.raises(ValueError):
        adjudication.resolve_conflict(
            work, conflict_id="cf:1", choice="随便", by="张三", reason="r"
        )
    assert _read_jsonl(work / "audit" / "decisions.jsonl") == []


def test_an_unknown_conflict_is_refused_without_writing(work: Path) -> None:
    with pytest.raises(KeyError):
        adjudication.resolve_conflict(
            work, conflict_id="cf:nope", choice="left", by="张三", reason="r"
        )
    assert _read_jsonl(work / "audit" / "decisions.jsonl") == []


def test_a_missing_human_is_refused_before_anything_is_written(work: Path) -> None:
    with pytest.raises(ValueError):
        adjudication.resolve_conflict(
            work, conflict_id="cf:1", choice="left", by="  ", reason="r"
        )
    assert _read_jsonl(work / "audit" / "decisions.jsonl") == []
    conflicts = _read_jsonl(work / "audit" / "conflicts.jsonl")
    assert conflicts[0]["status"] == "open", "被拒的裁决不许改状态"


# ── quarantine adjudication ──────────────────────────────────────────────────
@pytest.mark.parametrize("decision", ["keep", "drop", "retain"])
def test_every_quarantine_decision_is_recorded(work: Path, decision: str) -> None:
    result = adjudication.adjudicate_quarantine(
        work, item_id=QUARANTINE["item_id"], decision=decision,
        by="张三", reason="人工看过",
    )
    assert result["ok"] is True
    decisions = _read_jsonl(work / "audit" / "decisions.jsonl")
    assert len(decisions) == 1
    assert decisions[0]["kind"] == "quarantine"
    assert decisions[0]["subject"] == QUARANTINE["item_id"]
    assert decisions[0]["choice"] == decision
    assert decisions[0]["decided_by"] == "张三"


def test_the_quarantine_item_is_annotated_not_deleted(work: Path) -> None:
    """旧件保留：隔离项还在，只是被标了裁决。"""
    adjudication.adjudicate_quarantine(
        work, item_id=QUARANTINE["item_id"], decision="keep",
        by="张三", reason="r",
    )
    items = _read_jsonl(work / "assemble" / "quarantine.jsonl")
    assert len(items) == 1, "隔离项不许被删"
    assert items[0]["adjudication"]["choice"] == "keep"
    assert items[0]["adjudication"]["decided_by"] == "张三"


def test_an_unknown_quarantine_item_is_refused(work: Path) -> None:
    with pytest.raises(KeyError):
        adjudication.adjudicate_quarantine(
            work, item_id="item:nope", decision="keep", by="张三", reason="r"
        )
    assert _read_jsonl(work / "audit" / "decisions.jsonl") == []


def test_an_unknown_quarantine_decision_is_refused(work: Path) -> None:
    with pytest.raises(ValueError):
        adjudication.adjudicate_quarantine(
            work, item_id=QUARANTINE["item_id"], decision="随便",
            by="张三", reason="r",
        )


def test_the_decision_rows_carry_no_box_material(work: Path) -> None:
    """台账行不许夹带材料（value/roles/quote 等）——那是投射段的硬口径，
    这里同一条纪律：决策账只记裁决，不抄材料。"""
    adjudication.adjudicate_quarantine(
        work, item_id=QUARANTINE["item_id"], decision="keep",
        by="张三", reason="r",
    )
    row = _read_jsonl(work / "audit" / "decisions.jsonl")[0]
    for banned in ("value", "roles", "quote"):
        assert banned not in row, f"决策行不许夹带 {banned}"
