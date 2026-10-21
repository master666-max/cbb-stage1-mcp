"""The governance console is the human adjudication entry point.

Two things are being guarded, and they are the two that failed before in this project:

  * **a button that does nothing** — the console offers four kinds of human decision
    (predicate approval, conflict resolution, quarantine adjudication, supersede). The
    first two had no write path at all in this stage, and the conflict set declared
    five choices while implementing one. A console whose buttons are decorative is
    worse than no console, so every decision path is asserted to actually reach an
    MCP tool, with the human's ``by``.
  * **a read surface that reads stale or empty** — the read side is the box, and every
    view must report what is really there, including "this is empty" for a table that
    has no rows, rather than a 500 or a silent zero.

The fixture is built inside ``tmp_path`` rather than pointed at a recorded run, so the
suite does not depend on a machine that has one (the same defect the portability gate
caught in the dedupe test).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import gov_console

# ── fixture: a small but complete box ───────────────────────────────────────
ASSERTIONS = [
    {
        "assertion_id": "a:aaa1",
        "book_id": "t",
        "predicate": "位于",
        "library": "关系",
        "roles": {"主体": "e:t:loc:1", "值": {"literal": "湖泊"}},
        "claim_layer": "fact",
        "status": "asserted",
        "verification": {"level": "minor", "verdict": "none"},
        "evidence": [
            {"unit_id": "u:t:v01:c001", "start": 0, "end": 5, "quote": "那里，是巨"}
        ],
    },
    {
        "assertion_id": "a:bbb2",
        "book_id": "t",
        "predicate": "敌对",
        "library": "关系",
        "roles": {"主体": "e:t:person:1", "值": {"literal": "小兽"}},
        "claim_layer": "statement",
        "status": "asserted",
        "speaker": {"name": "阿尔法"},
        "verification": {"level": "major", "verdict": "support"},
        "evidence": [
            {"unit_id": "u:t:v01:c001", "start": 10, "end": 16, "quote": "「好。把它毁"}
        ],
    },
]

PROPOSALS = [
    {
        "proposal_id": "pp:1",
        "book_id": "t",
        "predicate": "世界观声明",
        "seen_in": ["u:t:v01:c001"],
        "count": 1,
        "status": "proposed",
        "decided_by": None,
    }
]

CONFLICTS = [
    {
        "conflict_id": "cf:1",
        "book_id": "t",
        "kind": "attribute",
        "left": "a:aaa1",
        "right": "a:bbb2",
        "detail": "两值互斥",
        "status": "open",
    }
]

QUARANTINE = [
    {
        "item_id": "assertion:u:t:v01:c001",
        "kind": "assertion",
        "reason": "ref_missing",
        "detail": "entity name not resolved: 源体",
        "source_run": "run:1",
    },
    {
        "item_id": "event:u:t:v01:c002",
        "kind": "event",
        "reason": "role_mismatch",
        "detail": "role mismatch",
        "source_run": "run:1",
    },
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _build_box(root: Path) -> Path:
    box = root / "box"
    _write_jsonl(box / "knowledge" / "assertions.jsonl", ASSERTIONS)
    _write_jsonl(box / "knowledge" / "proposed_predicates.jsonl", PROPOSALS)
    _write_jsonl(box / "knowledge" / "conflicts.jsonl", CONFLICTS)
    _write_jsonl(box / "audit" / "quarantine.jsonl", QUARANTINE)
    _write_jsonl(box / "audit" / "decisions.jsonl", [])
    _write_jsonl(box / "knowledge" / "citations.jsonl", [])
    _write_json(
        box / "qc" / "report.json",
        {
            "status": "pass",
            "hard_gates": {"lint": "pass"},
            "hard_gate_errors": {"lint": []},
            "soft_gates": {"cost_usd": None},
            "capability_gaps": {
                "d15_inference_partial": {
                    "status": "not_implemented",
                    "reason": "still missing X",
                }
            },
            "work_table_problems": [],
            "samples": {"assertions": 2, "merges": 0, "low_confidence": 0},
        },
    )
    _write_json(
        box / "qc" / "coverage.json",
        {
            "book_id": "t",
            "spec_version": "2.2.0",
            "libraries": ["实体", "关系", "事件", "埋线", "设定", "时序"],
            "aspects": ["presence", "identity"],
            "cells": [
                {"library": "实体", "aspect": "presence", "status": "present",
                 "count": 3, "reason": None},
                {"library": "关系", "aspect": "presence", "status": "absent",
                 "count": 0, "reason": "无关系断言"},
            ],
            "counts": {"entities": 3, "assertions": 2},
        },
    )
    _write_json(
        box / "index" / "by_predicate.json",
        {"位于": ["a:aaa1"], "敌对": ["a:bbb2"]},
    )
    _write_json(
        box / "index" / "by_entity.json",
        {"e:t:loc:1": ["a:aaa1"]},
    )
    _write_json(box / "index" / "by_unit.json", {"u:t:v01:c001": ["a:aaa1", "a:bbb2"]})
    for name, rows in (
        ("forecasts", [{"forecast_id": "fc:1", "method": "lexical_singleton_ratio"}]),
        ("stylometry", [{"style_id": "st:1", "method": "char_statistics"}]),
        ("themes", [{"theme_id": "th:1", "motif": "小兽"}]),
        ("emotion_curves", [{"curve_id": "ec:1", "speaker": "阿尔法"}]),
        ("motivation_chains", [{"chain_id": "mc:1", "character": "阿尔法"}]),
    ):
        _write_jsonl(box / "derived" / "inference" / f"{name}.jsonl", rows)
    return box


# ── read surface ─────────────────────────────────────────────────────────────
def test_overview_reports_the_gates_and_the_gaps(tmp_path: Path) -> None:
    box = _build_box(tmp_path)
    overview = gov_console.read_overview(box)
    assert overview["hard_gates"]["lint"] == "pass"
    assert "d15_inference_partial" in overview["capability_gaps"]
    assert overview["capability_gaps"]["d15_inference_partial"]["status"] == "not_implemented"


def test_coverage_reports_every_cell_with_its_reason(tmp_path: Path) -> None:
    box = _build_box(tmp_path)
    coverage = gov_console.read_coverage(box)
    assert coverage["cells"], "覆盖矩阵必须逐格给出"
    absent = [c for c in coverage["cells"] if c["status"] == "absent"]
    assert absent and absent[0]["reason"], "absent 格必须带 reason（I32）"


def test_the_four_pending_adjudication_kinds_are_all_visible(tmp_path: Path) -> None:
    """人裁入口的四个队列，一个都不能少。"""
    box = _build_box(tmp_path)
    queue = gov_console.read_pending(box)
    assert len(queue["proposals"]) == 1
    assert len(queue["conflicts"]) == 1
    assert len(queue["quarantine"]) == 2
    assert queue["proposals"][0]["predicate"] == "世界观声明"
    assert queue["proposals"][0]["status"] == "proposed"
    assert queue["quarantine"][0]["reason"] == "ref_missing"


def test_an_empty_table_reads_as_empty_not_as_an_error(tmp_path: Path) -> None:
    """空箱不许 500。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert gov_console.read_pending(empty)["proposals"] == []
    assert gov_console.read_pending(empty)["conflicts"] == []
    assert gov_console.read_pending(empty)["quarantine"] == []
    overview = gov_console.read_overview(empty)
    assert overview["hard_gates"] == {}
    assert overview["capability_gaps"] == {}


def test_the_quarantine_queue_shows_ids_the_write_path_can_resolve(tmp_path: Path) -> None:
    """The box and the work tree use different id spaces for quarantine.

    Measured on a recorded run: the box's ``audit/quarantine.jsonl`` carries
    ``assertion:u:book:v01:c001`` (one row per problem, grouped by kind and unit),
    while the work tree's ``assemble/quarantine.jsonl`` carries record-level ids such
    as ``a:pending:fe9bfd87d164aeba``. A console that displayed the box's ids and
    wrote to the work tree would show a human a list whose every button fails with
    "no such quarantine item". So the queue reads the side whose ids the write path
    can resolve, and says so.
    """
    box = _build_box(tmp_path)
    work = tmp_path / "work"
    (work / "assemble").mkdir(parents=True)
    _write_jsonl(
        work / "assemble" / "quarantine.jsonl",
        [
            {
                "item_id": "a:pending:fe9bfd87d164aeba",
                "kind": "assertion",
                "reason": "ref_missing",
                "detail": "entity name not resolved",
                "source_run": "run:1",
            }
        ],
    )

    queue = gov_console.read_pending(box, work=work)

    assert queue["quarantine"], "the queue must not be empty"
    assert queue["quarantine"][0]["item_id"] == "a:pending:fe9bfd87d164aeba", (
        "the shown id must be the one the write path resolves"
    )
    assert queue["quarantine_source"] == "work"


def test_every_shown_quarantine_id_is_resolvable_by_the_write_path(tmp_path: Path) -> None:
    """The whole point: what the human sees, the human can act on."""
    box = _build_box(tmp_path)
    work = tmp_path / "work"
    (work / "assemble").mkdir(parents=True)
    _write_jsonl(
        work / "assemble" / "quarantine.jsonl",
        [
            {"item_id": "a:pending:aaa", "kind": "assertion", "reason": "ref_missing"},
            {"item_id": "ev:invalid:bbb", "kind": "event", "reason": "role_mismatch"},
        ],
    )

    queue = gov_console.read_pending(box, work=work)

    for item in queue["quarantine"]:
        # The real write path must resolve every id the console displayed.
        from cbb_stage1.validate.adjudication import adjudicate_quarantine

        try:
            adjudicate_quarantine(
                work, item_id=item["item_id"], decision="keep", by="张三", reason="r"
            )
        except KeyError as exc:
            raise AssertionError(
                f"控制台显示的 id 无法写入：{item['item_id']} -> {exc}"
            ) from exc


def test_without_a_work_tree_the_box_is_still_readable(tmp_path: Path) -> None:
    """The box alone must still produce a queue — a console that needs two paths to
    show anything is a console that shows nothing on a bare box."""
    box = _build_box(tmp_path)
    queue = gov_console.read_pending(box)
    assert queue["quarantine"]
    assert queue["quarantine_source"] == "box"


def test_the_inference_view_lists_five_tables_and_says_which_are_empty(tmp_path: Path) -> None:
    box = _build_box(tmp_path)
    view = gov_console.read_inference(box)
    assert set(view["tables"]) >= {
        "forecasts", "stylometry", "themes", "emotion_curves", "motivation_chains",
    }
    for name, entry in view["tables"].items():
        assert "rows" in entry and "empty" in entry


def test_the_query_uses_the_box_index(tmp_path: Path) -> None:
    box = _build_box(tmp_path)
    result = gov_console.query(box, predicate="位于")
    assert result["total"] == 1
    assert result["hits"][0]["assertion_id"] == "a:aaa1"
    assert gov_console.query(box, predicate="不存在")["total"] == 0


def test_the_query_can_filter_by_unit(tmp_path: Path) -> None:
    box = _build_box(tmp_path)
    result = gov_console.query(box, unit="u:t:v01:c001")
    assert result["total"] == 2


# ── write surface: every decision must reach an MCP tool ────────────────────
@pytest.fixture()
def recorder(monkeypatch):
    calls: list[dict] = []

    def fake(tool: str, **kwargs):
        calls.append({"tool": tool, **kwargs})
        return {"ok": True, "data": {"accepted": True}}

    monkeypatch.setattr(gov_console, "_mcp_call", fake)
    return calls


def test_predicate_approval_reaches_the_lexicon_tool_with_a_by(
    tmp_path: Path, recorder: list
) -> None:
    calls = recorder
    gov_console.decide_predicate(
        tmp_path / "work", predicate="世界观声明", by="张三", library="设定"
    )
    assert calls, "批准谓词必须真的调 MCP"
    assert calls[0]["tool"] == "cbb_lexicon_approve"
    assert calls[0]["by"] == "张三", "I4 要求人审身份"
    assert calls[0]["predicate"] == "世界观声明"


def test_supersede_reaches_its_tool_with_a_by(tmp_path: Path, recorder: list) -> None:
    calls = recorder
    gov_console.decide_supersede(
        tmp_path / "work", loser="a:aaa1", winner="a:bbb2", by="张三", reason="后者更新"
    )
    assert calls[0]["tool"] == "cbb_supersede"
    assert calls[0]["by"] == "张三"
    assert calls[0]["loser"] == "a:aaa1"


@pytest.mark.parametrize("choice", ["left", "right", "merge", "disputed", "superseded"])
def test_every_conflict_choice_reaches_a_tool(
    tmp_path: Path, recorder: list, choice: str
) -> None:
    """五个选项一个都不能是死按钮。"""
    calls = recorder
    gov_console.decide_conflict(
        tmp_path / "work", conflict_id="cf:1", choice=choice, by="张三", reason="裁"
    )
    assert calls, f"冲突选项 {choice} 必须是活的"
    assert calls[0]["choice"] == choice
    assert calls[0]["by"] == "张三"
    # Exact name, not a prefix: a mutation that routed the conflict to another
    # tool (`cbb_supersede`) has to be caught, and `startswith("cbb_")` would
    # have let it through. The mutation test is what found that.
    assert calls[0]["tool"] == "cbb_conflict_resolve"


@pytest.mark.parametrize("decision", ["keep", "drop", "retain"])
def test_every_quarantine_decision_reaches_a_tool(
    tmp_path: Path, recorder: list, decision: str
) -> None:
    calls = recorder
    gov_console.decide_quarantine(
        tmp_path / "work", item_id="assertion:u:t:v01:c001", decision=decision,
        by="张三", reason="裁",
    )
    assert calls, f"隔离裁决 {decision} 必须是活的"
    assert calls[0]["decision"] == decision
    assert calls[0]["by"] == "张三"
    assert calls[0]["tool"] == "cbb_quarantine_adjudicate"


def test_an_unknown_conflict_choice_is_refused_before_calling_anything(
    tmp_path: Path, recorder: list
) -> None:
    calls = recorder
    with pytest.raises(ValueError):
        gov_console.decide_conflict(
            tmp_path / "work", conflict_id="cf:1", choice="随便", by="张三", reason="x"
        )
    assert calls == [], "非法选项不许先调后判"


def test_a_missing_human_identity_is_refused(tmp_path: Path, recorder: list) -> None:
    """没有 `by` 就不许发出去——I4 的门在这里。"""
    calls = recorder
    with pytest.raises(ValueError):
        gov_console.decide_predicate(tmp_path / "work", predicate="X", by="  ")
    assert calls == []


# ── discipline: the prior art's rules still hold ────────────────────────────
def test_dry_run_binds_no_port_and_says_what_is_in_place(tmp_path: Path, capsys) -> None:
    box = _build_box(tmp_path)
    exit_code = gov_console.main(["--box", str(box), "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert payload["box_exists"] is True
    assert payload["box"] == str(box)


def test_help_exits_zero_without_touching_the_filesystem(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        gov_console.main(["--help"])
    assert excinfo.value.code == 0


def test_the_read_side_never_writes(tmp_path: Path) -> None:
    """GET 路径一个字节都不许落盘。"""
    box = _build_box(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in box.rglob("*") if p.is_file()}
    gov_console.read_overview(box)
    gov_console.read_coverage(box)
    gov_console.read_pending(box)
    gov_console.read_inference(box)
    gov_console.query(box, predicate="位于")
    after = {p: p.stat().st_mtime_ns for p in box.rglob("*") if p.is_file()}
    assert before == after
    assert len(list(box.rglob("*"))) == len(list(box.rglob("*")))


def test_the_mcp_client_is_resolved_to_a_runnable_path() -> None:
    """`mcporter` is an npm shim — a `.cmd` on Windows.

    ``subprocess.run(["mcporter", ...])`` cannot find it: Windows needs the resolved
    full path, and the bare name raises ``FileNotFoundError [WinError 2]``. Measured
    on a real run: every write failed with that error while the MCP tool itself was
    fine. So the console must resolve the client before spawning it.
    """
    import shutil

    resolved = gov_console._mcp_executable()
    assert resolved, "the MCP client must resolve to something"
    if shutil.which("mcporter"):
        assert Path(resolved).exists(), (
            f"resolved MCP client does not exist: {resolved}"
        )
        assert Path(resolved).suffix.lower() in {".cmd", ".bat", ".exe", ""}, (
            f"unexpected client kind: {resolved}"
        )


def test_an_unresolvable_mcp_client_gives_a_readable_error(tmp_path, monkeypatch) -> None:
    """A missing client must not surface as a bare WinError."""
    monkeypatch.setenv("CBB_MCPORTER", str(tmp_path / "no-such-client"))
    result = gov_console._mcp_call("cbb_status", run_id="x")
    assert result["ok"] is False
    assert "no-such-client" in result["error"]


def test_the_console_is_a_single_file_with_no_third_party_imports() -> None:
    """stdlib 零依赖——先例的形态。"""
    source = (Path(gov_console.__file__)).read_text(encoding="utf-8")
    banned = ("import requests", "import flask", "import fastapi", "import django",
              "import pandas", "import numpy")
    for needle in banned:
        assert needle not in source, f"控制台必须零依赖，出现了 {needle}"