"""Wired-pipeline integration test (P0-A).

Two layers:

1. **Pipeline wiring (always runs):** inject fake integration stage modules and
   assert that ``run_pipeline`` calls postprocess after extraction / before
   WRITING_OUTPUT, keeps the 9-stage progress contract, records the sub-step
   chain in ``work/plan/wiring.json``, and aborts loudly when a stage fails.
2. **Real integration chain (runs once the P0-B/C/D packages land):**
   synthetic ``EngineOutput`` (entities + relation/statement assertions + event
   + cue) through the real ``run_pipeline``; asserts every integration stage is
   ``ok`` and its work table exists. Skipped with an explicit reason while the
   other packages' ``integration.py`` are still landing.

Run::

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_wired_pipeline.py -q
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from cbb_stage1.config import Config
from cbb_stage1.engines.base import Engine
from cbb_stage1.engines.mock import MockEngine
from cbb_stage1.models import Assertion, Cue, EngineOutput, Entity, Event, Evidence, Mention
from cbb_stage1.orchestrator.pipeline import run_pipeline
from cbb_stage1.orchestrator.postprocess import (
    DEFAULT_STAGES,
    PostprocessError,
    StageSpec,
    load_wiring,
)
from cbb_stage1.orchestrator.progress import (
    STAGE_ORDER,
    add_progress_listener,
    remove_progress_listener,
)

EXPECTED_ORDER = [spec.name for spec in DEFAULT_STAGES]


def _missing_integrations() -> list[str]:
    missing: list[str] = []
    for spec in DEFAULT_STAGES:
        try:
            importlib.import_module(spec.module)
        except ImportError:
            missing.append(spec.module)
    return missing


MISSING_INTEGRATIONS = _missing_integrations()


# =============================================================================
# Fakes
# =============================================================================

class WiredFixtureEngine(Engine):
    """Synthetic EngineOutput: two entities, relation + statement assertion, event, cue."""

    def extract(self, unit_id, text, card, known_entities):
        start = text.find("贝塔心叶")
        if start < 0:
            start = 0
        quote = text[start : start + 4]
        evidence = Evidence(
            unit_id=unit_id,
            start=start,
            end=start + len(quote),
            quote=quote,
            occurrence=1,
            mode="narration",
        )
        return EngineOutput(
            unit_id=unit_id,
            mentions=[
                Mention(
                    unit_id=unit_id,
                    surface=quote,
                    start=start,
                    end=start + len(quote),
                )
            ],
            entities=[
                Entity(
                    book_id="t",
                    type="person",
                    canonical_name="贝塔心叶",
                    aliases=[],
                    tier="canonical",
                ),
                Entity(
                    book_id="t",
                    type="person",
                    canonical_name="黑之魔王",
                    tier="major",
                ),
            ],
            assertions=[
                Assertion(
                    book_id="t",
                    predicate="亲属",
                    library="关系",
                    roles={"施事": "贝塔心叶", "受事": "黑之魔王"},
                    evidence=[evidence],
                    confidence=0.9,
                    confidence_method="evidence_span",
                    status="asserted",
                ),
                Assertion(
                    book_id="t",
                    predicate="性质",
                    library="设定",
                    roles={"主体": "贝塔心叶", "值": {"literal": "看见影子"}},
                    claim_layer="statement",
                    speaker={"kind": "narrator", "name": "叙述者"},
                    evidence=[evidence],
                    confidence=0.7,
                    confidence_method="evidence_span",
                    status="asserted",
                ),
            ],
            events=[
                Event(
                    book_id="t",
                    type="相遇",
                    summary="贝塔心叶走近黑之魔王",
                    participants=["贝塔心叶", "黑之魔王"],
                    unit_id=unit_id,
                    evidence=[evidence],
                )
            ],
            cues=[
                Cue(
                    book_id="t",
                    kind="object",
                    subject={"literal": "苍蓝色的影子"},
                    setup={"unit_id": unit_id, "quote": quote},
                    expected="能力来源",
                    status="candidate",
                )
            ],
        )


def _write_book(tmp_path: Path) -> Path:
    book = tmp_path / "book.txt"
    book.write_text(
        "# 第1章 起\n\n贝塔心叶走近黑之魔王。她的影子是苍蓝色的。\n\n"
        "# 第2章 承\n\n贝塔心叶说：「我看见了。」黑之魔王没有回答。\n",
        encoding="utf-8",
    )
    return book


def _make_config(tmp_path: Path) -> Config:
    return Config(
        book_id="t",
        book_path=_write_book(tmp_path),
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        model="mock",
        provider="mock",
    )


def _fake_stage_module(name: str, calls: list[dict], *, exc: BaseException | None = None,
                       counts: dict | None = None) -> types.ModuleType:
    module = types.ModuleType(name)

    def run(work_root, *, book_id, outputs, units, created_at, upstream):
        calls.append(
            {
                "module": name,
                "outputs": outputs,
                "units": units,
                "upstream": dict(upstream),
            }
        )
        if exc is not None:
            raise exc
        return {"counts": counts or {f"{name}_rows": 1}, "paths": []}

    module.run = run  # type: ignore[attr-defined]
    return module


def _register_fakes(monkeypatch, prefix: str, calls: list[dict]) -> list[StageSpec]:
    stages: list[StageSpec] = []
    for name in EXPECTED_ORDER:
        module_name = f"{prefix}_{name}"
        monkeypatch.setitem(
            sys.modules, module_name, _fake_stage_module(module_name, calls)
        )
        stages.append(StageSpec(name, module_name))
    return stages


def _first_seen_stages(events: list) -> list:
    first_seen = []
    for event in events:
        if not first_seen or first_seen[-1] is not event.stage:
            first_seen.append(event.stage)
    return first_seen


# =============================================================================
# Layer 1: pipeline wiring (no dependency on package integrations)
# =============================================================================

def test_pipeline_calls_postprocess_as_substep_and_records_wiring(tmp_path, monkeypatch):
    calls: list[dict] = []
    stages = _register_fakes(monkeypatch, "w1wire", calls)

    events: list = []
    add_progress_listener(events.append)
    try:
        result = run_pipeline(
            _make_config(tmp_path),
            chapters="all",
            engine=MockEngine(),
            postprocess_stages=stages,
        )
    finally:
        remove_progress_listener(events.append)

    assert result.status == "complete"
    assert result.postprocess is not None
    assert result.postprocess["ok"] is True
    assert result.postprocess["stages_ok"] == 8

    # integrations ran in the fixed order and received the engine payloads/units
    assert [call["module"] for call in calls] == [f"w1wire_{name}" for name in EXPECTED_ORDER]
    assert calls[0]["outputs"] and "unit_id" in calls[0]["outputs"][0]
    assert calls[0]["units"] and "narrator_reliability" in calls[0]["units"][0]

    # 9-stage progress contract unchanged; postprocess appears as sub-steps
    assert _first_seen_stages(events) == list(STAGE_ORDER)
    sub_events = [event for event in events if event.sub_total == 8]
    assert [event.sub_current for event in sub_events] == list(range(1, 9))
    assert all(event.message and "postprocess" in event.message for event in sub_events)

    # wiring artifact records the call chain
    wiring = load_wiring(tmp_path / "work")
    assert wiring is not None
    assert wiring["contract"] == "cbb-wiring/1"
    assert wiring["ok"] is True
    assert wiring["order"] == EXPECTED_ORDER
    assert [stage["name"] for stage in wiring["stages"]] == EXPECTED_ORDER
    assert wiring["stages_ok"] == 8

    run_summary = json.loads((tmp_path / "work" / "plan" / "run.json").read_text(encoding="utf-8"))
    assert run_summary["status"] == "complete"
    assert run_summary["postprocess"]["ok"] is True
    assert run_summary["stages"] == [stage.name for stage in STAGE_ORDER]
    state = json.loads((tmp_path / "work" / "plan" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["postprocess"]["ok"] is True


def test_postprocess_failure_aborts_pipeline_loudly(tmp_path, monkeypatch):
    calls: list[dict] = []
    ok_module = _fake_stage_module("w1fail_ok", calls)
    boom_module = _fake_stage_module(
        "w1fail_boom", calls, exc=RuntimeError("assemble exploded")
    )
    never_module = _fake_stage_module("w1fail_never", calls)
    for name, module in (
        ("w1fail_ok", ok_module),
        ("w1fail_boom", boom_module),
        ("w1fail_never", never_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    stages = [
        StageSpec("resolve", "w1fail_ok"),
        StageSpec("assemble", "w1fail_boom"),
        StageSpec("claims", "w1fail_never"),
    ]
    with pytest.raises(PostprocessError, match="assemble exploded") as info:
        run_pipeline(
            _make_config(tmp_path),
            chapters="all",
            engine=MockEngine(),
            postprocess_stages=stages,
        )
    assert info.value.stage == "assemble"

    # failure is not silent: run/state record failed + wiring records the stage
    run_summary = json.loads((tmp_path / "work" / "plan" / "run.json").read_text(encoding="utf-8"))
    assert run_summary["status"] == "failed"
    assert run_summary["postprocess"]["ok"] is False
    assert run_summary["postprocess"]["stage"] == "assemble"
    state = json.loads((tmp_path / "work" / "plan" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["postprocess"]["ok"] is False
    wiring = load_wiring(tmp_path / "work")
    assert wiring is not None
    assert wiring["ok"] is False
    assert wiring["stages"][-1]["name"] == "assemble"
    assert wiring["stages"][-1]["status"] == "error"
    # no stage after the failing one ran
    assert [call["module"] for call in calls] == ["w1fail_ok", "w1fail_boom"]


# =============================================================================
# Layer 2: real integration chain (P0-B/C/D landing required)
# =============================================================================

@pytest.mark.skipif(
    bool(MISSING_INTEGRATIONS),
    reason=f"waiting for package integrations to land: {MISSING_INTEGRATIONS}",
)
def test_wired_pipeline_with_real_integrations(tmp_path):
    cfg = _make_config(tmp_path)
    result = run_pipeline(cfg, chapters="all", engine=WiredFixtureEngine())

    assert result.status == "complete", result.to_dict()
    assert result.units_total == 2
    assert result.postprocess is not None
    assert result.postprocess["ok"] is True
    assert result.postprocess["stages_ok"] == 8

    wiring = load_wiring(tmp_path / "work")
    assert wiring is not None
    assert wiring["ok"] is True
    assert [stage["name"] for stage in wiring["stages"]] == EXPECTED_ORDER
    assert all(stage["status"] == "ok" for stage in wiring["stages"]), wiring

    work = tmp_path / "work"
    # canonical work tables mandated by P0-B/C/D task descriptions
    for relative in (
        "resolve/merges.jsonl",
        "assemble/assertions.jsonl",
        "assemble/claims.jsonl",
        "assemble/cues.jsonl",
        "assemble/timeline.jsonl",
        "audit/conflicts.jsonl",
    ):
        assert (work / relative).exists(), f"missing wired work table: {relative}"
    assert list((work / "derive").glob("*.jsonl")), "derive integration wrote no table"

    # every path a stage reports exists under the work root (or is absolute)
    for stage in wiring["stages"]:
        for path in stage["paths"]:
            candidate = Path(path)
            assert candidate.is_absolute() or (work / candidate).exists(), (stage["name"], path)
