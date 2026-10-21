"""Unit tests for ``orchestrator.postprocess`` (P0-A wiring layer).

The eight package ``integration.py`` modules are exercised through injected fake
modules here, so this file pins the contract (order, upstream threading,
wiring.json, loud failures) without depending on the other P0 tasks landing.

Run::

    .venv\\Scripts\\python.exe -m pytest tests/unit/test_postprocess.py -q
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from cbb_stage1.orchestrator.postprocess import (
    DEFAULT_STAGES,
    StageSpec,
    PostprocessError,
    load_wiring,
    run_postprocess,
)

EXPECTED_ORDER = [
    "resolve",
    "assemble",
    "claims",
    "cues",
    "temporal",
    "validate",
    "dedupe",
    "derive",
]


def _fake_module(
    name: str,
    calls: list[dict],
    *,
    result: dict | None = None,
    exc: BaseException | None = None,
    with_run: bool = True,
) -> types.ModuleType:
    """Create a fake integration module whose run(...) records its inputs."""
    module = types.ModuleType(name)

    if with_run:

        def run(work_root, *, book_id, outputs, units, created_at, upstream):
            calls.append(
                {
                    "module": name,
                    "work_root": Path(work_root),
                    "book_id": book_id,
                    "outputs": outputs,
                    "units": units,
                    "created_at": created_at,
                    "upstream": dict(upstream),
                }
            )
            if exc is not None:
                raise exc
            return result if result is not None else {"counts": {}, "paths": []}

        module.run = run  # type: ignore[attr-defined]

    return module


def _register(monkeypatch, prefix: str, calls: list[dict], *, count: int = 8) -> None:
    for name in EXPECTED_ORDER[:count]:
        module_name = f"{prefix}_{name}"
        result = {
            "counts": {f"{name}_rows": 1},
            "paths": [f"{name}/table.jsonl"],
        }
        monkeypatch.setitem(
            sys.modules, module_name, _fake_module(module_name, calls, result=result)
        )


def _stages(prefix: str, names: list[str] | None = None) -> list[StageSpec]:
    return [StageSpec(name, f"{prefix}_{name}") for name in (names or EXPECTED_ORDER)]


# =============================================================================
# Contract: fixed order / modules
# =============================================================================

def test_default_stage_order_and_modules():
    assert [spec.name for spec in DEFAULT_STAGES] == EXPECTED_ORDER
    assert [spec.module for spec in DEFAULT_STAGES] == [
        "cbb_stage1.resolve.integration",
        "cbb_stage1.assemble.integration",
        "cbb_stage1.claims.integration",
        "cbb_stage1.cues.integration",
        "cbb_stage1.temporal.integration",
        "cbb_stage1.validate.integration",
        "cbb_stage1.dedupe.integration",
        "cbb_stage1.derive.integration",
    ]


# =============================================================================
# Happy path: order, upstream, wiring
# =============================================================================

def test_runs_stages_in_order_and_threads_upstream(tmp_path, monkeypatch):
    calls: list[dict] = []
    _register(monkeypatch, "w1ok", calls)

    result = run_postprocess(
        tmp_path,
        book_id="book1",
        outputs=[{"unit_id": "u:book1:v01:c001"}],
        units=[{"unit_id": "u:book1:v01:c001"}],
        created_at="2026-10-18T00:00:00",
        stages=_stages("w1ok"),
    )

    assert result["ok"] is True
    assert [call["module"] for call in calls] == [f"w1ok_{name}" for name in EXPECTED_ORDER]
    # upstream accumulates completed stages; first stage sees an empty mapping.
    assert calls[0]["upstream"] == {}
    assert set(calls[1]["upstream"]) == {"resolve"}
    assert set(calls[-1]["upstream"]) == set(EXPECTED_ORDER[:-1])
    # inputs are passed through to every integration
    assert all(call["book_id"] == "book1" for call in calls)
    assert all(call["created_at"] == "2026-10-18T00:00:00" for call in calls)
    assert all(call["outputs"][0]["unit_id"] == "u:book1:v01:c001" for call in calls)

    wiring = load_wiring(tmp_path)
    assert wiring is not None
    assert wiring["contract"] == "cbb-wiring/1"
    assert wiring["ok"] is True
    assert wiring["order"] == EXPECTED_ORDER
    assert wiring["stages_total"] == 8
    assert wiring["stages_ok"] == 8
    assert [stage["name"] for stage in wiring["stages"]] == EXPECTED_ORDER
    assert all(stage["status"] == "ok" for stage in wiring["stages"])
    # counts aggregate; paths are flattened and de-duplicated
    assert wiring["counts"]["resolve_rows"] == 1
    assert wiring["counts"]["derive_rows"] == 1
    assert wiring["paths"] == [f"{name}/table.jsonl" for name in EXPECTED_ORDER]


def test_progress_callback_runs_once_per_stage_in_order(tmp_path, monkeypatch):
    calls: list[dict] = []
    _register(monkeypatch, "w1cb", calls)
    seen: list[tuple[int, int, str]] = []

    run_postprocess(
        tmp_path,
        book_id="b",
        outputs=[],
        units=[],
        stages=_stages("w1cb"),
        progress_cb=lambda index, total, name: seen.append((index, total, name)),
    )
    assert seen == [(index, 8, name) for index, name in enumerate(EXPECTED_ORDER, 1)]


def test_upstream_seed_is_passed_to_first_stage(tmp_path, monkeypatch):
    calls: list[dict] = []
    _register(monkeypatch, "w1seed", calls, count=1)
    run_postprocess(
        tmp_path,
        book_id="b",
        outputs=[],
        units=[],
        upstream={"engine": {"unit_count": 2}},
        stages=_stages("w1seed", ["resolve"]),
    )
    assert calls[0]["upstream"] == {"engine": {"unit_count": 2}}


def test_counts_only_keep_numeric_values(tmp_path, monkeypatch):
    module = types.ModuleType("w1mixed")
    module.run = lambda *args, **kwargs: {
        "counts": {"rows": 3, "flag": True, "name": "x", "ratio": 0.5},
        "paths": {"table": "assemble/assertions.jsonl"},
    }
    monkeypatch.setitem(sys.modules, "w1mixed", module)

    result = run_postprocess(
        tmp_path,
        book_id="b",
        outputs=[],
        units=[],
        stages=[StageSpec("assemble", "w1mixed")],
    )
    assert result["counts"] == {"rows": 3, "ratio": 0.5}
    assert result["paths"] == ["assemble/assertions.jsonl"]


# =============================================================================
# Failures are loud and leave an auditable wiring artifact
# =============================================================================

def test_missing_integration_module_fails_loudly(tmp_path):
    with pytest.raises(PostprocessError, match="unavailable") as info:
        run_postprocess(
            tmp_path,
            book_id="b",
            outputs=[],
            units=[],
            stages=[StageSpec("resolve", "w1_no_such_integration_module")],
        )
    assert info.value.stage == "resolve"
    assert info.value.wiring_path is not None
    wiring = load_wiring(tmp_path)
    assert wiring is not None
    assert wiring["ok"] is False
    assert wiring["stages"][0]["status"] == "unavailable"
    assert "unavailable" in wiring["error"]


def test_missing_run_callable_fails_loudly(tmp_path, monkeypatch):
    module = types.ModuleType("w1norun")
    monkeypatch.setitem(sys.modules, "w1norun", module)
    with pytest.raises(PostprocessError, match="no callable run") as info:
        run_postprocess(
            tmp_path,
            book_id="b",
            outputs=[],
            units=[],
            stages=[StageSpec("resolve", "w1norun")],
        )
    assert info.value.stage == "resolve"
    wiring = load_wiring(tmp_path)
    assert wiring is not None and wiring["ok"] is False
    assert wiring["stages"][0]["status"] == "error"


def test_stage_exception_writes_partial_wiring_and_stops_chain(tmp_path, monkeypatch):
    calls: list[dict] = []
    ok_module = _fake_module("w1partial_ok", calls, result={"counts": {"a": 1}})
    boom_module = _fake_module(
        "w1partial_boom", calls, exc=RuntimeError("integration exploded")
    )
    monkeypatch.setitem(sys.modules, "w1partial_ok", ok_module)
    monkeypatch.setitem(sys.modules, "w1partial_boom", boom_module)
    never_module = _fake_module("w1partial_never", calls)
    monkeypatch.setitem(sys.modules, "w1partial_never", never_module)

    with pytest.raises(PostprocessError, match="integration exploded") as info:
        run_postprocess(
            tmp_path,
            book_id="b",
            outputs=[],
            units=[],
            stages=[
                StageSpec("resolve", "w1partial_ok"),
                StageSpec("assemble", "w1partial_boom"),
                StageSpec("claims", "w1partial_never"),
            ],
        )
    assert info.value.stage == "assemble"
    assert isinstance(info.value.cause, RuntimeError)
    # chain stopped after the failing stage
    assert [call["module"] for call in calls] == ["w1partial_ok", "w1partial_boom"]

    wiring = load_wiring(tmp_path)
    assert wiring is not None
    assert wiring["ok"] is False
    assert wiring["stages_ok"] == 1
    assert [stage["name"] for stage in wiring["stages"]] == ["resolve", "assemble"]
    assert wiring["stages"][0]["status"] == "ok"
    assert wiring["stages"][1]["status"] == "error"
    assert "integration exploded" in wiring["stages"][1]["error"]


def test_non_dict_result_fails_loudly(tmp_path, monkeypatch):
    module = types.ModuleType("w1list")
    module.run = lambda *args, **kwargs: ["not", "a", "dict"]
    monkeypatch.setitem(sys.modules, "w1list", module)
    with pytest.raises(PostprocessError, match="expected dict"):
        run_postprocess(
            tmp_path,
            book_id="b",
            outputs=[],
            units=[],
            stages=[StageSpec("resolve", "w1list")],
        )
    wiring = load_wiring(tmp_path)
    assert wiring is not None and wiring["ok"] is False


def test_empty_stage_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one stage"):
        run_postprocess(
            tmp_path, book_id="b", outputs=[], units=[], stages=[]
        )


def test_load_wiring_returns_none_before_any_run(tmp_path):
    assert load_wiring(tmp_path) is None
    assert not (tmp_path / "plan" / "wiring.json").exists()


def test_wiring_json_is_utf8_and_contract_stable(tmp_path, monkeypatch):
    calls: list[dict] = []
    _register(monkeypatch, "w1utf", calls, count=1)
    run_postprocess(
        tmp_path,
        book_id="书名",
        outputs=[],
        units=[],
        stages=_stages("w1utf", ["resolve"]),
    )
    raw = (tmp_path / "plan" / "wiring.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["contract"] == "cbb-wiring/1"
    assert payload["book_id"] == "书名"
    assert "resolve" in payload["order"]
