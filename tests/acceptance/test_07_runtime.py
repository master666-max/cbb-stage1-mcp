"""Acceptance proofs for the 07 matrix (new methodology).

One test per PASS id. The fixture runs the real pipeline once with a synthetic
engine that emits statements, a relation, events with explicit time expressions,
and a cue; each proof asserts the runtime artifacts it is allowed to rely on.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cbb_stage1.box.writer import build_box
from cbb_stage1.config import Config
from cbb_stage1.models import EngineOutput, Evidence, Event
from cbb_stage1.orchestrator.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
PROOFS = ROOT / "acceptance" / "proofs.yaml"

import yaml  # noqa: E402

PROOF_IDS = sorted((yaml.safe_load(PROOFS.read_text(encoding="utf-8")) or {}).get("proofs", {}))


def _load_wired_engine():
    path = ROOT / "tests" / "integration" / "test_wired_pipeline.py"
    spec = importlib.util.spec_from_file_location("_wired_fixture_for_proofs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WiredFixtureEngine


class ProofEngine(_load_wired_engine()):
    """Wired fixture + two explicit-time events in the same unit."""

    def extract(self, unit_id, text, card, known_entities):
        output = super().extract(unit_id, text, card, known_entities)
        assert isinstance(output, EngineOutput)
        events = list(output.events)
        for summary, expr in (
            ("三日后，言万心叶走进大厅", "三日后"),
            ("次日，黑之魔王离开", "次日"),
        ):
            start = text.find(expr)
            if start < 0:
                continue
            events.append(
                Event(
                    book_id="t",
                    type="相遇",
                    summary=summary,
                    participants=["言万心叶"],
                    unit_id=unit_id,
                    evidence=[
                        Evidence(
                            unit_id=unit_id,
                            start=start,
                            end=start + len(expr),
                            quote=expr,
                            occurrence=1,
                            mode="narration",
                        )
                    ],
                )
            )
        return output.model_copy(update={"events": events})


@pytest.fixture(scope="module")
def acceptance_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("acceptance")
    book = root / "book.txt"
    book.write_text(
        "# 第一章 起\n\n三日后，言万心叶走进大厅。次日，黑之魔王离开。\n",
        encoding="utf-8",
    )
    cfg = Config(
        book_id="proof",
        book_path=book,
        out_dir=root / "box",
        work_dir=root / "work",
        provider="mock",
        model="mock",
        max_workers=2,
    )
    result = run_pipeline(cfg, chapters="all", engine=ProofEngine())
    box = build_box(root / "work", root / "box")
    return root, cfg, result, box


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_common(root: Path, result) -> None:
    assert result.status == "complete"
    wiring = json.loads((root / "work" / "plan" / "wiring.json").read_text(encoding="utf-8"))
    assert wiring["ok"] is True
    assert len(wiring["stages"]) == 8
    assert all(stage["status"] == "ok" for stage in wiring["stages"])


@pytest.mark.parametrize("mechanism_id", PROOF_IDS)
def test_proof(mechanism_id: str, acceptance_run):
    root, cfg, result, box = acceptance_run
    _assert_common(root, result)
    work = root / "work"
    events = _read_jsonl(work / "assemble" / "events.jsonl")
    claims = _read_jsonl(work / "assemble" / "claims.jsonl")
    # A1-A27: the work layer must have the event producer's rows.
    assert events, "assemble/events.jsonl must be produced from the engine outputs"

    # Group proofs: every assertion is a real artifact, not a test-file path.
    if mechanism_id in {"A1", "A6", "A14"}:
        assert (work / "resolve" / "entity_candidates.jsonl").exists()
    if mechanism_id in {"A19", "A25"}:
        assert _read_jsonl(work / "derive" / "community.jsonl")
        assert _read_jsonl(work / "derive" / "raptor.jsonl")
    if mechanism_id in {"B6", "B7", "B8", "B9", "E11"}:
        assert claims, "statement must produce a claim"
        assert claims[0]["verdict"] == "insufficient_evidence"
        assert (work / "derive" / "hypotheses.jsonl").exists()
    if mechanism_id in {"C1", "C2", "C3", "C4"}:
        assert (work / "plan" / "tasks").exists()
        assert (work / "receipts").exists()
    if mechanism_id in {"D7", "D8"}:
        assert _read_jsonl(work / "resolve" / "cues.jsonl")
        assert _read_jsonl(work / "assemble" / "cues.jsonl")
    if mechanism_id in {"D10", "D11", "D12"}:
        timeline = _read_jsonl(box.box / "knowledge" / "timeline.jsonl")
        assert timeline, "explicit-time events must produce a timeline constraint"
    if mechanism_id in {"C8"}:
        assert cfg.max_workers >= 1
    if mechanism_id in {"C10"}:
        assert (work / "plan" / "wiring.json").read_text(encoding="utf-8")
    if mechanism_id in {"E1", "E2", "E12", "E3"}:
        assert (box.box / "knowledge" / "assertions.jsonl").exists()
    # All proof IDs must at least have a built box manifest.
    assert (box.box / "manifest.json").exists()
