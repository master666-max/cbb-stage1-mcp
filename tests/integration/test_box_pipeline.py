from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.box.lint import lint_box
from cbb_stage1.box.writer import build_box
from cbb_stage1.config import Config
from cbb_stage1.engines.base import Engine
from cbb_stage1.models import Assertion, EngineOutput, Entity, Evidence
from cbb_stage1.orchestrator.pipeline import run_pipeline


class FixtureEngine(Engine):
    def extract(self, unit_id, text, card, known_entities):
        start = text.find("言万心叶")
        if start < 0:
            start = 0
        quote = text[start : start + 4]
        ev = Evidence(unit_id=unit_id, start=start, end=start + len(quote), quote=quote)
        return EngineOutput(
            unit_id=unit_id,
            entities=[
                Entity(book_id="t", type="person", canonical_name="言万心叶", tier="canonical"),
                Entity(book_id="t", type="person", canonical_name="黑之魔王", tier="major"),
            ],
            assertions=[
                Assertion(
                    book_id="t",
                    predicate="亲属",
                    library="关系",
                    roles={"施事": "言万心叶", "受事": "黑之魔王"},
                    evidence=[ev],
                    confidence=0.9,
                    confidence_method="evidence_span",
                    status="asserted",
                ),
                Assertion(
                    book_id="t",
                    predicate="性质",
                    library="设定",
                    roles={"主体": "言万心叶", "值": {"literal": "她来过"}},
                    claim_layer="statement",
                    speaker={"kind": "narrator", "name": "叙述者"},
                    evidence=[ev],
                    confidence=0.7,
                    confidence_method="evidence_span",
                    status="asserted",
                ),
            ],
        )


def test_pipeline_to_box_with_fixture_engine(tmp_path: Path):
    book = tmp_path / "book.txt"
    book.write_text("# 第1章 起\n\n言万心叶走近黑之魔王。\n", encoding="utf-8")
    cfg = Config(
        book_id="t",
        book_path=book,
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        model="fixture",
        provider="fixture",
    )
    result = run_pipeline(cfg, chapters="all", engine=FixtureEngine())
    assert result.status == "complete"
    build_box(tmp_path / "work", tmp_path / "box")
    assert lint_box(tmp_path / "box") == []
    assertions = (tmp_path / "box" / "knowledge" / "assertions.jsonl").read_text(encoding="utf-8")
    assert "亲属" in assertions
    claims = [
        json.loads(line)
        for line in (tmp_path / "box" / "knowledge" / "claims.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Claims used to be uniformly `insufficient_evidence` only because nothing produced a
    # `claims_qa` entry. Now the local producer grounds what it can, so the contract is what
    # matters: a closed verdict set, and `supported` backed by real answers.
    assert claims, "the box has no claims"
    assert {c["verdict"] for c in claims} <= {
        "supported",
        "refuted",
        "insufficient_evidence",
        "unanswerable",
    }
    for claim in claims:
        if claim["verdict"] == "supported":
            assert claim["questions"], "a supported claim must carry its questions"
            assert any(q.get("answers") for q in claim["questions"]), (
                f"claim {claim['claim_id']} is supported with no answers behind it"
            )
    coverage = json.loads((tmp_path / "box" / "qc" / "coverage.json").read_text(encoding="utf-8"))
    assert len(coverage["cells"]) == 60
