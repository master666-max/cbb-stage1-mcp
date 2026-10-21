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
        start = text.find("贝塔心叶")
        if start < 0:
            start = 0
        quote = text[start : start + 4]
        ev = Evidence(unit_id=unit_id, start=start, end=start + len(quote), quote=quote)
        return EngineOutput(
            unit_id=unit_id,
            entities=[
                Entity(book_id="t", type="person", canonical_name="贝塔心叶", tier="canonical"),
                Entity(book_id="t", type="person", canonical_name="黑之魔王", tier="major"),
            ],
            assertions=[
                Assertion(
                    book_id="t",
                    predicate="亲属",
                    library="关系",
                    roles={"施事": "贝塔心叶", "受事": "黑之魔王"},
                    evidence=[ev],
                    confidence=0.9,
                    confidence_method="evidence_span",
                    status="asserted",
                ),
                Assertion(
                    book_id="t",
                    predicate="性质",
                    library="设定",
                    roles={"主体": "贝塔心叶", "值": {"literal": "她来过"}},
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
    book.write_text("# 第1章 起\n\n贝塔心叶走近黑之魔王。\n", encoding="utf-8")
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
    assert claims and all(c["verdict"] == "insufficient_evidence" for c in claims)
    coverage = json.loads((tmp_path / "box" / "qc" / "coverage.json").read_text(encoding="utf-8"))
    assert len(coverage["cells"]) == 60
