"""Unit tests for the P4 box writer (work/ -> immutable v2.2 box).

Covers:

* the real mock pipeline path (2 chapters) -> box that passes the canonical
  ``tools/box_lint.py`` gate;
* a richer synthetic work layer exercising entities/mentions/assertions/events/
  cues plus the optional claims/timeline/hypotheses/merges tables;
* span repair + quarantine policy, predicate proposals, unit text recompute;
* atomic/no-overwrite behaviour, CHECKSUMS/manifest self-consistency,
  reproducibility with a fixed timestamp, and incremental manifest fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.box import atomic
from cbb_stage1.box.writer import BuildError, build_box, canonical_lint_errors
from cbb_stage1.models import (
    Assertion,
    Cue,
    EngineOutput,
    Entity,
    Event,
    Evidence,
    Mention,
    Speaker,
)

FIXED = "2026-10-18T20:00:00+08:00"
BOOK = "testbook"
U1 = "u:testbook:v01:c001"
U2 = "u:testbook:v01:c002"
T1 = "我的名字叫言万心叶。他人的思想不由分说地在脑中响起。黑之魔王站在船头。"
T2 = "研究所位于地下 267 层。言万心叶看着窗外的雨。"


# ---------------------------------------------------------------------------
# work-layer fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unit(uid: str, text_rel: str, text: str, order: int, chapter: int) -> dict:
    return {
        "unit_id": uid,
        "book_id": BOOK,
        "volume": 1,
        "chapter": chapter,
        "title": f"第{chapter}章 示例",
        "order": order,
        "prev": None,
        "next": None,
        "text_path": text_rel,
        "chars": len(text),
        "sha256": atomic.sha256_text(text),
        "narrative_mode": "present",
        "narrator": {"kind": "third_person", "entity_id": None, "name": "叙述者"},
        "narrator_reliability": "reliable",
        "source_locator": {"href": f"chapter{chapter}.xhtml", "nav_index": chapter - 1},
        "scenes": [],
        "created_at": FIXED,
    }


def _evidence(unit_id: str, text: str, quote: str) -> Evidence:
    start = text.index(quote)
    return Evidence(
        unit_id=unit_id,
        start=start,
        end=start + len(quote),
        quote=quote,
        occurrence=1,
        mode="narration",
    )


def _mention(unit_id: str, text: str, surface: str, *, alias: bool = False) -> Mention:
    start = text.index(surface)
    return Mention(
        unit_id=unit_id,
        surface=surface,
        start=start,
        end=start + len(surface),
        occurrence=1,
        mention_type="alias" if alias else "name",
        confidence=0.9,
    )


def make_rich_work(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    (work / "plan" / "texts" / "v01").mkdir(parents=True)
    (work / "plan" / "texts" / "v01" / "u001.txt").write_text(T1, encoding="utf-8", newline="\n")
    (work / "plan" / "texts" / "v01" / "u002.txt").write_text(T2, encoding="utf-8", newline="\n")
    _write_jsonl(
        work / "plan" / "units.jsonl",
        [
            _unit(U1, "plan/texts/v01/u001.txt", T1, 1, 1),
            _unit(U2, "plan/texts/v01/u002.txt", T2, 2, 2),
        ],
    )
    _write_json(
        work / "plan" / "run.json",
        {
            "book_id": BOOK,
            "book_title": "异世界示例",
            "book_format": "text",
            "trace_id": "rich0001",
            "engine": "MockEngine:mock",
            "status": "complete",
            "started_at": "2026-10-18T20:00:00",
            "ended_at": "2026-10-18T20:00:01",
            "language": "zh-CN",
        },
    )

    statement_id = "a:" + "a" * 16
    event_1 = "ev:" + "b" * 16
    event_2 = "ev:" + "c" * 16
    cue_1 = "cue:" + "d" * 16

    e_person = Entity(
        book_id=BOOK,
        type="person",
        canonical_name="言万心叶",
        aliases=["心叶"],
        tier="canonical",
        confidence=0.95,
    )
    e_black = Entity(
        book_id=BOOK,
        type="person",
        canonical_name="黑之魔王",
        aliases=[],
        tier="major",
        confidence=0.9,
    )
    e_lab = Entity(
        book_id=BOOK,
        type="location",
        canonical_name="研究所",
        aliases=[],
        tier="major",
        confidence=0.85,
    )

    output_1 = EngineOutput(
        unit_id=U1,
        mentions=[
            _mention(U1, T1, "言万心叶"),
            _mention(U1, T1, "心叶", alias=True),
            _mention(U1, T1, "黑之魔王"),
            # bad span => quarantined (surface not found)
            Mention(
                unit_id=U1,
                surface="不存在的人",
                start=0,
                end=5,
                occurrence=1,
                mention_type="name",
            ),
        ],
        entities=[e_person, e_black],
        assertions=[
            Assertion(
                book_id=BOOK,
                predicate="性质",
                library="设定",
                roles={"主体": "言万心叶", "值": {"literal": "能读取他人心声"}},
                evidence=[_evidence(U1, T1, "他人的思想不由分说地在脑中响起")],
                confidence=0.8,
                confidence_method="evidence_span",
                claim_layer="fact",
                status="asserted",
            ),
            Assertion(
                book_id=BOOK,
                predicate="赐予",
                library="关系",
                roles={"施主": "黑之魔王", "受者": "言万心叶"},
                evidence=[_evidence(U1, T1, "黑之魔王站在船头")],
                confidence=0.7,
                status="asserted",
            ),
            Assertion(
                assertion_id=statement_id,
                book_id=BOOK,
                predicate="性质",
                library="设定",
                roles={"主体": "言万心叶", "值": {"literal": "能读取他人心声"}},
                evidence=[_evidence(U1, T1, "他人的思想不由分说地在脑中响起")],
                claim_layer="statement",
                speaker=Speaker(kind="narrator", name="叙述者"),
                confidence=0.6,
                confidence_method="engine_self",
                status="asserted",
            ),
            Assertion(
                book_id=BOOK,
                predicate="性质",
                library="设定",
                roles={"主体": "言万心叶", "值": {"literal": "不存在的证据"}},
                evidence=[
                    Evidence(
                        unit_id=U1,
                        start=0,
                        end=5,
                        quote="不存在的话",
                        occurrence=1,
                    )
                ],
                confidence=0.5,
                status="asserted",
            ),
        ],
        events=[
            Event(
                event_id=event_1,
                book_id=BOOK,
                type="登场",
                summary="黑之魔王站在船头",
                participants=["黑之魔王"],
                unit_id=U1,
                importance="high",
                evidence=[_evidence(U1, T1, "黑之魔王站在船头")],
                status="asserted",
            ),
            Event(
                book_id=BOOK,
                type="错误事件",
                summary="无法解析参与者",
                participants=["不存在者"],
                unit_id=U1,
                evidence=[_evidence(U1, T1, "黑之魔王站在船头")],
                status="asserted",
            ),
        ],
        cues=[
            Cue(
                cue_id=cue_1,
                book_id=BOOK,
                kind="object",
                subject={"entity_id": None, "literal": "船头"},
                setup={"unit_id": U1, "evidence": [_evidence(U1, T1, "黑之魔王站在船头")]},
                status="active",
                expected="后续登场",
                confidence=0.6,
            ),
            Cue(
                book_id=BOOK,
                kind="line",
                subject={"entity_id": None, "literal": "坏线索"},
                setup={
                    "unit_id": U1,
                    "evidence": [
                        Evidence(unit_id=U1, start=0, end=4, quote="不存在的线索", occurrence=1)
                    ],
                },
                status="active",
                expected="不会进箱",
                confidence=0.2,
            ),
        ],
        notes=["rich synthetic work layer"],
    )

    output_2 = EngineOutput(
        unit_id=U2,
        mentions=[_mention(U2, T2, "研究所"), _mention(U2, T2, "言万心叶")],
        entities=[e_lab],
        assertions=[
            Assertion(
                book_id=BOOK,
                predicate="位于",
                library="关系",
                roles={"主体": "研究所", "值": {"literal": "地下 267 层"}},
                evidence=[_evidence(U2, T2, "研究所位于地下 267 层。")],
                confidence=0.9,
                confidence_method="evidence_span",
                claim_layer="fact",
                status="asserted",
            )
        ],
        events=[
            Event(
                event_id=event_2,
                book_id=BOOK,
                type="探索",
                summary="研究所被发现",
                participants=["研究所"],
                unit_id=U2,
                importance="medium",
                evidence=[_evidence(U2, T2, "研究所位于地下 267 层。")],
                status="asserted",
            )
        ],
        notes=["rich synthetic work layer"],
    )

    _write_json(
        work / "engine" / "u_testbook_v01_c001.json",
        output_1.model_dump(mode="json"),
    )
    _write_json(
        work / "engine" / "u_testbook_v01_c002.json",
        output_2.model_dump(mode="json"),
    )

    # optional work tables
    _write_jsonl(
        work / "assemble" / "claims.jsonl",
        [
            {
                "statement_assertion_id": statement_id,
                "normalized_claim": "言万心叶能读取他人心声",
                "claim_type": "attribute",
                "questions": [
                    {
                        "qid": "q1",
                        "question": "言万心叶能读取什么？",
                        "answers": [
                            {
                                "text": "他人心声",
                                "unit_id": U1,
                                "quote": "他人的思想不由分说地在脑中响起",
                                "source": "narration",
                            }
                        ],
                        "stance": "support",
                    }
                ],
                "verdict": "supported",
                "confidence": 0.8,
                "status": "verified",
            }
        ],
    )
    _write_jsonl(
        work / "assemble" / "timeline.jsonl",
        [
            {
                "constraint_id": "t:" + "e" * 16,
                "kind": "before",
                "left": {"kind": "event", "id": event_1},
                "right": {"kind": "event", "id": event_2},
                "source": "inferred",
                "confidence": 0.5,
                "evidence": [],
                "status": "inferred",
                "deps": [event_1, event_2],
            }
        ],
    )
    _write_jsonl(
        work / "derive" / "hypotheses.jsonl",
        [
            {
                "claim": "黑之魔王与言万心叶日后敌对",
                "scope": "foreshadow",
                "evidence_sufficiency": "insufficient",
                "missing_premises": ["两人的直接冲突场景"],
                "support": [{"unit_id": U1, "quote": "黑之魔王站在船头"}],
                "derived_from": [U1],
                "confidence": 0.4,
            }
        ],
    )
    return work


# ---------------------------------------------------------------------------
# rich build / lint / tables
# ---------------------------------------------------------------------------


def test_rich_work_builds_lint_clean_box(tmp_path):
    work = make_rich_work(tmp_path)
    box = tmp_path / "box"
    result = build_box(work, box, created_at=FIXED, box_id="testbook-w5-0001")

    assert result.status == "built"
    assert result.box_id == "testbook-w5-0001"
    assert result.counts == {
        "units": 2,
        "mentions": 5,
        "entities": 3,
        "assertions": 3,
        "statements": 1,
        "claims": 1,
        "citations": 5,
        "narratives": 2,
        "graph_edges": 0,
        "assets": 0,
        "events": 2,
        "cues": 1,
        "conflicts": 0,
        "superseded": 0,
        # 7 problem rows from 4 rejected candidates (some candidates violate
        # more than one check; each check lands its own auditable row).
        "quarantined": 7,
    }
    assert result.lint_errors == ()
    assert canonical_lint_errors(box) == []

    manifest = json.loads((box / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["spec"] == "cbb-box/2"
    assert manifest["spec_version"] == "2.2.0"
    assert manifest["book"]["id"] == BOOK
    assert manifest["book"]["title"] == "异世界示例"
    assert manifest["counts"] == result.counts
    assert manifest["status"] == "built"
    assert manifest["parent"] is None
    assert manifest["incremental"]["parent"] is None
    assert manifest["incremental"]["changed_units"] == []
    assert manifest["incremental"]["dirty_entities"] == []
    assert manifest["source"]["unit_count"] == 2
    assert manifest["source"]["char_count"] == len(T1) + len(T2)
    assert manifest["engine"]["primary"] == "MockEngine"
    assert manifest["engine"]["model"] == "mock"
    assert manifest["profiles"]["genre_id"] == "light-novel-zh"
    assert manifest["qc"]["hard_gates"] == "pass"

    qc = json.loads((box / "qc" / "report.json").read_text(encoding="utf-8"))
    assert qc["status"] == "pass"
    assert all(value == "pass" for value in qc["hard_gates"].values())
    assert qc["generated_at"] == FIXED

    # Directory shape from box spec §2.
    for relative in (
        "manifest.json",
        "SPEC.md",
        "CHECKSUMS.json",
        "lexicon/genre.json",
        "lexicon/predicates.json",
        "lexicon/types.json",
        "sources/units.jsonl",
        "sources/texts/v01/u001.txt",
        "sources/texts/v01/u002.txt",
        "identity/mentions.jsonl",
        "identity/entities.jsonl",
        "identity/merges.jsonl",
        "knowledge/assertions.jsonl",
        "knowledge/claims.jsonl",
        "knowledge/events.jsonl",
        "knowledge/timeline.jsonl",
        "knowledge/cues.jsonl",
        "knowledge/proposed_predicates.jsonl",
        "derived/summaries.jsonl",
        "derived/summary_tree.json",
        "derived/entity_cards.jsonl",
        "derived/hypotheses.jsonl",
        "audit/runs.jsonl",
        "audit/provenance.jsonl",
        "audit/decisions.jsonl",
        "audit/quarantine.jsonl",
        "audit/conflicts.jsonl",
        "qc/report.json",
        "qc/gates.jsonl",
        "index/by_entity.json",
        "index/by_unit.json",
        "index/by_predicate.json",
        "index/timeline.json",
    ):
        assert (box / relative).exists(), relative

    # Tables: predicate proposal never leaks into assertions (I4).
    assertions = _read_jsonl(box / "knowledge" / "assertions.jsonl")
    predicates = {row["predicate"] for row in assertions}
    assert "赐予" not in predicates
    proposals = _read_jsonl(box / "knowledge" / "proposed_predicates.jsonl")
    assert [row["predicate"] for row in proposals] == ["赐予"]
    assert proposals[0]["count"] == 1

    # Quarantine uses only the closed reason set.
    quarantine = _read_jsonl(box / "audit" / "quarantine.jsonl")
    assert len(quarantine) == 7
    assert {row["reason"] for row in quarantine} == {
        "unresolved_quote",
        "evidence_not_found",
        "ref_missing",
    }

    # Entities linked; statement claim resolved; timeline/hypothesis written.
    mentions = _read_jsonl(box / "identity" / "mentions.jsonl")
    assert all(row["entity_id"] for row in mentions)
    claims = _read_jsonl(box / "knowledge" / "claims.jsonl")
    assert claims[0]["verdict"] == "supported"
    assert claims[0]["confidence_scope"] == "claim"
    timeline = _read_jsonl(box / "knowledge" / "timeline.jsonl")
    assert len(timeline) == 1
    hypotheses = _read_jsonl(box / "derived" / "hypotheses.jsonl")
    assert hypotheses[0]["evidence_sufficiency"] == "insufficient"
    assert hypotheses[0]["missing_premises"] == ["两人的直接冲突场景"]

    # Engine audit copies are contract-shaped.
    engine_files = sorted((box / "audit" / "engine").glob("*.json"))
    assert len(engine_files) == 2
    contract = json.loads(engine_files[0].read_text(encoding="utf-8"))
    assert contract["contract"] == "cbb-engine-output/1"
    assert contract["unit_id"].startswith("u:")
    assert set(contract) >= {"mentions", "entities", "assertions", "events", "cues"}


def test_checksums_and_merkle_are_self_consistent(tmp_path):
    work = make_rich_work(tmp_path)
    box = tmp_path / "box"
    build_box(work, box, created_at=FIXED, box_id="checksum-0001")

    manifest = json.loads((box / "manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((box / "CHECKSUMS.json").read_text(encoding="utf-8"))
    assert checksums["files"] == manifest["hashes"]["files"]
    assert checksums["merkle_root"] == manifest["hashes"]["merkle_root"]
    assert atomic.merkle_root(checksums["files"]) == checksums["merkle_root"]
    for relative, digest in checksums["files"].items():
        assert atomic.sha256_bytes((box / relative).read_bytes()) == digest
    assert "manifest.json" not in checksums["files"]
    assert "CHECKSUMS.json" not in checksums["files"]
    assert "SPEC.md" not in checksums["files"]


def test_build_is_reproducible_with_fixed_timestamp(tmp_path):
    work = make_rich_work(tmp_path)
    first = tmp_path / "box-a"
    second = tmp_path / "box-b"
    build_box(work, first, created_at=FIXED, box_id="repro-0001")
    build_box(work, second, created_at=FIXED, box_id="repro-0001")
    manifest_a = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["hashes"] == manifest_b["hashes"]
    assert manifest_a["created_at"] == manifest_b["created_at"] == FIXED


def test_unit_text_mismatch_is_recomputed_with_warning(tmp_path):
    work = make_rich_work(tmp_path)
    units = _read_jsonl(work / "plan" / "units.jsonl")
    units[0]["chars"] = 1
    units[0]["sha256"] = "sha256:" + "0" * 64
    _write_jsonl(work / "plan" / "units.jsonl", units)

    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="warn-0001")
    assert any("chars" in warning for warning in result.warnings)
    assert any("sha256 mismatch" in warning for warning in result.warnings)
    manifest = json.loads((result.box / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["units"] == 2
    assert manifest["source"]["char_count"] == len(T1) + len(T2)


# ---------------------------------------------------------------------------
# mock pipeline integration
# ---------------------------------------------------------------------------


def test_mock_two_chapter_pipeline_fails_measured_gates(tmp_path):
    """The mock pipeline emits units/mentions but zero knowledge rows.

    History: this fixture built a box that reported 7/7 hard gates ``pass`` —
    the per-row checks are vacuously true on empty tables. The measured gate
    must now fail it; pinning the old green was the bug.
    """
    from cbb_stage1.orchestrator.smoke import smoke

    work = tmp_path / "smoke-work"
    smoke(work)
    box = tmp_path / "smoke-box"
    result = build_box(
        work, box, created_at=FIXED, box_id="smoke-w5-0001", verify=False
    )

    assert result.status == "built"
    assert result.counts["units"] == 2
    assert result.counts["mentions"] == 2
    assert result.counts["entities"] == 2  # resolve entity candidates
    assert result.counts["assertions"] == 0
    assert result.counts["quarantined"] == 0

    # The measured report is not a constant: no knowledge -> hard gate fails.
    assert result.lint_errors
    qc = json.loads((box / "qc" / "report.json").read_text(encoding="utf-8"))
    assert "fail" in qc["hard_gates"].values()
    manifest = json.loads((box / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["qc"]["hard_gates"] == "fail"
    assert manifest["book"]["id"] == "smoke"
    assert manifest["source"]["format"] == "txt"

    # With the default gate (verify=True) the build must refuse to publish.
    with pytest.raises(BuildError) as excinfo:
        build_box(work, tmp_path / "smoke-box-verified", created_at=FIXED)
    assert "coverage invariant" in str(excinfo.value)
    assert not (tmp_path / "smoke-box-verified").exists()


# ---------------------------------------------------------------------------
# safety / atomicity
# ---------------------------------------------------------------------------


def test_build_refuses_to_overwrite_an_existing_box(tmp_path):
    work = make_rich_work(tmp_path)
    box = tmp_path / "box"
    build_box(work, box, created_at=FIXED, box_id="immutable-0001")
    before = (box / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_box(work, box, created_at=FIXED, box_id="immutable-0001")
    assert (box / "manifest.json").read_bytes() == before


def test_failed_gate_leaves_no_partial_box(tmp_path, monkeypatch):
    import cbb_stage1.box.writer as writer_module

    work = make_rich_work(tmp_path)
    box = tmp_path / "box"

    def fail_gate(*_args, **_kwargs):
        return ["injected schema failure"]

    monkeypatch.setattr(writer_module, "_validate_schema_rows", fail_gate)
    with pytest.raises(BuildError) as excinfo:
        build_box(work, box, created_at=FIXED, box_id="partial-0001")
    assert "schema" in str(excinfo.value)
    assert not box.exists()
    leftovers = list(tmp_path.glob(".box.build-*"))
    assert leftovers == []


def test_missing_work_root_raises(tmp_path):
    with pytest.raises(BuildError):
        build_box(tmp_path / "missing", tmp_path / "box", created_at=FIXED)


# ---------------------------------------------------------------------------
# optional merge table
# ---------------------------------------------------------------------------


def test_valid_merge_is_applied_and_remains_reversible(tmp_path):
    work = tmp_path / "work"
    text = "心叶与言万心叶同行。"
    (work / "plan" / "texts" / "v01").mkdir(parents=True)
    (work / "plan" / "texts" / "v01" / "u001.txt").write_text(text, encoding="utf-8", newline="\n")
    _write_jsonl(
        work / "plan" / "units.jsonl",
        [_unit(U1, "plan/texts/v01/u001.txt", text, 1, 1)],
    )
    _write_json(
        work / "plan" / "run.json",
        {
            "book_id": BOOK,
            "book_title": "合并测试",
            "trace_id": "merge0001",
            "engine": "MockEngine:mock",
            "started_at": FIXED,
            "ended_at": FIXED,
        },
    )
    alias_id = "e:testbook:person:" + "1" * 12
    canonical_id = "e:testbook:person:" + "2" * 12
    output = EngineOutput(
        unit_id=U1,
        mentions=[
            Mention(
                unit_id=U1,
                surface="心叶",
                start=text.index("心叶"),
                end=text.index("心叶") + 2,
                entity_id=alias_id,
            ),
            Mention(
                unit_id=U1,
                surface="言万心叶",
                start=text.index("言万心叶"),
                end=text.index("言万心叶") + 4,
                entity_id=canonical_id,
            ),
        ],
        entities=[
            Entity(
                entity_id=alias_id,
                book_id=BOOK,
                type="person",
                canonical_name="心叶",
                aliases=[],
                tier="minor",
            ),
            Entity(
                entity_id=canonical_id,
                book_id=BOOK,
                type="person",
                canonical_name="言万心叶",
                aliases=[],
                tier="canonical",
            ),
        ],
    )
    _write_json(work / "engine" / "u_testbook_v01_c001.json", output.model_dump(mode="json"))
    _write_jsonl(
        work / "resolve" / "merges.jsonl",
        [
            {
                "kind": "alias",
                "from": [alias_id],
                "to": canonical_id,
                "decided_by": "rule",
                "confidence": 0.9,
                "evidence": [],
                "at": FIXED,
            }
        ],
    )

    result = build_box(
        work, tmp_path / "box", created_at=FIXED, box_id="merge-0001", verify=False
    )
    assert result.counts["entities"] == 2
    entities = {
        row["entity_id"]: row
        for row in _read_jsonl(result.box / "identity" / "entities.jsonl")
    }
    assert entities[alias_id]["status"] == "merged"
    assert entities[alias_id]["merged_into"] == canonical_id
    assert entities[canonical_id]["status"] == "active"
    mentions = _read_jsonl(result.box / "identity" / "mentions.jsonl")
    assert {row["entity_id"] for row in mentions} == {canonical_id}
    merges = _read_jsonl(result.box / "identity" / "merges.jsonl")
    assert merges[0]["reversible"] is True
    # This fixture has units/mentions but no knowledge rows, so the measured
    # gate is expected to fail; the test pins merge behaviour, not greenness.
    assert any("coverage invariant" in error for error in canonical_lint_errors(result.box))


# ---------------------------------------------------------------------------
# incremental manifest fields
# ---------------------------------------------------------------------------


def test_changed_units_drive_incremental_manifest_fields(tmp_path):
    work = make_rich_work(tmp_path)
    box = tmp_path / "box"
    result = build_box(
        work,
        box,
        created_at=FIXED,
        box_id="incremental-0001",
        parent="testbook-w5-0000",
        changed_units=[U2],
    )
    manifest = json.loads((result.box / "manifest.json").read_text(encoding="utf-8"))
    incremental = manifest["incremental"]
    assert manifest["parent"] == "testbook-w5-0000"
    assert incremental["parent"] == "testbook-w5-0000"
    assert incremental["changed_units"] == [U2]
    # U2 mentions 研究所 + 言万心叶 → both entities are dirty.
    assert set(incremental["dirty_entities"]) == {
        row["entity_id"]
        for row in _read_jsonl(result.box / "identity" / "entities.jsonl")
        if row["canonical_name"] in ("研究所", "言万心叶")
    }
    summaries = _read_jsonl(result.box / "derived" / "summaries.jsonl")
    assert any(row["target"] == U2 and row["stale"] for row in summaries)


# ---------------------------------------------------------------------------
# module CLI
# ---------------------------------------------------------------------------


def test_module_cli_builds_a_box(tmp_path):
    from cbb_stage1.box.writer import main

    work = make_rich_work(tmp_path)
    box = tmp_path / "cli-box"
    assert (
        main(
            [
                "--work",
                str(work),
                "--out",
                str(box),
                "--created-at",
                FIXED,
                "--box-id",
                "cli-0001",
            ]
        )
        == 0
    )
    assert (box / "manifest.json").exists()
    assert canonical_lint_errors(box) == []
