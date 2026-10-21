"""P0-D tests: the box writer consumes postprocess work tables.

These tests pin the boundary introduced by task P0-D:

* claims / merges / timeline / cues / conflicts / derived rows come from the
  work tables written by the package integrations; the writer no longer
  synthesises skeletons, alias merges, conflicts, hypotheses or cycle pruning;
* missing tables are explicit empty tables, never invented content;
* a cyclic work timeline fails the build instead of being silently pruned;
* ``dedupe.integration`` / ``derive.integration`` write the work tables the
  writer consumes (entity cards + dedupe map; summaries/forecasts/hypotheses/
  community).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1.box import atomic
from cbb_stage1.box.writer import BuildError, build_box, canonical_lint_errors
from cbb_stage1.dedupe.integration import run as run_dedupe
from cbb_stage1.derive.integration import run as run_derive

FIXED = "2026-10-18T23:00:00+08:00"
BOOK = "t"
U1 = "u:t:v01:c001"
U2 = "u:t:v01:c002"
T1 = "贝塔心叶走近黑之魔王。"
T2 = "黑之魔王说：「我来了。」"

STATEMENT_ID = "a:" + "1" * 16
FACT_A = "a:" + "2" * 16
FACT_B = "a:" + "3" * 16
EVENT_1 = "ev:" + "1" * 16
EVENT_2 = "ev:" + "2" * 16
ENGINE_CUE = "cue:" + "1" * 16
WORK_CUE = "cue:" + "9" * 16


# ---------------------------------------------------------------------------
# work fixture helpers
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


def _unit(uid: str, text_rel: str, text: str, order: int) -> dict:
    return {
        "unit_id": uid,
        "book_id": BOOK,
        "volume": 1,
        "chapter": order,
        "title": f"第{order}章",
        "order": order,
        "prev": None,
        "next": None,
        "text_path": text_rel,
        "chars": len(text),
        "sha256": atomic.sha256_text(text),
        "narrative_mode": "present",
        "narrator": {"kind": "third_person", "entity_id": None, "name": "叙述者"},
        "narrator_reliability": "reliable",
        "source_locator": {"href": f"chapter{order}.xhtml", "nav_index": order - 1},
        "scenes": [],
        "created_at": FIXED,
    }


def _evidence(unit_id: str, text: str, quote: str) -> dict:
    start = text.index(quote)
    return {
        "unit_id": unit_id,
        "start": start,
        "end": start + len(quote),
        "quote": quote,
        "occurrence": 1,
        "mode": "narration",
    }


def _assertion(
    assertion_id: str,
    value: str,
    *,
    claim_layer: str = "fact",
    speaker: dict | None = None,
) -> dict:
    record = {
        "assertion_id": assertion_id,
        "book_id": BOOK,
        "predicate": "性质",
        "library": "设定",
        "roles": {"主体": "贝塔心叶", "值": {"literal": value}},
        "qualifiers": {"negated": False, "modality": "asserted", "time": None, "location": None},
        "claim_layer": claim_layer,
        "evidence": [_evidence(U1, T1, "贝塔心叶")],
        "confidence": 0.7,
        "confidence_method": "evidence_span",
        "status": "asserted",
        "derived_from": [],
        "deps": [U1],
    }
    if speaker is not None:
        record["speaker"] = speaker
    return record


def _event(event_id: str, summary: str) -> dict:
    return {
        "event_id": event_id,
        "book_id": BOOK,
        "type": "登场",
        "summary": summary,
        "participants": ["贝塔心叶"],
        "unit_id": U1,
        "importance": "high",
        "evidence": [_evidence(U1, T1, "贝塔心叶")],
        "status": "asserted",
        "deps": [U1],
    }


def make_work(tmp_path: Path) -> Path:
    """Two units + engine candidates; **no** postprocess work tables yet."""
    work = tmp_path / "work"
    (work / "plan" / "texts" / "v01").mkdir(parents=True)
    (work / "plan" / "texts" / "v01" / "u001.txt").write_text(T1, encoding="utf-8", newline="\n")
    (work / "plan" / "texts" / "v01" / "u002.txt").write_text(T2, encoding="utf-8", newline="\n")
    _write_jsonl(
        work / "plan" / "units.jsonl",
        [_unit(U1, "plan/texts/v01/u001.txt", T1, 1), _unit(U2, "plan/texts/v01/u002.txt", T2, 2)],
    )
    _write_json(
        work / "plan" / "run.json",
        {
            "book_id": BOOK,
            "book_title": "P0-D 示例",
            "book_format": "text",
            "trace_id": "p0d0001",
            "engine": "MockEngine:mock",
            "started_at": "2026-10-18T23:00:00",
            "ended_at": "2026-10-18T23:00:01",
        },
    )
    output_1 = {
        "unit_id": U1,
        "mentions": [
            {"unit_id": U1, "surface": "贝塔心叶", "start": 0, "end": 4, "occurrence": 1, "mention_type": "name"},
            {"unit_id": U1, "surface": "黑之魔王", "start": 5, "end": 9, "occurrence": 1, "mention_type": "name"},
        ],
        "entities": [
            {"book_id": BOOK, "type": "person", "canonical_name": "贝塔心叶", "aliases": ["心叶"], "tier": "canonical"},
            {"book_id": BOOK, "type": "person", "canonical_name": "黑之魔王", "aliases": [], "tier": "major"},
        ],
        "assertions": [
            _assertion(STATEMENT_ID, "能读取他人心声", claim_layer="statement", speaker={"kind": "narrator", "name": "叙述者"}),
            _assertion(FACT_A, "A"),
            _assertion(FACT_B, "B"),
        ],
        "events": [_event(EVENT_1, "贝塔心叶走近"), _event(EVENT_2, "黑之魔王回望")],
        "cues": [
            {
                "cue_id": ENGINE_CUE,
                "book_id": BOOK,
                "kind": "object",
                "subject": {"entity_id": None, "literal": "engine-cue"},
                "setup": {"unit_id": U1, "evidence": [_evidence(U1, T1, "贝塔心叶")]},
                "status": "active",
                "expected": "引擎线索",
                "confidence": 0.5,
            }
        ],
        "notes": ["p0d fixture"],
    }
    output_2 = {
        "unit_id": U2,
        "mentions": [
            {"unit_id": U2, "surface": "黑之魔王", "start": 0, "end": 4, "occurrence": 1, "mention_type": "name"}
        ],
        "entities": [],
        "notes": ["p0d fixture"],
    }
    _write_json(work / "engine" / "u_t_v01_c001.json", output_1)
    _write_json(work / "engine" / "u_t_v01_c002.json", output_2)
    return work


def _valid_claim() -> dict:
    return {
        "statement_assertion_id": STATEMENT_ID,
        "normalized_claim": "贝塔心叶能读取他人心声",
        "claim_type": "attribute",
        "questions": [
            {
                "qid": "q1",
                "question": "贝塔心叶能读取什么？",
                "answers": [
                    {
                        "text": "他人心声",
                        "unit_id": U1,
                        "quote": "贝塔心叶",
                        "source": "narration",
                    }
                ],
                "stance": "support",
            }
        ],
        "verdict": "supported",
        "confidence": 0.8,
        "confidence_method": "claim_qa",
        "confidence_scope": "claim",
        "verified_by": {"verifier": "fixture", "context_mode": "isolated_source"},
        "status": "verified",
    }


def _timeline_constraint(cid: str, left: str, right: str) -> dict:
    return {
        "constraint_id": cid,
        "book_id": BOOK,
        "kind": "before",
        "left": {"kind": "event", "id": left},
        "right": {"kind": "event", "id": right},
        "source": "inferred",
        "confidence": 0.5,
        "evidence": [],
        "status": "inferred",
        "deps": [left, right],
    }


def _conflict() -> dict:
    return {
        "conflict_id": "c:" + "4" * 16,
        "book_id": BOOK,
        "kind": "attribute",
        "left": FACT_A,
        "right": FACT_B,
        "detail": "贝塔心叶 的性质出现两个值：['A', 'B']",
        "status": "open",
        "decision": None,
        "resolved_by": None,
        "at": None,
    }


# ---------------------------------------------------------------------------
# writer consumes work tables (no parallel synthesis)
# ---------------------------------------------------------------------------


def test_missing_claims_stay_explicitly_empty(tmp_path):
    work = make_work(tmp_path)
    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="p0d-empty-claims")
    assert result.counts["claims"] == 0
    assert _read_jsonl(result.box / "knowledge" / "claims.jsonl") == []
    assert canonical_lint_errors(result.box) == []


def test_writer_reads_claims_from_work_table(tmp_path):
    work = make_work(tmp_path)
    _write_jsonl(work / "assemble" / "claims.jsonl", [_valid_claim()])
    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="p0d-claims")
    claims = _read_jsonl(result.box / "knowledge" / "claims.jsonl")
    assert len(claims) == 1
    assert claims[0]["statement_assertion_id"] == STATEMENT_ID
    assert claims[0]["verdict"] == "supported"
    # Only the statement with a work claim gets a row; the two fact assertions
    # do not acquire skeleton claims.
    assert result.counts["claims"] == 1
    assert canonical_lint_errors(result.box) == []


def test_writer_reads_work_assertion_ids_so_claims_resolve(tmp_path):
    """test_box_pipeline regression: claims reference assemble assertion ids."""
    work = make_work(tmp_path)
    work_statement_id = "a:" + "f" * 16
    _write_jsonl(
        work / "assemble" / "assertions.jsonl",
        [
            {
                "assertion_id": work_statement_id,
                "book_id": BOOK,
                "predicate": "性质",
                "library": "设定",
                "roles": {"主体": "贝塔心叶", "值": {"literal": "能读取他人心声"}},
                "claim_layer": "statement",
                "speaker": {"kind": "narrator", "entity_id": None, "name": "叙述者"},
                "evidence": [_evidence(U1, T1, "贝塔心叶")],
                "confidence": 0.7,
                "confidence_method": "evidence_span",
                "status": "asserted",
            }
        ],
    )
    claim = _valid_claim()
    claim["statement_assertion_id"] = work_statement_id
    _write_jsonl(work / "assemble" / "claims.jsonl", [claim])

    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="p0d-work-assert")
    assertion_ids = {row["assertion_id"] for row in _read_jsonl(result.box / "knowledge" / "assertions.jsonl")}
    assert work_statement_id in assertion_ids
    # The engine-derived duplicate statement must not be added twice.
    statements = [
        row
        for row in _read_jsonl(result.box / "knowledge" / "assertions.jsonl")
        if row.get("claim_layer") == "statement"
    ]
    assert len(statements) == 1
    claims = _read_jsonl(result.box / "knowledge" / "claims.jsonl")
    assert len(claims) == 1
    assert claims[0]["statement_assertion_id"] == work_statement_id
    # The work table is authoritative: the writer must say so and must not
    # silently resurrect engine candidates the Assembler rejected.
    assert any(
        "authoritative" in warning and "assertion" in warning
        for warning in result.warnings
    )
    assert canonical_lint_errors(result.box) == []


def test_writer_generates_conservative_entities_from_mentions(tmp_path):
    """Mock engines only yield mentions: the box must not silently be entity-less."""
    work = tmp_path / "work"
    text = "我走进房间。"
    (work / "plan" / "texts" / "v01").mkdir(parents=True)
    (work / "plan" / "texts" / "v01" / "u001.txt").write_text(text, encoding="utf-8", newline="\n")
    _write_jsonl(work / "plan" / "units.jsonl", [_unit(U1, "plan/texts/v01/u001.txt", text, 1)])
    _write_json(
        work / "plan" / "run.json",
        {
            "book_id": BOOK,
            "book_title": "mentions-only",
            "trace_id": "mentions0001",
            "engine": "MockEngine:mock",
            "started_at": FIXED,
            "ended_at": FIXED,
        },
    )
    _write_json(
        work / "engine" / "u_t_v01_c001.json",
        {
            "unit_id": U1,
            "mentions": [
                {"unit_id": U1, "surface": "我", "start": 0, "end": 1, "occurrence": 1, "mention_type": "name"}
            ],
            "entities": [],
        },
    )
    # resolve ran but produced no candidates (mock engine has only mentions).
    _write_jsonl(work / "resolve" / "entity_candidates.jsonl", [])

    result = build_box(
        work, tmp_path / "box", created_at=FIXED, box_id="p0d-mentions", verify=False
    )
    assert result.counts["entities"] == 1
    assert result.counts["mentions"] == 1
    entities = _read_jsonl(result.box / "identity" / "entities.jsonl")
    assert entities[0]["canonical_name"] == "我"
    assert entities[0]["type"] == "unknown"
    assert entities[0]["tier"] == "minor"
    assert entities[0]["status"] == "active"
    mentions = _read_jsonl(result.box / "identity" / "mentions.jsonl")
    assert mentions[0]["entity_id"] == entities[0]["entity_id"]
    assert any("mentions" in warning for warning in result.warnings)
    # Entities only -> no assertions/events; the measured gate must fail.
    assert any("coverage invariant" in error for error in canonical_lint_errors(result.box))


def test_writer_reads_merges_from_work_table_and_does_not_auto_merge(tmp_path):
    roots = make_merge_work(tmp_path)
    result = build_box(
        roots["plain"], tmp_path / "box-plain", created_at=FIXED,
        box_id="p0d-nomerge", verify=False,
    )
    # No work merge: the alias remains a separate active entity (no synthesis).
    entities = {row["entity_id"]: row for row in _read_jsonl(result.box / "identity" / "entities.jsonl")}
    assert entities["e:t:person:" + "a" * 12]["status"] == "active"
    assert _read_jsonl(result.box / "identity" / "merges.jsonl") == []

    merged_work = roots["merged"]
    _write_jsonl(
        merged_work / "assemble" / "merges.jsonl",
        [
            {
                "kind": "alias",
                "from": ["e:t:person:" + "a" * 12],
                "to": "e:t:person:" + "b" * 12,
                "decided_by": "rule",
                "confidence": 0.9,
                "evidence": [],
                "at": FIXED,
            }
        ],
    )
    merged = build_box(
        merged_work, tmp_path / "box-merged", created_at=FIXED,
        box_id="p0d-merge", verify=False,
    )
    entities = {row["entity_id"]: row for row in _read_jsonl(merged.box / "identity" / "entities.jsonl")}
    assert entities["e:t:person:" + "a" * 12]["status"] == "merged"
    assert entities["e:t:person:" + "a" * 12]["merged_into"] == "e:t:person:" + "b" * 12
    assert len(_read_jsonl(merged.box / "identity" / "merges.jsonl")) == 1
    # The fixture has units/entities but no knowledge rows -> gate fails.
    assert any("coverage invariant" in error for error in canonical_lint_errors(merged.box))


def make_merge_work(tmp_path: Path) -> dict[str, Path]:
    """Two work roots: one without a merge table, one with it."""
    text = "心叶与贝塔心叶同行。"
    alias_id = "e:t:person:" + "a" * 12
    canonical_id = "e:t:person:" + "b" * 12

    def build_root(name: str) -> Path:
        work = tmp_path / name
        (work / "plan" / "texts" / "v01").mkdir(parents=True)
        (work / "plan" / "texts" / "v01" / "u001.txt").write_text(text, encoding="utf-8", newline="\n")
        _write_jsonl(
            work / "plan" / "units.jsonl",
            [_unit(U1, "plan/texts/v01/u001.txt", text, 1)],
        )
        _write_json(
            work / "plan" / "run.json",
            {
                "book_id": BOOK,
                "book_title": "merge fixture",
                "trace_id": "merge0001",
                "engine": "MockEngine:mock",
                "started_at": FIXED,
                "ended_at": FIXED,
            },
        )
        _write_json(
            work / "engine" / "u_t_v01_c001.json",
            {
                "unit_id": U1,
                "mentions": [
                    {"unit_id": U1, "surface": "心叶", "start": 0, "end": 2, "occurrence": 1, "entity_id": alias_id},
                    {"unit_id": U1, "surface": "贝塔心叶", "start": 3, "end": 7, "occurrence": 1, "entity_id": canonical_id},
                ],
                "entities": [
                    {"entity_id": alias_id, "book_id": BOOK, "type": "person", "canonical_name": "心叶", "aliases": [], "tier": "minor"},
                    {"entity_id": canonical_id, "book_id": BOOK, "type": "person", "canonical_name": "贝塔心叶", "aliases": [], "tier": "canonical"},
                ],
            },
        )
        return work

    plain = build_root("plain")
    merged = build_root("merged")
    return {"plain": plain, "merged": merged}


def test_writer_prefers_work_cues_over_engine_cues(tmp_path):
    work = make_work(tmp_path)
    before = build_box(work, tmp_path / "box-engine", created_at=FIXED, box_id="p0d-engcue")
    engine_cues = _read_jsonl(before.box / "knowledge" / "cues.jsonl")
    assert [row["cue_id"] for row in engine_cues] == [ENGINE_CUE]

    _write_jsonl(
        work / "assemble" / "cues.jsonl",
        [
            {
                "cue_id": WORK_CUE,
                "book_id": BOOK,
                "kind": "line",
                "subject": {"entity_id": None, "literal": "work-cue"},
                "setup": {"unit_id": U1, "evidence": [_evidence(U1, T1, "贝塔心叶")]},
                "reinforce": [],
                "payoff": {"event_id": None, "unit_id": None, "evidence": []},
                "status": "active",
                "expected": "集成线索",
                "confidence": 0.6,
                "deps": [U1],
            }
        ],
    )
    after = build_box(work, tmp_path / "box-work", created_at=FIXED, box_id="p0d-workcue")
    cues = _read_jsonl(after.box / "knowledge" / "cues.jsonl")
    assert [row["cue_id"] for row in cues] == [WORK_CUE]
    assert canonical_lint_errors(after.box) == []


def test_writer_rejects_cyclic_work_timeline_instead_of_pruning(tmp_path):
    work = make_work(tmp_path)
    _write_jsonl(
        work / "assemble" / "timeline.jsonl",
        [
            _timeline_constraint("t:" + "1" * 16, EVENT_1, EVENT_2),
            _timeline_constraint("t:" + "2" * 16, EVENT_2, EVENT_1),
        ],
    )
    with pytest.raises(BuildError) as excinfo:
        build_box(work, tmp_path / "box", created_at=FIXED, box_id="p0d-cycle")
    assert "cycle" in str(excinfo.value)
    assert not (tmp_path / "box").exists()

    _write_jsonl(
        work / "assemble" / "timeline.jsonl",
        [_timeline_constraint("t:" + "3" * 16, EVENT_1, EVENT_2)],
    )
    result = build_box(work, tmp_path / "box-ok", created_at=FIXED, box_id="p0d-acyclic")
    assert len(_read_jsonl(result.box / "knowledge" / "timeline.jsonl")) == 1


def test_writer_reads_conflicts_only_from_validate_table(tmp_path):
    work = make_work(tmp_path)
    plain = build_box(work, tmp_path / "box-plain", created_at=FIXED, box_id="p0d-noconflict")
    # FACT_A/FACT_B disagree, but the writer does not run conflict detection.
    assert plain.counts["conflicts"] == 0
    assert _read_jsonl(plain.box / "knowledge" / "conflicts.jsonl") == []

    _write_jsonl(work / "audit" / "conflicts.jsonl", [_conflict()])
    result = build_box(work, tmp_path / "box-conflict", created_at=FIXED, box_id="p0d-conflict")
    conflicts = _read_jsonl(result.box / "knowledge" / "conflicts.jsonl")
    assert len(conflicts) == 1
    assert conflicts[0]["left"] == FACT_A
    assert result.counts["conflicts"] == 1
    assert len(_read_jsonl(result.box / "audit" / "conflicts.jsonl")) == 1
    assert canonical_lint_errors(result.box) == []


# ---------------------------------------------------------------------------
# dedupe integration
# ---------------------------------------------------------------------------


def test_dedupe_integration_writes_entity_cards_and_reversible_map(tmp_path):
    work = tmp_path / "work"
    outputs = [
        {
            "unit_id": U1,
            "entities": [
                {
                    "canonical_name": "贝塔心叶",
                    "type": "person",
                    "aliases": ["心叶"],
                    "description": "能读取他人心声。",
                },
                {
                    "canonical_name": "贝塔心叶 ",  # whitespace variant
                    "type": "person",
                    "aliases": [],
                    "description": " 能读取他人心声。 ",  # exact-duplicate fragment
                },
            ],
        }
    ]
    units = [_unit(U1, "plan/texts/v01/u001.txt", T1, 1)]
    result = run_dedupe(
        work,
        book_id=BOOK,
        outputs=outputs,
        units=units,
        created_at=FIXED,
        upstream={},
    )
    assert result["stage"] == "dedupe"
    assert result["status"] == "ok"
    cards = _read_jsonl(work / "assemble" / "entity_cards.jsonl")
    assert len(cards) == 1
    card = cards[0]
    assert card["canonical_name"] == "贝塔心叶"
    assert card["description"] is None
    assert card["description_mode"] == "observations_only"
    assert card["observed_descriptions"] == ["能读取他人心声。"]
    # Whitespace/case variants are reversible-map entries, not aliases.
    assert card["aliases"] == ["心叶"]
    assert card["produced_by"] == "dedupe:integration"
    assert result["counts"]["entity_duplicates"] == 1
    assert result["counts"]["description_duplicates"] == 1

    dedupe_map = json.loads((work / "assemble" / "dedupe_map.json").read_text(encoding="utf-8"))
    assert dedupe_map["contract"] == "cbb-dedupe-map/1"
    assert dedupe_map["entities"]["mapping"]["贝塔心叶 "] == "贝塔心叶"
    assert dedupe_map["descriptions"]["duplicates"][" 能读取他人心声。 "] == "能读取他人心声。"


def test_dedupe_integration_without_input_is_explicitly_empty(tmp_path):
    work = tmp_path / "work"
    result = run_dedupe(work, book_id=BOOK, outputs=[], units=[], created_at=FIXED, upstream={})
    assert result["counts"]["entity_cards"] == 0
    assert _read_jsonl(work / "assemble" / "entity_cards.jsonl") == []
    dedupe_map = json.loads((work / "assemble" / "dedupe_map.json").read_text(encoding="utf-8"))
    assert dedupe_map["entities"]["clusters"] == []


# ---------------------------------------------------------------------------
# derive integration
# ---------------------------------------------------------------------------


def test_derive_integration_writes_four_work_tables(tmp_path):
    work = make_work(tmp_path)
    _write_jsonl(work / "audit" / "conflicts.jsonl", [_conflict()])
    units = _read_jsonl(work / "plan" / "units.jsonl")

    result = run_derive(
        work,
        book_id=BOOK,
        outputs=[],
        units=units,
        created_at=FIXED,
        upstream={},
    )
    assert result["stage"] == "derive"
    assert result["status"] == "ok"
    for relative in (
        "derive/summaries.jsonl",
        "derive/forecasts.jsonl",
        "derive/hypotheses.jsonl",
        "derive/community.jsonl",
    ):
        assert (work / relative).exists(), relative

    summaries = _read_jsonl(work / "derive" / "summaries.jsonl")
    scopes = [row["scope"] for row in summaries]
    assert scopes[0] == "book"
    assert "volume" in scopes and "chapter" in scopes
    assert all(row["text"] is None for row in summaries)  # no fabricated summary
    assert all(row["produced_by"] == "derive:integration" for row in summaries)

    # forecasts: chapter-scoped PROXY, marked theoretical, method named.
    # §5.20 asks for an auxiliary reading, not the real 100-Endings method; the
    # proxy is legitimate as long as it cannot be mistaken for the paper's metric.
    forecasts = _read_jsonl(work / "derive" / "forecasts.jsonl")
    assert len(forecasts) == 2
    assert all(row["theory"] is True for row in forecasts)
    assert all(row["scope"] == "chapter" for row in forecasts)
    assert all(row["method"] == "lexical_singleton_ratio" for row in forecasts)
    assert all(0.0 <= row["no_rate"] <= 1.0 for row in forecasts)

    hypotheses = _read_jsonl(work / "derive" / "hypotheses.jsonl")
    assert len(hypotheses) == 1
    assert hypotheses[0]["evidence_sufficiency"] == "insufficient"
    assert hypotheses[0]["missing_premises"]
    assert set(hypotheses[0]["derived_from"]) == {FACT_A, FACT_B}

    assert _read_jsonl(work / "derive" / "community.jsonl") == []
    assert result["counts"]["community"] == 0


def test_derive_summaries_use_upstream_text_when_supplied(tmp_path):
    work = make_work(tmp_path)
    units = _read_jsonl(work / "plan" / "units.jsonl")
    result = run_derive(
        work,
        book_id=BOOK,
        outputs=[],
        units=units,
        created_at=FIXED,
        upstream={"unit_summaries": {U1: "上游给出的第一章摘要"}},
    )
    summaries = _read_jsonl(work / "derive" / "summaries.jsonl")
    chapter = next(row for row in summaries if row["target"] == U1)
    assert chapter["text"] == "上游给出的第一章摘要"
    assert result["counts"]["summaries"] == len(summaries)


# ---------------------------------------------------------------------------
# end-to-end: integrations -> work tables -> box
# ---------------------------------------------------------------------------


def test_writer_consumes_integration_derived_and_entity_card_tables(tmp_path):
    work = make_work(tmp_path)
    units = _read_jsonl(work / "plan" / "units.jsonl")
    outputs = [
        json.loads((work / "engine" / name).read_text(encoding="utf-8"))
        for name in ("u_t_v01_c001.json", "u_t_v01_c002.json")
    ]
    _write_jsonl(work / "audit" / "conflicts.jsonl", [_conflict()])
    _write_jsonl(work / "assemble" / "claims.jsonl", [_valid_claim()])
    _write_jsonl(
        work / "assemble" / "timeline.jsonl",
        [_timeline_constraint("t:" + "5" * 16, EVENT_1, EVENT_2)],
    )

    run_dedupe(work, book_id=BOOK, outputs=outputs, units=units, created_at=FIXED, upstream={})
    run_derive(work, book_id=BOOK, outputs=outputs, units=units, created_at=FIXED, upstream={})

    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="p0d-e2e")
    assert canonical_lint_errors(result.box) == []

    # derived tables are the integration's bytes, not writer-invented rows.
    work_cards = _read_jsonl(work / "assemble" / "entity_cards.jsonl")
    box_cards = _read_jsonl(result.box / "derived" / "entity_cards.jsonl")
    assert box_cards == work_cards

    work_summaries = _read_jsonl(work / "derive" / "summaries.jsonl")
    box_summaries = _read_jsonl(result.box / "derived" / "summaries.jsonl")
    assert box_summaries == work_summaries

    work_forecasts = _read_jsonl(work / "derive" / "forecasts.jsonl")
    box_forecasts = _read_jsonl(result.box / "derived" / "inference" / "forecasts.jsonl")
    assert box_forecasts == work_forecasts

    work_hypotheses = _read_jsonl(work / "derive" / "hypotheses.jsonl")
    box_hypotheses = _read_jsonl(result.box / "derived" / "hypotheses.jsonl")
    assert box_hypotheses == work_hypotheses

    assert _read_jsonl(result.box / "derived" / "community_reports.jsonl") == []
    assert result.counts["claims"] == 1
    assert result.counts["conflicts"] == 1
    assert len(_read_jsonl(result.box / "knowledge" / "timeline.jsonl")) == 1


def test_writer_does_not_resurrect_an_assertion_the_stage_rejected(tmp_path):
    """Measured on run 725edc6cc5eb: assemble accepted 9, the box shipped 10.

    `assemble/assertions.jsonl` is what the Assembler accepted after role,
    value and evidence checks. The writer used to re-admit any engine candidate
    whose content key was absent from that table — resurrecting exactly what the
    stage threw away. The content key cannot diff the two sides either: the stage
    stores post-resolution roles (`e:` ids) while the engine output holds
    pre-resolution surfaces, so no key ever matches.

    The symptom was also visible in `predicate_reconciliation`:
    `性质: {emitted: 5, accepted: 6}` — accepted exceeded emitted.
    """
    work = make_work(tmp_path)
    accepted_id = "a:" + "e" * 16
    _write_jsonl(
        work / "assemble" / "assertions.jsonl",
        [
            {
                "assertion_id": accepted_id,
                "book_id": BOOK,
                "predicate": "性质",
                "library": "设定",
                "roles": {"主体": "贝塔心叶", "值": {"literal": "能读取他人心声"}},
                "claim_layer": "fact",
                "speaker": None,
                "evidence": [_evidence(U1, T1, "贝塔心叶")],
                "confidence": 0.7,
                "confidence_method": "evidence_span",
                "status": "asserted",
            }
        ],
    )
    _write_jsonl(
        work / "assemble" / "quarantine.jsonl",
        [
            {
                "item_id": "a:pending:" + "d" * 16,
                "kind": "assertion",
                "reason": "role_mismatch",
                "detail": "entity role expects an 'e:' id",
                "source_run": "run:test",
            }
        ],
    )

    result = build_box(work, tmp_path / "box", created_at=FIXED, box_id="p0d-no-resurrect", verify=False)
    rows = _read_jsonl(result.box / "knowledge" / "assertions.jsonl")
    ids = {row["assertion_id"] for row in rows}
    assert accepted_id in ids
    # Anything else in the box would be a resurrection of a rejected candidate.
    extra = ids - {accepted_id}
    assert not extra, f"box contains assertions the stage did not accept: {sorted(extra)}"
    assert any("authoritative" in warning for warning in result.warnings)
