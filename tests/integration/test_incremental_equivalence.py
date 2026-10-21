"""Incremental equivalence tests (A20/D6, task P2-C).

The fixture is a **two-volume** light-novel book (volume headings in the text;
``ingest/units.py`` currently flattens volumes to ``v01`` — the incremental layer
is volume-agnostic, so the test covers the same unit/volume dynamics at the unit
level). Two complete work layers are built through the real integration chain
(``ingest`` → engine → ``run_postprocess``), then the incremental plan is applied
to the *old* work layer and compared against the full rebuild: engine artifacts
and every work table must be equal after sorting by primary id.

This file deliberately does **not** drive ``orchestrator.pipeline``: the plan
function is the shell-agnostic interface (W1 wires it into the pipeline later),
and the test must stay independent of the shell's in-flight concurrency work.

Run::

    .venv\\Scripts\\python.exe -m pytest tests/integration/test_incremental_equivalence.py -q
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pytest

from cbb_stage1.engines.base import Engine
from cbb_stage1.incremental import (
    IncrementalPlan,
    MissingDependenciesError,
    plan_incremental,
)
from cbb_stage1.ingest.loader import load_book
from cbb_stage1.ingest.units import build_units
from cbb_stage1.models import Assertion, EngineOutput, Entity, Event, Evidence
from cbb_stage1.orchestrator.postprocess import run_postprocess

# ---------------------------------------------------------------------------
# Fixture book: two volumes, four chapters across the scenarios.
# ---------------------------------------------------------------------------
VOL1 = "# 第一卷 边界\n# 第1章 起\n贝塔心叶走进大厅，看见黑之魔王。\n# 第2章 承\n拉法把剑放在桌上，说这是证据。\n"
VOL1_EDITED = (
    "# 第一卷 边界\n# 第1章 起\n贝塔心叶走进大厅，看见黑之魔王。\n"
    "# 第2章 承\n洛平把剑放在桌上，说这不是证据。\n"
)
VOL2 = "# 第二卷 回声\n# 第3章 转\n苍蓝色的影子从墙后出现。\n"

BOOK_V1 = VOL1                                 # 2 chapters, 1 volume section
BOOK_V2 = VOL1_EDITED + VOL2                   # 3 chapters, chapter 2 edited
BOOK_V3 = VOL1                                 # 2 chapters (drops the old ch3)

#: Work tables compared after the incremental apply. Paths mirror the eight
#: integration stages; rows are compared sorted by their primary key.
TABLE_PRIMARY_KEY = {
    "resolve/entity_candidates.jsonl": "entity_id",
    "assemble/assertions.jsonl": "assertion_id",
    "assemble/events.jsonl": "event_id",
    "assemble/entity_cards.jsonl": "entity_id",
    "assemble/claims.jsonl": "claim_id",
    "assemble/cues.jsonl": "cue_id",
    "derive/summaries.jsonl": "summary_id",
    "derive/hypotheses.jsonl": "hypothesis_id",
}

#: Non-deterministic keys the shell may embed; they are not part of the
#: incremental equivalence claim (design ``00`` §4.16 excludes timestamps/run ids).
VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "generated_at",
        "built_at",
        "started_at",
        "ended_at",
        "timestamp",
        "last_updated",
        "run_id",
        "trace_id",
    }
)


class ChapterEngine(Engine):
    """Deterministic engine whose output depends on the chapter text.

    Two entities (first/last two characters of the chapter) plus one relation
    assertion and one event per unit. Because the output is text-dependent, a
    plan that misses a changed unit leaves a stale engine artifact behind and the
    equivalence assertion fails.
    """

    def extract(self, unit_id, text, card, known_entities):  # noqa: ANN001 - interface
        head = text[:4]
        first, second = text[:2], text[2:4]
        evidence = Evidence(unit_id=unit_id, start=0, end=len(head), quote=head)
        return EngineOutput(
            unit_id=unit_id,
            entities=[
                Entity(book_id="t", type="person", canonical_name=first, tier="major"),
                Entity(book_id="t", type="person", canonical_name=second, tier="minor"),
            ],
            assertions=[
                Assertion(
                    book_id="t",
                    predicate="亲属",
                    library="关系",
                    roles={"施事": first, "受事": second},
                    evidence=[evidence],
                    confidence=0.9,
                    confidence_method="evidence_span",
                    status="asserted",
                )
            ],
            events=[
                Event(
                    book_id="t",
                    type="事件",
                    summary=head,
                    participants=[first],
                    unit_id=unit_id,
                    evidence=[evidence],
                )
            ],
        )


class EntityOnlyEngine(Engine):
    """Engine that emits entities only (no assertions/events/cues).

    The dirty closure must still reach the entity candidates of a changed unit;
    that edge comes from ``resolve/entity_candidates.jsonl[*]["units"]``, which
    ``deps.py`` alone does not wire.
    """

    def extract(self, unit_id, text, card, known_entities):  # noqa: ANN001 - interface
        return EngineOutput(
            unit_id=unit_id,
            entities=[
                Entity(book_id="t", type="person", canonical_name=text[:2], tier="major")
            ],
        )


# ---------------------------------------------------------------------------
# Small work-layer helpers (stdlib only; mirror the pipeline's file layout).
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8", newline="\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _engine_rel(unit_id: str) -> str:
    return "engine/%s.json" % unit_id.replace(":", "_")


def _build_work(book_path: Path, work_dir: Path, engine: Engine) -> list[dict[str, Any]]:
    """Build one complete work layer: units → engine artifacts → postprocess."""
    book = load_book(book_path, book_id="t", title=book_path.stem)
    work = Path(work_dir)
    units = build_units(
        book,
        work,
        book_id="t",
        chapters="all",
        chunk_target_tokens=1200,
        chunk_overlap_tokens=100,
        chars_per_token=1.5,
    )
    records = [unit.to_record() for unit in units]
    (work / "plan").mkdir(parents=True, exist_ok=True)
    _write_jsonl(work / "plan" / "units.jsonl", records)

    outputs: list[dict[str, Any]] = []
    for unit in units:
        output = engine.extract(unit.unit_id, unit.text, {}, [])
        payload = output.model_dump()
        outputs.append(payload)
        _write_json(work / _engine_rel(unit.unit_id), payload)

    result = run_postprocess(work, book_id="t", outputs=outputs, units=records)
    assert result["ok"] is True, result
    return records


def _unit_text(work: Path, record: dict[str, Any]) -> str:
    return (work / record["text_path"]).read_text(encoding="utf-8")


def _drop_replace_rows(work: Path, plan: IncrementalPlan) -> None:
    """Apply ``plan.replace`` to the old tables (the recompute half of dirty)."""
    for relative, ids in plan.replace.items():
        path = work / relative
        if not path.is_file():
            continue
        key = TABLE_PRIMARY_KEY.get(relative, "id")
        dropped = set(ids)
        kept = [row for row in _read_jsonl(path) if str(row.get(key)) not in dropped]
        _write_jsonl(path, kept)


def _apply_incremental(
    work_old: Path,
    work_full: Path,
    work_inc: Path,
    plan: IncrementalPlan,
    units_new: list[dict[str, Any]],
    engine: Engine,
) -> None:
    """Simulate the shell's apply step: reuse + cleanup + re-extract + replace."""
    if work_inc.exists():
        shutil.rmtree(work_inc)
    shutil.copytree(work_old, work_inc)

    for relative in plan.cleanup_files:
        target = work_inc / relative
        if target.is_file():
            target.unlink()

    _drop_replace_rows(work_inc, plan)

    by_id = {record["unit_id"]: record for record in units_new}
    for unit_id in plan.extract_units:
        text = _unit_text(work_full, by_id[unit_id])
        payload = engine.extract(unit_id, text, {}, []).model_dump()
        _write_json(work_inc / _engine_rel(unit_id), payload)

    outputs = [_read_json(work_inc / _engine_rel(record["unit_id"])) for record in units_new]
    result = run_postprocess(work_inc, book_id="t", outputs=outputs, units=units_new)
    assert result["ok"] is True, result


# ---------------------------------------------------------------------------
# Equivalence assertion: engine artifacts + work tables, sorted by id.
# ---------------------------------------------------------------------------
def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _sort_key(relative: str, row: dict[str, Any]) -> str:
    key = TABLE_PRIMARY_KEY.get(relative)
    if key and row.get(key) is not None:
        return str(row[key])
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _assert_equivalent(work_inc: Path, work_full: Path) -> None:
    inc_engine = sorted(p.name for p in (work_inc / "engine").glob("*.json"))
    full_engine = sorted(p.name for p in (work_full / "engine").glob("*.json"))
    assert inc_engine == full_engine, "engine artifact set differs"
    for name in full_engine:
        assert _normalize(_read_json(work_inc / "engine" / name)) == _normalize(
            _read_json(work_full / "engine" / name)
        ), f"engine artifact differs: {name}"

    compared = 0
    for relative in sorted(TABLE_PRIMARY_KEY):
        inc_rows = _read_jsonl(work_inc / relative)
        full_rows = _read_jsonl(work_full / relative)
        assert len(inc_rows) == len(full_rows), f"row count differs: {relative}"
        inc_sorted = sorted((_normalize(row) for row in inc_rows), key=lambda r: _sort_key(relative, r))
        full_sorted = sorted((_normalize(row) for row in full_rows), key=lambda r: _sort_key(relative, r))
        assert inc_sorted == full_sorted, f"work table differs: {relative}"
        compared += 1
    assert compared == len(TABLE_PRIMARY_KEY)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_incremental_edit_and_append_equals_full(tmp_path: Path) -> None:
    book_v1 = tmp_path / "v1.txt"
    book_v1.write_text(BOOK_V1, encoding="utf-8")
    book_v2 = tmp_path / "v2.txt"
    book_v2.write_text(BOOK_V2, encoding="utf-8")

    engine = ChapterEngine()
    work_old = tmp_path / "work_old"
    units_old = _build_work(book_v1, work_old, engine)
    work_full = tmp_path / "work_full"
    units_new = _build_work(book_v2, work_full, engine)

    ids = [record["unit_id"] for record in units_new]
    assert ids == ["u:t:v01:c001", "u:t:v01:c002", "u:t:v01:c003"]

    plan = plan_incremental(work_old, units_new, previous_state=units_old)
    assert plan.full_run is False
    assert plan.changed_units == (ids[1], ids[2])       # edited ch2 + appended ch3
    assert plan.unchanged_units == (ids[0],)
    assert plan.extract_units == (ids[1], ids[2])
    assert plan.reuse_engine_outputs == ((ids[0], _engine_rel(ids[0])),)
    assert plan.removed_units == ()
    assert plan.dependency_edges > 0
    assert plan.counts["extract"] == 2 and plan.counts["reuse"] == 1

    # The dirty closure must reach the old records that depended on the edited
    # unit, and the replace plan must name them.
    old_assertions = _read_jsonl(work_old / "assemble" / "assertions.jsonl")
    dependent = [
        row["assertion_id"]
        for row in old_assertions
        if ids[1] in (row.get("deps") or [])
    ]
    assert dependent, "fixture lost the unit→assertion dependency"
    assert set(dependent) <= set(plan.replace["assemble/assertions.jsonl"])
    assert plan.replace["resolve/entity_candidates.jsonl"], "no dirty entity candidates"
    assert any(ids[1] in (row.get("units") or []) for row in
               _read_jsonl(work_old / "resolve" / "entity_candidates.jsonl")
               if row.get("entity_id") in plan.replace.get("resolve/entity_candidates.jsonl", ()))

    work_inc = tmp_path / "work_inc"
    _apply_incremental(work_old, work_full, work_inc, plan, units_new, engine)
    _assert_equivalent(work_inc, work_full)


def test_removed_unit_cleanup_equals_full(tmp_path: Path) -> None:
    book_3 = tmp_path / "v3ch.txt"
    book_3.write_text(VOL1 + VOL2, encoding="utf-8")
    book_2 = tmp_path / "v2ch.txt"
    book_2.write_text(VOL1, encoding="utf-8")

    engine = ChapterEngine()
    work_old = tmp_path / "work_old"
    units_old = _build_work(book_3, work_old, engine)
    work_full = tmp_path / "work_full"
    units_new = _build_work(book_2, work_full, engine)

    removed_id = "u:t:v01:c003"
    assert [record["unit_id"] for record in units_old][-1] == removed_id

    plan = plan_incremental(work_old, units_new, previous_state=units_old)
    assert plan.removed_units == (removed_id,)
    assert plan.changed_units == ()
    assert plan.extract_units == ()
    assert len(plan.reuse_engine_outputs) == 2
    assert _engine_rel(removed_id) in plan.cleanup_files
    assert "plan/texts/v01/u003.txt" in plan.cleanup_files

    old_events = _read_jsonl(work_old / "assemble" / "events.jsonl")
    removed_events = [row["event_id"] for row in old_events if removed_id in (row.get("deps") or [])]
    assert removed_events, "fixture lost the removed unit's event"
    assert set(removed_events) <= set(plan.replace["assemble/events.jsonl"])

    work_inc = tmp_path / "work_inc"
    _apply_incremental(work_old, work_full, work_inc, plan, units_new, engine)
    _assert_equivalent(work_inc, work_full)


def test_renamed_units_are_planned_as_cleanup(tmp_path: Path) -> None:
    book = tmp_path / "v1.txt"
    book.write_text(BOOK_V1, encoding="utf-8")
    engine = ChapterEngine()
    work_old = tmp_path / "work_old"
    units_old = _build_work(book, work_old, engine)

    # A renumbering re-ingest: same text (same sha256), different unit ids.
    renamed_previous = {
        "parent": "box:old",
        "units": [
            dict(record, unit_id=record["unit_id"].replace(":c00", ":c10"))
            for record in units_old
        ],
    }
    plan = plan_incremental(work_old, units_old, previous_state=renamed_previous)
    assert plan.full_run is False
    assert plan.parent == "box:old"
    assert plan.renamed_units == (
        ("u:t:v01:c101", "u:t:v01:c001"),
        ("u:t:v01:c102", "u:t:v01:c002"),
    )
    assert plan.removed_units == ()
    assert plan.changed_units == ("u:t:v01:c001", "u:t:v01:c002")
    assert plan.extract_units == ("u:t:v01:c001", "u:t:v01:c002")
    assert "engine/u_t_v01_c101.json" in plan.cleanup_files
    assert "engine/u_t_v01_c102.json" in plan.cleanup_files


def test_entity_only_change_dirties_entity_candidates(tmp_path: Path) -> None:
    book_v1 = tmp_path / "v1.txt"
    book_v1.write_text(BOOK_V1, encoding="utf-8")
    book_v2 = tmp_path / "v2.txt"
    book_v2.write_text(VOL1_EDITED, encoding="utf-8")

    engine = EntityOnlyEngine()
    work_old = tmp_path / "work_old"
    units_old = _build_work(book_v1, work_old, engine)
    work_full = tmp_path / "work_full"
    units_new = _build_work(book_v2, work_full, engine)

    plan = plan_incremental(work_old, units_new, previous_state=units_old)
    assert plan.changed_units == ("u:t:v01:c002",)
    assert plan.extract_units == ("u:t:v01:c002",)

    old_candidates = _read_jsonl(work_old / "resolve" / "entity_candidates.jsonl")
    dirty_candidates = {
        row["entity_id"]
        for row in old_candidates
        if "u:t:v01:c002" in (row.get("units") or [])
    }
    assert dirty_candidates, "fixture produced no entity candidate for the edited unit"
    replaced = set(plan.replace.get("resolve/entity_candidates.jsonl", ()))
    assert dirty_candidates <= replaced, "entity-only change did not dirty its candidates"

    work_inc = tmp_path / "work_inc"
    _apply_incremental(work_old, work_full, work_inc, plan, units_new, engine)
    _assert_equivalent(work_inc, work_full)


def test_missing_dependencies_fails_loudly(tmp_path: Path) -> None:
    book_v1 = tmp_path / "v1.txt"
    book_v1.write_text(BOOK_V1, encoding="utf-8")
    book_v2 = tmp_path / "v2.txt"
    book_v2.write_text(BOOK_V2, encoding="utf-8")
    engine = ChapterEngine()
    work_old = tmp_path / "work_old"
    units_old = _build_work(book_v1, work_old, engine)
    units_new = _build_work(book_v2, tmp_path / "work_full", engine)

    # Keep only the unit rows: no table carries a dependency edge anymore.
    scratch = tmp_path / "work_nodeps"
    (scratch / "plan").mkdir(parents=True)
    shutil.copyfile(work_old / "plan" / "units.jsonl", scratch / "plan" / "units.jsonl")

    with pytest.raises(MissingDependenciesError, match="cannot be proven"):
        plan_incremental(scratch, units_new, previous_state=units_old)

    # A previous state with a missing work root is also an explicit failure.
    with pytest.raises(MissingDependenciesError, match="work layer is missing"):
        plan_incremental(tmp_path / "does-not-exist", units_new, previous_state=units_old)

    # Without a previous state the same call is a legitimate full run.
    full = plan_incremental(scratch, units_new, previous_state=None)
    assert full.full_run is True
    assert full.extract_units == tuple(record["unit_id"] for record in units_new)
    assert full.changed_units == full.extract_units


def test_previous_state_box_and_work_root_forms(tmp_path: Path) -> None:
    book_v1 = tmp_path / "v1.txt"
    book_v1.write_text(BOOK_V1, encoding="utf-8")
    book_v2 = tmp_path / "v2.txt"
    book_v2.write_text(BOOK_V2, encoding="utf-8")
    engine = ChapterEngine()
    work_old = tmp_path / "work_old"
    units_old = _build_work(book_v1, work_old, engine)
    units_new = _build_work(book_v2, tmp_path / "work_full", engine)

    # A previous box directory: manifest.json + sources/units.jsonl.
    box = tmp_path / "box"
    (box / "sources").mkdir(parents=True)
    _write_jsonl(box / "sources" / "units.jsonl", units_old)
    _write_json(
        box / "manifest.json",
        {
            "box_id": "box:t:v1",
            "hashes": {"merkle_root": "sha256:deadbeef"},
            "incremental": {"dirty_entities": ["e:t:person:carry"]},
        },
    )
    from_box = plan_incremental(work_old, units_new, previous_state=box)
    assert from_box.full_run is False
    assert from_box.parent == "box:t:v1"
    assert from_box.previous_state_hash == "sha256:deadbeef"
    assert "e:t:person:carry" in from_box.changed_entities
    assert from_box.changed_units == ("u:t:v01:c002", "u:t:v01:c003")

    # The old work root itself is also a legal previous state.
    from_work = plan_incremental(work_old, units_new, previous_state=work_old)
    assert from_work.changed_units == from_box.changed_units
    assert from_work.unchanged_units == ("u:t:v01:c001",)

    # A bare {unit_id: sha256} mapping is normalized too.
    bare = {record["unit_id"]: record["sha256"] for record in units_old}
    from_bare = plan_incremental(work_old, units_new, previous_state=bare)
    assert from_bare.changed_units == ("u:t:v01:c002", "u:t:v01:c003")


def test_plan_is_deterministic_serializable_and_chainable(tmp_path: Path) -> None:
    book_v1 = tmp_path / "v1.txt"
    book_v1.write_text(BOOK_V1, encoding="utf-8")
    book_v2 = tmp_path / "v2.txt"
    book_v2.write_text(BOOK_V2, encoding="utf-8")
    engine = ChapterEngine()
    work_old = tmp_path / "work_old"
    units_old = _build_work(book_v1, work_old, engine)
    units_new = _build_work(book_v2, tmp_path / "work_full", engine)

    first = plan_incremental(work_old, units_new, previous_state=units_old)
    second = plan_incremental(work_old, units_new, previous_state=units_old)
    assert first.to_dict() == second.to_dict()
    assert first.previous_state_hash == plan_incremental(
        work_old, units_new, previous_state=units_old
    ).previous_state_hash
    assert json.dumps(first.to_dict(), ensure_ascii=False)  # JSON-serializable

    manifest = first.manifest_fields()
    assert set(manifest) == {"parent", "changed_units", "dirty_entities", "replay_base"}
    assert manifest["changed_units"] == list(first.changed_units)

    plan_path = first.write(tmp_path / "plan.json")
    assert _read_json(plan_path) == first.to_dict()

    # The plan itself is a valid previous state for the next round.
    chained = plan_incremental(work_old, units_new, previous_state=first.to_dict())
    assert chained.full_run is False
    assert chained.previous_state_hash == first.current_state_hash
    assert chained.changed_units == ()
    assert chained.extract_units == ("u:t:v01:c003",)  # only missing artifact in old work
    assert len(chained.reuse_engine_outputs) == 2
    assert any("artifact missing" in warning for warning in chained.warnings)

    # Full run: everything is extract, no dependency claim is made.
    full = plan_incremental(work_old, units_new, previous_state=None)
    assert full.full_run is True
    assert full.changed_units == full.extract_units
    assert len(full.extract_units) == 3
    assert full.unchanged_units == ()
    assert full.dependency_edges == 0
