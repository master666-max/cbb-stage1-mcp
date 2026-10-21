"""Fact-level assertion dedupe, tested against the real recorded data.

Building synthetic assertions is brittle here: assemble resolves role values to entity
ids, and a made-up role value gets quarantined before it can be deduped. So this test
drives the recorded spk5 engine output through the postprocess chain — the case that
actually produced the duplicate — and asserts on the outcome.

The rule being pinned:

  * same predicate + roles + layer AND the same text location  -> merge (keep both quotes)
  * same predicate + roles but text far apart                  -> keep both (different events)
  * fact vs statement                                          -> keep both (§5.16)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from cbb_stage1.orchestrator.postprocess import run_postprocess

ROOT = Path(__file__).resolve().parents[2]

#: Recorded engine output from the spk5 run, which is what actually produced the
#: duplicate pair. It is 41 KB, so it is committed under `tests/data/` and travels with
#: the repository — the earlier version pinned an absolute path under a temp
#: directory, which
#: meant the test only ran on the machine that had produced it. An environment variable
#: still wins, for anyone replaying a different run.
SOURCE = Path(
    os.environ.get("CBB_RECORDED_SPK5_WORK")
    or (ROOT / "tests" / "data" / "spk5-run" / "work")
)
ENGINE = SOURCE / "engine" / "u_spk5_v01_c001.json"

def _fixture_is_for_this_test() -> bool:
    """The recorded engine output must be the book this test was written against.

    `tests/data/spk5-run` is one recording of one chapter of one book. Someone running
    this project on their own novel still has the file, so `exists()` alone would let the
    test run against data it was never meant for — passing, and proving nothing. The book
    id is checked as well as the file.
    """
    if not ENGINE.exists():
        return False
    try:
        payload = json.loads(ENGINE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # The engine output carries no top-level `book_id`; the unit id is the signal:
    # `u:spk5:v01:c001` -> book `spk5`. An explicit book_id wins if one is ever added.
    book = str(payload.get("book_id") or "")
    if not book:
        unit_id = str(payload.get("unit_id") or "")
        parts = unit_id.split(":")
        book = parts[1] if len(parts) > 2 and parts[0] == "u" else ""
    return book == "spk5"


#: Why this is a skip and not a silent `return`: a test that returns early reports PASS
#: while having verified nothing, which is the same "green gate illusion" the wiring
#: audit documented. `skip` says out loud that the fixture was missing — or belonged to
#: a different book than the one these assertions describe.
_MISSING = pytest.mark.skipif(
    not _fixture_is_for_this_test(),
    reason=(
        "the recorded spk5 fixture is absent or belongs to another book; point "
        "CBB_RECORDED_SPK5_WORK at a work directory whose engine output is book_id=spk5"
    ),
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(tmp_path: Path) -> list[dict]:
    work = tmp_path / "work"
    (work / "engine").mkdir(parents=True)
    (work / "plan").mkdir(parents=True)
    shutil.copy2(ENGINE, work / "engine" / ENGINE.name)
    for name in ("units.jsonl", "run.json"):
        source = SOURCE / "plan" / name
        if source.exists():
            shutil.copy2(source, work / "plan" / name)

    engine_output = json.loads((work / "engine" / ENGINE.name).read_text(encoding="utf-8"))
    units = _read_jsonl(work / "plan" / "units.jsonl")
    run_postprocess(
        work,
        book_id="spk5",
        outputs=[engine_output],
        units=units,
        created_at="2026-10-20T00:00:00+08:00",
        upstream={"run_id": "run:dedupe-test"},
    )
    return _read_jsonl(work / "assemble" / "assertions.jsonl")


def _grouped(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("predicate"),
            json.dumps(row.get("roles"), ensure_ascii=False, sort_keys=True),
            row.get("claim_layer"),
        )
        groups.setdefault(key, []).append(row)
    return groups


@_MISSING
def test_the_recorded_duplicate_pair_is_merged(tmp_path: Path) -> None:
    """spk5 produced one sentence twice: with and without its trailing '。'."""
    rows = _run(tmp_path)
    merged = [
        row
        for row in rows
        if row.get("predicate") == "参与" and len(row.get("evidence") or []) > 1
    ]
    assert merged, "the pair differing by a full stop should be merged into one row"
    quotes = [(ev.get("quote") or "") for ev in merged[0]["evidence"]]
    assert any(q.endswith("。") for q in quotes)
    assert any(not q.endswith("。") for q in quotes), (
        "the merged row must keep both quotes, not discard one"
    )


@_MISSING
def test_no_fact_survives_twice_at_the_same_location(tmp_path: Path) -> None:
    rows = _run(tmp_path)
    seen: dict[tuple, list[tuple]] = {}
    for row in rows:
        key = (
            row.get("predicate"),
            json.dumps(row.get("roles"), ensure_ascii=False, sort_keys=True),
            row.get("claim_layer"),
        )
        spans = tuple(
            sorted(
                (ev.get("start"), ev.get("end"))
                for ev in (row.get("evidence") or [])
                if ev.get("start") is not None
            )
        )
        seen.setdefault(key, []).append(spans)
    for key, span_sets in seen.items():
        for index, first in enumerate(span_sets):
            for second in span_sets[index + 1:]:
                for a in first:
                    for b in second:
                        close = abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2
                        assert not close, (
                            f"{key[0]} appears twice at nearly the same span "
                            f"{a} vs {b}; the dedupe missed it"
                        )


@_MISSING
def test_different_events_with_identical_roles_both_survive(tmp_path: Path) -> None:
    """The seam case: same mapped roles, text hundreds of characters apart."""
    rows = _run(tmp_path)
    grouped = _grouped(rows)
    kept_multiple = [
        key
        for key, members in grouped.items()
        if len(members) > 1 and key[0] == "参与"
    ]
    assert kept_multiple, (
        "two 参与 events with the same roles but different text must both survive"
    )
