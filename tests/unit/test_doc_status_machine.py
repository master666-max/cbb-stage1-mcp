"""Per-document processing state: six states, persisted, with legal transitions.

Measured gap (A22)
------------------
The project has no document status at all — a grep for `doc_status` across `src/` returned nothing, and
`ingest/registry.py` records only that a book was ingested. The source's `DocStatus` has six live states:

    pending -> parsing -> analyzing -> processing -> processed | failed

Without a status table a run that dies mid-book has no per-document record, so "resume" can only mean
"skip whatever has an engine output" — which conflates *not started*, *failed*, and *processed but
produced nothing*. Three situations, three different actions, one boolean.

What is pinned
--------------
* the closed set is the source's six (`preprocessed` is deprecated upstream and not carried);
* transitions follow the pipeline order, and an illegal move raises instead of being stored;
* a document must start at `pending`;
* `failed` can be retried, and the retry is a **new row** — the earlier failure stays visible, because it
  is evidence about the source file;
* reading the table yields the latest state per document, and groups them the way a resume needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cbb_stage1.ingest.doc_status import (
    DOC_STATES,
    TERMINAL_STATES,
    DocStatusError,
    DocStatusTable,
    can_transition,
    next_states,
    resume_work,
)

DOC = "book:/tmp/abc.epub"


# ══ the closed set ══════════════════════════════════════════════════
def test_the_six_states_are_the_sources_six() -> None:
    assert DOC_STATES == (
        "pending",
        "parsing",
        "analyzing",
        "processing",
        "processed",
        "failed",
    )


def test_the_deprecated_state_is_not_carried() -> None:
    assert "preprocessed" not in DOC_STATES, (
        "`preprocessed` is deprecated upstream; carrying it would keep a state nothing produces"
    )


def test_processed_and_failed_are_terminal_for_a_normal_run() -> None:
    assert TERMINAL_STATES == frozenset({"processed", "failed"})


# ══ transitions ═════════════════════════════════════════════════════
def test_the_pipeline_order_is_allowed() -> None:
    for current, following in (
        ("pending", "parsing"),
        ("parsing", "processing"),
        ("processing", "processed"),
    ):
        assert can_transition(current, following), f"{current} -> {following} should be legal"


def test_analyzing_may_be_skipped() -> None:
    """The source marks it optional; a text-only pipeline goes parsing -> processing."""
    assert can_transition("parsing", "processing")
    assert can_transition("parsing", "analyzing")


def test_skipping_the_work_is_rejected() -> None:
    assert not can_transition("pending", "processed"), (
        "pending -> processed would record a finished document that never ran"
    )


def test_an_unknown_state_is_not_a_transition() -> None:
    assert not can_transition("pending", "exploded")


def test_failed_can_be_retried() -> None:
    assert can_transition("failed", "pending")
    assert next_states("failed") == ("pending",)


# ══ storing ═════════════════════════════════════════════════════════
def test_a_document_must_start_at_pending(tmp_path: Path) -> None:
    table = DocStatusTable(path=tmp_path / "doc-status.jsonl")
    with pytest.raises(DocStatusError, match="must start at 'pending'"):
        table.transition(DOC, "processing")


def test_an_unknown_state_raises_on_write(tmp_path: Path) -> None:
    table = DocStatusTable(path=tmp_path / "doc-status.jsonl")
    with pytest.raises(DocStatusError, match="unknown document state"):
        table.transition(DOC, "done-ish")


def test_an_illegal_move_raises(tmp_path: Path) -> None:
    table = DocStatusTable(path=tmp_path / "doc-status.jsonl")
    table.transition(DOC, "pending")
    with pytest.raises(DocStatusError, match="cannot move"):
        table.transition(DOC, "processed")


def test_a_retry_is_a_new_row_not_an_overwrite(tmp_path: Path) -> None:
    """The earlier failure is evidence about the file; hiding it would make the file look clean."""
    table = DocStatusTable(path=tmp_path / "doc-status.jsonl")
    table.transition(DOC, "pending")
    table.transition(DOC, "parsing")
    table.transition(DOC, "failed", detail="parse error")
    table.transition(DOC, "pending")
    table.transition(DOC, "parsing")
    table.transition(DOC, "processing")
    table.transition(DOC, "processed")

    assert table.attempts_of(DOC) == 7, "the history was collapsed"
    assert table.state_of(DOC) == "processed"
    assert any(row.state == "failed" for row in table.rows), "the earlier failure disappeared"


# ══ persistence ═════════════════════════════════════════════════════
def test_the_table_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "doc-status.jsonl"
    table = DocStatusTable(path=path)
    table.transition(DOC, "pending")
    table.transition(DOC, "parsing")
    table.save()

    reloaded = DocStatusTable.load(path)
    assert reloaded.state_of(DOC) == "parsing"
    assert reloaded.attempts_of(DOC) == 2


def test_a_missing_table_is_empty(tmp_path: Path) -> None:
    assert DocStatusTable.load(tmp_path / "nope.jsonl").latest() == {}


def test_a_corrupt_line_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    path = tmp_path / "doc-status.jsonl"
    path.write_text('{"doc_id":"a","state":"pending"}\nnot json\n', encoding="utf-8")
    assert DocStatusTable.load(path).state_of("a") == "pending"


def test_by_state_groups_the_way_a_resume_needs(tmp_path: Path) -> None:
    table = DocStatusTable(path=tmp_path / "doc-status.jsonl")
    table.transition("a", "pending")
    table.transition("b", "pending")
    table.transition("b", "parsing")
    grouped = table.by_state()
    assert grouped["pending"] == ["a"]
    assert grouped["parsing"] == ["b"]


# ══ the resume view ═════════════════════════════════════════════════
def test_resume_separates_failed_from_unstarted_and_done(tmp_path: Path) -> None:
    """The whole point of six states: a boolean cannot tell these three apart."""
    table = DocStatusTable(path=tmp_path / "doc-status.jsonl")
    table.transition("fresh", "pending")
    table.transition("bad", "pending")
    table.transition("bad", "failed")
    table.transition("ok", "pending")
    table.transition("ok", "parsing")
    table.transition("ok", "processing")
    table.transition("ok", "processed")

    grouped = resume_work(table.rows)
    assert grouped["unstarted"] == ["fresh"]
    assert grouped["failed"] == ["bad"]
    assert grouped["done"] == ["ok"]
