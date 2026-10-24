"""`replay_base` is a field that is always null, so nothing can be replayed.

Measured gap (E7)
-----------------
`box/writer.py:2224` and `incremental/runner.py:660` both write `"replay_base": None`, and nothing else
in the package mentions replay except a tracing docstring. So the design's "replay from the sources and
diff" has a field and no implementation.

What replay means, and why the field is not enough
--------------------------------------------------
A replay re-derives the work tables from `sources/` **without** the engine outputs, then compares the
result with what is in the box. Its value is the difference report: which rows changed, which appeared,
which vanished — the answer to "did this rebuild actually change anything, and where".

So `replay_base` is not the feature; it is the digest a replay report points at. Writing `None` means
a reader cannot even ask.

What is pinned
--------------
* the manifest records the digest of the box a rebuild was derived from, when there was one;
* a replay produces a row-level diff (added / removed / changed) rather than a boolean;
* the diff is deterministic and comparable, so two replays of the same inputs agree;
* when there is nothing to compare against, the report says so instead of claiming a clean replay.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cbb_stage1.box.replay import diff_tables, replay_report


def _rows(*pairs: tuple[str, str, dict]) -> list[dict]:
    return [{"table": table, "key": key, **payload} for table, key, payload in pairs]


# ══ the diff is row-level ════════════════════════════════════════════
def test_an_identical_table_has_no_changes() -> None:
    rows = _rows(("assertions.jsonl", "a:1", {"confidence": 0.9}))
    diff = diff_tables({"assertions.jsonl": rows}, {"assertions.jsonl": rows})
    assert diff.added == []
    assert diff.removed == []
    assert diff.changed == []


def test_a_new_row_is_reported_as_added() -> None:
    before = _rows(("assertions.jsonl", "a:1", {"confidence": 0.9}))
    after = _rows(
        ("assertions.jsonl", "a:1", {"confidence": 0.9}),
        ("assertions.jsonl", "a:2", {"confidence": 0.7}),
    )
    diff = diff_tables({"assertions.jsonl": before}, {"assertions.jsonl": after})
    assert [(item["table"], item["key"]) for item in diff.added] == [("assertions.jsonl", "a:2")]
    assert diff.removed == []


def test_a_vanished_row_is_reported_as_removed() -> None:
    before = _rows(
        ("assertions.jsonl", "a:1", {"confidence": 0.9}),
        ("assertions.jsonl", "a:2", {"confidence": 0.7}),
    )
    after = _rows(("assertions.jsonl", "a:1", {"confidence": 0.9}))
    diff = diff_tables({"assertions.jsonl": before}, {"assertions.jsonl": after})
    assert [(item["table"], item["key"]) for item in diff.removed] == [("assertions.jsonl", "a:2")]


def test_a_changed_field_is_reported_with_both_values() -> None:
    before = _rows(("assertions.jsonl", "a:1", {"confidence": 0.9}))
    after = _rows(("assertions.jsonl", "a:1", {"confidence": 0.4}))
    diff = diff_tables({"assertions.jsonl": before}, {"assertions.jsonl": after})
    assert len(diff.changed) == 1
    change = diff.changed[0]
    assert change["key"] == "a:1"
    assert change["fields"]["confidence"] == {"before": 0.9, "after": 0.4}


# ══ determinism ═════════════════════════════════════════════════════
def test_the_diff_is_sorted() -> None:
    before: dict[str, list[dict]] = {}
    after = {
        "b.jsonl": _rows(("b.jsonl", "z", {}), ("b.jsonl", "a", {})),
        "a.jsonl": _rows(("a.jsonl", "m", {})),
    }
    diff = diff_tables(before, after)
    keys = [(item["table"], item["key"]) for item in diff.added]
    assert keys == sorted(keys), f"added rows are not sorted: {keys}"


def test_two_replays_of_the_same_input_agree() -> None:
    before = _rows(("a.jsonl", "k", {"v": 1}))
    after = _rows(("a.jsonl", "k", {"v": 2}))
    first = diff_tables({"a.jsonl": before}, {"a.jsonl": after}).as_dict()
    second = diff_tables({"a.jsonl": before}, {"a.jsonl": after}).as_dict()
    assert first == second


# ══ the report, and the honest empty case ═══════════════════════════
def test_a_report_names_the_base_it_compared_against() -> None:
    report = replay_report(
        base_id="box:aaaa",
        before={"a.jsonl": _rows(("a.jsonl", "k", {"v": 1}))},
        after={"a.jsonl": _rows(("a.jsonl", "k", {"v": 2}))},
    )
    assert report.base == "box:aaaa"
    assert report.status == "compared"


def test_a_report_without_a_base_says_so() -> None:
    """No base is not a clean replay; it is an absence of comparison."""
    report = replay_report(
        base_id=None, before={}, after={"a.jsonl": _rows(("a.jsonl", "k", {}))}
    )
    assert report.status == "no_base", (
        f"a replay without a base reported {report.status!r}, which reads as a successful comparison"
    )
    assert report.diff.added, "the rows present in the rebuild were not reported"


def test_the_report_is_json_serialisable() -> None:
    import json

    report = replay_report(base_id="box:x", before={}, after={"a.jsonl": _rows(("a.jsonl", "k", {}))})
    json.dumps(report.as_dict(), ensure_ascii=False)


# ══ the manifest must carry the base ═════════════════════════════════
def test_the_writer_does_not_hardcode_a_null_base() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src" / "cbb_stage1" / "box" / "writer.py"
    ).read_text(encoding="utf-8")
    assert '"replay_base": None,' not in source, (
        "replay_base is still hardcoded to None, so the field can never carry a digest"
    )
