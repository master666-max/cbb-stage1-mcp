"""The mechanism trace must distinguish "in the flow" from "merely in the tree".

The existing audit answers one question — does code exist — with a regex, and its probes are
broad (`def `), so it reported 81/81 PRESENT with zero gaps on a tree that also contained a
mechanism nobody had called yet. That is the false-green this file exists to prevent.

The experiment's question is different and needs its own evidence:

    在   does the code exist?                    -> mechanism_audit
    接线 did it actually reach the flow?          -> this tool, on a finished run

Four levels, strongest first:

    T3  produced output      a box table has rows, or a receipt counter is above zero
    T2  reachable call site   a call site exists in the pipeline
    T1  definition only
    T0  nothing

`> 0` versus `>= 0` is the whole point: a counter present and equal to zero means the
mechanism was wired but did not fire in this run, which is a different finding from "not
wired" and must not be reported as either success or absence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from mechanism_trace import (  # noqa: E402
    LEVELS,
    Trace,
    read_box_tables,
    read_receipt_counters,
    trace,
)


def _work(root: Path, counts: dict[str, object] | None = None) -> Path:
    receipts = root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "postprocess_book_08_derive.json").write_text(
        json.dumps(
            {
                "contract": "cbb-receipt/1",
                "task_id": "postprocess:book:08:derive",
                "status": "ok",
                "counts": {"outputs": 1, **(counts or {})},
            }
        ),
        encoding="utf-8",
    )
    return root


def _box(root: Path, tables: dict[str, list[dict]] | None = None) -> Path:
    for rel, rows in (tables or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
    return root


# ── reading the two evidence sources ─────────────────────────────────────
def test_box_table_row_counts_are_measured(tmp_path: Path) -> None:
    box = _box(
        tmp_path / "box",
        {
            "derived/inference/stylometry.jsonl": [{"a": 1}, {"a": 2}],
            "knowledge/assertions.jsonl": [],
        },
    )
    tables = read_box_tables(box)
    assert tables["derived/inference/stylometry.jsonl"] == 2
    assert tables["knowledge/assertions.jsonl"] == 0


def test_receipt_counters_are_summed_across_receipts(tmp_path: Path) -> None:
    work = _work(tmp_path / "work", counts={"cove_revised": 3})
    # a second receipt contributing to the same key
    (work / "receipts" / "postprocess_book_03_claims.json").write_text(
        json.dumps({"counts": {"cove_revised": 2, "abstain": 7}}), encoding="utf-8"
    )
    counters = read_receipt_counters(work)
    assert counters["cove_revised"] == 5
    assert counters["abstain"] == 7


def test_a_missing_receipts_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_receipt_counters(tmp_path / "nope") == {}


def test_a_missing_box_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_box_tables(tmp_path / "nope") == {}


def test_a_malformed_receipt_does_not_abort_the_scan(tmp_path: Path) -> None:
    work = _work(tmp_path / "work", counts={"ok_key": 1})
    (work / "receipts" / "broken.json").write_text("{not json", encoding="utf-8")
    counters = read_receipt_counters(work)
    assert counters.get("ok_key") == 1, "a broken file must be skipped, not fatal"


# ── the four levels ──────────────────────────────────────────────────────
def test_produced_output_is_t3(tmp_path: Path) -> None:
    work = _work(tmp_path / "work")
    box = _box(
        tmp_path / "box",
        {"derived/inference/stylometry.jsonl": [{"x": 1}]},
    )
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert results["D15"].wired == "T3"
    assert "stylometry" in results["D15"].evidence


def test_a_counter_above_zero_is_t3(tmp_path: Path) -> None:
    work = _work(tmp_path / "work", counts={"cove_revised": 2})
    box = _box(tmp_path / "box")
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert results["B1"].wired == "T3"


def test_a_counter_of_zero_is_not_t3(tmp_path: Path) -> None:
    """Wired but never fired. This is the finding the whole tool exists for."""
    work = _work(tmp_path / "work", counts={"cove_revised": 0})
    box = _box(tmp_path / "box")
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert results["B1"].wired != "T3", "zero is not evidence of production"
    assert results["B1"].wired in {"T0", "T1", "T2"}


def test_a_present_zero_counter_is_named_as_wired_but_silent(tmp_path: Path) -> None:
    """The middle state, stated in words: reachable, and it did not fire.

    Mutation M3 (``counters.get(key, 0.0)``) made an *absent* counter indistinguishable
    from an explicit zero, which fabricates this finding for mechanisms that were never
    even wired. The two tests below differ only in whether the key exists, so they pin
    the difference.
    """
    work = _work(tmp_path / "work", counts={"cove_revised": 0})
    box = _box(tmp_path / "box")
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert results["B1"].wired == "T2"
    assert "did not fire" in results["B1"].evidence, (
        "an explicit zero must be reported as wired-but-silent, not as silence"
    )


def test_an_absent_counter_is_not_reported_as_wired_but_silent(tmp_path: Path) -> None:
    """No key at all is not a zero. It means the evidence was never produced."""
    work = _work(tmp_path / "work")  # no cove_revised key whatsoever
    box = _box(tmp_path / "box")
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert "did not fire" not in results["B1"].evidence, (
        "a missing counter must not be dressed up as a wired-but-silent mechanism"
    )


def test_an_empty_table_is_not_t3(tmp_path: Path) -> None:
    work = _work(tmp_path / "work")
    box = _box(tmp_path / "box", {"derived/inference/themes.jsonl": []})
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert results["D15"].wired != "T3"


def test_nothing_observable_is_t0(tmp_path: Path) -> None:
    results = {t.ident: t for t in trace(work=tmp_path / "work", box=tmp_path / "box")}
    assert results["B1"].wired in LEVELS
    assert results["B1"].wired != "T3"


def test_every_declared_item_appears_exactly_once(tmp_path: Path) -> None:
    results = trace(work=tmp_path / "work", box=tmp_path / "box")
    ids = [t.ident for t in results]
    assert len(ids) == len(set(ids))


def test_the_trace_covers_the_whole_inventory(tmp_path: Path) -> None:
    """A trace that silently drops items under-reports the audit."""
    from mechanism_audit import ITEMS

    results = trace(work=tmp_path / "work", box=tmp_path / "box")
    assert {t.ident for t in results} == {i.ident for i in ITEMS}


def test_present_is_carried_from_the_audit_not_reinvented(tmp_path: Path) -> None:
    results = {t.ident: t for t in trace(work=tmp_path / "work", box=tmp_path / "box")}
    assert results["B1"].present, "the audit's presence verdict must be carried through"


def test_the_result_is_json_serialisable(tmp_path: Path) -> None:
    results = trace(work=tmp_path / "work", box=tmp_path / "box")
    payload = json.dumps([t.as_dict() for t in results], ensure_ascii=False)
    assert "wired" in payload


# ── a produced artefact that is a stand-in must say so ──────────────────
def test_a_known_proxy_is_flagged_in_the_evidence(tmp_path: Path) -> None:
    """D14's item is the 100-Endings original method, which was never implemented.

    `derived/inference/forecasts.jsonl` does exist and does reach the flow, but it holds
    lexical dispersal, not the paper's method. Grading that as unqualified production is
    the same false green this tool exists to prevent, so the artefact is reported with its
    nature attached.
    """
    work = _work(tmp_path / "work")
    box = _box(tmp_path / "box", {"derived/inference/forecasts.jsonl": [{"m": 1}]})
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert results["D14"].wired == "T3", "the proxy does reach the flow"
    assert "代理" in results["D14"].evidence, (
        "the evidence must name the artefact as a proxy, not the original method"
    )


def test_a_normal_item_carries_no_proxy_caveat(tmp_path: Path) -> None:
    work = _work(tmp_path / "work")
    box = _box(tmp_path / "box", {"derived/inference/themes.jsonl": [{"t": 1}]})
    results = {t.ident: t for t in trace(work=work, box=box)}
    assert "代理" not in results["D15"].evidence


def test_every_declared_proxy_names_a_real_item() -> None:
    from mechanism_audit import ITEMS
    from mechanism_trace import PROXY_NOTES

    ids = {i.ident for i in ITEMS}
    for ident in PROXY_NOTES:
        assert ident in ids, f"PROXY_NOTES names an unknown item: {ident}"


def test_the_limited_items_are_all_annotated() -> None:
    """Any item the audit calls LIMITED must be annotated here, or read as clean."""
    from mechanism_audit import LIMITED
    from mechanism_trace import PROXY_NOTES

    missing = [ident for ident in LIMITED if ident not in PROXY_NOTES]
    assert missing == [], f"LIMITED without a caveat: {missing}"
