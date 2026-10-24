"""A reversible merge must carry the information that makes it reversible.

Measured defect
---------------
`resolve/merges.py:103-104` builds the box record with a `undo` payload — the ledger's own module
docstring says it "keeps enough state to undo a merge". Three things then remove it:

* `box/records.py:1224-1235` rebuilds the row field by field and does not copy `undo`;
* `merge.schema.json` declares `additionalProperties: false` and has no `undo` property, so even if
  the value survived, the lint would reject the row;
* `reversible` is a constant `true` — so the box asserts reversibility while carrying nothing to
  reverse it with.

The design (§5.5 / E3) asks for the merge log to be a reversible operation log, "split 一等公民".
A row that says `reversible: true` and has no undo payload is a claim, not a log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_stage1 import schemas
from cbb_stage1.box.records import normalize_merge

ENTITY_FROM = "e:tst:person:aaaaaaaaaaaa"
ENTITY_TO = "e:tst:person:bbbbbbbbbbbb"

UNDO = {
    "kind": "unmerge",
    "from": [ENTITY_TO],
    "to": [ENTITY_FROM],
    "restored_aliases": ["心叶"],
}


def _row(**overrides) -> dict:
    row = {
        "merge_id": "mg:1111111111111111",
        "book_id": "tst",
        "kind": "alias",
        "from": [ENTITY_FROM],
        "to": ENTITY_TO,
        "decided_by": "rule",
        "confidence": 0.9,
        "evidence": [{"unit_id": "u:tst:v01:c001", "quote": "言万"}],
        "at": "2026-10-23T03:00:00+08:00",
        "reversible": True,
        "undo": dict(UNDO),
    }
    row.update(overrides)
    return row


# ══ the payload must survive the box projection ════════════════════════
def test_normalize_merge_keeps_the_undo_payload() -> None:
    record, problems = normalize_merge(
        _row(), book_id="tst", entity_ids={ENTITY_FROM, ENTITY_TO}
    )
    assert not problems, problems
    assert record is not None
    assert record.get("undo") == UNDO, (
        "the undo payload was dropped while projecting the merge into box shape"
    )


def test_a_merge_without_undo_still_normalizes() -> None:
    """Rows written before the payload existed must keep working."""
    row = _row()
    del row["undo"]
    record, problems = normalize_merge(
        row, book_id="tst", entity_ids={ENTITY_FROM, ENTITY_TO}
    )
    assert not problems, problems
    assert record is not None
    assert "undo" not in record or record.get("undo") is None


# ══ the schema must allow what the producer writes ═════════════════════
def test_the_merge_schema_allows_undo() -> None:
    directory = next((p for p in schemas.candidate_dirs() if p.is_dir()), None)
    assert directory is not None
    schema = json.loads((directory / "merge.schema.json").read_text(encoding="utf-8"))
    assert "undo" in (schema.get("properties") or {}), (
        "merge.schema.json has additionalProperties:false and no `undo`, so a reversible merge "
        "record cannot be written"
    )


def test_a_merge_row_with_undo_validates() -> None:
    import jsonschema

    directory = next((p for p in schemas.candidate_dirs() if p.is_dir()), None)
    assert directory is not None
    schema = json.loads((directory / "merge.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(_row(), schema)


def test_a_merge_row_without_undo_still_validates() -> None:
    import jsonschema

    directory = next((p for p in schemas.candidate_dirs() if p.is_dir()), None)
    assert directory is not None
    schema = json.loads((directory / "merge.schema.json").read_text(encoding="utf-8"))
    row = _row()
    del row["undo"]
    jsonschema.validate(row, schema)
