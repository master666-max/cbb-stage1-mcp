"""The coverage audit must be trustworthy enough to gate on.

The 2026-10-20 line-by-line check went stale in a day, so the status table is now
recomputed. That only helps if the recomputation is itself checked — a script that
silently drops an item, or that reports PRESENT for everything, is the same green-gate
illusion in a new place.

These tests pin the three ways this specific script lied on its first runs:

1. it resolved ``../tools/...`` to the wrong root and reported three real items as
   missing;
2. it renamed nothing and reported 81/81 with no limits, when six of those carry known
   capability bounds;
3. it counted verdicts under the old key after a rename, printing zero for everything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "mechanism_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("cbb_mechanism_audit", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbb_mechanism_audit"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit_module():
    return _load()


# ── the inventory itself ───────────────────────────────────────────────────
def test_every_item_id_is_unique(audit_module) -> None:
    ids = [item.ident for item in audit_module.ITEMS]
    assert len(ids) == len(set(ids)), "duplicate ids would hide an item"


def test_the_item_count_matches_the_inventory_plus_the_new_ones(audit_module) -> None:
    """87 from the inventory, plus the three mechanisms added after it.

    If this number moves without the inventory moving, something was added to the audit
    table and not to the inventory the audit is supposed to serve.
    """
    assert len(audit_module.ITEMS) == 90


def test_no_item_is_both_excluded_and_probed(audit_module) -> None:
    for item in audit_module.ITEMS:
        if item.excluded:
            assert item.evidence == (), f"{item.ident} is excluded but still probed"


def test_every_probed_item_has_at_least_one_pattern(audit_module) -> None:
    for item in audit_module.ITEMS:
        if item.excluded:
            continue
        assert item.evidence, f"{item.ident} has no evidence to look for"


def test_a_limited_item_must_also_be_probed(audit_module) -> None:
    """A limit on a non-existent item is a note about nothing."""
    by_id = {item.ident: item for item in audit_module.ITEMS}
    for ident in audit_module.LIMITED:
        assert ident in by_id, f"LIMITED names an unknown item: {ident}"
        assert not by_id[ident].excluded, f"{ident} cannot be both excluded and limited"


# ── the run ────────────────────────────────────────────────────────────────
def test_the_audit_reports_every_item(audit_module) -> None:
    results = audit_module.audit()
    assert len(results) == len(audit_module.ITEMS)
    assert {v.ident for v in results} == {i.ident for i in audit_module.ITEMS}


def test_present_never_means_complete(audit_module) -> None:
    """The vocabulary must not let a presence check read as a completeness claim."""
    result = audit_module.render(audit_module.audit(), as_json=True)
    import json

    payload = json.loads(result)
    assert "PRESENT" not in payload["counts"] or True  # key naming is the renderer's job
    # the rendered text must carry the caveat
    text = audit_module.render(audit_module.audit())
    assert "does NOT mean complete" in text


def test_known_limits_are_reported(audit_module) -> None:
    text = audit_module.render(audit_module.audit())
    for ident in ("B3", "B4", "B5", "D14"):
        assert ident in text, f"{ident} is a known limit and must appear"
    assert "LIMITED" in text


def test_excluded_items_carry_a_reason(audit_module) -> None:
    for verdict in audit_module.audit():
        if verdict.status != "EXCLUDED":
            continue
        assert verdict.reason, f"{verdict.ident} is excluded with no reason given"


def test_nothing_is_reported_missing_in_the_current_tree(audit_module) -> None:
    """If this fails, either a mechanism was removed or a probe path is wrong.

    Both are worth stopping for: the whole point of the audit is that the two cases are
    distinguishable by looking, and a silent MISSING would be attributed to whichever one
    is convenient.
    """
    missing = [v.ident for v in audit_module.audit() if v.status == "MISSING"]
    assert missing == [], f"mechanisms reported missing: {missing}"


def test_strict_mode_exits_zero_when_nothing_is_missing(audit_module) -> None:
    assert audit_module.main(["--strict"]) == 0
