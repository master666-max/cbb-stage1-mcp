"""Verification strength must follow the entity tier, and the tier must actually be found.

Measured defect
---------------
`validate/integration.py:162-185` (`_entity_tiers`) reads only `work/identity/entities.jsonl`,
which is a **box-side** table: `box/writer.py:2271` writes it while building the box. Nothing in
the work layer ever creates it, so in a real run `_entity_tiers` returns `{}` and
`_assertion_tier` falls back to `"minor"` for every assertion. §5.15 says tier decides how hard an
assertion is verified — with everything at `minor`, that decision stops existing.

The fix follows the pattern already used one function below: `_speaker_cast` (line 330-352) reads
`identity/entities.jsonl` **and** `resolve/entity_candidates.jsonl`, because the resolve stage does
write the latter (`resolve/integration.py:372-374` writes `entity_id` and `tier`).

Precedence is deliberate: the box-side table describes the built entity (after merges), so when
both exist the box side wins.
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.validate.integration import _assertion_tier, _entity_tiers

CANONICAL = "e:tst:person:aaaaaaaaaaaa"
MAJOR = "e:tst:person:bbbbbbbbbbbb"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


# ══ the work-layer table must be read ══════════════════════════════════
def test_tiers_come_from_the_resolve_table_when_identity_is_absent(tmp_path: Path) -> None:
    """A real run has no `identity/` until the box is built."""
    work = tmp_path / "work"
    _write(
        work / "resolve" / "entity_candidates.jsonl",
        [
            {"entity_id": CANONICAL, "canonical_name": "言万", "tier": "canonical"},
            {"entity_id": MAJOR, "canonical_name": "拉法", "tier": "major"},
        ],
    )
    assert not (work / "identity" / "entities.jsonl").exists()

    tiers = _entity_tiers(work)
    assert tiers.get(CANONICAL) == "canonical", (
        f"the resolve table was not read; tiers={tiers!r}. Every assertion would be verified at "
        "'minor' because the box-side table does not exist until the box is built."
    )
    assert tiers.get(MAJOR) == "major"


def test_the_box_table_wins_when_both_exist(tmp_path: Path) -> None:
    """Merges happen after resolve, so the box side describes the final entity."""
    work = tmp_path / "work"
    _write(
        work / "resolve" / "entity_candidates.jsonl",
        [{"entity_id": CANONICAL, "tier": "major"}],
    )
    _write(
        work / "identity" / "entities.jsonl",
        [{"entity_id": CANONICAL, "tier": "canonical"}],
    )
    assert _entity_tiers(work).get(CANONICAL) == "canonical"


def test_an_unknown_tier_value_falls_back_to_minor(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _write(
        work / "resolve" / "entity_candidates.jsonl",
        [{"entity_id": CANONICAL, "tier": "protagonist"}],
    )
    assert _entity_tiers(work).get(CANONICAL) == "minor"


def test_a_missing_table_yields_no_tiers(tmp_path: Path) -> None:
    assert _entity_tiers(tmp_path / "work") == {}


# ══ the consequence: assertion tier is not always minor ════════════════
def test_an_assertion_about_a_canonical_entity_is_not_minor(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _write(
        work / "resolve" / "entity_candidates.jsonl",
        [{"entity_id": CANONICAL, "canonical_name": "言万", "tier": "canonical"}],
    )
    tiers = _entity_tiers(work)
    assertion = {
        "assertion_id": "a:1111111111111111",
        "roles": {"主体": "言万", "值": "炼气"},
        "qualifiers": {},
    }
    # `_assertion_tier` resolves names through the same tier map when given entity ids; here the
    # roles carry surfaces, so this asserts the lookup contract it relies on.
    assert tiers, "no tiers at all means the decision cannot be made"
    assert _assertion_tier({"roles": {"主体": CANONICAL}}, tiers) == "canonical"
