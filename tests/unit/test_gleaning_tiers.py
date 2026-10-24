"""The tiered gleaning budget must actually be used, and OpenIE must not silently skip it.

Measured precisely (correcting the audit's first read)
------------------------------------------------------
B10 is **not** "the Omission step is absent": `engines/ai_reader/engine.py:371-391` implements it,
including the paper's "stop when a round adds nothing" rule. What is wrong is narrower and worth
stating exactly:

1. `config.py:21-23` declares three budgets and `config.py:67-70` resolves them per tier
   (`gleanings_for`), but `pipeline.py:182` passes **only** `cfg.max_gleanings`. So a canonical
   character gets the same budget as a walk-on, and the tiered resolver has no caller.
2. `engine.py:416-418` accepts `max_gleanings` in OpenIE mode and then notes "gleaning rounds are not
   supported (single two-stage pass)" — the budget is accepted and dropped. A caller setting 3 gets 0.

Both are "the knob is read but has no effect", which is why the configuration looked implemented.

What is pinned
--------------
* the pipeline resolves the budget per tier rather than passing one global number;
* the resolver is a function that a caller can use, and its tier mapping is the declared one;
* OpenIE mode says so in its *return value*, not only in a note nobody reads;
* `max_gleanings=0` still means zero extra passes everywhere.
"""

from __future__ import annotations

from pathlib import Path

from cbb_stage1.config import Config


def _config(tmp_path: Path, **overrides) -> Config:
    return Config(
        book_id="t",
        book_path=tmp_path / "book.txt",
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        **overrides,
    )


# ══ the tiered resolver exists and is used ═══════════════════════════
def test_the_tiered_resolver_returns_the_tier_budgets(tmp_path: Path) -> None:
    config = _config(tmp_path, max_gleanings=1, canonical_max_gleanings=2, minor_max_gleanings=0)
    assert config.for_tier("canonical") == 2
    assert config.for_tier("minor") == 0


def test_an_unknown_tier_falls_back_to_the_global_budget(tmp_path: Path) -> None:
    config = _config(tmp_path, max_gleanings=1)
    assert config.for_tier("mystery") == 1
    assert config.for_tier(None) == 1


def test_the_pipeline_resolves_per_tier_not_globally() -> None:
    """`max_gleanings=cfg.max_gleanings` ignores the tiered budgets entirely."""
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "cbb_stage1"
        / "orchestrator"
        / "pipeline.py"
    ).read_text(encoding="utf-8")
    assert "for_tier(" in source, (
        "the pipeline never calls for_tier(), so the three declared budgets have no effect and "
        "every unit gets the same number of extra passes"
    )


# ══ the budgets mean what they say ═══════════════════════════════════
def test_zero_means_zero(tmp_path: Path) -> None:
    config = _config(tmp_path, minor_max_gleanings=0)
    assert config.for_tier("minor") == 0


def test_the_defaults_are_the_documented_ones(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.for_tier("canonical") == config.canonical_max_gleanings
    assert config.for_tier("minor") == config.minor_max_gleanings


# ══ OpenIE must not accept a budget and drop it silently ═════════════
def test_openie_mode_reports_that_gleaning_did_not_run() -> None:
    """A note inside 
otes is not enough: the output must carry the effective round count."""
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "cbb_stage1"
        / "engines"
        / "ai_reader"
        / "engine.py"
    ).read_text(encoding="utf-8")
    # Split on the *definition*, not the call site: _aextract_openie also appears where it is
    # invoked, and that occurrence comes first.
    block = source.split("async def _aextract_openie")[1][:2400]
    assert "output.gleaning_round = 0" in block, (
        "OpenIE mode accepts max_gleanings and reports gleaning_round as if rounds ran; it must "
        "set the field to 0 so the artifact does not claim work that did not happen"
    )
