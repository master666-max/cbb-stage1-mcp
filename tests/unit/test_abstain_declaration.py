"""B12: the abstain gate must describe what it actually reads, and its threshold must be reachable.

Two defects in one place.

**The comment lies.** `claims/integration.py:512-516` says:

    Until now the semantic-entropy score was computed and stored but never consulted here

which reads as "it is consulted now". It is not, and it cannot be: semantic entropy is computed by
the **validate** stage, and `postprocess.py` runs `claims` before `validate`. What the gate reads is
`statement.confidence` — the engine's self-reported number (`normalize/facts.py:343-344`). So the
abstain gate is driven by the model's own opinion of itself, and the comment says otherwise. That is
the "documented but wrong" shape this whole audit exists to find.

**The threshold is a dead knob.** `claims/integration.py:446-447` reads
`upstream["abstain_threshold"]`, and nothing in `pipeline.py` ever puts it there, so it is
`DEFAULT_ABSTAIN_THRESHOLD` (0.5) in every run. A parameter nobody can set is not a parameter.

These tests bind the comment to the code and make the knob reachable from `Config`.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from cbb_stage1.claims import integration as claims_integration
from cbb_stage1.config import Config

SOURCE = Path(inspect.getsourcefile(claims_integration))
PIPELINE = Path(inspect.getsourcefile(claims_integration)).parent.parent / "orchestrator" / "pipeline.py"


def _claims_source() -> str:
    """Comment markers and wrapping removed, so a claim can be searched for as prose."""
    raw = SOURCE.read_text(encoding="utf-8")
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            stripped = stripped[1:].strip()
        lines.append(stripped)
    return " ".join(" ".join(lines).split())


# ══ the comment must not claim semantic entropy feeds the gate ═════════
def test_the_comment_does_not_claim_semantic_entropy_is_consulted() -> None:
    prose = _claims_source()
    assert "semantic-entropy score was computed and stored but never consulted here" not in prose, (
        "the comment still claims semantic entropy is consulted at this point; it is computed by "
        "validate, which runs after claims"
    )


def test_the_comment_names_the_confidence_it_reads() -> None:
    source = _claims_source()
    assert "statement.confidence" in source
    assert "self-report" in source or "self_report" in source or "engine" in source.lower(), (
        "the abstain gate must say whose confidence it is: the engine's own, not a measured score"
    )


def test_the_comment_states_the_ordering_limitation() -> None:
    source = _claims_source()
    assert re.search(r"validate", source), "the ordering fact must be written down"
    assert re.search(r"after|before|later|先行|之后", source), (
        "the comment must say which stage runs where, otherwise the reader cannot check it"
    )


# ══ the threshold must be reachable ════════════════════════════════════
def _config(tmp_path: Path, **overrides) -> Config:
    return Config(
        book_id="tst",
        book_path=tmp_path / "book.txt",
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        **overrides,
    )


def test_config_exposes_an_abstain_threshold(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert hasattr(config, "abstain_threshold"), (
        "Config has no abstain_threshold, so the only value the gate can ever see is the default"
    )


def test_the_configured_threshold_is_a_plausible_probability(tmp_path: Path) -> None:
    value = float(_config(tmp_path).abstain_threshold)
    assert 0.0 < value < 1.0, f"absurd abstain threshold: {value}"


def test_the_config_default_matches_the_code_default(tmp_path: Path) -> None:
    """Two defaults for one decision will drift; pin them."""
    from cbb_stage1.claims.verdicts import DEFAULT_ABSTAIN_THRESHOLD

    assert float(_config(tmp_path).abstain_threshold) == float(DEFAULT_ABSTAIN_THRESHOLD), (
        "Config's default disagrees with claims.verdicts.DEFAULT_ABSTAIN_THRESHOLD"
    )


def test_a_configured_threshold_is_accepted(tmp_path: Path) -> None:
    assert float(_config(tmp_path, abstain_threshold=0.7).abstain_threshold) == 0.7


def test_the_pipeline_plumbs_the_threshold_into_the_stage() -> None:
    source = PIPELINE.read_text(encoding="utf-8")
    assert "abstain_threshold" in source, (
        "the pipeline never supplies abstain_threshold, so claims always falls back to 0.5 and the "
        "parameter cannot be set by anyone"
    )


def test_the_stage_reads_it_from_upstream() -> None:
    """Belt and braces: the consumer side must keep accepting it once the producer supplies it."""
    source = _claims_source()
    assert 'upstream.get("abstain_threshold")' in source
