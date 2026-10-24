"""A docstring that states a default must state the right one.

Measured defect
---------------
`orchestrator/collectors.py:186-192`:

    min_weight: int = 2,
    ...
    ``min_weight`` are dropped (source default: 2).

The original B09 stage it says it is porting sets `min_weight: int = 1`
(`库/code/B-项目案例/B09-BookContentGenerator/src/extraction/connections.py:42`). So the code deviates
from the source **and the comment asserts fidelity while doing it** — the worst shape of drift,
because a reader checking the comment concludes there is nothing to check.

Two things are pinned here:

1. the comment must not claim a source value it does not have;
2. the documented default must equal the actual default — so the next time someone changes one
   without the other, the test goes red instead of the comment quietly becoming a lie.

The source library path is deliberately **not** hardcoded: tests must not be bound to this
machine (the portability gate enforces that). The binding is comment-to-code, which is the part
that drifts.
"""

from __future__ import annotations

import inspect
import re

from cbb_stage1.orchestrator.collectors import build_connections

SOURCE_DEFAULT_CLAIM = re.compile(r"source default:\s*(\d+)", re.IGNORECASE)


def _doc() -> str:
    return inspect.getdoc(build_connections) or ""


def _actual_default() -> int:
    signature = inspect.signature(build_connections)
    return int(signature.parameters["min_weight"].default)


# ══ the comment must be true ═══════════════════════════════════════════
def test_the_source_default_claim_matches_the_actual_default() -> None:
    """`source default: N` must be the source's value, which the comment now records."""
    match = SOURCE_DEFAULT_CLAIM.search(_doc())
    assert match, "the docstring no longer states the source default; say what it is or say nothing"
    claimed = int(match.group(1))
    # The B09 stage being ported uses 1. The comment must say 1, and separately explain ours.
    assert claimed == 1, (
        f"the comment claims the source default is {claimed}, but B09's connections stage uses 1. "
        "A comment may document a deviation; it may not invent the source's value."
    )


def test_the_documented_deviation_matches_the_code() -> None:
    """If the code says 2, the prose must say 2 as the value in use."""
    actual = _actual_default()
    doc = _doc()
    assert str(actual) in doc, f"the docstring does not mention the value in use ({actual})"


def test_the_deviation_is_explained_rather_than_hidden() -> None:
    """A deviation needs a reason, or it is just a drift with paperwork."""
    doc = _doc().lower()
    assert "source default: 1" in doc
    assert any(word in doc for word in ("deviat", "rather", "because", "chose", "uses 2", "we use")), (
        "the docstring states a different default without saying why"
    )


# ══ the value itself is unchanged (the fix is truthfulness, not behaviour) ══
def test_the_threshold_itself_is_unchanged() -> None:
    """This change corrects the record; changing the threshold would be a separate decision."""
    assert _actual_default() == 2
