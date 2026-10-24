"""Two capability declarations claim the code is complete when the method body is not.

Measured (B3/B5 vs `DISABLED_CAPABILITIES`)
-------------------------------------------
`tools/mcp_pipeline_server.py:1489-1506` tells an agent:

    uqlm_whitebox: why_kept = "换白盒后端即可启用，代码完整"
    uq_heads:      why_kept = "代码完整，可现场训练验证链路"

The audit found neither is complete:

* `confidence.sep` is a hand-written heuristic, not the paper's trained linear probe — no head, no
  checkpoint, no training data, no γ⋆ (B3). A whitebox backend would give it logprobs, which it
  does not use the way the paper does.
* `confidence.uq_heads` builds its features from 12 surface scalars rather than attention/logits,
  and the auto path never reads `config.uq_head_checkpoint` (B5). Training it as-is would train on
  the wrong input representation.

"代码完整" is the phrase an agent will rely on when deciding whether a missing capability is one
environment change away. Both entries therefore overstate, and that is the failure this batch is
about: not a missing mechanism, a false statement about a present one.

`narrative_forecasting` and `d14_100_endings` are left alone — they are already honest (one says
"proxy", the other "never implemented").

The declarations are read with `ast` rather than imported: `tools/mcp_pipeline_server.py` is a server
module, and a test should not boot it to read a dict.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import cbb_stage1

#: Derived from the package, never from this machine: a hardcoded path cannot run in the package.
_PACKAGE = Path(cbb_stage1.__file__).resolve()
_PROJECT = _PACKAGE.parents[2]
_DESIGN = _PACKAGE.parents[3] / "设计" / "抽取段-v6"

SERVER = _PROJECT / "tools" / "mcp_pipeline_server.py"
DECLARATION = "DISABLED_CAPABILITIES"

requires_server = pytest.mark.skipif(
    not SERVER.is_file(), reason=f"mcp pipeline server not present at {SERVER}"
)


def _declarations() -> dict:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == DECLARATION:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == DECLARATION for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{DECLARATION} not found in {SERVER.name}")


# ══ the two overstating entries ═══════════════════════════════════════
@requires_server
def test_uqlm_whitebox_does_not_claim_the_code_is_complete() -> None:
    entry = _declarations()["uqlm_whitebox"]
    assert "代码完整" not in entry["why_kept"], (
        "uqlm_whitebox still says '代码完整'; `confidence.sep` is a heuristic, not the paper's "
        "trained probe, so a whitebox backend alone does not make it the paper's method"
    )


@requires_server
def test_uqlm_whitebox_names_what_is_missing() -> None:
    entry = _declarations()["uqlm_whitebox"]
    text = entry["why_kept"] + entry.get("what_it_is_not", "")
    assert "探针" in text or "probe" in text or "训练" in text, (
        f"the entry does not say the trained probe is missing: {text!r}"
    )


@requires_server
def test_uq_heads_does_not_claim_the_code_is_complete() -> None:
    entry = _declarations()["uq_heads"]
    assert "代码完整" not in entry["why_kept"], (
        "uq_heads still says '代码完整'; its features are surface scalars rather than "
        "attention/logits, and the auto path never reads the checkpoint"
    )


@requires_server
def test_uq_heads_names_what_is_missing() -> None:
    entry = _declarations()["uq_heads"]
    text = entry["why_kept"] + entry.get("what_it_is_not", "")
    assert "特征" in text or "feature" in text or "surface" in text, (
        f"the entry does not say the input representation is substituted: {text!r}"
    )


# ══ the honest entries stay honest ════════════════════════════════════
@requires_server
def test_the_proxy_entry_still_declares_itself_a_proxy() -> None:
    entry = _declarations()["narrative_forecasting"]
    assert entry["kind"] == "proxy"
    assert "不是" in entry["what_it_is_not"]


@requires_server
def test_the_unimplemented_entry_still_says_unimplemented() -> None:
    entry = _declarations()["d14_100_endings"]
    assert entry["kind"] == "not_implemented"


# ══ the F3 inventory row is stale ════════════════════════════════════
INVENTORY = _DESIGN / "07-吸收机制总清单-20261018.md"

requires_inventory = pytest.mark.skipif(
    not INVENTORY.is_file(), reason=f"inventory not present at {INVENTORY}"
)


def _f3_row() -> str:
    for line in INVENTORY.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("| F3 "):
            return line
    raise AssertionError("F3 row not found")


@requires_inventory
def test_the_f3_row_no_longer_says_unimplemented() -> None:
    row = _f3_row()
    assert "**未实现**" not in row, (
        "F3 still reads '未实现；仅 derived/inference/ 占位', but stylometry / themes / emotion / "
        "motivation / forecasts are implemented, written to disk and copied into the box"
    )


@requires_inventory
def test_the_f3_row_names_what_is_actually_missing() -> None:
    row = _f3_row()
    text = row
    assert "冰山" in text or "粉丝理论" in text, (
        f"the row must name the parts that really are missing: {text[:200]!r}"
    )
