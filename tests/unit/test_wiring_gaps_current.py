"""The wiring probe must report the two knobs that no code actually reads.

Measured
--------
**B5 (`uq_head_checkpoint`).** `confidence/uq_heads.py:19` documents `config.uq_head_checkpoint` as one
of its inputs, and `cli.py:173` passes it into `Config(...)` — but nothing reads the attribute. So the
auto selection can never see a checkpoint even when one is configured, and the confidence layer always
reports `unavailable`.

**A9 (`checkpoint_dir`).** `engines/ai_reader/factory.py:55` and `adapter.py:91` accept
`checkpoint_dir`, and `adapter.py:225` treats `None` as "no checkpoint". Nothing upstream ever passes
it, so the default always applies and the checkpoint is off in every run.

Both are "parameter exists, nobody supplies it" — precisely what the wiring probe is for. And both
need a staleness guard: `WIRE-04` stayed in this list for who knows how long after the gap closed, so
each entry here is pinned to a check that fails the moment the gap is fixed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import cbb_stage1

from cbb_stage1.preflight import _probe_wiring

PACKAGE = Path(cbb_stage1.__file__).resolve().parent


def _gaps() -> list[str]:
    return list((_probe_wiring().get("evidence") or {}).get("gaps") or [])


def _trees():
    """Parse every module in the package once."""
    import ast

    for path in PACKAGE.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file is a different problem
            continue
        yield path, tree


# ══ B5: nobody reads the configured checkpoint ════════════════════════
def _auto_method_ignores_the_config() -> bool:
    """True while `_auto_method` decides from kwargs only — which is the B5 gap.

    Precise on purpose: earlier versions scanned text and matched a docstring, then matched
    `confidence_method="uq_heads"` string literals in seven unrelated modules. Neither says anything
    about whether the configured checkpoint is consulted.
    """
    path = PACKAGE / "confidence" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "_auto_method":
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "resolve_checkpoint"
            ):
                return False
            if isinstance(inner, ast.Attribute) and inner.attr == "uq_head_checkpoint":
                return False
        return True
    raise AssertionError("_auto_method not found in confidence/__init__.py")


def test_the_resolver_itself_reads_the_config() -> None:
    """The gap is reachability, not correctness — the probe text must say which one."""
    import ast

    path = PACKAGE / "confidence" / "uq_heads.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "uq_head_checkpoint"
    ]
    assert reads, "resolve_checkpoint no longer reads the config field; the gap changed shape"


def test_b5_is_still_unwired() -> None:
    ignores_config = _auto_method_ignores_the_config()
    assert ignores_config, (
        "_auto_method now consults the configured checkpoint; remove the WIRE-07 probe entry "
        "instead of leaving a stale gap"
    )


def test_the_probe_reports_the_unread_checkpoint() -> None:
    joined = " ".join(_gaps())
    assert "uq_head_checkpoint" in joined, (
        f"the probe does not report that config.uq_head_checkpoint is never read: {_gaps()}"
    )


# ══ A9: nobody supplies the checkpoint dir ════════════════════════════
def _checkpoint_dir_suppliers() -> list[str]:
    """Call sites passing `checkpoint_dir=`; a function signature accepting it does not count."""
    import ast

    found: list[str] = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "checkpoint_dir":
                    found.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    return found


def test_a9_is_still_unwired() -> None:
    suppliers = _checkpoint_dir_suppliers()
    assert not suppliers, (
        f"checkpoint_dir is now supplied ({suppliers}); remove the WIRE-08 probe entry"
    )


def test_the_probe_reports_the_unsupplied_checkpoint_dir() -> None:
    joined = " ".join(_gaps())
    assert "checkpoint_dir" in joined, (
        f"the probe does not report that checkpoint_dir is never supplied: {_gaps()}"
    )


# ══ every reported gap has to be readable ═════════════════════════════
def test_every_gap_has_a_readable_description() -> None:
    for gap in _gaps():
        assert len(gap) >= 20, f"gap entry too terse to act on: {gap!r}"
        assert re.match(r"WIRE-\d+", gap), f"gap entry needs an id: {gap!r}"


def test_gap_ids_are_unique() -> None:
    ids = [gap.split()[0] for gap in _gaps()]
    assert len(ids) == len(set(ids)), f"duplicate gap ids: {ids}"
