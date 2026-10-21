"""Measured hard gates for a built box.

Why this module exists
----------------------

``qc/report.json`` used to carry seven literal ``"pass"`` strings and the
manifest a literal ``"pass"``; no measurement participated, so a box with
``units > 0`` but ``assertions == 0`` / ``events == 0`` still reported 7/7 pass.

This module does not re-implement any check. It calls the canonical validator
(:func:`cbb_stage1.box.lint.canonical_lint_errors`, i.e. ``tools/box_lint.py``)
and classifies the errors it actually reports onto the seven hard-gate names
from the v2.2 spec. A gate is ``"pass"`` only when the canonical validator
reported no error in that bucket; nothing is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "GATE_NAMES",
    "GateReport",
    "classify_error",
    "measure_hard_gates",
]

#: The seven hard-gate names from ``01-箱规格-v2`` §qc.
GATE_NAMES = (
    "schema",
    "evidence",
    "references",
    "roles",
    "timeline",
    "ids",
    "hashes",
)


@dataclass(frozen=True)
class GateReport:
    """Result of one canonical lint pass, bucketed by gate name."""

    gates: dict[str, str]
    errors: tuple[str, ...]
    by_gate: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not self.errors


def classify_error(message: str) -> str:
    """Map one canonical ``box_lint`` error string onto a hard-gate name.

    The rules are ordered from most specific to most generic. Anything the
    classifier does not recognise lands in ``schema`` (the structural gate)
    rather than being dropped, so an unknown failure can never look like pass.
    """
    text = message.strip()
    if text.startswith("[x]"):
        # canonical_lint_errors keeps the CLI prefix on purpose; classify the
        # message body so ``startswith("timeline" / "coverage invariant")`` works.
        text = text[3:].strip()

    # Table-level non-vacuity checks (the new coverage invariants) are
    # reference/coverage integrity: sources exist but the referenced knowledge
    # is missing.
    if text.startswith("coverage invariant:"):
        return "references"

    lowered = text.lower()
    if (
        "checksum" in lowered
        or "merkle" in lowered
        or "sha256" in lowered
        or "content_hash" in text
    ):
        return "hashes"
    if text.startswith("timeline") or "constraint_id" in text:
        return "timeline"
    if "角色" in text or "谓词" in text or "library" in lowered:
        return "roles"
    if (
        "quote" in lowered
        or "evidence" in lowered
        or "span" in lowered
        or "原文不符" in text
        or "sentence" in lowered
    ):
        return "evidence"
    if "不存在" in text or "anchor" in lowered or "端点" in text:
        return "references"
    if "重复" in text or "counts." in text or "superseded_by" in text:
        return "ids"
    return "schema"


def measure_hard_gates(
    box: Path,
    extra_errors: Iterable[str] = (),
) -> GateReport:
    """Run the canonical validator on ``box`` and bucket its real errors.

    Args:
        box: Box directory to validate (usually the build's temp directory).
        extra_errors: Errors produced by another gate in the same build (the
            writer's JSON-schema pass). They are classified with the same rules
            so a schema failure cannot be silently overwritten by a lint pass.
    """
    # Imported lazily: ``lint`` loads ``tools/box_lint.py`` at call time, which
    # must stay a runtime concern (the project is also vendored/copied).
    from cbb_stage1.box.lint import canonical_lint_errors

    errors = [str(item) for item in extra_errors]
    errors.extend(canonical_lint_errors(Path(box), quiet=True))

    by_gate: dict[str, list[str]] = {name: [] for name in GATE_NAMES}
    for message in errors:
        by_gate[classify_error(message)].append(message)
    gates = {
        name: ("fail" if by_gate[name] else "pass")
        for name in GATE_NAMES
    }
    return GateReport(
        gates=gates,
        errors=tuple(errors),
        by_gate={name: tuple(messages) for name, messages in by_gate.items()},
    )
