"""The codify gate: five named conditions decide whether the cue pass runs, and the reason is recorded.

Ported from `库/code/B-项目案例/B06-AI-Reader-V2/backend/src/services/world_structure_agent.py:579-619`,
whose `_should_trigger_llm` uses exactly this shape for world-structure updates.

The conditions, in the source's order (first match wins, so the reported reason is stable):

1. ``bootstrap`` — the first five chapters always run. Cues are planted early; a gate that skips the setup
   window would look for payoffs that were never recorded.
2. ``signal`` — an explicit trigger signal is present. Event-driven, no threshold needed.
3. ``new_kind`` — a cue kind never seen before. Structural novelty: a prophecy behaves differently from an
   object, and the state machine has not seen one yet.
4. ``volume`` — two or more new candidates. One cue alone is not worth a pass; two suggests the chapter is
   doing foreshadowing work.
5. ``periodic`` — every twenty chapters, as a safety net for slow drifts no single chapter reveals.

The source returns a bare `bool`. That is the half the audit objects to: a run cannot explain afterwards why
a chapter triggered, so the F-T-P segments end up recorded but never consumed by a decision. Returning the
fired condition is what makes the gate auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BOOTSTRAP_CHAPTERS",
    "CODIFY_CONDITIONS",
    "CodifyDecision",
    "PERIODIC_INTERVAL",
    "VOLUME_THRESHOLD",
    "codify_decision",
]

#: The source's bootstrap window and periodic interval.
BOOTSTRAP_CHAPTERS = 5
PERIODIC_INTERVAL = 20

#: Two candidates suggest the chapter is doing foreshadowing work; one does not.
VOLUME_THRESHOLD = 2

#: The closed set of reasons, in priority order.
CODIFY_CONDITIONS: tuple[str, ...] = (
    "bootstrap",
    "signal",
    "new_kind",
    "volume",
    "periodic",
)


@dataclass(frozen=True)
class CodifyDecision:
    """Whether to run the cue pass for one unit, and why."""

    unit_id: str
    should_run: bool
    condition: str
    reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "should_run": self.should_run,
            "condition": self.condition,
            "reason": self.reason,
            "detail": self.detail,
        }


def _chapter_of(unit: Mapping[str, Any]) -> int:
    try:
        return int(unit.get("chapter") or 0)
    except (TypeError, ValueError):
        return 0


def _kinds(cues: Iterable[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for cue in cues or ():
        if isinstance(cue, Mapping):
            kind = str(cue.get("kind") or "").strip()
            if kind:
                out.add(kind)
    return out


def codify_decision(
    unit: Mapping[str, Any],
    *,
    new_cues: Sequence[Mapping[str, Any]] = (),
    known_kinds: Iterable[str] | None = None,
    signals: Sequence[Mapping[str, Any]] = (),
) -> CodifyDecision:
    """Decide whether the cue pass should run for ``unit``.

    ``known_kinds`` is what earlier chapters already produced; ``None`` means "unknown", in which case the
    novelty condition cannot fire — guessing novelty from an empty set would trigger it on every chapter.
    """
    unit_id = str(unit.get("unit_id") or "")
    chapter = _chapter_of(unit)
    candidates = [cue for cue in (new_cues or ()) if isinstance(cue, Mapping)]

    if chapter and chapter <= BOOTSTRAP_CHAPTERS:
        return CodifyDecision(
            unit_id=unit_id,
            should_run=True,
            condition="bootstrap",
            reason=f"chapter {chapter} is inside the bootstrap window ({BOOTSTRAP_CHAPTERS})",
        )

    trigger_signals = [
        signal
        for signal in (signals or ())
        if isinstance(signal, Mapping) and str(signal.get("type") or "") == "trigger"
    ]
    if trigger_signals:
        return CodifyDecision(
            unit_id=unit_id,
            should_run=True,
            condition="signal",
            reason="an explicit trigger signal is present",
            detail=str(trigger_signals[0].get("detail") or ""),
        )

    if known_kinds is not None:
        seen = {str(kind) for kind in known_kinds}
        fresh = _kinds(candidates) - seen
        if fresh:
            return CodifyDecision(
                unit_id=unit_id,
                should_run=True,
                condition="new_kind",
                reason="a cue kind this book has not produced before",
                detail=", ".join(sorted(fresh)),
            )

    if len(candidates) >= VOLUME_THRESHOLD:
        return CodifyDecision(
            unit_id=unit_id,
            should_run=True,
            condition="volume",
            reason=f"{len(candidates)} new cue candidates in this unit",
        )

    if chapter and chapter % PERIODIC_INTERVAL == 0:
        return CodifyDecision(
            unit_id=unit_id,
            should_run=True,
            condition="periodic",
            reason=f"periodic check every {PERIODIC_INTERVAL} chapters",
        )

    return CodifyDecision(
        unit_id=unit_id,
        should_run=False,
        condition="",
        reason=(
            f"no condition matched (chapter {chapter or '?'}, {len(candidates)} candidate(s)); "
            "the cue pass is skipped and this is recorded rather than left silent"
        ),
    )
