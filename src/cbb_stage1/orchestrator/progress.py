"""Progress events - semantic 9 stages, integrity guards, interrupt gating.

Ported from B09 ``src/observability/progress.py`` (read-only source). The shape
is preserved (``ProgressStage`` / ``ProgressEvent`` / ``ProgressGuard`` /
listeners / interrupt), with two adaptations:

* no import of a repo-global config; the progress log directory is explicit;
* the 9 stage values are the v6 pipeline contract and are exported as
  ``STAGE_ORDER`` so the pipeline (and tests) can assert the shape.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

__all__ = [
    "ProgressStage",
    "ProgressEvent",
    "ProgressGuard",
    "STAGE_ORDER",
    "add_progress_listener",
    "remove_progress_listener",
    "emit_progress",
    "save_progress_log",
    "get_progress_warnings",
    "get_progress_events",
    "reset_progress",
    "request_interrupt",
    "check_interrupt",
    "clear_interrupt",
    "set_interactive_mode",
    "gate_unsafe_step",
]


# =============================================================================
# SEMANTIC STAGES - the 9-stage contract (design 08 §2.1)
# =============================================================================

class ProgressStage(Enum):
    """The 9 semantic pipeline stages kept from the B09 shell.

    Order matters: it is the observable pipeline contract.
    """

    LOADING_BOOK = "Loading book"
    ANALYZING_STRUCTURE = "Analyzing book structure"
    EXTRACTING_CHARACTERS_AND_ENTITIES = "Extracting characters and entities"
    RESOLVING_DUPLICATE_NAMES = "Resolving duplicate names"
    GENERATING_DESCRIPTIONS = "Generating descriptions"
    BUILDING_CONNECTIONS = "Building entity connections"
    WRITING_OUTPUT = "Writing output"
    SAVING_STATE = "Saving state"
    COMPLETE = "Complete"


STAGE_ORDER: tuple[ProgressStage, ...] = tuple(ProgressStage)

#: Number of stages in the pipeline contract (asserted by tests).
TOTAL_STAGES: int = len(STAGE_ORDER)


# =============================================================================
# PROGRESS EVENT
# =============================================================================

@dataclass
class ProgressEvent:
    """A single progress update with integrity guarantees.

    Attributes:
        stage: Current semantic stage.
        current: Current stage index (1-based).
        total: Total stages expected (9).
        message: Optional human-readable status message.
        timestamp: When this event was created.
        can_interrupt: Whether it is safe to stop after this event.
        approval_required: If True, human approval is needed before proceeding.
        sub_current: Sub-step index (e.g. unit within the extract stage).
        sub_total: Total sub-steps.
    """

    stage: ProgressStage
    current: int
    total: int
    message: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    can_interrupt: bool = True
    approval_required: bool = False
    sub_current: int = 0
    sub_total: int = 0

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        raw_pct = (self.current / self.total) * 100
        # Integrity: never report > 95% until actually at 100%.
        if raw_pct > 95.0 and self.current < self.total:
            return 95.0
        return raw_pct

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "stage_name": self.stage.name,
            "current": self.current,
            "total": self.total,
            "percentage": round(self.percentage, 1),
            "message": self.message,
            "can_interrupt": self.can_interrupt,
            "approval_required": self.approval_required,
            "sub_current": self.sub_current,
            "sub_total": self.sub_total,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# PROGRESS GUARD - integrity enforcement
# =============================================================================

class ProgressGuard:
    """Validate progress events and detect silent long-running steps.

    Warns when ``current > total`` (clamped) or when no event has been emitted
    for longer than ``silent_threshold_s``.
    """

    def __init__(self, silent_threshold_s: float = 60.0):
        self._last_event_time: float = time.monotonic()
        self._silent_threshold = silent_threshold_s
        self._warnings: list[str] = []

    def validate(self, event: ProgressEvent) -> ProgressEvent:
        now = time.monotonic()

        if event.current > event.total and event.total > 0:
            warning = (
                f"Progress overflow: {event.current}/{event.total} "
                f"for stage {event.stage.value}"
            )
            self._warnings.append(warning)
            print(f"  ! {warning}")
            event.current = event.total

        elapsed = now - self._last_event_time
        if elapsed > self._silent_threshold:
            warning = (
                f"Silent step detected: {elapsed:.0f}s since last progress "
                f"update (stage: {event.stage.value})"
            )
            self._warnings.append(warning)
            print(f"  ! {warning}")

        self._last_event_time = now
        return event

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)


# =============================================================================
# GLOBAL STATE (per-process; reset at the start of every pipeline run)
# =============================================================================

_progress_listeners: list[Callable[[ProgressEvent], None]] = []
_progress_guard = ProgressGuard()
_event_log: list[dict] = []


def add_progress_listener(callback: Callable[[ProgressEvent], None]) -> None:
    """Register a callback to receive progress events."""
    _progress_listeners.append(callback)


def remove_progress_listener(callback: Callable[[ProgressEvent], None]) -> None:
    """Remove a previously registered progress listener."""
    if callback in _progress_listeners:
        _progress_listeners.remove(callback)


def emit_progress(
    stage: ProgressStage,
    current: int,
    total: int,
    message: str | None = None,
    can_interrupt: bool = True,
    approval_required: bool = False,
    sub_current: int = 0,
    sub_total: int = 0,
    quiet: bool = False,
) -> ProgressEvent:
    """Emit a validated progress event, log it, and notify listeners."""
    event = ProgressEvent(
        stage=stage,
        current=current,
        total=total,
        message=message,
        can_interrupt=can_interrupt,
        approval_required=approval_required,
        sub_current=sub_current,
        sub_total=sub_total,
    )
    event = _progress_guard.validate(event)
    _event_log.append(event.to_dict())

    if not quiet:
        marker = "=" if not can_interrupt else "-"
        if approval_required:
            marker = "H"
        pct = f"{event.percentage:5.1f}%" if total > 0 else ""
        msg_part = f" - {message}" if message else ""
        count_part = f" ({current}/{total})" if total > 0 else ""
        sub_part = f" [sub: {sub_current}/{sub_total}]" if sub_total > 0 else ""
        print(f"  {marker} [{pct}] {stage.value}{msg_part}{count_part}{sub_part}")

    for listener in _progress_listeners:
        try:
            listener(event)
        except Exception:
            pass

    return event


def save_progress_log(out_dir: str | Path, trace_id: str | None = None) -> Path:
    """Save the in-memory progress event log under ``out_dir``.

    Args:
        out_dir: Directory for the progress log (work/plan/progress).
        trace_id: Optional trace id included in the file name.

    Returns:
        Path to the saved JSON log file.
    """
    log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{trace_id}" if trace_id else ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"progress{suffix}_{stamp}.json"
    log_data = {
        "events": _event_log,
        "warnings": _progress_guard.warnings,
        "saved_at": datetime.now().isoformat(),
    }
    log_path.write_text(
        json.dumps(log_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return log_path


def get_progress_warnings() -> list[str]:
    """Return integrity warnings detected during tracking."""
    return _progress_guard.warnings


def get_progress_events() -> list[dict]:
    """Return a copy of the in-memory event log."""
    return list(_event_log)


def reset_progress() -> None:
    """Reset all progress state for a new pipeline run."""
    global _progress_guard, _event_log
    _progress_guard = ProgressGuard()
    _event_log = []


# =============================================================================
# INTERRUPT HANDLING
# =============================================================================

_interrupt_requested = False


def request_interrupt() -> None:
    """Request the pipeline to stop at its next safe point."""
    global _interrupt_requested
    _interrupt_requested = True
    print("! Interrupt requested - will stop at next safe point")


def check_interrupt() -> bool:
    """Return True if ``request_interrupt()`` has been called."""
    return _interrupt_requested


def clear_interrupt() -> None:
    """Clear the interrupt flag after handling."""
    global _interrupt_requested
    _interrupt_requested = False


# =============================================================================
# APPROVAL GATING (kept from the source shell; batch mode always proceeds)
# =============================================================================

_interactive_mode = False
_approval_callback: Callable[[str, str], bool] | None = None


def set_interactive_mode(
    enabled: bool,
    approval_callback: Callable[[str, str], bool] | None = None,
) -> None:
    """Enable/disable interactive approval gating."""
    global _interactive_mode, _approval_callback
    _interactive_mode = enabled
    _approval_callback = approval_callback


def gate_unsafe_step(stage: str, description: str) -> bool:
    """Gate an unsafe step behind human approval in interactive mode.

    Returns True to proceed, False to abort. Batch mode always proceeds.
    """
    if not _interactive_mode:
        return True
    if _approval_callback:
        print(f"  H Approval required: {description}")
        return _approval_callback(stage, description)
    return True
