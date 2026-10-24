"""A run must be pausable, not only interruptible.

Measured gap (A5)
-----------------
`orchestrator/progress.py:281-299` has exactly one control flag:

    request_interrupt() / check_interrupt() / clear_interrupt()

and `pipeline.py:831` stops at the next unit safe point when it is set. There is no pause: the only
"stop" is terminal, so a run that needs to wait — for a human decision, for a rate limit window, for a
long transcription — has to be killed and started over, paying again for every unit already extracted.

The difference is not cosmetic:

* interrupt = "do not continue this run"; the work stops and stays stopped;
* pause = "do not proceed *yet*"; the work stops at a safe point and resumes when released.

Conflating them is why the pipeline can only be stopped by killing it, which this session had to do
twice.

What is pinned
--------------
* pause and interrupt are separate flags — setting one must not imply the other;
* a paused run's safe-point check reports "wait", distinct from "stop";
* releasing a pause lets the run continue from the same place;
* an interrupt still wins over a pause, so a run that is both must stop.
"""

from __future__ import annotations

import pytest

from cbb_stage1.orchestrator import progress


@pytest.fixture(autouse=True)
def _clean_state():
    """Every test starts and ends with no control flag set."""
    progress.clear_interrupt()
    progress.resume()
    yield
    progress.clear_interrupt()
    progress.resume()


# ══ the two controls are separate ════════════════════════════════════
def test_pause_does_not_request_an_interrupt() -> None:
    progress.pause()
    assert progress.check_pause() is True
    assert progress.check_interrupt() is False, (
        "pausing also set the interrupt flag, so a paused run cannot be resumed"
    )


def test_interrupt_does_not_pause() -> None:
    progress.request_interrupt()
    assert progress.check_interrupt() is True
    assert progress.check_pause() is False


def test_resume_releases_only_the_pause() -> None:
    progress.pause()
    progress.request_interrupt()
    progress.resume()
    assert progress.check_pause() is False
    assert progress.check_interrupt() is True, "resume cleared the interrupt as well"


# ══ the safe-point decision is three-valued ═════════════════════════
def test_a_clear_run_says_continue() -> None:
    assert progress.safe_point_action() == "continue"


def test_a_paused_run_says_wait() -> None:
    progress.pause()
    assert progress.safe_point_action() == "wait", (
        "a paused run must report 'wait', not 'stop': the work is not abandoned"
    )


def test_an_interrupted_run_says_stop() -> None:
    progress.request_interrupt()
    assert progress.safe_point_action() == "stop"


def test_an_interrupt_wins_over_a_pause() -> None:
    progress.pause()
    progress.request_interrupt()
    assert progress.safe_point_action() == "stop", (
        "a run that is both paused and interrupted must stop, not wait forever"
    )


# ══ pausing is idempotent and inspectable ═══════════════════════════
def test_pausing_twice_is_harmless() -> None:
    progress.pause()
    progress.pause()
    assert progress.check_pause() is True


def test_resuming_an_unpaused_run_is_harmless() -> None:
    progress.resume()
    assert progress.check_pause() is False


def test_the_state_is_reported_for_status() -> None:
    progress.pause()
    state = progress.control_state()
    assert state == {"paused": True, "interrupt": False}, f"unexpected control state: {state}"


# ══ the pipeline must consult it ════════════════════════════════════
def test_the_pipeline_consults_the_safe_point_action() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "cbb_stage1"
        / "orchestrator"
        / "pipeline.py"
    ).read_text(encoding="utf-8")
    assert "safe_point_action" in source, (
        "the pipeline only checks `check_interrupt()`, so a paused run would keep working"
    )
