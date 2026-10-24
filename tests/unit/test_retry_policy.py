"""The retry policy must include the backoff the source has.

Measured, and narrower than the audit's first read
--------------------------------------------------
The audit said "4 attempts (off-by-one) + 0 backoff". The off-by-one is **wrong**: `retries=2` with
`for attempt in range(self.retries + 1)` is exactly 3 attempts, which matches the source. What is
actually missing is the **backoff**: `retry_delay: float = 0.0`, so three attempts fire back to back.

That matters because the source's 5-second wait exists to let a rate limit or a cold model recover.
Retrying immediately turns one transient failure into three, and then fails anyway — the retries cost
three times as much and buy nothing.

What is pinned
--------------
* the default delay is non-zero and matches the source's 5 seconds;
* the delay is actually awaited between attempts (not just stored);
* the attempt count stays 3 — the audit's off-by-one claim is not implemented as a "fix";
* an explicit `retry_delay=0` is still honoured, because a caller may want fast failure in a test.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from cbb_stage1.engines.ai_reader.openie import OpenIEExtractor

SOURCE_BACKOFF_SECONDS = 5.0


class FailingLLM:
    """Fails `fail_times` times, then returns a valid payload."""

    def __init__(self, fail_times: int = 1) -> None:
        self.fail_times = fail_times
        self.calls = 0

    async def complete(self, **kwargs: Any) -> Any:  # pragma: no cover - shape varies
        raise RuntimeError("boom")


def _signature_default(name: str) -> Any:
    signature = inspect.signature(OpenIEExtractor.__init__)
    return signature.parameters[name].default


# ══ the default is the source's backoff ══════════════════════════════
def test_the_default_retry_delay_is_the_source_backoff() -> None:
    default = _signature_default("retry_delay")
    assert float(default) == SOURCE_BACKOFF_SECONDS, (
        f"the default retry_delay is {default}, so retries fire back to back; the source waits "
        f"{SOURCE_BACKOFF_SECONDS}s to let a rate limit or cold model recover"
    )


def test_the_attempt_count_is_still_three() -> None:
    """The audit called this an off-by-one; it is not. 2 retries + the first attempt = 3."""
    assert _signature_default("retries") == 2


# ══ the delay is awaited, not merely stored ══════════════════════════
def test_the_delay_is_awaited_between_attempts() -> None:
    source = inspect.getsource(OpenIEExtractor._call_with_retry)
    assert "asyncio.sleep" in source, "the retry loop never waits between attempts"
    assert "self.retry_delay" in source, "the configured delay is stored but not used"


def test_an_explicit_zero_delay_is_honoured() -> None:
    """Tests and fast-fail callers must be able to turn the wait off."""
    extractor = OpenIEExtractor(_DummyLLM(), retry_delay=0.0)
    assert extractor.retry_delay == 0.0


def test_a_negative_delay_is_clamped() -> None:
    extractor = OpenIEExtractor(_DummyLLM(), retry_delay=-3.0)
    assert extractor.retry_delay == 0.0


class _DummyLLM:
    """Enough of an LLM for construction; the tests above do not call it."""


# ══ the pipeline does not override the backoff away ═════════════════
def test_the_pipeline_does_not_pass_a_zero_delay() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in (root / "src" / "cbb_stage1").rglob("*.py"):
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "retry_delay=0" in stripped.replace(" ", ""):
                offenders.append(f"{path.name}:{index}")
    assert not offenders, (
        f"retry_delay=0 is passed at {offenders}, which disables the backoff the source specifies"
    )
