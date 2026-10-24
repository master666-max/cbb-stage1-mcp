"""Per-unit summary synthesis must run concurrently.

`derive` asks the model for a 2-4 sentence summary of every unit, one at a time. A 300k-character
book at 6-18k characters per unit is 17-50 units, so this one is small compared to `claims` —
but it is the same defect, and it sits after `claims` in the chain, so leaving it serial would
put a 50-call sequential tail on the end of every run for no reason.

`workers` is an explicit keyword: the module has no view of the run's policy, and
`synthesize_tree_texts` (the bottom-up reduction over volume/book rows) stays serial because
each level depends on the one below it — that is a real ordering constraint, not an oversight.
"""

from __future__ import annotations

import threading
import time

from cbb_stage1.derive.summarize import synthesize_unit_summaries


class SummaryProbe:
    def __init__(self, *, delay: float = 0.03, fail_on: str | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.threads: set[str] = set()

    def generate(self, prompt: str) -> str:
        with self.lock:
            self.in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self.in_flight)
            self.threads.add(threading.current_thread().name)
        try:
            time.sleep(self.delay)
            if self.fail_on and self.fail_on in prompt:
                raise RuntimeError("probe failure")
            return f"摘要：{prompt[:12]}"
        finally:
            with self.lock:
                self.in_flight -= 1


def _units(count: int) -> list[dict]:
    return [{"unit_id": f"u:t:v01:c{index:03d}"} for index in range(count)]


def _texts(count: int) -> dict[str, str]:
    return {f"u:t:v01:c{index:03d}": f"第{index}章正文内容" for index in range(count)}


# ══ the change ══════════════════════════════════════════════════════════
def test_unit_summaries_run_concurrently() -> None:
    probe = SummaryProbe()
    out = synthesize_unit_summaries(_units(8), _texts(8), probe, workers=4)
    assert len(out) == 8
    assert probe.peak >= 2, f"peak in-flight summaries was {probe.peak}; synthesis is serial"


def test_one_worker_stays_serial() -> None:
    probe = SummaryProbe()
    synthesize_unit_summaries(_units(8), _texts(8), probe, workers=1)
    assert probe.peak == 1


def test_worker_threads_are_named_for_the_stage() -> None:
    probe = SummaryProbe()
    synthesize_unit_summaries(_units(8), _texts(8), probe, workers=4)
    assert any(name.startswith("cbb-derive") for name in probe.threads), sorted(probe.threads)


# ══ what must not change ════════════════════════════════════════════════
def test_the_result_is_identical_to_the_serial_run() -> None:
    serial = synthesize_unit_summaries(_units(8), _texts(8), SummaryProbe(), workers=1)
    parallel = synthesize_unit_summaries(_units(8), _texts(8), SummaryProbe(), workers=4)
    assert parallel == serial


def test_no_llm_returns_empty() -> None:
    assert synthesize_unit_summaries(_units(4), _texts(4), None, workers=4) == {}


def test_units_without_text_are_skipped() -> None:
    texts = _texts(4)
    del texts["u:t:v01:c001"]
    probe = SummaryProbe()
    out = synthesize_unit_summaries(_units(4), texts, probe, workers=4)
    assert "u:t:v01:c001" not in out
    assert len(out) == 3
    assert probe.calls == 3


def test_one_failing_unit_does_not_lose_the_others() -> None:
    """A failed summary leaves `text=None` for that unit; it must not abort the stage."""
    probe = SummaryProbe(fail_on="第2章")
    out = synthesize_unit_summaries(_units(8), _texts(8), probe, workers=4)
    assert len(out) == 7, f"expected 7 summaries, got {len(out)}"
    assert "u:t:v01:c002" not in out
