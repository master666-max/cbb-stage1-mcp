"""The two primitives every parallel stage will share.

Why a shared module
-------------------
Four stages (claims / validate / dedupe / derive) all have the same shape: a list of
independent items, one model call each, results consumed in input order. GraphRAG solves this
with one primitive (`derive_from_rows`) instead of four ad-hoc pools, and its default
concurrency is 25 (`defaults.py:189`). Writing the same pool four times invites four
different bugs.

`stage_workers` decides how many threads an item count deserves:

    missing config          -> 1   (conservative: we do not know the caller's limits)
    item_count < 4          -> 1   (below the threshold, pool overhead > gain; mirrors
                                     pipeline.py _PARALLEL_MIN_UNITS = 4)
    otherwise               -> min(config.max_workers, 32)

`run_ordered` applies a transform to each item concurrently and returns results **in input
order**. Order preservation is the contract that lets each stage keep its existing serial
write path untouched: compute in parallel, persist in order, single writer.

The `max_concurrency` trap
--------------------------
`model_gateway` gates every model call through `BoundedSemaphore(policy.max_concurrency)`.
`cbb_start_run` used to write 2 there while `Config.max_workers` defaulted to 4, so a stage
could ask for 4 threads and only ever have 2 calls in flight. These tests pin the rule that
the caller must reconcile the two, and that we never ask for more threads than the gateway
will admit.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from cbb_stage1.orchestrator.parallel import (
    MAX_STAGE_WORKERS,
    PARALLEL_MIN_ITEMS,
    run_ordered,
    stage_workers,
)


# ══ stage_workers ═══════════════════════════════════════════════════════
def test_no_upstream_means_one_worker() -> None:
    """Without a config we do not guess a limit."""
    assert stage_workers(None, 100) == 1
    assert stage_workers({}, 100) == 1


def test_below_the_threshold_stays_serial() -> None:
    upstream = {"config": SimpleNamespace(max_workers=8)}
    assert PARALLEL_MIN_ITEMS == 4
    assert stage_workers(upstream, 1) == 1
    assert stage_workers(upstream, 3) == 1


def test_at_the_threshold_it_parallelises() -> None:
    upstream = {"config": SimpleNamespace(max_workers=8)}
    assert stage_workers(upstream, 4) == 8


def test_it_never_exceeds_the_configured_workers() -> None:
    upstream = {"config": SimpleNamespace(max_workers=2)}
    assert stage_workers(upstream, 100) == 2


def test_it_is_capped() -> None:
    upstream = {"config": SimpleNamespace(max_workers=10_000)}
    assert stage_workers(upstream, 100) == MAX_STAGE_WORKERS


def test_an_explicit_upstream_value_wins_over_config() -> None:
    """The pipeline passes `max_workers` directly so a stage need not know Config's shape."""
    upstream = {"max_workers": 6, "config": SimpleNamespace(max_workers=2)}
    assert stage_workers(upstream, 100) == 6


def test_non_positive_config_falls_back_to_serial() -> None:
    for bad in (0, -1):
        upstream = {"config": SimpleNamespace(max_workers=bad)}
        assert stage_workers(upstream, 100) == 1, f"max_workers={bad} must not spawn threads"


def test_a_string_config_is_coerced_or_ignored() -> None:
    upstream = {"config": SimpleNamespace(max_workers="8")}
    assert stage_workers(upstream, 100) == 8


def test_garbage_config_does_not_raise() -> None:
    upstream = {"config": object()}
    assert stage_workers(upstream, 100) == 1


# ══ run_ordered ═════════════════════════════════════════════════════════
def test_results_come_back_in_input_order() -> None:
    """The single-writer discipline depends on this."""
    items = list(range(20))

    def slow_first(item: int) -> int:
        time.sleep(0.01 * (20 - item))  # later items finish first
        return item * 2

    results = run_ordered(items, slow_first, workers=8, thread_name_prefix="t")
    assert results == [item * 2 for item in items]


def test_workers_one_runs_on_the_calling_thread() -> None:
    """Serial fallback must not spawn a pool at all."""
    seen: list[str] = []

    def record(item: int) -> int:
        seen.append(threading.current_thread().name)
        return item

    results = run_ordered([1, 2, 3], record, workers=1, thread_name_prefix="t")
    assert results == [1, 2, 3]
    assert seen == [threading.current_thread().name] * 3


def test_it_really_runs_concurrently() -> None:
    """The point of the change: more than one item in flight at a time."""
    barrier = threading.Barrier(4, timeout=5)
    peak = {"value": 0}
    lock = threading.Lock()

    def wait_for_peers(item: int) -> int:
        with lock:
            peak["value"] = max(peak["value"], threading.active_count())
        barrier.wait()  # deadlocks unless 4 items really are in flight
        return item

    results = run_ordered(list(range(4)), wait_for_peers, workers=4, thread_name_prefix="t")
    assert results == [0, 1, 2, 3]


def test_an_empty_list_returns_empty() -> None:
    assert run_ordered([], lambda x: x, workers=8, thread_name_prefix="t") == []


def test_an_exception_propagates() -> None:
    """A stage must not silently swallow a failure of the primitive itself."""

    def boom(item: int) -> int:
        if item == 2:
            raise RuntimeError("item 2 failed")
        return item

    with pytest.raises(RuntimeError, match="item 2 failed"):
        run_ordered([0, 1, 2, 3], boom, workers=4, thread_name_prefix="t")


def test_the_thread_name_prefix_is_used() -> None:
    """Threads must be identifiable in a dump; four stages mean four prefixes."""
    names: list[str] = []
    lock = threading.Lock()

    def record(item: int) -> int:
        with lock:
            names.append(threading.current_thread().name)
        return item

    run_ordered([0, 1, 2, 3], record, workers=4, thread_name_prefix="cbb-claims")
    assert all(name.startswith("cbb-claims") for name in names), names
