"""Cluster merging must not be one LM call at a time.

`dedupe_clusters` walks the connected components and, for every multi-member cluster, asks the
merger to pick a canonical surface. That is one model call per multi-member cluster, serial.

The same reasoning as the other stages applies (A13-AFEV §4.4 "each claim can be processed
independently"; A09-CoVVE "can be run in parallel, and hence be batched"; A04-GraphRAG's map
step "independently and in parallel"), and the clusters here are even more obviously
independent: a cluster's canonical choice depends only on its own members.

Where the worker count comes from
---------------------------------
`dedupe_clusters` is a library function and knows nothing about the run's policy, so it takes
an explicit `merge_workers` keyword. The integration stage decides it with
`stage_workers(upstream, cluster_count)`. Keeping the policy at the caller is what lets this
primitive stay testable in isolation.

Order and counting
------------------
`selected`, `clusters`, `mapping` and the two counters all keep their serial values and order:
choices are computed in parallel, then applied in cluster order by the single caller.
"""

from __future__ import annotations

import threading
import time

import pytest

from cbb_stage1.dedupe.semhash import dedupe_clusters


class MergerProbe:
    """A ClusterMerger that records peak concurrency and returns a valid member."""

    def __init__(self, *, delay: float = 0.03, answer: str | None = None) -> None:
        self.delay = delay
        self.answer = answer
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.seen: list[tuple[str, ...]] = []
        self.threads: set[str] = set()

    def merge(self, cluster):
        with self.lock:
            self.in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self.in_flight)
            self.seen.append(tuple(cluster))
            self.threads.add(threading.current_thread().name)
        try:
            time.sleep(self.delay)
            return self.answer if self.answer is not None else None
        finally:
            with self.lock:
                self.in_flight -= 1


def _items(clusters: int, size: int = 2) -> list[str]:
    """`clusters` groups of near-identical strings; each group clusters together."""
    values: list[str] = []
    for index in range(clusters):
        base = f"言万心叶第{index}次出现"
        values.append(base)
        for repeat in range(size - 1):
            values.append(f"{base}　")  # near-identical, clusters with the base
    return values


# ══ the change ══════════════════════════════════════════════════════════
def test_cluster_merges_run_concurrently() -> None:
    merger = MergerProbe()
    result = dedupe_clusters(_items(8), merger=merger, merge_workers=4)
    assert result.merger_calls >= 4, f"only {result.merger_calls} clusters needed a merge"
    assert merger.peak >= 2, f"peak in-flight merges was {merger.peak}; merging is serial"


def test_one_worker_stays_serial() -> None:
    merger = MergerProbe()
    dedupe_clusters(_items(8), merger=merger, merge_workers=1)
    assert merger.peak == 1


def test_the_library_honours_the_caller_s_worker_count() -> None:
    """No hidden threshold here: the policy lives in stage_workers, at the stage.

    With two clusters and merge_workers=4 the library must actually use up to 4 — deciding
    "two is too few to bother" is the caller's job, and duplicating that rule in two places
    is how the two disagree later.
    """
    merger = MergerProbe()
    dedupe_clusters(_items(2), merger=merger, merge_workers=4)
    assert merger.calls == 2, f"expected 2 merges, got {merger.calls}"
    assert merger.peak == 2, f"two clusters with 4 workers should overlap, peak={merger.peak}"


def test_worker_threads_are_named_for_the_stage() -> None:
    merger = MergerProbe()
    dedupe_clusters(_items(8), merger=merger, merge_workers=4)
    assert any(name.startswith("cbb-dedupe") for name in merger.threads), sorted(merger.threads)


# ══ what must not change ════════════════════════════════════════════════
def test_the_result_is_identical_to_the_serial_run() -> None:
    serial = dedupe_clusters(_items(8), merger=MergerProbe(), merge_workers=1)
    parallel = dedupe_clusters(_items(8), merger=MergerProbe(), merge_workers=4)

    assert parallel.selected == serial.selected
    assert parallel.clusters == serial.clusters
    assert parallel.mapping == serial.mapping
    assert parallel.merger_calls == serial.merger_calls
    assert parallel.mergers_applied == serial.mergers_applied


def test_the_merge_choice_is_honoured() -> None:
    """A merger that picks a real member must win, in both paths."""
    values = _items(8)
    first_cluster_second_member = values[1]

    class SecondMember:
        """A merger must expose .merge; a bare callable is rejected with a TypeError."""

        def merge(self, cluster):
            return cluster[1]

    for workers in (1, 4):
        result = dedupe_clusters(values, merger=SecondMember(), merge_workers=workers)
        assert result.mergers_applied == result.merger_calls
        assert result.selected[0] == result.clusters[0][1]


def test_an_invalid_choice_still_raises() -> None:
    """Validation must not be lost when the call moves to a worker thread."""
    from cbb_stage1.dedupe.semhash import DedupeRuleError

    with pytest.raises(DedupeRuleError):
        dedupe_clusters(_items(8), merger=MergerProbe(answer="不在簇里"), merge_workers=4)


def test_no_merger_keeps_the_deterministic_path() -> None:
    result = dedupe_clusters(_items(8), merger=None, merge_workers=4)
    assert result.lm_merger == "unavailable"
    assert result.merger_calls == 0
    assert result.selected[0] == result.clusters[0][0]
