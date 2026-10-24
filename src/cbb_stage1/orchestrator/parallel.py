"""Shared concurrency primitives for the postprocess stages.

Why this module exists
----------------------
Four stages have the same shape: a list of independent items, one model call each, results
consumed in input order. Before this module each of them ran a plain ``for`` loop, so a
5000-character chapter that needed ~220 model calls issued them strictly one at a time.

The original designs say not to do that:

* A13-AFEV §4.4 — "since no gradient computation is involved and **each claim can be
  processed independently, this process is parallel and highly scalable during inference**".
* A09-CoVVe — the factored variant answers questions as separate prompts and they "**can be
  run in parallel, and hence be batched**".
* A04-GraphRAG — the map step produces "partial answers ... **independently and in parallel**".

The reference implementation in the library does the same with one primitive:
``graphrag/index/utils/derive_from_rows.py`` applies a transform per row across
``num_threads`` workers, and ``defaults.py`` sets ``concurrent_requests: int = 25``.

Two decisions worth stating
---------------------------
**Threads, not asyncio.** The gateway already exposes an async path (``achat``) but every stage
calls the synchronous ``llm(prompt)`` callable supplied by ``upstream``. Rewriting four stages
as coroutines would change their signatures and every caller; a thread pool changes nothing
but the loop. ``ThreadPoolExecutor`` is what ``pipeline.py`` already uses for extraction
(line 869), so this matches the house pattern rather than introducing a second one.

**Compute in parallel, persist in order.** Each stage keeps its existing serial write path.
The pool returns results ordered by input position, and the stage writes them exactly as
before. That keeps the single-writer discipline that the box depends on, and it means a
parallel run and a serial run produce byte-identical artefacts — which is the regression test
that makes this change safe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

__all__ = [
    "MAX_STAGE_WORKERS",
    "PARALLEL_MIN_ITEMS",
    "run_ordered",
    "stage_workers",
]

Item = TypeVar("Item")
Result = TypeVar("Result")

#: Below this many items a pool costs more than it saves. Mirrors
#: ``pipeline.py:_PARALLEL_MIN_UNITS`` so the whole codebase uses one threshold.
PARALLEL_MIN_ITEMS = 4

#: Hard ceiling. GraphRAG defaults to 25 concurrent requests; 32 leaves room to raise it
#: without letting a misconfiguration spawn hundreds of threads against a rate-limited
#: endpoint (or against a host agent that cannot answer that many queue requests).
MAX_STAGE_WORKERS = 32


def _coerce_positive_int(value: Any) -> int | None:
    """Return ``value`` as an int when it means "spawn this many", else ``None``.

    Deliberately forgiving about the container (a dataclass, a mapping, a string number) and
    deliberately strict about the meaning: zero, negative, and unparseable are all "do not
    parallelise", never "guess something".
    """
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _configured_workers(upstream: Any) -> int | None:
    """Read the configured worker count from ``upstream`` or its ``config`` object."""
    if not isinstance(upstream, dict):
        return None

    explicit = _coerce_positive_int(upstream.get("max_workers"))
    if explicit is not None:
        return explicit

    config = upstream.get("config")
    if config is None:
        return None
    if isinstance(config, dict):
        return _coerce_positive_int(config.get("max_workers"))
    return _coerce_positive_int(getattr(config, "max_workers", None))


def stage_workers(upstream: Any, item_count: int) -> int:
    """How many threads ``item_count`` independent items deserve.

    Returns 1 whenever parallelism is not clearly wanted or not clearly safe:

    * no ``upstream`` / no configured worker count — the caller did not say, so do not guess
    * fewer than :data:`PARALLEL_MIN_ITEMS` items — pool overhead exceeds the gain
    * a non-positive or unparseable configured value

    Otherwise ``min(configured, MAX_STAGE_WORKERS)``.
    """
    if item_count < PARALLEL_MIN_ITEMS:
        return 1
    configured = _configured_workers(upstream)
    if configured is None:
        return 1
    return max(1, min(configured, MAX_STAGE_WORKERS))


def run_ordered(
    items: Sequence[Item] | Iterable[Item],
    transform: Callable[[Item], Result],
    *,
    workers: int,
    thread_name_prefix: str,
) -> list[Result]:
    """Apply ``transform`` to every item, returning results **in input order**.

    With ``workers <= 1`` the transform runs on the calling thread and no pool is created —
    the serial path stays byte-identical to the old code, which matters because most stages
    are called with one unit in tests.

    Exceptions propagate: a stage that cannot compute one item must fail loudly rather than
    silently drop it. Stages that need per-item resilience catch inside their own transform,
    where they can record the failure against the item.
    """
    ordered = list(items)
    if not ordered:
        return []

    if workers <= 1:
        return [transform(item) for item in ordered]

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix=thread_name_prefix
    ) as pool:
        futures = [pool.submit(transform, item) for item in ordered]
        # Collect strictly in submission order. `.result()` in sequence blocks on each item
        # in turn, which is what preserves the order the stages write in.
        return [future.result() for future in futures]
