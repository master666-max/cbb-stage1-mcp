"""Token budget auto-scaling by context window (port of B06 `infra/context_budget`).

The source read a mutable module-level ``config.CONTEXT_WINDOW_SIZE``; here the
budget is an explicit value so the engine is testable offline and never
imports application config.  Formula and anchors are unchanged:
8K -> conservative values, 128K -> generous values, linear in between.
"""

from __future__ import annotations

from dataclasses import dataclass

# Interpolation anchors (unchanged from source).
_CTX_MIN = 8192
_CTX_MAX = 131072

# Cloud/default context window used when the caller does not know better.
DEFAULT_CONTEXT_WINDOW = 131072


def _scale(ctx: int, min_val: int, max_val: int) -> int:
    t = max(0.0, min(1.0, (ctx - _CTX_MIN) / (_CTX_MAX - _CTX_MIN)))
    return int(min_val + t * (max_val - min_val))


@dataclass(frozen=True)
class TokenBudget:
    """All extraction prompt/context parameters derived from context_window."""

    context_window: int

    # chapter_fact_extractor.py
    max_chapter_len: int
    retry_len: int
    segment_enabled: bool
    extraction_num_ctx: int

    # context_summary_builder.py
    context_max_chars: int
    char_limit: int
    rel_limit: int
    loc_limit: int
    item_limit: int
    max_hierarchy_chains: int
    world_summary_chars: int


def compute_budget(context_window: int = DEFAULT_CONTEXT_WINDOW) -> TokenBudget:
    """Pure function: compute all budget parameters from ``context_window``."""
    ctx = int(context_window)
    return TokenBudget(
        context_window=ctx,
        max_chapter_len=_scale(ctx, 8000, 50000),
        retry_len=_scale(ctx, 6000, 30000),
        segment_enabled=ctx >= 32768,
        extraction_num_ctx=min(ctx, _scale(ctx, 16384, 32768)),
        context_max_chars=_scale(ctx, 6000, 18000),
        char_limit=_scale(ctx, 30, 60),
        rel_limit=_scale(ctx, 20, 40),
        loc_limit=_scale(ctx, 30, 80),
        item_limit=_scale(ctx, 15, 30),
        max_hierarchy_chains=_scale(ctx, 10, 15),
        world_summary_chars=_scale(ctx, 800, 1500),
    )


def get_budget(context_window: int = DEFAULT_CONTEXT_WINDOW) -> TokenBudget:
    """Return the budget for ``context_window`` (see :func:`compute_budget`)."""
    return compute_budget(context_window)
