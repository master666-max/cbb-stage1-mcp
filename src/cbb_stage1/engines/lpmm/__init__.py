"""LPMM (two-stage NER -> RDF) extraction adapter.

Public entry point: :class:`LpmmEngine` (Engine Protocol).  See
:mod:`cbb_stage1.engines.lpmm.adapter` for the ported behaviour (thread pool,
retry, tolerant JSON repair, per-paragraph checkpoint/resume).
"""

from cbb_stage1.engines.lpmm.adapter import LpmmAdapter, LpmmEngine, LpmmSettings

__all__ = ["LpmmAdapter", "LpmmEngine", "LpmmSettings"]
