"""Incremental layer: entity-level deps, dirty closure, and the incremental runner.

See :mod:`cbb_stage1.incremental.deps` (design ``00`` §4.16, invariant I13) and
:mod:`cbb_stage1.incremental.runner` (A20/D6 planning half). The layer is pure:
no engine calls, no writes to the work root; the caller (pipeline shell) applies
the returned :class:`~cbb_stage1.incremental.runner.IncrementalPlan`.
"""

from __future__ import annotations

from .deps import (
    DependencyGraph,
    DirtyClosure,
    build_dependency_graph,
    dirty_closure,
    entity_deps,
)
from .runner import (
    IncrementalPlan,
    MissingDependenciesError,
    PreviousState,
    load_previous_state,
    plan_incremental,
    unit_fingerprint,
    work_engine_path,
)

__all__ = [
    "DependencyGraph",
    "DirtyClosure",
    "IncrementalPlan",
    "MissingDependenciesError",
    "PreviousState",
    "build_dependency_graph",
    "dirty_closure",
    "entity_deps",
    "load_previous_state",
    "plan_incremental",
    "unit_fingerprint",
    "work_engine_path",
]
