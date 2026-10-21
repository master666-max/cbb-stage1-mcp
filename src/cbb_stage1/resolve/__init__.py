"""Resolution primitives: aliases (Union-Find + safety table + fanout) + merges.

* :mod:`cbb_stage1.resolve.alias` -- thin wrapper over the ported AI Reader
  ``alias_resolver`` (Union-Find, three-level safety table, canonical-name
  scoring) plus the Lorebase fanout clean.  Generic / kinship / title surfaces
  can never become Union-Find keys.
* :mod:`cbb_stage1.resolve.merges` -- reversible merge ledger
  (``identity/merges.jsonl`` shape: ``kind``/``from``/``to``/``decided_by``/
  ``evidence``/``reversible``/``undo``); apply, resolve chains, undo.

Boundary: turning merges into entity-table snapshots and writing
``entities.jsonl`` is the box builder's job (P4); this package decides identity
and records the decision, it does not write the box.
"""

from __future__ import annotations

from .alias import (
    BLACKLIST,
    DEFAULT_FANOUT,
    SAFETY_BLOCKED,
    SAFETY_SAFE,
    SAFETY_SUSPICIOUS,
    AliasHost,
    FanoutFinding,
    FanoutResult,
    UnionFind,
    alias_is_safe,
    alias_safety_level,
    build_alias_map,
    build_alias_map_from_entities,
    canonical_surface,
    fanout_clean,
    filter_aliases,
    surface_hosts,
)
from .merges import (
    DecidedBy,
    MergeError,
    MergeKind,
    MergeLedger,
    MergeRecord,
    make_merge,
    merge_id_for,
)
from .integration import run

__all__ = [
    # alias
    "BLACKLIST",
    "DEFAULT_FANOUT",
    "SAFETY_BLOCKED",
    "SAFETY_SUSPICIOUS",
    "SAFETY_SAFE",
    "AliasHost",
    "FanoutFinding",
    "FanoutResult",
    "UnionFind",
    "alias_is_safe",
    "alias_safety_level",
    "filter_aliases",
    "build_alias_map",
    "build_alias_map_from_entities",
    "canonical_surface",
    "fanout_clean",
    "surface_hosts",
    # merges
    "DecidedBy",
    "MergeKind",
    "MergeError",
    "MergeRecord",
    "MergeLedger",
    "merge_id_for",
    "make_merge",
    # integration
    "run",
]
