"""Compatibility re-exports for :mod:`cbb_stage1.ids`.

The canonical ID/SHA formulas live in :mod:`cbb_stage1.ids` (single neutral,
stdlib-only module). This module only keeps the historical
``cbb_stage1.temporal.ids`` import path for existing callers:

* :func:`stable_id` -> ``cbb_stage1.ids.stable_id`` (byte-identical: compact
  canonical JSON payload, 16-64 hex chars);
* :func:`canonical_json` -> ``cbb_stage1.ids.canonical_json``.

New code should import :mod:`cbb_stage1.ids` directly and use the named formulas
(``timeline_id`` / ``cue_id`` / ``conflict_id`` / ``quarantine_id``); ``cue:`` and
``t:`` follow those canonical formulas, while existing valid ids are preserved.
"""

from __future__ import annotations

from ..ids import canonical_json, stable_id

DEFAULT_ID_LENGTH = 16

__all__ = ["DEFAULT_ID_LENGTH", "canonical_json", "stable_id"]
