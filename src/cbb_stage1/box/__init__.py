"""Box v2.2 layer: single writer, atomic publish, canonical lint gate.

* :mod:`cbb_stage1.box.writer` -- :func:`build_box` reads ``work/`` and writes
  an immutable ``cbb-box/2`` directory (manifest, tables, CHECKSUMS, qc, index).
* :mod:`cbb_stage1.box.records` -- strict candidate -> box-record translation
  (span repair, predicate registry, quarantine instead of silent drops).
* :mod:`cbb_stage1.box.lexicon` -- genre/types/predicate registry (six libraries).
* :mod:`cbb_stage1.box.atomic` -- hashing + temp-dir helpers.
* :mod:`cbb_stage1.box.lint` -- thin wrapper around ``tools/box_lint.py``.
"""

from __future__ import annotations

from .atomic import (
    merkle_root,
    publish_dir,
    sha256_bytes,
    sha256_text,
)
from .lexicon import (
    SIX_LIBRARIES,
    Lexicon,
    PredicateSpec,
    RoleSpec,
    default_lexicon,
    entity_type_slug,
    load_lexicon,
)
from .lint import lint_box
from .records import (
    CLOSED_REASONS,
    Problem,
    RunMeta,
)
from .writer import BuildError, BuildResult, build_box, canonical_lint_errors

__all__ = [
    # writer
    "BuildError",
    "BuildResult",
    "build_box",
    "canonical_lint_errors",
    "lint_box",
    # records
    "CLOSED_REASONS",
    "Problem",
    "RunMeta",
    # lexicon
    "SIX_LIBRARIES",
    "Lexicon",
    "PredicateSpec",
    "RoleSpec",
    "default_lexicon",
    "entity_type_slug",
    "load_lexicon",
    # hashing
    "merkle_root",
    "publish_dir",
    "sha256_bytes",
    "sha256_text",
]
