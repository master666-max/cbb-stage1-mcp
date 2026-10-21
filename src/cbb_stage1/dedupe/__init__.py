"""Dedupe/enrich primitives for stage 1 (pure functions, no LLM).

Two source ports land here:

* :mod:`cbb_stage1.dedupe.semhash` -- ``C01-kg-gen`` (MIT)
  ``utils/deduplicate.py`` + ``steps/_3_deduplicate.py``: NFKC/singularize
  normalisation, semantic-hash grouping at a similarity threshold, reversible
  ``duplicates`` mapping, and relation rewriting. ``semhash`` is the
  deterministic default; an optional injected ``embedder`` enables cosine KNN
  (``method="embedding_knn"``), and ``lm_based``/``full`` raise
  ``NotImplementedError`` when their embedder/merger prerequisites are absent.
* :mod:`cbb_stage1.dedupe.describe` -- exact description-fragment dedupe plus
  the ``describe()`` interface skeleton. It returns the deduplicated
  observation list and never calls a model; LLM synthesis is P4
  (``enrich-merge``).

Boundary: structural chunking from ``B07-lorebase/scripts/01_chunking.py``
(Apache-2.0) is **not** duplicated here. It was ported by W1 into
:mod:`cbb_stage1.ingest.chunk`; this package only consumes text fragments.
"""

from __future__ import annotations

from .integration import run as run_dedupe
from .describe import (
    FRAGMENT_HASH_PREFIX,
    DescribeResult,
    DescriptionSynthesizer,
    FragmentDedupeResult,
    Observation,
    dedupe_description_fragments,
    dedupe_fragments,
    describe,
    fragment_key,
    merge_observations,
    normalize_fragment,
    render_observations,
)
from .semhash import (
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_METHOD,
    SEMHASH_METHOD,
    SEMHASH_PREFIX,
    DedupeMethod,
    DedupeResult,
    GraphDedupe,
    dedupe_graph,
    dedupe_items,
    dedupe_relations,
    normalize_text,
    semhash_key,
    similarity,
    singularize,
    tokenize,
)

__all__ = [
    # semhash
    "DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "EMBEDDING_METHOD",
    "SEMHASH_METHOD",
    "SEMHASH_PREFIX",
    "DedupeMethod",
    "DedupeResult",
    "GraphDedupe",
    "normalize_text",
    "singularize",
    "tokenize",
    "semhash_key",
    "similarity",
    "dedupe_items",
    "dedupe_relations",
    "dedupe_graph",
    # describe
    "FRAGMENT_HASH_PREFIX",
    "Observation",
    "FragmentDedupeResult",
    "DescribeResult",
    "DescriptionSynthesizer",
    "normalize_fragment",
    "fragment_key",
    "dedupe_fragments",
    "dedupe_description_fragments",
    "merge_observations",
    "describe",
    "render_observations",
    # postprocess integration (P0-D)
    "run_dedupe",
]
