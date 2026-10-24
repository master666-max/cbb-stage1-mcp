"""Semantic-hash dedupe (pure functions).

Ported from ``C01-kg-gen`` ``src/kg_gen/utils/deduplicate.py`` (MIT) and
``src/kg_gen/steps/_3_deduplicate.py`` (MIT), minus its heavy runtime
dependencies (``semhash`` / ``inflect`` / ``sentence_transformers`` / ``dspy``).

The upstream ``DeduplicateList`` does: NFKC normalize -> singularize English
tokens -> semantic hash -> self-deduplicate at a similarity threshold, keeping
a ``duplicates`` mapping so the merge can be rolled back. This module keeps the
same shape with deterministic, dependency-free pure functions:

* :func:`normalize_text` / :func:`singularize` / :func:`tokenize` -- the
  normalisation chain (NFKC + whitespace collapse + casefold, conservative
  English singularization, CJK characters kept as individual tokens);
* :func:`semhash_key` -- a content hash over the *token bag* (order-insensitive);
* :func:`similarity` -- Jaccard over token bags, so near variants that differ
  only in plural/space/case/punctuation collapse to one group;
* :func:`dedupe_items` -- greedy first-wins grouping, returning selected
  representatives **plus a reversible mapping** (``surface -> representative``);
* :func:`knn_candidates` / :func:`dedupe_clusters` -- the A12 half: KNN
  candidate pairs (token-bag Jaccard by default, cosine over real embeddings
  when an optional ``embedder`` is injected), then connected-component
  clustering with an **optional in-cluster LM merger** (injected protocol). With
  no merger the result says ``lm_merger="unavailable"`` explicitly -- no score
  or merge decision is invented. The executed method is recorded truthfully
  (``"semhash"`` or ``"embedding_knn"``), never the old misleading
  ``"knn_cluster"`` label;
* :func:`dedupe_relations` / :func:`dedupe_graph` -- apply the mapping to
  triples / entities / edges, mirroring ``run_semhash_deduplication``.

Deliberate method differences from C01 (A12 — the shape matches, the measuring does not):

* **clustering**: C01 runs KMeans into about 128 clusters; this uses KNN candidate pairs plus connected
  components. Components chain, so a long drift of near-variants can collapse into one group where KMeans
  would have split it;
* **retrieval**: C01 retrieves candidates with BM25 plus vectors, top-16; this uses token-bag Jaccard, or
  cosine when an ``embedder`` is injected, and has no BM25 at all. A rare surface form that BM25 would have
  surfaced can be missed;
* **merge output**: C01's LM step returns a ``duplicates`` set **and an alias**; this returns the chosen
  representative only. A consumer expecting an alias will get ``None``, which is why the field is named
  ``representative`` rather than ``alias``.

Deliberate boundaries:

* ``semhash`` is the deterministic default. ``lm_based`` / ``full`` require an
  injected ``embedder`` plus an in-cluster ``merger``; :func:`dedupe_graph`
  raises :class:`NotImplementedError` when those prerequisites are absent
  instead of silently degrading.
* Structural chunking from ``B07-lorebase/scripts/01_chunking.py`` is **not**
  here: it landed in :mod:`cbb_stage1.ingest.chunk` (W1 scope). This module
  only consumes the resulting text fragments.

All default-path functions are pure: no I/O, no global state, deterministic
input order. The optional embedder is the caller's injected I/O boundary.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from cbb_stage1.ids import sha256_hex_text

__all__ = [
    "DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "EMBEDDING_METHOD",
    "SEMHASH_METHOD",
    "SEMHASH_PREFIX",
    "ClusterDedupeResult",
    "ClusterMerger",
    "DedupeMethod",
    "DedupeResult",
    "DedupeRuleError",
    "GraphDedupe",
    "KnnCandidate",
    "normalize_text",
    "singularize",
    "tokenize",
    "semhash_key",
    "similarity",
    "dedupe_items",
    "knn_candidates",
    "dedupe_clusters",
    "dedupe_relations",
    "dedupe_graph",
]

#: Upstream ``_3_deduplicate.py`` default (``semhash_similarity_threshold=0.95``).
DEFAULT_SIMILARITY_THRESHOLD = 0.95

#: Cosine threshold for the *embedding* KNN path (vectors, not token bags).
#: Deliberately strict: name variants collapse, true synonyms do not.
DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD = 0.92

#: Truthful method ids written to artifacts (never ``knn_cluster``).
SEMHASH_METHOD = "semhash"
EMBEDDING_METHOD = "embedding_knn"

#: Version prefix so a future key algorithm change is visible in artifacts.
SEMHASH_PREFIX = "sh1:"

_WHITESPACE = re.compile(r"\s+")
#: Latin/digit words + individual CJK characters. Punctuation (including the
#: interpunct in ``哈利·波特``) is ignored for the token bag.
_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
#: Trailing ASCII possessive so ``stark's`` tokenises as ``stark``.
_POSSESSIVE = re.compile(r"'s\b")

#: English tokens that are already singular / mass nouns: never strip ``-s``.
_INVARIANT = frozenset(
    {
        "always",
        "analysis",
        "bus",
        "data",
        "economics",
        "famous",
        "gas",
        "glass",
        "mathematics",
        "means",
        "media",
        "news",
        "perhaps",
        "physics",
        "politics",
        "previous",
        "series",
        "species",
        "status",
        "this",
        "various",
    }
)

#: English irregular plurals (subset of ``inflect``; documented approximation).
_IRREGULAR = {
    "analyses": "analysis",
    "children": "child",
    "crises": "crisis",
    "feet": "foot",
    "geese": "goose",
    "indices": "index",
    "matrices": "matrix",
    "men": "man",
    "mice": "mouse",
    "oxen": "ox",
    "people": "person",
    "phenomena": "phenomenon",
    "teeth": "tooth",
    "vertices": "vertex",
    "women": "woman",
    # Common verb / contraction forms that merely end in ``-s``.
    "does": "do",
    "goes": "go",
    "has": "have",
    # -ves plurals whose stem is not simply ``-f``.
    "calves": "calf",
    "elves": "elf",
    "halves": "half",
    "hooves": "hoof",
    "knives": "knife",
    "leaves": "leaf",
    "lives": "life",
    "loaves": "loaf",
    "selves": "self",
    "shelves": "shelf",
    "thieves": "thief",
    "wives": "wife",
    "wolves": "wolf",
}


def normalize_text(text: str) -> str:
    """NFKC-normalise, collapse whitespace, strip, and casefold ``text``.

    Mirrors upstream ``DeduplicateList.normalize`` (NFKC) plus the extra
    whitespace/case stability needed for reproducible keys.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _singularize_token(token: str) -> str:
    """Singularize one lower-cased English token, conservatively.

    Rules are intentionally narrow: unknown words are returned unchanged, which
    matches the upstream "only singularize when it looks plural" behaviour and
    avoids mangling Chinese tokens or proper nouns.
    """
    low = token.casefold()
    if low in _IRREGULAR:
        return _IRREGULAR[low]
    if len(low) <= 3 or low in _INVARIANT:
        return token
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if low.endswith(("ches", "shes", "sses", "xes", "zes", "ses")):
        return low[:-2]
    if low.endswith("s") and not low.endswith(("ss", "us", "is")):
        return low[:-1]
    return token


def singularize(text: str) -> str:
    """Singularize the English tokens of ``text``; CJK tokens are unchanged.

    Expects (but does not require) :func:`normalize_text` input. Changed tokens
    are returned lower-cased; tokens that do not look plural keep their form.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return " ".join(_singularize_token(tok) for tok in text.split()).strip()


def tokenize(text: str) -> tuple[str, ...]:
    """Return the order-preserving token bag used for semantic hashing.

    Applies :func:`normalize_text` then :func:`singularize`, then extracts
    Latin/digit words and individual CJK characters. Returns a tuple (not a set)
    so callers can assert deterministic order; :func:`semhash_key` sorts it.
    """
    prepared = singularize(_POSSESSIVE.sub("", normalize_text(text)))
    return tuple(_TOKEN.findall(prepared))


def semhash_key(text: str) -> str:
    """Hash the order-insensitive token bag of ``text``.

    ``"New York"`` and ``"york new"`` collide by design (same bag); different
    words do not. The ``sh1:`` prefix versions the algorithm.
    """
    bag = " ".join(sorted(set(tokenize(text))))
    digest = sha256_hex_text(bag)
    return f"{SEMHASH_PREFIX}{digest}"


def _bag(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def similarity(left: str, right: str) -> float:
    """Jaccard similarity of two token bags, in ``[0.0, 1.0]``.

    Empty-vs-empty is ``1.0`` (both carry no content, so they are the same
    fragment); empty-vs-non-empty is ``0.0``.
    """
    left_bag, right_bag = _bag(left), _bag(right)
    if not left_bag and not right_bag:
        return 1.0
    union = left_bag | right_bag
    if not union:
        return 1.0
    return len(left_bag & right_bag) / len(union)


def _check_threshold(threshold: float) -> float:
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a number")
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("threshold must be in (0.0, 1.0]")
    return float(threshold)


@dataclass(frozen=True)
class DedupeResult:
    """Result of :func:`dedupe_items`.

    Attributes:
        selected: Representative surfaces, in first-seen input order.
        mapping: Every input surface -> its representative surface (a
            representative maps to itself). Reversible: callers can roll a
            merge back by iterating ``mapping`` in input order.
        clusters: One group per representative; ``clusters[i][0]`` is
            ``selected[i]`` and the rest are the merged duplicate surfaces.
        total_items: Number of input items (duplicates included).
    """

    selected: tuple[str, ...]
    mapping: dict[str, str] = field(default_factory=dict)
    clusters: tuple[tuple[str, ...], ...] = ()
    total_items: int = 0

    @property
    def deduplicated_items(self) -> int:
        return len(self.selected)

    @property
    def duplicate_items(self) -> int:
        return self.total_items - self.deduplicated_items

    @property
    def reduction(self) -> float:
        """Percentage of input items removed (upstream ``reduction`` stat)."""
        if self.total_items == 0:
            return 0.0
        return self.duplicate_items / self.total_items * 100.0

    @property
    def duplicates(self) -> dict[str, str]:
        """Only the non-identity entries of :attr:`mapping`."""
        return {item: rep for item, rep in self.mapping.items() if item != rep}

    def canonical_of(self, item: str) -> str:
        """Return the representative for ``item`` (itself when not merged)."""
        return self.mapping.get(item, item)

    def rollback(self, representative: str) -> tuple[str, ...]:
        """All surfaces merged into ``representative`` (input order)."""
        return tuple(
            item for item, rep in self.mapping.items() if rep == representative
        )

    def stats(self) -> str:
        """One-line stats, mirroring upstream ``DeduplicateList.stats``."""
        return (
            f"Total items: {self.total_items}; "
            f"Deduplicated items: {self.deduplicated_items}; "
            f"Duplicate items: {self.duplicate_items}; "
            f"Reduction: {self.reduction:.1f}%"
        )


def dedupe_items(
    items: Iterable[str],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> DedupeResult:
    """Group near-duplicate strings greedily, first occurrence wins.

    Two items merge when their :func:`semhash_key` is equal or their
    :func:`similarity` is at least ``threshold``. Each item is compared to the
    current cluster *representatives* only (no transitive union), which keeps
    the mapping deterministic and fully reversible.

    Args:
        items: Surfaces to deduplicate, in priority order (first wins).
        threshold: Jaccard threshold in ``(0, 1]``; upstream default ``0.95``.

    Returns:
        :class:`DedupeResult`; empty input yields empty results, never raises.
    """
    threshold = _check_threshold(threshold)
    selected: list[str] = []
    clusters: list[list[str]] = []
    mapping: dict[str, str] = {}
    representatives: list[tuple[frozenset[str], str]] = []
    exact: dict[str, int] = {}
    total = 0

    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"items must be str, got {type(item).__name__}")
        total += 1
        tokens = _bag(item)
        key = semhash_key(item)

        index = exact.get(key)
        if index is None:
            for candidate_index, (candidate_tokens, _) in enumerate(representatives):
                if not tokens and not candidate_tokens:
                    index = candidate_index
                    break
                union = tokens | candidate_tokens
                if union and len(tokens & candidate_tokens) / len(union) >= threshold:
                    index = candidate_index
                    break

        if index is not None:
            representative = representatives[index][1]
            mapping.setdefault(item, representative)
            clusters[index].append(item)
        else:
            index = len(selected)
            selected.append(item)
            clusters.append([item])
            representatives.append((tokens, item))
            exact[key] = index
            mapping[item] = item

    return DedupeResult(
        selected=tuple(selected),
        mapping=mapping,
        clusters=tuple(tuple(cluster) for cluster in clusters),
        total_items=total,
    )


# ── A12: KNN candidates + cluster merge (optional in-cluster LM) ──────────


class DedupeRuleError(ValueError):
    """A cluster merger broke its contract (e.g. returned a non-member surface)."""


class ClusterMerger(Protocol):
    """Optional in-cluster merger (the LM decision in upstream ``_3_deduplicate``).

    Receives one cluster (input order, more than one member) and returns the
    canonical member, or ``None`` for "no decision" (the deterministic
    first-seen member wins). Implementations must never invent a surface that is
    not in the cluster; doing so raises :class:`DedupeRuleError`.
    """

    def merge(self, cluster: Sequence[str]) -> str | None:
        ...


def _check_top_k(top_k: int | None) -> int | None:
    if top_k is None:
        return None
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an int or None")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    return top_k


@dataclass(frozen=True)
class KnnCandidate:
    """One near-duplicate candidate pair (``left_index < right_index``).

    Candidates are **hints only** (design ``00`` §4.12: "向量/近邻，仅提示");
    merging is :func:`dedupe_clusters`' job, not the candidate list's.
    """

    left: str
    right: str
    left_index: int
    right_index: int
    similarity: float

    def as_record(self) -> dict[str, object]:
        """Deterministic JSON shape (symmetric pair: ``a``/``b`` + score)."""
        return {"a": self.left, "b": self.right, "similarity": round(self.similarity, 6)}


def _coerce_embedder(embedder: Any) -> Callable[[list[str]], Any]:
    """Return a ``texts -> vectors`` callable from a callable/``.encode`` object."""
    if hasattr(embedder, "encode") and callable(embedder.encode):
        return embedder.encode
    if callable(embedder):
        return embedder
    raise TypeError("embedder must be callable or expose .encode(texts)")


def _embed_texts(embed: Callable[[list[str]], Any], texts: Sequence[str]) -> list[list[float]]:
    """Embed ``texts`` once and validate the returned vector matrix."""
    raw = embed(list(texts))
    try:
        vectors = [list(map(float, vector)) for vector in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"embedder returned non-numeric vectors: {exc}") from exc
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(texts)} texts"
        )
    widths = {len(vector) for vector in vectors}
    if len(widths) > 1:
        raise ValueError(f"embedder vectors must be equal length, got widths {sorted(widths)}")
    if widths and 0 in widths:
        raise ValueError("embedder vectors must be non-empty")
    return vectors


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 for a zero vector)."""
    if len(left) != len(right):
        raise ValueError(
            f"vector width mismatch: {len(left)} vs {len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def knn_candidates(
    items: Iterable[str],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k: int | None = None,
    embedder: Any | None = None,
) -> tuple[KnnCandidate, ...]:
    """KNN candidate pairs above ``threshold``.

    Two interchangeable similarity backends:

    * **no embedder** -- deterministic semhash token-bag Jaccard over every pair
      ``i < j`` (the pre-existing path, kept verbatim for offline runs);
    * **embedder** -- texts are embedded once (callable ``texts -> vectors`` or
      an object with ``.encode(texts)``) and candidates use cosine similarity,
      so a real embedding KNN replaces the "pure semhash stand-in".

    With ``top_k`` each left item keeps at most ``top_k`` neighbours. Output
    order is deterministic: ``(-similarity, left_index, right_index)``.
    """
    threshold = _check_threshold(threshold)
    top_k = _check_top_k(top_k)
    values = list(items)
    for item in values:
        if not isinstance(item, str):
            raise TypeError(f"items must be str, got {type(item).__name__}")
    embed: Callable[[list[str]], Any] | None = (
        _coerce_embedder(embedder) if embedder is not None else None
    )
    vectors = _embed_texts(embed, values) if embed is not None and values else []
    if embed is not None and len(vectors) != len(values):  # pragma: no cover - guarded
        raise ValueError("embedder vector count mismatch")

    rows: list[KnnCandidate] = []
    for left_index, left in enumerate(values):
        neighbours: list[tuple[float, int]] = []
        for right_index in range(left_index + 1, len(values)):
            if embed is not None:
                score = _cosine(vectors[left_index], vectors[right_index])
            else:
                score = similarity(left, values[right_index])
            if score >= threshold:
                neighbours.append((score, right_index))
        neighbours.sort(key=lambda pair: (-pair[0], pair[1]))
        if top_k is not None:
            neighbours = neighbours[:top_k]
        for score, right_index in neighbours:
            rows.append(
                KnnCandidate(
                    left=left,
                    right=values[right_index],
                    left_index=left_index,
                    right_index=right_index,
                    similarity=score,
                )
            )
    rows.sort(key=lambda row: (-row.similarity, row.left_index, row.right_index))
    return tuple(rows)


@dataclass(frozen=True)
class ClusterDedupeResult(DedupeResult):
    """Result of :func:`dedupe_clusters` (KNN graph -> connected components).

    Same reversible fields as :class:`DedupeResult` (``selected`` / ``mapping``
    / ``clusters`` / ``total_items``) plus the executed method:

    Attributes:
        method: ``"semhash"`` (token-bag Jaccard) or ``"embedding_knn"``
            (cosine over real embeddings) -- the truthful artifact value.
        lm_merger: ``"provided"`` when an in-cluster merger ran, else
            ``"unavailable"`` (**explicit** -- no merge decision is invented).
        merger_calls: Clusters with more than one member offered to the merger.
        mergers_applied: Calls where the merger returned a member (not ``None``).
        knn_pairs: Candidate pairs that fed the connected-component graph.
        candidates: The candidate pairs themselves (hints, not merges).
        notes: Human-readable caveats (e.g. the unavailable-merger note).
    """

    method: str = SEMHASH_METHOD
    lm_merger: str = "unavailable"
    merger_calls: int = 0
    mergers_applied: int = 0
    knn_pairs: int = 0
    candidates: tuple[KnnCandidate, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def reversible(self) -> bool:
        """Cluster merges keep the full mapping/cluster audit trail."""
        return True

    def stats(self) -> str:
        """One-line stats including the executed method and merger status."""
        return (
            f"Total items: {self.total_items}; "
            f"Deduplicated items: {self.deduplicated_items}; "
            f"Duplicate items: {self.duplicate_items}; "
            f"Reduction: {self.reduction:.1f}%; "
            f"Method: {self.method}; LM merger: {self.lm_merger}"
        )


def dedupe_clusters(
    items: Iterable[str],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k: int | None = None,
    merger: ClusterMerger | None = None,
    embedder: Any | None = None,
    merge_workers: int = 1,
) -> ClusterDedupeResult:
    """Cluster near-duplicates over the KNN graph (transitive union).

    Similarity comes from the semhash token bag by default, or from cosine over
    real embeddings when ``embedder`` is supplied (``method="embedding_knn"``);
    without one the deterministic path is unchanged (``method="semhash"``).
    Callers using an embedder should pass a cosine ``threshold`` (see
    :data:`DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD`).

    Unlike :func:`dedupe_items` (greedy, no transitive union), this unions
    connected components: ``a~b`` and ``b~c`` put all three in one cluster even
    when ``a`` and ``c`` are below ``threshold``. The cluster representative is
    the first-seen member unless ``merger`` (the optional LM decision) returns
    another member; without a merger the result is explicitly
    ``lm_merger="unavailable"``. The mapping/clusters stay reversible.

    ``merge_workers`` asks the merger for several clusters at once. Clusters are
    independent — a canonical choice depends only on that cluster's members — so this is a
    scheduling change only: ``selected``, ``clusters``, ``mapping`` and the two counters keep
    their serial values and their order. The caller owns the policy; this module does not
    read the run's configuration.
    """
    threshold = _check_threshold(threshold)
    top_k = _check_top_k(top_k)
    values = list(items)
    for item in values:
        if not isinstance(item, str):
            raise TypeError(f"items must be str, got {type(item).__name__}")
    if merger is not None and not callable(getattr(merger, "merge", None)):
        raise TypeError("merger must provide merge(cluster: Sequence[str]) -> str | None")
    if embedder is not None:
        _coerce_embedder(embedder)

    pairs = knn_candidates(values, threshold=threshold, top_k=top_k, embedder=embedder)
    method = EMBEDDING_METHOD if embedder is not None else SEMHASH_METHOD
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for pair in pairs:
        left_root, right_root = find(pair.left_index), find(pair.right_index)
        if left_root == right_root:
            continue
        # Keep the smaller input index as the root so cluster order stays stable.
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    grouped: dict[int, list[int]] = {}
    for index in range(len(values)):
        grouped.setdefault(find(index), []).append(index)
    index_clusters = sorted(grouped.values(), key=lambda members: members[0])

    selected: list[str] = []
    clusters: list[tuple[str, ...]] = []
    mapping: dict[str, str] = {}
    merger_calls = merger_mergers_applied = 0
    member_groups = [
        [values[index] for index in member_indices] for member_indices in index_clusters
    ]

    # One question per multi-member cluster, asked concurrently. `positions` keeps the
    # answers addressable by cluster so the application below stays in cluster order.
    positions = [
        position
        for position, members in enumerate(member_groups)
        if merger is not None and len(members) > 1
    ]
    if positions:
        from cbb_stage1.orchestrator.parallel import run_ordered

        choices = run_ordered(
            [tuple(member_groups[position]) for position in positions],
            merger.merge,
            workers=max(1, int(merge_workers)),
            thread_name_prefix="cbb-dedupe",
        )
    else:
        choices = []
    choice_at = dict(zip(positions, choices))

    for position, members in enumerate(member_groups):
        canonical = members[0]
        if merger is not None and len(members) > 1:
            merger_calls += 1
            choice = choice_at[position]
            if choice is not None:
                if not isinstance(choice, str) or choice not in members:
                    raise DedupeRuleError(
                        f"merger returned {choice!r}, expected None or a member of {members!r}"
                    )
                canonical = choice
                merger_mergers_applied += 1
        selected.append(canonical)
        clusters.append(tuple(members))
        for member in members:
            mapping[member] = canonical

    notes: list[str] = []
    if merger is None and any(len(cluster) > 1 for cluster in clusters):
        notes.append(
            "lm_merger unavailable: multi-member clusters used the first-seen member "
            "(deterministic; no LM merge decision was made)"
        )
    return ClusterDedupeResult(
        selected=tuple(selected),
        mapping=mapping,
        clusters=tuple(clusters),
        total_items=len(values),
        method=method,
        lm_merger="provided" if merger is not None else "unavailable",
        merger_calls=merger_calls,
        mergers_applied=merger_mergers_applied,
        knn_pairs=len(pairs),
        candidates=pairs,
        notes=tuple(notes),
    )


class DedupeMethod(str, Enum):
    """Dedupe modes from upstream ``_3_deduplicate.DeduplicateMethod``.

    :attr:`SEMHASH` is the deterministic token-bag path. :attr:`LM_BASED` and
    :attr:`FULL` require an injected ``embedder`` (and an in-cluster ``merger``);
    without those prerequisites :func:`dedupe_graph` raises
    :class:`NotImplementedError` rather than degrading.
    """

    SEMHASH = "semhash"
    LM_BASED = "lm_based"
    FULL = "full"


def dedupe_relations(
    relations: Iterable[Sequence[str]],
    entity_dedupe: DedupeResult | Mapping[str, str] | None = None,
    edge_dedupe: DedupeResult | Mapping[str, str] | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Rewrite ``(subject, predicate, object)`` triples through dedupe maps.

    Mirrors upstream ``run_semhash_deduplication._get_relation``: entity maps
    apply to positions 0/2, edge maps to position 1. Unknown surfaces are kept
    verbatim. Result order is first-seen; duplicate triples are collapsed.
    """
    entity_map: Mapping[str, str] = (
        entity_dedupe.mapping
        if isinstance(entity_dedupe, DedupeResult)
        else (entity_dedupe or {})
    )
    edge_map: Mapping[str, str] = (
        edge_dedupe.mapping
        if isinstance(edge_dedupe, DedupeResult)
        else (edge_dedupe or {})
    )

    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for relation in relations:
        if len(relation) != 3:
            raise ValueError(
                f"relation must have exactly 3 items, got {len(relation)}: {relation!r}"
            )
        subject, predicate, obj = (str(part) for part in relation)
        triple = (
            entity_map.get(subject, subject),
            edge_map.get(predicate, predicate),
            entity_map.get(obj, obj),
        )
        if triple not in seen:
            seen.add(triple)
            out.append(triple)
    return tuple(out)


@dataclass(frozen=True)
class GraphDedupe:
    """Deduplicated graph components (stage-1: no metadata merge).

    ``method`` records what actually ran: ``"semhash"`` (token-bag path) or
    ``"embedding_knn"`` (cosine KNN over the injected embedder).
    """

    entities: tuple[str, ...]
    edges: tuple[str, ...]
    relations: tuple[tuple[str, str, str], ...]
    entity_clusters: tuple[tuple[str, ...], ...]
    edge_clusters: tuple[tuple[str, ...], ...]
    entity_map: dict[str, str] = field(default_factory=dict)
    edge_map: dict[str, str] = field(default_factory=dict)
    method: str = SEMHASH_METHOD


def dedupe_graph(
    entities: Iterable[str],
    edges: Iterable[str],
    relations: Iterable[Sequence[str]],
    *,
    method: DedupeMethod | str = DedupeMethod.SEMHASH,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    embedding_threshold: float = DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
    embedder: Any | None = None,
    merger: ClusterMerger | None = None,
) -> GraphDedupe:
    """Deduplicate entities/edges and rewrite relations.

    With ``embedder`` (callable ``texts -> vectors`` or an object exposing
    ``.encode``) entities/edges are clustered by cosine KNN
    (``method="embedding_knn"``); without one the deterministic token-bag path
    runs unchanged (``method="semhash"``).

    Raises:
        NotImplementedError: for ``lm_based`` / ``full`` when their prerequisites
            are absent (an ``embedder`` and an in-cluster ``merger``); no mode is
            silently downgraded.
    """
    try:
        resolved = DedupeMethod(method)
    except ValueError as exc:
        raise ValueError(f"unknown dedupe method: {method!r}") from exc
    if resolved is not DedupeMethod.SEMHASH:
        if embedder is None:
            raise NotImplementedError(
                f"dedupe method {resolved.value!r} needs an embedder; pass "
                "embedder=<callable|object with .encode(texts)>"
            )
        if merger is None:
            raise NotImplementedError(
                f"dedupe method {resolved.value!r} needs an in-cluster LM merger; "
                "pass merger=<merge(cluster) -> str | None>"
            )

    if embedder is None:
        entity_dedupe = dedupe_items(entities, threshold)
        edge_dedupe = dedupe_items(edges, threshold)
        return GraphDedupe(
            entities=entity_dedupe.selected,
            edges=edge_dedupe.selected,
            relations=dedupe_relations(relations, entity_dedupe, edge_dedupe),
            entity_clusters=entity_dedupe.clusters,
            edge_clusters=edge_dedupe.clusters,
            entity_map=dict(entity_dedupe.mapping),
            edge_map=dict(edge_dedupe.mapping),
            method=SEMHASH_METHOD,
        )

    entity_dedupe = dedupe_clusters(
        entities, threshold=embedding_threshold, embedder=embedder, merger=merger
    )
    edge_dedupe = dedupe_clusters(
        edges, threshold=embedding_threshold, embedder=embedder, merger=merger
    )
    return GraphDedupe(
        entities=entity_dedupe.selected,
        edges=edge_dedupe.selected,
        relations=dedupe_relations(relations, entity_dedupe, edge_dedupe),
        entity_clusters=entity_dedupe.clusters,
        edge_clusters=edge_dedupe.clusters,
        entity_map=dict(entity_dedupe.mapping),
        edge_map=dict(edge_dedupe.mapping),
        method=EMBEDDING_METHOD,
    )
