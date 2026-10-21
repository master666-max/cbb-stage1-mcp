"""B3 — SEPs: semantic entropy from white-box signals (hidden states / token log-probs).

SEPs (semantic entropy proxies) approximate the meaning-space entropy without N
full generations — the design lists it as 5–10× cheaper but requiring white-box
access.  This adapter accepts either:

* ``hidden_states``: one vector per generation (or one per token span); vectors are
  greedily cosine-clustered and the cluster entropy gives confidence.  This is the
  SEP shape proper.
* ``logprobs``: the chosen-token log-probabilities of one generation; confidence is
  the geometric-mean token probability ``exp(mean(logprob))`` — a cheap white-box
  proxy, labelled as such in ``details["approximation"]``.

Without white-box access the method is **unavailable** (never a fabricated score):
no ``llm`` ⇔ ``unavailable``; black-box-only ``llm`` ⇔ ``unavailable``; an async
white-box client ⇔ ``unavailable`` (this adapter is synchronous; the caller must
pass plain ``logprobs``/``hidden_states`` from its own loop).

``method`` is reported as ``semantic_entropy`` because §5.15 has no separate
``sep`` value in its closed enum; ``details["approximation"]="sep"`` preserves the
mechanism name (B3) for the audit trail.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Sequence

from . import STATUS_OK, ConfidenceResult, as_mapping
from .semantic_entropy import _discard_awaitable, build_prompt, claim_text

__all__ = [
    "cluster_vectors",
    "cosine",
    "logprob_confidence",
    "score",
]

_EPS = 1e-12


# ── vectors ────────────────────────────────────────────────────────────────
def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("向量维度不一致：%d vs %d" % (len(a), len(b)))
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return dot / (na * nb)


def cluster_vectors(vectors: Sequence[Sequence[float]], threshold: float = 0.8) -> list[list[int]]:
    """Greedy cosine clustering; deterministic in input order (returns index lists)."""
    clusters: list[list[int]] = []
    for idx, vec in enumerate(vectors):
        for cluster in clusters:
            if cosine(vec, vectors[cluster[0]]) >= threshold:
                cluster.append(idx)
                break
        else:
            clusters.append([idx])
    return clusters


def logprob_confidence(logprobs: Sequence[float]) -> dict[str, Any]:
    """Geometric-mean token probability of one generation, clipped to [0,1]."""
    values = [float(x) for x in logprobs]
    if not values:
        raise ValueError("logprobs 为空：没有 token 概率可用")
    if any(v > 1e-6 for v in values):
        raise ValueError("logprobs 必须是 <=0 的对数概率：%r" % (values[:3],))
    mean_lp = sum(values) / len(values)
    confidence = max(0.0, min(1.0, math.exp(mean_lp)))
    return {
        "tokens": len(values),
        "mean_logprob": mean_lp,
        "perplexity": math.exp(-mean_lp),
        "confidence": confidence,
    }


def _confidence_from_sizes(sizes: Sequence[int]) -> dict[str, Any]:
    total = sum(sizes)
    if total < 1:
        raise ValueError("空簇")
    probs = [size / total for size in sizes]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(total) if total > 1 else 0.0
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    return {
        "clusters": len(sizes),
        "cluster_sizes": list(sizes),
        "entropy_nats": entropy,
        "entropy_normalized": normalised,
        "confidence": max(0.0, min(1.0, 1.0 - normalised)),
    }


def _as_vectors(raw: Any) -> list[list[float]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("hidden_states 必须是向量的序列")
    vectors: list[list[float]] = []
    for row in raw:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            raise ValueError("hidden_states 的每一项必须是非空数值序列：%r" % (row,))
        vectors.append([float(x) for x in row])
    return vectors


def _as_logprobs(raw: Any) -> list[float]:
    if hasattr(raw, "tolist") and not isinstance(raw, (list, tuple)):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("logprobs 必须是数值序列")
    out: list[float] = []
    for item in raw:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            # per-token distribution: take the chosen (max) log-prob
            out.append(max(float(x) for x in item))
        else:
            out.append(float(item))
    return out


# ── white-box client ───────────────────────────────────────────────────────
def _whitebox_from_llm(llm: Any, record: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Pull hidden states or log-probs from a white-box client (sync only)."""
    prompt = None
    if hasattr(llm, "hidden_states") or hasattr(llm, "token_logprobs") or hasattr(llm, "logprobs"):
        text = claim_text(record)
        if not text:
            return None, "记录没有可比较的主张文本（白盒探针没有输入）"
        prompt = build_prompt(record)
    if hasattr(llm, "hidden_states"):
        raw = llm.hidden_states(prompt)
        if inspect.isawaitable(raw):
            _discard_awaitable(raw)
            return None, "白盒接口异步：本同步适配器不驱动协程（请直接传 hidden_states）"
        return {"hidden_states": raw, "source": "llm.hidden_states"}, None
    getter = getattr(llm, "token_logprobs", None) or getattr(llm, "logprobs", None)
    if getter is not None:
        raw = getter(prompt)
        if inspect.isawaitable(raw):
            _discard_awaitable(raw)
            return None, "白盒接口异步：本同步适配器不驱动协程（请直接传 logprobs）"
        return {"logprobs": raw, "source": "llm.token_logprobs"}, None
    return None, "llm 不带白盒接口（无 hidden_states/token_logprobs）"


def score(
    record: Any,
    llm: Any = None,
    *,
    logprobs: Sequence[float] | None = None,
    hidden_states: Sequence[Sequence[float]] | None = None,
    similarity_threshold: float = 0.8,
    **_: Any,
) -> ConfidenceResult:
    """SEP score from explicit white-box inputs or a white-box ``llm``."""
    rec = as_mapping(record)
    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise ValueError("similarity_threshold 必须在 [0,1]：%r" % (similarity_threshold,))
    source = "explicit"
    if hidden_states is None and logprobs is None:
        if llm is None:
            return ConfidenceResult.unavailable(
                "SEPs 需要白盒访问（hidden_states 或 token_logprobs）；既没给输入也没有 llm —— 不可用，不伪造分数",
                method="semantic_entropy",
                approximation="sep",
            )
        pulled, why = _whitebox_from_llm(llm, rec)
        if pulled is None:
            return ConfidenceResult.unavailable(
                why or "白盒信号不可用",
                method="semantic_entropy",
                approximation="sep",
            )
        source = pulled.pop("source", "llm")
        hidden_states = pulled.get("hidden_states")
        logprobs = pulled.get("logprobs")

    if hidden_states is not None:
        vectors = _as_vectors(hidden_states)
        if len(vectors) < 2:
            return ConfidenceResult.unavailable(
                "hidden_states 只有 %d 条：语义熵需要 >=2 条" % len(vectors),
                method="semantic_entropy", approximation="sep",
            )
        clusters = cluster_vectors(vectors, similarity_threshold)
        stats = _confidence_from_sizes([len(c) for c in clusters])
        return ConfidenceResult(
            method="semantic_entropy",
            confidence=round(float(stats["confidence"]), 6),
            status=STATUS_OK,
            scope="assertion",
            details={
                "approximation": "sep_hidden_states",
                "source": source,
                "vectors": len(vectors),
                "dim": len(vectors[0]),
                "cluster_count": stats["clusters"],
                "cluster_sizes": stats["cluster_sizes"],
                "entropy_nats": round(float(stats["entropy_nats"]), 6),
                "entropy_normalized": round(float(stats["entropy_normalized"]), 6),
                "similarity_threshold": float(similarity_threshold),
            },
        )

    values = _as_logprobs(logprobs)
    stats = logprob_confidence(values)
    return ConfidenceResult(
        method="semantic_entropy",
        confidence=round(float(stats["confidence"]), 6),
        status=STATUS_OK,
        scope="assertion",
        details={
            "approximation": "sep_logprob_geom_mean",
            "source": source,
            "tokens": stats["tokens"],
            "mean_logprob": round(float(stats["mean_logprob"]), 6),
            "perplexity": round(float(stats["perplexity"]), 6),
        },
    )
