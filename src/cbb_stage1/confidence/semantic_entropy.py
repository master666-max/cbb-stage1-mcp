"""B2 — semantic entropy (meaning-space entropy from N samples).

Algorithm (all deterministic, stdlib only)::

    sample the same claim N times  →  normalise answers  →  greedy meaning-space
    clustering (char bigram Jaccard / SequenceMatcher)  →  cluster distribution p
    →  H = -Σ p·ln p  →  H_norm = H / ln(N)  →  confidence = 1 - H_norm

* ``N < 2`` ⇒ ``unavailable`` (entropy is undefined from one sample).
* no ``llm``/``sampler`` ⇒ ``unavailable`` (no fabricated score).
* async ``llm.generate`` ⇒ ``unavailable``: this is a synchronous adapter; the
  caller must pass a synchronous ``sampler`` (we never block on or fake a value).
* a record with no claim text ⇒ ``unavailable`` (nothing to ask about).

The LLM protocol accepted here is intentionally tiny (W1 can adapt its client)::

    llm.sample(prompt, n) -> Sequence[str]              # preferred
    llm.generate(prompt)   -> str | (str, usage)        # called n times

or pass ``sampler=callable(prompt, n) -> Sequence[str]``.
"""

from __future__ import annotations

import difflib
import inspect
import math
from typing import Any, Callable, Mapping, Sequence

from cbb_stage1.ids import short_hash

from . import (
    STATUS_OK,
    ConfidenceResult,
    as_mapping,
)

__all__ = [
    "build_prompt",
    "claim_text",
    "cluster_answers",
    "normalize_answer",
    "score",
    "semantic_entropy",
    "similar",
]

#: Ordered fields that carry the comparable claim text (first non-empty wins).
_CLAIM_FIELDS = ("normalized_claim", "claim", "statement", "text", "summary")

#: Prompt used when the caller does not pass one.
_PROMPT_TEMPLATE = (
    "请只复述下面这条主张的内容（不要解释、不要增删事实），用一句话回答：\n{claim}"
)


def claim_text(record: Any) -> str | None:
    """Extract the claim/assertion text that all N samples are about."""
    rec = as_mapping(record)
    for name in _CLAIM_FIELDS:
        value = rec.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # assertion shape: predicate + roles
    predicate = rec.get("predicate") or rec.get("pred") or rec.get("pred_name")
    roles = rec.get("roles")
    if predicate:
        if isinstance(roles, Mapping) and roles:
            body = "；".join("%s=%s" % (k, v) for k, v in sorted(roles.items()))
            return "%s：%s" % (predicate, body)
        if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)) and roles:
            parts = []
            for item in roles:
                if isinstance(item, Mapping):
                    role = item.get("role") or item.get("name") or "?"
                    value = item.get("value")
                    parts.append("%s=%s" % (role, value))
                else:
                    parts.append(str(item))
            return "%s：%s" % (predicate, "；".join(parts))
        return str(predicate)
    # evidence quotes as a last resort (still a real claim surface)
    evidence = rec.get("evidence")
    quotes: list[str] = []
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        for item in evidence:
            if isinstance(item, str) and item.strip():
                quotes.append(item.strip())
            elif isinstance(item, Mapping) and isinstance(item.get("quote"), str):
                quotes.append(item["quote"].strip())
    if quotes:
        return "；".join(q for q in quotes if q)
    return None


def build_prompt(record: Any, template: str | None = None) -> str:
    """Prompt shared by all samples (same input for every sample)."""
    text = claim_text(record)
    if not text:
        raise ValueError("记录没有可比较的主张文本（claim_text 为空）")
    return (template or _PROMPT_TEMPLATE).format(claim=text)


def normalize_answer(text: str) -> str:
    """Whitespace/case/punctuation normalisation used before clustering."""
    if not isinstance(text, str):
        return ""
    out = "".join(" " if ch.isspace() else ch for ch in text).strip().lower()
    # strip common trailing/leading punctuation only (keep inner structure)
    out = out.strip(" \t\r\n。.!！?？;；,，、:：\"'“”‘’()（）[]【】")
    return " ".join(out.split())


def _char_bigrams(text: str) -> set[str]:
    compact = "".join(text.split())
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _char_bigrams(a), _char_bigrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Meaning-space similarity proxy: normalised equality, bigram Jaccard or ratio."""
    if a == b:
        return True
    if _jaccard(a, b) >= threshold:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def cluster_answers(answers: Sequence[str], threshold: float = 0.6) -> list[list[str]]:
    """Greedy clustering; the order of ``answers`` fixes the result (deterministic)."""
    clusters: list[list[str]] = []
    for raw in answers:
        normalised = normalize_answer(raw)
        if not normalised:
            continue
        for cluster in clusters:
            if similar(normalised, cluster[0], threshold):
                cluster.append(normalised)
                break
        else:
            clusters.append([normalised])
    return clusters


def semantic_entropy(answers: Sequence[str], threshold: float = 0.6) -> dict[str, Any]:
    """Entropy over meaning clusters → confidence in ``[0, 1]``."""
    clusters = cluster_answers(answers, threshold)
    total = sum(len(c) for c in clusters)
    if total < 2:
        return {
            "usable": False,
            "reason": "有效样本 <2，熵无定义（N=%d）" % total,
            "clusters": clusters,
        }
    probs = [len(c) / total for c in clusters]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(total)
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    confidence = max(0.0, min(1.0, 1.0 - normalised))
    return {
        "usable": True,
        "samples": total,
        "clusters": clusters,
        "cluster_sizes": [len(c) for c in clusters],
        "entropy_nats": entropy,
        "entropy_normalized": normalised,
        "confidence": confidence,
    }


# ── sampling ───────────────────────────────────────────────────────────────
def _discard_awaitable(obj: Any) -> None:
    """Close an async result instead of leaving an un-awaited coroutine behind."""
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _sample(llm: Any, prompt: str, n: int, sampler: Callable[..., Any] | None) -> tuple[list[str] | None, str | None]:
    raw: Any
    if sampler is not None:
        raw = sampler(prompt, n)
        if inspect.isawaitable(raw):
            _discard_awaitable(raw)
            return None, "sampler 异步：本同步接口不驱动协程（请给同步 sampler，不伪造分数）"
        return _clean_samples(raw), None
    if llm is None:
        return None, "没有 llm/sampler：无法采样 N 次（B2 需要模型）"
    if hasattr(llm, "sample"):
        raw = llm.sample(prompt, n)
        if inspect.isawaitable(raw):
            _discard_awaitable(raw)
            return None, "llm.sample 异步：本同步接口不驱动协程（请给同步 sampler）"
        return _clean_samples(raw), None
    if hasattr(llm, "generate"):
        answers: list[str] = []
        for _ in range(n):
            item = llm.generate(prompt)
            if inspect.isawaitable(item):
                _discard_awaitable(item)
                return None, "llm.generate 异步：本同步接口不驱动协程（请给同步 sampler）"
            answers.append(_generate_text(item))
        return _clean_samples(answers), None
    return None, "llm 没有 sample/generate 采样接口（无法算语义熵）"


def _generate_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item:
        return _generate_text(item[0])
    if isinstance(item, Mapping) and isinstance(item.get("text"), str):
        return item["text"]
    return "" if item is None else str(item)


def _clean_samples(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(x) for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]
    return []


def score(
    record: Any,
    llm: Any = None,
    *,
    samples: int = 8,
    similarity_threshold: float = 0.6,
    prompt: str | None = None,
    sampler: Callable[..., Any] | None = None,
    template: str | None = None,
    **_: Any,
) -> ConfidenceResult:
    """Sample N times → meaning clusters → entropy → ``ConfidenceResult``."""
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples 必须是 >=2 的整数（熵需要 >=2 个样本）：%r" % (samples,))
    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise ValueError("similarity_threshold 必须在 [0,1]：%r" % (similarity_threshold,))
    rec = as_mapping(record)
    try:
        text_prompt = prompt if prompt else build_prompt(rec, template)
    except ValueError as exc:
        return ConfidenceResult.unavailable(str(exc), method="semantic_entropy")
    answers, why = _sample(llm, text_prompt, samples, sampler)
    if answers is None:
        return ConfidenceResult.unavailable(why or "采样不可用", method="semantic_entropy")
    result = semantic_entropy(answers, similarity_threshold)
    if not result["usable"]:
        return ConfidenceResult.unavailable(
            str(result["reason"]),
            method="semantic_entropy",
            samples=result.get("clusters"),
            requested_samples=samples,
        )
    return ConfidenceResult(
        method="semantic_entropy",
        confidence=round(float(result["confidence"]), 6),
        status=STATUS_OK,
        scope="assertion",
        details={
            "samples": result["samples"],
            "cluster_count": len(result["clusters"]),
            "cluster_sizes": result["cluster_sizes"],
            "cluster_examples": [c[0] for c in result["clusters"]],
            "entropy_nats": round(float(result["entropy_nats"]), 6),
            "entropy_normalized": round(float(result["entropy_normalized"]), 6),
            "similarity_threshold": float(similarity_threshold),
            "prompt_sha1": short_hash(text_prompt, 12),
            "approximation": "semantic_entropy_clustering",
        },
    )

