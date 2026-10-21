"""B4 — UQLM: white-box / black-box / judge uncertainty scores in ``[0, 1]``.

Three classes, each an **adapter over a signal the caller already has**; this
module never invents a score:

* ``whitebox`` — token probabilities (``llm.token_logprobs`` or explicit
  ``logprobs``) → geometric-mean token probability.  Method ``uqlm_whitebox``
  (cheap, needs logprobs).
* ``blackbox`` — N sampled answers → meaning-space agreement
  ``Σ size² / N²`` (probability that two samples agree).  Method
  ``uqlm_blackbox`` (expensive, no logprobs needed).
* ``judge`` — an LLM judge returns a 0–1 support score.  §5.15 has no
  ``uqlm_judge`` value, so the canonical method recorded is ``claim_qa`` (the
  closest closed-enum class) while ``details["uqlm_class"]="judge"`` keeps the
  UQLM class visible.  A judge that does not return a parseable ``[0,1]`` number
  is recorded as ``error`` — never rounded into a fake confidence.

If the required signal is missing the result is ``unavailable`` (with a reason),
not ``0.0``.  This is the difference between "uncertain" and "no measurement".
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Mapping, Sequence

from cbb_stage1.ids import short_hash

from . import STATUS_OK, ConfidenceResult, as_mapping
from .semantic_entropy import (
    _discard_awaitable,
    _sample,
    build_prompt,
    claim_text,
    cluster_answers,
)
from .sep import logprob_confidence

__all__ = [
    "MODES",
    "parse_judge_score",
    "score",
]

MODES = ("whitebox", "blackbox", "judge")

_METHOD_BY_MODE = {
    "whitebox": "uqlm_whitebox",
    "blackbox": "uqlm_blackbox",
    "judge": "claim_qa",
}

_JUDGE_PROMPT = (
    "你是判卷官。只根据下面的主张文本判断其可支持程度，并只输出一个 0 到 1 之间的数"
    "（0=完全不可信，1=完全可信）：\n{claim}"
)

_NUMBER_RX = re.compile(r"(?<![0-9.])(0(?:\.\d+)?|1(?:\.0+)?)(?![0-9.])")


# ── whitebox ───────────────────────────────────────────────────────────────
def _pull_logprobs(llm: Any, record: Any, logprobs: Sequence[float] | None) -> tuple[list[float] | None, str | None]:
    if logprobs is not None:
        return [float(x) for x in logprobs], None
    if llm is None:
        return None, "白盒需要 logprobs 或带 token_logprobs 的 llm"
    getter = getattr(llm, "token_logprobs", None) or getattr(llm, "logprobs", None)
    if getter is None:
        return None, "llm 没有 token_logprobs/logprobs（不是白盒客户端）"
    text = claim_text(record)
    if not text:
        return None, "记录没有可比较的主张文本（白盒探针没有输入）"
    raw = getter(build_prompt(record))
    if inspect.isawaitable(raw):
        _discard_awaitable(raw)
        return None, "白盒接口异步：本同步适配器不驱动协程（请直接传 logprobs）"
    if hasattr(raw, "tolist") and not isinstance(raw, (list, tuple)):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None, "token_logprobs 返回值不是数值序列"
    return [float(x) for x in raw], None


def _whitebox(record: Any, llm: Any, logprobs: Sequence[float] | None) -> ConfidenceResult:
    values, why = _pull_logprobs(llm, record, logprobs)
    if values is None:
        return ConfidenceResult.unavailable(why or "白盒信号不可用", method="uqlm_whitebox")
    try:
        stats = logprob_confidence(values)
    except ValueError as exc:
        return ConfidenceResult.error(str(exc), method="uqlm_whitebox")
    return ConfidenceResult(
        method="uqlm_whitebox",
        confidence=round(float(stats["confidence"]), 6),
        status=STATUS_OK,
        scope="assertion",
        details={
            "uqlm_class": "whitebox",
            "tokens": stats["tokens"],
            "mean_logprob": round(float(stats["mean_logprob"]), 6),
            "perplexity": round(float(stats["perplexity"]), 6),
        },
    )


# ── blackbox ───────────────────────────────────────────────────────────────
def _blackbox(
    record: Any,
    llm: Any,
    *,
    samples: int,
    sampler: Any,
    similarity_threshold: float,
) -> ConfidenceResult:
    text = claim_text(record)
    if not text:
        return ConfidenceResult.unavailable(
            "记录没有可比较的主张文本（黑盒采样没有输入）", method="uqlm_blackbox")
    prompt = build_prompt(record)
    answers, why = _sample(llm, prompt, samples, sampler)
    if answers is None:
        return ConfidenceResult.unavailable(why or "采样不可用", method="uqlm_blackbox")
    clusters = cluster_answers(answers, similarity_threshold)
    total = sum(len(c) for c in clusters)
    if total < 2:
        return ConfidenceResult.unavailable(
            "有效样本 <2（N=%d）：一致性无定义" % total, method="uqlm_blackbox")
    agreement = sum((len(c) / total) ** 2 for c in clusters)
    return ConfidenceResult(
        method="uqlm_blackbox",
        confidence=round(max(0.0, min(1.0, agreement)), 6),
        status=STATUS_OK,
        scope="assertion",
        details={
            "uqlm_class": "blackbox",
            "samples": total,
            "cluster_count": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "cluster_examples": [c[0] for c in clusters],
            "agreement": round(float(agreement), 6),
            "similarity_threshold": float(similarity_threshold),
            "definition": "P(two sampled answers agree) = Σ size² / N²",
        },
    )


# ── judge ──────────────────────────────────────────────────────────────────
def parse_judge_score(raw: Any) -> tuple[float | None, str | None]:
    """Accept ``float`` / ``{"score": x}`` / text containing one ``[0,1]`` number."""
    if isinstance(raw, bool):
        return None, "judge 返回 bool：不是 0-1 分数"
    if isinstance(raw, (int, float)):
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value, None
        return None, "judge 返回 %r：不在 [0,1]（不猜百分比口径）" % (value,)
    if isinstance(raw, Mapping):
        for key in ("score", "confidence", "value"):
            if key in raw:
                return parse_judge_score(raw[key])
        return None, "judge 返回的对象里没有 score/confidence/value"
    if isinstance(raw, str):
        found = _NUMBER_RX.findall(raw)
        if len(found) == 1:
            return float(found[0]), None
        if not found:
            return None, "judge 文本里找不到 [0,1] 数字：%r" % (raw.strip()[:60],)
        return None, "judge 文本里有多个候选数字（无法确定）：%r" % (found[:4],)
    return None, "judge 返回类型不认：%r" % (type(raw).__name__,)


def _judge(record: Any, judge_llm: Any, prompt_template: str | None) -> ConfidenceResult:
    if judge_llm is None:
        return ConfidenceResult.unavailable(
            "judge 类需要 judge_llm（第二个模型）；未提供 ⇒ 不可用，不拿别的模型冒充",
            method="claim_qa", uqlm_class="judge",
        )
    text = claim_text(record)
    if not text:
        return ConfidenceResult.unavailable(
            "记录没有可比较的主张文本（判卷没有题目）", method="claim_qa", uqlm_class="judge")
    prompt = (prompt_template or _JUDGE_PROMPT).format(claim=text)
    raw: Any
    if hasattr(judge_llm, "judge"):
        raw = judge_llm.judge(prompt)
    elif hasattr(judge_llm, "generate"):
        raw = judge_llm.generate(prompt)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
            raw = raw[0]
    else:
        return ConfidenceResult.unavailable(
            "judge_llm 没有 judge/generate 接口", method="claim_qa", uqlm_class="judge")
    if inspect.isawaitable(raw):
        _discard_awaitable(raw)
        return ConfidenceResult.unavailable(
            "judge 接口异步：本同步适配器不驱动协程（请给同步 judge）",
            method="claim_qa", uqlm_class="judge")
    value, why = parse_judge_score(raw)
    if value is None:
        return ConfidenceResult.error(
            why or "judge 分数不可解析", method="claim_qa", uqlm_class="judge", raw=str(raw)[:120])
    return ConfidenceResult(
        method="claim_qa",
        confidence=round(float(value), 6),
        status=STATUS_OK,
        scope="claim",
        details={
            "uqlm_class": "judge",
            "note": "§5.15 无 uqlm_judge 闭集值；按冲突解决律记最近类 claim_qa，"
                    "UQLM 类名保留在 details.uqlm_class",
            "prompt_sha1": short_hash(prompt, 12),
        },
    )


def score(
    record: Any,
    llm: Any = None,
    *,
    mode: str | None = None,
    judge_llm: Any = None,
    samples: int = 5,
    logprobs: Sequence[float] | None = None,
    sampler: Any = None,
    similarity_threshold: float = 0.6,
    judge_prompt_template: str | None = None,
    **_: Any,
) -> ConfidenceResult:
    """UQLM entry; ``mode`` ∈ ``whitebox|blackbox|judge`` (``None`` = auto)."""
    rec = as_mapping(record)
    if mode is not None and mode not in MODES:
        raise ValueError("mode 不属闭集 %s：%r" % (MODES, mode))
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples 必须是 >=2 的整数：%r" % (samples,))
    if not 0.0 <= float(similarity_threshold) <= 1.0:
        raise ValueError("similarity_threshold 必须在 [0,1]：%r" % (similarity_threshold,))

    if mode is None:
        if logprobs is not None or hasattr(llm, "token_logprobs") or hasattr(llm, "logprobs"):
            mode = "whitebox"
        elif sampler is not None or (llm is not None and hasattr(llm, "sample")):
            mode = "blackbox"
        elif llm is not None and hasattr(llm, "generate"):
            mode = "blackbox"
        elif judge_llm is not None:
            mode = "judge"
        else:
            return ConfidenceResult.unavailable(
                "UQLM 三类都不可用：没有 logprobs（白盒）、没有可采样 llm（黑盒）、没有 judge_llm（裁判）",
            )

    if mode == "whitebox":
        return _whitebox(rec, llm, logprobs)
    if mode == "blackbox":
        return _blackbox(rec, llm, samples=samples, sampler=sampler,
                         similarity_threshold=similarity_threshold)
    return _judge(rec, judge_llm, judge_prompt_template)

