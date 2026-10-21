"""B5 — UQ Heads: inference adapter for a trained uncertainty head.

UQ Heads are claim-level uncertainty probes (attention/logit auxiliary heads) that
must be trained before they can score anything.  The honest failure modes are:

* trained head / loader / valid ``cbb-uq-head/1`` checkpoint ⇒ real inference;
* no artifact anywhere ⇒ ``status="external"`` (EXTERNAL) with the missing source
  named — **never** a fabricated score;
* an artifact that exists but is unreadable / from another feature schema / broken
  ⇒ ``status="external"`` naming the path and the reason (still no score);
* a head that returns a value outside ``[0, 1]`` or an unparseable shape ⇒
  ``status="error"``.

Checkpoint resolution order (first non-empty wins, explicit beats implicit):

1. ``head=...`` callable (or object with ``score``/``predict``/``infer``);
2. ``checkpoint=...`` argument, loaded with ``loader`` if given else the built-in
   :func:`cbb_stage1.confidence.uq_checkpoint.load_checkpoint`;
3. ``config.uq_head_checkpoint`` (the pipeline ``Config`` field);
4. environment variable ``CBB_UQ_HEAD_CHECKPOINT``.

``tools/train_uq_head.py`` trains the head (pure-stdlib logistic regression over
:mod:`cbb_stage1.confidence.uq_features`) and writes the checkpoint this adapter
reads.  Production weights are a deployment artifact (EXTERNAL when absent); the
repo only ships fixture weights produced inside tests.

Design reference: ``07`` B5 (claim-level uncertainty, optional; needs training),
``01`` §5.15 ``confidence_scope="claim"``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from . import STATUS_OK, ConfidenceResult, as_mapping
from .uq_checkpoint import UQHeadCheckpointError, is_checkpoint, load_checkpoint

__all__ = [
    "CHECKPOINT_ENV",
    "EXTERNAL_REASON",
    "available",
    "resolve_checkpoint",
    "score",
]

CHECKPOINT_ENV = "CBB_UQ_HEAD_CHECKPOINT"

EXTERNAL_REASON = (
    "UQ Heads 需已训练头：未找到可用 checkpoint（依次查 head 参数、checkpoint 参数、"
    "Config.uq_head_checkpoint、环境变量 CBB_UQ_HEAD_CHECKPOINT；训练用 tools/train_uq_head.py）"
    " ⇒ EXTERNAL，不伪造分数。"
)

_PARSE_KEYS = ("confidence", "score", "value", "prob", "probability")


def resolve_checkpoint(
    checkpoint: Any = None, config: Any = None
) -> tuple[str | None, str | None]:
    """Return ``(path, source)``; ``source`` ∈ ``explicit|config|env|None``."""
    if checkpoint is not None:
        return os.fspath(checkpoint), "explicit"
    if config is not None:
        value = getattr(config, "uq_head_checkpoint", None)
        if value:
            return os.fspath(value), "config"
    env = os.environ.get(CHECKPOINT_ENV)
    if env:
        return env, "env"
    return None, None


def available(
    *,
    head: Any = None,
    checkpoint: Any = None,
    loader: Any = None,
    config: Any = None,
) -> bool:
    """True only when a usable head is in hand (or a loadable checkpoint exists)."""
    if head is not None:
        return True
    resolved, _source = resolve_checkpoint(checkpoint, config)
    if resolved is None:
        return False
    if loader is not None:
        return True  # the caller promises to produce a head from the path
    return is_checkpoint(resolved)


def _call_head(head: Any, record: Mapping[str, Any]) -> Any:
    if callable(head):
        return head(dict(record) if isinstance(record, Mapping) else record)
    for name in ("score", "predict", "infer"):
        fn = getattr(head, name, None)
        if callable(fn):
            return fn(dict(record) if isinstance(record, Mapping) else record)
    raise TypeError("head 既不可调用也没有 score/predict/infer 方法：%r" % (type(head).__name__,))


def _parse_head_output(raw: Any) -> tuple[float | None, str | None, str | None]:
    """Return ``(confidence, scope, why)``; ``why`` non-None means untrusted output."""
    scope = None
    if isinstance(raw, bool):
        return None, None, "head 返回 bool：不是 0-1 分数"
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, Mapping):
        raw_scope = raw.get("scope")
        if raw_scope in ("assertion", "claim"):
            scope = raw_scope
        for key in _PARSE_KEYS:
            if key in raw:
                return _parse_head_output(raw[key])[0], scope, None
        return None, scope, "head 返回的对象里没有 confidence/score/value"
    else:
        return None, None, "head 返回类型不认：%r" % (type(raw).__name__,)
    if not 0.0 <= value <= 1.0:
        return None, scope, "head 返回 %r：不在 [0,1]（不裁剪、不猜口径）" % (value,)
    return value, scope, None


def score(
    record: Any,
    llm: Any = None,
    *,
    head: Any = None,
    checkpoint: Any = None,
    loader: Callable[[Any], Any] | None = None,
    config: Any = None,
    scope: str = "claim",
    **_: Any,
) -> ConfidenceResult:
    """Run a trained head; never fabricate a score when the artifact is absent."""
    if scope not in ("assertion", "claim"):
        raise ValueError("scope 必须在 assertion/claim：%r" % (scope,))
    rec = as_mapping(record)
    resolved, source = resolve_checkpoint(checkpoint, config)

    if head is None and resolved is not None:
        if loader is not None:
            try:
                head = loader(resolved)
            except Exception as exc:  # noqa: BLE001 - external artifact failure is a result
                return ConfidenceResult.external(
                    "loader(checkpoint) 失败 ⇒ EXTERNAL（%s 来源 %s：%s）"
                    % (resolved, source, exc),
                    method="uq_heads", checkpoint=resolved, checkpoint_source=source,
                    error="%s: %s" % (type(exc).__name__, exc),
                )
            if head is None:
                return ConfidenceResult.external(
                    "loader(checkpoint) 返回 None（没有加载出头）⇒ EXTERNAL（来源 %s）" % source,
                    method="uq_heads", checkpoint=resolved, checkpoint_source=source,
                )
        else:
            try:
                head = load_checkpoint(resolved)
            except UQHeadCheckpointError as exc:
                return ConfidenceResult.external(
                    "checkpoint 不可用 ⇒ EXTERNAL：%s（来源 %s）" % (exc, source),
                    method="uq_heads", checkpoint=resolved, checkpoint_source=source,
                )
    if head is None:
        return ConfidenceResult.external(EXTERNAL_REASON, method="uq_heads")

    try:
        raw = _call_head(head, rec)
    except Exception as exc:  # noqa: BLE001 - head failure is reported, never scored
        return ConfidenceResult.error(
            "head 调用失败：%s：%s" % (type(exc).__name__, exc),
            method="uq_heads", head_type=type(head).__name__,
        )
    value, out_scope, why = _parse_head_output(raw)
    if why is not None:
        return ConfidenceResult.error(why, method="uq_heads", head_type=type(head).__name__,
                                      raw=str(raw)[:120])
    return ConfidenceResult(
        method="uq_heads",
        confidence=round(float(value), 6),
        status=STATUS_OK,
        scope=out_scope or scope,
        details={
            "head_type": type(head).__name__,
            "head_kind": getattr(head, "kind", None),
            "feature_schema": getattr(head, "feature_schema", None),
            "checkpoint": resolved,
            "checkpoint_source": source,
            "scope_default": scope,
            "note": "经已训练头推理；本仓不附带生产权重（缺失记 EXTERNAL）；不替代金标 ECE 校准",
        },
    )
