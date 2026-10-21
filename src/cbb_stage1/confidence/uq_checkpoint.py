"""Checkpoint format + loader for the B5 UQ head (``cbb-uq-head/1``).

``tools/train_uq_head.py`` writes this JSON; :class:`LogRegUQHead` reads it and
produces a real ``[0, 1]`` claim-level score at inference time.  The format is
deliberately plain JSON (no pickle, no torch): the head is a small standardized
logistic regression, so a checkpoint is auditable and portable, and the loader
works on any machine with the stdlib.

A checkpoint is **not** a score.  When the file is absent, unreadable, from another
feature schema, or malformed, :func:`load_checkpoint` raises
:class:`UQHeadCheckpointError`; the caller (:mod:`cbb_stage1.confidence.uq_heads`)
then reports ``EXTERNAL`` — never a fabricated number.

Format (version 1)::

    {
      "format": "cbb-uq-head/1",
      "kind": "logreg",
      "feature_schema": 1,
      "features": ["claim_chars", …],          # exactly uq_features.FEATURE_NAMES
      "mean":  [...], "std": [...],            # standardization of those features
      "weights": [...], "bias": 0.0,           # decision function
      "threshold": 0.5,                        # label boundary (not used for score)
      "data_sha256": "…", "n_rows": 40, "n_positive": 20, "seed": 0,
      "metrics": {...},                        # train/holdout logloss+accuracy
      "feature_schema_owner": "cbb_stage1.confidence.uq_features"
    }
"""

from __future__ import annotations

import io
import json
import math
import os
from typing import Any, Mapping

from .uq_features import FEATURE_NAMES, FEATURE_SCHEMA, extract_features

__all__ = [
    "CHECKPOINT_FORMAT",
    "KINDS",
    "LogRegUQHead",
    "UQHeadCheckpointError",
    "is_checkpoint",
    "load_checkpoint",
]

CHECKPOINT_FORMAT = "cbb-uq-head/1"
KINDS = ("logreg",)


class UQHeadCheckpointError(ValueError):
    """The artifact exists but cannot be trusted as a ``cbb-uq-head/1`` head."""


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _as_floats(payload: Mapping[str, Any], key: str, path: str) -> tuple[float, ...]:
    raw = payload.get(key)
    if not isinstance(raw, (list, tuple)):
        raise UQHeadCheckpointError("%s: 字段 %r 必须是数值数组" % (path, key))
    out: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise UQHeadCheckpointError("%s: 字段 %r 含非数值项：%r" % (path, key, item))
        value = float(item)
        if not math.isfinite(value):
            raise UQHeadCheckpointError("%s: 字段 %r 含非有限值：%r" % (path, key, item))
        out.append(value)
    return tuple(out)


class LogRegUQHead:
    """Standardized logistic-regression head loaded from a ``cbb-uq-head/1`` file."""

    kind = "logreg"

    def __init__(self, payload: Mapping[str, Any], path: str | None = None) -> None:
        self.path = path
        if not isinstance(payload, Mapping):
            raise UQHeadCheckpointError("checkpoint 根不是对象")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise UQHeadCheckpointError(
                "checkpoint format 不是 %r：%r" % (CHECKPOINT_FORMAT, payload.get("format")))
        if payload.get("kind") not in KINDS:
            raise UQHeadCheckpointError(
                "checkpoint kind 不属闭集 %s：%r" % (KINDS, payload.get("kind")))
        if int(payload.get("feature_schema", -1)) != FEATURE_SCHEMA:
            raise UQHeadCheckpointError(
                "feature_schema 不匹配：checkpoint=%r 代码=%r（模型与特征不同源，拒载）"
                % (payload.get("feature_schema"), FEATURE_SCHEMA))
        features = payload.get("features")
        if list(features or []) != list(FEATURE_NAMES):
            raise UQHeadCheckpointError(
                "checkpoint features 与代码 FEATURE_NAMES 不一致（拒绝静默错位）")
        self.mean = _as_floats(payload, "mean", self.path or "<checkpoint>")
        self.std = _as_floats(payload, "std", self.path or "<checkpoint>")
        self.weights = _as_floats(payload, "weights", self.path or "<checkpoint>")
        n = len(FEATURE_NAMES)
        for name, vec in (("mean", self.mean), ("std", self.std), ("weights", self.weights)):
            if len(vec) != n:
                raise UQHeadCheckpointError(
                    "%s 长度 %d != 特征数 %d" % (name, len(vec), n))
        if any(value <= 0.0 for value in self.std):
            raise UQHeadCheckpointError("std 必须 > 0（零方差特征不能标准化）")
        bias = payload.get("bias", 0.0)
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(float(bias)):
            raise UQHeadCheckpointError("bias 不是有限数值：%r" % (bias,))
        self.bias = float(bias)
        threshold = payload.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                or not 0.0 <= float(threshold) <= 1.0:
            raise UQHeadCheckpointError("threshold 必须在 [0,1]：%r" % (threshold,))
        self.threshold = float(threshold)
        self.feature_schema = FEATURE_SCHEMA
        self.data_sha256 = payload.get("data_sha256")
        self.metrics = dict(payload.get("metrics") or {})

    def score(self, record: Any) -> float:
        """Real inference: standardize → logit → sigmoid, in ``(0, 1)``."""
        values = extract_features(record)
        z = self.bias
        for value, mean, std, weight in zip(values, self.mean, self.std, self.weights):
            z += weight * ((value - mean) / std)
        return _sigmoid(z)

    def predict(self, record: Any) -> float:
        """Label using the checkpoint threshold (``1.0``/``0.0``)."""
        return 1.0 if self.score(record) >= self.threshold else 0.0


def load_checkpoint(path: str | os.PathLike[str]) -> LogRegUQHead:
    """Read and validate a ``cbb-uq-head/1`` checkpoint."""
    name = os.fspath(path)
    if not os.path.isfile(name):
        raise UQHeadCheckpointError("checkpoint 不存在：%s" % name)
    try:
        payload = json.loads(io.open(name, encoding="utf-8").read())
    except (OSError, ValueError) as exc:
        raise UQHeadCheckpointError("checkpoint 读不动：%s（%s）" % (name, exc)) from exc
    return LogRegUQHead(payload, name)


def is_checkpoint(path: str | os.PathLike[str]) -> bool:
    """Cheap probe used by ``available()`` — never raises."""
    try:
        load_checkpoint(path)
        return True
    except (UQHeadCheckpointError, TypeError, ValueError):
        return False
