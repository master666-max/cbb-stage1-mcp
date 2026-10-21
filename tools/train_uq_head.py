#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_uq_head — reproducible trainer for the B5 UQ head (``cbb-uq-head/1``).

Motivation
==========
``07`` B5 (UQ Heads) needs a *trained* claim-level uncertainty probe; the repo
must not ship fake numbers.  This tool trains a small, auditable head and writes a
checkpoint that :func:`cbb_stage1.confidence.uq_heads.score` can load and use for
real inference.  Production weights stay a deployment artifact (absent ⇒ EXTERNAL).

Model
=====
Standardized logistic regression over :data:`cbb_stage1.confidence.uq_features`
(12 statistical claim/answer/evidence features), trained with full-batch gradient
descent in pure stdlib (no numpy/torch dependency).  Deterministic: fixed order,
no timestamps — same data + args ⇒ byte-identical checkpoint.

Input JSONL (one label per line)::

    {"record": {"claim": "甲是青匠", "answer": "甲是铁匠", "evidence": [{"quote": "甲是青牛镇的铁匠"}]}, "label": 1}
    {"claim": "甲是魔王", "answer": "不知道", "evidence": [], "label": 0}     # flat form also accepted

Usage::

    python tools/train_uq_head.py --data labels.jsonl --out artifacts/uq_head.json
    python tools/train_uq_head.py --data labels.jsonl --out head.json --epochs 300 --lr 0.3 --holdout 0.2

Exit codes: 0 trained / 2 bad arguments, bad JSONL, or a one-class dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

# Run from the repo root even when the package is not installed.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cbb_stage1.confidence.uq_checkpoint import CHECKPOINT_FORMAT  # noqa: E402
from cbb_stage1.confidence.uq_features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    extract_features,
)

_LABEL_KEYS = ("label", "y")
_EPS = 1e-12


class TrainError(ValueError):
    """Bad input or impossible training request (reported as rc=2, no silent run)."""


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read labeled rows: ``[{"record": {...}, "label": 0|1}, ...]`` (file order)."""
    name = os.fspath(path)
    if not os.path.isfile(name):
        raise TrainError("标注文件不在：%s" % name)
    rows: list[dict[str, Any]] = []
    with open(name, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except ValueError as exc:
                raise TrainError("第 %d 行不是合法 JSON：%s" % (lineno, exc)) from exc
            if not isinstance(raw, dict):
                raise TrainError("第 %d 行必须是对象：%r" % (lineno, text[:60]))
            label = None
            for key in _LABEL_KEYS:
                if key in raw:
                    label = raw[key]
                    break
            if label is None:
                raise TrainError("第 %d 行缺 label（0/1）：%r" % (lineno, text[:60]))
            if isinstance(label, bool):
                label = int(label)
            if not isinstance(label, int) or label not in (0, 1):
                raise TrainError("第 %d 行 label 必须是 0/1：%r" % (lineno, label))
            record = raw.get("record")
            if record is None:
                record = {k: v for k, v in raw.items() if k not in _LABEL_KEYS}
            if not isinstance(record, dict) or not record:
                raise TrainError("第 %d 行 record 必须是非空对象" % lineno)
            rows.append({"record": record, "label": int(label), "lineno": lineno})
    if not rows:
        raise TrainError("标注文件没有任何行：%s" % name)
    return rows


def _round(value: float) -> float:
    return round(float(value), 12)


def _standardize(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[float], list[list[float]]]:
    n, dim = len(matrix), len(matrix[0])
    mean = [sum(row[j] for row in matrix) / n for j in range(dim)]
    std = []
    for j in range(dim):
        var = sum((row[j] - mean[j]) ** 2 for row in matrix) / n
        std.append(math.sqrt(var) if var > 0.0 else 1.0)
    z = [[(row[j] - mean[j]) / std[j] for j in range(dim)] for row in matrix]
    return mean, std, z


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _loss_metrics(
    z: Sequence[Sequence[float]], labels: Sequence[int], weights: Sequence[float], bias: float
) -> dict[str, float]:
    n = len(labels)
    logloss = 0.0
    correct = 0
    for row, label in zip(z, labels):
        logit = bias + sum(w * x for w, x in zip(weights, row))
        p = min(max(_sigmoid(logit), _EPS), 1.0 - _EPS)
        logloss -= label * math.log(p) + (1 - label) * math.log(1.0 - p)
        if (p >= 0.5) == (label == 1):
            correct += 1
    return {"n": float(n), "logloss": _round(logloss / n), "accuracy": _round(correct / n)}


def _fit(
    z: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[list[float], float]:
    n, dim = len(z), len(z[0])
    weights = [0.0] * dim
    bias = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for row, label in zip(z, labels):
            logit = bias + sum(w * x for w, x in zip(weights, row))
            err = _sigmoid(logit) - label
            grad_b += err
            for j in range(dim):
                grad_w[j] += err * row[j]
        for j in range(dim):
            weights[j] -= lr * (grad_w[j] / n + l2 * weights[j])
        bias -= lr * (grad_b / n)
    return weights, bias


def _data_sha(rows: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps({"record": row["record"], "label": row["label"]},
                   ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train(
    rows: Sequence[dict[str, Any]],
    *,
    epochs: int = 400,
    lr: float = 0.2,
    l2: float = 1e-3,
    seed: int = 0,
    holdout: float = 0.0,
) -> dict[str, Any]:
    """Fit a ``cbb-uq-head/1`` checkpoint payload from labeled rows."""
    if epochs < 1:
        raise TrainError("--epochs 必须 >=1：%r" % epochs)
    if lr <= 0.0 or not math.isfinite(lr):
        raise TrainError("--lr 必须 >0：%r" % lr)
    if l2 < 0.0 or not math.isfinite(l2):
        raise TrainError("--l2 必须 >=0：%r" % l2)
    if not 0.0 <= holdout < 0.9:
        raise TrainError("--holdout 必须在 [0, 0.9)：%r" % holdout)
    labels = [row["label"] for row in rows]
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise TrainError(
            "标注只有一类（pos=%d neg=%d）：逻辑回归无法训练，需要两类标注" % (n_pos, n_neg))

    indices = list(range(len(rows)))
    if holdout > 0.0:
        random.Random(seed).shuffle(indices)
        n_hold = int(round(len(rows) * holdout))
        if n_hold < 1 or n_hold >= len(rows):
            raise TrainError("holdout=%r 在 %d 行上分不出训练/留出集" % (holdout, len(rows)))
        hold_idx, train_idx = indices[:n_hold], indices[n_hold:]
    else:
        hold_idx, train_idx = [], indices
    train_labels = [labels[i] for i in train_idx]
    if len(set(train_labels)) < 2:
        raise TrainError("训练集只剩一类（holdout 切分把另一类全划走了）——换 seed/holdout")

    matrix = [extract_features(rows[i]["record"]) for i in train_idx]
    mean, std, z_train = _standardize(matrix)
    weights, bias = _fit(z_train, train_labels, epochs=epochs, lr=lr, l2=l2)
    metrics_train = _loss_metrics(z_train, train_labels, weights, bias)
    metrics_holdout = None
    if hold_idx:
        hold_matrix = [extract_features(rows[i]["record"]) for i in hold_idx]
        z_hold = [[(row[j] - mean[j]) / std[j] for j in range(len(mean))] for row in hold_matrix]
        metrics_holdout = _loss_metrics(z_hold, [labels[i] for i in hold_idx], weights, bias)

    return {
        "format": CHECKPOINT_FORMAT,
        "kind": "logreg",
        "feature_schema": FEATURE_SCHEMA,
        "features": list(FEATURE_NAMES),
        "mean": [_round(x) for x in mean],
        "std": [_round(max(x, 1e-9)) for x in std],
        "weights": [_round(w) for w in weights],
        "bias": _round(bias),
        "threshold": 0.5,
        "data_sha256": _data_sha(rows),
        "n_rows": len(rows),
        "n_positive": n_pos,
        "seed": seed,
        "holdout": holdout,
        "metrics": {"train": metrics_train, "holdout": metrics_holdout},
        "trained_by": "tools/train_uq_head.py",
    }


def write_checkpoint(payload: dict[str, Any], out: str | os.PathLike[str]) -> str:
    name = os.fspath(out)
    parent = os.path.dirname(os.path.abspath(name))
    if not os.path.isdir(parent):
        raise TrainError("输出目录不在位：%s（本工具不代建目录）" % parent)
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    with open(name, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a cbb-uq-head/1 logistic-regression UQ head.")
    parser.add_argument("--data", required=True, help="labeled JSONL (record + label 0/1)")
    parser.add_argument("--out", required=True, help="checkpoint JSON path (parent dir must exist)")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        rows = read_jsonl(args.data)
        payload = train(rows, epochs=args.epochs, lr=args.lr, l2=args.l2,
                        seed=args.seed, holdout=args.holdout)
        ckpt_sha = write_checkpoint(payload, args.out)
    except TrainError as exc:
        print("ERROR: %s" % exc)
        return 2
    metrics = payload["metrics"]
    train_m = metrics["train"]
    print("TRAIN-UQ-HEAD｜data=%s rows=%d pos=%d neg=%d features=%d"
          % (args.data, payload["n_rows"], payload["n_positive"],
             payload["n_rows"] - payload["n_positive"], len(payload["features"])))
    print("  train   logloss=%.6f accuracy=%.6f n=%d"
          % (train_m["logloss"], train_m["accuracy"], int(train_m["n"])))
    hold = metrics["holdout"]
    if hold:
        print("  holdout logloss=%.6f accuracy=%.6f n=%d"
              % (hold["logloss"], hold["accuracy"], int(hold["n"])))
    print("  OUT %s sha256=%s data_sha256=%s"
          % (args.out, ckpt_sha[:16], payload["data_sha256"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
