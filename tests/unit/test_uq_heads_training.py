"""Unit tests for B5 UQ Heads **training + inference** (task 24 / P3-C).

Coverage:

* ``tools/train_uq_head.py`` trains a reproducible ``cbb-uq-head/1`` checkpoint
  from labeled JSONL (claim/answer/evidence statistics);
* ``load_checkpoint`` validates the artifact and ``uq_heads.score`` runs real
  inference producing a ``[0, 1]`` score;
* checkpoint resolution: explicit argument > ``Config.uq_head_checkpoint`` >
  ``CBB_UQ_HEAD_CHECKPOINT``;
* missing production weights are reported explicitly as ``EXTERNAL`` — never
  faked into a number; malformed artifacts are not silently trusted;
* the tool fails loudly (rc=2) on one-class data / bad JSONL / missing out dir.

The trainer is exercised in-process through its ``main(argv)`` entry (the same
code path the CLI runs).  Spawning a child Python process is avoided on purpose:
the DSH Windows sandbox forbids the piped-stdio child processes that
``capture_output`` needs, which would make an otherwise deterministic test
environment-dependent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cbb_stage1.confidence import (
    STATUS_EXTERNAL,
    STATUS_OK,
    uq_heads,
)
from cbb_stage1.confidence.uq_checkpoint import (
    CHECKPOINT_FORMAT,
    UQHeadCheckpointError,
    load_checkpoint,
)
from cbb_stage1.confidence.uq_features import FEATURE_NAMES, extract_features
from cbb_stage1.config import Config

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "train_uq_head.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("train_uq_head", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAINER = _load_trainer()

SUPPORTED = {
    "claim": "甲在青牛镇打铁",
    "answer": "甲是铁匠",
    "evidence": [{"quote": "甲在青牛镇打铁为生"}],
}
UNSUPPORTED = {
    "claim": "甲是魔王的转世",
    "answer": "魔王",
    "evidence": [],
}


@pytest.fixture(autouse=True)
def _clean_checkpoint_env(monkeypatch):
    """Keep resolution tests independent from a developer's shell env."""
    monkeypatch.delenv("CBB_UQ_HEAD_CHECKPOINT", raising=False)


def _rows(n_positive: int = 12, n_negative: int = 12) -> list[dict]:
    rows: list[dict] = []
    for i in range(max(n_positive, n_negative)):
        if i < n_positive:
            rows.append({"record": dict(SUPPORTED, claim=SUPPORTED["claim"] + str(i)), "label": 1})
        if i < n_negative:
            rows.append({"record": dict(UNSUPPORTED, claim=UNSUPPORTED["claim"] + str(i)), "label": 0})
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _run_train(tmp_path: Path, rows: list[dict], *extra: str) -> tuple[int, Path]:
    data = _write_jsonl(tmp_path / "labels.jsonl", rows)
    out = tmp_path / "uq_head.json"
    rc = TRAINER.main(["--data", str(data), "--out", str(out), *extra])
    return rc, out


def _config(tmp_path: Path, checkpoint: str | None) -> Config:
    return Config(
        book_id="testbook",
        book_path=tmp_path / "book.txt",
        out_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        uq_head_checkpoint=checkpoint,
    )


# ── feature contract ───────────────────────────────────────────────────────
def test_feature_schema_is_stable():
    values = extract_features(SUPPORTED)
    assert len(values) == len(FEATURE_NAMES) == 12
    assert all(isinstance(v, float) for v in values)
    names = dict(zip(FEATURE_NAMES, values))
    assert names["claim_bigram_recall"] > 0.5
    assert names["evidence_empty"] == 0.0
    negative = dict(zip(FEATURE_NAMES, extract_features(UNSUPPORTED)))
    assert negative["evidence_empty"] == 1.0
    assert negative["claim_bigram_recall"] < names["claim_bigram_recall"]


# ── train → load → score ───────────────────────────────────────────────────
def test_train_then_load_then_score(tmp_path, capsys):
    rc, out = _run_train(tmp_path, _rows())
    assert rc == 0
    assert "TRAIN-UQ-HEAD" in capsys.readouterr().out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["kind"] == "logreg"
    assert payload["feature_schema"] == 1
    assert payload["features"] == list(FEATURE_NAMES)
    assert payload["n_positive"] == 12 and payload["n_rows"] == 24

    head = load_checkpoint(out)
    assert head.kind == "logreg" and head.feature_schema == 1

    positive = uq_heads.score(SUPPORTED, checkpoint=out)
    negative = uq_heads.score(UNSUPPORTED, checkpoint=out)
    assert positive.status == STATUS_OK
    assert negative.status == STATUS_OK
    assert positive.method == "uq_heads" and negative.method == "uq_heads"
    assert positive.scope == "claim"
    assert 0.0 <= positive.confidence <= 1.0
    assert 0.0 <= negative.confidence <= 1.0
    assert positive.confidence > negative.confidence
    assert positive.details["checkpoint_source"] == "explicit"
    assert positive.details["head_kind"] == "logreg"
    assert positive.details["feature_schema"] == 1


def test_training_is_reproducible(tmp_path):
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    rc_first, out_first = _run_train(first_dir, _rows())
    rc_second, out_second = _run_train(second_dir, _rows())
    assert rc_first == 0 and rc_second == 0
    assert out_first.read_bytes() == out_second.read_bytes()


def test_holdout_metrics_are_reported(tmp_path):
    rc, out = _run_train(tmp_path, _rows(), "--holdout", "0.25")
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"]["holdout"] is not None
    assert payload["metrics"]["holdout"]["n"] > 0
    assert uq_heads.score(SUPPORTED, checkpoint=out).status == STATUS_OK


# ── checkpoint resolution ──────────────────────────────────────────────────
def test_env_checkpoint_is_used(tmp_path, monkeypatch):
    _rc, out = _run_train(tmp_path, _rows())
    monkeypatch.setenv("CBB_UQ_HEAD_CHECKPOINT", str(out))
    result = uq_heads.score(SUPPORTED)
    assert result.status == STATUS_OK
    assert result.details["checkpoint_source"] == "env"
    assert result.details["head_kind"] == "logreg"


def test_config_checkpoint_is_used_without_env(tmp_path):
    _rc, out = _run_train(tmp_path, _rows())
    result = uq_heads.score(SUPPORTED, config=_config(tmp_path, str(out)))
    assert result.status == STATUS_OK
    assert result.details["checkpoint_source"] == "config"


def test_config_beats_env(tmp_path, monkeypatch):
    _rc, out = _run_train(tmp_path, _rows())
    monkeypatch.setenv("CBB_UQ_HEAD_CHECKPOINT", str(tmp_path / "missing.json"))
    result = uq_heads.score(SUPPORTED, config=_config(tmp_path, str(out)))
    assert result.status == STATUS_OK
    assert result.details["checkpoint_source"] == "config"


# ── missing / invalid production weights ───────────────────────────────────
def test_missing_production_checkpoint_is_external_not_faked(tmp_path, monkeypatch):
    missing = tmp_path / "weights" / "uq_head.json"
    monkeypatch.setenv("CBB_UQ_HEAD_CHECKPOINT", str(missing))
    result = uq_heads.score(SUPPORTED)
    assert result.status == STATUS_EXTERNAL
    assert result.confidence is None
    assert "checkpoint 不存在" in result.reason
    assert result.details["checkpoint"] == str(missing)
    assert result.details["checkpoint_source"] == "env"
    assert not uq_heads.available(config=_config(tmp_path, None))


def test_corrupt_checkpoint_is_external_not_faked(tmp_path):
    bad = tmp_path / "bad_head.json"
    bad.write_text('{"format": "not-a-uq-head"}', encoding="utf-8")
    result = uq_heads.score(SUPPORTED, checkpoint=bad)
    assert result.status == STATUS_EXTERNAL
    assert result.confidence is None
    assert "format" in result.reason


def test_feature_schema_mismatch_is_rejected(tmp_path):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "logreg",
        "feature_schema": 999,
        "features": list(FEATURE_NAMES),
        "mean": [0.0] * 12,
        "std": [1.0] * 12,
        "weights": [0.0] * 12,
        "bias": 0.0,
        "threshold": 0.5,
    }
    bad = tmp_path / "schema_mismatch.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UQHeadCheckpointError, match="feature_schema"):
        load_checkpoint(bad)
    result = uq_heads.score(SUPPORTED, checkpoint=bad)
    assert result.status == STATUS_EXTERNAL and result.confidence is None


# ── tool failure modes ─────────────────────────────────────────────────────
def test_one_class_dataset_is_rejected(tmp_path, capsys):
    rows = [{"record": SUPPORTED, "label": 1} for _ in range(6)]
    rc, _out = _run_train(tmp_path, rows)
    assert rc == 2
    assert "一类" in capsys.readouterr().out


def test_bad_jsonl_is_rejected(tmp_path, capsys):
    data = tmp_path / "labels.jsonl"
    data.write_text("not json at all\n", encoding="utf-8")
    out = tmp_path / "head.json"
    rc = TRAINER.main(["--data", str(data), "--out", str(out)])
    assert rc == 2
    assert "JSON" in capsys.readouterr().out


def test_missing_out_dir_is_rejected(tmp_path, capsys):
    data = _write_jsonl(tmp_path / "labels.jsonl", _rows())
    rc = TRAINER.main(["--data", str(data),
                       "--out", str(tmp_path / "no_such_dir" / "head.json")])
    assert rc == 2
    assert "输出目录" in capsys.readouterr().out


# ── availability ───────────────────────────────────────────────────────────
def test_available_helper_with_checkpoint(tmp_path):
    _rc, out = _run_train(tmp_path, _rows())
    assert uq_heads.available(checkpoint=out)
    assert not uq_heads.available()
    assert uq_heads.available(head=object())
    assert not uq_heads.available(checkpoint=tmp_path / "missing.json")
