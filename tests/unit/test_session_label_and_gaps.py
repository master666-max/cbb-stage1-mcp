"""The main reasoning model changes with the agent, so it must be declared.

The host transport means "the agent answers". The same run may be served by one
model today and another tomorrow, but the gateway hardcoded ``model="session"``,
so ``model-calls.jsonl`` could not answer "which model produced this box" and the
D4 decoupling check could not even be evaluated — two calls might be the same
model or different ones and the audit could not tell.

``session_model_label`` is the caller's declaration. It is not probed: the host
protocol does not expose model identity, and inventing one would be worse than
recording nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.box.writer import _capability_gaps
from cbb_stage1.model_gateway import ModelPolicy


def test_policy_defaults_to_no_label() -> None:
    policy = ModelPolicy.from_dict({})
    assert policy.session_model_label == ""
    assert policy.judge_model == ""
    assert policy.nli_model == ""
    assert policy.whitebox_backend is False


def test_policy_reads_the_new_declarations() -> None:
    policy = ModelPolicy.from_dict(
        {
            "session_model_label": "deepseek-v4.1-flash",
            "judge_base_url": "http://127.0.0.1:9000/v1",
            "judge_model": "qwen3-4b",
            "nli_base_url": "http://127.0.0.1:9100/v1",
            "nli_model": "nli-zh-small",
            "whitebox_backend": True,
        }
    )
    assert policy.session_model_label == "deepseek-v4.1-flash"
    assert policy.judge_model == "qwen3-4b"
    assert policy.nli_model == "nli-zh-small"
    assert policy.whitebox_backend is True


def test_capability_gaps_names_every_unavailable_mechanism(tmp_path: Path) -> None:
    """The gap list used to be assembled by hand from separate receipts."""
    calls = tmp_path / "plan" / "model-calls.jsonl"
    calls.parent.mkdir(parents=True, exist_ok=True)
    calls.write_text(
        "\n".join(
            [
                json.dumps({"kind": "chat", "role": "extraction", "model": "session"}),
                json.dumps({"kind": "chat", "role": "verifier", "model": "session"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    soft = {"model_decoupled": False}
    gaps = _capability_gaps(tmp_path, soft)

    assert "b3_sep" in gaps
    assert "no token logprobs" in gaps["b3_sep"]["reason"]
    assert "b4_uqlm_whitebox" in gaps
    assert "b4_uqlm_judge" in gaps
    assert "second model" in gaps["b4_uqlm_judge"]["reason"]
    assert gaps["b5_uq_heads"]["status"] == "external"
    assert gaps["d4_decoupled"]["status"] == "false"
    assert "§5.15" in gaps["d4_decoupled"]["reason"]


def test_capability_gaps_clears_when_signals_were_present(tmp_path: Path) -> None:
    calls = tmp_path / "plan" / "model-calls.jsonl"
    calls.parent.mkdir(parents=True, exist_ok=True)
    calls.write_text(
        "\n".join(
            [
                json.dumps({"kind": "chat", "role": "extraction", "method": "sep"}),
                json.dumps({"kind": "chat", "role": "judge", "method": "claim_qa"}),
                json.dumps({"kind": "chat", "role": "extraction", "method": "uq_heads"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gaps = _capability_gaps(tmp_path, {"model_decoupled": True})
    assert "b3_sep" not in gaps
    assert "b4_uqlm_judge" not in gaps
    assert "b5_uq_heads" not in gaps
    assert "d4_decoupled" not in gaps


def test_capability_gaps_reports_not_measured_without_verifier(tmp_path: Path) -> None:
    calls = tmp_path / "plan" / "model-calls.jsonl"
    calls.parent.mkdir(parents=True, exist_ok=True)
    calls.write_text(
        json.dumps({"kind": "chat", "role": "extraction", "model": "session"}) + "\n",
        encoding="utf-8",
    )
    gaps = _capability_gaps(tmp_path, {"model_decoupled": None})
    assert gaps["d4_decoupled"]["status"] == "not measured"
