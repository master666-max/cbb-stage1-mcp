"""White-box signals are reachable after all — measured, not assumed.

Two capabilities were written off as "blocked by the host transport": B3 SEPs and
B4 whitebox UQ. That was half right. The host *queue* carries only text, but the
local backend behind it does serve ``logprobs=true`` for standard autoregressive
models. Measured against LM Studio / gemma-3-1b-it:

    logprobs -> [-0.243, -0.039, -1.303, -0.031, ...]
    SEPs     -> status=ok confidence=0.716
    UQLM     -> status=ok confidence=0.716

The same backend refuses for a diffusion-style model ("the current context does
not support logits computation"), so the distinction between "no signal" and
"signal of zero" has to survive: a missing logprobs block returns a reason, never
an empty list.

Token accounting was broken the same way: the contract carried ``usage`` and the
gateway read it, but ``cbb_reply`` hardcoded ``None``, so 73/73 calls in a real
run had no token counts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from cbb_stage1.model_gateway import _usage_mapping

SERVER = Path(__file__).resolve().parents[2] / "tools" / "mcp_model_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("mcp_model_under_test", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── token accounting ──────────────────────────────────────────────────────────
def test_usage_mapping_accepts_a_dict() -> None:
    assert _usage_mapping({"prompt_tokens": 3}) == {"prompt_tokens": 3}


def test_usage_mapping_accepts_a_namespace() -> None:
    """Transports hand back SimpleNamespace; passing it through broke the writer."""
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7)
    assert _usage_mapping(usage) == {"prompt_tokens": 5, "completion_tokens": 7}


def test_usage_mapping_accepts_a_pydantic_model() -> None:
    class Usage:
        def model_dump(self):
            return {"total_tokens": 9}

    assert _usage_mapping(Usage()) == {"total_tokens": 9}


def test_usage_mapping_returns_none_when_absent() -> None:
    """Absent must stay distinguishable from zero."""
    assert _usage_mapping(None) is None


def test_cbb_reply_forwards_usage(tmp_path: Path) -> None:
    """The hardcoded None is what made cost_usd unmeasurable."""
    spec = importlib.util.spec_from_file_location(
        "mcp_pipeline_under_test",
        Path(__file__).resolve().parents[2] / "tools" / "mcp_pipeline_server.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    work = tmp_path / "work"
    runs = work / "plan" / "mcp-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "r1.json").write_text(
        json.dumps({"run_id": "r1", "work": str(work)}), encoding="utf-8"
    )

    result = module.cbb_reply(
        {
            "work": str(work),
            "run_id": "r1",
            "request_id": "req-1",
            "text": "answer",
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
    )
    assert result.get("accepted") is True, result

    written = json.loads(
        (work / "plan" / "host-model" / "responses" / "req-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["usage"] == {"prompt_tokens": 11, "completion_tokens": 4}


# ── white-box signals ─────────────────────────────────────────────────────────
def test_fetch_logprobs_reports_a_reason_when_the_backend_refuses(monkeypatch) -> None:
    """A diffusion-style engine refuses; that must be a reason, not an empty list."""
    module = _load_server()

    class FakeResponse:
        status_code = 400

        def raise_for_status(self):
            raise RuntimeError("the current context does not support logits computation")

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    module._http = lambda: type("H", (), {"Client": FakeClient})
    module._MODEL = "diffusion-model"
    values, reason = module._fetch_logprobs("prompt")
    assert values is None
    assert reason is not None
    assert "logits" in reason or "RuntimeError" in reason


def test_fetch_logprobs_returns_the_values_when_present(monkeypatch) -> None:
    module = _load_server()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": "甲", "logprob": -0.243},
                                {"token": "是", "logprob": -0.039},
                            ]
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    module._http = lambda: type("H", (), {"Client": FakeClient})
    module._MODEL = "gemma-3-1b-it"
    values, reason = module._fetch_logprobs("prompt")
    assert reason is None
    assert values == [-0.243, -0.039]


def test_real_logprobs_drive_seps_to_a_real_score() -> None:
    """The whole point: a logprob vector must produce a computed confidence."""
    from cbb_stage1.confidence import score

    record = {
        "assertion_id": "a:1",
        "predicate": "性质",
        "roles": {},
        "evidence": [],
        "confidence": 0.6,
        "confidence_method": "engine_self",
    }
    logprobs = [-0.243, -0.039, -1.303, -0.031, -0.836, -0.662]
    sep = score(record, method="sep", logprobs=logprobs)
    assert sep.status == "ok"
    assert sep.confidence is not None
    assert 0.0 <= sep.confidence <= 1.0
    assert (sep.details or {}).get("approximation") == "sep_logprob_geom_mean"

    uqlm = score(record, method="uqlm_whitebox", logprobs=logprobs)
    assert uqlm.status == "ok"
    assert uqlm.confidence == sep.confidence
