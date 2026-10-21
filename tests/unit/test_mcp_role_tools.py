"""MCP scheduling surface for the roles that must NOT reuse the main model.

Three capabilities were previously unreachable through MCP: NLI entailment (D11),
a judge model (B4 judge), and token-level signals (B3 SEPs / B4 whitebox). The
algorithms existed; there was simply no tool to ask for them.

The design rule these tests pin: **a fallback must be labelled, and a missing
second model must refuse rather than borrow the main one.** A general chat model
answering "does A entail B" is not an NLI model, and reusing the extracting model
as judge turns §5.15's independence requirement into a self-assessment that still
reports a number.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[2] / "tools" / "mcp_model_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("mcp_model_under_test", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_surface_exposes_the_three_new_roles():
    module = _load()
    names = {tool["name"] for tool in module.TOOLS}
    assert {"model_generate", "model_embed"} <= names
    assert {"model_nli", "model_judge", "model_generate_whitebox"} <= names


def test_judge_refuses_without_a_second_model():
    """No fallback: borrowing the main model would defeat the check."""
    module = _load()
    module._JUDGE_BASE = ""
    module._JUDGE_MODEL = ""
    result = module._dispatch("model_judge", {"claim": "甲是铁匠"})
    assert result["isError"] is True
    payload = result["structuredContent"]
    assert payload["score"] is None
    assert "not configured" in payload["error"]


def test_nli_labels_a_main_model_fallback():
    """Falling back is allowed; pretending it is an NLI model is not."""
    module = _load()
    module._NLI_BASE = ""
    module._NLI_MODEL = ""
    calls: list[tuple[str, str]] = []

    def fake_post(base, model, system, prompt, *, timeout):
        calls.append((base, model))
        return "yes"

    module._post_chat = fake_post
    module._discover_model = lambda kind: "main-chat-model"
    result = module._dispatch(
        "model_nli", {"premise": "A 先于 B", "hypothesis": "A 在 B 之前"}
    )
    payload = result["structuredContent"]
    assert payload["result"] == "yes"
    assert payload["verifier"] == "fallback-main-model"
    assert payload["model"] == "main-chat-model"


def test_nli_prefers_the_dedicated_endpoint():
    module = _load()
    module._NLI_BASE = "http://127.0.0.1:9999/v1"
    module._NLI_MODEL = "dedicated-nli-model"
    seen: list[str] = []

    def fake_post(base, model, system, prompt, *, timeout):
        seen.append(f"{base}|{model}")
        return "no"

    module._post_chat = fake_post
    result = module._dispatch(
        "model_nli", {"premise": "A 先于 B", "hypothesis": "A 在 B 之后"}
    )
    payload = result["structuredContent"]
    assert payload["verifier"] == "dedicated-nli"
    assert payload["result"] == "no"
    assert seen and "dedicated-nli-model" in seen[0]


def test_nli_unparseable_answer_is_unknown_not_guessed():
    module = _load()
    module._NLI_BASE = ""
    module._NLI_MODEL = ""
    module._post_chat = lambda *a, **k: "I think maybe it could be either"
    module._discover_model = lambda kind: "m"
    result = module._dispatch("model_nli", {"premise": "p", "hypothesis": "h"})
    assert result["structuredContent"]["result"] == "unknown"


def test_nli_backend_failure_is_reported():
    module = _load()
    module._NLI_BASE = ""
    module._NLI_MODEL = ""

    def boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    module._post_chat = boom
    module._discover_model = lambda kind: "m"
    result = module._dispatch("model_nli", {"premise": "p", "hypothesis": "h"})
    assert result["isError"] is True
    assert "connection refused" in str(result["structuredContent"].get("error"))


def test_whitebox_reports_missing_signals_with_a_reason():
    """'no signal' and 'signal is zero' must not look alike."""
    module = _load()

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": "甲是铁匠"},
                        "logprobs": None,
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
    module._MODEL = "m"
    result = module._dispatch(
        "model_generate_whitebox",
        {"prompt": "hi", "want": ["logprobs", "hidden_states"]},
    )
    payload = result["structuredContent"]
    assert payload["text"] == "甲是铁匠"
    assert payload["signals"] is None
    unavailable = payload["signals_unavailable"]
    assert "logprobs" in unavailable
    assert "hidden_states" in unavailable
    assert "not exposed" in unavailable["hidden_states"]


def test_whitebox_passes_logprobs_through_when_present():
    module = _load()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": "甲是铁匠"},
                        "logprobs": {
                            "content": [
                                {"token": "甲", "logprob": -0.1},
                                {"token": "是", "logprob": -0.2},
                            ]
                        },
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
    module._MODEL = "m"
    result = module._dispatch(
        "model_generate_whitebox", {"prompt": "hi", "want": ["logprobs"]}
    )
    payload = result["structuredContent"]
    assert payload["signals"]["logprobs"] == pytest.approx([-0.1, -0.2])
    assert payload["signals_unavailable"] is None
