"""The verifier arrives in shapes that cannot all answer a text question.

Measured on spk10: every speaker check recorded

    verifier failed: AttributeError: 'EndpointSubagent' object has no attribute 'generate'

because the module assumed a `generate` interface while the pipeline passes an
`EndpointSubagent`. The codebase already had the adapter for that shape
(`agents.subagent.SubagentLLM`), so the fix is to use it rather than to invent one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cbb_stage1.agents.subagent import EndpointSubagent, SubagentLLM, SubagentResult, SubagentTask
from cbb_stage1.validate.integration import _speaker_asker


class FakeSyncLLM:
    """Shape 1: already has `generate`."""

    def generate(self, prompt: str) -> str:  # noqa: ANN001
        return "support"


def test_an_llm_with_generate_is_used_directly() -> None:
    llm = FakeSyncLLM()
    asker, same_model = _speaker_asker({"verifier_llm": llm})
    assert asker is llm
    assert same_model is False, "a dedicated verifier is not the same model"


def test_an_endpoint_subagent_is_wrapped_in_the_existing_adapter() -> None:
    """Shape 2: `EndpointSubagent` has `run`, and `SubagentLLM` bridges it."""
    subagent = EndpointSubagent(role="verifier", generate=lambda prompt: "support")
    asker, same_model = _speaker_asker({"verifier_llm": subagent})
    assert isinstance(asker, SubagentLLM), "must reuse the shipped adapter"
    assert same_model is False
    assert asker.generate("任意问题") == "support"


def test_it_falls_back_to_the_aux_model_and_says_so() -> None:
    """No adaptor available: run on the extractor's model, and report that."""
    llm = FakeSyncLLM()
    asker, same_model = _speaker_asker({"verifier_llm": object(), "llm": llm})
    assert asker is llm
    assert same_model is True, "§5.15 independence must not be claimed here"


def test_nothing_usable_yields_no_asker() -> None:
    asker, same_model = _speaker_asker({})
    assert asker is None
    assert same_model is False


def test_the_wrapped_subagent_actually_reaches_the_model() -> None:
    """End to end through the adapter: a prompt in, the model's text out."""
    seen: list[str] = []

    def generate(prompt: str) -> str:
        seen.append(prompt)
        return '{"verdict": "support", "rationale": "ok"}'

    subagent = EndpointSubagent(role="verifier", generate=generate)
    asker, _ = _speaker_asker({"verifier_llm": subagent})
    answer = asker.generate("引文是谁说的？")
    assert "引文是谁说的？" in seen[0]
    assert "support" in answer


def test_subagent_result_shape_is_what_the_adapter_reads() -> None:
    """Guard the contract the adapter depends on."""
    result = SubagentResult(task_id="t", role="verifier", status="ok", text="support")
    assert result.text == "support"
    assert SubagentTask(task_id="t", role="verifier", instruction="i").instruction == "i"