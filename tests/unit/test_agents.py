from __future__ import annotations

from types import SimpleNamespace

from cbb_stage1.agents import (
    EndpointSubagent,
    SubagentLLM,
    SubagentPool,
    SubagentTask,
    build_subagent_pool,
)


class _OneShot:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.replies.pop(0) if self.replies else "unsure"


def test_endpoint_subagent_keeps_transcript_and_verdict():
    agent = EndpointSubagent("verifier", _OneShot(["support", "support"]).generate)
    result = agent.run(
        SubagentTask(
            task_id="verify:u1",
            role="verifier",
            instruction="判断引文",
            payload={"unit_id": "u1", "quote": "原文"},
        )
    )
    assert result.status == "ok"
    assert result.text == "support"
    assert result.transcript[0].role == "system"
    assert result.transcript[-1].content == "support"
    assert agent({"unit_id": "u1", "quote": "原文"}) == "support"


def test_subagent_llm_exposes_generate_and_sample():
    agent = EndpointSubagent("assistant", _OneShot(["a", "a", "b"]).generate)
    llm = SubagentLLM(agent)
    assert llm.generate("p") == "a"
    assert llm.sample("p", 2) == ["a", "b"]


def test_pool_roles_are_independent_and_empty_for_mock():
    pool = SubagentPool(SimpleNamespace(generate=lambda prompt: "support"))
    assert pool.get("reasoner") is not None
    assert pool.get("verifier") is not None
    assert len(pool) == 7
    empty = build_subagent_pool(SimpleNamespace(provider="mock", model="mock"))
    assert not empty
    assert empty.get("assistant") is None

