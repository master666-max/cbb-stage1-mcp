from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cbb_stage1.orchestrator.llm_bridge import (
    ReasoningLLM,
    SyncLLM,
    build_aux_llm,
    build_embedder,
)


class _FakeClient:
    async def generate(self, system, prompt):
        if "三元组" in prompt:
            return json.dumps([{"subject": "A", "predicate": "认识", "object": "B"}]), None
        if "拆成" in prompt:
            return json.dumps(["原子事实一", "原子事实二"]), None
        if "打分" in prompt:
            return json.dumps({"consistency": 0.8, "plausibility": 0.7}), None
        return "support", None


def test_sync_llm_and_reasoning_adapters():
    llm = SyncLLM(_FakeClient())
    assert llm.generate("随便") == "support"
    assert llm.sample("随便", 2) == ["support", "support"]
    reasoning = ReasoningLLM(llm)
    assert reasoning.decompose("断言", None) == ["原子事实一", "原子事实二"]
    assert reasoning.triples("断言")[0]["predicate"] == "认识"
    assert reasoning.verify("事实", ["证据"])["verdict"] == "unverifiable"
    assert reasoning.score("断言", ["前提"]) == {"consistency": 0.8, "plausibility": 0.7}


def test_mock_config_never_builds_network_client():
    cfg = SimpleNamespace(provider="mock", model="mock")
    assert build_aux_llm(cfg) is None
    assert build_embedder(cfg) is None


def test_embedder_requires_explicit_env(monkeypatch):
    cfg = SimpleNamespace(provider="openai", model="m")
    monkeypatch.delenv("CBB_EMBEDDING_MODEL", raising=False)
    assert build_embedder(cfg) is None
    monkeypatch.setenv("CBB_EMBEDDING_MODEL", "e")
    monkeypatch.setenv("CBB_EMBEDDING_BASE_URL", "http://127.0.0.1:1")
    assert build_embedder(cfg) is not None
