from __future__ import annotations

import asyncio

from cbb_stage1.temporal.nli_client import build_nli


class FakeHostClient:
    async def generate(self, system, prompt, **kwargs):
        return "yes", None


def test_nli_host_provider_uses_injected_client():
    nli = build_nli({}, client=FakeHostClient(), provider="host")
    result = nli("ev:a", "ev:b", "ev:a 是否先于 ev:b？")
    assert result.result == "yes"
    assert result.verifier.startswith("nli_client:host")
