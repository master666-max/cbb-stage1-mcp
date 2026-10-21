from __future__ import annotations

import asyncio

import pytest

from cbb_stage1.agents.host_queue import (
    HostQueueClient,
    HostQueueTimeout,
    build_host_queue_client,
)


def test_host_queue_round_trip(tmp_path):
    client = HostQueueClient(tmp_path / "queue", timeout=5, poll=0.01)

    async def run():
        task = asyncio.create_task(client.generate("sys", "prompt", max_tokens=8))
        await asyncio.sleep(0.05)
        pending = client.drain()
        assert len(pending) == 1
        assert pending[0]["prompt"] == "prompt"
        client.reply(pending[0]["request_id"], "host-answer", usage={"tokens": 1})
        return await task

    text, usage = asyncio.run(run())
    assert text == "host-answer"
    assert usage == {"tokens": 1}


def test_host_queue_times_out_without_reply(tmp_path):
    client = HostQueueClient(tmp_path / "queue", timeout=0.05, poll=0.01)
    with pytest.raises(HostQueueTimeout):
        asyncio.run(client.generate("s", "p"))


def test_host_queue_disabled_without_env(monkeypatch):
    monkeypatch.delenv("CBB_HOST_MODEL_QUEUE", raising=False)
    assert build_host_queue_client(None) is None
