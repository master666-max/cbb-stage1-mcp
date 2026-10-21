"""In-conversation subagents backed by an external inference endpoint.

The pipeline's auxiliary passes (verifier / reasoner / description / community)
expect synchronous ``generate``/``sample`` hooks.  This package turns an
external LLM endpoint into a *conversation-scoped subagent*: every call is a
message turn, the subagent owns its role prompt and transcript, and the result
is a structured ``SubagentResult`` that can also produce a file-handoff receipt.

The endpoint itself is still the provider from :mod:`cbb_stage1.engines`; this is
only the conversion layer, not a second model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

__all__ = [
    "ROLE_PROMPTS",
    "SubagentMessage",
    "SubagentTask",
    "SubagentResult",
    "EndpointSubagent",
    "SubagentLLM",
    "SubagentPool",
    "build_subagent_pool",
]

ROLE_PROMPTS: Mapping[str, str] = {
    "assistant": "你是小说抽取流水线的辅助推理子代理。只返回请求要求的 JSON 或短文本，不编造原文。",
    "reasoner": "你是断言核验/溯因推理子代理。只依据给定证据，输出严格 JSON；无法判断就返回 unverifiable。",
    "verifier": "你是独立验证子代理。只依据原文与 span 判断 support/against/unsure，不信任抽取器的引文。",
    "resolver": "你是身份消歧子代理。只在候选名中选择规范名，不发明新名字。",
    "arbiter": "你是冲突裁决子代理。只输出裁决与理由，不删除、不覆盖原记录。",
    "abducer": "你是溯因子代理。只提出缺失前提/候选假设，标注证据不足。",
    "summarizer": "你是摘要子代理。只压缩已有观察，不补充原文没有的事实。",
}


@dataclass
class SubagentMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class SubagentTask:
    task_id: str
    role: str
    instruction: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    unit_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "unit_id": self.unit_id,
            "instruction": self.instruction,
            "payload": dict(self.payload),
        }


@dataclass
class SubagentResult:
    task_id: str
    role: str
    status: str  # ok | abstained | failed
    text: str
    structured: Any = None
    error: str | None = None
    transcript: list[SubagentMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "status": self.status,
            "text": self.text,
            "structured": self.structured,
            "error": self.error,
            "transcript": [message.to_dict() for message in self.transcript],
        }


class EndpointSubagent:
    """One role-bound conversation over an external endpoint."""

    def __init__(
        self,
        role: str,
        generate: Callable[[str], str],
        *,
        system: str | None = None,
        max_transcript: int = 40,
    ) -> None:
        if role not in ROLE_PROMPTS:
            raise ValueError(f"unknown subagent role {role!r}")
        self.role = role
        self._generate = generate
        self.system = system or ROLE_PROMPTS[role]
        self.max_transcript = max(2, int(max_transcript))
        self.transcript: list[SubagentMessage] = [
            SubagentMessage("system", self.system)
        ]

    def _prompt(self, task: SubagentTask) -> str:
        return (
            f"系统角色：{self.system}\n"
            f"任务：{task.instruction}\n"
            f"输入：{json.dumps(dict(task.payload), ensure_ascii=False, sort_keys=True)}"
        )

    def run(self, task: SubagentTask) -> SubagentResult:
        prompt = self._prompt(task)
        self.transcript.append(SubagentMessage("user", prompt))
        try:
            text = str(self._generate(prompt))
        except Exception as exc:  # noqa: BLE001 - endpoint failure is a result
            self.transcript.append(SubagentMessage("assistant", f"ERROR: {exc}"))
            return SubagentResult(
                task.task_id,
                task.role,
                "failed",
                "",
                error=f"{type(exc).__name__}: {exc}",
                transcript=list(self.transcript),
            )
        self.transcript.append(SubagentMessage("assistant", text))
        if len(self.transcript) > self.max_transcript:
            self.transcript = [self.transcript[0]] + self.transcript[-(self.max_transcript - 1):]
        structured = _parse_json(text)
        return SubagentResult(
            task.task_id,
            task.role,
            "abstained" if not text.strip() else "ok",
            text,
            structured=structured,
            transcript=list(self.transcript),
        )

    def __call__(self, record: Mapping[str, Any]) -> str:
        """Verifier-compatible callable: record -> support/against/unsure."""
        task = SubagentTask(
            task_id=f"verify:{record.get('unit_id', '')}",
            role="verifier",
            instruction="根据原文与 span 判断候选引文是否支持该断言；只输出 support/against/unsure。",
            payload=dict(record),
            unit_id=str(record.get("unit_id") or "") or None,
        )
        result = self.run(task)
        text = result.text.strip().lower()
        for verdict in ("support", "against", "unsure"):
            if verdict in text:
                return verdict
        return "unsure"


class SubagentLLM:
    """Adapter so existing ``generate``/``sample`` call sites use a subagent."""

    def __init__(self, subagent: EndpointSubagent) -> None:
        self.subagent = subagent
        self.provider = f"subagent:{subagent.role}"

    def generate(self, prompt: str) -> str:
        task = SubagentTask(
            task_id=f"{self.subagent.role}:call",
            role=self.subagent.role,
            instruction="回答下面的输入；遵守系统角色约束。",
            payload={"prompt": prompt},
        )
        return self.subagent.run(task).text

    def sample(self, prompt: str, n: int) -> list[str]:
        return [self.generate(prompt) for _ in range(max(1, int(n)))]


class SubagentPool:
    """Role -> subagent; empty when the run is mock/offline."""

    def __init__(self, llm: Any | None = None) -> None:
        self._agents: dict[str, EndpointSubagent] = {}
        if llm is not None:
            for role in ROLE_PROMPTS:
                self._agents[role] = EndpointSubagent(role, llm.generate)

    @classmethod
    def from_gateway(cls, gateway: Any) -> "SubagentPool":
        """One role-specific sync client per role (role_models policy honored)."""
        pool = cls()
        for role in ROLE_PROMPTS:
            pool._agents[role] = EndpointSubagent(
                role, gateway.sync_llm(role).generate
            )
        return pool

    def get(self, role: str) -> SubagentLLM | None:
        agent = self._agents.get(role)
        return SubagentLLM(agent) if agent is not None else None

    def verifier(self) -> EndpointSubagent | None:
        return self._agents.get("verifier")

    def __bool__(self) -> bool:
        return bool(self._agents)

    def __len__(self) -> int:
        return len(self._agents)


def build_subagent_pool(cfg: Any, gateway: Any = None) -> SubagentPool:
    """Build the role pool from the configured gateway/endpoint (mock => empty)."""
    if gateway is not None:
        return SubagentPool.from_gateway(gateway)
    from cbb_stage1.orchestrator.llm_bridge import build_aux_llm

    return SubagentPool(build_aux_llm(cfg))


def _parse_json(text: str) -> Any:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
