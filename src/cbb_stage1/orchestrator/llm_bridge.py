"""Bridge real provider clients to the synchronous hooks used by postprocess.

The audit found several adapters that were defined but never wired: confidence
sampling, reasoning decomposition, description synthesis, NLI, etc.  This module
is the single place that turns the configured provider into synchronous
callables; it returns ``None`` for mock/offline runs so no network call is ever
made by accident.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Sequence

from cbb_stage1.config import Config
from cbb_stage1.engines._async import run_coroutine_sync
from cbb_stage1.engines.ai_reader.llm_client import (
    get_llm_client,
    load_settings_from_env,
)

__all__ = [
    "SyncLLM",
    "ReasoningLLM",
    "VerifierLLM",
    "build_aux_llm",
    "build_reasoning_llm",
    "build_verifier_llm",
    "build_embedder",
]

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _json_value(text: str, default: Any) -> Any:
    raw = text.strip()
    match = _JSON_BLOCK.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001 - malformed model output is not a crash
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:  # noqa: BLE001
                return default
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:  # noqa: BLE001
                return default
        return default


class SyncLLM:
    """A small synchronous facade over the async provider client."""

    def __init__(self, client: Any, *, system: str = "You are a careful extraction verifier.") -> None:
        self.client = client
        self.system = system

    def generate(self, prompt: str) -> str:
        result = run_coroutine_sync(self.client.generate(self.system, prompt))
        text = result[0] if isinstance(result, tuple) else result
        if isinstance(text, dict):
            return json.dumps(text, ensure_ascii=False)
        return str(text)

    def sample(self, prompt: str, n: int) -> list[str]:
        return [self.generate(prompt) for _ in range(max(1, int(n)))]


class ReasoningLLM:
    """Implement the AFEV / VeGraph / Abduction protocols over one SyncLLM."""

    def __init__(self, llm: SyncLLM) -> None:
        self.llm = llm

    def decompose(self, claim: str, context: str | None) -> Sequence[str]:
        prompt = (
            "把断言拆成可逐条核验的原子事实。只输出 JSON 数组，每项一句话。\n"
            f"断言：{claim}\n上下文：{context or '无'}"
        )
        value = _json_value(self.llm.generate(prompt), [])
        return [str(item) for item in value if str(item).strip()][:12] if isinstance(value, list) else []

    def verify(self, fact: str, evidence: Sequence[str]) -> dict[str, Any]:
        prompt = (
            "只根据给定证据核验原子事实。输出 JSON: "
            '{"verdict":"true|false|unverifiable","rationale":"..."}\n'
            f"事实：{fact}\n证据：\n" + "\n".join(f"- {item}" for item in evidence)
        )
        value = _json_value(self.llm.generate(prompt), {})
        if not isinstance(value, dict):
            return {"verdict": "unverifiable", "rationale": "bad model output"}
        verdict = str(value.get("verdict") or "unverifiable").lower()
        if verdict not in {"true", "false", "unverifiable"}:
            verdict = "unverifiable"
        return {"verdict": verdict, "rationale": value.get("rationale")}

    def triples(self, claim: str) -> Sequence[dict[str, Any]]:
        prompt = (
            "把断言拆成 {subject, predicate, object} 三元组，只输出 JSON 数组。\n"
            f"断言：{claim}"
        )
        value = _json_value(self.llm.generate(prompt), [])
        if not isinstance(value, list):
            return []
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and item.get("subject") and item.get("predicate") and item.get("object"):
                out.append({key: str(item[key]) for key in ("subject", "predicate", "object")})
        return out

    def verify_subassertion(self, triple: dict[str, Any]) -> str:
        prompt = (
            "只输出 support/refute/irrelevant/unanswerable 之一。\n"
            f"三元组：{json.dumps(triple, ensure_ascii=False)}"
        )
        value = self.llm.generate(prompt).strip().lower()
        for stance in ("support", "refute", "irrelevant", "unanswerable"):
            if stance in value:
                return stance
        return "unanswerable"

    def missing_premises(self, claim: str, evidence: Sequence[str]) -> Sequence[str]:
        prompt = (
            "指出要判定该断言还缺哪些原文前提。只输出 JSON 数组，每项一句话。\n"
            f"断言：{claim}\n证据：\n" + "\n".join(f"- {item}" for item in evidence)
        )
        value = _json_value(self.llm.generate(prompt), [])
        return [str(item) for item in value if str(item).strip()][:8] if isinstance(value, list) else []

    def score(self, claim: str, premises: Sequence[str]) -> dict[str, Any]:
        prompt = (
            "给假设打分，只输出 JSON: "
            '{"consistency":0..1,"plausibility":0..1}\n'
            f"断言：{claim}\n前提：\n" + "\n".join(f"- {item}" for item in premises)
        )
        value = _json_value(self.llm.generate(prompt), {})
        out: dict[str, Any] = {}
        for key in ("consistency", "plausibility"):
            try:
                number = float((value or {}).get(key, 0.0))
            except (TypeError, ValueError):
                number = 0.0
            out[key] = min(1.0, max(0.0, number))
        return out


def build_aux_llm(cfg: Config, gateway: Any = None) -> Any:
    """Return a synchronous LLM for auxiliary passes, or ``None`` for mock."""
    if gateway is not None:
        return gateway.sync_llm("assistant")
    provider = (cfg.provider or "").strip().lower()
    if provider in {"", "mock"} or (cfg.model or "").strip().lower() == "mock":
        return None
    try:
        settings = load_settings_from_env()
        if cfg.model and cfg.model != "mock":
            settings.model = cfg.model
        if provider not in {"ai_reader", "ai-reader"}:
            settings.provider = provider
        client = get_llm_client(settings)
        return SyncLLM(client)
    except Exception:  # noqa: BLE001 - explicit offline/unconfigured state
        return None


def build_reasoning_llm(llm: SyncLLM | None) -> ReasoningLLM | None:
    return ReasoningLLM(llm) if llm is not None else None


class VerifierLLM:
    """Callable expected by ``role_runtime.verify_evidence``."""

    def __init__(self, llm: SyncLLM) -> None:
        self.llm = llm

    def __call__(self, record: dict[str, Any]) -> str:
        prompt = (
            "只根据原文片段判断候选引文是否支持该断言。"
            "只输出 support/against/unsure 之一。\n"
            f"原文：{record.get('text', '')}\n"
            f"span：{json.dumps(record.get('span'), ensure_ascii=False)}\n"
            f"候选引文：{record.get('quote_expected')}\n"
            f"实际片段：{record.get('quote_observed')}"
        )
        answer = self.llm.generate(prompt).strip().lower()
        for verdict in ("support", "against", "unsure"):
            if verdict in answer:
                return verdict
        return "unsure"


class JudgeLLM:
    """Callable that scores claim support in ``[0, 1]`` (B4 judge class).

    Separate from :class:`VerifierLLM` on purpose. A verifier answers "does this
    quote support this assertion" for the *extractor's own* candidate — it can run
    on the same model without much harm because it is checking a span, not judging
    a claim. A judge evaluates whether the evidence justifies the claim, and
    §5.15 requires that judgement not come from the model that produced it.

    Returns ``None`` when the model's answer cannot be parsed as a number in
    ``[0, 1]``. That is deliberate: a judge that rambles should be recorded as
    "no score", never rounded into a confident-looking 0.5.
    """

    def __init__(self, llm: SyncLLM) -> None:
        self.llm = llm

    def __call__(self, record: dict[str, Any]) -> float | None:
        claim = str(record.get("claim") or record.get("normalized_claim") or "")
        evidence = str(record.get("evidence") or record.get("quote") or "")
        prompt = (
            "判断给出的证据在多大程度上支持这条主张。"
            "只输出一个 0 到 1 之间的小数，不要输出其它内容。\n"
            f"主张：{claim}\n证据：{evidence}"
        )
        raw = self.llm.generate(prompt).strip()
        import re as _re

        match = _re.search(r"(\d+(?:\.\d+)?)", raw)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        if not 0.0 <= value <= 1.0:
            return None
        return value


def build_http_llm(base_url: str, model: str, *, api_key: str = "") -> SyncLLM:
    """A synchronous LLM pointed at an explicit OpenAI-compatible endpoint.

    Used for the roles that must not share the main model (B4 judge, dedicated
    NLI). The endpoint is given, not inferred, so a misconfiguration fails loudly
    at construction rather than quietly reusing the main model.
    """
    from cbb_stage1.engines.ai_reader.llm_client import (
        LLMSettings,
        OpenAICompatibleClient,
        SyncLLM,
    )

    settings = LLMSettings(
        provider="openai",
        provider_format="openai",
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
    )
    return SyncLLM(OpenAICompatibleClient(settings))


def build_judge_llm(llm: SyncLLM | None) -> JudgeLLM | None:
    return JudgeLLM(llm) if llm is not None else None


def build_verifier_llm(llm: SyncLLM | None) -> VerifierLLM | None:
    return VerifierLLM(llm) if llm is not None else None


def build_embedder(
    cfg: Config, gateway: Any = None
) -> Callable[[list[str]], list[list[float]]] | None:
    """Build an OpenAI-compatible embedder only when explicitly configured."""
    if gateway is not None:
        return gateway.embed
    model = os.environ.get("CBB_EMBEDDING_MODEL", "").strip()
    if not model:
        return None
    base = os.environ.get("CBB_EMBEDDING_BASE_URL") or "http://127.0.0.1:8080/v1"
    key = os.environ.get("CBB_EMBEDDING_API_KEY") or os.environ.get("CBB_LLM_API_KEY", "")
    if not base:
        return None

    def embed(texts: list[str]) -> list[list[float]]:
        try:
            import httpx
        except ImportError:  # pragma: no cover - dependency is declared
            return []
        # Same configurable budget as ``ModelGateway.embed``: a cold local
        # embedding model can exceed the old hard-coded 60 s.
        from cbb_stage1.model_gateway import embedding_timeout

        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=embedding_timeout()) as client:
            response = client.post(
                f"{base.rstrip('/')}/embeddings",
                headers=headers,
                json={"model": model, "input": list(texts)},
            )
            response.raise_for_status()
            payload = response.json()
        return [list(map(float, item["embedding"])) for item in payload.get("data", [])]

    return embed
