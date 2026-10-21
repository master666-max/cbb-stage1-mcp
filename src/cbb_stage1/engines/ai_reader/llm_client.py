"""Async LLM clients: Ollama / OpenAI-compatible / Anthropic (httpx only).

Ported from B06-AI-Reader-V2 `infra/llm_client.py`, `infra/openai_client.py` and
`infra/anthropic_client.py`, with two deliberate changes:

* no ``src.infra.config`` import — provider settings arrive as an explicit
  :class:`LLMSettings` (or from ``CBB_LLM_*`` environment variables);
* no FastAPI/DB coupling — this module only knows ``generate()``.

All three clients expose the same coroutine::

    await client.generate(system=..., prompt=..., format=None, temperature=0.1,
                          max_tokens=4096, timeout=120, num_ctx=None)
    -> (content: str | dict, usage: LlmUsage)

``format`` triggers JSON parsing on the client side (and native JSON mode where
the provider supports it).  Clients set ``is_cloud = True/False`` so the
extractor can decide whether to embed the JSON schema in the system prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from cbb_stage1.engines.ai_reader.budget import DEFAULT_CONTEXT_WINDOW

logger = logging.getLogger(__name__)


# ── Usage / errors ────────────────────────────────────────────────


@dataclass
class LlmUsage:
    """Token usage from one LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "LlmUsage") -> "LlmUsage":
        return LlmUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMError(Exception):
    """Base exception for LLM client errors."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""


class LLMParseError(LLMError):
    """Raised when JSON parsing of an LLM response fails."""


# ── Response cleanup helpers ──────────────────────────────────────


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks, including unclosed (truncated) ones."""
    result = re.sub(r"<think>[\s\S]*?</think>", "", text)
    result = re.sub(r"<think>[\s\S]*$", "", result)
    return result


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may carry fences or commentary."""
    cleaned = _strip_thinking(text)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            loaded = json.loads(match.group())
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

    raise LLMParseError(f"Failed to extract JSON from LLM response: {cleaned[:200]}...")


def _repair_truncated_json(text: str) -> str:
    """Repair JSON cut off at max_tokens by trimming to a safe cut and closing.

    Ported verbatim in behaviour from ``openai_client._repair_truncated_json``:
    scan for positions after complete array/object elements, trim there, drop
    trailing commas / orphan keys, then append the missing closers.
    """
    text = text.rstrip()

    safe_cuts: list[int] = []
    stack: list[str] = []  # '{' or '['
    in_string = False
    escape = False
    after_colon = False

    i = 0
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
            i += 1
            continue
        if ch == "\\" and in_string:
            escape = True
            i += 1
            continue
        if ch == '"':
            if in_string:
                in_string = False
                if after_colon or (stack and stack[-1] == "["):
                    safe_cuts.append(i + 1)
                    after_colon = False
            else:
                in_string = True
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if ch in (" ", "\n", "\r", "\t"):
            i += 1
            continue
        if ch in ("{", "["):
            stack.append(ch)
            after_colon = False
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            safe_cuts.append(i + 1)
            after_colon = False
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
            safe_cuts.append(i + 1)
            after_colon = False
        elif ch == ":":
            after_colon = True
        elif ch == ",":
            after_colon = False
        elif after_colon or (stack and stack[-1] == "["):
            j = i
            while j < len(text) and text[j] not in (",", "}", "]", " ", "\n", "\r", "\t"):
                j += 1
            safe_cuts.append(j)
            after_colon = False
            i = j
            continue
        i += 1

    if safe_cuts:
        text = text[:safe_cuts[-1]]

    text = text.rstrip().rstrip(",")

    stripped = text.rstrip()
    if stripped and stripped[-1] not in ('"', '}', ']', '{', '['):
        j = len(stripped) - 1
        while j > 0 and stripped[j - 1] not in (":", ",", "[", "{", " ", "\n"):
            j -= 1
        trailing = stripped[j:]
        valid_primitives = {"true", "false", "null"}
        try:
            float(trailing)
            is_valid = True
        except ValueError:
            is_valid = trailing in valid_primitives
        if not is_valid:
            text = stripped[:j].rstrip().rstrip(":").rstrip().rstrip(",")

    text = re.sub(r',\s*"[^"]*"\s*:?\s*$', "", text)
    text = re.sub(r'(\{)\s*"[^"]*"\s*:?\s*$', r"\1", text)
    text = text.rstrip().rstrip(",")

    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    closers = "".join("]" if opener == "[" else "}" for opener in reversed(stack))
    repaired = text + closers
    logger.info("Repaired truncated JSON: trimmed to %d chars, added %d closers", len(text), len(closers))
    return repaired


# ── Semaphores (one GPU / ≤3 cloud requests in flight) ────────────
#
# Concurrency limiters are keyed by the running event loop.  The LPMM adapter
# runs each paragraph inside its own ``asyncio.run`` loop from a thread pool,
# and an ``asyncio.Semaphore`` is loop-bound once contended; one semaphore per
# loop keeps the per-loop in-flight cap without cross-loop RuntimeErrors.

_local_semaphores: dict[int, asyncio.Semaphore] = {}
_cloud_semaphores: dict[int, asyncio.Semaphore] = {}
_semaphore_lock = threading.Lock()


def _get_local_semaphore() -> asyncio.Semaphore:
    key = id(asyncio.get_running_loop())
    with _semaphore_lock:
        sem = _local_semaphores.get(key)
        if sem is None:
            sem = _local_semaphores[key] = asyncio.Semaphore(1)
        return sem


def _get_cloud_semaphore() -> asyncio.Semaphore:
    key = id(asyncio.get_running_loop())
    with _semaphore_lock:
        sem = _cloud_semaphores.get(key)
        if sem is None:
            sem = _cloud_semaphores[key] = asyncio.Semaphore(3)
        return sem


def _make_http_client(timeout: float | httpx.Timeout) -> httpx.AsyncClient:
    """httpx client that ignores system proxy env vars (source behaviour)."""
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


# ── Ollama ───────────────────────────────────────────────────────


class LLMClient:
    """Async client for the local Ollama chat API (structured output support)."""

    is_cloud = False

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen2.5:7b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(
        self,
        system: str,
        prompt: str,
        format: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        num_ctx: int | None = None,
    ) -> tuple[str | dict, LlmUsage]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        options: dict = {"temperature": temperature, "num_predict": max_tokens}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }
        if format is not None:
            payload["format"] = format

        sem = _get_local_semaphore()
        async with sem:
            try:
                async with _make_http_client(httpx.Timeout(timeout, connect=10.0)) as client:
                    resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                    resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"Ollama request timed out after {timeout}s") from exc
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Ollama HTTP error {exc.response.status_code}: {exc.response.text[:300]}"
                ) from exc

        data = resp.json()
        content: str = data.get("message", {}).get("content", "")
        if not content:
            raise LLMError("Empty response from Ollama")

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = LlmUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        if format is not None:
            content = _strip_thinking(content).strip()
            try:
                return json.loads(content), usage
            except json.JSONDecodeError:
                return _extract_json(content), usage

        return content, usage

    async def generate_stream(
        self,
        system: str,
        prompt: str,
        timeout: int = 180,
    ) -> AsyncIterator[str]:
        """Stream tokens (no semaphore: chat must stay responsive)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "think": False,
        }
        async with _make_http_client(httpx.Timeout(timeout, connect=10.0)) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break


# ── OpenAI-compatible ────────────────────────────────────────────


class OpenAICompatibleClient:
    """Async client for OpenAI-compatible APIs (DeepSeek, Kimi, Qwen, vLLM…)."""

    is_cloud = True

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def generate(
        self,
        system: str,
        prompt: str,
        format: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        num_ctx: int | None = None,  # ignored, Ollama-only
    ) -> tuple[str | dict, LlmUsage]:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if format is not None:
            payload["response_format"] = {"type": "json_object"}

        sem = _get_cloud_semaphore()
        async with sem:
            try:
                async with _make_http_client(httpx.Timeout(timeout, connect=10.0)) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                    )
                    resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"Cloud API request timed out after {timeout}s") from exc
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Cloud API HTTP error {exc.response.status_code}: {exc.response.text[:300]}"
                ) from exc

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise LLMError("Empty choices in cloud API response")
        choice = choices[0]
        content: str = choice.get("message", {}).get("content", "")
        if not content:
            raise LLMError("Empty content in cloud API response")

        finish_reason = choice.get("finish_reason", "")
        usage_data = data.get("usage", {})
        usage = LlmUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        if format is not None:
            if finish_reason == "length":
                logger.warning(
                    "Cloud API output truncated (finish_reason=length), repairing JSON (%d chars)",
                    len(content),
                )
                content = _repair_truncated_json(content)
            content = _strip_thinking(content).strip()
            try:
                return json.loads(content), usage
            except json.JSONDecodeError:
                return _extract_json(content), usage

        return content, usage

    async def generate_stream(
        self,
        system: str,
        prompt: str,
        timeout: int = 180,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        async with _make_http_client(httpx.Timeout(timeout, connect=10.0)) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token


# ── Anthropic ───────────────────────────────────────────────────

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient:
    """Async client for the Anthropic Messages API (different protocol, same iface)."""

    is_cloud = True

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def generate(
        self,
        system: str,
        prompt: str,
        format: dict | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        num_ctx: int | None = None,  # ignored — Anthropic manages context internally
    ) -> tuple[str | dict, LlmUsage]:
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }

        sem = _get_cloud_semaphore()
        async with sem:
            try:
                async with _make_http_client(httpx.Timeout(timeout, connect=10.0)) as client:
                    resp = await client.post(
                        f"{self.base_url}/v1/messages", json=payload, headers=self._headers()
                    )
                    resp.raise_for_status()
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"Anthropic API request timed out after {timeout}s") from exc
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Anthropic API HTTP error {exc.response.status_code}: {exc.response.text[:300]}"
                ) from exc

        data = resp.json()
        content_blocks = data.get("content", [])
        if not content_blocks:
            raise LLMError("Empty content in Anthropic API response")
        content: str = content_blocks[0].get("text", "")
        if not content:
            raise LLMError("Empty text in Anthropic API response")

        stop_reason = data.get("stop_reason", "")
        usage_data = data.get("usage", {})
        usage = LlmUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        if format is not None:
            if stop_reason == "max_tokens":
                logger.warning(
                    "Anthropic output truncated (stop_reason=max_tokens), repairing JSON (%d chars)",
                    len(content),
                )
                content = _repair_truncated_json(content)
            try:
                return json.loads(content), usage
            except json.JSONDecodeError:
                return _extract_json(content), usage

        return content, usage

    async def generate_stream(
        self,
        system: str,
        prompt: str,
        timeout: int = 180,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async with _make_http_client(httpx.Timeout(timeout, connect=10.0)) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/messages", json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if not raw:
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    chunk_type = chunk.get("type", "")
                    if chunk_type == "content_block_delta":
                        delta = chunk.get("delta", {})
                        if delta.get("type") == "text_delta":
                            token = delta.get("text", "")
                            if token:
                                yield token
                    elif chunk_type in ("message_stop", "message_delta"):
                        if chunk_type == "message_stop" or chunk.get("delta", {}).get("stop_reason"):
                            break


# ── Settings / factory ──────────────────────────────────────────

LLMClientLike = LLMClient | OpenAICompatibleClient | AnthropicClient


@dataclass
class LLMSettings:
    """Explicit provider configuration (no hidden module-level config reads)."""

    provider: str = "openai"  # "openai" | "ollama"
    provider_format: str = "openai"  # "openai" | "anthropic"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = 8192
    vot_spatial_enabled: bool = False


def load_settings_from_env(environ: dict[str, str] | None = None) -> LLMSettings:
    """Build :class:`LLMSettings` from ``CBB_LLM_*`` environment variables."""
    env = os.environ if environ is None else environ
    return LLMSettings(
        provider=env.get("CBB_LLM_PROVIDER", "openai"),
        provider_format=env.get("CBB_LLM_PROVIDER_FORMAT", "openai"),
        base_url=env.get("CBB_LLM_BASE_URL", ""),
        api_key=env.get("CBB_LLM_API_KEY", ""),
        model=env.get("CBB_LLM_MODEL", ""),
        context_window=int(env.get("CBB_LLM_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW)),
        max_tokens=int(env.get("CBB_LLM_MAX_TOKENS", "8192")),
        vot_spatial_enabled=env.get("CBB_VOT_SPATIAL_ENABLED", "").lower() in ("1", "true", "yes"),
    )


def get_llm_client(settings: LLMSettings | None = None) -> LLMClientLike:
    """Build a client from explicit settings (default: environment)."""
    cfg = settings or load_settings_from_env()
    if cfg.provider == "ollama" or cfg.provider == "local":
        return LLMClient(
            base_url=cfg.base_url or os.environ.get("CBB_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=cfg.model or os.environ.get("CBB_OLLAMA_MODEL", "qwen2.5:7b"),
        )
    if not cfg.base_url:
        raise ValueError("LLM base_url is required for provider=openai")
    if not cfg.api_key:
        raise ValueError("LLM api_key is required for provider=openai")
    if cfg.provider_format == "anthropic":
        return AnthropicClient(
            base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model or "claude-sonnet-4-5"
        )
    return OpenAICompatibleClient(
        base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model or "gpt-4o"
    )
