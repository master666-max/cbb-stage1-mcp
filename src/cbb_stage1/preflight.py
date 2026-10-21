"""Pre-flight dependency audit — run *before* a pipeline start.

Why this exists
---------------
There was no such check.  Every missing dependency surfaced mid-run instead:

* the embedding backend was only exercised inside the ``derive`` stage, so a
  dead endpoint or an unloaded model killed the run ~40 minutes in;
* the box directory was created by ``cbb_start_run`` and only rejected by
  ``build_box`` at the very last step;
* ``mcporter`` had to be discovered by trial and error (``WinError 2``).

The audit is evidence-first: every check carries the probe that produced the
verdict, so a reader can recompute it instead of trusting a summary.  Verdicts
use the project's four states: READY / BLOCKED / OBSERVED / N/A.

Coverage follows ``19-外部依赖与接线清单-20261019.md`` classes A–F:

* A external services   — LLM endpoint/credentials, NLI, embedding backend
* B external artifacts  — UQ Head checkpoint, labelled data, gold sets, budget
* C runtime infra       — disk, work/out dirs, concurrency ceiling
* D config              — every ``CBB_*`` the run depends on
* E wiring gaps         — WIRED≠LIVE: report which injections are still missing
* F acceptance inputs   — the book actually being read

B-class items are *artifacts*, not services: a reachable endpoint does not make
``CBB_UQ_HEAD_CHECKPOINT`` usable, and an absent checkpoint silently downgrades
``confidence`` to "unavailable" for the whole run.  They are probed separately
for exactly that reason.

Nothing here mutates a shared resource except a temporary probe file inside the
``work`` directory it was asked to verify (created and deleted immediately).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from cbb_stage1.ids import sha256_hex_bytes

READY = "READY"
BLOCKED = "BLOCKED"
OBSERVED = "OBSERVED"
N_A = "N/A"

#: Minimum free bytes on the volume that will receive the work/box output.
MIN_FREE_BYTES = 512 * 1024 * 1024
#: Python below this cannot run the project. Kept equal to `requires-python` in
#: ``pyproject.toml`` on purpose: a preflight that accepts a version pip refuses is
#: worse than no preflight, because it reports READY for an environment that cannot be
#: installed. They disagreed (3.11 here, ">=3.12" there) until this was checked.
MIN_PYTHON = (3, 12)
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8080/v1"
#: Format string a B5 UQ Head checkpoint must declare (``confidence/uq_checkpoint``).
UQ_CHECKPOINT_FORMAT = "cbb-uq-head/1"
#: Default concurrency the pipeline will use when nothing overrides it
#: (``19-外部依赖与接线清单`` D.1: ``Config.max_workers`` = 4).
DEFAULT_MAX_WORKERS = 4
#: Documented default for ``CBB_LLM_MAX_TOKENS`` (19 清单 D.1).
DEFAULT_LLM_MAX_TOKENS = 8192


def _check(name: str, status: str, detail: str, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "evidence": dict(evidence or {}),
    }


def _probe_python() -> dict[str, Any]:
    version = ".".join(str(part) for part in sys.version_info[:3])
    evidence = {"executable": sys.executable, "version": version}
    if tuple(sys.version_info[:2]) < MIN_PYTHON:
        return _check(
            "python",
            BLOCKED,
            f"python {version} is older than the required "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            evidence,
        )
    return _check("python", READY, f"python {version}", evidence)


def _probe_book(book: str | Path | None) -> dict[str, Any]:
    if not book:
        return _check("book", N_A, "no book path was supplied to the audit")
    path = Path(book)
    evidence: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        return _check("book", BLOCKED, "book file does not exist", evidence)
    if not path.is_file():
        return _check("book", BLOCKED, "book path is not a regular file", evidence)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("book", BLOCKED, "book file is not readable", evidence)
    evidence["bytes"] = len(payload)
    # Contract hashing lives in ``ids`` (see qc/id_sources): no local formulas.
    evidence["sha256_16"] = sha256_hex_bytes(payload, 16)
    if not payload:
        return _check("book", BLOCKED, "book file is empty", evidence)
    return _check("book", READY, f"{len(payload)} bytes readable", evidence)


def _probe_work(work: str | Path | None) -> dict[str, Any]:
    if not work:
        return _check("work_dir", N_A, "no work path was supplied to the audit")
    root = Path(work)
    evidence: dict[str, Any] = {"path": str(root)}
    try:
        (root / "plan").mkdir(parents=True, exist_ok=True)
        probe = root / "plan" / ".preflight-probe"
        probe.write_text("ok", encoding="utf-8")
        echoed = probe.read_text(encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("work_dir", BLOCKED, "work directory is not writable", evidence)
    if echoed != "ok":
        return _check("work_dir", BLOCKED, "work directory write did not round-trip", evidence)
    return _check("work_dir", READY, "writable (write/read/delete round-trip)", evidence)


def _probe_out(out: str | Path | None) -> dict[str, Any]:
    if not out:
        return _check("out_dir", N_A, "no out path was supplied to the audit")
    target = Path(out)
    evidence: dict[str, Any] = {"path": str(target), "exists": target.exists()}
    if target.exists():
        # A box is immutable: ``box.write_box`` refuses to publish into an
        # existing directory, so starting a run here guarantees a failure at the
        # last step.  Fail now, while it is still cheap.
        return _check(
            "out_dir",
            BLOCKED,
            "out already exists; a box is immutable and build_box refuses to "
            "overwrite it — remove it or choose another out path",
            evidence,
        )
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        probe = parent / ".preflight-out-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        evidence["parent"] = str(parent)
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("out_dir", BLOCKED, "out parent directory is not writable", evidence)
    evidence["parent"] = str(parent)
    return _check("out_dir", READY, "absent and its parent is writable", evidence)


def _probe_mcporter() -> dict[str, Any]:
    found = shutil.which("mcporter")
    evidence: dict[str, Any] = {"which": found or ""}
    if found:
        return _check("mcporter", READY, f"found on PATH: {found}", evidence)
    npm_cmd = Path(os.environ.get("APPDATA", "")) / "npm" / "mcporter.cmd"
    evidence["npm_fallback"] = str(npm_cmd)
    if npm_cmd.is_file():
        return _check(
            "mcporter",
            OBSERVED,
            "not on PATH but the npm shim exists; call it by absolute path "
            "(a bare `mcporter` raises WinError 2)",
            evidence,
        )
    return _check(
        "mcporter",
        OBSERVED,
        "not found; only needed by the MCP *client* side, the server itself is "
        "already running",
        evidence,
    )


def _probe_embedding(base_url: str | None) -> dict[str, Any]:
    base = (base_url or os.environ.get("CBB_EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL).rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    evidence: dict[str, Any] = {"base_url": base, "probe_url": f"{root}/api/v0/models"}
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - declared dependency
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("embedding_backend", BLOCKED, "httpx is not installed", evidence)
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{root}/api/v0/models")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # noqa: BLE001 - any network failure blocks derive
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check(
            "embedding_backend",
            BLOCKED,
            "embedding backend is unreachable; the derive stage would die on "
            "RAPTOR instead",
            evidence,
        )
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    evidence["model_count"] = len(rows) if isinstance(rows, list) else 0
    if not isinstance(rows, list):
        return _check(
            "embedding_backend",
            BLOCKED,
            "backend answered without a model list (unexpected endpoint shape)",
            evidence,
        )
    loaded_embeddings = [
        str(item.get("id") or "")
        for item in rows
        if isinstance(item, Mapping)
        and str(item.get("state") or "").lower() == "loaded"
        and "embed" in str(item.get("id") or "").lower()
    ]
    evidence["loaded_embedding_models"] = loaded_embeddings
    if not loaded_embeddings:
        loaded_any = [
            str(item.get("id") or "")
            for item in rows
            if isinstance(item, Mapping) and str(item.get("state") or "").lower() == "loaded"
        ]
        evidence["loaded_other_models"] = loaded_any
        return _check(
            "embedding_backend",
            BLOCKED,
            "no *loaded* embedding model on the backend; a cold call would need a "
            "just-in-time load and the derive stage would time out",
            evidence,
        )
    return _check(
        "embedding_backend",
        READY,
        f"loaded embedding model: {loaded_embeddings[0]}",
        evidence,
    )


def _probe_disk(target: str | Path | None) -> dict[str, Any]:
    if not target:
        return _check("disk", N_A, "no target path was supplied to the audit")
    anchor = Path(target)
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    try:
        usage = shutil.disk_usage(anchor)
    except OSError as exc:
        return _check(
            "disk",
            BLOCKED,
            "cannot measure free space",
            {"anchor": str(anchor), "error": f"{type(exc).__name__}: {exc}"},
        )
    evidence = {"anchor": str(anchor), "free_bytes": usage.free, "required_bytes": MIN_FREE_BYTES}
    if usage.free < MIN_FREE_BYTES:
        return _check("disk", BLOCKED, f"only {usage.free} free bytes", evidence)
    return _check("disk", READY, f"{usage.free // (1024 * 1024)} MiB free", evidence)


# ── B class: external artifacts (19 清单 B / EXT-06..EXT-10) ────────────────


def _probe_uq_checkpoint(path: str | Path | None) -> dict[str, Any]:
    """EXT-06: the B5 UQ Head checkpoint, a *file*, not a service.

    Without it ``confidence`` stays ``unavailable`` for the entire run: the
    validate stage silently degrades to ``semantic_entropy``.  An endpoint being
    reachable says nothing about this file, which is why it is probed alone.
    """
    raw = str(path or os.environ.get("CBB_UQ_HEAD_CHECKPOINT") or "").strip()
    evidence: dict[str, Any] = {"configured": bool(raw), "format_expected": UQ_CHECKPOINT_FORMAT}
    if not raw:
        return _check(
            "uq_checkpoint",
            N_A,
            "not configured (CBB_UQ_HEAD_CHECKPOINT unset); confidence will report "
            "'unavailable' and validate stays on semantic_entropy",
            evidence,
        )
    target = Path(raw)
    evidence["path"] = str(target)
    if not target.is_file():
        return _check("uq_checkpoint", BLOCKED, "configured path does not exist", evidence)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("uq_checkpoint", BLOCKED, "checkpoint is not readable JSON", evidence)
    declared = str((payload or {}).get("format") or "")
    evidence["format"] = declared
    if declared != UQ_CHECKPOINT_FORMAT:
        return _check(
            "uq_checkpoint",
            BLOCKED,
            f"format {declared!r} != required {UQ_CHECKPOINT_FORMAT!r}",
            evidence,
        )
    evidence["bytes"] = target.stat().st_size
    return _check("uq_checkpoint", READY, f"valid {UQ_CHECKPOINT_FORMAT} head", evidence)


def _probe_labelled_data(path: str | Path | None) -> dict[str, Any]:
    """EXT-07: labelled JSONL for training the UQ head (``record`` + ``label``).

    Only checked when a path is configured; training is an offline step, so an
    unset path is N/A rather than a failure.
    """
    raw = str(path or os.environ.get("CBB_UQ_TRAIN_DATA") or "").strip()
    evidence: dict[str, Any] = {"configured": bool(raw)}
    if not raw:
        return _check(
            "uq_train_data",
            N_A,
            "not configured; the UQ head cannot be retrained in this run",
            evidence,
        )
    target = Path(raw)
    evidence["path"] = str(target)
    if not target.is_file():
        return _check("uq_train_data", BLOCKED, "configured path does not exist", evidence)
    rows = 0
    labelled = 0
    positives = 0
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("label") in (0, 1):
                labelled += 1
                positives += int(row["label"])
    except OSError as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("uq_train_data", BLOCKED, "labelled data is not readable", evidence)
    evidence.update({"rows": rows, "labelled": labelled, "positives": positives})
    if labelled == 0:
        return _check("uq_train_data", BLOCKED, "no row carries label 0/1", evidence)
    if positives in (0, labelled):
        # The trainer rejects single-class input; say so before it is spent on.
        return _check(
            "uq_train_data",
            BLOCKED,
            f"single-class data (positives={positives}/{labelled}); the trainer refuses it",
            evidence,
        )
    return _check("uq_train_data", READY, f"{labelled} labelled rows", evidence)


def _probe_budget(max_tokens: int | None, token_budget: int | None) -> dict[str, Any]:
    """EXT-10: a spend ceiling must exist before a live run, not after it.

    ``19-外部依赖与接线清单`` D.1 documents ``CBB_LLM_MAX_TOKENS`` defaulting to
    **8192**, so an unset value is not "no ceiling" — it is the documented
    default.  Only a *missing* per-run ``token_budget`` is a real gap, and that
    is reported OBSERVED (bounded per call, not per run) rather than BLOCKED.
    """
    resolved_max = max_tokens if max_tokens is not None else _env_int("CBB_LLM_MAX_TOKENS")
    if not resolved_max:
        resolved_max = DEFAULT_LLM_MAX_TOKENS
    resolved_budget = token_budget if token_budget is not None else _env_int("CBB_LLM_TOKEN_BUDGET")
    evidence = {
        "max_tokens": resolved_max,
        "max_tokens_source": "explicit" if max_tokens is not None else "env-or-documented-default",
        "token_budget": resolved_budget,
    }
    if resolved_max <= 0:
        return _check("budget", BLOCKED, "max_tokens must be positive", evidence)
    if not resolved_budget:
        return _check(
            "budget",
            OBSERVED,
            f"max_tokens={resolved_max} (documented default) but no per-run token_budget; "
            "cost is bounded per call, not per run",
            evidence,
        )
    return _check(
        "budget", READY, f"max_tokens={resolved_max}, token_budget={resolved_budget}", evidence
    )


def _probe_concurrency(max_workers: int | None) -> dict[str, Any]:
    """EXT-11/12: the worker ceiling the run will actually use."""
    resolved = max_workers if max_workers is not None else _env_int("CBB_MAX_WORKERS")
    if not resolved:
        resolved = DEFAULT_MAX_WORKERS
    evidence = {"max_workers": resolved, "default": DEFAULT_MAX_WORKERS}
    if resolved <= 0:
        return _check("concurrency", BLOCKED, "max_workers must be positive", evidence)
    if resolved > DEFAULT_MAX_WORKERS:
        return _check(
            "concurrency",
            OBSERVED,
            f"max_workers={resolved} exceeds the default {DEFAULT_MAX_WORKERS}; provider "
            "rate limits must be confirmed or the run will hit 429s",
            evidence,
        )
    return _check("concurrency", READY, f"max_workers={resolved}", evidence)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# ── E class: wiring gaps (WIRED != LIVE) ───────────────────────────────────


def _probe_wiring() -> dict[str, Any]:
    """Report which injections are still missing (19 清单 E / WIRE-01..06).

    A reachable endpoint does not mean a feature is live: several adapters exist
    but the pipeline never passes the llm/embedder/merger in.  Reporting them as
    OBSERVED keeps "has a key" from being read as "already effective".
    """
    try:
        from cbb_stage1.config import Config
    except Exception as exc:  # noqa: BLE001
        return _check("wiring", OBSERVED, "config not importable", {"error": str(exc)})

    fields = {name: getattr(Config, "model_fields", {}).get(name) is not None for name in ("nli_provider", "uq_head_checkpoint")}
    gaps = [
        "WIRE-01 confidence llm/sampler not injected into validate",
        "WIRE-04 description synthesizer not injected into dedupe",
        "WIRE-06 community has no LLM summary path",
    ]
    return _check(
        "wiring",
        OBSERVED,
        f"{len(gaps)} known injection gaps remain (WIRED != LIVE)",
        {"gaps": gaps, "config_fields": fields},
    )


# ── A class: external services (19 清单 A / EXT-01, EXT-02, EXT-13) ────────


def _probe_llm_endpoint(
    base_url: str | None, model: str | None, host_queue: bool | None = None
) -> dict[str, Any]:
    """EXT-01/EXT-13: is there a real chat endpoint, and is the model there?

    The host-queue transport (``CBB_HOST_MODEL_QUEUE=1``) needs no endpoint at
    all — the agent itself answers.  So "no base URL" is only a gap when the
    transport actually requires HTTP, which this probe reports rather than
    guesses.
    """
    # The caller may know the transport even when this process's environment does
    # not. That is exactly the MCP server's case: cbb_start_run decides the
    # transport from its own argument and writes the policy file, so it cannot
    # rely on CBB_HOST_MODEL_QUEUE being exported into the server process.
    if host_queue is None:
        host_queue = os.environ.get("CBB_HOST_MODEL_QUEUE", "").strip() in {"1", "true", "yes"}
    resolved = str(base_url or os.environ.get("CBB_LLM_BASE_URL", "")).strip()
    evidence: dict[str, Any] = {
        "host_queue": host_queue,
        "base_url_configured": bool(resolved),
        "model": model or os.environ.get("CBB_LLM_MODEL", ""),
    }
    if host_queue and not resolved:
        return _check(
            "llm_endpoint",
            READY,
            "host queue is the transport; no HTTP endpoint needed (the agent answers)",
            evidence,
        )
    if not resolved:
        return _check(
            "llm_endpoint",
            BLOCKED,
            "no CBB_LLM_BASE_URL and not using the host queue; the run would fall back "
            "to the mock engine and extract nothing",
            evidence,
        )
    evidence["base_url"] = resolved
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("llm_endpoint", BLOCKED, "httpx is not installed", evidence)
    try:
        import time

        started = time.monotonic()
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{resolved.rstrip('/')}/models")
            response.raise_for_status()
            payload = response.json()
        evidence["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    except Exception as exc:  # noqa: BLE001
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check(
            "llm_endpoint",
            BLOCKED,
            "chat endpoint is unreachable (network/proxy?) — EXT-13",
            evidence,
        )
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    ids = [
        str(item.get("id") or "")
        for item in (rows or [])
        if isinstance(item, Mapping)
    ]
    evidence["model_count"] = len(ids)
    wanted = evidence["model"]
    if wanted and wanted not in ids:
        evidence["models_sample"] = ids[:8]
        return _check(
            "llm_endpoint",
            BLOCKED,
            f"endpoint is up but model {wanted!r} is not listed",
            evidence,
        )
    return _check("llm_endpoint", READY, f"endpoint reachable, {len(ids)} model(s)", evidence)


#: Capabilities that depend on WHICH sub-agent serves the main reasoning.
#:
#: This is the defining property of the agent-driven architecture: the host
#: transport means "the agent answers", and the agent may be a different model on
#: every run. So the *capability set* is not a constant of the project — it varies
#: with whoever is driving. The point of this probe is to make that explicit at
#: start-up instead of discovering it three stages later when a mechanism silently
#: reports ``unavailable``.
#:
#: The trade is deliberate: switching models freely, at the price of not knowing
#: up-front what the model can do.
CAPABILITY_MATRIX: dict[str, dict[str, str]] = {
    # capability -> {transport kind -> status}
    "token_logprobs": {
        "host": "unknown",
        "mcp": "depends on backend",
        "http": "depends on vendor",
    },
    "hidden_states": {
        "host": "no",
        "mcp": "no",
        "http": "no (needs direct weight access)",
    },
    "second_model": {
        "host": "no (one responder)",
        "mcp": "configurable",
        "http": "configurable",
    },
    "stable_model_identity": {
        "host": "no (changes with the agent)",
        "mcp": "yes",
        "http": "yes",
    },
    "token_accounting": {
        "host": "only if the agent reports usage",
        "mcp": "yes",
        "http": "yes",
    },
}

#: Vendors whose API does not expose token logprobs at all.
_NO_LOGPROB_HINTS = ("anthropic", "claude")


def _probe_main_reasoner(
    *,
    transport: str | None = None,
    session_model_label: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """What the current main-reasoning sub-agent can and cannot do.

    Under the agent-driven architecture the main model is whoever drives the
    session, so capabilities are not fixed. That is the trade: free switching in
    exchange for an unknown capability set. This probe does not pretend to know
    the model — it records what is *determined by the transport* and marks the
    rest as depending on the driver.

    Reported as OBSERVED, never BLOCKED: an unknown capability set is the normal
    state of this architecture, not a defect. It becomes a blocker only when a
    mechanism that needs a specific signal is enabled while the signal is absent —
    which the per-capability probes (whitebox_backend, judge_backend) report.
    """
    resolved_transport = (
        str(transport or os.environ.get("CBB_MODEL_TRANSPORT", "")).strip().lower()
        or ("host" if os.environ.get("CBB_HOST_MODEL_QUEUE", "").strip() in {"1", "true", "yes"} else "")
        or "host"
    )
    label = str(
        session_model_label or os.environ.get("CBB_SESSION_MODEL_LABEL", "")
    ).strip()
    resolved_base = str(base_url or os.environ.get("CBB_LLM_BASE_URL", "")).strip()

    matrix = {
        name: kinds.get(resolved_transport, "unknown")
        for name, kinds in CAPABILITY_MATRIX.items()
    }

    evidence: dict[str, Any] = {
        "transport": resolved_transport,
        "session_model_label": label or None,
        "base_url_configured": bool(resolved_base),
        "capabilities": matrix,
    }

    # A hosted vendor that never returns logprobs is knowable from the URL.
    lowered = resolved_base.lower()
    if any(hint in lowered for hint in _NO_LOGPROB_HINTS):
        matrix["token_logprobs"] = "no (vendor does not expose logprobs)"
        evidence["logprob_verdict"] = "vendor-excluded"

    if resolved_transport == "host":
        detail = (
            "main reasoning is served by the driving sub-agent"
            + (f" ({label})" if label else " (unlabelled)")
            + "; capability set varies with the driver — B3/B4-whitebox depend on "
            "whether that model exposes logprobs, and the audit cannot probe it. "
            "Set session_model_label so the box records who produced it."
        )
        return _check("main_reasoner", OBSERVED, detail, evidence)

    detail = (
        f"transport={resolved_transport}; capability set is determined by the "
        "configured backend rather than by the driving agent"
    )
    return _check("main_reasoner", OBSERVED, detail, evidence)


def _probe_llm_credentials() -> dict[str, Any]:
    """EXT-01: credentials exist. Never prints the value, only its shape."""
    key = os.environ.get("CBB_LLM_API_KEY", "")
    evidence = {"configured": bool(key), "length": len(key)}
    if not key:
        # A local endpoint (LM Studio / Ollama) legitimately needs no key.
        return _check(
            "llm_credentials",
            OBSERVED,
            "CBB_LLM_API_KEY unset; fine for a local endpoint, fatal for a hosted one",
            evidence,
        )
    evidence["prefix"] = key[:3] + "…" if len(key) > 3 else "…"
    return _check("llm_credentials", READY, "CBB_LLM_API_KEY is set", evidence)


def _probe_nli(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """EXT-02: without an NLI model, temporal verify is permanently ``unknown``."""
    cfg = config or {}
    provider = str(cfg.get("nli_provider") or os.environ.get("CBB_NLI_PROVIDER") or "disabled")
    model = str(cfg.get("nli_model") or os.environ.get("CBB_NLI_MODEL") or "")
    evidence = {"nli_provider": provider, "nli_model": model}
    if provider == "disabled":
        return _check(
            "nli",
            OBSERVED,
            "nli_provider=disabled; temporal verify stays 'unknown' and D11 cannot judge",
            evidence,
        )
    if not model:
        return _check(
            "nli", BLOCKED, f"nli_provider={provider!r} but no nli_model configured", evidence
        )
    return _check("nli", READY, f"nli_provider={provider}, model={model}", evidence)


def _probe_judge(base_url: str | None, model: str | None) -> dict[str, Any]:
    """B4 judge: a *second* model. Reusing the main one defeats the check."""
    resolved_base = str(base_url or os.environ.get("CBB_MCP_JUDGE_BASE_URL", "")).strip()
    resolved_model = str(model or os.environ.get("CBB_MCP_JUDGE_MODEL", "")).strip()
    evidence = {"base_url": resolved_base, "model": resolved_model}
    if not resolved_base or not resolved_model:
        return _check(
            "judge_backend",
            N_A,
            "no judge model configured; B4 judge stays unavailable and the verifier "
            "remains the same model as the extractor (D4 cannot be satisfied)",
            evidence,
        )
    try:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{resolved_base.rstrip('/')}/models")
            response.raise_for_status()
            rows = response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("judge_backend", BLOCKED, "judge endpoint is unreachable", evidence)
    ids = [str(item.get("id") or "") for item in rows if isinstance(item, Mapping)]
    evidence["model_count"] = len(ids)
    if resolved_model not in ids:
        evidence["models_sample"] = ids[:6]
        return _check(
            "judge_backend", BLOCKED, f"judge model {resolved_model!r} is not listed", evidence
        )
    return _check("judge_backend", READY, f"judge model available: {resolved_model}", evidence)


def _probe_nli_backend(base_url: str | None, model: str | None) -> dict[str, Any]:
    """D11: a dedicated entailment model. The main model can stand in, but that is
    a fallback and the NLI tool reports it as such."""
    resolved_base = str(base_url or os.environ.get("CBB_MCP_NLI_BASE_URL", "")).strip()
    resolved_model = str(model or os.environ.get("CBB_MCP_NLI_MODEL", "")).strip()
    evidence = {"base_url": resolved_base, "model": resolved_model}
    if not resolved_base or not resolved_model:
        return _check(
            "nli_backend",
            N_A,
            "no dedicated NLI model; temporal verification would fall back to the "
            "main chat model and be reported as verifier='fallback-main-model'",
            evidence,
        )
    try:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{resolved_base.rstrip('/')}/models")
            response.raise_for_status()
            rows = response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return _check("nli_backend", BLOCKED, "NLI endpoint is unreachable", evidence)
    ids = [str(item.get("id") or "") for item in rows if isinstance(item, Mapping)]
    evidence["model_count"] = len(ids)
    if resolved_model not in ids:
        evidence["models_sample"] = ids[:6]
        return _check(
            "nli_backend", BLOCKED, f"NLI model {resolved_model!r} is not listed", evidence
        )
    return _check("nli_backend", READY, f"NLI model available: {resolved_model}", evidence)


def _probe_whitebox(enabled: bool | None) -> dict[str, Any]:
    """Whether the configured backend can return token-level signals (B3/B4)."""
    resolved = bool(enabled) if enabled is not None else bool(
        os.environ.get("CBB_WHITEBOX_BACKEND", "").strip() in {"1", "true", "yes"}
    )
    evidence = {"whitebox_backend": resolved}
    if not resolved:
        return _check(
            "whitebox_backend",
            OBSERVED,
            "backend cannot return logprobs/hidden_states: B3 SEPs and B4 whitebox "
            "UQ will report unavailable (the algorithm exists, the signal does not)",
            evidence,
        )
    return _check("whitebox_backend", READY, "backend declared white-box capable", evidence)


def _probe_gate_tools() -> dict[str, Any]:
    """Are the standalone gate tools present and runnable?

    ``field_coverage.py`` and ``call_wiring.py`` are what catch "written but never
    invoked" and "required field always null". They live outside the package, so
    nothing guarantees they ship with it — and a missing gate tool is worse than no
    gate, because the acceptance run silently skips it.

    Located through ``schemas.tools_dir`` rather than ``parents[2] / "tools"``: the package
    flattens the scripts to its root, so the arithmetic reported both tools missing on an
    install that carried them. Measured there: ``tools_dir = <install-dir>\\tools``, absent.
    """
    from cbb_stage1.schemas import tools_dir as _tools_dir

    resolved = _tools_dir()
    expected = ("field_coverage.py", "call_wiring.py")
    present: dict[str, bool] = {}
    for name in expected:
        present[name] = (resolved / name).is_file()

    evidence: dict[str, Any] = {
        "tools_dir": str(resolved),
        "present": present,
    }
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        evidence["missing"] = missing
        return _check(
            "gate_tools",
            BLOCKED,
            f"gate tool(s) missing: {missing} — call_wiring/field_coverage cannot "
            "run, so 'written but not wired' would go undetected",
            evidence,
        )

    # Import them to prove they are runnable, not merely present.
    import importlib.util
    import sys

    broken: list[str] = []
    for name in expected:
        path = resolved / name
        spec = importlib.util.spec_from_file_location(f"_pf_{path.stem}", path)
        if spec is None or spec.loader is None:
            broken.append(f"{name}: cannot load")
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - report, do not crash preflight
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    if broken:
        evidence["broken"] = broken
        return _check("gate_tools", BLOCKED, "gate tool(s) failed to load", evidence)
    return _check(
        "gate_tools",
        READY,
        "field_coverage + call_wiring present and importable",
        evidence,
    )


def _probe_schema_dir(schema_dir: str | Path | None = None) -> dict[str, Any]:
    """The box schemas are required by field_coverage and by box linting.

    They live in the design tree, not in the package, so a relocated or trimmed
    checkout breaks the field-coverage gate while every other check stays green.
    """
    if schema_dir:
        resolved = Path(schema_dir)
    else:
        # Resolve by chain (shipped schemas first, legacy sibling last). The old
        # version went straight to the sibling design tree, so a published checkout
        # reported schema_dir: BLOCKED on every preflight.
        from cbb_stage1.schemas import schemas_dir as _resolve_schemas

        try:
            resolved = _resolve_schemas()
        except FileNotFoundError:
            resolved = Path(__file__).resolve().parents[3] / "设计" / "抽取段-v6" / "schemas"
    evidence: dict[str, Any] = {"schema_dir": str(resolved)}
    if not resolved.is_dir():
        return _check(
            "schema_dir",
            BLOCKED,
            "box schema directory not found; field coverage and lint cannot run",
            evidence,
        )
    schemas = sorted(resolved.glob("*.schema.json"))
    evidence["count"] = len(schemas)
    if not schemas:
        return _check("schema_dir", BLOCKED, "no *.schema.json in the schema directory", evidence)
    return _check("schema_dir", READY, f"{len(schemas)} schema(s) found", evidence)


PROBES: tuple[tuple[str, Callable[..., dict[str, Any]]], ...] = (
    # A / C: services and runtime
    ("python", lambda ctx: _probe_python()),
    ("book", lambda ctx: _probe_book(ctx.get("book"))),
    ("work_dir", lambda ctx: _probe_work(ctx.get("work"))),
    ("out_dir", lambda ctx: _probe_out(ctx.get("out"))),
    ("mcporter", lambda ctx: _probe_mcporter()),
    (
        "llm_endpoint",
        lambda ctx: _probe_llm_endpoint(
            ctx.get("llm_base_url"), ctx.get("llm_model"), ctx.get("host_queue")
        ),
    ),
    ("llm_credentials", lambda ctx: _probe_llm_credentials()),
    ("nli", lambda ctx: _probe_nli(ctx.get("nli_config"))),
    ("embedding_backend", lambda ctx: _probe_embedding(ctx.get("embedding_base_url"))),
    ("disk", lambda ctx: _probe_disk(ctx.get("work") or ctx.get("out"))),
    # B: external artifacts
    ("uq_checkpoint", lambda ctx: _probe_uq_checkpoint(ctx.get("uq_head_checkpoint"))),
    ("uq_train_data", lambda ctx: _probe_labelled_data(ctx.get("uq_train_data"))),
    ("budget", lambda ctx: _probe_budget(ctx.get("max_tokens"), ctx.get("token_budget"))),
    # C: concurrency ceiling
    ("concurrency", lambda ctx: _probe_concurrency(ctx.get("max_workers"))),
    # F: the gate tools themselves (a missing gate is worse than no gate)
    ("gate_tools", lambda ctx: _probe_gate_tools()),
    ("schema_dir", lambda ctx: _probe_schema_dir(ctx.get("schema_dir"))),
    # A: who is serving the main reasoning (agent-driven => varies per run)
    ("main_reasoner", lambda ctx: _probe_main_reasoner(
        transport=ctx.get("transport"),
        session_model_label=ctx.get("session_model_label"),
        base_url=ctx.get("base_url"),
    )),
    # B/E: second-model roles and white-box capability
    ("judge_backend", lambda ctx: _probe_judge(ctx.get("judge_base_url"), ctx.get("judge_model"))),
    ("nli_backend", lambda ctx: _probe_nli_backend(ctx.get("nli_base_url"), ctx.get("nli_model"))),
    ("whitebox_backend", lambda ctx: _probe_whitebox(ctx.get("whitebox_backend"))),
    # E: wiring gaps
    ("wiring", lambda ctx: _probe_wiring()),
)


def run_preflight(
    *,
    book: str | Path | None = None,
    work: str | Path | None = None,
    out: str | Path | None = None,
    embedding_base_url: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    host_queue: bool | None = None,
    nli_config: Mapping[str, Any] | None = None,
    transport: str | None = None,
    session_model_label: str | None = None,
    base_url: str | None = None,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    nli_base_url: str | None = None,
    nli_model: str | None = None,
    whitebox_backend: bool | None = None,
    schema_dir: str | Path | None = None,
    uq_head_checkpoint: str | Path | None = None,
    uq_train_data: str | Path | None = None,
    max_tokens: int | None = None,
    token_budget: int | None = None,
    max_workers: int | None = None,
    skip: set[str] | None = None,
) -> dict[str, Any]:
    """Run every probe and return a verdict a caller can gate on.

    ``skip`` marks probes N/A (used by offline/mock paths that legitimately do
    not need a backend).  Skipped probes are reported, never silently dropped.

    Every argument here is threaded into ``context`` and read by exactly one
    probe.  A probe whose input is never passed would silently report N/A on a
    missing dependency — the same "field added, nothing wired" failure this
    audit exists to catch.
    """
    context = {
        "book": book,
        "work": work,
        "out": out,
        "embedding_base_url": embedding_base_url,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "host_queue": host_queue,
        "nli_config": nli_config,
        "transport": transport,
        "session_model_label": session_model_label,
        "base_url": base_url,
        "judge_base_url": judge_base_url,
        "judge_model": judge_model,
        "nli_base_url": nli_base_url,
        "nli_model": nli_model,
        "whitebox_backend": whitebox_backend,
        "schema_dir": schema_dir,
        "uq_head_checkpoint": uq_head_checkpoint,
        "uq_train_data": uq_train_data,
        "max_tokens": max_tokens,
        "token_budget": token_budget,
        "max_workers": max_workers,
    }
    checks: list[dict[str, Any]] = []
    for name, probe in PROBES:
        if skip and name in skip:
            checks.append(_check(name, N_A, "skipped by caller", {"skipped": True}))
            continue
        try:
            checks.append(probe(context))
        except Exception as exc:  # noqa: BLE001 - a probe must never crash the audit
            checks.append(
                _check(
                    name,
                    BLOCKED,
                    "probe raised",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
            )
    blocked = [item["name"] for item in checks if item["status"] == BLOCKED]
    observed = [item["name"] for item in checks if item["status"] == OBSERVED]
    return {
        "status": BLOCKED if blocked else READY,
        "blocked": blocked,
        "observed": observed,
        "checks": checks,
    }


def format_report(report: Mapping[str, Any]) -> str:
    """Human/agent readable rendering; keeps the probe evidence visible."""
    lines = [f"preflight: {report.get('status')}"]
    for item in report.get("checks", []):
        lines.append(f"  [{item['status']:8s}] {item['name']}: {item['detail']}")
        evidence = item.get("evidence") or {}
        if evidence:
            lines.append(f"             {json.dumps(evidence, ensure_ascii=False)}")
    if report.get("blocked"):
        lines.append("blocked: " + ", ".join(report["blocked"]))
    return "\n".join(lines)
