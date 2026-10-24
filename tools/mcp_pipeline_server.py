"""MCP server: the agent-driven front door of the extraction pipeline.

The project is agent-driven. This server is the **primary driving interface**:
an agent starts a run, services the model requests the run raises, and builds
the box — all through MCP tool calls, not by shelling out and watching a
terminal.

Why the run is a subprocess: the pipeline blocks on model calls. If the model
requests were serviced by the same process that runs the pipeline, the agent
would deadlock (it is waiting for the tool result). So:

* ``cbb_start_run`` spawns ``cbb_stage1.cli run`` with ``CBB_HOST_MODEL_QUEUE=1``
  and returns immediately;
* ``cbb_next_requests`` lists the model requests that run is waiting on;
* ``cbb_reply`` answers one of them;
* ``cbb_status`` reports progress and the final status.

The model answers therefore come from the agent's own session subagents. No
local endpoint is used for main reasoning unless the run explicitly declares it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "cbb-pipeline-mcp", "version": "0.1.0"}

#: Directory holding the standalone gate tools (field_coverage, call_wiring).
_TOOLS_DIR = Path(__file__).resolve().parent

def _find_project_root(server_file: str | Path) -> Path:
    """Locate the tree this file belongs to, in either supported layout.

    The file sits at ``<project>/tools/<name>.py`` in the checkout and at
    ``<package>/server.py`` in the built package. ``_TOOLS_DIR.parent`` is right for the
    first and yields the package's **parent** for the second — measured on the real
    install, it resolved ``tools_dir`` to ``<install-dir>\\tools`` and ``root`` to
    ``<install-dir>\\src\\cbb_stage1``, both absent, so ``gate_tools``, ``schema_dir`` and
    ``call_wiring`` all failed on a package that contained every file they looked for.

    Marker walk rather than arithmetic: the nearest ancestor holding ``cbb_stage1/``
    (packaged) or ``src/cbb_stage1/`` (checkout). With no marker anywhere it returns the
    start directory — escaping to a parent that has nothing is how it broke.
    """
    start = Path(server_file).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "cbb_stage1").is_dir() or (
            candidate / "src" / "cbb_stage1"
        ).is_dir():
            return candidate
    return start


def _module_source_dir(root: Path) -> Path:
    """Where ``cbb_stage1`` is importable from: ``root/src`` in the checkout, ``root`` packaged."""
    nested = root / "src" / "cbb_stage1"
    return nested if nested.is_dir() else root / "cbb_stage1"


#: The repository root, resolved for whichever layout this file is running from.
#: Delegates to the library so the server, the preflight probe and the schema resolver
#: cannot disagree about where the tree is — two copies of this rule drift, and only one
#: of them gets fixed.
def _library_root() -> Path:
    import sys as _sys

    for extra in (_TOOLS_DIR / ".." / "src", _TOOLS_DIR):
        resolved = extra.resolve()
        if (resolved / "cbb_stage1").is_dir() and str(resolved) not in _sys.path:
            _sys.path.insert(0, str(resolved))
    from cbb_stage1.schemas import project_root as _project_root

    return _project_root(__file__)


PROJECT_ROOT = _library_root()

#: The importable package directory, which differs between the two layouts.
MODULE_SOURCE_DIR = _module_source_dir(PROJECT_ROOT)

_TOOL_MODULE_CACHE: dict[str, Any] = {}


def _load_tool_module(name: str) -> Any:
    """Import a ``tools/<name>.py`` module once, in-process.

    The two gate scripts predate the package layout and are also run by hand and by
    ``qc/acceptance.py``; moving them is a separate change with its own blast radius.
    Loading them here keeps one copy of the logic while letting the MCP server run the
    check itself rather than shelling out.
    """
    cached = _TOOL_MODULE_CACHE.get(name)
    if cached is not None:
        return cached
    import importlib.util

    path = _TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cbb_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules[f"cbb_tool_{name}"] = module
    spec.loader.exec_module(module)
    _TOOL_MODULE_CACHE[name] = module
    return module

REQUESTS = "requests"
RESPONSES = "responses"

#: Run state must survive across MCP calls: mcporter (and most MCP clients)
#: start a fresh server process per call, so an in-memory dict would be empty
#: by the time ``cbb_status`` runs. State lives under ``<work>/plan/mcp-runs``.
RUNS_DIR = "mcp-runs"


def _runs_dir(work: str | Path) -> Path:
    return Path(work) / "plan" / RUNS_DIR


def _run_path(work: str | Path, run_id: str) -> Path:
    return _runs_dir(work) / f"{run_id}.json"


def _save_run(work: Path, run_id: str, record: dict[str, Any]) -> None:
    _write_json(_run_path(work, run_id), record)


def _load_run(work: str | Path, run_id: str) -> dict[str, Any] | None:
    path = _run_path(work, run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int | None) -> bool:
    """Check liveness WITHOUT signalling the process.

    ``os.kill(pid, 0)`` looks like the obvious probe, but on Windows it is not a
    no-op: any signal other than CTRL_C/CTRL_BREAK is forwarded to
    ``TerminateProcess``. Calling it here killed the very pipeline we were
    checking on. Use a read-only handle instead.
    """
    if not pid:
        return False
    pid = int(pid)
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError, PermissionError):
            return False
        return True
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)

TOOLS = [
    {
        "name": "cbb_start_run",
        "description": "Start an extraction run. Returns immediately with a run_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "book": {"type": "string", "description": "EPUB/TXT path"},
                "book_id": {"type": "string"},
                "work": {"type": "string", "description": "work directory"},
                "out": {"type": "string", "description": "box output directory"},
                "chapters": {"type": "string", "default": "1-3"},
                "stages": {
                    "type": "string",
                    "description": (
                        "comma-separated postprocess stages to run instead of all eight: "
                        "resolve,assemble,claims,cues,temporal,validate,dedupe,derive. Useful for "
                        "re-running one stage after a failure; unknown names are rejected rather "
                        "than skipped."
                    ),
                },
                "live": {"type": "boolean", "default": True},
                "provider": {
                    "type": "string",
                    "default": "ai_reader",
                    "description": "engine provider (ai_reader/openie/lpmm); 'mock' would not extract",
                },
                "model": {"type": "string", "default": "session"},
                "transport": {
                    "type": "string",
                    "enum": ["host", "http"],
                    "default": "host",
                    "description": (
                        "who serves the main reasoning: 'host' (default) = the agent's own "
                        "subagents answer through work/plan/host-model, the native form for "
                        "this package; 'http' = a real endpoint answers, for runs that "
                        "should finish without the agent"
                    ),
                },
                "llm_base_url": {
                    "type": "string",
                    "description": (
                        "required when transport='http': OpenAI-compatible base URL, e.g. "
                        "http://127.0.0.1:8080/v1"
                    ),
                },
                "llm_model": {
                    "type": "string",
                    "description": "model id for the http transport; recorded as the session label",
                },
                "max_concurrency": {
                    "type": "integer",
                    # Must equal CONCURRENCY_DEFAULT (8): the handler uses that constant when the
                    # argument is absent, so advertising 2 made an agent plan for half the
                    # concurrency it actually got. Pinned by `test_mcp_schema_defaults.py`.
                    "default": 8,
                    "description": (
                        "How many model calls the gateway may admit at once (the "
                        "BoundedSemaphore size). Defaults to CONCURRENCY_DEFAULT and must not be "
                        "below Config.max_workers, or stage threads queue on the semaphore."
                    ),
                },
                "uq_head_checkpoint": {
                    "type": "string",
                    "description": "EXT-06 B5 UQ head artifact; without it confidence reports 'unavailable'",
                },
                "uq_train_data": {"type": "string", "description": "EXT-07 labelled JSONL"},
                "session_model_label": {
                    "type": "string",
                    "description": "who serves the main reasoning (recorded in model-calls.jsonl)",
                },
                "judge_base_url": {"type": "string", "description": "B4 judge endpoint"},
                "judge_model": {"type": "string", "description": "B4 judge model id"},
                "nli_base_url": {"type": "string", "description": "D11 dedicated NLI endpoint"},
                "nli_model": {"type": "string", "description": "D11 dedicated NLI model id"},
                "whitebox_backend": {
                    "type": "boolean",
                    "description": "backend can return token logprobs (unlocks B3/B4-whitebox)",
                },
                "max_tokens": {"type": "integer", "description": "EXT-10 per-call ceiling"},
                "token_budget": {"type": "integer", "description": "EXT-10 per-run ceiling"},
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "start even when preflight is BLOCKED (recorded, not hidden)",
                },
                "preflight_skip": {
                    "type": "string",
                    "description": "comma-separated probes to mark N/A for this run",
                },
            },
            "required": ["book", "book_id", "work", "out"],
        },
    },
    {
        "name": "cbb_pause_run",
        "description": (
            "Pause a run at its next unit safe point (resumable). Distinct from cbb_stop_run: a "
            "paused run keeps its work and continues where it stopped."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string", "description": "work directory the run used"},
            },
            "required": ["work"],
        },
    },
    {
        "name": "cbb_resume_run",
        "description": "Release a pause set by cbb_pause_run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string", "description": "work directory the run used"},
            },
            "required": ["work"],
        },
    },
    {
        "name": "cbb_stop_run",
        "description": (
            "Stop a run started by cbb_start_run: terminates the process tree and records who "
            "stopped it. A run that already finished is reported as a no-op."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string", "description": "work directory the run used"},
                "run_id": {"type": "string"},
                "by": {"type": "string", "description": "who stopped it (recorded in the run record)"},
            },
            "required": ["work", "run_id"],
        },
    },
    {
        "name": "cbb_status",
        "description": "Progress/status of a run, plus how many model requests wait.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string", "description": "work directory (run state lives under <work>/plan/mcp-runs)"},
                "run_id": {"type": "string"},
            },
            "required": ["work", "run_id"],
        },
    },
    {
        "name": "cbb_next_requests",
        "description": (
            "Take up to `limit` pending model requests. A returned request is CLAIMED for this "
            "consumer so another consumer will not answer it too; the claim expires after "
            "CLAIM_TTL_SECONDS. Answer with cbb_reply."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string", "description": "work directory (run state lives under <work>/plan/mcp-runs)"},
                "run_id": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["work", "run_id"],
        },
    },
    {
        "name": "cbb_reply",
        "description": (
            "Answer one pending model request with the agent's own answer. "
            "Use text_file for long answers (avoids shell quoting limits). "
            "The first answer to a request is kept; a later answer is reported as "
            "{duplicate: true, kept, rejected} instead of overwriting it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string", "description": "work directory (run state lives under <work>/plan/mcp-runs)"},
                "run_id": {"type": "string"},
                "request_id": {"type": "string"},
                "text": {"type": "string"},
                "text_file": {"type": "string", "description": "path to a UTF-8 file"},
                "usage": {
                    "type": "object",
                    "description": (
                        "token usage reported by the agent, e.g. "
                        "{prompt_tokens, completion_tokens}. Optional; when absent "
                        "cost stays unmeasured rather than guessed."
                    ),
                    "properties": {
                        "prompt_tokens": {"type": "integer"},
                        "completion_tokens": {"type": "integer"},
                        "total_tokens": {"type": "integer"},
                        "estimated": {"type": "boolean"},
                    },
                },
                "error": {"type": "string"},
            },
            "required": ["work", "run_id", "request_id"],
        },
    },
    {
        "name": "cbb_model_call",
        "description": (
            "One model call through the gateway, decided by the agent. Lets the "
            "driver probe a model or test a prompt without running the pipeline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string"},
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "prompt": {"type": "string"},
                "role": {"type": "string", "default": "assistant"},
                "system": {"type": "string"},
                "out": {"type": "string", "description": "write the answer here"},
                "book": {"type": "string"},
            },
            "required": ["work", "provider", "model", "prompt"],
        },
    },
    {
        "name": "cbb_model_stats",
        "description": "Summarise work/plan/model-calls.jsonl (roles, models, failures).",
        "inputSchema": {
            "type": "object",
            "properties": {"work": {"type": "string"}},
            "required": ["work"],
        },
    },
    {
        "name": "cbb_model_drain",
        "description": (
            "List pending host-model requests from the queue alone (no run id needed)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string"},
                "queue": {"type": "string"},
            },
        },
    },
    {
        "name": "cbb_model_policy",
        "description": "Write work/plan/model-policy.json without starting a run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string"},
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "transport": {"type": "string", "enum": ["host", "mcp", "http", "auto"]},
                "host_queue": {"type": "string"},
                "mcp_command": {"type": "string"},
                "mcp_tool": {"type": "string"},
                "embedding_base_url": {"type": "string"},
                "embedding_model": {"type": "string"},
                "role_models": {"type": "string", "description": "JSON object role -> model"},
                "session_model_label": {"type": "string"},
                "judge_base_url": {"type": "string"},
                "judge_model": {"type": "string"},
                "nli_base_url": {"type": "string"},
                "nli_model": {"type": "string"},
                "whitebox_backend": {"type": "boolean"},
                "max_concurrency": {"type": "integer"},
                "max_tokens": {"type": "integer"},
                "timeout": {"type": "integer"},
                "token_budget": {"type": "integer"},
            },
            "required": ["work"],
        },
    },
    {
        "name": "cbb_smoke",
        "description": "Offline mock smoke run: proves the pipeline works without any model.",
        "inputSchema": {
            "type": "object",
            "properties": {"work": {"type": "string"}},
        },
    },
    {
        "name": "cbb_capabilities",
        "description": (
            "What this channel can do: compares the CLI command set against the MCP "
            "tool set and reports anything the agent cannot reach."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "cbb_lexicon_list",
        "description": "D5: list unregistered-predicate proposals awaiting a decision.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string"},
                "json": {"type": "boolean", "default": True},
            },
            "required": ["work"],
        },
    },
    {
        "name": "cbb_lexicon_approve",
        "description": (
            "D5: approve one predicate proposal and register it. Requires `by`: I4 "
            "forbids predicates grown from data without a human gate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string"},
                "predicate": {"type": "string"},
                "by": {"type": "string", "description": "who approves it (mandatory)"},
                "library": {"type": "string"},
                "lexicon": {"type": "string"},
                "json": {"type": "boolean", "default": True},
            },
            "required": ["work", "predicate", "by"],
        },
    },
    {
        "name": "cbb_annotator_agreement",
        "description": (
            "B11: have several labellers judge the same claims independently and report "
            "Cohen's kappa. A soft gate: a small sample makes kappa noisy, so it is "
            "reported, never enforced."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["work"],
            "properties": {
                "work": {"type": "string"},
                "sample": {"type": "integer"},
                "annotator": {"type": "array", "items": {"type": "string"}},
                "json": {"type": "boolean"},
            },
        },
    },
    {
        "name": "cbb_conflict_resolve",
        "description": (
            "Record a human decision about one conflict. All five declared choices are "
            "accepted; the conflict row is annotated, never deleted. Requires `by`."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["work", "conflict_id", "choice", "by", "reason"],
            "properties": {
                "work": {"type": "string"},
                "conflict_id": {"type": "string"},
                "choice": {
                    "type": "string",
                    "enum": ["left", "right", "merge", "disputed", "superseded"],
                },
                "by": {"type": "string"},
                "reason": {"type": "string"},
                "at": {"type": "string"},
                "json": {"type": "boolean"},
            },
        },
    },
    {
        "name": "cbb_quarantine_adjudicate",
        "description": (
            "Record a human decision about one quarantined item. The item is "
            "annotated, never deleted. Requires `by`."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["work", "item_id", "decision", "by", "reason"],
            "properties": {
                "work": {"type": "string"},
                "item_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["keep", "drop", "retain"]},
                "by": {"type": "string"},
                "reason": {"type": "string"},
                "at": {"type": "string"},
                "json": {"type": "boolean"},
            },
        },
    },
    {
        "name": "cbb_supersede",
        "description": (
            "D9: mark one assertion superseded by another (retcon). A human decision: "
            "the old record is kept, the decision is written to audit/decisions.jsonl."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string"},
                "loser": {"type": "string"},
                "winner": {"type": "string"},
                "by": {"type": "string"},
                "reason": {"type": "string"},
                "at": {"type": "string"},
                "json": {"type": "boolean", "default": True},
            },
            "required": ["work", "loser", "winner", "by", "reason"],
        },
    },
    {
        "name": "cbb_field_coverage",
        "description": (
            "Check every schema-required box field for emptiness. Catches the failure "
            "where a required field is null in every row and all other gates stay green."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "box": {"type": "string"},
                "schema_dir": {"type": "string"},
                "json": {"type": "boolean"},
            },
            "required": ["box"],
        },
    },
    {
        "name": "cbb_call_wiring",
        "description": (
            "Public functions with no call site anywhere. cbb_gate(wiring) proves a "
            "module is imported; this proves its functions are invoked."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "json": {"type": "boolean"},
            },
        },
    },
    {
        "name": "cbb_build_box",
        "description": "Build the v2.2 box from a finished run's work directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "work": {"type": "string"},
                "out": {"type": "string"},
                "series_id": {"type": "string"},
                "volume_no": {"type": "integer"},
            },
            "required": ["work", "out"],
        },
    },
    {
        "name": "cbb_lint_box",
        "description": "Lint a box directory.",
        "inputSchema": {
            "type": "object",
            "properties": {"box": {"type": "string"}},
            "required": ["box"],
        },
    },
    {
        "name": "cbb_gate",
        "description": "Run a project gate: wiring | id_sources | acceptance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gate": {"type": "string", "enum": ["wiring", "id_sources", "acceptance"]},
                "matrix": {"type": "string"},
            },
            "required": ["gate"],
        },
    },
    {
        "name": "cbb_doctor",
        "description": (
            "Pre-flight dependency audit covering the classes in "
            "19-外部依赖与接线清单: A services (llm endpoint/credentials, nli, "
            "embedding), B artifacts (uq checkpoint, labelled data, budget), "
            "C runtime (python, book, work/out dirs, disk, concurrency), "
            "E wiring gaps (WIRED != LIVE), plus main_reasoner: which capabilities "
            "the current driver actually has (under agent-driven transport the main "
            "model changes per run, so this is reported rather than assumed). "
            "Returns per-probe verdicts with the evidence that produced them. "
            "cbb_start_run runs this too."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "book": {"type": "string"},
                "work": {"type": "string"},
                "out": {"type": "string"},
                "embedding_base_url": {"type": "string"},
                "llm_base_url": {"type": "string", "description": "EXT-01 chat endpoint"},
                "llm_model": {"type": "string", "description": "EXT-01 model id"},
                "transport": {
                    "type": "string",
                    "enum": ["host", "mcp", "http"],
                    "description": (
                        "who serves the main reasoning; host means the driving "
                        "sub-agent, so its capability set varies per run"
                    ),
                },
                "session_model_label": {
                    "type": "string",
                    "description": "name of the model driving the session (recorded in the box)",
                },
                "judge_base_url": {"type": "string", "description": "B4 judge endpoint"},
                "judge_model": {"type": "string", "description": "B4 judge model id"},
                "nli_base_url": {"type": "string", "description": "D11 dedicated NLI endpoint"},
                "whitebox_backend": {
                    "type": "boolean",
                    "description": "backend can return token logprobs (B3/B4-whitebox)",
                },
                "schema_dir": {"type": "string", "description": "box schema directory for the field-coverage gate"},
                "nli_provider": {"type": "string", "description": "EXT-02"},
                "nli_model": {"type": "string", "description": "EXT-02"},
                "uq_head_checkpoint": {"type": "string", "description": "EXT-06"},
                "uq_train_data": {"type": "string", "description": "EXT-07"},
                "max_tokens": {"type": "integer", "description": "EXT-10 per-call ceiling"},
                "token_budget": {"type": "integer", "description": "EXT-10 per-run ceiling"},
                "max_workers": {"type": "integer", "description": "EXT-11/12 concurrency"},
                "skip": {
                    "type": "string",
                    "description": "comma-separated probes to mark N/A",
                },
            },
        },
    },
]
# NOTE: there is deliberately no CLI-invoking helper here.
#
# The server used to have ``_cli(argv)`` that spawned ``python -m cbb_stage1.cli``
# and returned ``{returncode, stdout, stderr}``. Every tool that used it owned no
# capability: it delegated its jurisdiction to a development tool, handed the agent
# CLI text to parse, and paid a process spawn per call. Tools now call the library
# directly (``cbb_stage1.model_ops``, ``validate.adjudication``, ``box.writer`` …),
# and the CLI is a separate front-end over the same functions.

# ── helpers ────────────────────────────────────────────────────────────────


#: Queue location relative to the work root, for runs started before queues were per-run.
LEGACY_QUEUE_RELATIVE = "plan/host-model"

#: Per-queue directory holding one file per request a consumer has taken.
CLAIMS = "claims"

#: How long a claim is honoured before another consumer may take the request.
#:
#: A claim must expire or a consumer that dies mid-answer strands its request forever — the run
#: would then block until the host-queue timeout, which is the failure mode claims exist to
#: avoid on the other side. Five minutes is longer than any single answer takes and short enough
#: that a dead consumer is noticed inside one run.
CLAIM_TTL_SECONDS = 300.0


def _claim_path(queue: Path, request_id: str) -> Path:
    return queue / CLAIMS / f"{request_id}.json"


def _claim_is_live(path: Path) -> bool:
    """True when an unexpired claim already exists."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        claimed_at = float(row.get("claimed_at") or 0.0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # An unreadable claim file is treated as expired: a corrupt marker must not block a
        # request for the rest of the run.
        return False
    return (time.time() - claimed_at) < CLAIM_TTL_SECONDS


def _try_claim(queue: Path, request_id: str) -> bool:
    """Take `request_id` for this consumer, atomically. False when someone else has it."""
    path = _claim_path(queue, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        {"request_id": request_id, "claimed_at": time.time(), "pid": os.getpid()}
    )

    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if _claim_is_live(path):
            return False
        # Expired. Remove and race to recreate: two consumers both reaching this point is fine,
        # because only one O_EXCL create can win. Overwriting instead would let both believe
        # they own it.
        try:
            path.unlink()
        except OSError:
            return False
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False

    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    return True



def _queue_dir(work: str | Path, relative: str | None = None) -> Path:
    """Where a run's host-model queue lives.

    ``relative`` comes from the run record and is a path under the work root. When it is
    absent (a run started before this field existed) the legacy shared directory is used, so
    old work directories keep working. Deliberately not gated on the directory existing: the
    record is written before the pipeline starts, so a fresh run's queue has not been created
    yet.
    """
    base = Path(work)
    if relative:
        return base / relative
    return base / LEGACY_QUEUE_RELATIVE


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


#: Error kinds an agent can branch on. Every failure is classified here rather
#: than being an opaque string, because the driver needs to decide what to do:
#: a ``usage`` error means fix the arguments, a ``backend`` error means retry or
#: change the endpoint, ``timeout`` means wait, ``internal`` means report it.
ERROR_KINDS = ("usage", "backend", "timeout", "blocked", "internal")


def _classify_error(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn a handler's failure into a shape the agent can act on."""
    raw = payload.get("error")
    message = str(raw) if not isinstance(raw, dict) else str(raw.get("message") or raw)

    kind = "internal"
    if payload.get("blocked"):
        kind = "blocked"
    elif any(token in message.lower() for token in ("required", "is required", "unknown tool", "unknown gate")):
        kind = "usage"
    elif any(token in message.lower() for token in ("timeout", "timed out")):
        kind = "timeout"
    elif any(
        token in message.lower()
        for token in ("refused", "unreachable", "connection", "http", "endpoint", "backend")
    ):
        kind = "backend"
    if isinstance(raw, dict) and raw.get("kind") in ERROR_KINDS:
        kind = str(raw["kind"])

    return {"kind": kind, "message": message}


def _tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Uniform envelope over every tool response.

    Handlers returned a dozen different shapes (``{"error": ...}``,
    ``{"returncode": ...}``, a bare dict). An agent cannot write recovery logic
    against a dozen shapes, so every response is wrapped as::

        {"ok": bool, "error": {"kind", "message"} | None, "data": <original>}

    The original payload is preserved under ``data``, so existing callers that read
    ``returncode``/``stdout`` keep working — this adds a contract, it does not
    replace one.
    """
    failed = bool(payload.get("error"))
    envelope = {
        "ok": not failed,
        "error": _classify_error(payload) if failed else None,
        "data": payload,
    }
    return {
        "content": [{"type": "text", "text": json.dumps(envelope, ensure_ascii=False)}],
        "structuredContent": envelope,
        "isError": failed,
    }


# ── tools ──────────────────────────────────────────────────────────────────


#: How the main reasoning is served.
#:
#: `host` (default) — the agent's **subagents** answer through `work/plan/host-model`.
#: This is the package's native form: it hangs off an agent and needs no extra service.
#: `http` — a real endpoint answers, for runs that should finish without the agent.
#:
#: Do not read the measurement below as a property of `host`. It is what happens when the
#: queue is drained *synchronously by the top-level session*: on a 5000-character chapter,
#: 294,543 characters of prompt for 28,710 characters of answers (median 7), the same
#: chapter echoed 28 times, two 900-second timeouts, five hours unfinished. A subagent
#: pool draining the same queue concurrently does not pay that — the queue is built for
#: it (`max_concurrency`).
TRANSPORTS: tuple[str, ...] = ("host", "http")

#: Default number of concurrent model calls.
#:
#: This was 2, which is below the pipeline's own `Config.max_workers` default of 4 — so a
#: parallel stage could open 4 threads and have half of them blocked on the gateway's
#: `BoundedSemaphore`. The reference implementation in the library defaults to 25
#: (`graphrag .../defaults.py:189`); 8 is a deliberate middle: enough that four stages can
#: overlap, low enough that a host agent is not asked for 25 queue replies at once.
CONCURRENCY_DEFAULT = 8


def _transport_settings(args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return `(policy fields, child env)` for the requested transport.

    Refuses rather than degrades: an `http` run with no endpoint would fall back to the
    mock engine and produce a box that looks finished and contains nothing.
    """
    transport = str(args.get("transport") or "host").strip().lower()
    if transport not in TRANSPORTS:
        raise ValueError(
            f"unknown transport {transport!r}; expected one of {list(TRANSPORTS)}"
        )

    if transport == "host":
        return (
            {"transport": "host", "host_queue": "1"},
            {"CBB_HOST_MODEL_QUEUE": "1"},
        )

    base_url = str(args.get("llm_base_url") or "").strip()
    if not base_url:
        raise ValueError(
            "transport='http' needs llm_base_url; without it the run would fall back to "
            "the mock engine and extract nothing"
        )
    model = str(args.get("llm_model") or "").strip()
    label = str(args.get("session_model_label") or "").strip() or model
    env = {"CBB_LLM_BASE_URL": base_url}
    if model:
        env["CBB_LLM_MODEL"] = model
    return (
        {"transport": "http", "host_queue": "", "session_model_label": label},
        env,
    )


def cbb_start_run(args: dict[str, Any]) -> dict[str, Any]:
    work = Path(str(args["work"]))
    out = Path(str(args["out"]))
    work.mkdir(parents=True, exist_ok=True)
    # A box is immutable: ``box.write_box`` refuses to write into an existing
    # ``out``.  The pipeline itself never writes there (``cfg.out_dir`` is only
    # read, for an incremental baseline manifest), so creating the directory here
    # made the MCP flow unable to publish at all -- start_run succeeded and
    # build_box then died with FileExistsError.  Only the parent is created.
    out.parent.mkdir(parents=True, exist_ok=True)

    # Probe the environment *before* spending anything on the run.  Nothing used
    # to check dependencies: a dead embedding endpoint or an already-existing box
    # directory only surfaced at the stage that needed it (tens of minutes in),
    # or at the very last step.  A BLOCKED probe refuses to start unless the
    # caller explicitly forces it, and a forced start is recorded, not hidden.
    from cbb_stage1.preflight import run_preflight

    preflight_skip: set[str] = set()
    if not args.get("live", True):
        # Offline/mock runs legitimately need no model backend.
        preflight_skip = {"embedding_backend"}
    if args.get("preflight_skip"):
        preflight_skip |= {
            str(item).strip()
            for item in str(args["preflight_skip"]).split(",")
            if str(item).strip()
        }
    preflight = run_preflight(
        book=str(args["book"]),
        work=work,
        out=out,
        embedding_base_url=str(args.get("embedding_base_url") or "") or None,
        # B-class artifacts and A-class service wiring: a run missing these does
        # not fail loudly, it silently degrades (confidence unavailable, verify
        # unknown), so they are audited before any time is spent.
        uq_head_checkpoint=str(args.get("uq_head_checkpoint") or "") or None,
        uq_train_data=str(args.get("uq_train_data") or "") or None,
        max_tokens=args.get("max_tokens"),
        token_budget=args.get("token_budget"),
        max_workers=args.get("max_concurrency"),
        # This server always drives the run over the host queue (it writes that
        # policy below), so the transport is known here even though the server
        # process does not export CBB_HOST_MODEL_QUEUE.
        host_queue=True,
        skip=preflight_skip,
    )
    if preflight["status"] == "BLOCKED" and not args.get("force"):
        return {
            "error": "preflight blocked; run not started",
            "blocked": preflight["blocked"],
            "preflight": preflight,
            "hint": "fix the blocked probes, or pass force=true to start anyway",
        }
    forced = preflight["status"] == "BLOCKED"

    # Who serves the main reasoning is the caller's choice: the agent's subagents
    # (host, the default and the native form for a package that hangs off an
    # agent) or a real endpoint (http). See TRANSPORTS.
    transport_policy, transport_env = _transport_settings(args)
    policy_path = work / "plan" / "model-policy.json"
    _write_json(
        policy_path,
        {
            **transport_policy,
            "max_concurrency": int(args.get("max_concurrency") or CONCURRENCY_DEFAULT),
            "max_tokens": int(args.get("max_tokens") or 4096),
            "timeout": int(args.get("timeout") or 180),
            "embedding_base_url": str(args.get("embedding_base_url") or "http://127.0.0.1:8080/v1"),
            # Who is serving the main reasoning. The host protocol does not expose
            # model identity, so the caller declares it; without it every call in
            # model-calls.jsonl reads "session" and the box cannot say which model
            # produced it. The transport helper already filled it for an http run.
            "session_model_label": str(
                transport_policy.get("session_model_label")
                or args.get("session_model_label")
                or ""
            ),
            "judge_base_url": str(args.get("judge_base_url") or ""),
            "judge_model": str(args.get("judge_model") or ""),
            "nli_base_url": str(args.get("nli_base_url") or ""),
            "nli_model": str(args.get("nli_model") or ""),
            "whitebox_backend": bool(args.get("whitebox_backend") or False),
        },
    )

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    # The http path must NOT inherit a host-queue marker from this server's own
    # environment, or the run would go back to waiting on an agent that is not there.
    if "CBB_HOST_MODEL_QUEUE" not in transport_env:
        env.pop("CBB_HOST_MODEL_QUEUE", None)
    env.update(transport_env)
    argv = [
        sys.executable,
        # ``-u`` matters: the log is a redirected file, so Python block-buffers
        # stdout and wrote nothing for the whole 50-minute run. ``cbb_status``
        # then had an empty ``log_tail`` exactly when progress mattered.
        "-u",
        "-m",
        "cbb_stage1.cli",
        "run",
        "--book",
        str(args["book"]),
        "--book-id",
        str(args["book_id"]),
        "--work",
        str(work),
        "--out",
        str(out),
        "--chapters",
        str(args.get("chapters") or "1-3"),
    ]
    if args.get("live", True):
        argv.append("--live")
    # Without an explicit provider the run would silently use the mock engine
    # (Config.provider defaults to "mock") and "succeed" with no extraction.
    # "ai_reader" means: use the AI Reader engine, whose llm client is the
    # gateway (host transport) rather than a direct HTTP endpoint.
    argv += [
        "--provider",
        str(args.get("provider") or "ai_reader"),
        "--model",
        str(args.get("model") or "session"),
    ]
    # Stage selection: `--stages` is only added when asked for, so a plain run keeps the default
    # eight-stage chain.
    if args.get("stages"):
        argv.extend(["--stages", str(args["stages"])])

    run_id = uuid.uuid4().hex[:12]
    log_path = work / "plan" / f"mcp-run-{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # One queue per run. Without this, a second run against the same work directory reads the
    # previous run's unanswered requests as its own (measured: 16 pending, 14 answered). Only
    # the host transport has a queue at all.
    queue_relative = ""
    if env.get("CBB_HOST_MODEL_QUEUE"):
        queue_relative = f"{LEGACY_QUEUE_RELATIVE}/{run_id}"
        env["CBB_HOST_MODEL_QUEUE"] = queue_relative

    # The MCP client owns this server process and kills its process *tree* when
    # the call ends. A plain child — even DETACHED_PROCESS — is still a descendant
    # and dies with it. So the pipeline is launched through a .cmd shim via
    # ``start /b``: cmd exits immediately, the pipeline is re-parented, and it is
    # no longer in the server's tree. That is the only way a one-shot MCP call can
    # start a long-running job on Windows.
    pid_file = work / "plan" / f"mcp-run-{run_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    if os.name == "nt":
        shim_py = Path(__file__).resolve().parent / "_mcp_run_shim.py"
        shim = work / "plan" / f"mcp-run-{run_id}.cmd"
        args_for_cmd = [str(pid_file), argv[0], *argv[1:]]
        lines = ["@echo off"]
        lines += [f'set "{k}={v}"' for k, v in env.items() if k.startswith("CBB_")]
        command = " ".join(_cmd_quote(a) for a in [argv[0], str(shim_py), *args_for_cmd])
        # The redirect MUST be on the same line as the command: on its own line
        # it just creates an empty file and the real output is lost.
        lines.append(f'{command} > "{log_path}" 2>&1')
        # Batch files are parsed in the OEM codepage; UTF-8 would mangle the
        # Chinese paths. ``newline="\n"`` avoids the \r\r\n blank-line artifact.
        shim.write_text("\r\n".join(lines) + "\r\n", encoding="mbcs", newline="")
        subprocess.Popen(
            ["cmd", "/c", "start", "", "/b", str(shim)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS,
            close_fds=True,
        )
        process_pid = _wait_for_pid_file(pid_file)
    else:
        log_file = log_path.open("w", encoding="utf-8", errors="replace")
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            start_new_session=True,
        )
        log_file.close()
        process_pid = process.pid
        pid_file.write_text(str(process_pid), encoding="utf-8")
    record = {
        "run_id": run_id,
        "work": str(work),
        "out": str(out),
        "pid": process_pid,
        "started": time.time(),
        "chapters": str(args.get("chapters") or "1-3"),
        "stages": str(args.get("stages") or ""),
        "argv": argv,
        "log": str(log_path),
        "returncode": None,
        # Empty for http/mcp transports: those have no queue.
        "queue": queue_relative,
        "preflight_status": preflight["status"],
        "preflight_blocked": preflight["blocked"],
        "preflight_forced": forced,
    }
    _save_run(work, run_id, record)
    return {
        "run_id": run_id,
        "work": str(work),
        "out": str(out),
        "pid": process_pid,
        "chapters": record["chapters"],
        "preflight": {
            "status": preflight["status"],
            "blocked": preflight["blocked"],
            "forced": forced,
        },
        "next": "call cbb_next_requests with work+run_id, then cbb_reply for each request",
    }


def _cmd_quote(value: str) -> str:
    """Quote one argument for a generated .cmd line."""
    text = str(value)
    if not text:
        return '""'
    if any(ch in text for ch in ' \t&()[]{}^=;!\'"'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _wait_for_pid_file(path: Path, timeout: float = 15.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                value = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                value = 0
            if value > 0:
                return value
        time.sleep(0.05)
    raise RuntimeError(f"pipeline did not report its pid within {timeout}s ({path})")


def _resolve_run(args: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load a run record from disk; returns (record, error_payload)."""
    work = args.get("work")
    run_id = str(args.get("run_id") or "")
    if not work:
        return None, {"error": "work is required (run state lives under <work>/plan/mcp-runs)"}
    if not run_id:
        return None, {"error": "run_id is required"}
    record = _load_run(work, run_id)
    if record is None:
        return None, {"error": f"unknown run_id {run_id!r} under {work}"}
    return record, None


def _exit_record(work: Path, run_id: str) -> dict[str, Any] | None:
    """Read the exit code the launcher shim dropped next to the pid file.

    The shim used to keep ``child.wait()``'s return value to itself, so the run
    record kept ``returncode: null`` forever. ``_mcp_run_shim.py`` now writes
    ``<pidfile>.rc``; reading it here is what turns a finished run into a
    decidable result instead of "returncode unknown".
    """
    path = work / "plan" / f"mcp-run-{run_id}.pid.rc"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _model_call_summary(work: Path) -> dict[str, Any]:
    """Aggregate ``plan/model-calls.jsonl`` — the only truthful call log.

    Neither the run log nor the trace sees host-queue calls (the trace literally
    printed "0 LLM calls" after 39 of them), and embedding calls used to bypass
    the scheduler as well, so they appeared in no log at all. Embedding calls are
    now audited as ``kind="embed"``; surfacing this file through cbb_status is
    what makes the three sources comparable at all.
    """
    path = work / "plan" / "model-calls.jsonl"
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "total": 0,
        "by_kind": {},
        "by_role": {},
        "by_status": {},
        "failed": 0,
        "elapsed_ms_total": 0.0,
    }
    if not path.exists():
        return summary
    by_kind: Counter[str] = Counter()
    by_role: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    total = 0
    failed = 0
    elapsed_total = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        total += 1
        by_kind[str(row.get("kind") or "?")] += 1
        by_role[str(row.get("role") or "?")] += 1
        status = str(row.get("status") or "?")
        by_status[status] += 1
        if status != "ok":
            failed += 1
        try:
            elapsed_total += float(row.get("elapsed_ms") or 0.0)
        except (TypeError, ValueError):
            pass
    summary.update(
        {
            "total": total,
            "by_kind": dict(by_kind),
            "by_role": dict(by_role),
            "by_status": dict(by_status),
            "failed": failed,
            "elapsed_ms_total": round(elapsed_total, 1),
        }
    )
    return summary


def _telemetry_summary(work: Path) -> dict[str, Any]:
    """Latest telemetry snapshot: counters, duration and recorded errors."""
    traces = work / "plan" / "telemetry"
    if not traces.is_dir():
        return {"path": str(traces), "exists": False}
    files = sorted(traces.glob("telemetry_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {"path": str(traces), "exists": True, "files": 0}
    latest = files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(latest), "exists": True, "parsed": False}
    counters = payload.get("counters") or {}
    return {
        "path": str(latest),
        "exists": True,
        "files": len(files),
        "duration_s": payload.get("duration_s"),
        "error_count": payload.get("error_count"),
        "counters": counters,
    }


def _terminate_tree(pid: int) -> str:
    """Stop a run's process tree.

    Returns ``""`` when nothing was terminated, ``"tree"`` when the process and its children were
    reaped, or ``"process"`` when only the process itself could be ended. The distinction is not
    cosmetic: a run's children write into its work directory, so a ``"process"`` stop can leave
    writers behind.

    `taskkill /T /F` is the only portable way to reach the children on Windows. When spawning it is
    denied outright (`PermissionError` from `CreateProcess`, seen on this host), the fallback is
    `os.kill`, which reaches the process but not the tree.
    """
    if pid <= 0:
        return ""
    if os.name == "nt":
        try:
            # No `capture_output`: the exit status is all that is needed, and avoiding pipes keeps
            # this working under another framework's output capture.
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return "tree"
        except OSError:
            pass
        try:
            os.kill(pid, signal.SIGTERM)
            return "process"
        except OSError:
            return ""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return "tree"
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
            return "process"
        except OSError:
            return ""


def _control_flag(work: str, name: str) -> Path:
    """Where the pipeline polls for a cross-process control flag."""
    return Path(work).resolve() / "plan" / name


def cbb_pause_run(args: dict[str, Any]) -> dict[str, Any]:
    """Pause a run at its next unit safe point, resumably.

    Distinct from `cbb_stop_run`: a paused run keeps its work and resumes where it stopped, a stopped
    one does not. The flag is a file under `<work>/plan/` because the pipeline runs detached — a Python
    flag in this process would never reach it.
    """
    work = str(args.get("work") or "").strip()
    if not work:
        return {"ok": False, "error": "work is required (the flag lives under <work>/plan)"}
    flag = _control_flag(work, "pause.flag")
    try:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("paused", encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    # Also set the in-process flag: when this server embeds a pipeline, the file alone is not the only
    # surface that has to agree.
    try:
        from cbb_stage1.orchestrator.progress import pause as _pause

        _pause()
    except Exception:  # noqa: BLE001 - the file is the authority for a detached run
        pass
    return {"ok": True, "paused": True, "flag": str(flag)}


def cbb_resume_run(args: dict[str, Any]) -> dict[str, Any]:
    """Release a pause. Does nothing if the run was not paused."""
    work = str(args.get("work") or "").strip()
    if not work:
        return {"ok": False, "error": "work is required (the flag lives under <work>/plan)"}
    flag = _control_flag(work, "pause.flag")
    existed = flag.exists()
    try:
        flag.unlink(missing_ok=True)
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        from cbb_stage1.orchestrator.progress import resume as _resume

        _resume()
    except Exception:  # noqa: BLE001 - the file removal above is what matters for a detached run
        pass
    return {"ok": True, "paused": False, "was_paused": existed, "flag": str(flag)}


def cbb_stop_run(args: dict[str, Any]) -> dict[str, Any]:
    """Stop a run started through `cbb_start_run`.

    The surface could start work but not stop it, so ending a run meant reaching outside the
    protocol. This terminates the recorded process **and its children**: the pipeline runs detached
    and spawns worker activity, and killing only the parent leaves processes writing into the same
    work directory.

    A run that already finished is reported as a no-op rather than a kill, and an unknown run id is
    an error rather than a success — `stopped: true` must mean a process died.
    """
    work = str(args.get("work") or "").strip()
    if not work:
        return {"ok": False, "error": "work is required (run state lives under <work>/plan/mcp-runs)"}
    run_id = str(args.get("run_id") or "").strip()
    if not run_id:
        return {"ok": False, "error": "run_id is required"}
    by = str(args.get("by") or "agent").strip() or "agent"

    work_path = Path(work).resolve()
    record = _load_run(work_path, run_id)
    if record is None:
        return {"ok": False, "error": f"no such run: {run_id}"}

    if record.get("returncode") is not None:
        return {
            "ok": True,
            "stopped": False,
            "reason": "already finished",
            "returncode": record.get("returncode"),
            "run_id": run_id,
        }

    pid = int(record.get("pid") or 0)
    stop_mode = _terminate_tree(pid)
    did_stop = bool(stop_mode)

    record["stopped_by"] = by
    record["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # `cbb_status` reads `returncode` to decide whether a run is in flight; leaving it None would
    # keep reporting a stopped run as running.
    record["returncode"] = int(record.get("returncode") or -15)
    record["stop_mode"] = stop_mode
    record["stop_note"] = (
        "terminated by cbb_stop_run" if did_stop else "process already gone"
    )
    _save_run(work_path, run_id, record)

    return {
        "ok": True,
        "stopped": bool(did_stop),
        "run_id": run_id,
        "pid": pid,
        "mode": stop_mode,
        "reason": (
            ""
            if stop_mode == "tree"
            else "only the process was stopped; children may still be running"
            if stop_mode == "process"
            else "process was already gone"
        ),
        "stopped_by": by,
        "returncode": record["returncode"],
    }


def cbb_status(args: dict[str, Any]) -> dict[str, Any]:
    record, err = _resolve_run(args)
    # A5: "why is this run not moving?" is the question status exists to answer, and a paused run is
    # not the same as a stuck one.
    _control = {}
    try:
        from cbb_stage1.orchestrator.progress import control_state

        _control = dict(control_state())
    except Exception:  # noqa: BLE001 - status must not fail because a probe did
        _control = {}
    if err is not None:
        return err
    work = Path(record["work"])
    run_id = record["run_id"]
    queue = _queue_dir(work)
    pending = len(
        [
            p
            for p in (queue / REQUESTS).glob("*.json")
            if not (queue / RESPONSES / p.name).exists()
        ]
    )
    answered = len(list((queue / RESPONSES).glob("*.json")))
    state_path = work / "plan" / "state.json"
    state: Any = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = None
    alive = _pid_alive(record.get("pid"))
    exit_record = _exit_record(work, run_id)
    returncode = record.get("returncode")
    if returncode is None and exit_record is not None:
        returncode = exit_record.get("returncode")
    payload: dict[str, Any] = {
        "run_id": run_id,
        "work": str(work),
        "out": record.get("out"),
        "pid": record.get("pid"),
        "running": alive,
        "returncode": returncode,
        "exit_record": exit_record,
        "pending_requests": pending,
        "answered_requests": answered,
        "elapsed_sec": round(time.time() - float(record.get("started") or time.time()), 1),
        "state": state,
        "preflight_status": record.get("preflight_status"),
        "preflight_blocked": record.get("preflight_blocked"),
        "preflight_forced": record.get("preflight_forced"),
        "model_calls": _model_call_summary(work),
        "telemetry": _telemetry_summary(work),
    }
    log = Path(record.get("log") or "")
    if log and log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        payload["log"] = {
            "path": str(log),
            "bytes": len(text.encode("utf-8", errors="replace")),
            "lines": text.count("\n"),
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(log.stat().st_mtime)
            ),
        }
        payload["log_tail"] = text[-3000:]
        if alive and not text:
            payload["log_hint"] = (
                "log is empty while running — the launcher passes -u; if this is "
                "still empty the pipeline is buffering or has not started"
            )
    out_dir = Path(str(record.get("out") or ""))
    if record.get("out"):
        payload["box"] = {
            "path": str(out_dir),
            "exists": out_dir.exists(),
        }
    if not alive and returncode is None:
        payload["note"] = (
            "process exited but no exit record was found; the shim writes "
            "<pidfile>.rc on completion, so a missing file means the shim "
            "itself died"
        )
    return payload


def cbb_next_requests(args: dict[str, Any]) -> dict[str, Any]:
    record, err = _resolve_run(args)
    if err is not None:
        return err
    limit = max(1, int(args.get("limit") or 5))
    queue = _queue_dir(Path(record["work"]), record.get("queue"))
    requests: list[dict[str, Any]] = []
    for path in sorted((queue / REQUESTS).glob("*.json")):
        if (queue / RESPONSES / path.name).exists():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Taking a request marks it, so a second consumer does not answer the same question.
        # Without this, measured on a 12-request queue with three consumers: 15 touches for 12
        # requests, and two consumers answering the same verifier question differently.
        if not _try_claim(queue, str(row.get("request_id") or path.stem)):
            continue
        requests.append(row)
        if len(requests) >= limit:
            break
    return {
        "run_id": record["run_id"],
        "count": len(requests),
        "requests": requests,
    }


def cbb_reply(args: dict[str, Any]) -> dict[str, Any]:
    record, err = _resolve_run(args)
    if err is not None:
        return err
    request_id = str(args["request_id"])
    queue = _queue_dir(Path(record["work"]), record.get("queue"))
    error = args.get("error")
    text = str(args.get("text") or "")
    text_file = args.get("text_file")
    if text_file:
        text = Path(str(text_file)).read_text(encoding="utf-8")
    if not error and not text:
        return {"error": "reply needs text, text_file or error"}
    # First answer wins. With several consumers on one queue, "last write wins" makes the stored
    # answer depend on scheduling — and for a verifier request two consumers can disagree.
    existing_path = queue / RESPONSES / f"{request_id}.json"
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        return {
            "run_id": record["run_id"],
            "request_id": request_id,
            "accepted": False,
            "duplicate": True,
            "kept": existing.get("text"),
            "rejected": text,
        }
    _write_json(
        queue / RESPONSES / f"{request_id}.json",
        {
            "request_id": request_id,
            "status": "error" if error else "ok",
            "text": text,
            # Forwarded, not hardcoded. The host queue contract already carried a
            # ``usage`` slot and the gateway already read it; writing ``None`` here
            # is why 73/73 calls in a real run had no token counts, which made
            # ``cost_usd`` unmeasurable and ``token_budget`` inert.
            "usage": args.get("usage"),
            "error": error,
            "replied_at": time.time(),
        },
    )
    return {"run_id": record["run_id"], "request_id": request_id, "accepted": True}


def cbb_model_call(args: dict[str, Any]) -> dict[str, Any]:
    """One model call through the gateway, dispatched by the agent.

    Straight through ``model_ops.call_once``. The agent decides to make a single call
    (probe a model, test a prompt) without going through the pipeline.
    """
    from cbb_stage1.model_ops import call_once

    for required in ("work", "provider", "model", "prompt"):
        if not str(args.get(required) or "").strip():
            return {"error": f"{required} is required"}
    return call_once(
        str(args["work"]),
        provider=str(args["provider"]),
        model=str(args["model"]),
        prompt=str(args["prompt"]),
        role=args.get("role"),
        system=str(args.get("system") or ""),
        out=args.get("out"),
        book=args.get("book"),
    )


def cbb_model_stats(args: dict[str, Any]) -> dict[str, Any]:
    """Summarise work/plan/model-calls.jsonl (roles, models, failures)."""
    from cbb_stage1.model_ops import summarise_calls

    if not str(args.get("work") or "").strip():
        return {"error": "work is required"}
    return summarise_calls(str(args["work"]))


def cbb_model_drain(args: dict[str, Any]) -> dict[str, Any]:
    """List pending host-model requests, so the agent can answer them.

    ``cbb_next_requests`` needs a run id; this works on the queue alone, which is what
    an agent wants when it only has a work directory.
    """
    from cbb_stage1.model_ops import drain_queue

    return drain_queue(str(args.get("work") or ""), queue=args.get("queue"))


def cbb_model_policy(args: dict[str, Any]) -> dict[str, Any]:
    """Write work/plan/model-policy.json without starting a run."""
    from cbb_stage1.model_ops import write_policy

    if not str(args.get("work") or "").strip():
        return {"error": "work is required"}
    return write_policy(
        str(args["work"]),
        provider=args.get("provider"),
        model=args.get("model"),
        transport=args.get("transport"),
        host_queue=args.get("host_queue"),
        mcp_command=args.get("mcp_command"),
        mcp_tool=args.get("mcp_tool"),
        embedding_base_url=args.get("embedding_base_url"),
        embedding_model=args.get("embedding_model"),
        max_concurrency=args.get("max_concurrency"),
        max_tokens=args.get("max_tokens"),
        timeout=args.get("timeout"),
        token_budget=args.get("token_budget"),
        role_models=args.get("role_models"),
        session_model_label=args.get("session_model_label"),
        judge_base_url=args.get("judge_base_url"),
        judge_model=args.get("judge_model"),
        nli_base_url=args.get("nli_base_url"),
        nli_model=args.get("nli_model"),
        whitebox_backend=args.get("whitebox_backend"),
    )


def cbb_smoke(args: dict[str, Any]) -> dict[str, Any]:
    """Offline mock smoke run: proves the pipeline works without any model.

    Calls the library directly. Both this tool and the CLI are thin front-ends over
    ``orchestrator.smoke.smoke``; neither depends on the other.
    """
    from cbb_stage1.orchestrator.smoke import smoke

    work = str(args.get("work") or "").strip()
    if not work:
        import tempfile

        work = tempfile.mkdtemp(prefix="cbb-smoke-")
    result = smoke(work)
    return {
        "ok": True,
        "data": {
            "work": work,
            "status": getattr(result, "status", None),
            "units": len(getattr(result, "units", ()) or ()),
        },
    }


#: Capabilities that exist but are deliberately not active, with the reason.
#:
#: The distinction matters for an agent-driven project: "absent" means write code,
#: "disabled" means the code is there and you must not use it. Recording the second
#: kind here keeps the decision visible instead of buried in a docstring.
DISABLED_CAPABILITIES: dict[str, dict[str, str]] = {
    "narrative_forecasting": {
        "status": "disabled",
        "component": "derive.forecasts.forecast_chapter",
        "kind": "proxy",
        "what_it_does": (
            "章级词法离散度（singleton/token 比值），写 derived/inference/forecasts.jsonl"
        ),
        "what_it_is_not": (
            "不是 100-Endings 的 no_rate。原法需要「句数 × 100 次生成」，"
            "本书 18 章约 18 万次调用，当前不可行"
        ),
        "why_kept": (
            "§5.20 要求按章做理论辅助且标理论；该实现满足契约，产物标 theory=true、"
            "method=lexical_singleton_ratio，不进事实卡"
        ),
        "usable_as": "编辑参考；不得作为预测或张力结论引用",
    },
    "uqlm_whitebox": {
        "status": "disabled",
        "component": "confidence.sep / confidence.uqlm(whitebox)",
        "kind": "blocked_by_transport",
        "what_it_does": "用 token logprobs 估算不确定性",
        "what_it_is_not": "host 传输不提供 logprobs",
        "why_kept": (
            "换白盒后端只能让它拿到 logprobs；论文的 SEP 是训练出来的线性探针，本项目实现的是"
            "手写启发式——无 head、无 checkpoint、无训练数据、无 γ⋆，所以换后端也到不了论文方法"
        ),
        "usable_as": "不可用；capability_gaps 已报 unavailable",
    },
    "uq_heads": {
        "status": "disabled",
        "component": "confidence.uq_heads",
        "kind": "needs_artifact",
        "what_it_does": "claim 级不确定性头推理",
        "what_it_is_not": "没有生产 checkpoint；且主模型随 agent 变，head 绑死基座",
        "why_kept": (
            "可现场训练验证链路，但训练的是替代表示：特征层用表层标量绕开模型内部，"
            "不是论文 attention/logits 上的 transformer 头；auto 路径也没有读取 checkpoint"
        ),
        "usable_as": "不可用；capability_gaps 已报 external",
    },
    "d14_100_endings": {
        "status": "disabled",
        "component": "（无）",
        "kind": "not_implemented",
        "what_it_does": "—",
        "what_it_is_not": "100-Endings 论文原方法，从未实现",
        "why_kept": "07 清单判定降级：论文只验证短篇",
        "usable_as": "不做",
    },
}


def cbb_capabilities(args: dict[str, Any]) -> dict[str, Any]:
    """What the agent can and cannot do through this channel.

    An agent-driven project should be able to answer "is there anything I cannot
    reach?" without a human auditing the CLI. This compares the CLI command set
    against the MCP tool set and reports the difference.
    """
    cli_path = MODULE_SOURCE_DIR / "cli.py"
    covered = {
        "run": "cbb_start_run",
        "doctor": "cbb_doctor",
        "build-box": "cbb_build_box",
        "lint-box": "cbb_lint_box",
        "acceptance": "cbb_gate",
        "wiring": "cbb_gate",
        "lexicon-list": "cbb_lexicon_list",
        "lexicon-approve": "cbb_lexicon_approve",
        "supersede": "cbb_supersede",
        "conflict-resolve": "cbb_conflict_resolve",
        "annotator-agreement": "cbb_annotator_agreement",
        "quarantine-adjudicate": "cbb_quarantine_adjudicate",
        "model-reply": "cbb_reply",
        "model-call": "cbb_model_call",
        "model-stats": "cbb_model_stats",
        "model-drain": "cbb_model_drain",
        "model-policy": "cbb_model_policy",
        "smoke": "cbb_smoke",
    }
    commands: list[str] = []
    try:
        import re as _re

        commands = sorted(
            set(_re.findall(r'sub\.add_parser\(\s*"([a-z-]+)"', cli_path.read_text(encoding="utf-8")))
        )
    except OSError:
        commands = []

    missing = [command for command in commands if command not in covered]
    tools = sorted(tool["name"] for tool in TOOLS)
    return {
        "ok": not missing,
        "data": {
            "cli_commands": commands,
            "mcp_tools": tools,
            "covered": covered,
            "missing": missing,
            # Capabilities that exist but must not be used, and why. An agent that
            # reads only the tool list would not know these are off-limits.
            "disabled_capabilities": DISABLED_CAPABILITIES,
            "note": (
                "missing 为空表示 agent 可以通过 MCP 完成全部 CLI 能力"
                if not missing
                else "这些 CLI 能力无法通过 MCP 调度"
            ),
        },
    }


def cbb_lexicon_list(args: dict[str, Any]) -> dict[str, Any]:
    """D5: proposals awaiting a decision on an unregistered predicate."""
    from cbb_stage1.assemble.lexicon_evolution import list_proposals

    proposals = list_proposals(str(args["work"]))
    return {"ok": True, "data": {"proposals": proposals, "count": len(proposals)}}


def cbb_lexicon_approve(args: dict[str, Any]) -> dict[str, Any]:
    """D5: approve one proposal. Refuses without ``by`` — predicates are not grown
    from data (I4), so the human gate has to be explicit rather than defaulted."""
    from cbb_stage1.assemble.lexicon_evolution import ApprovalError, approve_proposal

    if not str(args.get("by") or "").strip():
        return {"error": "approved_by is required (I4 human gate)"}
    try:
        result = approve_proposal(
            str(args["work"]),
            str(args["predicate"]),
            approved_by=str(args["by"]),
            library=args.get("library"),
            lexicon_path=args.get("lexicon"),
        )
    except ApprovalError as exc:
        return {"error": str(exc)}
    return {"ok": True, "data": result.to_dict()}


def cbb_supersede(args: dict[str, Any]) -> dict[str, Any]:
    """D9: mark one assertion superseded by another. Human decision, never automatic.

    ``04-执行模型`` §8 says decisions are not taken by the pipeline, so this stays a
    command an operator invokes — but it has to be reachable from the agent that
    drives the project, not only from a shell.
    """
    from cbb_stage1.validate.adjudication import supersede_assertion

    for required in ("loser", "winner", "by", "reason"):
        if not str(args.get(required) or "").strip():
            return {"error": f"{required} is required"}
    try:
        return supersede_assertion(
            str(args["work"]),
            loser=str(args["loser"]),
            winner=str(args["winner"]),
            by=str(args["by"]),
            reason=str(args["reason"]),
            at=args.get("at"),
        )
    except (KeyError, ValueError) as exc:
        return {"error": str(exc)}


def cbb_annotator_agreement(args: dict[str, Any]) -> dict[str, Any]:
    """B11: independent labelling + Cohen's kappa.

    Runs in-process through ``qc.agreement``; the MCP surface owns the capability rather
    than delegating it to the CLI.
    """
    import json as _json
    from pathlib import Path as _Path

    from cbb_stage1.qc.agreement import independent_label

    work = _Path(str(args["work"]))
    claims: dict[str, str] = {}
    evidence: dict[str, list[str]] = {}
    table = work / "assemble" / "claims.jsonl"
    if table.exists():
        for line in table.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            claim_id = str(row.get("claim_id") or "")
            if not claim_id:
                continue
            claims[claim_id] = str(row.get("normalized_claim") or "")
            evidence[claim_id] = [
                str(item.get("quote") or "")
                for item in row.get("evidence") or ()
                if str(item.get("quote") or "").strip()
            ]

    report = independent_label(
        list(claims),
        llm=None,
        claims=claims,
        evidence_by_claim=evidence,
        annotators=tuple(args.get("annotator") or ("A", "B", "C")),
        sample_size=int(args.get("sample") or 20),
    )
    return {"ok": True, "data": report.as_dict()}


def cbb_conflict_resolve(args: dict[str, Any]) -> dict[str, Any]:
    """Record a human decision about one conflict.

    ``validate/conflicts.py`` declares five choices and implemented one. This gives
    the other four a door, so the conflict can be closed by a person instead of
    sitting open. The row is annotated, never deleted.
    """
    from cbb_stage1.validate.adjudication import resolve_conflict

    for required in ("conflict_id", "choice", "by", "reason"):
        if not str(args.get(required) or "").strip():
            return {"error": f"{required} is required"}
    try:
        return resolve_conflict(
            str(args["work"]),
            conflict_id=str(args["conflict_id"]),
            choice=str(args["choice"]),
            by=str(args["by"]),
            reason=str(args["reason"]),
            at=args.get("at"),
        )
    except (KeyError, ValueError) as exc:
        return {"error": str(exc)}


def cbb_quarantine_adjudicate(args: dict[str, Any]) -> dict[str, Any]:
    """Record a human decision about one quarantined item.

    The item is annotated, never deleted: what was quarantined and why is the
    evidence that the quarantine happened, and a judgement that erases its own
    subject is not auditable.
    """
    from cbb_stage1.validate.adjudication import adjudicate_quarantine

    for required in ("item_id", "decision", "by", "reason"):
        if not str(args.get(required) or "").strip():
            return {"error": f"{required} is required"}
    try:
        return adjudicate_quarantine(
            str(args["work"]),
            item_id=str(args["item_id"]),
            decision=str(args["decision"]),
            by=str(args["by"]),
            reason=str(args["reason"]),
            at=args.get("at"),
        )
    except (KeyError, ValueError) as exc:
        return {"error": str(exc)}


def cbb_field_coverage(args: dict[str, Any]) -> dict[str, Any]:
    """Check every schema-required box field for emptiness.

    This is the check that would have caught ``verification.level`` being null in
    every row while three other gates stayed green. Exposed here because the agent
    driving the project is the one that needs to run it after building a box.
    """
    box = str(args.get("box") or "")
    if not box:
        return {"error": "box is required"}
    coverage = _load_tool_module("field_coverage")
    schema_dir = Path(
        str(args.get("schema_dir") or (PROJECT_ROOT / "schemas"))
    )
    stats, notes = coverage.check_box(Path(box), schema_dir)
    defects = [
        {"table": s.table, "field": s.field, "empty": s.empty, "rows": s.rows}
        for s in stats
        if s.required and s.empty and not s.allowed_reason
    ]
    return {
        "ok": not defects,
        "data": {"box": box, "defects": defects, "notes": notes,
                 "tables": len({s.table for s in stats})},
    }
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-4000:],
    }


def cbb_call_wiring(args: dict[str, Any]) -> dict[str, Any]:
    """Public functions with no call site anywhere.

    Import-reachability (``cbb_gate wiring``) proves a module is loaded; this proves
    its functions are invoked. The difference is where 52 written-but-unwired
    functions hid.
    """
    root = str(args.get("root") or MODULE_SOURCE_DIR)
    cw = _load_tool_module("call_wiring")
    uncalled = cw.analyse(Path(root))
    return {
        "ok": not uncalled,
        "data": {"root": root, "uncalled": uncalled, "count": len(uncalled)},
    }
    if args.get("json"):
        argv.append("--json")
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    completed = subprocess.run(
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-4000:],
    }


def cbb_build_box(args: dict[str, Any]) -> dict[str, Any]:
    from cbb_stage1.box.writer import BuildError, build_box

    try:
        result = build_box(
            str(args["work"]),
            str(args["out"]),
            series_id=args.get("series_id"),
            volume_no=(
                int(args["volume_no"]) if args.get("volume_no") is not None else None
            ),
        )
    except BuildError as exc:
        return {"error": str(exc)}
    return {"ok": True, "data": {"box": str(getattr(result, "box", args["out"]))}}


def cbb_lint_box(args: dict[str, Any]) -> dict[str, Any]:
    from cbb_stage1.box.lint import lint_box

    problems = lint_box(Path(str(args["box"])))
    return {
        "ok": not problems,
        "data": {"box": str(args["box"]), "problems": list(problems)},
    }


def cbb_gate(args: dict[str, Any]) -> dict[str, Any]:
    """Run a project gate in-process.

    Three gates, three library functions. No subprocess: a gate is a capability of the
    project, so the server runs it rather than asking the CLI to run it.
    """
    gate = str(args["gate"])

    if gate == "wiring":
        from cbb_stage1.qc.wiring import dead_modules

        dead = dead_modules()
        return {"ok": not dead, "data": {"gate": "wiring", "dead_modules": dead}}

    if gate == "id_sources":
        from cbb_stage1.qc.id_sources import scan

        violations = scan()
        return {"ok": not violations, "data": {"gate": "id_sources", "violations": violations}}

    if gate == "acceptance":
        from cbb_stage1.qc.acceptance import verify

        matrix = Path(str(args.get("matrix") or "acceptance/07.yaml"))
        passed = verify(matrix, run=True)
        return {
            "ok": bool(passed),
            "data": {"gate": "acceptance", "matrix": str(matrix), "passed": bool(passed)},
        }

    return {"error": f"unknown gate {gate!r}; expected wiring | id_sources | acceptance"}


def cbb_doctor(args: dict[str, Any]) -> dict[str, Any]:
    """Pre-flight dependency audit, callable before cbb_start_run.

    Without it, a missing backend only failed at the stage that needed it — the
    embedding endpoint inside ``derive``, the box directory inside ``build_box``.
    """
    from cbb_stage1.preflight import run_preflight

    skip = {
        str(item).strip()
        for item in (args.get("skip") or "").split(",")
        if str(item).strip()
    }
    report = run_preflight(
        book=args.get("book"),
        work=args.get("work"),
        out=args.get("out"),
        embedding_base_url=args.get("embedding_base_url"),
        llm_base_url=args.get("llm_base_url"),
        llm_model=args.get("llm_model"),
        # Under the agent-driven architecture the main reasoning is served by
        # whichever sub-agent drives the session, so the capability set varies per
        # run. These two values are how the pre-flight learns what it is driving;
        # without them the report cannot say which mechanisms will be unavailable.
        transport=args.get("transport"),
        session_model_label=args.get("session_model_label"),
        base_url=args.get("llm_base_url"),
        judge_base_url=args.get("judge_base_url"),
        judge_model=args.get("judge_model"),
        nli_base_url=args.get("nli_base_url"),
        nli_model=args.get("nli_model"),
        whitebox_backend=args.get("whitebox_backend"),
        schema_dir=args.get("schema_dir"),
        nli_config={
            "nli_provider": args.get("nli_provider"),
            "nli_model": args.get("nli_model"),
        },
        uq_head_checkpoint=args.get("uq_head_checkpoint"),
        uq_train_data=args.get("uq_train_data"),
        max_tokens=args.get("max_tokens"),
        token_budget=args.get("token_budget"),
        max_workers=args.get("max_workers"),
        skip=skip,
    )
    return report


_DISPATCH = {
    "cbb_model_call": cbb_model_call,
    "cbb_model_stats": cbb_model_stats,
    "cbb_model_drain": cbb_model_drain,
    "cbb_model_policy": cbb_model_policy,
    "cbb_smoke": cbb_smoke,
    "cbb_capabilities": cbb_capabilities,
    "cbb_lexicon_list": cbb_lexicon_list,
    "cbb_lexicon_approve": cbb_lexicon_approve,
    "cbb_supersede": cbb_supersede,
    "cbb_conflict_resolve": cbb_conflict_resolve,
    "cbb_annotator_agreement": cbb_annotator_agreement,
    "cbb_quarantine_adjudicate": cbb_quarantine_adjudicate,
    "cbb_field_coverage": cbb_field_coverage,
    "cbb_call_wiring": cbb_call_wiring,
    "cbb_start_run": cbb_start_run,
    "cbb_status": cbb_status,
    "cbb_pause_run": cbb_pause_run,
    "cbb_resume_run": cbb_resume_run,
    "cbb_stop_run": cbb_stop_run,
    "cbb_next_requests": cbb_next_requests,
    "cbb_reply": cbb_reply,
    "cbb_build_box": cbb_build_box,
    "cbb_lint_box": cbb_lint_box,
    "cbb_gate": cbb_gate,
    "cbb_doctor": cbb_doctor,
}


# ── JSON-RPC ───────────────────────────────────────────────────────────────


def _send(payload: dict[str, Any]) -> None:
    # Use ASCII escapes: the stdio pipe on Windows inherits the console codepage
    # (GBK here), and a raw ``・`` in a model prompt crashes the write mid-line,
    # which the client sees as "connection closed".
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": SERVER_INFO,
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            handler = _DISPATCH.get(name)
            if handler is None:
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": _tool_result({"error": f"unknown tool {name!r}"}),
                    }
                )
                continue
            try:
                result = handler(arguments)
            except Exception as exc:  # noqa: BLE001 - report, never fabricate
                result = {"error": f"{type(exc).__name__}: {exc}"}
            _send({"jsonrpc": "2.0", "id": request_id, "result": _tool_result(result)})
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif request_id is not None:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
