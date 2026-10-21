"""C1-C4 role runtime: five roles, file handoff, deterministic verifier pre-check.

Execution-model contract (``设计/抽取段-v6/04-执行模型-v6.1.md``):

* **Extractor** -> ``work/engine/<unit>.json`` + receipt;
* **Verifier** -> ``work/verify/`` verdicts, input **must contain the original
  unit text and the span** (never just the claimed quote);
* **Resolver / Arbiter / Abducer** -> the postprocess integration stages.

This module is the in-process boundary for those handoffs. It reuses the
``roles`` dataclasses (``cbb-task/1`` / ``cbb-receipt/1`` / ``cbb-verdict/1``)
and adds:

* the **five** live roles (``extractor, verifier, resolver, arbiter, abducer``);
  ``summarizer`` is not part of the v6.1 live chain and is rejected here;
* atomic task/receipt/verdict file writes under ``work/plan/tasks``,
  ``work/receipts`` and ``work/verify``;
* a deterministic verifier: ``span == text[start:end]`` and ``quote`` equality
  are checked first; a mismatch is ``against`` (pseudo-quote rejected) and **no
  LLM call is made**. Without an LLM every verified item is ``unsure`` - the
  runtime never invents support/confidence;
* an extractor receipt exactly mirroring the persisted engine output;
* one verifier input artifact per unit that carries the original text + spans,
  so a later agent (or LLM) can judge without trusting the extractor's quote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from cbb_stage1.ids import sha256_text

from .files import atomic_write_json
from .roles import Receipt, Task, Verdict

__all__ = [
    "FIVE_ROLES",
    "VERDICTS",
    "POSTPROCESS_ROLE",
    "Task",
    "Receipt",
    "Verdict",
    "make_task",
    "make_receipt",
    "make_verdict",
    "task_path",
    "receipt_path",
    "verdict_path",
    "record_task",
    "record_receipt",
    "record_verdict",
    "evidence_fields",
    "build_verifier_input",
    "verify_evidence",
    "verify_output",
    "record_extraction",
    "record_verification",
    "record_postprocess_chain",
]

#: The five roles on the v6.1 live chain. ``summarizer`` is intentionally absent.
FIVE_ROLES = ("extractor", "verifier", "resolver", "arbiter", "abducer")

VERDICTS = ("support", "against", "unsure")

#: Postprocess stage -> role (C1-C4: deterministic stages are code, the role
#: boundary is only where a task/receipt is recorded).
POSTPROCESS_ROLE: dict[str, str] = {
    "resolve": "resolver",
    "assemble": "resolver",
    "claims": "verifier",
    "cues": "abducer",
    "temporal": "arbiter",
    "validate": "verifier",
    "dedupe": "resolver",
    "derive": "abducer",
}

#: Relative work-artifact locations (same layout as ``roles.py``).
_TASKS_DIR = ("plan", "tasks")
_RECEIPTS_DIR = ("receipts",)
_VERIFY_DIR = ("verify",)
_VERIFIER_INPUT_DIR = ("plan", "roles", "verifier_inputs")


# =============================================================================
# Helpers
# =============================================================================

def _safe_name(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "task"


def _rel(path: Path, work_root: Path) -> str:
    try:
        return path.relative_to(work_root).as_posix()
    except ValueError:
        return str(path)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def task_path(work_root: str | Path, task_id: str) -> Path:
    return Path(work_root).joinpath(*_TASKS_DIR, f"{_safe_name(task_id)}.json")


def receipt_path(work_root: str | Path, task_id: str) -> Path:
    return Path(work_root).joinpath(*_RECEIPTS_DIR, f"{_safe_name(task_id)}.json")


def verdict_path(work_root: str | Path, task_id: str, suffix: str = "") -> Path:
    name = _safe_name(task_id) + (f"_{_safe_name(suffix)}" if suffix else "")
    return Path(work_root).joinpath(*_VERIFY_DIR, f"{name}.json")


def _write_json(path: Path, payload: dict) -> dict:
    atomic_write_json(path, payload)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return {
        "path": str(path),
        "sha256": sha256_text(data),
    }


# =============================================================================
# Task / Receipt / Verdict construction
# =============================================================================

def make_task(
    task_id: str,
    role: str,
    *,
    unit_id: str | None = None,
    input_paths: Iterable[str] | None = None,
    output_path: str | None = None,
    prompt_sha256: str | None = None,
) -> Task:
    """Build a task, rejecting roles outside the five-role live chain."""
    if role not in FIVE_ROLES:
        raise ValueError(
            f"role {role!r} is not on the v6.1 live chain; allowed: {FIVE_ROLES}"
        )
    return Task(
        task_id=task_id,
        role=role,
        unit_id=unit_id,
        input_paths=list(input_paths or []),
        output_path=output_path,
        prompt_sha256=prompt_sha256,
    )


def make_receipt(
    task_id: str,
    status: str,
    *,
    outputs: Iterable[str] | None = None,
    counts: dict[str, int] | None = None,
    notes: Iterable[str] | None = None,
) -> Receipt:
    if status not in ("ok", "failed", "abstained"):
        raise ValueError(f"receipt status must be ok/failed/abstained, got {status!r}")
    return Receipt(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        outputs=list(outputs or []),
        counts=dict(counts or {}),
        notes=list(notes or []),
    )


def make_verdict(
    task_id: str,
    verdict: str,
    *,
    evidence: Iterable[dict[str, Any]] | None = None,
    reason: str = "",
) -> Verdict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    return Verdict(
        task_id=task_id,
        verdict=verdict,  # type: ignore[arg-type]
        evidence=list(evidence or []),
        reason=reason,
    )


# =============================================================================
# Persistence
# =============================================================================

def record_task(work_root: str | Path, task: Task) -> dict:
    root = Path(work_root)
    payload = task.to_dict()
    info = _write_json(task_path(root, task.task_id), payload)
    info["task_id"] = task.task_id
    info["relative_path"] = _rel(task_path(root, task.task_id), root)
    return info


def record_receipt(work_root: str | Path, receipt: Receipt) -> dict:
    root = Path(work_root)
    info = _write_json(receipt_path(root, receipt.task_id), receipt.to_dict())
    info["task_id"] = receipt.task_id
    info["relative_path"] = _rel(receipt_path(root, receipt.task_id), root)
    return info


def record_verdict(work_root: str | Path, verdict: Verdict, *, suffix: str = "") -> dict:
    root = Path(work_root)
    path = verdict_path(root, verdict.task_id, suffix)
    info = _write_json(path, verdict.to_dict())
    info["task_id"] = verdict.task_id
    info["relative_path"] = _rel(path, root)
    return info


# =============================================================================
# Verifier (C3): original text + span in, deterministic check first
# =============================================================================

def evidence_fields(evidence: Any) -> tuple[int | None, int | None, str | None]:
    """Extract ``(start, end, quote)`` from a pydantic Evidence or a dict."""
    start = _get(evidence, "start")
    end = _get(evidence, "end")
    quote = _get(evidence, "quote")
    try:
        start = int(start)  # type: ignore[arg-type]
        end = int(end)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, None, quote
    return start, end, quote


def build_verifier_input(
    unit_text: str,
    *,
    unit_id: str,
    evidence: Any,
) -> dict:
    """Build the verifier input record: **original text + span** + observed quote."""
    start, end, quote = evidence_fields(evidence)
    observed: str | None = None
    if start is not None and end is not None and 0 <= start <= end <= len(unit_text):
        observed = unit_text[start:end]
    return {
        "contract": "cbb-verifier-input/1",
        "unit_id": unit_id,
        "text": unit_text,
        "span": {"start": start, "end": end},
        "quote_expected": quote,
        "quote_observed": observed,
    }


def _normalize_llm_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("verdict")
    if value not in VERDICTS:
        raise ValueError(
            f"LLM verifier returned {value!r}; expected one of {VERDICTS}"
        )
    return str(value)


def verify_evidence(
    unit_text: str,
    *,
    unit_id: str,
    evidence: Any,
    task_id: str | None = None,
    llm: Callable[[dict], Any] | None = None,
) -> Verdict:
    """Verify one evidence item against the original unit text.

    Order (design 04 §8):

    1. deterministic span/quote check - a mismatch is ``against`` and the LLM is
       **never** called on a pseudo-quote;
    2. if no ``llm`` is supplied the verdict is ``unsure`` (never fabricated);
    3. otherwise the LLM's support/against/unsure is recorded as-is.

    The input passed to ``llm`` always contains the original text and the span.
    """
    record = build_verifier_input(unit_text, unit_id=unit_id, evidence=evidence)
    tid = task_id or f"verify:{unit_id}"
    start = record["span"]["start"]
    end = record["span"]["end"]
    expected = record["quote_expected"]
    observed = record["quote_observed"]

    determinism: list[dict[str, Any]] = [
        {
            "unit_id": unit_id,
            "span": {"start": start, "end": end},
            "quote_expected": expected,
            "quote_observed": observed,
        }
    ]

    if start is None or end is None or start < 0 or end < start or end > len(unit_text):
        return make_verdict(
            tid,
            "against",
            evidence=determinism,
            reason="span_out_of_range",
        )
    if expected is not None and observed != expected:
        return make_verdict(
            tid,
            "against",
            evidence=determinism,
            reason="quote_span_mismatch",
        )

    if llm is None:
        return make_verdict(
            tid,
            "unsure",
            evidence=determinism,
            reason="no_llm_verifier",
        )

    try:
        verdict = _normalize_llm_value(llm(record))
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - verifier failure is uncertainty
        return make_verdict(
            tid,
            "unsure",
            evidence=determinism,
            reason=f"llm_error: {type(exc).__name__}: {exc}",
        )
    return make_verdict(tid, verdict, evidence=determinism, reason="llm_judgement")


def verify_output(
    output: Any,
    unit_text: str,
    *,
    unit_id: str | None = None,
    llm: Callable[[dict], Any] | None = None,
) -> tuple[list[Verdict], dict[str, int], list[str]]:
    """Verify every assertion evidence item of one ``EngineOutput``.

    Returns ``(verdicts, counts, detail_suffixes)``. An assertion without
    evidence yields one ``unsure`` verdict (``no_evidence``) - never support.
    """
    resolved_unit_id = unit_id or str(_get(output, "unit_id", "unknown"))
    task_id = f"verify:{resolved_unit_id}"
    verdicts: list[Verdict] = []
    suffixes: list[str] = []
    counts = {name: 0 for name in VERDICTS}
    assertions = _get(output, "assertions", []) or []
    for assertion_index, assertion in enumerate(assertions):
        assertion_id = str(_get(assertion, "assertion_id", "") or f"a{assertion_index:03d}")
        evidence_list = _get(assertion, "evidence", []) or []
        if not evidence_list:
            suffix = f"{assertion_index:03d}_noev"
            verdict = make_verdict(
                task_id,
                "unsure",
                evidence=[{"assertion_id": assertion_id, "unit_id": resolved_unit_id}],
                reason="no_evidence",
            )
            verdicts.append(verdict)
            suffixes.append(suffix)
            counts["unsure"] += 1
            continue
        for evidence_index, evidence in enumerate(evidence_list):
            suffix = f"{assertion_index:03d}_{evidence_index:02d}"
            verdict = verify_evidence(
                unit_text,
                unit_id=resolved_unit_id,
                evidence=evidence,
                task_id=task_id,
                llm=llm,
            )
            # Keep the assertion id on the same evidence rows so a verifier
            # input artifact carries span + quote + assertion together.
            for entry in verdict.evidence:
                entry.setdefault("assertion_id", assertion_id)
            verdicts.append(verdict)
            suffixes.append(suffix)
            counts[verdict.verdict] += 1
    return verdicts, counts, suffixes


# =============================================================================
# Pipeline handoffs
# =============================================================================

def record_extraction(
    work_root: str | Path,
    unit: Any,
    output: Any,
    *,
    output_path: str,
    counts: dict[str, int],
    cached: bool,
) -> dict:
    """Record the Extractor task + receipt for one persisted engine output."""
    root = Path(work_root)
    task_id = f"extract:{unit.unit_id}"
    task = make_task(
        task_id,
        "extractor",
        unit_id=unit.unit_id,
        input_paths=[unit.text_path],
        output_path=output_path,
    )
    task_info = record_task(root, task)
    status = "abstained" if bool(_get(output, "abstain", False)) else "ok"
    notes = ["cached from idempotency/checkpoint"] if cached else []
    receipt = make_receipt(
        task_id,
        status,
        outputs=[output_path],
        counts=counts,
        notes=notes,
    )
    receipt_info = record_receipt(root, receipt)
    return {"task": task_info, "receipt": receipt_info, "task_id": task_id}


def record_verification(
    work_root: str | Path,
    unit: Any,
    output: Any,
    *,
    llm: Callable[[dict], Any] | None = None,
) -> dict:
    """Write the verifier input (original text + spans), task, verdicts, receipt."""
    root = Path(work_root)
    unit_id = unit.unit_id
    verdicts, counts, suffixes = verify_output(
        output, unit.text, unit_id=unit_id, llm=llm
    )

    input_payload = {
        "contract": "cbb-verifier-input/1",
        "unit_id": unit_id,
        "text": unit.text,
        "items": [entry for verdict in verdicts for entry in verdict.evidence],
    }
    input_path = root.joinpath(*_VERIFIER_INPUT_DIR, f"{_safe_name(unit_id)}.json")
    input_info = _write_json(input_path, input_payload)

    task_id = f"verify:{unit_id}"
    task = make_task(
        task_id,
        "verifier",
        unit_id=unit_id,
        input_paths=[unit.text_path, _rel(input_path, root)],
    )
    task_info = record_task(root, task)

    verdict_infos: list[dict] = []
    for verdict, suffix in zip(verdicts, suffixes):
        verdict_infos.append(record_verdict(root, verdict, suffix=suffix))

    receipt = make_receipt(
        task_id,
        "ok",
        outputs=[info["relative_path"] for info in verdict_infos],
        counts=counts,
        notes=["no LLM: uncertain items stay unsure"] if llm is None else [],
    )
    receipt_info = record_receipt(root, receipt)
    return {
        "task_id": task_id,
        "task": task_info,
        "receipt": receipt_info,
        "input_path": _rel(input_path, root),
        "input_sha256": input_info["sha256"],
        "counts": counts,
        "verdict_paths": [info["relative_path"] for info in verdict_infos],
    }


def record_postprocess_chain(
    work_root: str | Path,
    stages: Iterable[dict],
    *,
    book_id: str,
) -> list[dict]:
    """Record one resolver/arbiter/abducer task+receipt per postprocess stage."""
    root = Path(work_root)
    records: list[dict] = []
    upstream: list[str] = []
    for index, stage in enumerate(stages, 1):
        name = str(stage.get("name", ""))
        role = POSTPROCESS_ROLE.get(name)
        if role is None:
            raise ValueError(
                f"postprocess stage {name!r} has no C1-C4 role mapping; "
                f"known stages: {sorted(POSTPROCESS_ROLE)}"
            )
        status = "ok" if stage.get("status") == "ok" else "failed"
        task_id = f"postprocess:{book_id}:{index:02d}:{name}"
        input_paths = list(upstream)
        stage_paths = [str(path) for path in stage.get("paths", [])]
        if role == "verifier" and not input_paths:
            # The execution contract requires a verifier task to point at its
            # source input. The wiring artifact always exists at runtime and
            # records exactly what this verifier stage consumed.
            input_paths = stage_paths or ["plan/wiring.json"]
        task = make_task(
            task_id,
            role,
            input_paths=input_paths,
        )
        task_info = record_task(root, task)
        notes = [str(stage.get("error"))] if stage.get("error") else []
        receipt = make_receipt(
            task_id,
            status,
            outputs=[str(path) for path in stage.get("paths", [])],
            counts={
                key: int(value)
                for key, value in (stage.get("counts") or {}).items()
                if isinstance(value, int) and not isinstance(value, bool)
            },
            notes=notes,
        )
        receipt_info = record_receipt(root, receipt)
        upstream = [str(path) for path in stage.get("paths", [])]
        records.append(
            {
                "stage": name,
                "role": role,
                "task": task_info,
                "receipt": receipt_info,
                "status": status,
            }
        )
    return records
