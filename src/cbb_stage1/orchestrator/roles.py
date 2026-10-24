"""Execution roles and file handoff (v6.1).

The pipeline itself is deterministic; these helpers are the boundary used when
an agent (or a human) fills an Extractor/Verifier/Resolver/Arbiter role. They
write tasks, receipts and verdicts as files so the Lead never has to read a
teammate's full context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cbb_stage1.ids import sha256_text

ROLES = ("extractor", "verifier", "resolver", "arbiter", "abducer", "summarizer")


@dataclass
class Task:
    task_id: str
    role: str
    unit_id: str | None
    input_paths: list[str] = field(default_factory=list)
    output_path: str | None = None
    prompt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")
        if self.role == "verifier" and not self.input_paths:
            raise ValueError("verifier task must include the source unit/span input")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "cbb-task/1",
            "task_id": self.task_id,
            "role": self.role,
            "unit_id": self.unit_id,
            "input_paths": self.input_paths,
            "output_path": self.output_path,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass
class Receipt:
    task_id: str
    status: Literal["ok", "failed", "abstained"]
    outputs: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "cbb-receipt/1",
            "task_id": self.task_id,
            "status": self.status,
            "outputs": self.outputs,
            "counts": self.counts,
            "notes": self.notes,
        }


@dataclass
class Verdict:
    task_id: str
    verdict: Literal["support", "against", "unsure"]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "cbb-verdict/1",
            "task_id": self.task_id,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "reason": self.reason,
        }


def _atomic_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return sha256_text(data)


def _safe_name(value: str) -> str:
    """Task ids contain ``:``/``/``; filesystems do not."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "task"


def write_task(work: Path, task: Task) -> str:
    return _atomic_json(work / "plan" / "tasks" / f"{_safe_name(task.task_id)}.json", task.to_dict())


def write_receipt(work: Path, receipt: Receipt) -> str:
    return _atomic_json(work / "receipts" / f"{_safe_name(receipt.task_id)}.json", receipt.to_dict())


def write_verdict(work: Path, verdict: Verdict) -> str:
    return _atomic_json(work / "verify" / f"{_safe_name(verdict.task_id)}.json", verdict.to_dict())

#: Where the exec receipt contract lives, relative to a resolved schema directory.
EXEC_RECEIPT_SCHEMA = "exec/receipt.schema.json"


def receipt_contract_gaps() -> tuple[str, ...]:
    """Fields the exec receipt schema requires that :class:`Receipt` does not carry.

    The shortfall is real and unresolved, not a formatting difference: `schemas/exec/receipt
    .schema.json` demands 14 fields and `Receipt` carries five. Two of the missing ones cannot be
    supplied honestly here — `tokens` and `cost_usd` need usage reporting that the `host` transport
    does not provide, and writing zeros would assert "no tokens were used", which is a fabricated
    number of exactly the kind this project forbids elsewhere.

    Computed from the schema rather than listed by hand, so it cannot go stale: a field added to
    `Receipt` shrinks the tuple, and a field added to the schema grows it.

    Returns:
        Sorted, de-duplicated field names. Empty would mean the contract is met.
    """
    import json

    from cbb_stage1 import schemas

    for directory in schemas.candidate_dirs():
        path = directory / EXEC_RECEIPT_SCHEMA
        if not path.is_file():
            continue
        try:
            contract = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required = contract.get("required") or ()
        carried = set(Receipt.__dataclass_fields__)
        return tuple(sorted({str(name) for name in required} - carried))
    return ()
