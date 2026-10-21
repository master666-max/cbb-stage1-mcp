"""Work-table producer/consumer contract gate.

The audit found orphan tables (``promoted_facts``) and tables with no producer
(``assemble/events``).  This module makes the contract explicit and checkable.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

CONTRACT = Path(__file__).resolve().parents[3] / "acceptance" / "table_contract.yaml"
EMPTY_POLICIES = {"EMPTY-VALID", "GAP", "EXTERNAL"}


def load_contract(path: Path | None = None) -> dict[str, dict]:
    raw = yaml.safe_load((path or CONTRACT).read_text(encoding="utf-8")) or {}
    return dict(raw.get("tables") or {})


def check_contract(path: Path | None = None) -> list[str]:
    problems: list[str] = []
    for table, spec in load_contract(path).items():
        policy = spec.get("empty_policy")
        if policy not in EMPTY_POLICIES:
            problems.append(f"{table}: empty_policy={policy!r} not in {sorted(EMPTY_POLICIES)}")
        producer = str(spec.get("producer") or "")
        if not producer:
            problems.append(f"{table}: no producer")
        else:
            try:
                importlib.import_module(producer)
            except Exception as exc:  # noqa: BLE001 - report, do not hide
                problems.append(f"{table}: producer {producer} import failed: {exc}")
        consumers = spec.get("consumers") or []
        if not consumers:
            problems.append(f"{table}: no consumers")
        for consumer in consumers:
            try:
                importlib.import_module(str(consumer))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{table}: consumer {consumer} import failed: {exc}")
    return problems


def check_work_tables(work_root: Path, path: Path | None = None) -> list[str]:
    """Runtime check: GAP tables must exist and be non-empty; EMPTY-VALID may be absent/empty."""
    problems: list[str] = []
    for table, spec in load_contract(path).items():
        if spec.get("empty_policy") != "GAP":
            continue
        target = Path(work_root) / table
        if not target.exists():
            problems.append(f"{table}: GAP table missing")
        elif target.stat().st_size == 0:
            problems.append(f"{table}: GAP table empty")
    return problems
