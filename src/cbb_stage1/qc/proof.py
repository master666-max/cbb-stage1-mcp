"""Proof-layer validation for the 07 acceptance matrix.

The old gate accepted ``test:some_module_test.py`` as enough.  This module
requires every PASS item to declare a proof: entry, modules, work/box artifacts,
a concrete acceptance test node, and a level.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "acceptance" / "07.yaml"
PROOFS = ROOT / "acceptance" / "proofs.yaml"
PROOF_TESTS = ROOT / "tests" / "acceptance"

LEVELS = {"WIRED", "LIVE", "EXTERNAL", "BLOCKED"}
PARTIAL_WORDS = ("部分", "未实现", "未接", "结构版", "EXTERNAL", "只进")


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_proofs() -> dict[str, dict]:
    return dict(load(PROOFS).get("proofs") or {})


def check_proofs() -> list[str]:
    problems: list[str] = []
    items = {str(i["id"]): i for i in load(MATRIX).get("items", [])}
    proofs = load_proofs()
    for iid, item in items.items():
        if item.get("expected") != "PASS":
            continue
        proof = proofs.get(iid)
        if not proof:
            problems.append(f"{iid}: PASS without proof")
            continue
        level = proof.get("level")
        if level not in LEVELS:
            problems.append(f"{iid}: bad level {level!r}")
        test = str(proof.get("test") or "")
        test_path = test.split("::", 1)[0]
        if not test_path or not (ROOT / test_path).exists():
            problems.append(f"{iid}: proof test missing {test!r}")
        for key in ("work", "box"):
            if key not in proof:
                problems.append(f"{iid}: proof missing {key}")
        if level in {"EXTERNAL", "BLOCKED"}:
            problems.append(f"{iid}: PASS cannot be {level}; reclassify expected or supply artifact")
    extra = sorted(set(proofs) - {iid for iid, item in items.items() if item.get("expected") == "PASS"})
    if extra:
        problems.append(f"proofs for non-PASS/unknown ids: {extra}")
    return problems


def status_conflicts() -> list[str]:
    out: list[str] = []
    for item in load(MATRIX).get("items", []):
        if item.get("expected") != "PASS":
            continue
        status = str(item.get("status_text") or "")
        if any(word in status for word in PARTIAL_WORDS):
            out.append(f"{item['id']}: PASS but status_text={status!r}")
    return out


def run_proof_tests() -> int:
    cmd = [sys.executable, "-m", "pytest", str(PROOF_TESTS), "-q"]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode
