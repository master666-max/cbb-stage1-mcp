"""Static gate: contract IDs/SHA must come from ``cbb_stage1.ids``.

This catches the historical failure mode where ``a:/e:/ev:/...`` formulas were
copied into several layers and slowly drifted.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "ids.py",
    "box/atomic.py",
    "evidence/anchor.py",
    "temporal/ids.py",
    "box/lexicon.py",
}
ID_PREFIXES = ("a:", "e:", "ev:", "m:", "cue:", "t:", "mg:", "pp:", "cl:", "u:", "sm:", "hy:", "cm:", "cr:", "rt:", "c:", "q:", "fc:")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scan(root: Path | None = None) -> list[str]:
    root = root or ROOT
    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = _relative(path)
        if rel in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"_sha16", "_short_hash"}:
                problems.append(f"{rel}:{node.lineno}: duplicate hash helper {node.name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"sha256", "sha1"} and getattr(node.func.value, "id", "") == "hashlib":
                    problems.append(f"{rel}:{node.lineno}: direct hashlib.{node.func.attr}")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left = node.left
                if isinstance(left, ast.Constant) and isinstance(left.value, str):
                    if left.value in ID_PREFIXES:
                        problems.append(f"{rel}:{node.lineno}: local id prefix {left.value!r}")
    return problems


def main() -> int:
    problems = scan()
    for problem in problems:
        print(problem)
    print(f"{len(problems)} id-source violation(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
