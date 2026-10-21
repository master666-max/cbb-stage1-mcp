"""Static wiring gate: every functional module must be reachable from an entry.

This is deliberately stricter than unit tests: a module imported only by tests
does not count.  Run ``python -m cbb_stage1.qc.wiring --check`` in CI.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ENTRYPOINTS = (
    "cbb_stage1.cli",
    "cbb_stage1.orchestrator.pipeline",
    "cbb_stage1.box.writer",
    "cbb_stage1.qc.acceptance",
)

#: Package ``__init__`` modules are implicit; do not report them as dead.
_IGNORE = {"cbb_stage1"}


def _modules(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        rel = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        if rel.endswith(".__init__"):
            rel = rel[: -len(".__init__")]
        out[rel] = path
    return out


def _resolve_from(current: str, node: ast.ImportFrom, modules: dict[str, Path]) -> set[str]:
    if node.level == 0:
        base = node.module or ""
    else:
        pkg = current.split(".")[:-1]
        up = node.level - 1
        if up:
            pkg = pkg[:-up] if up <= len(pkg) else []
        base = ".".join(pkg + ([node.module] if node.module else []))
    found: set[str] = set()
    if base in modules:
        found.add(base)
    if node.level > 0 and node.module:
        direct_module = f"{current}.{node.module}"
        if direct_module in modules:
            found.add(direct_module)
    for alias in node.names:
        if alias.name == "*":
            continue
        # ``from . import x`` inside a package __init__ resolves against the
        # package itself (we normalized __init__ to the package name).
        if node.level > 0:
            direct = f"{current}.{alias.name}"
            if direct in modules:
                found.add(direct)
        candidate = f"{base}.{alias.name}" if base else alias.name
        if candidate in modules:
            found.add(candidate)
    return found


def build_graph(root: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    modules = _modules(root)
    edges: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                edges[name] |= _resolve_from(name, node, modules)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        edges[name].add(alias.name)
        # Dynamic stage specs / importlib.import_module("cbb_stage1.x") are
        # explicit wiring points; count only those calls, not every string in
        # docstrings or __all__.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_import = (
                isinstance(func, ast.Attribute) and func.attr in ("import_module", "__import__")
            ) or (isinstance(func, ast.Name) and func.id == "__import__")
            if is_import and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in modules:
                        edges[name].add(arg.value)
            for kw in node.keywords:
                if kw.arg == "module" and isinstance(kw.value, ast.Constant):
                    value = kw.value.value
                    if isinstance(value, str) and value in modules:
                        edges[name].add(value)
            if isinstance(func, ast.Name) and func.id == "StageSpec" and len(node.args) >= 2:
                arg = node.args[1]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in modules:
                        edges[name].add(arg.value)
    return modules, edges


def dead_modules(root: Path | None = None) -> list[str]:
    root = root or Path(__file__).resolve().parents[1]
    modules, edges = build_graph(root)
    seen: set[str] = set()

    # An entrypoint that does not exist is the worst case: walking it silently
    # succeeds and the gate reports "0 dead modules" for a tree missing a whole
    # package. Report it as dead so the gate fails instead of lying.
    dead: list[str] = [entry for entry in ENTRYPOINTS if entry not in modules]

    def walk(name: str) -> None:
        if name in seen or name not in edges:
            return
        seen.add(name)
        for child in edges[name]:
            walk(child)
        # A module imported dynamically (e.g. StageSpec importlib) still brings
        # its package __init__ into the runtime; walk the parent package too.
        parent = name.rsplit(".", 1)[0]
        if parent and parent in edges:
            walk(parent)

    for entry in ENTRYPOINTS:
        walk(entry)
    for name in sorted(modules):
        if name in _IGNORE:
            continue
        # A package __init__ is implicit if any child is reachable.
        if any(other.startswith(name + ".") for other in seen):
            continue
        if name not in seen:
            dead.append(name)
    return dead


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbb-wiring")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    dead = dead_modules(Path(args.root) if args.root else None)
    for name in dead:
        print(f"DEAD {name}")
    print(f"{len(dead)} dead modules")
    return 0 if not dead else 1


if __name__ == "__main__":
    sys.exit(main())
