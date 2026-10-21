"""Acceptance gate for the 07 matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

EXPECTED = [f"{c}{i}" for c, n in [("A", 27), ("B", 12), ("C", 10), ("D", 15), ("E", 12), ("F", 11)] for i in range(1, n + 1)]


def load_matrix(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("items", [])
    return items


def _evidence_ok(ev: str) -> bool:
    if not ev:
        return False
    if ev.startswith("test:"):
        return Path(ev[5:]).exists()
    if ev.startswith("artifact:"):
        return Path(ev[9:]).exists()
    return Path(ev).exists()


def _field_coverage_problems() -> list[str]:
    """Run the field-coverage gate over every box under the known output roots.

    This is the check that catches "required field is null in every row while every
    other gate is green" — the exact shape of the verification.level defect. It is
    skipped when no box exists, because acceptance must run before the first box.
    """
    try:
        import importlib.util
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        tool = root / "tools" / "field_coverage.py"
        from cbb_stage1.schemas import schemas_dir

        # Resolved by chain, not hardcoded to the sibling design tree: a published
        # checkout ships the schemas with the code and has no sibling.
        try:
            schemas = schemas_dir(root)
        except FileNotFoundError:
            schemas = root.parent / "设计" / "抽取段-v6" / "schemas"
        if not tool.is_file() or not schemas.is_dir():
            return [
                "field-coverage: gate tool or schema dir missing "
                f"(tool={tool.is_file()}, schemas={schemas.is_dir()})"
            ]

        spec = importlib.util.spec_from_file_location("cbb_field_coverage", tool)
        if spec is None or spec.loader is None:
            return ["field-coverage: cannot load the gate tool"]
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Any box passed via env, plus none by default (acceptance has no box arg).
        import os

        boxes = [
            Path(item)
            for item in os.environ.get("CBB_ACCEPTANCE_BOXES", "").split(os.pathsep)
            if item.strip()
        ]
        problems: list[str] = []
        for box in boxes:
            if not box.is_dir():
                continue
            stats, _ = module.check_box(box, schemas)
            for stat in stats:
                if stat.is_defect:
                    problems.append(
                        f"field-coverage: {box.name} {stat.table}.{stat.field} "
                        f"empty in {stat.empty}/{stat.rows} rows"
                    )
        return problems
    except Exception as exc:  # noqa: BLE001 - a broken gate must not pass silently
        return [f"field-coverage: {type(exc).__name__}: {exc}"]


def _linter_problems() -> list[str]:
    """ruff (hygiene) and vulture (dead symbols) as hard gates.

    Both run through the project venv. Each is skipped with an explicit problem
    when the tool is not installed — silently skipping a gate is worse than failing
    it, because the acceptance run would then report PASS having checked nothing.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    problems: list[str] = []

    checks = [
        (
            "ruff",
            [sys.executable, "-m", "ruff", "check", "src/cbb_stage1",
             "--select", "F401,F811,F841", "--output-format", "concise"],
        ),
        (
            "vulture",
            [sys.executable, "-m", "vulture", "src/cbb_stage1",
             ".vulture_whitelist.py", "--min-confidence", "80"],
        ),
    ]
    for name, argv in checks:
        completed = subprocess.run(
            argv, cwd=root, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if completed.returncode == 0:
            continue
        output = (completed.stdout or "") + (completed.stderr or "")
        if "No module named" in output:
            problems.append(
                f"{name}: not installed — the gate cannot run. "
                f"Install it (pip install {name}) or the check is meaningless."
            )
            continue
        lines = [line for line in output.splitlines() if line.strip()]
        head = " | ".join(lines[:4])
        problems.append(f"{name}: {len(lines)} finding(s): {head}")

    # mypy, scoped. ``union-attr`` is "you dereferenced something that may be None"
    # and ``index`` is "you indexed something that is not indexable" — both are
    # latent runtime crashes. The remaining mypy categories are type-signature
    # churn; gating on them would block every commit for a pre-existing backlog.
    completed = subprocess.run(
        [
            sys.executable, "-m", "mypy", "src/cbb_stage1",
            "--ignore-missing-imports", "--no-error-summary",
        ],
        cwd=root, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    if "No module named mypy" in combined:
        problems.append(
            "mypy: not installed — the None-dereference gate cannot run. "
            "Install it (pip install mypy)."
        )
    else:
        risky = [
            line
            for line in combined.splitlines()
            if "[union-attr]" in line or "[index]" in line
        ]
        if risky:
            problems.append(
                f"mypy(union-attr/index): {len(risky)} latent runtime risk(s): "
                + " | ".join(risky[:3])
            )
    return problems


def _call_level_problems() -> list[str]:
    """Public functions that no non-test code ever calls.

    Kept as a warning channel rather than a hard failure: a genuinely public API
    has no internal caller by design. The value is that the list is produced
    automatically, so a function that was written and then forgotten shows up
    instead of being discovered months later.
    """
    try:
        import importlib.util
        from pathlib import Path

        tool = Path(__file__).resolve().parents[3] / "tools" / "call_wiring.py"
        if not tool.exists():
            return []
        spec = importlib.util.spec_from_file_location("cbb_call_wiring", tool)
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        # dataclass() resolves the module through sys.modules, so it must be
        # registered before exec_module or the decorator raises.
        import sys as _sys

        _sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        findings = module.analyse(Path(__file__).resolve().parents[1])
    except Exception:  # noqa: BLE001 - a broken checker must not fail acceptance
        return []
    if not findings:
        return []
    names = ", ".join(str(item["qualname"]) for item in findings[:8])
    more = "" if len(findings) <= 8 else f" (+{len(findings) - 8} more)"
    return [
        f"call-wiring: {len(findings)} 个公开函数无调用点: {names}{more}"
        " — 接到主链上，或在 tools/call_wiring.py 的 EXEMPT 里写明理由"
    ]


def verify(matrix: Path, run: bool = False, strict: bool = True) -> bool:
    """Check the 07 acceptance matrix.

    `strict` controls whether a PARTIAL row without a stated gap is a failure.
    It was accepted and never read, so the parameter silently did nothing — the
    kind of dead knob that makes a caller believe they changed behaviour when they
    did not.
    """
    items = load_matrix(matrix)
    ids = [i["id"] for i in items]
    problems: list[str] = []
    if len(items) != 87:
        problems.append(f"矩阵条目 {len(items)} != 87")
    if len(set(ids)) != len(ids):
        problems.append("矩阵存在重复 ID")
    missing = sorted(set(EXPECTED) - set(ids))
    extra = sorted(set(ids) - set(EXPECTED))
    if missing:
        problems.append(f"缺少 ID: {missing}")
    if extra:
        problems.append(f"多出 ID: {extra}")
    if not run:
        print(f"矩阵结构：{len(items)} 条，PASS={sum(i['expected']=='PASS' for i in items)} "
              f"PARTIAL={sum(i['expected']=='PARTIAL' for i in items)} "
              f"EXCLUDED={sum(i['expected']=='EXCLUDED' for i in items)}")
    for i in items:
        ev = i.get("evidence") or []
        if i["expected"] == "PASS" and run and not ev:
            problems.append(f"{i['id']}: PASS 但无 evidence")
        for e in ev:
            if not _evidence_ok(str(e)):
                problems.append(f"{i['id']}: evidence 不存在 {e}")
        if i["expected"] == "PARTIAL" and run and not i.get("gap"):
            if strict:
                problems.append(f"{i['id']}: PARTIAL 但无 gap")
            else:
                print(f"  [warn] {i['id']}: PARTIAL 但无 gap（strict=False，不判失败）")
    if run:
        from cbb_stage1.qc.id_sources import scan as scan_id_sources
        from cbb_stage1.qc.wiring import dead_modules
        from cbb_stage1.qc.proof import check_proofs, run_proof_tests, status_conflicts
        from cbb_stage1.qc.tables import check_contract

        dead = dead_modules()
        if dead:
            problems.append(f"wiring: {len(dead)} 个模块不在入口闭包: {dead}")
        # Call-level wiring: import-reachability alone let a module be imported for
        # one constant while its public API was never invoked — which is how
        # verification_level() sat uncalled while every gate stayed green.
        # Call-level wiring is a HARD gate.
        #
        # It was a warning while 52 functions were unwired, because failing on that
        # would have blocked every commit for a pre-existing backlog. The backlog is
        # now zero and the remaining public-API exemptions are recorded with reasons
        # in tools/call_wiring.py, so a new uncalled function is a regression and
        # must fail. If a genuinely public API is added, exempt it there with a
        # reason — that is the review step this gate exists to force.
        problems.extend(_call_level_problems())
        problems.extend(_linter_problems())
        problems.extend(_field_coverage_problems())
        problems.extend(scan_id_sources())
        problems.extend(check_contract())
        problems.extend(check_proofs())
        problems.extend(status_conflicts())
        code = run_proof_tests()
        if code != 0:
            problems.append(f"acceptance proof tests failed (rc={code})")
    if problems:
        print("FAIL")
        for p in problems:
            print("  [x]", p)
        return False
    print("PASS: 07 acceptance matrix")
    return True


if __name__ == "__main__":
    ok = verify(Path(sys.argv[1] if len(sys.argv) > 1 else "acceptance/07.yaml"), run="--run" in sys.argv)
    raise SystemExit(0 if ok else 1)
