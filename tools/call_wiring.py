"""Call-level wiring: is a module's public API actually CALLED, not just imported?

The existing gate (``qc/wiring.py``) proves every module is reachable from an entry
point via the import graph. That is necessary but far too weak: a module can be
imported for one constant and its entire public API never invoked, and the gate
still reports "0 dead modules".

That gap is exactly how ``verification.level`` stayed null through three green
gates: ``records.py`` was imported (so wiring passed), and the code that *validates*
the field ran on every row (so tests passed) — but nothing ever *computed* it.

This checker is deliberately conservative. It reports a function as "uncalled" only
when its name appears nowhere outside its own definition and the tests, because
false positives here are worse than misses: an over-eager report gets ignored, and
then the real ones are ignored too.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path

#: Functions that are part of a public contract but legitimately have no internal
#: caller. Each entry needs a reason; the list is reviewed, not appended to.
EXEMPT: dict[str, str] = {
    #: A public API with no internal caller is not a defect. Forcing a call
    #: site would make this check pass while teaching the next reader that it
    #: means nothing. Every entry states why it has no caller.
    "cbb_stage1.assemble.assertions.assemble_assertions": "批量装配入口；integration 逐个调 assemble_one（批量版留给外部调用方）",
    "cbb_stage1.assemble.registry.seed_registry": "种子注册表构造入口；内部直接用 SEED_LEXICON 常量",
    "cbb_stage1.box.lexicon.iter_libraries": "遍历六库的只读工具",
    "cbb_stage1.box.lexicon.lexicon_from_work": "从 work 读 lexicon；writer 自己读了同样的文件",
    "cbb_stage1.box.projection_tables.build_assets": "废弃：已被 extract_assets 取代，保留向后兼容的空表",
    "cbb_stage1.claims.qa.make_answer": "answer 构造工厂，供调用方",
    "cbb_stage1.claims.verdicts.apply_abstain": "abstain_decision 的封装；claims 直接用了后者",
    "cbb_stage1.confidence.available_methods": "诊断用：列出可用置信方法",
    "cbb_stage1.confidence.uq_features.feature_dict": "UQ 特征导出，供训练/调试",
    "cbb_stage1.cues.state.can_transition": "apply_event 的 bool 封装；已接线的是 apply_event",
    "cbb_stage1.cues.state.transition_cue": "apply_event 的 dict 封装",
    "cbb_stage1.dedupe.describe.dedupe_description_fragments": "dedupe_fragments 的别名",
    "cbb_stage1.dedupe.describe.merge_observations": "合成器路径已覆盖；此函数是底层原语",
    "cbb_stage1.dedupe.describe.render_observations": "调试用渲染",
    "cbb_stage1.derive.community.drilldown": "社区 drill-down 查询接口（段二用）",
    "cbb_stage1.derive.community.validate_community_hierarchy": "层级校验",
    "cbb_stage1.derive.hypotheses.derive_hypotheses": "abduce 的别名路径；integration 用 abduce",
    "cbb_stage1.derive.raptor.validate_raptor_tree": "树校验",
    "cbb_stage1.evidence.anchor.locate_surface": "choose_occurrence 的便利入口",
    "cbb_stage1.incremental.deps.entity_deps": "E7 entity 级 deps；runner 自己维护 deps 字典，公开入口留给增量写入器",
    "cbb_stage1.ingest.chunk.chars_to_tokens_approx": "估算工具；chunk 内部有等价实现",
    "cbb_stage1.normalize.facts.sanitize_mapped_output": "clean_engine_output 的封装",
    "cbb_stage1.orchestrator.idempotency.generate_content_hash": "幂等哈希；IdempotencyStore 内部直接调 sha256_hex_text",
    "cbb_stage1.orchestrator.progress.add_progress_listener": "进度订阅（宿主/UI 用）",
    "cbb_stage1.orchestrator.progress.gate_unsafe_step": "危险步骤闸",
    "cbb_stage1.orchestrator.progress.get_progress_events": "进度事件查询",
    "cbb_stage1.orchestrator.progress.get_progress_warnings": "进度告警查询",
    "cbb_stage1.orchestrator.progress.remove_progress_listener": "进度退订",
    "cbb_stage1.orchestrator.progress.request_interrupt": "中断请求",
    "cbb_stage1.orchestrator.progress.set_interactive_mode": "交互模式开关",
    "cbb_stage1.orchestrator.roles.write_receipt": "角色回执落盘",
    "cbb_stage1.orchestrator.roles.write_task": "角色任务落盘（04 执行模型回执面）",
    "cbb_stage1.orchestrator.roles.write_verdict": "角色判定落盘",
    "cbb_stage1.orchestrator.tracer.get_active_trace": "trace 查询",
    "cbb_stage1.orchestrator.tracer.get_current_trace_id": "trace 查询",
    "cbb_stage1.orchestrator.tracer.load_trace": "trace 加载（离线分析用）",
    "cbb_stage1.resolve.alias.alias_is_safe": "别名安全判定；resolve 内部有等价逻辑",
    "cbb_stage1.resolve.alias.canonical_surface": "规范表面名工具",
    "cbb_stage1.temporal.nli.expected_answer": "NLI 期望答案，测试辅助",
    "cbb_stage1.validate.conflicts.validate_conflict": "单条冲突校验，供外部检查",
    "cbb_stage1.dedupe.semhash.dedupe_graph": "A12 图级去重；graph_edges 是投影表（从断言派生），无独立图输入可去重",
    "cbb_stage1.engines.ai_reader.alias_resolver.invalidate_alias_cache": "增量场景的别名缓存失效；增量链路尚未在真实多章书上跑过",
    "cbb_stage1.engines.ai_reader.budget.get_budget": "compute_budget 的封装；调用方直接用 compute_budget",
    "cbb_stage1.derive.forecasts.forecast_chapter": "占位：始终返回 None 并携带「为何不做」的理由；真实现落位时替换它",
    "cbb_stage1.ids.forecast_id": "forecast 表仍写在契约里（当前为空），id 公式留待真实现使用",
}


@dataclass
class FunctionInfo:
    module: str
    name: str
    lineno: int
    private: bool

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _collect_definitions(root: Path) -> list[FunctionInfo]:
    out: list[FunctionInfo] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        module = _module_name(root, path)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(
                    FunctionInfo(
                        module=module,
                        name=node.name,
                        lineno=node.lineno,
                        private=node.name.startswith("_"),
                    )
                )
    return out


def _collect_references(paths: list[Path]) -> dict[str, int]:
    """Count how often each referenced name appears outside its definition.

    ``import x as y`` is followed: the *original* name is credited, because
    ``from cbb_stage1.ids import citation_id as _ids_citation_id`` is a real call
    site of ``citation_id``. Missing this produced false positives — the first
    scan reported ``ids.citation_id`` as uncalled while projection_tables used it
    on every row under a private alias.
    """
    counts: dict[str, int] = {}
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        definitions = {
            (node.lineno, node.name)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # Aliased imports: local name -> original name.
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for item in node.names:
                    if item.asname:
                        aliases[item.asname] = item.name
            elif isinstance(node, ast.Import):
                for item in node.names:
                    if item.asname:
                        aliases[item.asname] = item.name.split(".")[-1]

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                key = (getattr(node, "lineno", -1), node.id)
                if key in definitions:
                    continue
                target = aliases.get(node.id, node.id)
                counts[target] = counts.get(target, 0) + 1
            elif isinstance(node, ast.Attribute):
                target = aliases.get(node.attr, node.attr)
                counts[target] = counts.get(target, 0) + 1
    return counts


#: Where call sites may live, relative to the package root. The MCP server is a
#: first-class front-end over this library, and it lives in ``tools/``; scanning only
#: ``src/`` reported every function that the MCP server calls as "never called". That is
#: a false positive of exactly the kind this checker exists to avoid.
EXTRA_CALL_ROOTS = ("tools",)


def analyse(root: Path, *, include_tests: bool = False) -> list[dict[str, object]]:
    definitions = _collect_definitions(root)
    sources = [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    # Call sites outside the package: the MCP server, which is a front-end, not a
    # consumer of the package through some other layer.
    # `root` is normally `src/cbb_stage1`; the project root is two levels up.
    for extra in EXTRA_CALL_ROOTS:
        for depth in (1, 0, 2):
            candidate = (root.parents[depth - 1] / extra) if depth else (root / extra)
            resolved = candidate.resolve()
            if resolved.is_dir():
                sources.extend(
                    p for p in resolved.rglob("*.py") if "__pycache__" not in p.parts
                )
                break
    counts = _collect_references(sources)

    findings: list[dict[str, object]] = []
    for info in definitions:
        if info.private:
            continue
        if info.qualname in EXEMPT:
            continue
        if counts.get(info.name, 0) == 0:
            findings.append(
                {
                    "module": info.module,
                    "function": info.name,
                    "lineno": info.lineno,
                    "qualname": info.qualname,
                }
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="package root to scan (default: src/cbb_stage1)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    findings = analyse(root)

    if args.json:
        print(json.dumps({"uncalled": findings, "count": len(findings)}, ensure_ascii=False, indent=2))
    else:
        print("=" * 78)
        print("CALL-LEVEL WIRING — public functions never referenced outside their module")
        print("=" * 78)
        if not findings:
            print("\nnone")
        else:
            print()
            for item in findings:
                print(f"  {item['module']:44s} {item['function']}()  line {item['lineno']}")
            print()
            print(
                "Each of these may be a real gap (written, never called) or a "
                "legitimate public API. Add it to EXEMPT with a reason if the latter."
            )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
