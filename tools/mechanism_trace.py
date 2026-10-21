"""Trace each mechanism from the tree into an actual run.

Two separate questions, two separate evidence sources
-----------------------------------------------------
The coverage audit (`mechanism_audit.py`) answers *does the code exist* with a regex. Its
probes are broad (`def `), so it reported 81/81 PRESENT with zero gaps on a tree that also
held mechanisms nothing had ever called. Presence is necessary and nowhere near sufficient.

This tool answers the other question: **did the mechanism reach the flow.** It reads a
finished run — the box and the work directory — and grades each item on four levels:

    T3  produced output     a box table has rows, or a receipt counter is above zero
    T2  reachable call site  a call site exists
    T1  definition only
    T0  nothing observable

Why ``> 0`` and not ``>= 0``
----------------------------
A receipt counter that exists and reads zero means *wired but did not fire in this run*.
That is a third state, distinct from success and from absence, and collapsing it into
either one is how a green board gets built over an untested mechanism. `cove_revised == 0`
says CoVe was reachable and produced no revision — not that CoVe works, and not that it is
missing.

Reading the evidence
--------------------
* **Box tables** — `box/<path>` line counts. A table that exists but is empty is not
  production.
* **Receipt counters** — `work/receipts/*.json` each carry `counts`; keys are summed across
  receipts, because a mechanism's work can be split over stages.
* A malformed receipt or an absent directory is skipped, never fatal: an audit that dies on
  one bad file reports nothing about the other eighty-nine.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mechanism_audit

__all__ = [
    "LEVELS",
    "POINTERS",
    "PROXY_NOTES",
    "Trace",
    "main",
    "read_box_tables",
    "read_receipt_counters",
    "trace",
]

PROJECT = Path(__file__).resolve().parents[1]
LEVELS = ("T0", "T1", "T2", "T3")

#: Pointer kinds, weakest to strongest by the level each can best establish.
#:   ("module",  <project-relative path or glob>)   -> T1
#:   ("callsite", <relative path>, <regex>)         -> T2
#:   ("table",   <box-relative path>)               -> T3 when rows > 0
#:   ("counter", <receipt counts key>)              -> T3 when sum > 0
POINTERS: dict[str, tuple[tuple[str, ...], ...]] = {
    # ── A 抽取管线 ────────────────────────────────────────────────────────
    "A1": (("counter", "prescan_entities"), ("callsite", "src/cbb_stage1/orchestrator/pipeline.py", r"_prescan_for_extraction")),
    "A2": (("module", "src/cbb_stage1/engines/ai_reader/schema.py"),),
    "A3": (("module", "src/cbb_stage1/engines/ai_reader/schema.py"),),
    "A4": (("module", "src/cbb_stage1/engines/ai_reader/prompts.py"),),
    "A5": (("module", "src/cbb_stage1/orchestrator/checkpoint.py"),),
    "A6": (("counter", "merges"), ("counter", "merged_entities")),
    "A7": (("counter", "conflicts"), ("module", "src/cbb_stage1/fact_validator.py")),
    "A8": (("module", "src/cbb_stage1/engines/lpmm/adapter.py"),),
    "A9": (("module", "src/cbb_stage1/engines/ai_reader/openie.py"),),
    "A10": (("module", "src/cbb_stage1/incremental/"),),
    "A11": (("module", "src/cbb_stage1/ingest/chunk.py"),),
    "A12": (("counter", "duplicates"), ("module", "src/cbb_stage1/dedupe/")),
    "A13": (("module", "src/cbb_stage1/ingest/chunk.py"),),
    "A14": (("module", "src/cbb_stage1/dedupe/"),),
    "A15": (("table", "sources/units.jsonl"),),
    "A16": (("module", "src/cbb_stage1/orchestrator/idempotency.py"),),
    "A17": (("module", "src/cbb_stage1/engines/ai_reader/"),),
    "A18": (("table", "identity/entities.jsonl"),),
    "A19": (("table", "derived/community_reports.jsonl"),),
    "A20": (("module", "src/cbb_stage1/incremental/"),),
    "A21": (("module", "src/cbb_stage1/engines/ai_reader/"),),
    "A22": (("module", "src/cbb_stage1/orchestrator/"),),
    "A23": (("module", "src/cbb_stage1/ingest/chunk.py"),),
    "A24": (("module", "src/cbb_stage1/dedupe/"),),
    "A25": (("table", "derived/raptor.jsonl"),),
    "A26": (("module", "src/cbb_stage1/engines/"),),
    "A27": (("table", "knowledge/assertions.jsonl"),),
    # ── B 验证/置信/推断 ───────────────────────────────────────────────────
    "B1": (("counter", "cove_revised"), ("callsite", "src/cbb_stage1/claims/integration.py", r"cove_verify")),
    "B2": (("counter", "confidence_computed"), ("counter", "semantic_entropy"), ("module", "src/cbb_stage1/confidence/semantic_entropy.py")),
    "B3": (("module", "src/cbb_stage1/confidence/sep.py"),),
    "B4": (("module", "src/cbb_stage1/confidence/uqlm.py"),),
    "B5": (("module", "src/cbb_stage1/confidence/uq_heads.py"),),
    "B6": (("counter", "verdicts"), ("counter", "evidence_sufficiency"), ("callsite", "src/cbb_stage1/claims/integration.py", r"evidence_sufficiency")),
    "B7": (("counter", "afev"), ("counter", "atomic_facts"), ("callsite", "src/cbb_stage1/claims/integration.py", r"afev_verify")),
    "B8": (("table", "derived/hypotheses.jsonl"),),
    "B9": (("counter", "vegraph"), ("callsite", "src/cbb_stage1/claims/integration.py", r"vegraph_verify")),
    "B10": (("table", "knowledge/citations.jsonl"),),
    "B11": (("counter", "kappa"), ("callsite", "tools/mcp_pipeline_server.py", r"cbb_annotator_agreement")),
    "B12": (("counter", "abstain"), ("counter", "abstained"), ("callsite", "src/cbb_stage1/claims/integration.py", r"abstain_threshold")),
    "B13": (("counter", "speaker_verified"), ("counter", "quote_verdicts"), ("callsite", "src/cbb_stage1/validate/integration.py", r"verify_speaker_quote")),
    # ── C 执行模型 ─────────────────────────────────────────────────────────
    "C1": (("module", "src/cbb_stage1/agents/"),),
    "C2": (("counter", "outputs"),),
    "C3": (("module", "src/cbb_stage1/agents/subagent.py"),),
    "C4": (("module", "src/cbb_stage1/agents/"),),
    "C5": (("table", "derived/summaries.jsonl"),),
    "C6": (("module", "src/cbb_stage1/derive/"),),
    "C7": (("module", "src/cbb_stage1/orchestrator/"),),
    "C8": (("module", "src/cbb_stage1/model_gateway.py"),),
    "C9": (("module", "src/cbb_stage1/box/atomic.py"),),
    "C10": (("module", "src/cbb_stage1/preflight.py"),),
    "C11": (("module", "tools/gov_console.py"),),
    "C12": (("module", "tools/mcp_pipeline_server.py"),),
    # ── D 闭环/增量/叙述 ───────────────────────────────────────────────────
    "D1": (("table", "knowledge/narrative.jsonl"), ("counter", "claim_layer")),
    "D2": (("callsite", "src/cbb_stage1/assemble/assertions.py", r"narrator_reliability"),),
    "D3": (("module", "src/cbb_stage1/confidence/"),),
    "D4": (("module", "src/cbb_stage1/model_gateway.py"),),
    "D5": (("counter", "units_to_rescan"), ("module", "src/cbb_stage1/assemble/lexicon_evolution.py")),
    "D6": (("module", "src/cbb_stage1/incremental/"),),
    "D7": (("table", "knowledge/cues.jsonl"),),
    "D8": (("counter", "triggers"), ("table", "knowledge/cues.jsonl")),
    "D9": (("table", "audit/conflicts.jsonl"), ("table", "knowledge/conflicts.jsonl")),
    "D10": (("table", "knowledge/timeline.jsonl"),),
    "D11": (("counter", "nli"), ("table", "knowledge/timeline.jsonl")),
    "D12": (("counter", "clues"), ("table", "knowledge/cues.jsonl")),
    "D13": (("table", "derived/raptor.jsonl"),),
    "D14": (("table", "derived/inference/forecasts.jsonl"),),
    "D15": (
        ("table", "derived/inference/stylometry.jsonl"),
        ("table", "derived/inference/themes.jsonl"),
        ("table", "derived/inference/emotion_curves.jsonl"),
        ("table", "derived/inference/motivation_chains.jsonl"),
    ),
    # ── E 箱与契约 ─────────────────────────────────────────────────────────
    "E1": (("table", "knowledge/citations.jsonl"), ("counter", "evidence_verified")),
    "E2": (("counter", "candidates"),),
    "E3": (("table", "identity/merges.jsonl"),),
    "E4": (("counter", "cards"), ("module", "src/cbb_stage1/box/writer.py")),
    "E5": (("counter", "promoted"), ("counter", "verdicts")),
    "E6": (("module", "src/cbb_stage1/box/writer.py"),),
    "E7": (("module", "src/cbb_stage1/orchestrator/"),),
    "E8": (("module", "src/cbb_stage1/qc/acceptance.py"),),
    "E9": (("module", "src/cbb_stage1/box/writer.py"),),
    "E10": (("table", "identity/entities.jsonl"), ("table", "derived/entity_cards.jsonl")),
    "E11": (("table", "knowledge/claims.jsonl"), ("table", "knowledge/assertions.jsonl")),
    "E12": (("table", "audit/provenance.jsonl"), ("module", "src/cbb_stage1/ids.py")),
    # ── F 段二/未进箱（清单自带排除）──────────────────────────────────────
    "F1": (),
    "F2": (),
    "F3": (
        ("table", "derived/inference/stylometry.jsonl"),
        ("table", "derived/inference/themes.jsonl"),
        ("table", "derived/inference/emotion_curves.jsonl"),
        ("table", "derived/inference/motivation_chains.jsonl"),
        ("table", "derived/inference/forecasts.jsonl"),
    ),
    "F4": (("table", "sources/units.jsonl"),),
    "F5": (),
    "F6": (),
    "F7": (),
    "F8": (),
    "F9": (),
    "F10": (),
    "F11": (),
}


#: Items whose observable artefact is a **stand-in**, not the thing the inventory names.
#: Without this, a table with rows grades as unqualified production and a proxy is reported
#: as the original method -- the same false green the four-level scale exists to prevent.
#: Every item `mechanism_audit.LIMITED` names must appear here (enforced by a test).
PROXY_NOTES: dict[str, str] = {
    "D14": "⚠ 代理：自有实现是 lexical_singleton_ratio（词法离散度），非 100-Endings 的 no_rate",
    "B3": "⚠ 白盒受限：需 logprobs，host 传输不提供",
    "B4": "⚠ 白盒受限：黑盒分支可用，白盒分支拿不到信号",
    "B5": "⚠ 需 checkpoint：主模型随 agent 变，头绑死基座",
    "B11": "⚠ 需模型：无模型时 κ 显式 unavailable",
}


@dataclass(frozen=True)
class Trace:
    ident: str
    name: str
    present: str
    wired: str
    evidence: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.ident,
            "name": self.name,
            "present": self.present,
            "wired": self.wired,
            "evidence": self.evidence,
        }


# ── reading the run ──────────────────────────────────────────────────────
def read_box_tables(box: Path) -> dict[str, int]:
    """Box-relative table path -> row count. Absent box yields nothing, not an error."""
    counts: dict[str, int] = {}
    if not box or not box.is_dir():
        return counts
    for path in box.rglob("*.jsonl"):
        try:
            rel = path.relative_to(box).as_posix()
        except ValueError:
            continue
        rows = 0
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        rows += 1
        except OSError:
            continue
        counts[rel] = rows
    return counts


def read_receipt_counters(work: Path) -> dict[str, float]:
    """Sum every numeric ``counts`` value across ``work/receipts/*.json``."""
    totals: dict[str, float] = {}
    receipts = (work / "receipts") if work else None
    if not receipts or not receipts.is_dir():
        return totals
    for path in sorted(receipts.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # One unreadable receipt must not cost the audit its other 89 findings.
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in (payload.get("counts") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[str(key)] = totals.get(str(key), 0.0) + float(value)
    return totals


def _callsite_exists(rel: str, pattern: str) -> bool:
    import re

    path = PROJECT / rel
    if not path.is_file():
        return False
    try:
        return bool(re.search(pattern, path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def _module_exists(rel: str) -> bool:
    path = PROJECT / rel
    if path.is_file():
        return True
    if path.is_dir():
        return any(p.suffix == ".py" for p in path.rglob("*.py"))
    return bool(list(PROJECT.glob(rel)))


def _grade(
    pointers: Iterable[tuple[str, ...]],
    *,
    tables: dict[str, int],
    counters: dict[str, float],
) -> tuple[str, str]:
    """Highest level any pointer earns, with the pointer that earned it."""
    best = "T0"
    evidence = ""
    rank = {level: index for index, level in enumerate(LEVELS)}

    def better(level: str, note: str) -> None:
        nonlocal best, evidence
        if rank[level] > rank[best]:
            best, evidence = level, note

    for pointer in pointers:
        kind = pointer[0]
        if kind == "table" and len(pointer) >= 2:
            rel = pointer[1]
            rows = tables.get(rel, 0)
            if rows > 0:
                better("T3", f"box/{rel} rows={rows}")
            elif rel in tables:
                better("T2", f"box/{rel} exists but is empty")
        elif kind == "counter" and len(pointer) >= 2:
            key = pointer[1]
            value = counters.get(key)
            if value is not None and value > 0:
                better("T3", f"receipt counts.{key}={value:g}")
            elif value is not None:
                # Present and zero: wired, did not fire. Deliberately not T3.
                better("T2", f"receipt counts.{key}=0 (wired, did not fire)")
        elif kind == "callsite" and len(pointer) >= 3:
            if _callsite_exists(pointer[1], pointer[2]):
                better("T2", f"callsite {pointer[1]}")
        elif kind == "module" and len(pointer) >= 2:
            if _module_exists(pointer[1]):
                better("T1", f"definition {pointer[1]}")
    return best, evidence


def trace(*, work: Path, box: Path) -> list[Trace]:
    """Grade every inventory item against one finished run."""
    tables = read_box_tables(box)
    counters = read_receipt_counters(work)
    presence = {v.ident: v for v in mechanism_audit.audit()}

    results: list[Trace] = []
    for item in mechanism_audit.ITEMS:
        present = presence.get(item.ident)
        level, evidence = _grade(
            POINTERS.get(item.ident, ()), tables=tables, counters=counters
        )
        if item.excluded:
            # The inventory rules it out; there is nothing to wire.
            level, evidence = "T0", item.excluded
        note = PROXY_NOTES.get(item.ident, "")
        if note and level in {"T2", "T3"}:
            evidence = f"{evidence}  {note}" if evidence else note
        results.append(
            Trace(
                ident=item.ident,
                name=item.name,
                present=(present.status if present else "?"),
                wired=level,
                evidence=evidence,
            )
        )
    return results


def render(results: list[Trace]) -> str:
    wired = Counter(r.wired for r in results)
    present = Counter(r.present for r in results)
    lines = [
        "=" * 84,
        "MECHANISM TRACE — 在 (present) vs 接线 (wired), from a finished run",
        "=" * 84,
        "",
        f"  items        : {len(results)}",
        f"  present      : " + "  ".join(f"{k}={v}" for k, v in sorted(present.items())),
        f"  wired        : " + "  ".join(f"{k}={wired.get(k, 0)}" for k in LEVELS),
        "",
        "  T3 = produced output   T2 = reachable/gated   T1 = definition only   T0 = none",
        "",
    ]
    for level in ("T3", "T2", "T1", "T0"):
        group = [r for r in results if r.wired == level]
        if not group:
            continue
        lines.append(f"── {level} ({len(group)}) " + "-" * (78 - len(level) - 8))
        for r in group:
            lines.append(f"  {r.ident:5s} {r.name[:34]:34s} present={r.present:9s} {r.evidence[:52]}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade mechanisms against a finished run: present vs wired"
    )
    parser.add_argument("--work", required=True)
    parser.add_argument("--box", required=True)
    parser.add_argument("--json", action="store_true")
    ns = parser.parse_args(argv)

    results = trace(work=Path(ns.work), box=Path(ns.box))
    if ns.json:
        print(json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())