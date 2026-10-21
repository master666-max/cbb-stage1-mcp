#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""箱 v2 校验器：schema + 不变量。

用法：
  py -3 tools/box_lint.py [example]        # 默认 example
  py -3 tools/box_lint.py <box_dir>
  py -3 tools/box_lint.py --no-coverage-invariant <box_dir>
      # 关闭“有来源就必须有知识 / coverage 缺席率”两条表级不变量
      # （默认开启；也可用环境变量 CBB_BOX_LINT_DISABLE_COVERAGE_INVARIANT=1）

退出码：0 = 全过；1 = 有错（逐条打印）。
"""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

BASE = Path(__file__).resolve().parents[1]
SCHEMAS = BASE / "schemas"
TABLES = {
    "sources/units.jsonl": "unit.schema.json",
    "identity/mentions.jsonl": "mention.schema.json",
    "identity/entities.jsonl": "entity.schema.json",
    "identity/merges.jsonl": "merge.schema.json",
    "knowledge/assertions.jsonl": "assertion.schema.json",
    "knowledge/claims.jsonl": "claim.schema.json",
    "knowledge/citations.jsonl": "citation.schema.json",
    "knowledge/narrative.jsonl": "narrative.schema.json",
    "knowledge/graph_edges.jsonl": "graph-edge.schema.json",
    "knowledge/conflicts.jsonl": "conflict.schema.json",
    "sources/assets.jsonl": "asset.schema.json",
    "knowledge/events.jsonl": "event.schema.json",
    "knowledge/timeline.jsonl": "timeline.schema.json",
    "knowledge/cues.jsonl": "cue.schema.json",
    "derived/hypotheses.jsonl": "hypothesis.schema.json",
    "derived/research.jsonl": "research.schema.json",
    "audit/runs.jsonl": "run.schema.json",
}

#: Table-level coverage invariants (the "empty box is not green" line).
#:
#: Every check above is a *per-row* check: with zero rows it is vacuously true,
#: so a box with sources but no knowledge printed 7/7 pass.  These two guards
#: are intentionally table-level.
#:
#: ``COVERAGE_ABSENT_RATIO_LIMIT`` is the fraction of ``qc/coverage.json`` cells
#: allowed to be ``absent``.  It is 0.8 rather than 0.5 on measured evidence:
#:
#: * the project's fully-populated rich fixture is 35/60 = 0.583 absent;
#: * a real pipeline-fixture box (``tests/integration/test_box_pipeline.py``)
#:   is 44/60 = 0.733 absent — a 0.7 threshold false-killed it;
#: * the empty real box is 51/60 = 0.850 absent.
#:
#: 0.8 keeps both compliant boxes and still fails the empty one with margin.
COVERAGE_ABSENT_RATIO_LIMIT = 0.8

#: Set to ``1`` / ``true`` in the environment (or pass
#: ``--no-coverage-invariant``) to disable the table-level invariants for the
#: example fixture / offline smoke.  Default is ON.
COVERAGE_INVARIANT_ENV = "CBB_BOX_LINT_DISABLE_COVERAGE_INVARIANT"


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _coverage_invariant_disabled(argv):
    return "--no-coverage-invariant" in argv or _env_flag(COVERAGE_INVARIANT_ENV)


def _top_quarantine_reason(reasons):
    if not reasons:
        return "audit/quarantine.jsonl 无行（无最高频 reason）"
    reason, count = reasons.most_common(1)[0]
    return f"audit/quarantine.jsonl 最高频 reason = {reason} (×{count})"


def sha(s):
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_b(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def load_schema(name):
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{i} JSON 解析失败: {e}") from e
    return out


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    positional = [arg for arg in raw_args if not arg.startswith("-")]
    arg = positional[0] if positional else "example"
    box = (BASE / arg) if not Path(arg).is_absolute() else Path(arg)
    if not box.exists():
        print(f"箱不存在：{box}")
        return 1
    errors = []
    warnings = []

    def err(msg):
        errors.append(msg)

    def warn(msg):
        warnings.append(msg)

    # ---------- schema 校验 ----------
    manifest = read_json(box / "manifest.json")
    v = Draft202012Validator(load_schema("box.schema.json"))
    for e in sorted(v.iter_errors(manifest), key=lambda e: list(e.path)):
        err(f"manifest: {'/'.join(map(str, e.path))}: {e.message}")
    rows = {}
    for rel, schema_name in TABLES.items():
        p = box / rel
        try:
            data = read_jsonl(p)
        except ValueError as e:
            err(str(e))
            data = []
        rows[rel] = data
        sv = Draft202012Validator(load_schema(schema_name))
        for i, row in enumerate(data, 1):
            for e in sorted(sv.iter_errors(row), key=lambda e: list(e.path)):
                err(f"{rel}:{i}: {'/'.join(map(str, e.path))}: {e.message}")

    # ---------- 基础表 ----------
    units = {u["unit_id"]: u for u in rows["sources/units.jsonl"]}
    texts = {}
    for uid, u in units.items():
        tp = box / u["text_path"]
        if not tp.exists():
            err(f"unit {uid}: text_path 不存在 {u['text_path']}")
            continue
        t = tp.read_text(encoding="utf-8")
        texts[uid] = t
        if len(t) != u["chars"]:
            err(f"unit {uid}: chars {u['chars']} != 实际 {len(t)}")
        if sha_b(tp.read_bytes()) != u["sha256"]:
            err(f"unit {uid}: sha256 与 text 文件不符")
    # 顺序与 prev/next
    ordered = sorted(units.values(), key=lambda x: x["order"])
    for i, u in enumerate(ordered):
        prev = ordered[i - 1]["unit_id"] if i > 0 else None
        nxt = ordered[i + 1]["unit_id"] if i + 1 < len(ordered) else None
        if u.get("prev") != prev:
            err(f"unit {u['unit_id']}: prev 应为 {prev}")
        if u.get("next") != nxt:
            err(f"unit {u['unit_id']}: next 应为 {nxt}")

    entities = {e["entity_id"]: e for e in rows["identity/entities.jsonl"]}
    for e in entities.values():
        if e["status"] == "merged" and not e.get("merged_into"):
            err(f"entity {e['entity_id']}: merged 但无 merged_into")
        if e.get("merged_into") and e["merged_into"] not in entities:
            err(f"entity {e['entity_id']}: merged_into 不存在")
        for uf in ("first_unit", "last_unit"):
            if e.get(uf) and e[uf] not in units:
                err(f"entity {e['entity_id']}: {uf} 不存在 {e[uf]}")

    # lexicon
    lexicon = read_json(box / "lexicon" / "predicates.json")
    preds = {p["pred"]: p for p in lexicon["predicates"]}
    types = read_json(box / "lexicon" / "types.json")["types"]

    # ---------- mentions ----------
    for m in rows["identity/mentions.jsonl"]:
        uid = m["unit_id"]
        if uid not in texts:
            err(f"mention {m['mention_id']}: unit 不存在")
            continue
        t = texts[uid]
        if not (0 <= m["start"] < m["end"] <= len(t)):
            err(f"mention {m['mention_id']}: span 越界")
        elif t[m["start"]:m["end"]] != m["surface"]:
            err(f"mention {m['mention_id']}: 原文不符 {t[m['start']:m['end']]!r} != {m['surface']!r}")
        if m.get("sentence_span") and m.get("sentence") is not None:
            s, e = m["sentence_span"]
            if t[s:e] != m["sentence"]:
                err(f"mention {m['mention_id']}: sentence 不符")
        if m.get("entity_id") and m["entity_id"] not in entities:
            err(f"mention {m['mention_id']}: entity 不存在")

    # ---------- assertions ----------
    def check_evidence(evs, label, require=True):
        if require and not evs:
            err(f"{label}: asserted/required evidence 为空")
        for ev in evs or []:
            uid = ev["unit_id"]
            if uid not in texts:
                err(f"{label}: evidence unit 不存在 {uid}")
                continue
            t = texts[uid]
            s, e = ev["start"], ev["end"]
            if not (0 <= s < e <= len(t)):
                err(f"{label}: evidence span 越界")
            elif t[s:e] != ev["quote"]:
                err(f"{label}: quote 不在原文 span：{t[s:e]!r} != {ev['quote']!r}")
            if sha(ev["quote"]) != ev["quote_sha256"]:
                err(f"{label}: quote_sha256 不符")

    seen_a = set()
    for a in rows["knowledge/assertions.jsonl"]:
        aid = a["assertion_id"]
        if aid in seen_a:
            err(f"assertion {aid}: 重复")
        seen_a.add(aid)
        p = preds.get(a["predicate"])
        if not p:
            err(f"assertion {aid}: 谓词未注册 {a['predicate']}")
        else:
            if a["library"] != p["library"]:
                err(f"assertion {aid}: library {a['library']} != 注册表 {p['library']}")
            declared = {r["name"]: r for r in p["roles"]}
            for name, spec in declared.items():
                if spec.get("required") and name not in a["roles"]:
                    err(f"assertion {aid}: 缺必填角色 {name}")
            for name, val in a["roles"].items():
                if name not in declared:
                    err(f"assertion {aid}: 未声明角色 {name}")
                    continue
                spec = declared[name]
                vals = val if isinstance(val, list) else [val]
                for v in vals:
                    if spec["type"] == "entity":
                        if not (isinstance(v, str) and v.startswith("e:")):
                            err(f"assertion {aid}: 角色 {name} 应为实体 id")
                        elif v not in entities:
                            err(f"assertion {aid}: 角色 {name} 实体不存在 {v}")
                    else:
                        if not (isinstance(v, dict) and "literal" in v):
                            err(f"assertion {aid}: 角色 {name} 应为 literal")
        check_evidence(a.get("evidence", []), f"assertion {aid}", require=(a["status"] == "asserted"))
        if a.get("claim_layer") == "statement" and not a.get("speaker"):
            err(f"assertion {aid}: statement 缺 speaker")
        if a.get("confidence", 0) and not a.get("confidence_method"):
            err(f"assertion {aid}: 有 confidence 但缺 confidence_method")
        if a["status"] == "inferred" and not (a.get("derived_from") or a.get("evidence")):
            err(f"assertion {aid}: inferred 但既无 derived_from 也无 evidence")
        if a.get("superseded_by"):
            if a["superseded_by"] not in {x["assertion_id"] for x in rows["knowledge/assertions.jsonl"]}:
                err(f"assertion {aid}: superseded_by 不存在")
        if a["status"] == "superseded" and not a.get("superseded_by"):
            err(f"assertion {aid}: superseded 但无 superseded_by")

    # ---------- claims（v2.2） ----------
    assertion_ids = {a["assertion_id"]: a for a in rows["knowledge/assertions.jsonl"]}
    for c in rows["knowledge/claims.jsonl"]:
        cid = c["claim_id"]
        st = assertion_ids.get(c["statement_assertion_id"])
        if not st:
            err(f"claim {cid}: statement_assertion_id 不存在")
        elif st.get("claim_layer", "fact") != "statement":
            err(f"claim {cid}: 指向的不是 statement 断言")
        stances = []
        for q in c.get("questions", []):
            stances.append(q["stance"])
            for ans in q.get("answers", []):
                check_evidence([ans], f"claim {cid}/{q['qid']}")
        has_support = "support" in stances
        has_refute = "refute" in stances
        expect = ("conflicting" if (has_support and has_refute) else
                  "supported" if has_support else
                  "refuted" if has_refute else "insufficient_evidence")
        if c["verdict"] != expect:
            err(f"claim {cid}: verdict {c['verdict']} 不符合四值规则，应为 {expect}")
        if c["verdict"] == "insufficient_evidence" and st and st.get("claim_layer") != "statement":
            err(f"claim {cid}: insufficient 不得升 fact")
        for af in c.get("atomic_facts", []):
            if af["verdict"] == "false" and c["verdict"] == "supported":
                err(f"claim {cid}: atomic false 但 verdict supported")

    # ---------- hypotheses（derived） ----------
    for h in rows["derived/hypotheses.jsonl"]:
        if h["evidence_sufficiency"] == "sufficient" and h.get("missing_premises"):
            err(f"hypothesis {h['hypothesis_id']}: sufficient 不应有 missing_premises")
        if h["evidence_sufficiency"] == "insufficient" and not h.get("missing_premises"):
            warn(f"hypothesis {h['hypothesis_id']}: insufficient 但无 missing_premises")

    # ---------- projection-facing tables（v2.2 addendum） ----------
    citations = {}
    for c in rows["knowledge/citations.jsonl"]:
        cid = c["citation_id"]
        if cid in citations:
            err(f"citation {cid}: 重复")
        citations[cid] = c
        check_evidence([c], f"citation {cid}")
        uid = c["unit_id"]
        if uid in units and c["source_hash"] != units[uid]["sha256"]:
            err(f"citation {cid}: source_hash 与 unit sha256 不符")

    for seg in rows["knowledge/narrative.jsonl"]:
        sid = seg["seg_id"]
        if seg["anchor"] not in citations:
            err(f"narrative {sid}: anchor 不存在")
        if seg["content_hash"] != sha(f"{seg['type']}\x1f{seg['text']}"):
            err(f"narrative {sid}: content_hash 不符")

    for edge in rows["knowledge/graph_edges.jsonl"]:
        eid = edge["edge_id"]
        if edge["from"] not in entities or edge["to"] not in entities:
            err(f"graph edge {eid}: 端点不是实体")
        if edge["assertion_id"] not in assertion_ids:
            err(f"graph edge {eid}: assertion 不存在")

    for conflict in rows["knowledge/conflicts.jsonl"]:
        for side in ("left", "right"):
            if conflict[side] not in assertion_ids:
                err(f"conflict {conflict['conflict_id']}: {side} 不存在")

    coverage_path = box / "qc" / "coverage.json"
    if coverage_path.exists():
        coverage = read_json(coverage_path)
        cells = coverage.get("cells", [])
        if len(cells) != 60:
            err(f"coverage: cells {len(cells)} != 60")
        for cell in cells:
            if cell.get("status") not in ("present", "partial", "absent"):
                err(f"coverage: 非法 status {cell.get('status')}")
            if cell.get("status") == "absent" and not cell.get("reason"):
                err("coverage: absent 格缺 reason")

    # ---------- events ----------
    event_ids = set()
    for ev in rows["knowledge/events.jsonl"]:
        eid = ev["event_id"]
        if eid in event_ids:
            err(f"event {eid}: 重复")
        event_ids.add(eid)
        for pid in ev["participants"]:
            if pid not in entities:
                err(f"event {eid}: participant 不存在 {pid}")
        if ev.get("location") and ev["location"] not in entities:
            err(f"event {eid}: location 不存在")
        if ev["unit_id"] not in units:
            err(f"event {eid}: unit 不存在")
        check_evidence(ev["evidence"], f"event {eid}")
        anchor = (ev.get("time") or {}).get("anchor_unit")
        if anchor and anchor not in units:
            err(f"event {eid}: anchor_unit 不存在")

    # ---------- timeline ----------
    def node_exists(n):
        if n["kind"] == "event":
            return n["id"] in event_ids
        if n["kind"] == "unit":
            return n["id"] in units
        return True

    edges = {}
    for c in rows["knowledge/timeline.jsonl"]:
        for side in ("left", "right"):
            if not node_exists(c[side]):
                err(f"timeline {c['constraint_id']}: {side} 不存在 {c[side]}")
        if c["kind"] in ("before", "after"):
            l, r = c["left"]["id"], c["right"]["id"]
            if c["kind"] == "after":
                l, r = r, l
            edges.setdefault(l, set()).add(r)
        verify = c.get("verify")
        if verify:
            if verify.get("method") not in ("nli_pairwise", "explicit_expression", "topological"):
                err(f"timeline {c['constraint_id']}: verify.method 非法")
            if verify.get("result") not in ("yes", "no", "unknown"):
                err(f"timeline {c['constraint_id']}: verify.result 非法")
    # 环检测
    color = {}

    def dfs(n, stack):
        color[n] = 1
        for m in edges.get(n, ()):
            if color.get(m) == 1:
                err(f"timeline: 检测到环 {' -> '.join(stack + [n, m])}")
                return True
            if color.get(m, 0) == 0 and dfs(m, stack + [n]):
                return True
        color[n] = 2
        return False

    for n in list(edges):
        if color.get(n, 0) == 0:
            dfs(n, [])

    # ---------- cues ----------
    for c in rows["knowledge/cues.jsonl"]:
        su = c["setup"]["unit_id"]
        if su not in units:
            err(f"cue {c['cue_id']}: setup unit 不存在")
        check_evidence(c["setup"]["evidence"], f"cue {c['cue_id']}")
        po = c["payoff"]
        if po.get("event_id") and po["event_id"] not in event_ids:
            err(f"cue {c['cue_id']}: payoff event 不存在")
        if po.get("unit_id") and po["unit_id"] not in units:
            err(f"cue {c['cue_id']}: payoff unit 不存在")
        status = c.get("status")
        if status in ("triggered", "paid") and not (c.get("trigger") or {}).get("condition"):
            err(f"cue {c['cue_id']}: {status} 但 trigger 无 condition")
        if status == "paid":
            if not c["payoff"].get("evidence"):
                err(f"cue {c['cue_id']}: paid 但 payoff evidence 为空")
            if c.get("triggered_at") is None and (c.get("trigger") or {}).get("condition"):
                err(f"cue {c['cue_id']}: paid 前未 triggered")
        if status == "falsified" and not c.get("falsified_reason"):
            err(f"cue {c['cue_id']}: falsified 无 reason")

    # ---------- merges ----------
    for mg in rows["identity/merges.jsonl"]:
        for f in mg["from"]:
            if f not in entities:
                err(f"merge {mg['merge_id']}: from 不存在 {f}")
        if mg["to"] not in entities:
            err(f"merge {mg['merge_id']}: to 不存在")
        if not mg.get("reversible"):
            err(f"merge {mg['merge_id']}: reversible 必须 true")

    # ---------- 谓词提案 ----------
    proposed = {p["predicate"] for p in read_jsonl(box / "knowledge/proposed_predicates.jsonl")}
    for a in rows["knowledge/assertions.jsonl"]:
        if a["predicate"] in proposed:
            err(f"assertion {a['assertion_id']}: 谓词 {a['predicate']} 仍是提案，不得进正式断言")

    # ---------- engine outputs（契约样例） ----------
    eo_schema = Draft202012Validator(load_schema("engine-output.schema.json"))
    for p in sorted((box / "audit" / "engine").glob("*.json")):
        try:
            eo = read_json(p)
        except ValueError as e:
            err(f"engine {p.name}: {e}")
            continue
        for e in sorted(eo_schema.iter_errors(eo), key=lambda e: list(e.path)):
            err(f"engine {p.name}: {'/'.join(map(str, e.path))}: {e.message}")
        if eo.get("unit_id") not in units:
            err(f"engine {p.name}: unit_id 不存在")

    # ---------- counts ----------
    counts = manifest["counts"]
    actual = {
        "units": len(units), "mentions": len(rows["identity/mentions.jsonl"]),
        "entities": len(entities), "assertions": len(rows["knowledge/assertions.jsonl"]),
        "events": len(event_ids), "cues": len(rows["knowledge/cues.jsonl"]),
        "quarantined": len(read_jsonl(box / "audit/quarantine.jsonl")),
    }
    for k, v in actual.items():
        if counts.get(k) != v:
            err(f"counts.{k}: manifest {counts.get(k)} != 实际 {v}")
    optional_counts = {
        "statements": sum(1 for a in rows["knowledge/assertions.jsonl"] if a.get("claim_layer") == "statement"),
        "claims": len(rows["knowledge/claims.jsonl"]),
        "citations": len(rows["knowledge/citations.jsonl"]),
        "narratives": len(rows["knowledge/narrative.jsonl"]),
        "graph_edges": len(rows["knowledge/graph_edges.jsonl"]),
        "assets": len(rows["sources/assets.jsonl"]),
        "conflicts": len(rows["knowledge/conflicts.jsonl"]),
        "superseded": sum(1 for a in rows["knowledge/assertions.jsonl"] if a.get("status") == "superseded"),
    }
    for k, v in optional_counts.items():
        if k in counts and counts[k] != v:
            err(f"counts.{k}: manifest {counts[k]} != 实际 {v}")

    # ---------- checksums ----------
    checks = read_json(box / "CHECKSUMS.json") if (box / "CHECKSUMS.json").exists() else {"files": {}}
    if not checks.get("files"):
        warn("CHECKSUMS.json 缺失或为空")
    for rel, h in checks.get("files", {}).items():
        p = box / rel
        if not p.exists():
            err(f"checksum: 文件缺失 {rel}")
        elif sha_b(p.read_bytes()) != h:
            err(f"checksum: {rel} 不符")
    if checks.get("files") != manifest["hashes"]["files"]:
        err("CHECKSUMS.files 与 manifest.hashes.files 不一致")
    merkle = sha("\n".join(f"{k}:{v}" for k, v in sorted(checks.get("files", {}).items())))
    if merkle != manifest["hashes"]["merkle_root"]:
        err("merkle_root 不符")

    # ---------- qc ----------
    qc = read_json(box / "qc" / "report.json")
    if any(v != "pass" for v in qc.get("hard_gates", {}).values()):
        err("qc.hard_gates 存在非 pass")
    if manifest["status"] == "built" and qc.get("hard_gates", {}).get("hashes") != "pass":
        err("status=built 但 hashes 硬门未过")

    # ---------- coverage invariants（空表不得空真通过） ----------
    # Kept as a separate section on purpose: the per-row checks above stay
    # untouched, these guards only add the table-level non-vacuity condition.
    if _coverage_invariant_disabled(raw_args):
        print(
            "coverage invariant DISABLED "
            f"({COVERAGE_INVARIANT_ENV}=1 / --no-coverage-invariant): "
            "空表不再被判定为失败，仅限 example / 离线冒烟使用"
        )
    else:
        unit_count = len(units)
        if unit_count:
            quarantine_reasons = Counter(
                str(row.get("reason") or "<missing>")
                for row in read_jsonl(box / "audit/quarantine.jsonl")
                if isinstance(row, dict)
            )
            top_reason = _top_quarantine_reason(quarantine_reasons)
            # "不得为 0 行" is enforced on the knowledge layer as a whole: a
            # real pipeline box may legitimately emit assertions without events
            # (see tests/integration/test_box_pipeline.py). What must never pass
            # is a box with *no knowledge at all*.
            if not rows["knowledge/events.jsonl"] and not rows["knowledge/assertions.jsonl"]:
                err(
                    "coverage invariant: sources/units.jsonl 有 "
                    f"{unit_count} 个 unit，但 knowledge/events.jsonl 与 "
                    "knowledge/assertions.jsonl 都是 0 行 —— 有来源就必须抽出"
                    f"知识，空表不能让逐行检查空真通过；{top_reason}"
                )

        coverage_path = box / "qc" / "coverage.json"
        if coverage_path.exists():
            coverage = read_json(coverage_path)
            cells = [cell for cell in coverage.get("cells", []) if isinstance(cell, dict)]
            absent = [cell for cell in cells if cell.get("status") == "absent"]
            if cells:
                ratio = len(absent) / len(cells)
                if ratio > COVERAGE_ABSENT_RATIO_LIMIT:
                    reasons = Counter(
                        str(cell.get("reason") or "<missing>") for cell in absent
                    )
                    main_reasons = ", ".join(
                        f"{reason} ×{count}"
                        for reason, count in reasons.most_common(3)
                    ) or "无 reason"
                    err(
                        "coverage invariant: qc/coverage.json 缺席 "
                        f"{len(absent)}/{len(cells)} 格（{ratio:.3f} > 阈值 "
                        f"{COVERAGE_ABSENT_RATIO_LIMIT:.2f}，常量 "
                        "COVERAGE_ABSENT_RATIO_LIMIT）——覆盖度不足以判 PASS；"
                        f"主要 reason: {main_reasons}"
                    )

            # ---- per-predicate reconciliation（部分掏空防线） ----
            # The two guards above only catch a *total* loss. A box that keeps
            # 1 of 9 assertions has a non-empty assertions table and a low absent
            # ratio, so it would ship green — that is exactly the "partial
            # hollowing" shape the external audit flagged. Comparing emitted vs
            # accepted per predicate makes the shape of the loss visible.
            reconciliation = coverage.get("predicate_reconciliation") or {}
            hollowed = {
                predicate: entry
                for predicate, entry in reconciliation.items()
                if isinstance(entry, dict)
                and entry.get("emitted", 0) > 0
                and entry.get("accepted", 0) == 0
            }
            if hollowed:
                detail = ", ".join(
                    f"{predicate}（发射 {entry.get('emitted')}，入箱 0）"
                    for predicate, entry in sorted(hollowed.items())
                )
                err(
                    "coverage invariant: 以下谓词整类被隔离、箱里一条都没有 —— "
                    "逐行检查与表级非空门都看不出来，必须按谓词对账；"
                    f"{detail}"
                )

    # ---------- 汇总 ----------
    print(f"箱：{box}")
    print(f"表：units={actual['units']} mentions={actual['mentions']} entities={actual['entities']} "
          f"assertions={actual['assertions']} events={actual['events']} cues={actual['cues']}")
    for w in warnings:
        print(f"WARN  {w}")
    if errors:
        print(f"\nFAIL：{len(errors)} 个错误")
        for e in errors[:80]:
            print("  [x]", e)
        if len(errors) > 80:
            print(f"  ... 还有 {len(errors) - 80} 条")
        return 1
    print("\nPASS：schema + 不变量全过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
