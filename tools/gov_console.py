"""gov_console.py — 抽取段网页治理控制台（人裁入口）。

形态沿用旧山一代的 ``cbb/tools/web_console.py``：**标准库零依赖**、``--dry-run`` 卫生口
（零绑定零监听）、``--help`` exit 0。形态可抄，纪律要加一条：

    旧件是「只读观察台」，本件是**人裁入口**。所以写面被分成两半——

    * **读面**读箱（``--box``）：治理面、待裁队列、简单查询。GET，不落盘。
    * **写面**走 MCP（``--work``）：所有裁决都调 ``cbb_*`` 工具，**本件自己不开写口**。

不开写口是刻意的：这个项目里谓词注册与改判已经有实现
（``assemble/lexicon_evolution.py::approve_proposal``、
``validate/conflicts.py::apply_supersede``），MCP 侧已暴露
``cbb_lexicon_approve``／``cbb_supersede``。控制台再写一遍就是第二条写路，
而第二条写路正是本项目反复吃过亏的地方。

**每次裁决必须带 ``by``**（人审身份）——I4 规定谓词不许从数据里自己长出来。
本件在调用 MCP 之前就校验它，不是之后。

只读的箱与可写的 work 是两个目录：裁决落在 work，箱是交付件，页面上会写明
「已裁决，需重建箱」。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

__all__ = [
    "CONFLICT_CHOICES",
    "QUARANTINE_DECISIONS",
    "INFERENCE_TABLES",
    "decide_conflict",
    "decide_predicate",
    "decide_quarantine",
    "decide_supersede",
    "main",
    "query",
    "read_coverage",
    "read_inference",
    "read_overview",
    "read_pending",
]

HERE = Path(__file__).resolve().parent

#: §5.20 的五个推断层文件。缺表要写「本轮为空」，不许假装有。
INFERENCE_TABLES = (
    "forecasts",
    "stylometry",
    "themes",
    "emotion_curves",
    "motivation_chains",
)

#: 冲突裁决的五个选项（``validate/conflicts.py`` 的 ``DECISIONS``）。
CONFLICT_CHOICES = ("left", "right", "merge", "disputed", "superseded")

#: 隔离区裁决。**语义待第一次真实裁决定**——本件只记决策、不删件。
QUARANTINE_DECISIONS = ("keep", "drop", "retain")

#: MCP 服务器名。可用环境变量覆盖（换服务名不必改代码）。
MCP_SERVER = os.environ.get("CBB_MCP_SERVER") or "cbb-pipeline"


# ── 小工具 ────────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


# ── 读面：全部只读，全部读箱 ─────────────────────────────────────────────────
def read_overview(box: str | Path) -> dict[str, Any]:
    """硬门/软门、能力缺口登记、工作台账问题。"""
    report = _read_json(Path(box) / "qc" / "report.json")
    return {
        "status": report.get("status"),
        "hard_gates": report.get("hard_gates") or {},
        "hard_gate_errors": report.get("hard_gate_errors") or {},
        "soft_gates": report.get("soft_gates") or {},
        "capability_gaps": report.get("capability_gaps") or {},
        "work_table_problems": report.get("work_table_problems") or [],
        "samples": report.get("samples") or {},
    }


def read_coverage(box: str | Path) -> dict[str, Any]:
    """6 库 × 10 方面 = 60 格，逐格带 status/count/reason（I32）。"""
    data = _read_json(Path(box) / "qc" / "coverage.json")
    return {
        "book_id": data.get("book_id"),
        "spec_version": data.get("spec_version"),
        "libraries": data.get("libraries") or [],
        "aspects": data.get("aspects") or [],
        "cells": data.get("cells") or [],
        "counts": data.get("counts") or {},
        "predicate_reconciliation": data.get("predicate_reconciliation") or {},
    }


def read_pending(
    box: str | Path,
    *,
    work: str | Path | None = None,
) -> dict[str, Any]:
    """待裁队列：谓词提案 / 冲突 / 隔离项。

    Two id spaces, and the quarantine queue has to cross between them
    ------------------------------------------------------------------
    Measured on a recorded run: the box's ``audit/quarantine.jsonl`` carries
    ``assertion:u:book:v01:c001`` — one row per problem, grouped by kind and unit —
    while the work tree's ``assemble/quarantine.jsonl`` carries record-level ids such
    as ``a:pending:fe9bfd87d164aeba``. They are different id spaces.

    The write path resolves against the work tree, so a queue that displayed the
    box's ids would show a human a list whose every button fails with "no such
    quarantine item". The quarantine rows therefore come from the work tree when one
    is given, and ``quarantine_source`` says which side was used. The other two
    queues read the box: their ids are the same on both sides.
    """
    root = Path(box)
    proposals = _read_jsonl(root / "knowledge" / "proposed_predicates.jsonl")
    conflicts = _read_jsonl(root / "knowledge" / "conflicts.jsonl")

    quarantine: list[dict[str, Any]] = []
    source = "box"
    if work is not None:
        work_quarantine = _read_jsonl(
            Path(work) / "assemble" / "quarantine.jsonl"
        )
        if work_quarantine:
            quarantine = work_quarantine
            source = "work"
    if not quarantine:
        quarantine = _read_jsonl(root / "audit" / "quarantine.jsonl")
        source = "box"

    return {
        "proposals": proposals,
        "conflicts": conflicts,
        "quarantine": quarantine,
        "quarantine_source": source,
    }


def read_assertions(
    box: str | Path,
    *,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    """断言列表（分页）。单条的证据在 rows 里，不另开一份。"""
    rows = _read_jsonl(Path(box) / "knowledge" / "assertions.jsonl")
    window = rows[offset: offset + limit]
    return {
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "rows": window,
        "empty": not rows,
    }


def read_inference(box: str | Path) -> dict[str, Any]:
    """推断层五表。缺表写「本轮为空」，不许假装有。"""
    root = Path(box) / "derived" / "inference"
    tables: dict[str, Any] = {}
    for name in INFERENCE_TABLES:
        rows = _read_jsonl(root / f"{name}.jsonl")
        tables[name] = {
            "rows": len(rows),
            "empty": not rows,
            "sample": rows[0] if rows else None,
        }
    return {"tables": tables, "total": sum(t["rows"] for t in tables.values())}


def query(
    box: str | Path,
    *,
    q: str = "",
    predicate: str = "",
    entity: str = "",
    unit: str = "",
) -> dict[str, Any]:
    """简单查询。优先走箱内 ``index/``（构建期已算好），不扫全表。"""
    root = Path(box)
    assertions = {
        row["assertion_id"]: row
        for row in _read_jsonl(root / "knowledge" / "assertions.jsonl")
        if row.get("assertion_id")
    }

    hits: list[dict[str, Any]] = []
    if predicate:
        ids = _read_json(root / "index" / "by_predicate.json").get(predicate) or []
        hits = [assertions[i] for i in ids if i in assertions]
    elif entity:
        ids = _read_json(root / "index" / "by_entity.json").get(entity) or []
        hits = [assertions[i] for i in ids if i in assertions]
    elif unit:
        ids = _read_json(root / "index" / "by_unit.json").get(unit) or []
        hits = [assertions[i] for i in ids if i in assertions]
    else:
        hits = list(assertions.values())

    needle = str(q or "").strip().lower()
    if needle:
        hits = [
            row
            for row in hits
            if needle in json.dumps(row, ensure_ascii=False).lower()
        ]
    return {
        "total": len(hits),
        "hits": hits,
        "query": {"q": q, "predicate": predicate, "entity": entity, "unit": unit},
    }


# ── 写面：唯一出口是 MCP ────────────────────────────────────────────────────
def _mcp_executable() -> str:
    """解析 MCP 客户端的可执行路径。

    ``mcporter`` 是 npm 的 shim——在 Windows 上是 ``mcporter.cmd``。
    ``subprocess.run(["mcporter", ...])`` **找不到它**：Windows 要的是解析后的全路径，
    裸名会抛 ``FileNotFoundError [WinError 2]``。

    实测：一次真实跑批里，每一次写操作都带着这个错误，而 MCP 工具本身是好的
    （直接调它返回的是「提案没有 library」这种正常业务回执）。所以这条不是
    锦上添花，是写面的全部。
    """
    override = os.environ.get("CBB_MCPORTER")
    if override:
        return override
    return shutil.which("mcporter") or shutil.which("mcporter.cmd") or "mcporter"


def _mcp_call(tool: str, **kwargs: Any) -> dict[str, Any]:
    """调一个 MCP 工具。**这是本件唯一的写侧出口**——测试也打桩这里。

    参数一律走 ``key=value``（mcporter 的约定），值里可能有中文与路径，
    所以用列表传参而不是拼字符串。
    """
    client = _mcp_executable()
    if not Path(client).exists():
        return {
            "ok": False,
            "error": (
                f"MCP 客户端不可用：{client} 不在位。"
                "装上它，或用 CBB_MCPORTER 指向可执行文件的全路径"
            ),
        }
    cmd = [
        client,
        "call",
        f"{MCP_SERVER}.{tool}",
        *[f"{key}={value}" for key, value in kwargs.items()],
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out = (proc.stdout or "").strip()
    try:
        payload = json.loads(out)
    except (json.JSONDecodeError, ValueError):
        payload = {
            "ok": proc.returncode == 0,
            "data": {"raw": out, "stderr": (proc.stderr or "")[-600:]},
        }
    if not isinstance(payload, dict):
        payload = {"ok": proc.returncode == 0, "data": payload}
    return payload


def _require_by(by: Any) -> str:
    """人审身份。空的不许发出去——这道门在调用之前，不在之后。"""
    who = str(by or "").strip()
    if not who:
        raise ValueError(
            "裁决必须带人审身份（by）：I4 规定谓词不许从数据里自己长出来，"
            "supersede 也是人裁"
        )
    return who


def decide_predicate(
    work: str | Path,
    *,
    predicate: str,
    by: str,
    library: str | None = None,
) -> dict[str, Any]:
    """批准一个未注册谓词（D5）。"""
    who = _require_by(by)
    args: dict[str, Any] = {
        "work": str(work),
        "predicate": str(predicate),
        "by": who,
    }
    if library:
        args["library"] = str(library)
    return _mcp_call("cbb_lexicon_approve", **args)


def decide_supersede(
    work: str | Path,
    *,
    loser: str,
    winner: str,
    by: str,
    reason: str,
) -> dict[str, Any]:
    """一条断言被另一条取代（retcon）。旧件保留，决策写 ``audit/decisions.jsonl``。"""
    who = _require_by(by)
    return _mcp_call(
        "cbb_supersede",
        work=str(work),
        loser=str(loser),
        winner=str(winner),
        by=who,
        reason=str(reason),
    )


def decide_conflict(
    work: str | Path,
    *,
    conflict_id: str,
    choice: str,
    by: str,
    reason: str,
) -> dict[str, Any]:
    """冲突裁决。五个选项一个都不许是死按钮。"""
    who = _require_by(by)
    picked = str(choice or "").strip()
    if picked not in CONFLICT_CHOICES:
        raise ValueError(
            f"冲突选项 {choice!r} 不在 {list(CONFLICT_CHOICES)} 里；"
            "非法选项不许先调后判"
        )
    return _mcp_call(
        "cbb_conflict_resolve",
        work=str(work),
        conflict_id=str(conflict_id),
        choice=picked,
        by=who,
        reason=str(reason),
    )


def decide_quarantine(
    work: str | Path,
    *,
    item_id: str,
    decision: str,
    by: str,
    reason: str,
) -> dict[str, Any]:
    """隔离项裁决。**只记决策，不删件**（与「旧件保留」同纪律）。"""
    who = _require_by(by)
    picked = str(decision or "").strip()
    if picked not in QUARANTINE_DECISIONS:
        raise ValueError(
            f"隔离裁决 {decision!r} 不在 {list(QUARANTINE_DECISIONS)} 里"
        )
    return _mcp_call(
        "cbb_quarantine_adjudicate",
        work=str(work),
        item_id=str(item_id),
        decision=picked,
        by=who,
        reason=str(reason),
    )


# ── HTTP ─────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    """只读 GET + 只调 MCP 的 POST。本件不直写箱、不直写 work。"""

    box = Path(".")
    work = Path(".")

    def _send(self, code: int, body: str, ctype: str = "application/json; charset=utf-8"):
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False))

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名
        try:
            parsed = urllib.parse.urlparse(self.path)
            query_pairs = dict(urllib.parse.parse_qsl(parsed.query))
            path = parsed.path

            if path == "/":
                html = HERE / "gov_console.html"
                if not html.exists():
                    return self._json(500, {"error": "gov_console.html 不在位"})
                return self._send(200, html.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            if path == "/api/overview":
                return self._json(200, read_overview(self.box))
            if path == "/api/coverage":
                return self._json(200, read_coverage(self.box))
            if path == "/api/pending":
                return self._json(200, read_pending(self.box, work=self.work))
            if path == "/api/inference":
                return self._json(200, read_inference(self.box))
            if path == "/api/assertions":
                return self._json(
                    200,
                    read_assertions(
                        self.box,
                        limit=int(query_pairs.get("limit") or 200),
                        offset=int(query_pairs.get("offset") or 0),
                    ),
                )
            if path == "/api/query":
                return self._json(
                    200,
                    query(
                        self.box,
                        q=query_pairs.get("q", ""),
                        predicate=query_pairs.get("predicate", ""),
                        entity=query_pairs.get("entity", ""),
                        unit=query_pairs.get("unit", ""),
                    ),
                )
            self._json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001 - 页面不许因一个端点崩掉整个控制台
            self._json(500, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return self._json(400, {"error": "请求体不是 JSON"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "请求体必须是对象"})

            by = body.get("by")
            path = parsed.path
            if path == "/api/decide/predicate":
                result = decide_predicate(
                    self.work,
                    predicate=body.get("predicate", ""),
                    by=by,
                    library=body.get("library"),
                )
            elif path == "/api/decide/supersede":
                result = decide_supersede(
                    self.work,
                    loser=body.get("loser", ""),
                    winner=body.get("winner", ""),
                    by=by,
                    reason=body.get("reason", ""),
                )
            elif path == "/api/decide/conflict":
                result = decide_conflict(
                    self.work,
                    conflict_id=body.get("conflict_id", ""),
                    choice=body.get("choice", ""),
                    by=by,
                    reason=body.get("reason", ""),
                )
            elif path == "/api/decide/quarantine":
                result = decide_quarantine(
                    self.work,
                    item_id=body.get("item_id", ""),
                    decision=body.get("decision", ""),
                    by=by,
                    reason=body.get("reason", ""),
                )
            else:
                return self._json(404, {"error": "not found"})
            return self._json(200, {"result": result, "needs_rebuild": True})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        sys.stderr.write("[gov_console] " + (fmt % args) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="抽取段网页治理控制台（人裁入口）——读面读箱，写面走 MCP 工具"
    )
    ap.add_argument("--box", required=True, help="箱目录（读面）")
    ap.add_argument("--work", default="", help="工作目录（写面：MCP 工具的 work 参数）")
    ap.add_argument("--port", type=int, default=8913, help="监听端口（默认 8913）")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印配置与数据面在位情况即退出，不绑端口不启服务",
    )
    ns = ap.parse_args(argv)

    box = Path(ns.box)
    work = Path(ns.work) if ns.work else box.parent / "work"

    if ns.dry_run:
        print(
            json.dumps(
                {
                    "gov_console": True,
                    "dry_run": True,
                    "port": ns.port,
                    "box": str(box),
                    "box_exists": box.exists(),
                    "work": str(work),
                    "work_exists": work.exists(),
                    "mcporter": _mcp_executable(),
                    "mcp_server": MCP_SERVER,
                    "纪律": "读面只读；写面只调 MCP 工具，本件不开写口",
                },
                ensure_ascii=False,
            )
        )
        return 0

    Handler.box = box
    Handler.work = work
    print(
        json.dumps(
            {
                "gov_console": True,
                "url": f"http://127.0.0.1:{ns.port}",
                "box": str(box),
                "work": str(work),
                "纪律": "读面只读；写面只调 MCP 工具",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        ThreadingHTTPServer(("127.0.0.1", ns.port), Handler).serve_forever()
    except OSError as exc:
        # 端口被占是常事，给一句能看懂的话，不甩栈。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"端口 {ns.port} 起不来：{type(exc).__name__}: {exc}",
                    "hint": "换一个 --port，或先停掉占用该端口的进程",
                },
                ensure_ascii=False,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
