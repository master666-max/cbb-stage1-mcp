"""Build the MCP install package: extract, register, done.

The package is deliberately shaped so the target device needs **no pip install and no
project paths**:

    cbb-mcp/
      server.py            <- the MCP server
      field_coverage.py    <- the two gate scripts it loads in-process
      call_wiring.py
      cbb_stage1/          <- the library, next to the server
      schemas/             <- the box schemas lint/field-coverage read
      mcp.json             <- the manifest the host reads
      README.md            <- three steps

Two defects this layout caused, both found by running the installed package rather than
the checkout, are fixed here and in `mcp_pipeline_server.py`:

1. **`schemas/` was never copied.** `build_publish.py` ships it; this builder did not, so
   `cbb_lint_box` and `cbb_field_coverage` had no schemas on a fresh install. The copy list
   is now :func:`package_plan` — explicit data, assertable without running a build.
2. **The server resolved its own root one level high.** `_TOOLS_DIR.parent` is correct at
   `<project>/tools/<file>` and is the package's *parent* at `<package>/server.py`, so
   `tools_dir`, `schema_dir` and `call_wiring --root` all pointed outside the package.
   `_find_project_root` walks markers instead.

`server.py`'s own directory goes on `sys.path` automatically (that is how Python starts
a script), so `import cbb_stage1` resolves without installation. The `_cli` helper that
used to spawn a child is gone, so there is no child process to give a PYTHONPATH to.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

# Derived from this file: the script lives in `<project>/tools/`, so parents[1] is
# the project. Pinning an absolute path here would also put it in the published
# tree, which the publish rescan rejects.
PROJECT = Path(__file__).resolve().parents[1]
DIST = PROJECT / "dist" / "cbb-mcp"

#: The box schemas live beside the design documents, not in the code tree. Located through
#: the library's own resolver so there is one place that knows the search order.
DESIGN = PROJECT.parent / "设计" / "抽取段-v6"


def package_plan(project: Path) -> list[tuple[Path, str]]:
    """``(source, destination)`` pairs the package is built from.

    Exposed as data rather than inlined in the copy statements, because the schemas were
    absent for exactly as long as this list was implicit. A test asserts the plan contains
    them and that every source exists.
    """
    from cbb_stage1.schemas import schemas_dir

    schemas = Path(schemas_dir(project))
    return [
        (project / "tools" / "field_coverage.py", "field_coverage.py"),
        (project / "tools" / "call_wiring.py", "call_wiring.py"),
        # `cbb_start_run` spawns the pipeline through this shim, resolved next to the
        # server. It was left out, so every run died with
        # `can't open file '<install>\_mcp_run_shim.py'` and `cbb_start_run` reported
        # "pipeline did not report its pid within 15.0s" — the same class of defect as the
        # missing schemas, and found the same way: by running the install.
        (project / "tools" / "_mcp_run_shim.py", "_mcp_run_shim.py"),
        # renamed on the way in: the host reads ``server.py``
        (project / "tools" / "mcp_pipeline_server.py", "server.py"),
        (project / "src" / "cbb_stage1", "cbb_stage1"),
        (schemas, "schemas"),
    ]


def build(project: Path = PROJECT, dist: Path = DIST) -> None:
    """Build the package. Side effects live here, never at import time."""
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)

    # ── 1. everything the package is made of ──────────────────────────────────
    for source, destination in package_plan(project):
        if not source.exists():
            raise SystemExit(f"package plan names a missing source: {source}")
        target = dist / destination
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        else:
            shutil.copy2(source, target)

    # ── 3. the manifest ───────────────────────────────────────────────────────
    manifest = {
        "mcpServers": {
            "cbb": {
                "command": "python",
                "args": ["-u", str(dist / "server.py")],
                "description": (
                    "CBB 抽取段 v6.1 主驱动：起跑/服务模型请求/建箱/跑门/人裁（21 个工具）"
                ),
            }
        }
    }
    (dist / "mcp.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ── 4. the readme ─────────────────────────────────────────────────────────
    README = """# cbb-mcp — 抽取段 MCP 包

    解压、注册、连上。**不需要 pip install，不需要知道任何项目路径。**

    ## 三步

    ### 1. 放到任意目录

    把整个 `cbb-mcp/` 解压到哪都行（`cbb_stage1/` 必须和 `server.py` 同目录）。

    ### 2. 注册到宿主的 MCP 配置

    把 `mcp.json` 里这段并进宿主的 MCP 配置（`mcpServers` 那一层）：

    ```json
    {
      "mcpServers": {
        "cbb": {
          "command": "python",
          "args": ["-u", "<解压目录>/server.py"]
        }
      }
    }
    ```

    `command` 用宿主自己的 Python 解释器（>= 3.12）。`-u` 别去掉：日志是重定向文件，
    块缓冲会让长跑期间 `log_tail` 一片空白。

    ### 3. 重开会话，工具面出现

    宿主连上后，工具面会直接出现：

    ```
    cbb__cbb_start_run        cbb__cbb_next_requests   cbb__cbb_reply
    cbb__cbb_status           cbb__cbb_build_box       cbb__cbb_lint_box
    cbb__cbb_gate             cbb__cbb_field_coverage  cbb__cbb_call_wiring
    cbb__cbb_lexicon_list     cbb__cbb_lexicon_approve cbb__cbb_supersede
    cbb__cbb_conflict_resolve cbb__cbb_quarantine_adjudicate
    cbb__cbb_model_call       cbb__cbb_model_stats     cbb__cbb_model_drain
    cbb__cbb_model_policy     cbb__cbb_doctor          cbb__cbb_capabilities
    cbb__cbb_smoke
    ```

    共 **21 个**。**这就是全部管辖面**——不需要 shell、不需要 CLI、不需要 mcporter。

    ## 依赖

    宿主 Python 需要这些（`pip install` 一次即可）：

    ```
    pydantic>=2.7  httpx>=0.27  ebooklib>=0.18  beautifulsoup4>=4.12
    rapidfuzz>=3.9  jieba>=0.42  PyYAML>=6.0  jsonschema>=4.22
    ```

    ## 为什么要这个包，而不是让 agent 自己起服务

    **传输成本**：用 CLI 客户端（如 mcporter）调 stdio server，**每次调用都现起一个
    server 进程**，实测 **8.4–16.8 秒/次**。三卷抽取一万多条请求 ⇒ 光传输就 20+ 小时。
    托管连接后这个成本消失。

    **工具本身很快**：进程内实测 `cbb_status` 65 ms、`cbb_next_requests` 0.01 ms、
    `cbb_model_stats` 10 ms。所以快的办法是托管传输，不是改工具。

    ## 架构

    ```
    MCP 工具 ──┐
               ├──> cbb_stage1 库（全部真实逻辑）
    CLI      ──┘
    ```

    两个前端，同一个库，互不依赖。**MCP 是管辖面；CLI 只是开发与修 bug 用的工具。**

    `cbb_start_run` 是唯一一处例外：它必须把一个长任务脱离出去（要活过这次工具调用），
    所以它起一个独立进程——那是进程管理，不是把能力转手给 CLI。
    """
    (dist / "README.md").write_text(README, encoding="utf-8")

    # ── 5. verify the package runs standalone ────────────────────────────────
    import subprocess
    import sys

    print("=" * 70)
    print("package contents")
    print("=" * 70)
    for entry in sorted(dist.iterdir()):
        kind = "dir " if entry.is_dir() else "file"
        size = "" if entry.is_dir() else f"  {entry.stat().st_size}B"
        print(f"  {kind} {entry.name}{size}")
    total = sum(p.stat().st_size for p in dist.rglob("*") if p.is_file())
    print(f"  total: {total / 1024:.0f} KB")

    print()
    print("=" * 70)
    print("standalone check: import the library using ONLY the package dir")
    print("=" * 70)
    # The check is a small script of its own, written without %-formatting: the previous
    # version used "%s" placeholders and the embedded "%d" collided with them, so the check
    # raised TypeError instead of running. A packaging script that cannot verify its own
    # output is worse than no check.
    # Dedented deliberately: this literal lives inside a function, and indenting the
    # surrounding block indented *its contents* too, so the heredoc became an
    # `IndentationError` the moment it was executed. `textwrap.dedent` decouples the
    # script's indentation from the indentation of the code holding it.
    CHECK = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {dist!r})
        import cbb_stage1.box.lint, cbb_stage1.model_ops, cbb_stage1.validate.adjudication
        print("library imports OK")

        import importlib.util as u
        spec = u.spec_from_file_location("mcp", {server!r})
        module = u.module_from_spec(spec)
        sys.modules["mcp"] = module
        spec.loader.exec_module(module)
        print("server tools:", len(module._DISPATCH))

        result = module.cbb_capabilities({{}})
        print("cbb_capabilities ok:", result.get("ok"))
        print("STANDALONE PASS")
        """
    )
    check = CHECK.format(dist=str(dist), server=str(dist / "server.py"))
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", check],
        cwd=str(dist), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(result.stdout.strip() or result.stderr[-800:])
    print(f"exit={result.returncode}")
    if result.returncode != 0:
        raise SystemExit("the package does not run standalone")

    # The check above imports the package from dist, which leaves __pycache__ behind. Strip
    # it so the zipped artefact carries code, not bytecode compiled on the build machine
    # (57 .pyc entries in the first release, roughly doubling the archive).
    for cache in list(dist.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    for stale in list(dist.rglob("*.pyc")):
        stale.unlink(missing_ok=True)
    print("stripped __pycache__ from the package")


if __name__ == "__main__":
    build()
