# cbb-stage1

抽取段 v6.1 管线（箱 v2.2 契约）。**MCP 驱动**：21 个工具构成完整的管辖面，
CLI 只是开发与修 bug 用的前端。

```
MCP 工具 ──┐
           ├──> cbb_stage1 库（全部真实逻辑）
CLI      ──┘
```

## 快速开始

### 方式一：安装包（推荐，给另一台设备）

从 [Releases](../../releases) 下载 `cbb-mcp-*.zip`，解压到任意目录，把 `mcp.json`
并进宿主的 MCP 配置，重开会话即可。**不需要 pip install，不需要任何项目路径。**

```json
{
  "mcpServers": {
    "cbb": { "command": "python", "args": ["-u", "<解压目录>/server.py"] }
  }
}
```

宿主 Python 需 3.12+，依赖：

```
pydantic>=2.7  httpx>=0.27  ebooklib>=0.18  beautifulsoup4>=4.12
rapidfuzz>=3.9  jieba>=0.42  PyYAML>=6.0  jsonschema>=4.22
```

### 方式二：从源码

```powershell
$env:PYTHONUTF8='1'
py -0                                   # 看清 3.12 的标识（uv 装的会带命名空间）
py -V:<标识> -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q
```

## 21 个工具

| 类别 | 工具 |
|---|---|
| 运行 | `cbb_start_run` `cbb_status` `cbb_next_requests` `cbb_reply` |
| 建箱/校验 | `cbb_build_box` `cbb_lint_box` `cbb_field_coverage` `cbb_call_wiring` |
| 门禁 | `cbb_gate`(wiring / id_sources / acceptance) |
| 人裁 | `cbb_lexicon_list` `cbb_lexicon_approve` `cbb_supersede` `cbb_conflict_resolve` `cbb_quarantine_adjudicate` |
| 模型 | `cbb_model_call` `cbb_model_stats` `cbb_model_drain` `cbb_model_policy` |
| 其他 | `cbb_doctor` `cbb_capabilities` `cbb_smoke` |

## 为什么是 MCP 而不是 CLI

**实测**：用 CLI 型 MCP 客户端调 stdio server，**每次调用现起一个 server 进程**，
8.4–16.8 秒/次；一万条回填光传输就 20+ 小时。托管连接后这个成本消失。

**工具本身很快**：进程内实测 `cbb_status` 65 ms、`cbb_next_requests` 0.01 ms、
`cbb_model_stats` 10 ms。所以正确的优化是托管传输，不是改工具。

`cbb_start_run` 是唯一一处例外——它必须把长任务脱离出去（要活过这次工具调用），
那是进程管理，不是把能力转手给 CLI。

## 纪律

- **I4**：谓词不许从数据里自己长出来。未注册谓词进 `proposed_predicates`，
  必须人工批准（带 `by`）才能注册。
- **I27**：`derived/inference/*` 只在派生层，每条带证据或统计口径，不参与硬门。
- **人裁不删件**：裁决追加到 `audit/decisions.jsonl`，被裁的记录只被标注。
- **测试的参照物是规范**，不是实现。

## 门禁

```powershell
.\.venv\Scripts\python -X utf8 -m cbb_stage1.qc.id_sources
.\.venv\Scripts\python -X utf8 -m cbb_stage1.cli wiring
.\.venv\Scripts\python -m ruff check src\cbb_stage1 --select F401,F811,F841
.\.venv\Scripts\python -m vulture src\cbb_stage1 .vulture_whitelist.py --min-confidence 80
.\.venv\Scripts\python -X utf8 tools\call_wiring.py --root src\cbb_stage1
```

## 已知限制

- 占位与代理件是**明标**的：`derived/inference/forecasts.jsonl` 的 `method` 字段写着
  `lexical_singleton_ratio`，不是 E07 的原法；情绪曲线标 `lexical_polarity_window`，
  不是 UED。
- `cbb_start_run` 起的是独立进程；宿主环境若会连坐杀进程树，长跑会被打断。
  判据：`plan/mcp-runs/*.pid.rc` 不存在就是被杀，不是自己退出的。
- 推断层（动机链/情绪/主题）在单章上只能验证机制，不能验证价值。
- **`example/` 箱子夹具不随仓库发布**（它由受版权保护的原文构建）。需要它的两条测试
  （`test_box_hollowing_invariant` / `test_box_lint_example`）会在 clone 里具名 skip，
  只在作者工作区运行。

## 许可

MIT，见 `LICENSE`。
