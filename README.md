# cbb_stage1

抽取段 v6.1 管线（箱 v2.2 契约）。设计见 `../设计/抽取段-v6/`，验收见
`../设计/抽取段-v6/07-吸收机制总清单-v2-20261020.md`。

## ⚠️ 本包不是自包含的：必须同时带上 `设计/抽取段-v6/`

`src/` 与 `tests/` 都靠**兄弟目录** `../设计/抽取段-v6/schemas/` 工作：
`box/writer.py` 建箱时从那里读 19 份 schema，`tests/unit/test_*` 也从那里读。
只复制 `cbb_stage1_v6/` 一个目录过去，建箱会直接抛：

```
BuildError: box schemas not found: <parent>/设计/抽取段-v6/schemas
```

2026-10-21 实测：只复制 `cbb_stage1_v6/` 时 **46 个测试失败**；把 `设计/` 一起复制后
**1027 个全过**。所以拷贝时保持这个相对位置：

```
<任意父目录>/
├── cbb_stage1_v6/          # 本包
└── 设计/抽取段-v6/schemas/  # 必需，不是文档
```

## 新机器上跑起来

需要 **Python 3.12**（`pyproject.toml` 的 `requires-python`，`preflight.MIN_PYTHON`
与它保持一致——两处曾一个写 3.11 一个写 3.12，preflight 会对 pip 装不上的环境报 READY）。

```powershell
$env:PYTHONUTF8='1'                     # 中文路径必须；否则 Windows 按 ANSI 代码页读写

# 先看这台机器上有哪个 3.12。`py -3.12` 只在官方安装包下有效；
# uv 装的 Python 会被注册到命名空间里（如 Astral/CPython3.12.14），必须用全名。
py -0                                   # 列出现有解释器与它们的标识
py -3.12 -m venv .venv                  # 若上面显示的就是 3.12，用这行
# 否则（uv / 其它命名空间）：
# py -V:Astral/CPython3.12.14 -m venv .venv

.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
```

2026-10-21 在本机实测：`py -3.12` 报 exit=103（找不到），
`py -V:Astral/CPython3.12.14` 正常。**不要照抄某台机器的绝对路径，先 `py -0` 看清单。**

`jieba` 首次分词会构建词典缓存（写到 `%TEMP%`），第一次跑会慢几秒，属正常。

## 模型端点（都可覆盖）

| 用途 | 默认 | 覆盖用的环境变量 |
|---|---|---|
| 主推理 / 嵌入 | `http://127.0.0.1:8080/v1`（LM Studio） | `CBB_LLM_BASE_URL`、`CBB_EMBEDDING_BASE_URL` |
| Ollama 备选 | `http://127.0.0.1:11434` | `CBB_OLLAMA_BASE_URL` |
| API key | 空 | `CBB_LLM_API_KEY`、`CBB_EMBEDDING_API_KEY` |
| judge（异档验证） | 未配置 | `CBB_MCP_JUDGE_BASE_URL` |

默认 **mock 引擎，不联网**；真实 LLM 只在 `--live` 下启用。没有模型服务时管线仍能跑完
（推理类能力会记为 unavailable，不伪造结果）。

## 命令

```powershell
.venv\Scripts\python -m cbb_stage1.cli lint-box ..\设计\抽取段-v6\example
.venv\Scripts\python -m cbb_stage1.cli run --book <book> --chapters 1-3 --out artifacts\box-3ch
.venv\Scripts\python -m cbb_stage1.cli run --book <book> --chapters 1 --live   # 走真实模型
.venv\Scripts\python -m cbb_stage1.cli build-box --work <work> --out <box>
.venv\Scripts\python -m cbb_stage1.cli acceptance acceptance\07.yaml
```

## 门禁（提交前应全绿）

```powershell
.venv\Scripts\python -X utf8 -m cbb_stage1.qc.id_sources          # ID 单一来源
.venv\Scripts\python -X utf8 -m cbb_stage1.cli wiring             # 模块可达
.venv\Scripts\python -m ruff check src\cbb_stage1 --select F401,F811,F841
.venv\Scripts\python -m vulture src\cbb_stage1 .vulture_whitelist.py --min-confidence 80
.venv\Scripts\python -X utf8 tools\call_wiring.py --root src\cbb_stage1     # 公开 API 是否真被调用
.venv\Scripts\python -X utf8 tools\field_coverage.py <box>                  # 必填字段是否恒空
```

`tests/unit/test_portability.py` 把「换一台 Windows 11 机器还能跑」这件事也钉成了测试：
源码里不许有绝对路径 / 用户名 / 桌面目录，端点必须可被环境变量覆盖，测试不许绑定本机路径。

## 换机时的已知依赖

| 依赖 | 说明 |
|---|---|
| 中文路径 | 所有 Python 调用带 `-X utf8` 或 `PYTHONUTF8=1`；git 看中文路径要 `-c core.quotepath=false` |
| 录制夹具 | `tests/data/spk5-run/` 随仓库走（41 KB），不需要本机历史运行产物 |
| 环境变量夹具 | 别的录制跑可用 `CBB_RECORDED_SPK5_WORK` 指向自己的 work 目录 |
| 长路径 | 仓库内最长 121 字符；放到很深的目录时注意 Windows 260 上限 |
