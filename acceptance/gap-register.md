# Gap Register

初始按 07 表格逐行解析为 60 PASS / 16 PARTIAL / 11 EXCLUDED。
实现后：A7、D13 经代码+测试升级为 PASS；D14 实现按章 forecast 后转为 PARTIAL。
当前矩阵：**62 PASS / 15 PARTIAL / 10 EXCLUDED（共 87）**，与 07 文末统计 62/15/10 对齐。

PARTIAL 缺口逐条在 `acceptance/gaps/<ID>.md`；EXCLUDED 见 `EXCLUDED.md`。
