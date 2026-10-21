"""Prompts for the LPMM two-stage extractor.

Copied verbatim from B05T-MaiBot-0.12.0 ``src/chat/knowledge/prompt_template.py``
(the LPMM NER -> RDF prompts).  Kept in Chinese and unchanged so the engine's
output style matches the source pipeline; only the message-splitting helper is
new (the source concatenated system + paragraph into one context string).
"""

from __future__ import annotations

import json

ENTITY_EXTRACT_SYSTEM_PROMPT = """你是一个性能优异的实体提取系统。请从段落中提取出所有实体，并以JSON列表的形式输出。

输出格式示例：
[ "实体A", "实体B", "实体C" ]

请注意以下要求：
- 将代词（如“你”、“我”、“他”、“她”、“它”等）转化为对应的实体命名，以避免指代不清。
- 尽可能多的提取出段落中的全部实体；
"""

RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT = """你是一个性能优异的RDF（资源描述框架，由节点和边组成，节点表示实体/资源、属性，边则表示了实体和实体之间的关系以及实体和属性的关系。）构造系统。你的任务是根据给定的段落和实体列表构建RDF图。

请使用JSON回复，使用三元组的JSON列表输出RDF图中的关系（每个三元组代表一个关系）。

输出格式示例：
[
        ["某实体","关系","某属性"],
        ["某实体","关系","某实体"],
        ["某资源","关系","某属性"]
]

请注意以下要求：
- 每个三元组应包含每个段落的实体命名列表中的至少一个命名实体，但最好是两个。
- 将代词（如“你”、“我”、“他”、“她”、“它”等）转化为对应的实体命名，以避免指代不清。
"""


def build_entity_prompt(paragraph: str) -> str:
    """User message for stage 1 (NER)."""
    return f"""段落：
```
{paragraph}
```"""


def build_rdf_prompt(paragraph: str, entities: list[str]) -> str:
    """User message for stage 2 (RDF triples)."""
    entities_json = json.dumps(entities, ensure_ascii=False)
    return f"""段落：
```
{paragraph}
```

实体列表：
```
{entities_json}
```"""
