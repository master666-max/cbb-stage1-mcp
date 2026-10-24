"""The extraction prompts must carry the paper's one-shot demonstration.

Measured drift (A26)
--------------------
A26's paper (OpenIE / the 1-shot extractor) states at p.30, and in the prompt text itself:

    "1-shot prompting ... with the prompts shown in Appendix I"
    Appendix I contains a "One-Shot Demonstration" — a worked input/output pair.

The implementation is 0-shot: `openie.py:58-82` has system prompts that show an *output format* sample
(`[ "实体A", "实体B" ]`), which is a schema hint, not a demonstration of the task performed on real
input. Nothing in either prompt contains an input passage plus the correct extraction for it.

The distinction is not pedantic. A format sample tells the model what the JSON looks like; a one-shot
demonstration shows it what counts as an entity, how pronouns are resolved, and what granularity of
triple is wanted — which is exactly the behaviour the paper measured.

What is pinned
--------------
* each prompt contains a demonstration with both an input and its expected output;
* the demonstration is marked as an example, so the model does not extract from it;
* the real paragraph is still the thing to extract from, and appears after the demonstration;
* a caller can turn it off, because a one-shot prompt costs tokens on every call.
"""

from __future__ import annotations

from cbb_stage1.engines.ai_reader.openie import (
    ENTITY_EXTRACT_SYSTEM_PROMPT,
    RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT,
    build_entity_extract_prompt,
    build_rdf_triple_prompt,
)

PARAGRAPH = "言万推开木门，看见拉法正在院中练剑。"


# ══ the demonstration is in the system prompts ════════════════════════
def test_the_entity_prompt_contains_a_demonstration() -> None:
    text = ENTITY_EXTRACT_SYSTEM_PROMPT
    assert "示例输入" in text or "示例：输入" in text, (
        "the entity prompt has no demonstration input; a format sample is not a one-shot example"
    )
    assert "示例输出" in text or "示例：输出" in text, "the demonstration has no expected output"


def test_the_triple_prompt_contains_a_demonstration() -> None:
    text = RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT
    assert "示例输入" in text or "示例：输入" in text, (
        "the triple prompt has no demonstration input"
    )
    assert "示例输出" in text or "示例：输出" in text, "the demonstration has no expected output"


def test_the_demonstration_shows_pronoun_resolution() -> None:
    """The prompt asks for pronoun resolution; the example must demonstrate it, not just assert it."""
    text = ENTITY_EXTRACT_SYSTEM_PROMPT
    marker = text.split("示例输入")
    assert len(marker) > 1, "no demonstration input to inspect"
    block = marker[1][:600]
    assert "他" in block or "她" in block or "你" in block, (
        f"the demonstration does not show a pronoun being resolved: {block[:200]!r}"
    )


def test_the_demonstration_output_is_parseable() -> None:
    """An example the parser could not read teaches the wrong shape."""
    import json
    import re

    text = ENTITY_EXTRACT_SYSTEM_PROMPT
    blocks = re.findall(r"示例输出[：:]\s*(\[[^\]]*\])", text, re.DOTALL)
    assert blocks, "no JSON array found in the entity demonstration output"
    assert isinstance(json.loads(blocks[0]), list)


def test_the_triple_demonstration_output_is_parseable() -> None:
    import json
    import re

    text = RDF_TRIPLE_EXTRACT_SYSTEM_PROMPT
    # The demonstration spans several lines; take everything from the marker to the closing bracket
    # so the nested arrays are not truncated by a single-line pattern.
    marker = text.split("示例输出")
    assert len(marker) > 1, "no demonstration output to inspect"
    body = marker[1]
    start = body.find("[")
    assert start != -1, "the triple demonstration output is not a JSON array"
    # Walk to the matching close bracket instead of guessing with a regex.
    depth = 0
    end = -1
    for index in range(start, len(body)):
        if body[index] == "[":
            depth += 1
        elif body[index] == "]":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    assert end != -1, "the triple demonstration array is unbalanced"
    parsed = json.loads(body[start:end])
    assert isinstance(parsed, list) and parsed, "the demonstration parsed to nothing"
    assert all(len(item) == 3 for item in parsed), (
        f"the triple demonstration is not a list of triples: {parsed!r}"
    )


# ══ the real paragraph must not be confused with the example ═════════
def test_the_user_prompt_marks_the_example_as_an_example() -> None:
    prompt = build_entity_extract_prompt(PARAGRAPH)
    assert PARAGRAPH in prompt, "the paragraph to extract from is missing"
    assert "示例" in prompt or "example" in prompt.lower(), (
        "the user prompt does not mark the demonstration as an example, so the model may extract "
        "from it"
    )


def test_the_paragraph_comes_after_the_example() -> None:
    prompt = build_entity_extract_prompt(PARAGRAPH)
    example_at = prompt.find("示例")
    paragraph_at = prompt.find(PARAGRAPH)
    assert example_at != -1 and paragraph_at != -1
    assert example_at < paragraph_at, (
        "the paragraph precedes the demonstration; the model would read the example as a correction"
    )


# ══ it can be turned off ══════════════════════════════════════════════
def test_the_demonstration_can_be_disabled() -> None:
    prompt = build_entity_extract_prompt(PARAGRAPH, one_shot=False)
    assert "示例输入" not in prompt, "one_shot=False still included the demonstration"
    assert PARAGRAPH in prompt
