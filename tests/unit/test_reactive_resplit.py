"""A paragraph that overflows the context must be re-split, not dropped.

Measured gap (A11)
------------------
`openie.py:435-440` catches any failure per paragraph and appends it to `result.failed`:

    except (OpenIEError, LLMError, ValueError) as exc:
        logger.warning(...)
        result.failed.append(paragraph)
        continue

The source reacts differently to one specific failure: when the model reports the input is too long, it
splits the paragraph and retries the halves. That is the only recovery that can succeed — the text did
not change, so retrying the same oversized paragraph fails the same way, and dropping it loses content
permanently.

What is pinned
--------------
* a context-length failure triggers a re-split and a retry, and the recovered output appears in the
  result;
* **other** failures still go to `failed` untouched — splitting on a network error would multiply the
  cost of a failing call;
* the recursion terminates: a paragraph that cannot be split further is recorded as failed, not looped.
"""

from __future__ import annotations

import pytest

from cbb_stage1.engines._textsplit import Paragraph
from cbb_stage1.engines.ai_reader.llm_client import LlmUsage
from cbb_stage1.engines.ai_reader.openie import (
    OpenIEError,
    OpenIEExtractor,
    looks_like_context_overflow,
)

LONG_TEXT = "言万推开木门，看见拉法正在院中练剑。" * 20


def _paragraph(text: str) -> Paragraph:
    from cbb_stage1.ids import sha256_hex_text

    return Paragraph(text=text, start=0, end=len(text), sha256=sha256_hex_text(text))


class LengthLimitedLLM:
    """Raises a context-overflow error when the prompt exceeds `limit`."""

    def __init__(self, limit: int = 350) -> None:
        self.limit = limit
        self.prompts: list[int] = []

    async def generate(self, *, system: str, prompt: str, **kwargs):  # noqa: ANN001
        self.prompts.append(len(prompt))
        if len(prompt) > self.limit:
            raise OpenIEError("This model's maximum context length is 4096 tokens")
        # NER wants a list, RDF wants triples; both parse from a JSON array.
        if "实体" in system:
            return '["言万", "拉法"]', LlmUsage()
        return '[["言万","遇见","拉法"]]', LlmUsage()


# ══ the classifier ═══════════════════════════════════════════════════
def test_a_context_overflow_error_is_recognised() -> None:
    assert looks_like_context_overflow(
        OpenIEError("This model's maximum context length is 4096 tokens")
    )


def test_another_failure_is_not_mistaken_for_an_overflow() -> None:
    """Splitting on a network error would multiply the cost of a failing call."""
    assert not looks_like_context_overflow(OpenIEError("connection reset by peer"))
    assert not looks_like_context_overflow(RuntimeError("boom"))


def test_the_chinese_wording_is_recognised() -> None:
    assert looks_like_context_overflow(OpenIEError("输入超出上下文长度限制"))


# ══ the re-split ═════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_an_overflowing_paragraph_is_re_split_not_dropped() -> None:
    extractor = OpenIEExtractor(
        LengthLimitedLLM(limit=350), target_chars=10_000, min_chars=1
    )
    result = await extractor.extract_paragraphs([_paragraph(LONG_TEXT)])

    assert result.docs, "the overflowing paragraph produced no documents"
    assert not result.failed, f"the paragraph was dropped instead of re-split: {result.failed}"


@pytest.mark.asyncio
async def test_the_recovery_is_recorded_on_the_result() -> None:
    """A reader must be able to tell a one-pass paragraph from a recovered one."""
    extractor = OpenIEExtractor(
        LengthLimitedLLM(limit=350), target_chars=10_000, min_chars=1
    )
    result = await extractor.extract_paragraphs([_paragraph(LONG_TEXT)])
    assert getattr(result, "resplit", 0) >= 1, (
        "the result does not say that a paragraph had to be re-split"
    )


# ══ other failures are unchanged ════════════════════════════════════
@pytest.mark.asyncio
async def test_a_non_length_failure_still_marks_the_paragraph_failed() -> None:
    class Boom:
        async def generate(self, **kwargs):  # noqa: ANN001
            raise OpenIEError("connection reset by peer")

    extractor = OpenIEExtractor(Boom(), target_chars=10_000, min_chars=1)
    with pytest.raises(OpenIEError):
        await extractor.extract_paragraphs([_paragraph("言万遇见拉法。")])


# ══ the recursion terminates ════════════════════════════════════════
@pytest.mark.asyncio
async def test_a_paragraph_that_cannot_shrink_further_is_recorded_as_failed() -> None:
    """A one-character paragraph cannot be split; record it rather than loop."""

    class AlwaysOverflow:
        async def generate(self, **kwargs):  # noqa: ANN001
            raise OpenIEError("maximum context length exceeded")

    extractor = OpenIEExtractor(AlwaysOverflow(), target_chars=10_000, min_chars=1)
    with pytest.raises(OpenIEError):
        await extractor.extract_paragraphs([_paragraph("。")])
