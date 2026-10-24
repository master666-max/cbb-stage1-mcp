"""Produce `story_time` from `time_expr`, conservatively.

The box schema wants `{start, end, precision}`. The extractor gives free text. Only expressions that
name a calendar anchor are converted; a time of day (清晨/黄昏) is not a calendar interval and is left
null rather than dressed up as one, because inventing a span from "morning" is the same error as
inventing a confidence.

`start`/`end` are opaque story-time labels, not ISO dates: this book has no absolute calendar, so
`第3年` is the honest representation. A downstream reader that needs ordering can compare the year
number; one that needs a date must not pretend it has one.
"""

from __future__ import annotations

import re

#: Chinese numerals used in story expressions.
_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

#: Month names that appear in this corpus's register.
_MONTHS = {
    "正月": 1, "元月": 1, "一月": 1, "二月": 2, "三月": 3, "四月": 4,
    "五月": 5, "六月": 6, "七月": 7, "八月": 8, "九月": 9, "十月": 10,
    "冬月": 11, "十一月": 11, "腊月": 12, "十二月": 12,
}

_YEAR_PATTERNS = (
    re.compile(r"第\s*([0-9]+|[零一二三四五六七八九十]+)\s*年"),
    re.compile(r"([0-9]+|[零一二三四五六七八九十]+)\s*年后"),
    re.compile(r"([0-9]+|[零一二三四五六七八九十]+)\s*年之后"),
    re.compile(r"([0-9]+|[零一二三四五六七八九十]+)\s*年过去"),
)


def _to_int(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if token == "十":
        return 10
    if token.startswith("十"):
        rest = token[1:]
        return 10 + (_CN_DIGITS.get(rest, 0) if rest else 0)
    if token.endswith("十"):
        head = token[:-1]
        return _CN_DIGITS.get(head, 0) * 10 if head else 10
    if "十" in token:
        head, _, tail = token.partition("十")
        return _CN_DIGITS.get(head, 0) * 10 + (_CN_DIGITS.get(tail, 0) if tail else 0)
    return _CN_DIGITS.get(token)


def parse_story_time(expression: str | None) -> dict[str, str] | None:
    """Return a box-shaped ``story_time`` for a calendar expression, else ``None``.

    Only year and month anchors are converted. A time of day is deliberately rejected: it carries no
    interval, and the schema's ``precision`` enum has no value for it, so the honest output is ``None``
    with the caller recording the expression as text.
    """
    text = " ".join(str(expression or "").split())
    if not text:
        return None

    # Year anchors first: 三年后 is more specific than any month it might mention.
    for pattern in _YEAR_PATTERNS:
        match = pattern.search(text)
        if match:
            number = _to_int(match.group(1))
            if number is None:
                continue
            label = f"第{number}年"
            return {"start": label, "end": label, "precision": "year"}

    for name, month in _MONTHS.items():
        if name in text:
            label = f"{month}月"
            return {"start": label, "end": label, "precision": "month"}

    # 清晨 / 黄昏 / 正午 / 夜: a time of day, not an interval. Refused on purpose.
    return None
