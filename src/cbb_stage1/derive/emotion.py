"""Emotion curves for the inference layer (§5.20 ``emotion_curves.jsonl``).

What this is, and what it is not
--------------------------------
The published method for this is Utterance Emotion Dynamics (Vilhublotla et al.,
ACL 2024 Findings): a 500-word sliding window, NRC-VAD valence/arousal/dominance, and
fourteen derived metrics, measured **per character** because the paper reports
near-zero correlation between the narration arc and the dialogue arc (0.06–0.09) and
between characters (0.02–0.04).

That paper is **not in this project's resource library.** So this module does not claim
to implement it. `method` says `lexical_polarity_window`, and the note says so in words:
a small hand-written Chinese polarity lexicon, summed over each sentence and smoothed.
It is a reading, not UED.

Why per speaker matters even at this level
------------------------------------------
Merging narration and dialogue into one arc blots out both, per the same paper. The
lanes are therefore kept separate: the empty-string speaker is the narration voice, and
a named lane is one character's quoted lines. That is also why this file could not have
been written before the speaker work — without attribution every quote lands in one
undifferentiated bucket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from cbb_stage1 import ids

__all__ = [
    "METHOD",
    "NEGATIVE",
    "POSITIVE",
    "build_emotion_curves",
    "curve_for_unit",
]

#: Named for what it is. Not ``ued``, because UED is not implemented.
METHOD = "lexical_polarity_window"

#: A deliberately small lexicon. Documented as small: a reader must not read these
#: numbers as a validated affect measure. Chinese VAD resources are not in the library.
POSITIVE = frozenset(
    """好 笑 喜欢 爱 高兴 开心 快乐 幸福 温柔 温暖 希望 相信 努力 坚持 温柔
    放心 安心 感谢 谢谢 可爱 美丽 漂亮 温柔 勇敢 坚强 善良 圆满 成功 期待
    安慰 拥抱 守护 保护 珍贵 宝贵 甜蜜 安心 舒服 轻松 明亮 清澈 有趣""".split()
)

NEGATIVE = frozenset(
    """死 痛 痛苦 伤心 悲伤 绝望 残酷 可怕 恐怖 恨 讨厌 恶心 难受 流泪 哭
    废物 废弃 滞销 杀死 毁掉 破坏 崩坏 腐烂 腐朽 阴霾 空虚 孤独 寂 冷酷
    无能 无力 白费 徒劳 牺牲 遗憾 残酷 极端 黑暗 死掉 毁 惨 恐惧 无所谓的""".split()
)

_SENTENCE_END = re.compile(r"[。！？…；!?;]|\n")
_QUOTE = re.compile(r"[「『]([^」』]*)[」』]")
_QUOTE_SELF = re.compile(r"[「『][^」』]*[」』]")


def _sentences(text: str) -> list[str]:
    """Split into sentences without cutting through a quoted line.

    A dialogue line contains its own terminators — 「好。把它毁掉吧。」 has a full stop
    inside it — so a plain split produced 「好 and 把它毁掉吧。」 as two "sentences". The
    consequence was not cosmetic: neither fragment equalled the quote recorded on the
    assertion, so every attributed line fell into the narration lane and the per-speaker
    split quietly did nothing.

    Quoted spans are masked, the text is split, and the spans are restored.
    """
    spans: list[str] = []

    def mask(match: re.Match) -> str:
        spans.append(match.group(0))
        # A placeholder with no terminator characters, restored after the split.
        return f"\x00{len(spans) - 1}\x00"

    masked = _QUOTE_SELF.sub(mask, text or "")
    pieces = [piece.strip() for piece in _SENTENCE_END.split(masked)]
    restored: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        for index, span in enumerate(spans):
            piece = piece.replace(f"\x00{index}\x00", span)
        restored.append(piece)
    return restored


def _polarity(sentence: str) -> float | None:
    """Mean lexicon hit per sentence, or ``None`` when the sentence says nothing.

    Returning ``None`` rather than 0.0 matters: a neutral sentence and a sentence with
    no scoring words are different, and averaging zeros in would drag every curve toward
    the middle.
    """
    # Match lexicon terms directly instead of tokenising first. A greedy token regex
    # swallowed 「毁掉」 into 「把它毁掉」, so a clearly negative sentence scored positive;
    # substring lookup against the lexicon cannot make that mistake.
    hits = 0
    score = 0
    for term in POSITIVE:
        if term in sentence:
            hits += 1
            score += 1
    for term in NEGATIVE:
        if term in sentence:
            hits += 1
            score -= 1
    if not hits:
        return None
    return max(-1.0, min(1.0, score / hits))


@dataclass(frozen=True)
class Curve:
    """One lane of one unit."""

    book_id: str
    unit_id: str
    speaker: str
    points: list[dict[str, Any]]
    mean_polarity: float
    turning_points: list[int]

    def to_row(self, *, created_at: str | None = None) -> dict[str, Any]:
        return {
            "curve_id": ids.emotion_curve_id(self.book_id, self.unit_id, self.speaker),
            "book_id": self.book_id,
            "unit_id": self.unit_id,
            "speaker": self.speaker,
            # window_chars / step_chars are omitted rather than zeroed: this curve is
            # sentence-based, not window-based, and the schema's minimum for them is 1.
            # Emitting 0 was a contract violation the schema caught immediately.
            "points": self.points,
            "turning_points": self.turning_points,
            "mean_polarity": self.mean_polarity,
            "method": METHOD,
            "note": (
                "小型中文极性词典逐句打分并分轨（叙述/各角色）；"
                "**不是 UED**——UED 论文不在本项目资料库内，方法未实现"
            ),
            "produced_by": "derive:emotion",
            "created_at": created_at,
        }


def _speaker_lookup(statements: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """quote -> speaker name, from the statement assertions that carry one."""
    lookup: dict[str, str] = {}
    for row in statements:
        if str(row.get("claim_layer") or "") != "statement":
            continue
        speaker = row.get("speaker") or {}
        name = str((speaker or {}).get("name") or "")
        if not name:
            continue
        for evidence in row.get("evidence") or ():
            quote = str(evidence.get("quote") or "")
            if quote:
                lookup[quote] = name
    return lookup


def curve_for_unit(
    book_id: str,
    unit_id: str,
    text: str,
    *,
    speakers: Mapping[str, str] | None = None,
) -> list[Curve]:
    """Build one curve per lane: the narration voice, plus each attributed character."""
    lookup = speakers or {}
    lanes: dict[str, list[dict[str, Any]]] = {}
    offset = 0

    for sentence in _sentences(text or ""):
        start = (text or "").find(sentence, offset)
        if start < 0:
            start = offset
        offset = start + len(sentence)

        quote = _QUOTE.search(sentence)
        if quote is None:
            lane = ""
        else:
            # The lane is the character this quoted line was attributed to. Without an
            # attribution the line falls back to narration, which is visibly a
            # different lane rather than a silent guess.
            lane = lookup.get(quote.group(0), "")
        polarity = _polarity(sentence)
        if polarity is None:
            continue
        lanes.setdefault(lane, []).append(
            {"offset": start, "polarity": polarity, "quote": sentence[:120]}
        )

    curves: list[Curve] = []
    for speaker in sorted(lanes):
        points = lanes[speaker]
        polarities = [point["polarity"] for point in points]
        mean = round(sum(polarities) / len(polarities), 6)
        turns: list[int] = []
        for previous, current in zip(points, points[1:]):
            if (previous["polarity"] - mean) * (current["polarity"] - mean) < 0:
                turns.append(current["offset"])
        curves.append(
            Curve(
                book_id=book_id,
                unit_id=unit_id,
                speaker=speaker,
                points=points,
                mean_polarity=mean,
                turning_points=turns,
            )
        )
    return curves


def build_emotion_curves(
    book_id: str,
    units: Iterable[Mapping[str, Any]],
    *,
    texts: Mapping[str, str] | None = None,
    statements: Sequence[Mapping[str, Any]] = (),
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """One row per (unit, speaker lane) that produced at least one scored sentence."""
    lookup = _speaker_lookup(statements)
    body = texts or {}
    rows: list[dict[str, Any]] = []
    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        text = body.get(unit_id)
        if not unit_id or not text:
            continue
        for curve in curve_for_unit(book_id, unit_id, text, speakers=lookup):
            rows.append(curve.to_row(created_at=created_at))
    rows.sort(key=lambda row: row["curve_id"])
    return rows
