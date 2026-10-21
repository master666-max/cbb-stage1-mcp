"""Deterministic time-expression extraction (07 D10 / D11).

This is the missing **producer** for timeline constraints: it finds explicit
time expressions in unit text, attaches them to events whose summary/evidence
contains them, and builds only *inferred* ordering constraints between events
with explicit time mentions in the same unit.  It never derives story order from
plain narrative order.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from ..ids import sha256_text, timeline_id

__all__ = [
    "TimeMention",
    "extract_time_mentions",
    "annotate_event_times",
    "build_timeline_from_events",
]

_REL = re.compile(
    r"(?:\d+|[一二三四五六七八九十百千万两]+)\s*(?:日|天|周|星期|个月|月|年)\s*(?:后|之后|以后|前|之前|以前)"
)
_SIMPLE = re.compile(r"(?:次日|翌日|第二天|同日|当天|数日后|不久后|片刻后|稍后)")


class TimeMention:
    __slots__ = ("text", "start", "end")

    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start = start
        self.end = end


def extract_time_mentions(text: str) -> list[TimeMention]:
    found: list[TimeMention] = []
    for pattern in (_REL, _SIMPLE):
        for match in pattern.finditer(text or ""):
            found.append(TimeMention(match.group(0), match.start(), match.end()))
    return sorted(found, key=lambda m: (m.start, m.end))


def annotate_event_times(
    events: Iterable[dict[str, Any]],
    unit_texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Attach the first explicit time mention found in an event's own evidence."""
    out: list[dict[str, Any]] = []
    for event in events:
        event = dict(event)
        unit_id = str(event.get("unit_id") or "")
        text = unit_texts.get(unit_id, "")
        mentions = extract_time_mentions(text)
        chosen = None
        probes = [str(event.get("summary") or "")]
        probes.extend(str(ev.get("quote") or "") for ev in event.get("evidence") or [])
        for probe in probes:
            for mention in mentions:
                if mention.text and mention.text in probe:
                    chosen = mention
                    break
            if chosen:
                break
        if chosen:
            time = dict(event.get("time") or {})
            time.setdefault("anchor_unit", unit_id)
            time["expression"] = chosen.text
            time["time_expr"] = chosen.text
            event["time"] = time
        out.append(event)
    return out


def _event_mention(event: Mapping[str, Any], unit_texts: Mapping[str, str]) -> TimeMention | None:
    expr = str((event.get("time") or {}).get("time_expr") or "")
    if not expr:
        return None
    text = unit_texts.get(str(event.get("unit_id") or ""), "")
    for mention in extract_time_mentions(text):
        if mention.text == expr:
            return mention
    return None


def build_timeline_from_events(
    book_id: str,
    events: Iterable[Mapping[str, Any]],
    unit_texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build ``before`` constraints only between events with explicit time mentions.

    Same-unit, explicit-time only; source=inferred, confidence=0.5.  Events
    without an id or a time mention are skipped.  This is deliberately
    conservative: no mention => no constraint.
    """
    by_unit: dict[str, list[tuple[int, Mapping[str, Any], TimeMention]]] = {}
    for event in events:
        eid = str(event.get("event_id") or "")
        unit_id = str(event.get("unit_id") or "")
        if not eid or not unit_id:
            continue
        mention = _event_mention(event, unit_texts)
        if mention is None:
            continue
        by_unit.setdefault(unit_id, []).append((mention.start, event, mention))

    rows: list[dict[str, Any]] = []
    for unit_id, items in sorted(by_unit.items()):
        items.sort(key=lambda item: item[0])
        for (_, left, _), (_, right, right_mention) in zip(items, items[1:]):
            left_id = str(left["event_id"])
            right_id = str(right["event_id"])
            rows.append(
                {
                    # canonical t: formula (ids.timeline_id); same book/kind/nodes
                    # => same constraint id across reruns (I7).
                    "constraint_id": timeline_id(
                        book_id, "before", "event", left_id, "event", right_id
                    ),
                    "book_id": book_id,
                    "kind": "before",
                    "left": {"kind": "event", "id": left_id},
                    "right": {"kind": "event", "id": right_id},
                    "source": "inferred",
                    "confidence": 0.5,
                    "evidence": [
                        {
                            "unit_id": unit_id,
                            "start": right_mention.start,
                            "end": right_mention.end,
                            "quote": right_mention.text,
                            "quote_sha256": sha256_text(right_mention.text),
                            "occurrence": 1,
                        }
                    ],
                    "status": "inferred",
                    "deps": [left_id, right_id],
                }
            )
    return sorted(rows, key=lambda row: row["constraint_id"])
