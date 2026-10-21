"""Projection-facing tables: citations, narrative, graph edges, conflicts,
coverage, assets.

These are derived deterministically from the box's canonical tables.  They do
not call a model and do not invent facts: every citation is an exact span, every
graph edge points back to an assertion, and coverage counts are recomputed.

P0-D downgrade: the box writer **no longer calls** ``fill_claim_skeletons``,
``build_alias_merges``, ``build_conflicts`` or ``build_insufficient_hypotheses``.
Those tables are now produced by the claims/resolve/validate/derive integrations
and consumed from ``work/``; the helpers stay here only for the projection unit
tests that pin their standalone behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from cbb_stage1.ids import (
    citation_id as _ids_citation_id,
    sha256_bytes as _ids_sha256_bytes,
    sha256_text as _ids_sha256_text,
    short_hash as _ids_short_hash,
)

#: Media reference patterns for :func:`extract_assets`.
_MD_IMAGE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+[\"'](?P<title>[^\"']*)[\"'])?\)"
)
_HTML_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_MEDIA = re.compile(r"<(?:audio|video)\b[^>]*>", re.IGNORECASE)
_HTML_SRC = re.compile(r"""\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_HTML_ALT = re.compile(r"""\balt\s*=\s*["']([^"']*)["']""", re.IGNORECASE)

#: Extension -> media type for the asset table.
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}

ASPECTS = (
    "presence",
    "identity",
    "attributes",
    "relations",
    "events",
    "time",
    "evidence",
    "summary",
    "cue",
    "quality",
)
LIBRARIES = ("实体", "关系", "事件", "埋线", "设定", "时序")


def _sha(text: str) -> str:
    """Thin alias to the single formula source (:mod:`cbb_stage1.ids`)."""
    return _ids_sha256_text(text)


def _short(text: str, n: int = 16) -> str:
    """Thin alias to the single formula source (:mod:`cbb_stage1.ids`)."""
    return _ids_short_hash(text, n)


def _iter_evidence(rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for row in rows:
        for ev in row.get("evidence") or []:
            if isinstance(ev, Mapping):
                yield ev


def _cue_evidence(cues: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for cue in cues:
        for part in ("setup", "trigger", "payoff"):
            block = cue.get(part) or {}
            for ev in block.get("evidence") or []:
                if isinstance(ev, Mapping):
                    yield ev
        for ev in cue.get("reinforce") or []:
            if isinstance(ev, Mapping):
                yield ev


def build_citations(
    book_id: str,
    units: list[Mapping[str, Any]],
    assertions: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    cues: list[Mapping[str, Any]],
    claims: list[Mapping[str, Any]],
    narratives: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unit_sha = {u["unit_id"]: u["sha256"] for u in units}
    found: dict[str, dict[str, Any]] = {}

    def add(unit_id: str, start: int, end: int, quote: str, occurrence: int = 1) -> str:
        if unit_id not in unit_sha:
            raise ValueError(f"citation references missing unit {unit_id}")
        cid = _ids_citation_id(unit_id, int(start), int(end), quote)
        if cid not in found:
            found[cid] = {
                "citation_id": cid,
                "book_id": book_id,
                "unit_id": unit_id,
                "start": int(start),
                "end": int(end),
                "quote": quote,
                "quote_sha256": _sha(quote),
                "occurrence": int(occurrence),
                "source_hash": unit_sha[unit_id],
            }
        return cid

    for ev in _iter_evidence(assertions):
        add(ev["unit_id"], ev["start"], ev["end"], ev["quote"], ev.get("occurrence", 1))
    for ev in _iter_evidence(events):
        add(ev["unit_id"], ev["start"], ev["end"], ev["quote"], ev.get("occurrence", 1))
    for ev in _cue_evidence(cues):
        add(ev["unit_id"], ev["start"], ev["end"], ev["quote"], ev.get("occurrence", 1))
    for claim in claims:
        for question in claim.get("questions") or []:
            for answer in question.get("answers") or []:
                add(
                    answer["unit_id"],
                    answer["start"],
                    answer["end"],
                    answer["quote"],
                    1,
                )
    for seg in narratives:
        add(seg["unit_id"], seg["start"], seg["end"], seg["text"], 1)
    return sorted(found.values(), key=lambda row: row["citation_id"])


def build_narrative(
    units: list[Mapping[str, Any]],
    unit_texts: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    heading = re.compile(r"^第[0-9〇一二三四五六七八九十百千]+[章节卷回]")
    for unit in sorted(units, key=lambda u: u["order"]):
        text = unit_texts[unit["unit_id"]]
        offset = 0
        order = 0
        for line in text.splitlines(keepends=True):
            raw = line.rstrip("\r\n")
            start = offset
            end = start + len(raw)
            offset += len(line)
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or heading.match(stripped):
                continue
            order += 1
            seg_type = "quote" if stripped[0] in "「“‘" else "paragraph"
            content_hash = _sha(f"{seg_type}\x1f{raw}")
            rows.append(
                {
                    "seg_id": "n:" + _short(f"{unit['unit_id']}|{order}|{content_hash}"),
                    "unit_id": unit["unit_id"],
                    "entity_id": None,
                    "order": order,
                    "type": seg_type,
                    "text": raw,
                    "content_hash": content_hash,
                    # internal fields, stripped by the writer
                    "start": start,
                    "end": end,
                }
            )
    return rows


def attach_narrative_anchors(narratives: list[dict[str, Any]]) -> None:
    for seg in narratives:
        seg["anchor"] = _ids_citation_id(
            seg["unit_id"], seg["start"], seg["end"], seg["text"]
        )


def build_graph_edges(book_id: str, assertions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def entity_ids(roles: Mapping[str, Any]) -> list[str]:
        out: list[str] = []
        for value in roles.values():
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str) and item.startswith("e:"):
                    out.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("entity"), str):
                    out.append(item["entity"])
        return list(dict.fromkeys(out))

    for assertion in assertions:
        ids = entity_ids(assertion.get("roles") or {})
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                key = f"{left}|{right}|{assertion['predicate']}|{assertion['assertion_id']}"
                eid = "g:" + _short(key)
                if eid in found:
                    continue
                found[eid] = {
                    "edge_id": eid,
                    "book_id": book_id,
                    "from": left,
                    "to": right,
                    "predicate": assertion["predicate"],
                    "assertion_id": assertion["assertion_id"],
                    "library": assertion["library"],
                    "strength": float(assertion.get("confidence") or 0.0),
                    "evidence_count": len(assertion.get("evidence") or []),
                    "status": assertion.get("status", "asserted"),
                }
    return sorted(found.values(), key=lambda row: row["edge_id"])


def build_coverage(
    book_id: str,
    entities: list[Mapping[str, Any]],
    assertions: list[Mapping[str, Any]],
    events: list[Mapping[str, Any]],
    timeline: list[Mapping[str, Any]],
    cues: list[Mapping[str, Any]],
    claims: list[Mapping[str, Any]],
    summaries: list[Mapping[str, Any]],
    citations: list[Mapping[str, Any]],
    graph_edges: list[Mapping[str, Any]],
    conflicts: list[Mapping[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    by_lib = {lib: 0 for lib in LIBRARIES}
    for assertion in assertions:
        lib = assertion.get("library")
        if lib in by_lib:
            by_lib[lib] += 1
    base = {
        "实体": len(entities),
        "关系": by_lib["关系"] + len(graph_edges),
        "事件": len(events) + len(timeline),
        "埋线": len(cues),
        "设定": by_lib["设定"],
        "时序": len(timeline),
    }
    counts = {
        "presence": base,
        "identity": {"实体": len(entities), **{lib: base[lib] for lib in LIBRARIES if lib != "实体"}},
        "attributes": {"实体": by_lib["实体"], "关系": by_lib["关系"], "事件": 0, "埋线": 0, "设定": by_lib["设定"], "时序": 0},
        "relations": {"实体": 0, "关系": len(graph_edges), "事件": 0, "埋线": 0, "设定": 0, "时序": 0},
        "events": {"实体": 0, "关系": 0, "事件": len(events), "埋线": 0, "设定": 0, "时序": 0},
        "time": {"实体": 0, "关系": 0, "事件": len(events), "埋线": 0, "设定": 0, "时序": len(timeline)},
        "evidence": {"实体": len(citations), "关系": len(citations), "事件": len(citations), "埋线": len(citations), "设定": len(citations), "时序": len(citations)},
        "summary": {"实体": 0, "关系": 0, "事件": len(summaries), "埋线": 0, "设定": 0, "时序": 0},
        "cue": {"实体": 0, "关系": 0, "事件": 0, "埋线": len(cues), "设定": 0, "时序": 0},
        "quality": {"实体": len(conflicts), "关系": len(conflicts), "事件": len(conflicts), "埋线": len(conflicts), "设定": len(conflicts), "时序": len(conflicts)},
    }
    cells: list[dict[str, Any]] = []
    for lib in LIBRARIES:
        for aspect in ASPECTS:
            value = counts[aspect][lib]
            cells.append(
                {
                    "library": lib,
                    "aspect": aspect,
                    "status": "present" if value > 0 else "absent",
                    "count": int(value),
                    "reason": None if value > 0 else "no_records",
                }
            )
    return {
        "book_id": book_id,
        "spec_version": "2.2.0",
        "aspects": list(ASPECTS),
        "libraries": list(LIBRARIES),
        "cells": cells,
        "counts": {
            "entities": len(entities),
            "assertions": len(assertions),
            "events": len(events),
            "timeline": len(timeline),
            "cues": len(cues),
            "claims": len(claims),
            "summaries": len(summaries),
            "citations": len(citations),
            "graph_edges": len(graph_edges),
            "conflicts": len(conflicts),
        },
        "generated_at": generated_at,
    }


def _media_type(path: str) -> str:
    suffix = Path(path.split("?", 1)[0].split("#", 1)[0]).suffix.lower()
    return MEDIA_TYPES.get(suffix, "application/octet-stream")


def _asset_sha(path: str, media_roots: Iterable[Path]) -> str:
    """Hash the media bytes when resolvable, else the reference URI.

    The box asset schema has a single ``sha256`` field and no "unresolved" flag.
    Using the URI digest keeps the row deterministic; this behaviour is
    documented in :func:`extract_assets`.
    """
    relative = path.split("?", 1)[0].split("#", 1)[0].lstrip("/\\")
    for root in media_roots:
        candidate = Path(root) / relative
        try:
            if candidate.is_file():
                return _sha_bytes(candidate.read_bytes())
        except OSError:
            continue
    return _sha_bytes(path.encode("utf-8"))


def _sha_bytes(data: bytes) -> str:
    """Thin alias to the single formula source (:mod:`cbb_stage1.ids`)."""
    return _ids_sha256_bytes(data)


def _scan_media_references(text: str) -> list[tuple[int, str, str | None]]:
    """Return ``(position, src, alt)`` for markdown/HTML media references."""
    found: list[tuple[int, str, str | None]] = []
    for match in _MD_IMAGE.finditer(text or ""):
        alt = match.group("alt") or match.group("title") or None
        found.append((match.start(), match.group("src"), alt))
    for match in _HTML_IMG.finditer(text or ""):
        tag = match.group(0)
        src_match = _HTML_SRC.search(tag)
        if not src_match:
            continue
        alt_match = _HTML_ALT.search(tag)
        found.append((match.start(), src_match.group(1), alt_match.group(1) if alt_match else None))
    for match in _HTML_MEDIA.finditer(text or ""):
        src_match = _HTML_SRC.search(match.group(0))
        if src_match:
            found.append((match.start(), src_match.group(1), None))
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def extract_assets(
    book_id: str,
    units: Iterable[Mapping[str, Any]],
    unit_texts: Mapping[str, str],
    *,
    media_roots: Iterable[str | Path] = (),
) -> list[dict[str, Any]]:
    """Extract media references from unit text into ``sources/assets.jsonl`` rows.

    Sources: markdown images ``![alt](path "title")``, HTML ``<img src alt>``,
    and HTML ``<audio|video src>``. Alt/title text becomes the asset ``alt``
    ("media description" in the source).

    ``sha256`` is the media file's digest when the binary can be found under
    ``media_roots``; otherwise it is the digest of the reference URI (deterministic
    and explicitly not a media digest). No media references => empty list, which
    is the valid "EMPTY-VALID" box state.

    Args:
        book_id: Box book id.
        units: Unit records (``unit_id``/``order``).
        unit_texts: unit_id -> text.
        media_roots: Candidate roots for resolving referenced media files.

    Returns:
        Asset rows sorted by ``asset_id``; duplicates (same URI+alt) collapse to
        the first unit that referenced them.
    """
    roots = [Path(root) for root in media_roots]
    rows: dict[str, dict[str, Any]] = {}
    for unit in sorted(units, key=lambda item: (int(item.get("order") or 0), str(item.get("unit_id") or ""))):
        unit_id = str(unit.get("unit_id") or "")
        text = unit_texts.get(unit_id, "")
        for _, src, alt in _scan_media_references(text):
            reference = str(src).strip()
            if not reference:
                continue
            key = f"{book_id}|{reference}|{alt or ''}"
            asset_id = "as:" + _short(key)
            if asset_id in rows:
                continue
            digest = _asset_sha(reference, roots)
            row = {
                "asset_id": asset_id,
                "book_id": book_id,
                "path": reference,
                "sha256": digest,
                "media_type": _media_type(reference),
                "source_unit_id": unit_id or None,
                "alt": alt,
            }
            rows[asset_id] = row
    return [rows[key] for key in sorted(rows)]


def build_assets(book_id: str) -> list[dict[str, Any]]:
    """Backwards-compatible empty table (writer now uses :func:`extract_assets`)."""
    return []
