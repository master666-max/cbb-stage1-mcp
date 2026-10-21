"""Neutral, dependency-free ID/SHA formulas for the whole extraction stage.

Every contract ID and hash is computed here.  Layer modules import these
functions (or keep a thin compatibility wrapper) instead of mirroring a formula.

Design constraints:

* **No project imports.**  This module only uses the standard library, so it can
  be imported from engines, box, work integrations and tools without cycles.
* **Byte compatibility.**  Existing formulas are preserved exactly for
  ``a:``/``e:``/``ev:``/``cl:`` and the other families listed in the plan;
  canonical choices for ``cue:``/``t:``/``mg:``/``pp:`` are documented in
  ``25-统一ID模块-20261019.md``.
* **Separators matter.**  Assertion IDs use default ``json.dumps`` separators;
  stable-id payloads use compact separators.  Do not unify them blindly.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "TYPE_ALIASES",
    "sha256_bytes",
    "sha256_text",
    "short_hash",
    "sha256_hex_text",
    "sha256_hex_bytes",
    "canonical_json",
    "identity_json",
    "merkle_root",
    "entity_type_slug",
    "mention_id",
    "entity_id",
    "assertion_id",
    "event_id",
    "cue_id",
    "claim_id",
    "timeline_id",
    "merge_id",
    "proposal_id",
    "unit_id",
    "summary_id",
    "hypothesis_id",
    "abduction_hypothesis_id",
    "community_id",
    "report_id",
    "raptor_cluster_id",
    "forecast_id",
    "style_id",
    "theme_id",
    "motivation_chain_id",
    "emotion_curve_id",
    "conflict_id",
    "quarantine_id",
    "citation_id",
    "stable_id",
    "invalid_id",
]

#: Chinese/AI-Reader type labels -> schema-safe ``[a-z_]+`` slugs.
TYPE_ALIASES: dict[str, str] = {
    "人物": "person",
    "角色": "person",
    "人": "person",
    "地点": "location",
    "场所": "location",
    "地理": "location",
    "组织": "organization",
    "势力": "faction",
    "门派": "faction",
    "家族": "faction",
    "机构": "organization",
    "物品": "item",
    "器物": "item",
    "道具": "item",
    "概念": "concept",
    "设定": "concept",
    "技能": "skill",
    "体系": "skill",
    "能力": "skill",
    "事件": "event",
    "生物": "creature",
    "怪物": "creature",
    "神祇": "deity",
    "神明": "deity",
    "线索": "clue",
    "伏笔": "clue",
    "未知": "unknown",
}

_BOOK_ID_RE = re.compile(r"^[a-z0-9_-]+$")


def sha256_bytes(data: bytes) -> str:
    """Return ``sha256:`` + hex digest for raw bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return ``sha256:`` + hex digest for UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def short_hash(text: str, length: int = 16) -> str:
    """Return the first ``length`` hex chars of SHA-256 (ids use 16)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sha256_hex_text(text: str, length: int = 64) -> str:
    """Bare SHA-256 hex for non-contract fingerprints (no ``sha256:`` prefix)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sha256_hex_bytes(data: bytes, length: int = 64) -> str:
    """Bare SHA-256 hex for raw bytes (file/artifact fingerprints)."""
    return hashlib.sha256(data).hexdigest()[:length]


def canonical_json(value: Any) -> str:
    """Compact canonical JSON (stable_id payloads use this)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def identity_json(value: Any) -> str:
    """Default-separator JSON (assertion IDs intentionally use this byte form)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def merkle_root(files: Mapping[str, str]) -> str:
    """Merkle-style root over ``path:sha256`` lines, sorted by path."""
    payload = "\n".join(f"{path}:{digest}" for path, digest in sorted(files.items()))
    return sha256_text(payload)


def entity_type_slug(value: str | None) -> str:
    """Map an arbitrary type label to a schema-safe ``[a-z_]+`` slug."""
    raw = str(value or "").strip()
    if not raw:
        return "unknown"
    candidate = raw.lower().replace("-", "_").replace(" ", "_")
    if candidate and all(ch.isascii() and (ch.isalnum() or ch == "_") for ch in candidate):
        return candidate
    return TYPE_ALIASES.get(raw, "unknown")


def mention_id(unit_id: str, start: int, end: int, surface: str) -> str:
    return f"m:{unit_id}:{int(start)}-{int(end)}:{short_hash(surface, 8)}"


def entity_id(book_id: str, entity_type: str, canonical_name: str) -> str:
    slug = entity_type_slug(entity_type)
    return f"e:{book_id}:{slug}:{short_hash(f'{slug}|{canonical_name}', 12)}"


def _quote_sha(value: str) -> str:
    text = str(value or "")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def assertion_id(
    book_id: str,
    predicate: str,
    roles: Mapping[str, Any],
    quote_sha256: str,
) -> str:
    key = "|".join(
        [
            book_id,
            predicate,
            identity_json(dict(roles)),
            _quote_sha(quote_sha256),
        ]
    )
    return "a:" + short_hash(key, 16)


def event_id(
    book_id: str,
    event_type: str,
    unit_id: str,
    participant_ids: Iterable[str],
    summary: str,
) -> str:
    key = "|".join(
        [book_id, event_type, unit_id, ",".join(sorted(participant_ids)), summary]
    )
    return "ev:" + short_hash(key, 16)


def cue_id(
    book_id: str,
    kind: str,
    subject: Mapping[str, Any],
    setup_unit: str | None,
) -> str:
    key = "|".join(
        [book_id, kind, identity_json(dict(subject)), str(setup_unit or "")]
    )
    return "cue:" + short_hash(key, 16)


def claim_id(statement_assertion_id: str, normalized_claim: str) -> str:
    return "cl:" + short_hash(f"{statement_assertion_id}|{normalized_claim}", 16)


def timeline_id(
    book_id: str,
    kind: str,
    left_kind: str,
    left_id: str,
    right_kind: str,
    right_id: str,
) -> str:
    key = f"{book_id}|{kind}|{left_kind}:{left_id}|{right_kind}:{right_id}"
    return "t:" + short_hash(key, 16)


def merge_id(
    book_id: str,
    kind: str,
    from_ids: Iterable[str],
    to_id: str,
) -> str:
    return "mg:" + short_hash(
        f"{book_id}|{kind}|{'|'.join(sorted(from_ids))}|{to_id}", 16
    )


def proposal_id(book_id: str, predicate: str, seen_in: Iterable[str]) -> str:
    return "pp:" + short_hash(
        f"{book_id}|{predicate}|{'|'.join(sorted(seen_in))}", 16
    )


def unit_id(book_id: str, volume: int, chapter: int, seq: int = 0) -> str:
    if not isinstance(book_id, str) or not _BOOK_ID_RE.match(book_id):
        raise ValueError(f"invalid book_id {book_id!r}")
    if not 1 <= int(volume) <= 99:
        raise ValueError(f"volume out of range 1-99: {volume}")
    if not 1 <= int(chapter) <= 999:
        raise ValueError(f"chapter out of range 1-999: {chapter}")
    if not 0 <= int(seq) <= 99:
        raise ValueError(f"chunk seq out of range 0-99: {seq}")
    base = f"u:{book_id}:v{int(volume):02d}:c{int(chapter):03d}"
    return base if seq == 0 else f"{base}:s{int(seq):02d}"


def summary_id(book_id: str, scope: str, target: str) -> str:
    return "sm:" + short_hash(f"{book_id}|{scope}|{target}", 16)


def hypothesis_id(book_id: str, scope: str, claim: str) -> str:
    return "hy:" + short_hash(f"{book_id}|{scope}|{claim}", 16)


def abduction_hypothesis_id(
    book_id: str, scope: str, claim: str, premises: Sequence[str]
) -> str:
    joined = "\x1f".join(premises)
    return "hy:" + short_hash(f"{book_id}|{scope}|{claim}|{joined}", 16)


def community_id(
    book_id: str,
    level: int,
    member_entities: Iterable[str],
    *,
    volume: int | None = None,
) -> str:
    suffix = "" if volume is None else f"|v{volume}"
    return "cm:" + short_hash(
        f"{book_id}|{int(level)}{suffix}|" + "|".join(sorted(member_entities)), 16
    )


def report_id(community_id_value: str) -> str:
    return "cr:" + short_hash(community_id_value, 16)


def raptor_cluster_id(
    book_id: str, volume: int, level: int, member_units: Iterable[str]
) -> str:
    return "rt:" + short_hash(
        f"{book_id}|v{int(volume)}|L{int(level)}|" + "|".join(sorted(member_units)),
        16,
    )


def forecast_id(book_id: str, unit_id_value: str) -> str:
    return "fc:" + short_hash(f"{book_id}|{unit_id_value}", 16)


def style_id(book_id: str, unit_id_value: str) -> str:
    """One stylometry row per unit (§5.20)."""
    return "st:" + short_hash(f"{book_id}|{unit_id_value}", 16)


def theme_id(book_id: str, motif: str) -> str:
    """One row per motif; the units carrying it live inside the row (§5.20)."""
    return "th:" + short_hash(f"{book_id}|{motif}", 16)


def motivation_chain_id(book_id: str, character: str) -> str:
    """One chain per character (§5.20)."""
    return "mc:" + short_hash(f"{book_id}|{character}", 16)


def emotion_curve_id(book_id: str, unit_id_value: str, speaker: str) -> str:
    """One curve per (unit, speaker); ``speaker`` may be the empty string for the
    narration voice, which is a legitimate lane of its own (§5.20)."""
    return "ec:" + short_hash(f"{book_id}|{unit_id_value}|{speaker}", 16)


def stable_id(prefix: str, payload: Any, length: int = 16) -> str:
    if not prefix.endswith(":"):
        prefix = prefix + ":"
    if length < 16 or length > 64:
        raise ValueError("id hex length must be within [16, 64]")
    return prefix + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def conflict_id(book_id: str, kind: str, left: str, right: str, detail: str) -> str:
    del book_id  # preserved in the record, not in the id payload
    return stable_id("c:", {"kind": kind, "left": left, "right": right, "detail": detail})


def quarantine_id(payload: Any, kind: str | None = None) -> str:
    """Preserve the old ``stable_id("q:", payload)`` form when kind is omitted."""
    body = payload if kind is None else {"kind": kind, "payload": payload}
    return stable_id("q:", body)


def citation_id(unit_id_value: str, start: int, end: int, quote: str) -> str:
    key = f"{unit_id_value}|{int(start)}|{int(end)}|{sha256_text(quote)}"
    return "c:" + short_hash(key, 16)


def invalid_id(prefix: str, payload: Any) -> str:
    if not prefix.endswith(":"):
        prefix = prefix + ":"
    return prefix + "invalid:" + short_hash(canonical_json(payload), 16)
