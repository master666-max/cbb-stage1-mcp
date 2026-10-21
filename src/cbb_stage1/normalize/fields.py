"""Normalize LLM JSON payloads before Pydantic validation.

Ported from B06-AI-Reader-V2 `chapter_fact_extractor._normalize_field_names`.
LLMs occasionally use shorter field names (``name`` instead of ``item_name``)
or put non-dict items inside array fields; fix those in place so a single
validation layer (Pydantic) is the only schema gate.
"""

from __future__ import annotations

from typing import Any

# Array fields that must contain dicts.
_DICT_ARRAY_FIELDS: tuple[str, ...] = (
    "characters",
    "relationships",
    "locations",
    "item_events",
    "org_events",
    "events",
    "spatial_relationships",
    "new_concepts",
    "world_declarations",
)

# (array field, expected key, accepted alias key)
_FIELD_ALIASES: tuple[tuple[str, str, str], ...] = (
    ("item_events", "item_name", "name"),
    ("item_events", "item_type", "type"),
    ("org_events", "org_name", "name"),
    ("org_events", "org_type", "type"),
)


def normalize_llm_payload(data: Any) -> dict[str, Any]:
    """Return a shallow-cleaned copy of a raw ChapterFact payload.

    * non-dict members are dropped from array fields,
    * known short field names are renamed to their schema names,
    * a top-level list is unwrapped to its first dict element (some models
      answer ``[{...}]`` for an object schema).

    Values inside dict members are mutated on the returned structure only; the
    input dict is copied one level down so callers can keep the raw payload.
    """
    if isinstance(data, list):
        dict_items = [item for item in data if isinstance(item, dict)]
        if not dict_items:
            raise ValueError("LLM returned a list with no dict elements")
        data = dict_items[0]
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")

    cleaned: dict[str, Any] = dict(data)

    for field in _DICT_ARRAY_FIELDS:
        value = cleaned.get(field)
        if isinstance(value, list):
            cleaned[field] = [item for item in value if isinstance(item, dict)]

    for field, expected_key, alias_key in _FIELD_ALIASES:
        for item in cleaned.get(field, []) or []:
            if expected_key not in item and alias_key in item:
                item[expected_key] = item.pop(alias_key)

    return cleaned
