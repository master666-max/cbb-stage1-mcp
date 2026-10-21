"""Predicate/type/genre lexicon for the box (v2.2 six-library contract).

The registry is the **only** source of truth for ``predicate -> library`` and
role names (box spec §5.6 / invariant I3-I4). Data never defines its own
library or roles: records that use an unregistered predicate go to
``knowledge/proposed_predicates.jsonl`` instead of ``assertions``.

A caller may supply a pre-approved lexicon directory with
``genre.json`` / ``predicates.json`` / ``types.json`` (same shape as
``example/lexicon``). Without one, :func:`default_lexicon` provides the
minimal six-predicate stage-1 registry used by mock/manual runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from cbb_stage1.ids import TYPE_ALIASES, entity_type_slug

__all__ = [
    "SIX_LIBRARIES",
    "TYPE_ALIASES",
    "PredicateSpec",
    "RoleSpec",
    "Lexicon",
    "default_lexicon",
    "load_lexicon",
    "entity_type_slug",
]

#: Closed six-library set from box spec §5.6.
SIX_LIBRARIES = ("实体", "关系", "事件", "埋线", "设定", "时序")

# ``TYPE_ALIASES`` and ``entity_type_slug`` are re-exported from
# :mod:`cbb_stage1.ids`; ``DEFAULT_TYPES`` below is box-specific registry data.

DEFAULT_TYPES: dict[str, dict[str, str]] = {
    "person": {"zh": "人物"},
    "organization": {"zh": "组织"},
    "faction": {"zh": "势力"},
    "skill": {"zh": "技能/体系"},
    "location": {"zh": "地点"},
    "item": {"zh": "器物"},
    "concept": {"zh": "概念"},
    "event": {"zh": "事件"},
    "creature": {"zh": "生物"},
    "deity": {"zh": "神祇"},
    "clue": {"zh": "线索"},
    "unknown": {"zh": "未知"},
}


@dataclass(frozen=True)
class RoleSpec:
    """One declared role of a predicate."""

    name: str
    type: str = "entity"
    required: bool = False
    repeatable: bool = False

    def __post_init__(self) -> None:
        if self.type not in ("entity", "literal"):
            raise ValueError(f"role {self.name!r}: type must be entity|literal")

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.required:
            record["required"] = True
        if self.repeatable:
            record["repeatable"] = True
        return record


@dataclass(frozen=True)
class PredicateSpec:
    """One registered predicate (``library`` + role schema)."""

    pred: str
    library: str
    roles: tuple[RoleSpec, ...]

    def __post_init__(self) -> None:
        if self.library not in SIX_LIBRARIES:
            raise ValueError(
                f"predicate {self.pred!r}: library {self.library!r} is not one of {SIX_LIBRARIES}"
            )
        if not self.pred:
            raise ValueError("predicate name is required")
        names = [role.name for role in self.roles]
        if len(names) != len(set(names)):
            raise ValueError(f"predicate {self.pred!r}: duplicate role names")

    @property
    def role_by_name(self) -> dict[str, RoleSpec]:
        return {role.name: role for role in self.roles}

    @property
    def required_roles(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles if role.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pred": self.pred,
            "library": self.library,
            "roles": [role.to_dict() for role in self.roles],
        }


@dataclass(frozen=True)
class Lexicon:
    """Genre profile + closed type set + predicate registry."""

    genre_id: str
    genre: dict[str, Any]
    types: dict[str, dict[str, Any]]
    predicates: dict[str, PredicateSpec]

    def predicate(self, name: str) -> PredicateSpec | None:
        return self.predicates.get(name)

    def genre_file(self) -> dict[str, Any]:
        return dict(self.genre)

    def types_file(self) -> dict[str, Any]:
        return {"contract": "cbb-lexicon/1", "types": dict(self.types)}

    def predicates_file(self) -> dict[str, Any]:
        return {
            "contract": "cbb-lexicon/1",
            "predicates": [
                self.predicates[name].to_dict() for name in sorted(self.predicates)
            ],
        }


def default_lexicon(genre_id: str = "light-novel-zh") -> Lexicon:
    """Stage-1 default registry (mirrors the minimal ``example`` box)."""
    genre = {
        "contract": "cbb-lexicon/1",
        "id": genre_id,
        "types": list(DEFAULT_TYPES),
        "alias_rules": [
            "泛称不是别名",
            "亲属称谓不是别名",
            "官职不是别名",
            "别名必须有文本依据",
        ],
        "narrative_modes": ["present", "flashback", "prophecy", "mixed"],
    }
    predicates = {
        "性质": PredicateSpec(
            "性质",
            "设定",
            (
                RoleSpec("主体", "entity", required=True),
                RoleSpec("值", "literal", required=True),
            ),
        ),
        "别名": PredicateSpec(
            "别名",
            "实体",
            (
                RoleSpec("主体", "entity", required=True),
                RoleSpec("值", "literal", required=True),
            ),
        ),
        "位于": PredicateSpec(
            "位于",
            "关系",
            (
                RoleSpec("主体", "entity", required=True),
                RoleSpec("值", "literal", required=True),
            ),
        ),
        "敌对": PredicateSpec(
            "敌对",
            "关系",
            (
                RoleSpec("施事", "entity", required=True),
                RoleSpec("受事", "entity", required=True),
            ),
        ),
        "亲属": PredicateSpec(
            "亲属",
            "关系",
            (
                RoleSpec("施事", "entity", required=True),
                RoleSpec("受事", "entity", required=True),
            ),
        ),
        "参与": PredicateSpec(
            "参与",
            "事件",
            (
                RoleSpec("事件", "entity", required=True),
                RoleSpec("参与者", "entity", required=True, repeatable=True),
            ),
        ),
    }
    return Lexicon(
        genre_id=genre_id,
        genre=genre,
        types={name: dict(value) for name, value in DEFAULT_TYPES.items()},
        predicates=predicates,
    )


def _read_lexicon_file(directory: Path, name: str) -> dict[str, Any]:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"lexicon file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    return value


def _parse_predicates(raw: Mapping[str, Any]) -> dict[str, PredicateSpec]:
    rows = raw.get("predicates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("predicates.json: 'predicates' must be a non-empty array")
    registry: dict[str, PredicateSpec] = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"predicates.json: predicate #{index} must be an object")
        pred = str(row.get("pred") or "").strip()
        library = str(row.get("library") or "").strip()
        role_rows = row.get("roles")
        if not isinstance(role_rows, list) or not role_rows:
            raise ValueError(f"predicates.json: predicate {pred!r} needs roles")
        roles: list[RoleSpec] = []
        for role in role_rows:
            if not isinstance(role, Mapping):
                raise ValueError(f"predicates.json: {pred!r} role must be an object")
            roles.append(
                RoleSpec(
                    name=str(role.get("name") or "").strip(),
                    type=str(role.get("type") or "entity"),
                    required=bool(role.get("required", False)),
                    repeatable=bool(role.get("repeatable", False)),
                )
            )
        spec = PredicateSpec(pred, library, tuple(roles))
        if pred in registry:
            raise ValueError(f"predicates.json: duplicate predicate {pred!r}")
        registry[pred] = spec
    return registry


def load_lexicon(lexicon_dir: str | Path | None = None) -> Lexicon:
    """Load a pre-approved lexicon directory, or the built-in default.

    Args:
        lexicon_dir: Directory containing ``genre.json``, ``predicates.json``
            and ``types.json``. ``None`` selects :func:`default_lexicon`.

    Raises:
        FileNotFoundError/ValueError: the directory is incomplete or malformed.
            The caller must fix the lexicon (no silent fallback to defaults for
            an explicitly supplied directory).
    """
    if lexicon_dir is None:
        return default_lexicon()
    directory = Path(lexicon_dir)
    genre = _read_lexicon_file(directory, "genre.json")
    types_raw = _read_lexicon_file(directory, "types.json")
    predicates = _parse_predicates(_read_lexicon_file(directory, "predicates.json"))
    types = types_raw.get("types")
    if not isinstance(types, dict) or not types:
        raise ValueError("types.json: 'types' must be a non-empty object")
    genre_id = str(genre.get("id") or "light-novel-zh")
    return Lexicon(
        genre_id=genre_id,
        genre=genre,
        types={str(name): dict(value) for name, value in types.items()},
        predicates=predicates,
    )


def lexicon_from_work(
    work_root: str | Path, *, fallback: bool = True
) -> tuple[Lexicon, Path | None]:
    """Use ``work_root/lexicon`` when present, else the built-in default."""
    directory = Path(work_root) / "lexicon"
    if directory.is_dir():
        return load_lexicon(directory), directory
    if not fallback:
        raise FileNotFoundError(f"lexicon directory not found: {directory}")
    return default_lexicon(), None


def role_names(spec: PredicateSpec) -> tuple[str, ...]:
    """Convenience: declared role names in registry order."""
    return tuple(role.name for role in spec.roles)


def iter_libraries() -> Iterable[str]:
    """Iterate the closed six-library set."""
    return iter(SIX_LIBRARIES)
