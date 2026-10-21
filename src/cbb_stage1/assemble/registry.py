"""Predicate registry: the single source of truth for predicates, roles, libraries.

Design contract (``01-箱规格-v2.md`` §5.6 / §5.10, ``02-引擎契约.md`` §4.1,
``00-架构总设计.md`` §4.2 and §4.9):

* the registry, not the extracted data, defines a predicate's ``library`` (one
  of the six: 实体/关系/事件/埋线/设定/时序), its role names, which roles are
  required and which are repeatable;
* roles whose keys are not the registered ones, or whose values do not match
  the role type, are a ``role_mismatch`` quarantine, never a silent pass;
* an unregistered predicate is *not* auto-registered: it goes to
  ``proposed_predicates.jsonl`` (``status=proposed``) and only a human-approved
  add may call :meth:`PredicateRegistry.register`.

The registry is the machine form of ``lexicon/predicates.json``; it is loaded
from that file (or an equivalent mapping) and its digest is recorded in the
manifest as ``predicates_sha256``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cbb_stage1 import ids

__all__ = [
    "LIBRARIES",
    "LIBRARY_ALIASES",
    "PredicateProposal",
    "PredicateRegistry",
    "PredicateSpec",
    "RegistryError",
    "RoleCheck",
    "RoleProblem",
    "RoleSpec",
    "SEED_LEXICON",
    "normalize_library",
    "propose_predicate",
    "seed_registry",
]

Library = Literal["实体", "关系", "事件", "埋线", "设定", "时序"]
LIBRARIES: tuple[str, ...] = ("实体", "关系", "事件", "埋线", "设定", "时序")

#: The W2 normalizer and some early design rows use English library names; the
#: assembler accepts them as aliases but always writes the Chinese canonical
#: value from the registry ("不允许数据自定", §5.6).
LIBRARY_ALIASES: dict[str, str] = {
    "entity": "实体",
    "relation": "关系",
    "event": "事件",
    "cue": "埋线",
    "foreshadow": "埋线",
    "setting": "设定",
    "world": "设定",
    "temporal": "时序",
    "timeline": "时序",
}


class RegistryError(ValueError):
    """Lexicon payload or registry transition is invalid."""


def normalize_library(value: str) -> str:
    """Canonicalize a library name; raise :class:`RegistryError` if unknown."""
    text = (value or "").strip()
    canonical = LIBRARY_ALIASES.get(text.lower(), text)
    if canonical not in LIBRARIES:
        raise RegistryError(
            f"library {value!r} is not one of the six libraries {LIBRARIES}"
        )
    return canonical


# ── Lexicon models ────────────────────────────────────────────────


class RoleSpec(BaseModel):
    """One role slot of a predicate (``name`` must be the roles key)."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    type: Literal["entity", "literal"] = "entity"
    required: bool = True
    repeatable: bool = False


class PredicateSpec(BaseModel):
    """One registered predicate: library + role schema (from the lexicon)."""

    model_config = ConfigDict(extra="ignore")

    pred: str = Field(min_length=1)
    library: Library
    roles: list[RoleSpec] = Field(min_length=1)
    note: str | None = None

    @field_validator("library", mode="before")
    @classmethod
    def _normalize_library(cls, value: Any) -> str:
        return normalize_library(str(value))

    @field_validator("roles")
    @classmethod
    def _validate_roles(cls, roles: list[RoleSpec]) -> list[RoleSpec]:
        names = [role.name for role in roles]
        if len(set(names)) != len(names):
            raise ValueError("role names must be unique within a predicate")
        return roles

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles)

    @property
    def required_roles(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles if role.required)

    def role(self, name: str) -> RoleSpec | None:
        return next((role for role in self.roles if role.name == name), None)


class PredicateProposal(BaseModel):
    """``knowledge/proposed_predicates.jsonl`` row (§5.10) -- never registered."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    book_id: str
    predicate: str
    seen_in: list[str] = Field(default_factory=list)
    count: int = 1
    role_shape_examples: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["proposed"] = "proposed"
    decided_by: str | None = None
    first_seen: str | None = None


def _proposal_id(book_id: str, predicate: str, seen_in: Sequence[str]) -> str:
    return ids.proposal_id(book_id, predicate, seen_in)


def propose_predicate(
    predicate: str,
    *,
    book_id: str,
    seen_in: Iterable[str] = (),
    count: int = 1,
    role_shape_examples: Iterable[Mapping[str, Any]] = (),
    first_seen: str | None = None,
) -> PredicateProposal:
    """Build a proposal row for an unregistered predicate (no auto-register)."""
    seen = [str(unit) for unit in seen_in]
    return PredicateProposal(
        proposal_id=_proposal_id(book_id, predicate, seen),
        book_id=book_id,
        predicate=predicate,
        seen_in=seen,
        count=count,
        role_shape_examples=[dict(example) for example in role_shape_examples],
        status="proposed",
        decided_by=None,
        first_seen=first_seen,
    )


# ── Role checking ────────────────────────────────────────────────


@dataclass(frozen=True)
class RoleProblem:
    """One reason a ``roles`` mapping does not match the registry."""

    kind: Literal["roles_shape", "missing_role", "unknown_role", "role_type"]
    role: str
    detail: str


@dataclass(frozen=True)
class RoleCheck:
    """Result of :meth:`PredicateRegistry.check_roles`."""

    ok: bool
    predicate: str
    library: str | None
    problems: tuple[RoleProblem, ...]

    @property
    def reason(self) -> str:
        """Quarantine reason for the closed set (``role_mismatch``) or empty."""
        return "" if self.ok else "role_mismatch"

    def detail(self) -> str:
        if self.ok:
            return ""
        return "; ".join(f"{problem.kind}({problem.role}): {problem.detail}" for problem in self.problems)


def _role_value_problems(role: RoleSpec, value: Any) -> list[RoleProblem]:
    problems: list[RoleProblem] = []
    values = value if isinstance(value, (list, tuple)) else [value]
    if isinstance(value, (list, tuple)):
        if not role.repeatable:
            problems.append(
                RoleProblem(
                    "role_type",
                    role.name,
                    "list value given for a non-repeatable role",
                )
            )
            return problems
        if not values:
            problems.append(RoleProblem("role_type", role.name, "empty list for required role"))
            return problems

    for item in values:
        if role.type == "entity":
            if not (isinstance(item, str) and item.startswith("e:")):
                problems.append(
                    RoleProblem(
                        "role_type",
                        role.name,
                        f"entity role expects an 'e:' id, got {item!r}",
                    )
                )
        else:  # literal
            # A bare string is NOT accepted here on purpose. The value shape is
            # owned by the emitting end (`predicate_map.literal`); tolerating a
            # bare string at this layer too would give the contract two safety
            # nets and make the full-chain gate decorative — measured: with both
            # layers tolerant, no single-point mutation could turn the gate red.
            # `box/records.py` accepting a bare string is a *reader* being
            # lenient, which is fine; a *writer* contract must be strict.
            if not isinstance(item, Mapping) or not isinstance(item.get("literal"), str):
                problems.append(
                    RoleProblem(
                        "role_type",
                        role.name,
                        "literal role expects {'literal': str, ...}, "
                        f"got {item!r}",
                    )
                )
                continue
            datatype = item.get("datatype")
            if datatype is not None and datatype not in ("string", "number", "time", "boolean"):
                problems.append(
                    RoleProblem(
                        "role_type",
                        role.name,
                        f"unknown literal datatype {datatype!r}",
                    )
                )
    return problems


# ── The registry ─────────────────────────────────────────────────


class PredicateRegistry:
    """Versioned predicate registry loaded from ``lexicon/predicates.json``."""

    def __init__(
        self,
        predicates: Iterable[PredicateSpec],
        *,
        contract: str = "cbb-lexicon/1",
        version: str | None = None,
    ) -> None:
        self.contract = contract
        self.version = version
        self._predicates: dict[str, PredicateSpec] = {}
        for spec in predicates:
            self._insert(spec)

    def _insert(self, spec: PredicateSpec) -> None:
        if spec.pred in self._predicates:
            raise RegistryError(f"duplicate predicate {spec.pred!r} in lexicon")
        self._predicates[spec.pred] = spec

    # -- loading -------------------------------------------------------
    @classmethod
    def from_lexicon(
        cls,
        data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        version: str | None = None,
    ) -> "PredicateRegistry":
        """Accept the ``lexicon/predicates.json`` object or a bare predicate list."""
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
            payload: Mapping[str, Any] = {"predicates": list(data)}
        else:
            payload = dict(data)  # type: ignore[arg-type]
        raw_predicates = payload.get("predicates")
        if not isinstance(raw_predicates, Sequence) or isinstance(raw_predicates, (str, bytes)):
            raise RegistryError("lexicon payload must contain a 'predicates' list")
        specs = [PredicateSpec.model_validate(item) for item in raw_predicates]
        contract = str(payload.get("contract") or "cbb-lexicon/1")
        return cls(
            specs,
            contract=contract,
            version=version or payload.get("version"),
        )

    @classmethod
    def from_path(cls, path: str | Path, *, version: str | None = None) -> "PredicateRegistry":
        target = Path(path)
        if not target.exists():
            raise RegistryError(f"lexicon file not found: {target}")
        with target.open("r", encoding="utf-8") as fh:
            return cls.from_lexicon(json.load(fh), version=version)

    # -- introspection -------------------------------------------------
    def __len__(self) -> int:
        return len(self._predicates)

    def __contains__(self, predicate: str) -> bool:
        return predicate in self._predicates

    @property
    def predicates(self) -> tuple[PredicateSpec, ...]:
        return tuple(self._predicates.values())

    def get(self, predicate: str) -> PredicateSpec | None:
        return self._predicates.get(predicate)

    def library_of(self, predicate: str) -> str | None:
        spec = self._predicates.get(predicate)
        return spec.library if spec else None

    def libraries_in_use(self) -> tuple[str, ...]:
        return tuple(sorted({spec.library for spec in self._predicates.values()}))

    def to_lexicon(self) -> dict[str, Any]:
        """Serialize back to the lexicon shape (used for the digest)."""
        return {
            "contract": self.contract,
            "version": self.version,
            "predicates": [
                spec.model_dump(exclude_none=True) for spec in self._predicates.values()
            ],
        }

    def digest(self) -> str:
        """Canonical ``sha256:...`` of the registry (manifest ``predicates_sha256``)."""
        return ids.sha256_text(ids.canonical_json(self.to_lexicon()))

    # -- mutations (human-approved only) -------------------------------
    def register(self, spec: PredicateSpec, *, approved_by: str | None = None) -> PredicateSpec:
        """Add a predicate; callers must come from a human-approved decision."""
        if not approved_by:
            raise RegistryError(
                "register() requires approved_by (human gate); "
                "unregistered predicates go to propose_predicate()"
            )
        if spec.pred in self._predicates:
            raise RegistryError(f"predicate {spec.pred!r} is already registered")
        self._insert(spec)
        return spec

    # -- validation ----------------------------------------------------
    def check_roles(self, predicate: str, roles: Mapping[str, Any] | None) -> RoleCheck:
        """Validate a ``roles`` mapping against the predicate's role schema."""
        spec = self._predicates.get(predicate)
        if spec is None:
            return RoleCheck(
                ok=False,
                predicate=predicate,
                library=None,
                problems=(
                    RoleProblem(
                        "unknown_role",
                        "",
                        f"predicate {predicate!r} is not registered",
                    ),
                ),
            )
        if not isinstance(roles, Mapping) or not roles:
            return RoleCheck(
                ok=False,
                predicate=predicate,
                library=spec.library,
                problems=(RoleProblem("roles_shape", "", "roles must be a non-empty mapping"),),
            )

        problems: list[RoleProblem] = []
        for role_name in roles:
            if spec.role(role_name) is None:
                problems.append(
                    RoleProblem(
                        "unknown_role",
                        role_name,
                        f"predicate {predicate!r} expects {list(spec.role_names)}",
                    )
                )
        for role in spec.roles:
            if role.name not in roles:
                if role.required:
                    problems.append(
                        RoleProblem("missing_role", role.name, "required role is missing")
                    )
                continue
            problems.extend(_role_value_problems(role, roles[role.name]))

        return RoleCheck(
            ok=not problems,
            predicate=predicate,
            library=spec.library,
            problems=tuple(problems),
        )

    def role_shape(self, predicate: str) -> dict[str, Any] | None:
        """Compact role schema used in proposal examples / error messages."""
        spec = self._predicates.get(predicate)
        if spec is None:
            return None
        return {
            "library": spec.library,
            "roles": [
                {
                    "name": role.name,
                    "type": role.type,
                    "required": role.required,
                    "repeatable": role.repeatable,
                }
                for role in spec.roles
            ],
        }


#: Minimal seed matching the box builder's default six-predicate registry
#: (``box/lexicon.py`` ``default_lexicon``).  It is **not** the human-owned
#: production profile: the pipeline must load the real
#: ``lexicon/predicates.json`` and record its digest.  Kept for smoke tests and
#: cold-start wiring; keeping it identical to the box default avoids a work/box
#: disagreement on which predicates are registered.
SEED_LEXICON: dict[str, Any] = {
    "contract": "cbb-lexicon/1",
    "predicates": [
        {
            "pred": "性质",
            "library": "设定",
            "roles": [
                {"name": "主体", "type": "entity", "required": True},
                {"name": "值", "type": "literal", "required": True},
            ],
        },
        {
            "pred": "别名",
            "library": "实体",
            "roles": [
                {"name": "主体", "type": "entity", "required": True},
                {"name": "值", "type": "literal", "required": True},
            ],
        },
        {
            "pred": "位于",
            "library": "关系",
            "roles": [
                {"name": "主体", "type": "entity", "required": True},
                {"name": "值", "type": "literal", "required": True},
            ],
        },
        {
            "pred": "敌对",
            "library": "关系",
            "roles": [
                {"name": "施事", "type": "entity", "required": True},
                {"name": "受事", "type": "entity", "required": True},
            ],
        },
        {
            "pred": "亲属",
            "library": "关系",
            "roles": [
                {"name": "施事", "type": "entity", "required": True},
                {"name": "受事", "type": "entity", "required": True},
            ],
        },
        {
            "pred": "参与",
            "library": "事件",
            "roles": [
                {"name": "事件", "type": "entity", "required": True},
                {"name": "参与者", "type": "entity", "required": True, "repeatable": True},
            ],
        },
    ],
}


def seed_registry() -> PredicateRegistry:
    """Registry built from :data:`SEED_LEXICON` (smoke / cold start only)."""
    return PredicateRegistry.from_lexicon(SEED_LEXICON)
