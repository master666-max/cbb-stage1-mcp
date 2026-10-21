"""VeGraph: claim -> triples -> entity resolution -> per-sub-assertion verification.

Design refs: ``05-资料库前沿检索与吸收-20261018.md`` §3.1/§3.3-3.6 (B9) and
``01-箱规格-v2.md`` §5.19/§5.16.

Order matters (VeGraph): **resolve entities first, then judge stances.** Surfaces
that do not resolve are recorded and *block* stance judgement -- the module never
lets an unresolved entity reach the verifier and never guesses one. Entity
resolution reuses :func:`cbb_stage1.claims.qa.resolve_entities` (single source).

Boundary: no LLM call. Without a model there are no triples and the only honest
verdict is ``insufficient_evidence``. No file write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from cbb_stage1 import ids
from cbb_stage1.claims.qa import EntityResolution, resolve_entities
from cbb_stage1.claims.verdicts import verdict_from_stances

__all__ = [
    "STANCES",
    "Triple",
    "VeGraphLLM",
    "VeGraphResult",
    "VeGraphRuleError",
    "vegaph_verify",
]

STANCES: tuple[str, ...] = ("support", "refute", "irrelevant", "unanswerable")


class VeGraphRuleError(ValueError):
    """The VeGraph model returned something outside the contract."""


class VeGraphLLM(Protocol):
    """The two model capabilities VeGraph needs."""

    def triples(self, claim: str) -> Sequence[Mapping[str, Any]]:
        """Turn the claim into ``{subject, predicate, object}`` triples."""

    def verify_subassertion(self, triple: Mapping[str, Any]) -> str:
        """Return the stance of one triple: support/refute/irrelevant/unanswerable."""


@dataclass(frozen=True)
class Triple:
    """One claim triple (also the sub-assertion unit VeGraph verifies)."""

    subject: str
    predicate: str
    obj: str

    def __post_init__(self) -> None:
        for name, value in (("subject", self.subject), ("predicate", self.predicate), ("object", self.obj)):
            if not isinstance(value, str) or not value.strip():
                raise VeGraphRuleError(f"triple {name} must be a non-empty string, got {value!r}")

    def as_record(self) -> dict[str, str]:
        return {"subject": self.subject.strip(), "predicate": self.predicate.strip(),
                "object": self.obj.strip()}

    @property
    def surfaces(self) -> tuple[str, str]:
        return (self.subject.strip(), self.obj.strip())

    @property
    def triple_id(self) -> str:
        key = "|".join((self.subject.strip(), self.predicate.strip(), self.obj.strip()))
        return f"tr:{ids.short_hash(key)}"


def _clean_triples(raw: Sequence[Any]) -> tuple[Triple, ...]:
    triples: list[Triple] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise VeGraphRuleError(f"triples() must return mappings, got {type(item).__name__}")
        missing = [key for key in ("subject", "predicate", "object") if not item.get(key)]
        if missing:
            raise VeGraphRuleError(f"triple missing {missing}: {dict(item)!r}")
        triples.append(Triple(str(item["subject"]), str(item["predicate"]), str(item["object"])))
    return tuple(triples)


@dataclass(frozen=True)
class VeGraphResult:
    """VeGraph verification of one claim."""

    claim: str
    book_id: str
    triples: tuple[Triple, ...]
    sub_stances: tuple[tuple[Triple, str], ...]
    resolution: EntityResolution
    unresolved_entities: tuple[str, ...]
    verdict: str
    degenerate: bool
    notes: tuple[str, ...] = ()

    def sub_assertion_records(self) -> list[dict[str, Any]]:
        """``[{subject,predicate,object,stance}]`` -- verifier material, not box rows."""
        return [dict(triple.as_record(), stance=stance) for triple, stance in self.sub_stances]


def vegraph_verify(
    claim: str,
    *,
    book_id: str = "",
    surfaces: Sequence[str] = (),
    name_to_id: Mapping[str, str],
    alias_map: Mapping[str, str] | None = None,
    llm: VeGraphLLM | None = None,
) -> VeGraphResult:
    """Resolve claim entities first, then verify every triple as a sub-assertion.

    Without ``llm`` the result is degenerate: no triples, no stances,
    ``verdict="insufficient_evidence"``. Unresolved surfaces block stance
    judgement (the verifier is not even called).
    """
    if not isinstance(claim, str) or not claim.strip():
        raise VeGraphRuleError("claim must be a non-empty string")
    claim = claim.strip()
    empty_resolution = EntityResolution(resolved={}, unresolved=())

    if llm is None:
        return VeGraphResult(
            claim=claim,
            book_id=book_id,
            triples=(),
            sub_stances=(),
            resolution=empty_resolution,
            unresolved_entities=(),
            verdict="insufficient_evidence",
            degenerate=True,
            notes=("no_llm=insufficient_evidence",),
        )

    triples = _clean_triples(list(llm.triples(claim)))
    if not triples:
        return VeGraphResult(
            claim=claim,
            book_id=book_id,
            triples=(),
            sub_stances=(),
            resolution=empty_resolution,
            unresolved_entities=(),
            verdict="insufficient_evidence",
            degenerate=False,
            notes=("no_triples",),
        )

    all_surfaces: list[str] = []
    for surface in list(surfaces) + [name for triple in triples for name in triple.surfaces]:
        surface = (surface or "").strip()
        if surface and surface not in all_surfaces:
            all_surfaces.append(surface)
    resolution = resolve_entities(all_surfaces, name_to_id=name_to_id, alias_map=alias_map)

    if resolution.unresolved:
        # VeGraph order: unresolved surfaces block stance judgement entirely.
        return VeGraphResult(
            claim=claim,
            book_id=book_id,
            triples=triples,
            sub_stances=(),
            resolution=resolution,
            unresolved_entities=resolution.unresolved,
            verdict="insufficient_evidence",
            degenerate=False,
            notes=("unresolved_entities_block_stance",),
        )

    stances: list[tuple[Triple, str]] = []
    for triple in triples:
        stance = llm.verify_subassertion(triple.as_record())
        if stance not in STANCES:
            raise VeGraphRuleError(f"sub-assertion stance {stance!r} not in {STANCES}")
        stances.append((triple, stance))

    verdict = verdict_from_stances(stance for _triple, stance in stances)
    return VeGraphResult(
        claim=claim,
        book_id=book_id,
        triples=triples,
        sub_stances=tuple(stances),
        resolution=resolution,
        unresolved_entities=(),
        verdict=verdict,
        degenerate=False,
    )
