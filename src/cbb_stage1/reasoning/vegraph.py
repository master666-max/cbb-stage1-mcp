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

#: The paper's iteration bound for the clarification stage (B9: "up to k=5 rounds").
#:
#: A bound, not the rule: the loop stops as soon as a round produces nothing new, which is what makes
#: repeated clarification cheap. Kept at the paper's value so a run that hits it is comparable.
DEFAULT_MAX_ROUNDS = 5


class VeGraphRuleError(ValueError):
    """The VeGraph model returned something outside the contract."""


class VeGraphLLM(Protocol):
    """The two model capabilities VeGraph needs."""

    def triples(self, claim: str) -> Sequence[Mapping[str, Any]]:
        """Turn the claim into ``{subject, predicate, object}`` triples."""

    def verify_subassertion(self, triple: Mapping[str, Any]) -> str:
        """Return the stance of one triple: support/refute/irrelevant/unanswerable."""

    #: Optional (B9 stage 2): ask a clarifying question about a sub-assertion that could not be
    #: judged, and return the answer's rationale for the next round. Declared here as documentation;
    #: the loop probes for it with `getattr` so a model without it still works.
    #:
    #: def clarify(self, triple: Mapping[str, Any], question: str) -> str: ...


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
    #: Clarification rounds actually used. 1 means the single-pass path ran (no clarifying model).
    rounds: int = 1

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
    max_rounds: int = DEFAULT_MAX_ROUNDS,
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

    # B9 stage 2: clarify -> feed the rationale back -> re-judge, bounded by k.
    #
    # A single pass over the triples is a dictionary hit with a model in the middle: a sub-assertion
    # that comes back `unanswerable` stays unanswerable because nothing asks what is missing.
    clarify = getattr(llm, "clarify", None)
    rationales: dict[int, str] = {}
    notes: list[str] = []
    rounds = 0
    stances: list[tuple[Triple, str]] = []

    for round_no in range(1, int(max_rounds or 1) + 1):
        rounds = round_no
        stances = []
        for triple in triples:
            record = dict(triple.as_record())
            rationale = rationales.get(id(triple), "")
            if rationale:
                record["rationale"] = rationale
            stance = llm.verify_subassertion(record)
            if stance not in STANCES:
                raise VeGraphRuleError(f"sub-assertion stance {stance!r} not in {STANCES}")
            stances.append((triple, stance))

        pending = [triple for triple, stance in stances if stance == "unanswerable"]
        if not pending:
            break
        if not callable(clarify):
            # No clarifying capability: say so rather than looping pointlessly.
            if round_no == 1:
                notes.append("no_clarify_capability=single_pass")
            break

        changed = False
        for triple in pending:
            question = (
                f"要判断「{triple.subject} {triple.predicate} {triple.obj}」还缺什么？"
                "请只依据原文说明。"
            )
            try:
                answer = str(clarify(triple.as_record(), question) or "").strip()
            except Exception:  # noqa: BLE001 - a failed clarification must not abort the stage
                answer = ""
            if answer and answer != rationales.get(id(triple), ""):
                rationales[id(triple)] = answer
                changed = True
        if not changed:
            # A round that learns nothing: the same question cannot produce a different answer.
            notes.append(f"stopped_on_no_new_information_at_round_{round_no}")
            break

    if rounds >= int(max_rounds or 1) and any(
        stance == "unanswerable" for _triple, stance in stances
    ):
        notes.append("unanswerable_after_max_rounds")

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
        notes=tuple(notes),
        rounds=rounds,
    )
