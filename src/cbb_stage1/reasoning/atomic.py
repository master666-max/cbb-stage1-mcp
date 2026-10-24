"""AFEV: iterative atomic decomposition -> per-fact verification -> aggregation.

Design refs: ``05-资料库前沿检索与吸收-20261018.md`` §3.4 (B7) and
``01-箱规格-v2.md`` §5.19 (``atomic_facts``).

Algorithm (paper: ``库/papers/A13-AFEV-2025.pdf``): a complex claim is decomposed
into atomic facts; each atomic fact is retrieved against the source, then
verified; the rationale of unverifiable facts feeds the next decomposition round.
Aggregation is the rule from §5.19 rule 3: all ``true`` => ``supported``; any
``false`` => ``refuted``; any ``unverifiable`` => ``insufficient_evidence``.

Boundary: no LLM here. The model is an injected :class:`AtomicLLM`; **without it
the only honest result is one atomic fact with verdict ``unverifiable``** -- this
module never fabricates a verdict or evidence span. No file write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from cbb_stage1 import ids
from cbb_stage1.claims.verdicts import aggregate_atomic_facts
from cbb_stage1.models import AtomicFact

__all__ = [
    "ATOMIC_VERDICTS",
    "DEFAULT_MAX_ROUNDS",
    "AFEVResult",
    "AtomicLLM",
    "AtomicRuleError",
    "afev_verify",
]

ATOMIC_VERDICTS: tuple[str, ...] = ("true", "false", "unverifiable")
#: Upper bound on the decompose/verify rounds.
#:
#: **This is a bounded approximation, not the paper's rule.** A13-AFEV-2025 p.9 has the model decide
#: each iteration whether "the extracted facts collectively provide sufficient coverage of the
#: claim"; this constant caps the loop instead. The model *is* asked (`coverage_sufficient` below)
#: and can stop the loop early, but a model that does not implement that method is bounded by this
#: number alone, and the stop is then recorded as `unverifiable_after_max_rounds` rather than being
#: presented as a coverage judgement.
DEFAULT_MAX_ROUNDS = 3

#: Retrieval hook: ``fact_text -> evidence quotes``. Optional; without it all
#: facts see the claim-level evidence passed by the caller.
Retriever = Callable[[str], Sequence[str]]



#: Optional capability: "do the facts so far cover the claim?" (A13 p.9).
#:
#: Declared separately from :class:`AtomicLLM` on purpose — it is optional, so a model that lacks it
#: keeps working and simply cannot stop the loop early. The check is a ``getattr``, not an isinstance.
COVERAGE_METHOD = "coverage_sufficient"

class AtomicRuleError(ValueError):
    """The AFEV model returned something that is not a legal atomic fact."""


class AtomicLLM(Protocol):
    """The two model capabilities AFEV needs (implemented by the caller/verifier)."""

    def decompose(self, claim: str, context: str | None) -> Sequence[str]:
        """Split ``claim`` into atomic fact texts; ``context`` is last round's rationale."""

    def verify(self, fact: str, evidence: Sequence[str]) -> Mapping[str, Any]:
        """Return ``{"verdict": true|false|unverifiable, "rationale": str|None}``."""


def _fact_id(book_id: str, text: str, index: int) -> str:
    """Atomic-fact id; hashes via :mod:`cbb_stage1.ids` (bytes unchanged)."""
    return f"af:{ids.short_hash(f'{book_id}|{index}|{text}')}"


def _clean_texts(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise AtomicRuleError(f"decompose() must return non-empty strings, got {value!r}")
        text = value.strip()
        if text not in out:
            out.append(text)
    return out


def _verify_one(fact: str, evidence: Sequence[str], llm: AtomicLLM, book_id: str, index: int) -> AtomicFact:
    """Run one model verification and shape it into an :class:`AtomicFact`.

    Evidence spans are deliberately **empty**: this module has no unit/offset
    information, and inventing one would fabricate provenance. The caller keeps
    the quotes; they are not box evidence rows.
    """
    result = llm.verify(fact, list(evidence))
    if not isinstance(result, Mapping):
        raise AtomicRuleError(f"verify() must return a mapping, got {type(result).__name__}")
    verdict = result.get("verdict")
    if verdict not in ATOMIC_VERDICTS:
        raise AtomicRuleError(f"atomic verdict {verdict!r} not in {ATOMIC_VERDICTS}")
    rationale = result.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise AtomicRuleError(f"rationale must be str|None, got {type(rationale).__name__}")
    return AtomicFact(
        fact_id=_fact_id(book_id, fact, index),
        text=fact,
        verdict=verdict,  # type: ignore[arg-type]
        evidence=[],
        rationale=rationale,
    )


@dataclass(frozen=True)
class AFEVResult:
    """AFEV verification of one claim (box §5.19 ``atomic_facts`` material)."""

    claim: str
    book_id: str
    atomic_facts: tuple[AtomicFact, ...]
    verdict: str
    rounds: int
    degenerate: bool
    notes: tuple[str, ...] = ()

    @property
    def aggregate(self) -> str:
        """The §5.19 rule-3 aggregate of :attr:`atomic_facts`."""
        return aggregate_atomic_facts(self.atomic_facts)

    def atomic_records(self) -> list[dict[str, Any]]:
        """``atomic_facts[]`` rows as box-shaped dicts (no fabricated evidence)."""
        return [fact.model_dump(mode="json") for fact in self.atomic_facts]


def afev_verify(
    claim: str,
    *,
    book_id: str = "",
    evidence: Sequence[str] = (),
    llm: AtomicLLM | None = None,
    retrieve: Retriever | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> AFEVResult:
    """Decompose/verify/aggregate one claim.

    Without ``llm`` the result is exactly one ``unverifiable`` atomic fact and
    ``verdict="insufficient_evidence"`` (``degenerate=True``) -- never a
    fabricated verdict. With ``llm`` the decomposition is iterated up to
    ``max_rounds``, feeding unverifiable rationales as the next round's context.
    """
    if not isinstance(claim, str) or not claim.strip():
        raise AtomicRuleError("claim must be a non-empty string")
    if max_rounds < 1:
        raise AtomicRuleError("max_rounds must be >= 1")
    claim = claim.strip()
    base_evidence = tuple(evidence)

    if llm is None:
        fact = AtomicFact(
            fact_id=_fact_id(book_id, claim, 0),
            text=claim,
            verdict="unverifiable",
            evidence=[],
            rationale="no-llm: AFEV needs a verifier; unverifiable by rule (no fabrication)",
        )
        facts = (fact,)
        return AFEVResult(
            claim=claim,
            book_id=book_id,
            atomic_facts=facts,
            verdict=aggregate_atomic_facts(facts),
            rounds=0,
            degenerate=True,
            notes=("no_llm=unverifiable",),
        )

    facts: tuple[AtomicFact, ...] = ()
    context: str | None = None
    rounds = 0
    previous_texts: tuple[str, ...] | None = None
    stopped_on_repeat = False
    stopped_on_coverage = False
    for round_no in range(1, max_rounds + 1):
        rounds = round_no
        texts = _clean_texts(list(llm.decompose(claim, context)))
        if not texts:
            raise AtomicRuleError("decompose() returned no atomic facts (no fabrication fallback)")
        # A round that decomposes to exactly the previous round's facts has learned
        # nothing: same facts, same evidence, therefore the same verdicts. Verifying them
        # again and then running round 3 is pure cost. Measured on a 5000-character run:
        # every claim whose evidence simply does not cover it burned all three rounds on
        # identical facts, which is a third of the AFEV requests per claim.
        current_texts = tuple(texts)
        if previous_texts is not None and current_texts == previous_texts:
            stopped_on_repeat = True
            break
        previous_texts = current_texts
        shaped: list[AtomicFact] = []
        for index, text in enumerate(texts):
            fact_evidence = tuple(retrieve(text)) if retrieve is not None else base_evidence
            shaped.append(_verify_one(text, fact_evidence, llm, book_id, index))
        facts = tuple(shaped)
        if all(fact.verdict in ("true", "false") for fact in facts):
            break
        # A13 p.9: the model decides whether coverage is sufficient. Optional — a model without the
        # method simply cannot stop early, and the cap below still applies.
        ask_coverage = getattr(llm, COVERAGE_METHOD, None)
        if callable(ask_coverage):
            try:
                sufficient = bool(
                    ask_coverage(claim, [fact.text for fact in facts])
                )
            except Exception:  # noqa: BLE001 - a failed judgement must not abort verification
                sufficient = False
            if sufficient:
                stopped_on_coverage = True
                break
        context = "\n".join(
            f"{fact.text}: {fact.rationale or ''}".strip() for fact in facts
            if fact.verdict == "unverifiable"
        ) or None
        if round_no == max_rounds:
            break

    notes: list[str] = []
    if stopped_on_coverage:
        notes.append("stopped_on_sufficient_coverage")
    if stopped_on_repeat:
        # Named so a reader can tell "stopped because nothing changed" apart from
        # "ran out of rounds" — those are different findings about the evidence.
        notes.append("stopped_on_repeated_decomposition")
    if context and not stopped_on_coverage:
        notes.append("unverifiable_after_max_rounds")

    return AFEVResult(
        claim=claim,
        book_id=book_id,
        atomic_facts=facts,
        verdict=aggregate_atomic_facts(facts),
        rounds=rounds,
        degenerate=False,
        notes=tuple(notes),
    )
