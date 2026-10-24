"""Assertion assembly: predicate registry -> roles -> claim layer -> confidence.

The assembler is the deterministic gate between engine output and the box
(``00-架构总设计.md`` §4.9, ``01-箱规格-v2.md`` §5.6 / §5.15 / §5.16):

* ``predicate`` must be registered; an unknown predicate goes to the proposal
  table and is **not** accepted ("表外谓词不自动注册").
* ``roles`` keys must equal the registry role names and values must match the
  role type; anything else is ``role_mismatch`` quarantine.
* ``library`` comes from the registry; data cannot choose it.
* ``claim_layer`` is ``fact`` / ``statement`` / ``inferred``.  Evidence
  ``mode=dialogue|inner|memory|document|letter`` defaults to ``statement``;
  ``narration`` defaults to ``fact`` (``statement`` when the unit narrator is
  unreliable).  A ``statement`` without a ``speaker`` is
  ``claim_layer_conflict`` quarantine.
* ``confidence`` without a ``confidence_method`` is not trusted; the method must
  be in the closed enum, and ``calibrated`` must carry a version.
* ``speaker`` / ``mode`` are passed through untouched (they are the narrative
  layer's only evidence of who claims what).

Quarantine reasons use the closed set from §5.12 so ``qc``/``quarantine`` rows
stay lint-able.  This module does not write files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from cbb_stage1 import ids
from cbb_stage1.evidence.anchor import sha256_text
from cbb_stage1.models import Assertion, ClaimLayer, Evidence, Speaker

from .registry import (
    PredicateProposal,
    PredicateRegistry,
    normalize_library,
    propose_predicate,
)

__all__ = [
    "AssembleResult",
    "Assembler",
    "CONFIDENCE_METHODS",
    "CONFIDENCE_METHOD_ALIASES",
    "EVIDENCE_MODES",
    "LayerDowngrade",
    "MODE_DEFAULT_CLAIM_LAYER",
    "QUARANTINE_REASONS",
    "QuarantineItem",
    "assemble_assertions",
    "assertion_id_for",
    "infer_claim_layer",
    "normalize_confidence_method",
    "resolve_claim_layer",
]

#: Closed quarantine ``reason`` set (01 §5.12).
QUARANTINE_REASONS: frozenset[str] = frozenset({
    "schema",
    "evidence_not_found",
    "role_mismatch",
    "ref_missing",
    "timeline_cycle",
    "contradiction",
    "low_confidence",
    "duplicate",
    "unresolved_quote",
    "claim_layer_conflict",
})

#: ``confidence_method`` enum from §5.15.
CONFIDENCE_METHODS: frozenset[str] = frozenset({
    "engine_self",
    "evidence_span",
    "uqlm_whitebox",
    "uqlm_blackbox",
    "semantic_entropy",
    "uq_heads",
    "claim_qa",
    "human",
    "calibrated",
})

#: Legacy spellings observed in engine/adapter output.  Mapped, not accepted
#: verbatim: the box records the canonical enum value.
CONFIDENCE_METHOD_ALIASES: dict[str, str] = {
    "engine_self_report": "engine_self",
    "engine_report": "engine_self",
    "self_report": "engine_self",
    "evidence": "evidence_span",
    "span": "evidence_span",
}

EVIDENCE_MODES: tuple[str, ...] = (
    "narration",
    "dialogue",
    "inner",
    "memory",
    "document",
    "letter",
)

#: §5.16 default claim layer per evidence mode.
MODE_DEFAULT_CLAIM_LAYER: dict[str, ClaimLayer] = {
    "narration": "fact",
    "dialogue": "statement",
    "inner": "statement",
    "memory": "statement",
    "document": "statement",
    "letter": "statement",
}

_CALIBRATION_KEYS = ("calibration_version",)


# ── helpers ──────────────────────────────────────────────────────


def _pending_assertion_id(raw: Assertion) -> str:
    return "a:pending:" + ids.short_hash(
        ids.canonical_json({"predicate": raw.predicate, "roles": raw.roles}), 16
    )


def normalize_confidence_method(method: str | None) -> str | None:
    """Canonical method name, ``None`` when absent/invalid."""
    if not method:
        return None
    value = str(method).strip()
    if not value:
        return None
    value = CONFIDENCE_METHOD_ALIASES.get(value, value)
    return value if value in CONFIDENCE_METHODS else None


def infer_claim_layer(
    evidence: Sequence[Evidence],
    *,
    narrator_reliability: str = "reliable",
) -> ClaimLayer:
    """Raw mode-based default: any non-narration evidence -> ``statement``."""
    if not evidence:
        return "fact"
    if all(item.mode == "narration" for item in evidence):
        return "statement" if narrator_reliability == "unreliable" else "fact"
    return "statement"


def resolve_claim_layer(
    requested: ClaimLayer,
    evidence: Sequence[Evidence],
    *,
    narrator_reliability: str = "reliable",
) -> tuple[ClaimLayer, bool]:
    """Return ``(layer, downgraded)``; explicit ``statement``/``inferred`` win.

    A record asking for ``fact`` while carrying dialogue/inner/... evidence is
    downgraded to ``statement`` (the caller may still keep ``fact`` when a
    verifier recorded ``verdict=support``; see :class:`Assembler`).
    """
    if requested in ("statement", "inferred"):
        return requested, False
    inferred = infer_claim_layer(evidence, narrator_reliability=narrator_reliability)
    return inferred, inferred != "fact"


def _evidence_primary_sha(evidence: Sequence[Evidence]) -> str:
    """Box-style primary quote hash (``sha256:`` prefixed, recomputed)."""
    if not evidence:
        return ""
    primary = evidence[0]
    value = primary.quote_sha256 or f"sha256:{sha256_text(primary.quote)}"
    if not value.startswith("sha256:"):
        value = f"sha256:{value}"
    return value


def assertion_id_for(
    book_id: str,
    predicate: str,
    roles: Mapping[str, Any],
    evidence: Sequence[Evidence] = (),
) -> str:
    """Thin wrapper over :func:`cbb_stage1.ids.assertion_id` (same ``a:`` bytes).

    Only the primary quote hash is extracted here; the key formula lives in the
    neutral ``ids`` module.  Evidence-less (``inferred``) records share the
    same formula; they never reach the box, which requires anchored evidence.
    """
    return ids.assertion_id(book_id, predicate, roles, _evidence_primary_sha(evidence))


def _role_entity_ids(roles: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            if value.startswith("e:"):
                ids.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
        elif isinstance(value, Mapping):
            for item in value.values():
                _walk(item)

    for value in roles.values():
        _walk(value)
    return ids


def _has_verified_support(raw: Assertion) -> bool:
    verification = raw.verification or {}
    return isinstance(verification, Mapping) and verification.get("verdict") == "support"


def _calibration_version(raw: Assertion) -> str | None:
    for container in (raw.verification, raw.produced_by):
        if isinstance(container, Mapping):
            for key in _CALIBRATION_KEYS:
                value = container.get(key)
                if value:
                    return str(value)
    return None


#: Evidence modes that mean "someone other than the narrator said this".
_ATTRIBUTED_MODES = frozenset(
    {"dialogue", "inner", "memory", "document", "letter"}
)


def _is_narration_only(raw: Assertion) -> bool:
    """True when nothing in the evidence claims to be someone else's words.

    Used to decide whether a downgrade may be attributed to the narrator. An assertion with no
    evidence at all is treated as narration: the narrator is then the only speaker left, and the
    alternative is dropping the record entirely.
    """
    modes = {
        str(getattr(item, "mode", "") or "")
        for item in (raw.evidence or ())
    }
    return not (modes & _ATTRIBUTED_MODES)


def _resolve_speaker(raw: Assertion) -> Speaker | None:
    if raw.speaker is not None:
        return raw.speaker
    for item in raw.evidence:
        if item.speaker is not None:
            return item.speaker
    return None


def _item_id(raw: Assertion) -> str:
    if raw.assertion_id:
        return raw.assertion_id
    return _pending_assertion_id(raw)


# ── result records ───────────────────────────────────────────────


class QuarantineItem(BaseModel):
    """``audit/quarantine.jsonl`` row (§5.12)."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    kind: str = "assertion"
    reason: str
    detail: str
    source_run: str | None = None

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        if value not in QUARANTINE_REASONS:
            raise ValueError(f"unknown quarantine reason {value!r}")
        return value


@dataclass(frozen=True)
class LayerDowngrade:
    """One ``fact -> statement`` downgrade forced by the evidence modes."""

    assertion_ref: str
    requested: str
    applied: str
    reason: str


@dataclass
class AssembleResult:
    """Batch outcome: accepted assertions, quarantines, proposals, downgrades."""

    accepted: list[Assertion] = field(default_factory=list)
    quarantined: list[QuarantineItem] = field(default_factory=list)
    proposals: list[PredicateProposal] = field(default_factory=list)
    downgrades: list[LayerDowngrade] = field(default_factory=list)
    duplicates_dropped: int = 0

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined)

    @property
    def proposal_count(self) -> int:
        return len(self.proposals)

    def extend(self, other: "AssembleResult") -> "AssembleResult":
        self.accepted.extend(other.accepted)
        self.quarantined.extend(other.quarantined)
        self.proposals.extend(other.proposals)
        self.downgrades.extend(other.downgrades)
        self.duplicates_dropped += other.duplicates_dropped
        return self


# ── assembler ────────────────────────────────────────────────────


class Assembler:
    """Deterministic assertion assembler for one book/unit stream."""

    def __init__(
        self,
        registry: PredicateRegistry,
        *,
        book_id: str | None = None,
        narrator_reliability: str = "reliable",
        run_id: str | None = None,
        statement_ids: Iterable[str] | None = None,
    ) -> None:
        self.registry = registry
        self.book_id = book_id
        self.narrator_reliability = narrator_reliability
        self.run_id = run_id
        self.statement_ids = frozenset(statement_ids or ())
        self._accepted_ids: set[str] = set()

    # -- public ------------------------------------------------------
    def assemble(self, assertions: Iterable[Assertion]) -> AssembleResult:
        result = AssembleResult()
        for raw in assertions:
            self._assemble_one(raw, result)
        return result

    def assemble_one(self, raw: Assertion) -> AssembleResult:
        """Assemble a single assertion (useful for incremental streams)."""
        result = AssembleResult()
        self._assemble_one(raw, result)
        return result

    # -- internals ---------------------------------------------------
    def _quarantine(self, result: AssembleResult, raw: Assertion, reason: str, detail: str) -> None:
        result.quarantined.append(
            QuarantineItem(
                item_id=_item_id(raw),
                kind="assertion",
                reason=reason,
                detail=detail,
                source_run=self.run_id,
            )
        )

    def _propose(
        self, result: AssembleResult, raw: Assertion, reason_detail: str
    ) -> None:
        book_id = raw.book_id or self.book_id or ""
        seen_in = sorted({e.unit_id for e in raw.evidence})
        proposal = propose_predicate(
            raw.predicate,
            book_id=book_id,
            seen_in=seen_in,
            count=1,
            role_shape_examples=[dict(raw.roles)],
        )
        result.proposals.append(proposal)
        self._quarantine(result, raw, "schema", reason_detail)

    def _assemble_one(self, raw: Assertion, result: AssembleResult) -> None:
        book_id = raw.book_id or self.book_id
        if not book_id:
            self._quarantine(result, raw, "schema", "assertion has no book_id")
            return

        # 1) predicate closure --------------------------------------
        spec = self.registry.get(raw.predicate)
        if spec is None:
            self._propose(
                result,
                raw,
                f"unregistered predicate {raw.predicate!r} -> proposed_predicates "
                "(not auto-registered)",
            )
            return

        # 2) roles -------------------------------------------------
        check = self.registry.check_roles(raw.predicate, raw.roles)
        if not check.ok:
            self._quarantine(result, raw, "role_mismatch", check.detail())
            return

        # 3) library comes from the registry ------------------------
        if raw.library:
            try:
                requested_library = normalize_library(raw.library)
            except ValueError as exc:
                self._quarantine(result, raw, "schema", str(exc))
                return
            if requested_library != spec.library:
                self._quarantine(
                    result,
                    raw,
                    "schema",
                    f"library {raw.library!r} does not match registry library "
                    f"{spec.library!r} for predicate {raw.predicate!r}",
                )
                return
        library = spec.library

        # 4) claim layer + speaker + status -------------------------
        requested_layer: ClaimLayer = raw.claim_layer
        layer, downgraded = resolve_claim_layer(
            requested_layer,
            raw.evidence,
            narrator_reliability=self.narrator_reliability,
        )
        if downgraded and _has_verified_support(raw):
            layer, downgraded = "fact", False
        if downgraded:
            result.downgrades.append(
                LayerDowngrade(
                    assertion_ref=_item_id(raw),
                    requested=requested_layer,
                    applied=layer,
                    reason="evidence modes default to statement (§5.16)",
                )
            )

        speaker: Speaker | None = None
        if layer == "statement":
            speaker = _resolve_speaker(raw)
            if speaker is None and downgraded and _is_narration_only(raw):
                # The downgrade is what created this statement, and for narration the speaker can
                # only be the narrator: "the narrator claims X" is the honest reading. Without this
                # the assertion was quarantined, and `narrator_reliability=unreliable` silently
                # deleted narration instead of lowering its layer.
                #
                # Guarded on the evidence mode: a downgraded *dialogue* line belongs to whoever said
                # it, and blaming the narrator for someone else's line would wave it through.
                speaker = Speaker(kind="narrator", name="narrator")
            if speaker is None:
                # 01-箱规格 §5.16 / I16: a `statement` MUST carry a speaker. This is a
                # hard rule, not a preference — the quarantine is the specified outcome,
                # not a lost record. Downgrading to `fact` would silently turn "someone
                # said X" into "X is true", which is exactly the confusion the layer
                # exists to prevent.
                #
                # (I tried the downgrade first; the spec check killed it. Two tests
                # asserted the quarantine and they were right.)
                self._quarantine(
                    result,
                    raw,
                    "claim_layer_conflict",
                    "claim_layer=statement requires a speaker "
                    "(entity or narrator) on the assertion or its evidence",
                )
                return
        elif layer == "fact":
            if self._statement_derivation_violation(raw):
                self._quarantine(
                    result,
                    raw,
                    "claim_layer_conflict",
                    "claim_layer=fact derives only from statement assertions; "
                    "it needs independent narration evidence or a verified support",
                )
                return

        if layer == "inferred":
            if not raw.derived_from and not raw.evidence:
                self._quarantine(
                    result,
                    raw,
                    "claim_layer_conflict",
                    "claim_layer=inferred requires derived_from or premise evidence",
                )
                return
        elif not raw.evidence:
            self._quarantine(
                result,
                raw,
                "evidence_not_found",
                "asserted assertion has no anchored evidence",
            )
            return

        # 5) confidence contract -----------------------------------
        method = normalize_confidence_method(raw.confidence_method)
        if raw.confidence_method and method is None:
            self._quarantine(
                result,
                raw,
                "schema",
                f"confidence_method {raw.confidence_method!r} is not in the closed enum",
            )
            return
        if method is None and raw.confidence > 0:
            self._quarantine(
                result,
                raw,
                "schema",
                "confidence without confidence_method is not trusted",
            )
            return
        if method == "calibrated" and _calibration_version(raw) is None:
            self._quarantine(
                result,
                raw,
                "schema",
                "confidence_method=calibrated requires calibration_version",
            )
            return

        # 6) id + deps + record ------------------------------------
        assertion_id = assertion_id_for(book_id, raw.predicate, raw.roles, raw.evidence)
        if assertion_id in self._accepted_ids:
            result.duplicates_dropped += 1
            return

        deps: list[str] = list(raw.deps)
        for item in raw.evidence:
            if item.unit_id not in deps:
                deps.append(item.unit_id)
        for entity_id in _role_entity_ids(raw.roles):
            if entity_id not in deps:
                deps.append(entity_id)

        status = "inferred" if layer == "inferred" else "asserted"
        accepted = raw.model_copy(
            update={
                "assertion_id": assertion_id,
                "book_id": book_id,
                "library": library,
                "claim_layer": layer,
                "speaker": speaker if layer == "statement" else raw.speaker,
                "confidence_method": method,
                "status": status,
                "deps": deps,
            }
        )
        self._accepted_ids.add(assertion_id)
        result.accepted.append(accepted)

    def _statement_derivation_violation(self, raw: Assertion) -> bool:
        """I16: a fact may not rest only on statement assertions."""
        narration_evidence = any(item.mode == "narration" for item in raw.evidence)
        return bool(
            raw.derived_from
            and self.statement_ids
            and all(item in self.statement_ids for item in raw.derived_from)
            and not narration_evidence
        )


def assemble_assertions(
    assertions: Iterable[Assertion],
    registry: PredicateRegistry,
    **kwargs: Any,
) -> AssembleResult:
    """One-shot convenience wrapper around :class:`Assembler`."""
    return Assembler(registry, **kwargs).assemble(assertions)
