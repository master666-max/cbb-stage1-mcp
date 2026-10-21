"""Shared records. These mirror the box v2.2 contract at the Python boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ClaimLayer = Literal["fact", "statement", "inferred"]
EvidenceMode = Literal["narration", "dialogue", "inner", "memory", "document", "letter"]
CueStatus = Literal["candidate", "active", "triggered", "dormant", "paid", "falsified", "unresolved"]
ClaimVerdict = Literal["supported", "refuted", "conflicting", "insufficient_evidence"]


class Speaker(BaseModel):
    kind: Literal["entity", "narrator"] = "narrator"
    entity_id: str | None = None
    name: str | None = None


class Evidence(BaseModel):
    unit_id: str
    start: int
    end: int
    quote: str
    quote_sha256: str | None = None
    occurrence: int = 1
    speaker: Speaker | None = None
    mode: EvidenceMode = "narration"


class Mention(BaseModel):
    mention_id: str | None = None
    unit_id: str
    surface: str
    start: int
    end: int
    occurrence: int = 1
    entity_id: str | None = None
    mention_type: str = "name"
    confidence: float = 0.0


class Entity(BaseModel):
    entity_id: str | None = None
    book_id: str
    type: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    description_quote: str | None = None
    tier: Literal["canonical", "major", "minor"] = "minor"
    status: Literal["active", "merged", "inactive"] = "active"
    confidence: float = 0.0


class Assertion(BaseModel):
    assertion_id: str | None = None
    book_id: str
    predicate: str
    library: str
    roles: dict[str, Any]
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    claim_layer: ClaimLayer = "fact"
    speaker: Speaker | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_method: str | None = None
    confidence_scope: Literal["assertion", "claim"] = "assertion"
    status: str = "asserted"
    verification: dict[str, Any] | None = None
    superseded_by: str | None = None
    derived_from: list[str] = Field(default_factory=list)
    deps: list[str] = Field(default_factory=list)
    produced_by: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    event_id: str | None = None
    book_id: str
    type: str
    summary: str
    participants: list[str] = Field(default_factory=list)
    location: str | None = None
    time: dict[str, Any] = Field(default_factory=dict)
    importance: str = "medium"
    unit_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    status: str = "asserted"
    superseded_by: str | None = None
    deps: list[str] = Field(default_factory=list)


class Cue(BaseModel):
    cue_id: str | None = None
    book_id: str
    kind: str = "object"
    subject: dict[str, Any] = Field(default_factory=dict)
    setup: dict[str, Any] = Field(default_factory=dict)
    reinforce: list[Any] = Field(default_factory=list)
    trigger: dict[str, Any] = Field(default_factory=dict)
    payoff: dict[str, Any] = Field(default_factory=dict)
    status: CueStatus = "candidate"
    activated_at: str | None = None
    triggered_at: str | None = None
    dormant_since: str | None = None
    falsified_reason: str | None = None
    premature_payoff: bool = False
    expected: str | None = None
    confidence: float = 0.0
    deps: list[str] = Field(default_factory=list)


class ClaimQuestion(BaseModel):
    qid: str
    question: str
    answers: list[dict[str, Any]] = Field(default_factory=list)
    stance: Literal["support", "refute", "irrelevant", "unanswerable"] = "unanswerable"


class AtomicFact(BaseModel):
    fact_id: str
    text: str
    verdict: Literal["true", "false", "unverifiable"] = "unverifiable"
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str | None = None


class Claim(BaseModel):
    claim_id: str | None = None
    book_id: str
    statement_assertion_id: str
    normalized_claim: str
    claim_type: str = "attribute"
    questions: list[ClaimQuestion] = Field(default_factory=list)
    atomic_facts: list[AtomicFact] = Field(default_factory=list)
    verdict: ClaimVerdict = "insufficient_evidence"
    verdict_rule: str = "av-eritec"
    justification: str | None = None
    confidence: float = 0.0
    confidence_method: str = "claim_qa"
    confidence_scope: Literal["assertion", "claim"] = "claim"
    verified_by: dict[str, Any] = Field(default_factory=dict)
    status: str = "open"


class EngineOutput(BaseModel):
    unit_id: str
    mentions: list[Mention] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    cues: list[Cue] = Field(default_factory=list)
    summary: str | None = None
    abstain: bool = False
    notes: list[str] = Field(default_factory=list)
    gleaning_round: int = 0
