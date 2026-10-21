"""Reversible merge ledger (``identity/merges.jsonl`` shape, box v2).

A merge record is *always* reversible: it stores ``from`` / ``to`` plus the
minimal ``undo`` payload, and the ledger keeps enough state to undo a merge and
to re-resolve every merged id.  Old entity ids are never reused; a chain such
as ``A -> B``, ``B -> C`` is legal and ``resolve(A) == C``.

Design references: ``01-箱规格-v2.md`` §5.5 (merge record: ``kind`` /
``decided_by`` / ``evidence`` / ``reversible`` / ``undo``), §4.8 Resolver
(``merges.jsonl``, old id never reused, ``name_history`` kept by the entity
table), and ``schemas/merge.schema.json`` (field shape; ``reversible: true``).

Not in scope here: detecting *which* ids should merge (that is the resolver /
LLM / human decision) and writing quarantine/conflict records (see
``validate/``).  This module only records, validates, applies, resolves and
undoes merges.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cbb_stage1 import ids
from cbb_stage1.models import Evidence

__all__ = [
    "DecidedBy",
    "MergeError",
    "MergeKind",
    "MergeLedger",
    "MergeRecord",
    "make_merge",
    "merge_id_for",
]

MergeKind = Literal["alias", "coref", "type_unify"]
DecidedBy = Literal["rule", "llm", "human", "panel"]


class MergeError(ValueError):
    """A merge record or ledger transition is illegal."""


def _decision_key(record: "MergeRecord") -> tuple[Any, ...]:
    """Structural identity of a merge decision (timestamp/evidence excluded)."""
    return (
        record.book_id,
        record.kind,
        tuple(sorted(record.from_ids)),
        record.to,
        record.decided_by,
    )


def merge_id_for(book_id: str, kind: str, from_ids: Iterable[str], to_id: str) -> str:
    """Thin wrapper over :func:`cbb_stage1.ids.merge_id` (same bytes, ``mg:``)."""
    return ids.merge_id(book_id, kind, from_ids, to_id)


class MergeRecord(BaseModel):
    """One reversible merge (field names match ``schemas/merge.schema.json``).

    The JSON field is ``from`` (a Python reserved word), so the model field is
    ``from_ids`` with an alias; ``model_dump(by_alias=True)`` emits the box
    shape.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    merge_id: str
    book_id: str
    kind: MergeKind = "alias"
    from_ids: list[str] = Field(alias="from", min_length=1)
    to: str
    decided_by: DecidedBy = "rule"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    at: str
    reversible: Literal[True] = True
    undo: dict[str, Any] = Field(default_factory=dict)

    @field_validator("from_ids")
    @classmethod
    def _validate_from(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("'from' contains duplicate entity ids")
        if any(not value.startswith("e:") for value in values):
            raise ValueError("every 'from' id must start with 'e:'")
        return values

    @field_validator("to")
    @classmethod
    def _validate_to(cls, value: str) -> str:
        if not value.startswith("e:"):
            raise ValueError("'to' id must start with 'e:'")
        return value

    def box_dict(self) -> dict[str, Any]:
        """Serializable record in box shape (``from``/``to``/``undo``...)."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_box_dict(cls, payload: Mapping[str, Any]) -> "MergeRecord":
        return cls.model_validate(dict(payload))


def _as_evidence_dicts(evidence: Iterable[Evidence | Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in evidence or ():
        if isinstance(item, Evidence):
            out.append(item.model_dump(mode="json"))
        elif isinstance(item, Mapping):
            out.append(dict(item))
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported merge evidence type: {type(item).__name__}")
    return out


def make_merge(
    book_id: str,
    from_ids: Sequence[str],
    to_id: str,
    *,
    kind: MergeKind = "alias",
    decided_by: DecidedBy = "rule",
    confidence: float = 1.0,
    evidence: Iterable[Evidence | Mapping[str, Any]] | None = None,
    at: str | None = None,
) -> MergeRecord:
    """Build a validated record with the minimal reversible ``undo`` payload."""
    from_list = [str(f) for f in from_ids]
    record = MergeRecord(
        merge_id=merge_id_for(book_id, kind, from_list, to_id),
        book_id=book_id,
        kind=kind,
        from_ids=from_list,
        to=to_id,
        decided_by=decided_by,
        confidence=confidence,
        evidence=_as_evidence_dicts(evidence),
        at=at or datetime.now(timezone.utc).isoformat(),
        reversible=True,
        undo={"from": list(from_list), "to": to_id},
    )
    if record.to in record.from_ids:
        raise MergeError(f"merge target {record.to!r} is also a source id")
    return record


class MergeLedger:
    """In-memory ledger for one book: apply / resolve / undo.

    * ``apply`` is idempotent by ``merge_id`` (a rerun does not duplicate rows and
      does not resurrect an undone merge: undo is a human decision).
    * A source id can only be merged once while its record is active.
    * A merge that would close a cycle (``B -> A`` after ``A -> B``) is refused.
    * ``undo`` only deactivates the record; the record itself stays in the log
      (box tables are append-only: "被 supersede 的记录永久保留").
    """

    def __init__(self, records: Iterable[MergeRecord] = (), *, book_id: str | None = None) -> None:
        self.book_id = book_id
        self._records: list[MergeRecord] = []
        self._by_id: dict[str, MergeRecord] = {}
        self._undone: set[str] = set()
        # active source id -> merge id, so resolve() is O(depth), not O(records)
        self._source_index: dict[str, str] = {}
        for record in records:
            self.apply(record)

    # -- introspection -------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[MergeRecord, ...]:
        return tuple(self._records)

    @property
    def active(self) -> tuple[MergeRecord, ...]:
        return tuple(r for r in self._records if r.merge_id not in self._undone)

    @property
    def undone_ids(self) -> frozenset[str]:
        return frozenset(self._undone)

    def get(self, merge_id: str) -> MergeRecord | None:
        return self._by_id.get(merge_id)

    # -- transitions ---------------------------------------------------
    def apply(self, record: MergeRecord) -> MergeRecord:
        """Validate and append *record*; return the stored record.

        Reruns are idempotent: the same ``merge_id`` (same ``book/kind/from/to``
        decision) returns the stored record even when the rerun carries a new
        ``at`` timestamp or re-collected evidence.  A reused id with a different
        structural decision is a ledger corruption and raises.
        """
        if self.book_id is not None and record.book_id != self.book_id:
            raise MergeError(
                f"merge {record.merge_id} belongs to book {record.book_id!r}, "
                f"ledger is for {self.book_id!r}"
            )
        if record.merge_id in self._by_id:
            existing = self._by_id[record.merge_id]
            if _decision_key(existing) != _decision_key(record):
                raise MergeError(
                    f"merge id {record.merge_id} already used by a different decision"
                )
            return existing

        for source in record.from_ids:
            if source == record.to:
                raise MergeError("merge target is also a source id")
            if self.resolve(source) == self.resolve(record.to):
                raise MergeError(
                    f"merge {source} -> {record.to} would close a cycle "
                    "(target already resolves to the source group)"
                )
            if self._active_record_for(source) is not None:
                raise MergeError(f"entity {source} is already merged; old ids are write-once")

        self._records.append(record)
        self._by_id[record.merge_id] = record
        for source in record.from_ids:
            self._source_index[source] = record.merge_id
        return record

    def add(
        self,
        from_ids: Sequence[str],
        to_id: str,
        *,
        book_id: str | None = None,
        **kwargs: Any,
    ) -> MergeRecord:
        """Create (via :func:`make_merge`) and apply a merge."""
        bid = book_id or self.book_id
        if not bid:
            raise MergeError("book_id is required (pass it to the ledger or to add())")
        return self.apply(make_merge(bid, from_ids, to_id, **kwargs))

    def undo(self, merge_id: str) -> MergeRecord:
        """Reverse one merge; the record is kept and only deactivated."""
        record = self._by_id.get(merge_id)
        if record is None:
            raise MergeError(f"unknown merge id: {merge_id}")
        if merge_id in self._undone:
            raise MergeError(f"merge {merge_id} is already undone")
        self._undone.add(merge_id)
        for source in record.from_ids:
            self._source_index.pop(source, None)
        return record

    def reversal(self, merge_id: str) -> dict[str, Any]:
        """Return the minimal rollback payload of one merge."""
        record = self._by_id.get(merge_id)
        if record is None:
            raise MergeError(f"unknown merge id: {merge_id}")
        return dict(record.undo)

    # -- resolution ----------------------------------------------------
    def _active_record_for(self, entity_id: str) -> MergeRecord | None:
        merge_id = self._source_index.get(entity_id)
        return self._by_id.get(merge_id) if merge_id is not None else None

    def resolve(self, entity_id: str) -> str:
        """Follow active merge records to the surviving id (path compressed)."""
        seen: set[str] = set()
        current = entity_id
        while True:
            if current in seen:
                raise MergeError(f"merge cycle detected at {current!r}")
            seen.add(current)
            record = self._active_record_for(current)
            if record is None:
                return current
            current = record.to

    def merged_into_map(self) -> dict[str, str]:
        """``old_id -> resolved_id`` for every id that was merged (any depth)."""
        result: dict[str, str] = {}
        for record in self.active:
            for source in record.from_ids:
                result[source] = self.resolve(source)
        return result

    def merge_groups(self) -> dict[str, list[str]]:
        """Resolved id -> all ids that resolve to it (groups of size >= 2)."""
        groups: dict[str, list[str]] = {}
        for record in self.active:
            for source in record.from_ids:
                root = self.resolve(source)
                members = groups.setdefault(root, [root])
                if source not in members:
                    members.append(source)
        return {root: sorted(members) for root, members in groups.items() if len(members) >= 2}

    def history(self, entity_id: str) -> list[MergeRecord]:
        """Merge records (both active and undone) whose source is *entity_id*."""
        return [r for r in self._records if entity_id in r.from_ids]

    # -- persistence ---------------------------------------------------
    def to_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record.box_dict(), ensure_ascii=False) + "\n")
        return target

    @classmethod
    def from_jsonl(cls, path: str | Path, *, book_id: str | None = None) -> "MergeLedger":
        ledger = cls(book_id=book_id)
        target = Path(path)
        if not target.exists():
            return ledger
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ledger.apply(MergeRecord.from_box_dict(json.loads(line)))
        return ledger
