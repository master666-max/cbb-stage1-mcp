"""Per-document processing state, derived from its tests.

The six states are the source's live `DocStatus` members. `preprocessed` is deprecated upstream and is
deliberately absent: carrying a state nothing produces makes the table lie about what can be seen.

Transitions follow the pipeline order, with one allowance the source also makes: `analyzing` is optional,
so `parsing` may go straight to `processing`. Everything else raises, because a table that accepts any
move cannot answer "what happened to this document".

Append-only. A retry writes a new row rather than overwriting the failure, because the earlier failure is
evidence about the source file — `attempts_of` and `state_of` are separate questions on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "DOC_STATES",
    "TERMINAL_STATES",
    "DocStatusError",
    "DocStatusRow",
    "DocStatusTable",
    "can_transition",
    "next_states",
    "resume_work",
]

#: The source's six live states, in pipeline order.
DOC_STATES: tuple[str, ...] = (
    "pending",
    "parsing",
    "analyzing",
    "processing",
    "processed",
    "failed",
)

#: States with nothing left to do in a normal run.
TERMINAL_STATES: frozenset[str] = frozenset({"processed", "failed"})

#: Legal moves. `parsing` may skip `analyzing` because the source marks it optional; `failed` may return to
#: `pending` because a retry is a real operation and not a new document.
_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "pending": ("parsing", "failed"),
    "parsing": ("analyzing", "processing", "failed"),
    "analyzing": ("processing", "failed"),
    "processing": ("processed", "failed"),
    # A rerun of a changed book is a new attempt on the same document, so a terminal state may be
    # re-armed. `failed` had this exit already; `processed` needs it for the same reason, or re-running a
    # book would raise instead of recording the new attempt.
    "processed": ("pending",),
    "failed": ("pending",),
}


class DocStatusError(ValueError):
    """An unknown state, or a transition the pipeline does not allow."""


def can_transition(current: str, new: str) -> bool:
    """True when ``current -> new`` is a legal move."""
    if current not in DOC_STATES or new not in DOC_STATES:
        return False
    return new in _TRANSITIONS[current]


def next_states(current: str) -> tuple[str, ...]:
    """The states a document may move to, for a caller deciding what to do next."""
    return _TRANSITIONS.get(current, ())


@dataclass
class DocStatusRow:
    doc_id: str
    state: str
    attempt: int = 1
    detail: str = ""
    at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "state": self.state,
            "attempt": self.attempt,
            "detail": self.detail,
            "at": self.at,
        }


@dataclass
class DocStatusTable:
    """Append-only per-document state log; the latest row for a document is its state."""

    path: Path
    rows: list[DocStatusRow] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "DocStatusTable":
        """Read a table, skipping unparseable lines rather than refusing the whole file."""
        table = cls(path=Path(path))
        if not table.path.is_file():
            return table
        for line in table.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            doc_id = payload.get("doc_id")
            state = payload.get("state")
            if not doc_id or not state:
                continue
            table.rows.append(
                DocStatusRow(
                    doc_id=str(doc_id),
                    state=str(state),
                    attempt=int(payload.get("attempt") or 1),
                    detail=str(payload.get("detail") or ""),
                    at=str(payload.get("at") or ""),
                )
            )
        return table

    # ── reads ────────────────────────────────────────────────────────
    def latest(self) -> dict[str, DocStatusRow]:
        out: dict[str, DocStatusRow] = {}
        for row in self.rows:
            out[row.doc_id] = row
        return dict(sorted(out.items()))

    def state_of(self, doc_id: str) -> str | None:
        row = self.latest().get(doc_id)
        return row.state if row else None

    def attempts_of(self, doc_id: str) -> int:
        return sum(1 for row in self.rows if row.doc_id == doc_id)

    def by_state(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {state: [] for state in DOC_STATES}
        for doc_id, row in self.latest().items():
            grouped.setdefault(row.state, []).append(doc_id)
        return grouped

    # ── writes ───────────────────────────────────────────────────────
    def transition(
        self, doc_id: str, new_state: str, *, detail: str = "", at: str = ""
    ) -> DocStatusRow:
        """Append a state change, rejecting an unknown state or an illegal move."""
        if new_state not in DOC_STATES:
            raise DocStatusError(
                f"unknown document state {new_state!r}; expected one of {list(DOC_STATES)}"
            )
        current = self.state_of(doc_id)
        if current is None and new_state != "pending":
            raise DocStatusError(
                f"{doc_id}: a document must start at 'pending', not {new_state!r}"
            )
        if current is not None and not can_transition(current, new_state):
            raise DocStatusError(
                f"{doc_id}: cannot move {current!r} -> {new_state!r}; "
                f"allowed: {list(next_states(current))}"
            )
        row = DocStatusRow(
            doc_id=doc_id,
            state=new_state,
            attempt=self.attempts_of(doc_id) + 1,
            detail=detail,
            at=at,
        )
        self.rows.append(row)
        return row

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "\n".join(json.dumps(row.as_dict(), ensure_ascii=False) for row in self.rows) + "\n",
            encoding="utf-8",
        )
        return self.path


def resume_work(rows: Iterable[DocStatusRow]) -> dict[str, list[str]]:
    """Documents a resume must act on, grouped by what needs doing.

    ``unstarted`` and ``failed`` need work; ``done`` does not; ``in_flight`` means the previous run died
    mid-document. Separating those four is the whole reason the six states exist — a boolean "done" cannot
    tell a failed document from a finished one, and stopping at the first is how a bad file gets retried
    forever or never.
    """
    latest: dict[str, DocStatusRow] = {}
    for row in rows:
        latest[row.doc_id] = row
    grouped: dict[str, list[str]] = {
        "unstarted": [],
        "failed": [],
        "in_flight": [],
        "done": [],
    }
    for doc_id, row in sorted(latest.items()):
        if row.state == "pending":
            grouped["unstarted"].append(doc_id)
        elif row.state == "failed":
            grouped["failed"].append(doc_id)
        elif row.state == "processed":
            grouped["done"].append(doc_id)
        else:
            grouped["in_flight"].append(doc_id)
    return grouped
