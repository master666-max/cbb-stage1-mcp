"""Timeline constraint solver: partial order, cycle quarantine, contradictions.

Contract: box ``knowledge/timeline.jsonl`` (design/01 §5.8) + invariants
**I5** (timeline acyclic, ``before/after`` antisymmetric) and **I25**
(``verify.result`` must not contradict the constraint; partial order is
allowed, a total order is never invented).

Semantics, written down so callers do not guess:

* ``before(L, R)``  -> strict edge ``L -> R`` (``R`` happens later).
* ``after(L, R)``   -> strict edge ``R -> L``.
* ``during`` / ``overlap`` are **interval relations**. Under the box's
  point/whole-predecessor abstraction they are incompatible with a strict
  precedence in either direction, so such a combination is reported as a
  contradiction (quarantine + conflict), never silently reordered.
* Only the partial order is promised. Incomparable nodes relate as ``unknown``.
* A contradictory ``nli_pairwise`` result becomes a conflict record; it never
  overwrites the constraint (design/01 §5.8, I25).

Pure functions only: no I/O, no clock, no model call (NLI is injected through
:mod:`cbb_stage1.temporal.nli`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from ..ids import conflict_id as make_conflict_id
from ..ids import quarantine_id as make_quarantine_id
from ..ids import timeline_id as make_timeline_id
from .nli import VERIFY_METHODS, VERIFY_RESULTS, NliImpl, verify_constraint

TIMELINE_KINDS = ("before", "after", "during", "overlap")
NODE_KINDS = ("event", "unit", "time_point")
TIMELINE_SOURCES = ("explicit", "inferred")
TIMELINE_STATUSES = ("asserted", "inferred", "disputed")

# 箱 §5.12 的拒因闭集里，本模块会产出的那几项
QUARANTINE_REASONS = ("schema", "ref_missing", "timeline_cycle", "timeline_contradiction")

CONFLICT_KIND = "timeline"


class TimelineContractError(ValueError):
    """A timeline record violates the box contract (closed set / required key)."""


@dataclass(frozen=True)
class Node:
    kind: str
    id: str

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise TimelineContractError("node.kind 不属闭集 %s：%r" % (NODE_KINDS, self.kind))
        if not isinstance(self.id, str) or not self.id:
            raise TimelineContractError("node.id 必须是非空字符串")


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    book_id: str
    kind: str
    left: Node
    right: Node
    source: str = "explicit"
    confidence: float = 0.0
    confidence_method: str | None = None
    verify: Mapping[str, Any] | None = None
    evidence: tuple[Mapping[str, Any], ...] = ()
    status: str = "asserted"
    deps: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        record = {
            "constraint_id": self.constraint_id,
            "book_id": self.book_id,
            "kind": self.kind,
            "left": {"kind": self.left.kind, "id": self.left.id},
            "right": {"kind": self.right.kind, "id": self.right.id},
            "source": self.source,
            "confidence": self.confidence,
            "confidence_method": self.confidence_method,
            "verify": dict(self.verify) if self.verify else None,
            "evidence": [dict(e) for e in self.evidence],
            "status": self.status,
            "deps": list(self.deps),
        }
        return record


@dataclass(frozen=True)
class QuarantineItem:
    """One rejected/held record; shape follows ``audit/quarantine.jsonl``."""

    item_id: str
    kind: str
    reason: str
    detail: str
    source_run: str | None = None

    def __post_init__(self) -> None:
        if self.reason not in QUARANTINE_REASONS:
            raise ValueError("quarantine reason 不属本模块闭集：%r" % (self.reason,))

    def as_record(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "kind": self.kind, "reason": self.reason,
                "detail": self.detail, "source_run": self.source_run}


@dataclass(frozen=True)
class ConflictItem:
    """Timeline contradiction / cycle; shape follows ``audit/conflicts.jsonl``."""

    conflict_id: str
    book_id: str
    kind: str
    left: str
    right: str
    detail: str
    status: str = "open"
    decision: str | None = None
    resolved_by: str | None = None
    at: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {"conflict_id": self.conflict_id, "book_id": self.book_id, "kind": self.kind,
                "left": self.left, "right": self.right, "detail": self.detail,
                "status": self.status, "decision": self.decision,
                "resolved_by": self.resolved_by, "at": self.at}


def make_conflict(book_id: str, left: str, right: str, detail: str) -> ConflictItem:
    """Deterministic conflict id so reruns diff cleanly (I7)."""
    return ConflictItem(
        conflict_id=make_conflict_id(book_id, CONFLICT_KIND, left, right, detail),
        book_id=book_id, kind=CONFLICT_KIND, left=left, right=right, detail=detail)


@dataclass(frozen=True)
class SolveResult:
    constraints: tuple[Constraint, ...]
    quarantine: tuple[QuarantineItem, ...]
    conflicts: tuple[ConflictItem, ...]
    nodes: tuple[str, ...]
    edges: Mapping[str, tuple[str, ...]]
    reachability: Mapping[str, frozenset[str]]
    topological_order: tuple[str, ...] | None
    cycles: tuple[tuple[str, ...], ...]

    def precedes(self, left: str, right: str) -> bool:
        """True iff ``left`` strictly precedes ``right`` in the partial order."""
        return right in self.reachability.get(left, frozenset())

    def relates(self, left: str, right: str) -> str:
        """``before`` / ``after`` / ``same`` / ``unknown`` (partial order, I25)."""
        if left == right:
            return "same"
        if self.precedes(left, right):
            return "before"
        if self.precedes(right, left):
            return "after"
        return "unknown"

    @property
    def has_cycle(self) -> bool:
        return bool(self.cycles)

    @property
    def order(self) -> tuple[str, ...] | None:
        return self.topological_order


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def _node(raw: Any) -> Node:
    if not isinstance(raw, Mapping):
        raise TimelineContractError("left/right 必须是 {kind, id} 对象")
    return Node(kind=str(raw.get("kind")), id=str(raw.get("id")))


def parse_constraint(record: Mapping[str, Any]) -> Constraint:
    """Parse one timeline record; raises :class:`TimelineContractError` on violation."""
    if not isinstance(record, Mapping):
        raise TimelineContractError("timeline 记录必须是对象")
    kind = record.get("kind")
    if kind not in TIMELINE_KINDS:
        raise TimelineContractError("kind 不属闭集 %s：%r" % (TIMELINE_KINDS, kind))
    book_id = record.get("book_id")
    if not isinstance(book_id, str) or not book_id:
        raise TimelineContractError("book_id 必须是非空字符串")
    left, right = _node(record.get("left")), _node(record.get("right"))
    source = record.get("source", "explicit")
    if source not in TIMELINE_SOURCES:
        raise TimelineContractError("source 不属闭集 %s：%r" % (TIMELINE_SOURCES, source))
    status = record.get("status", "asserted")
    if status not in TIMELINE_STATUSES:
        raise TimelineContractError("status 不属闭集 %s：%r" % (TIMELINE_STATUSES, status))
    confidence = record.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise TimelineContractError("confidence 必须在 [0,1]：%r" % (confidence,))
    verify = record.get("verify")
    if verify is not None:
        if not isinstance(verify, Mapping):
            raise TimelineContractError("verify 必须是对象或 null")
        if verify.get("method") not in VERIFY_METHODS:
            raise TimelineContractError("verify.method 不属闭集 %s：%r"
                                        % (VERIFY_METHODS, verify.get("method")))
        if verify.get("result") not in VERIFY_RESULTS:
            raise TimelineContractError("verify.result 不属闭集 %s：%r"
                                        % (VERIFY_RESULTS, verify.get("result")))
    evidence = tuple(dict(e) for e in (record.get("evidence") or []) if isinstance(e, Mapping))
    deps = tuple(str(d) for d in (record.get("deps") or []))
    cid = record.get("constraint_id")
    if not (isinstance(cid, str) and cid.strip()):
        # 旧合法 ID 保留（现有约束不重编号）；缺号才用统一 t: 公式补。
        cid = make_timeline_id(book_id, kind, left.kind, left.id, right.kind, right.id)
    return Constraint(constraint_id=str(cid), book_id=book_id, kind=kind, left=left, right=right,
                      source=source, confidence=float(confidence),
                      confidence_method=record.get("confidence_method"),
                      verify=dict(verify) if verify else None, evidence=evidence,
                      status=status, deps=deps)


def parse_constraints(records: Iterable[Mapping[str, Any]]) -> tuple[tuple[Constraint, ...],
                                                                     tuple[QuarantineItem, ...]]:
    """Split records into parsed constraints and contract-rejected quarantine items."""
    parsed: list[Constraint] = []
    rejected: list[QuarantineItem] = []
    for record in records:
        try:
            parsed.append(parse_constraint(record))
        except TimelineContractError as exc:
            rejected.append(QuarantineItem(
                item_id=make_quarantine_id({"detail": str(exc), "record": record}),
                kind=CONFLICT_KIND, reason="schema", detail=str(exc)))
    return tuple(parsed), tuple(rejected)


# ---------------------------------------------------------------------------
# graph primitives (pure)
# ---------------------------------------------------------------------------
def _adjacency(constraints: Iterable[Constraint]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for c in constraints:
        if c.kind == "before":
            src, dst = c.left.id, c.right.id
        elif c.kind == "after":
            src, dst = c.right.id, c.left.id
        else:
            continue
        adj.setdefault(src, set()).add(dst)
    return adj


def _reachable(adj: Mapping[str, set[str]], start: str) -> frozenset[str]:
    seen: set[str] = set()
    stack = list(adj.get(start, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adj.get(node, ()))
    return frozenset(seen)


def _transitive_closure(adj: Mapping[str, set[str]], nodes: Iterable[str]) -> dict[str, frozenset[str]]:
    return {node: _reachable(adj, node) for node in nodes}


def _reaches(adj: Mapping[str, set[str]], src: str, dst: str) -> bool:
    if src == dst:
        return True
    return dst in _reachable(adj, src)


def _canonical_cycle(cycle: Sequence[str]) -> tuple[str, ...]:
    seq = tuple(cycle)
    if not seq:
        return seq
    pivot = min(range(len(seq)), key=lambda i: seq[i])
    return seq[pivot:] + seq[:pivot]


def find_cycles(adj: Mapping[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    """All elementary-ish cycles (deduped by canonical rotation); small graphs only."""
    found: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for start in sorted(adj):
        stack: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for nxt in sorted(adj.get(node, ())):
                if nxt == start:
                    key = _canonical_cycle(path)
                    if key not in seen:
                        seen.add(key)
                        found.append(key)
                elif nxt not in path:
                    stack.append((nxt, path + (nxt,)))
    return tuple(found)


def topological_order(nodes: Iterable[str], adj: Mapping[str, set[str]]) -> tuple[str, ...] | None:
    """Kahn order over all nodes; ``None`` when the graph is cyclic."""
    indegree = {node: 0 for node in nodes}
    for src, targets in adj.items():
        for dst in targets:
            indegree.setdefault(src, 0)
            indegree[dst] = indegree.get(dst, 0) + 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for nxt in sorted(adj.get(node, ())):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    return tuple(order) if len(order) == len(indegree) else None


def _cycle_edges(adj: Mapping[str, set[str]]) -> set[tuple[str, str]]:
    return {(src, dst) for src, targets in adj.items() for dst in targets
            if src == dst or _reaches(adj, dst, src)}


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def solve(records: Iterable[Mapping[str, Any]], *,
          known_events: Iterable[str] | None = None,
          known_units: Iterable[str] | None = None,
          nli: NliImpl | None = None,
          verify: bool | Iterable[str] = False) -> SolveResult:
    """Solve a batch of timeline records.

    ``verify``: ``False`` (default) = no NLI; ``True`` = verify every parsed
    constraint; an iterable = verify only those ``constraint_id``s.
    Verification never mutates the caller's records and never overwrites a
    constraint: a ``no`` verdict is emitted as a conflict (I25).
    """
    parsed, rejected = parse_constraints(records)
    quarantine: list[QuarantineItem] = list(rejected)
    conflicts: list[ConflictItem] = []

    events = set(known_events) if known_events is not None else None
    units = set(known_units) if known_units is not None else None
    kept: list[Constraint] = []
    for c in parsed:
        if (c.left.kind == "event" and events is not None and c.left.id not in events) or \
           (c.right.kind == "event" and events is not None and c.right.id not in events) or \
           (c.left.kind == "unit" and units is not None and c.left.id not in units) or \
           (c.right.kind == "unit" and units is not None and c.right.id not in units):
            quarantine.append(QuarantineItem(
                item_id=make_quarantine_id({"constraint_id": c.constraint_id,
                                            "reason": "ref_missing"}),
                kind=CONFLICT_KIND, reason="ref_missing",
                detail="端点不在已知集合：%s/%s 或 %s/%s"
                       % (c.left.kind, c.left.id, c.right.kind, c.right.id)))
            continue
        kept.append(c)

    adj = _adjacency(kept)
    cycles = find_cycles(adj)
    cyc_edges = _cycle_edges(adj)
    cycle_for_edge: dict[tuple[str, str], tuple[str, ...]] = {}
    for cycle in cycles:
        edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        for edge in edges:
            cycle_for_edge.setdefault(edge, cycle)

    clean: list[Constraint] = []
    for c in kept:
        edge = ((c.left.id, c.right.id) if c.kind == "before"
                else (c.right.id, c.left.id) if c.kind == "after" else None)
        if edge is not None and edge in cyc_edges:
            cycle = cycle_for_edge.get(edge, ())
            detail = "时序环：%s（约束 %s 落在环上，整条进隔离，回炉重抽）" % (
                " -> ".join(cycle), c.constraint_id)
            quarantine.append(QuarantineItem(
                item_id=make_quarantine_id({"constraint_id": c.constraint_id,
                                            "reason": "timeline_cycle"}),
                kind=CONFLICT_KIND, reason="timeline_cycle", detail=detail))
            conflicts.append(make_conflict(c.book_id, c.left.id, c.right.id, detail))
            continue
        clean.append(c)

    clean_adj = _adjacency(clean)
    nodes = sorted({c.left.id for c in kept} | {c.right.id for c in kept})
    # A cycle means the hard gate I5 cannot pass; do not hand out a tidy-looking
    # total order over the surviving subgraph (it would hide the quarantine).
    order = None if cycles else topological_order(nodes, clean_adj)
    closure = _transitive_closure(clean_adj, nodes)

    # interval relations: containment/overlap cannot coexist with a strict
    # precedence in either direction (whole-interval semantics, documented).
    # Quarantined constraints are held OUT of the accepted set: the timeline
    # table must not contain records that are simultaneously in quarantine.
    accepted: list[Constraint] = list(clean)
    quarantined_ids: set[str] = set()
    for c in clean:
        if c.kind not in ("during", "overlap"):
            continue
        left_before = c.right.id in closure.get(c.left.id, frozenset())
        right_before = c.left.id in closure.get(c.right.id, frozenset())
        if left_before or right_before:
            direction = "%s 先于 %s" % ((c.left.id, c.right.id) if left_before
                                        else (c.right.id, c.left.id))
            detail = "%s 与严格先后矛盾（%s；%s）" % (c.kind, direction, c.constraint_id)
            quarantine.append(QuarantineItem(
                item_id=make_quarantine_id({"constraint_id": c.constraint_id,
                                            "reason": "timeline_contradiction"}),
                kind=CONFLICT_KIND, reason="timeline_contradiction", detail=detail))
            conflicts.append(make_conflict(c.book_id, c.left.id, c.right.id, detail))
            quarantined_ids.add(c.constraint_id)
    if quarantined_ids:
        accepted = [c for c in accepted if c.constraint_id not in quarantined_ids]

    # NLI verification: additive record, contradiction -> conflict (never overwrite)
    if nli is not None and verify is not False:
        wanted = None if verify is True else set(verify)
        verified: list[Constraint] = []
        for c in accepted:
            if wanted is not None and c.constraint_id not in wanted:
                verified.append(c)
                continue
            record, contradicted = verify_constraint(c, nli)
            verified.append(replace(c, verify=record))
            if contradicted:
                detail = "NLI 判 no，与约束相反（%s %s→%s；%s）" % (
                    c.kind, c.left.id, c.right.id, c.constraint_id)
                conflicts.append(make_conflict(c.book_id, c.left.id, c.right.id, detail))
        accepted = verified

    edges = {src: tuple(sorted(targets)) for src, targets in sorted(clean_adj.items())}
    unique_quarantine = tuple(dict(((q.item_id, q.reason, q.detail), q)
                                   for q in quarantine).values())
    unique_conflicts = tuple(dict(((c.conflict_id), c) for c in conflicts).values())
    return SolveResult(constraints=tuple(accepted), quarantine=unique_quarantine,
                       conflicts=unique_conflicts, nodes=tuple(nodes), edges=edges,
                       reachability={k: v for k, v in closure.items()},
                       topological_order=order, cycles=cycles)
