"""Replay: re-derive from the sources and report what changed, row by row.

`replay_base` was a field hardcoded to `None`, which meant a reader could not even ask "what changed
since the box this build was derived from". This module is the missing half: a deterministic row-level
diff and a report that says whether a comparison actually happened.

The diff is keyed by `(table, key)` and compares the **payload** only — the key fields are identity,
not content. Reporting a change when only the key's own copy of a field differs would produce noise for
every table whose rows carry their own id twice.

`status` is three-valued on purpose:

* `compared` — a base was given and the diff is the answer;
* `no_base` — nothing to compare against. The rows present in the rebuild are still reported as added,
  because that is the honest reading of "everything is new"; calling it a clean replay would not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["ReplayDiff", "ReplayReport", "diff_tables", "replay_report"]

#: Row fields that identify a row rather than describe it. Excluded from the payload comparison.
_IDENTITY_FIELDS = frozenset({"table", "key"})


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in row.items() if name not in _IDENTITY_FIELDS}


def _index(rows: Sequence[Mapping[str, Any]], table: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("key") or "")
        if key:
            indexed[key] = _payload(row)
    return indexed


@dataclass(frozen=True)
class ReplayDiff:
    added: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    changed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": list(self.changed),
            "counts": {
                "added": len(self.added),
                "removed": len(self.removed),
                "changed": len(self.changed),
            },
        }


def diff_tables(
    before: Mapping[str, Sequence[Mapping[str, Any]]],
    after: Mapping[str, Sequence[Mapping[str, Any]]],
) -> ReplayDiff:
    """Row-level diff between two table sets, keyed by ``(table, key)``.

    Deterministic: every list is sorted by ``(table, key)``, so two runs over the same inputs produce
    byte-identical reports and a diff can be asserted in a test.
    """
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []

    for table in sorted(set(before) | set(after)):
        old = _index(before.get(table) or (), table)
        new = _index(after.get(table) or (), table)
        for key in sorted(set(new) - set(old)):
            added.append({"table": table, "key": key, "payload": new[key]})
        for key in sorted(set(old) - set(new)):
            removed.append({"table": table, "key": key, "payload": old[key]})
        for key in sorted(set(old) & set(new)):
            if old[key] == new[key]:
                continue
            fields: dict[str, Any] = {}
            for name in sorted(set(old[key]) | set(new[key])):
                if old[key].get(name) != new[key].get(name):
                    fields[name] = {
                        "before": old[key].get(name),
                        "after": new[key].get(name),
                    }
            changed.append({"table": table, "key": key, "fields": fields})

    added.sort(key=lambda item: (item["table"], item["key"]))
    removed.sort(key=lambda item: (item["table"], item["key"]))
    changed.sort(key=lambda item: (item["table"], item["key"]))
    return ReplayDiff(added=added, removed=removed, changed=changed)


@dataclass(frozen=True)
class ReplayReport:
    base: str | None
    status: str
    diff: ReplayDiff

    def as_dict(self) -> dict[str, Any]:
        return {"base": self.base, "status": self.status, **self.diff.as_dict()}


def replay_report(
    *,
    base_id: str | None,
    before: Mapping[str, Sequence[Mapping[str, Any]]],
    after: Mapping[str, Sequence[Mapping[str, Any]]],
) -> ReplayReport:
    """The replay answer: what changed, and whether there was anything to compare against.

    ``base_id`` is the digest/label of the box the rebuild was derived from. When it is missing the
    status is ``no_base`` even though the diff still reports the rebuilt rows — "we could not compare"
    and "nothing changed" are different answers and must not share one.
    """
    diff = diff_tables(before, after)
    status = "compared" if base_id else "no_base"
    return ReplayReport(base=base_id, status=status, diff=diff)
