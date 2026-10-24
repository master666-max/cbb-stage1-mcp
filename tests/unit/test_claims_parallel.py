"""`claims` must verify statements concurrently — the A13-AFEV §4.4 claim.

The source
----------
AFEV's paper, §4.4 Computational Complexity Analysis (PDF page 13):

    "since no gradient computation is involved and **each claim can be processed
     independently, this process is parallel and highly scalable during inference**"

Our stage ran `for statement in statements:` — strictly serial. On a 5000-character chapter
that is 44 statements x (AFEV decompose + per-fact verify x rounds + VeGraph) = ~220 model
calls, one at a time. For a 300k-character book it is ~3,100 statements and 12,000-22,000
serial calls.

What must not change
--------------------
Parallelism is a scheduling change, not a semantic one. So:

* the same statements must produce the same claims, in the same order;
* the same number of model calls must be issued (no duplicates, no omissions);
* per-statement failures must stay isolated and stay recorded in encounter order.

The last point is why the pool cannot simply append to shared lists: `refused` and
`afev_errors` are read by humans and by the box, and their order is part of their meaning.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from cbb_stage1.assemble import run as assemble_run
from cbb_stage1.claims import run as claims_run
from cbb_stage1.resolve import run as resolve_run

BOOK = "tst"
CREATED = "2026-10-18T23:00:00+08:00"
UNIT = "u:tst:v01:c001"


# ── a model that reports how many calls are in flight at once ──────────
class ConcurrencyProbe:
    """Implements AtomicLLM and VeGraphLLM, recording peak concurrency.

    `explode_after` raises on the Nth call rather than on a text match: the eight synthetic
    statements share the same normalised claim text, so a text trigger would either hit all
    of them or none.
    """

    def __init__(self, *, delay: float = 0.03, explode_after: int | None = None) -> None:
        self.delay = delay
        self.explode_after = explode_after
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.threads: set[str] = set()

    def _work(self, label: str) -> None:
        with self.lock:
            self.in_flight += 1
            self.calls += 1
            ordinal = self.calls
            self.peak = max(self.peak, self.in_flight)
            self.threads.add(threading.current_thread().name)
        try:
            time.sleep(self.delay)
            if self.explode_after is not None and ordinal == self.explode_after:
                raise RuntimeError(f"probe explosion on call {ordinal} ({label[:20]})")
        finally:
            with self.lock:
                self.in_flight -= 1

    # AtomicLLM
    def decompose(self, claim: str, context: str | None) -> list[str]:
        self._work(claim)
        return [f"原子事实一：{claim}", f"原子事实二：{claim}"]

    def verify(self, fact: str, evidence) -> dict:
        self._work(fact)
        return {"verdict": "true", "rationale": "证据直接支持"}

    # VeGraphLLM
    def triples(self, claim: str) -> list[dict]:
        self._work(claim)
        # The contract wants `predicate`; `relation` is rejected with VeGraphRuleError.
        return [{"subject": "言万心叶", "predicate": "敌对", "object": "拉法"}]

    def verify_subassertion(self, triple: dict) -> str:
        self._work(str(triple))
        return "support"


# ── fixtures ───────────────────────────────────────────────────────────
def _units() -> list[dict]:
    return [
        {
            "unit_id": UNIT,
            "book_id": BOOK,
            "volume": 1,
            "chapter": 1,
            "order": 1,
            "narrator_reliability": "reliable",
        }
    ]


def _outputs(count: int) -> list[dict]:
    """`count` independent statements.

    Two things decide whether a row reaches the claims stage at all:

    1. `claim_layer` must be "statement" — `_load_assertions` skips "fact" rows, so a fact
       fixture yields `statements: 0` and the probe is never called (measured).
    2. The assemble dedupe key is (predicate, roles, claim_layer, evidence SPANS) — quote
       text is deliberately not in it, so identical spans collapse into one assertion with
       merged evidence (measured: 8 rows became 1 with 8 evidence items). Distinct spans
       keep them independent.
    """
    speaker = {"kind": "entity", "entity_id": None, "name": "言万心叶"}
    assertions = []
    for index in range(count):
        start = index * 100
        quote = f"言万心叶在第{index}次交锋中与拉法对阵"
        assertions.append(
            {
                "predicate": "敌对",
                "library": "关系",
                "roles": {"施事": "言万心叶", "受事": "拉法"},
                "claim_layer": "statement",
                "speaker": speaker,
                "evidence": [
                    {
                        "unit_id": UNIT,
                        "start": start,
                        "end": start + len(quote),
                        "quote": quote,
                        "occurrence": 1,
                        "mode": "dialogue",
                        "speaker": speaker,
                    }
                ],
                "confidence": 0.9,
                "confidence_method": "evidence_span",
            }
        )
    return [
        {
            "unit_id": UNIT,
            "entities": [
                {
                    "canonical_name": "言万心叶",
                    "type": "person",
                    "aliases": ["心叶"],
                    "tier": "canonical",
                    "confidence": 0.9,
                },
                {
                    "canonical_name": "拉法",
                    "type": "person",
                    "aliases": ["拉法尔"],
                    "tier": "major",
                    "confidence": 0.8,
                },
            ],
            "assertions": assertions,
        }
    ]


def _run_claims(
    tmp_path: Path, *, count: int, workers: int, probe: ConcurrencyProbe
) -> tuple[dict, Path]:
    """Run resolve -> assemble -> claims and return (result, work_dir)."""
    work = tmp_path / f"work-{count}-{workers}"
    kwargs = dict(book_id=BOOK, outputs=_outputs(count), units=_units(), created_at=CREATED)
    resolve_run(work, upstream={}, **kwargs)
    assemble_run(work, upstream={}, **kwargs)
    result = claims_run(
        work,
        upstream={"reasoning_llm": probe, "max_workers": workers},
        **kwargs,
    )
    return result, work


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ══ the change: concurrent model calls ══════════════════════════════════
def test_statements_are_verified_concurrently(tmp_path: Path) -> None:
    probe = ConcurrencyProbe()
    _run_claims(tmp_path, count=8, workers=4, probe=probe)
    assert probe.peak >= 2, (
        f"peak in-flight model calls was {probe.peak}; claims is still serial "
        "(A13-AFEV §4.4: each claim can be processed independently)"
    )


def test_one_worker_stays_serial(tmp_path: Path) -> None:
    """The serial path must remain available and must not spawn a pool."""
    probe = ConcurrencyProbe()
    _run_claims(tmp_path, count=8, workers=1, probe=probe)
    assert probe.peak == 1, f"workers=1 still ran {probe.peak} calls at once"


def test_few_statements_stay_serial(tmp_path: Path) -> None:
    """Below PARALLEL_MIN_ITEMS a pool costs more than it saves."""
    probe = ConcurrencyProbe()
    _run_claims(tmp_path, count=2, workers=8, probe=probe)
    assert probe.peak == 1


# ══ what must not change ════════════════════════════════════════════════
def test_the_model_call_count_is_unchanged_by_parallelism(tmp_path: Path) -> None:
    """Parallelism must not introduce duplicate or skipped verification."""
    serial = ConcurrencyProbe()
    _run_claims(tmp_path, count=8, workers=1, probe=serial)

    parallel = ConcurrencyProbe()
    _run_claims(tmp_path, count=8, workers=4, probe=parallel)

    assert parallel.calls == serial.calls, (
        f"serial issued {serial.calls} calls, parallel issued {parallel.calls}"
    )


def test_the_claims_file_is_identical_to_the_serial_run(tmp_path: Path) -> None:
    """The regression anchor: a scheduling change must not change the artefact."""
    _serial_result, serial_work = _run_claims(
        tmp_path, count=8, workers=1, probe=ConcurrencyProbe()
    )
    _parallel_result, parallel_work = _run_claims(
        tmp_path, count=8, workers=4, probe=ConcurrencyProbe()
    )

    serial_rows = _read_jsonl(serial_work / "assemble" / "claims.jsonl")
    parallel_rows = _read_jsonl(parallel_work / "assemble" / "claims.jsonl")

    assert len(serial_rows) == 8, f"serial baseline produced {len(serial_rows)} claims"
    assert [row["claim_id"] for row in parallel_rows] == [
        row["claim_id"] for row in serial_rows
    ], "parallel run wrote claims in a different order"
    assert parallel_rows == serial_rows, "parallel run produced different claim rows"


def test_per_statement_failures_stay_isolated(tmp_path: Path) -> None:
    """One bad model call must not sink the stage; the failure is recorded, not swallowed."""
    probe = ConcurrencyProbe(explode_after=7)
    result, _work = _run_claims(tmp_path, count=8, workers=4, probe=probe)

    assert result["status"] == "ok"
    assert result["afev_error_rows"], "the exploding call produced no recorded error"
    assert {row["stage"] for row in result["afev_error_rows"]} <= {"afev", "vegraph", "cove"}
    assert result["counts"]["claims"] == 8, "statements without errors must still yield claims"
    assert result["counts"]["afev_errors"] >= 1


def test_the_counts_are_the_same_as_serial(tmp_path: Path) -> None:
    serial_result, _ = _run_claims(tmp_path, count=8, workers=1, probe=ConcurrencyProbe())
    parallel_result, _ = _run_claims(tmp_path, count=8, workers=4, probe=ConcurrencyProbe())

    for key in ("statements", "claims", "refused", "promoted_facts", "afev_errors", "abstained"):
        assert parallel_result["counts"][key] == serial_result["counts"][key], key


def test_worker_threads_are_named_for_the_stage(tmp_path: Path) -> None:
    """A thread dump must say which stage the threads belong to."""
    probe = ConcurrencyProbe()
    _run_claims(tmp_path, count=8, workers=4, probe=probe)
    named = {name for name in probe.threads if name.startswith("cbb-claims")}
    assert named, f"no cbb-claims thread observed; saw {sorted(probe.threads)}"
