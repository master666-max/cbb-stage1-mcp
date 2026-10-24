"""`validate`'s AVeriTeC speaker check must run concurrently.

The stage asks one question per statement — "was this quote actually said by the supposed
speaker" — and did it in a `for assertion in assertions:` loop. Measured on a real chapter it
is 2 checks; on a book with a few thousand statements it is a few thousand sequential model
calls sitting between `claims` and the box.

Two defects are being fixed at once, and they are different in kind:

1. **Serial** — the same class of deviation the other stages have (A13-AFEV §4.4, A09-CoVVE,
   A04-GraphRAG all prescribe parallel or batched verification).
2. **Re-reading a file per iteration** — `_speaker_cast(root)` was called *inside* the loop,
   so the roster was rebuilt from disk once per statement.

The dedupe contract
-------------------
`verify_speaker_quote` memoises on `(quote, speaker)` through a plain dict passed in by the
caller. Sharing that dict across threads would make the check-then-set a race: two threads
could both miss and both call the model for the same pair, so the call count would depend on
thread interleaving. Instead the stage now resolves the **distinct** pairs first, asks each
exactly once in parallel, and applies the verdicts to every occurrence in input order. That
makes the call count deterministic and never larger than the serial one.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from cbb_stage1.assemble import run as assemble_run
from cbb_stage1.resolve import run as resolve_run
from cbb_stage1.validate import run as validate_run

BOOK = "tst"
CREATED = "2026-10-18T23:00:00+08:00"
UNIT = "u:tst:v01:c001"


class SpeakerProbe:
    """An `llm_with_generate` that records peak concurrency."""

    def __init__(self, *, delay: float = 0.03) -> None:
        self.delay = delay
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.prompts: list[str] = []
        self.threads: set[str] = set()

    def generate(self, prompt: str) -> str:
        with self.lock:
            self.in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self.in_flight)
            self.prompts.append(prompt)
            self.threads.add(threading.current_thread().name)
        try:
            time.sleep(self.delay)
            return "support"
        finally:
            with self.lock:
                self.in_flight -= 1


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


def _outputs(count: int, *, repeat_quote: bool = False) -> list[dict]:
    speaker = {"kind": "entity", "entity_id": None, "name": "言万心叶"}
    assertions = []
    for index in range(count):
        start = index * 100
        # repeat_quote makes every statement quote the SAME sentence with the same speaker,
        # which is the case the (quote, speaker) memo exists for.
        quote = (
            "言万心叶说同一句话"
            if repeat_quote
            else f"言万心叶在第{index}次交锋中说了一句不同的话"
        )
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


def _prepare(tmp_path: Path, *, count: int, repeat_quote: bool = False) -> tuple[Path, list[dict], list[dict]]:
    """Build a work dir whose statements are ready for the speaker check."""
    work = tmp_path / "work"
    outputs = _outputs(count, repeat_quote=repeat_quote)
    units = _units()
    kwargs = dict(book_id=BOOK, outputs=outputs, units=units, created_at=CREATED)
    resolve_run(work, upstream={}, **kwargs)
    assemble_run(work, upstream={}, **kwargs)

    path = work / "assemble" / "assertions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if str(row.get("claim_layer")) == "statement":
            # `validate` only checks rows whose verification level is not "none".
            row["verification"] = {"level": "span"}
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return work, outputs, units


def _run_validate(work: Path, outputs, units, *, workers: int, probe: SpeakerProbe) -> dict:
    return validate_run(
        work,
        book_id=BOOK,
        outputs=outputs,
        units=units,
        created_at=CREATED,
        upstream={
            "verify_speaker": True,
            "verifier_llm": probe,
            "max_workers": workers,
        },
    )


# ══ the change ══════════════════════════════════════════════════════════
def test_speaker_checks_run_concurrently(tmp_path: Path) -> None:
    work, outputs, units = _prepare(tmp_path, count=8)
    probe = SpeakerProbe()
    _run_validate(work, outputs, units, workers=4, probe=probe)
    assert probe.calls >= 4, f"only {probe.calls} speaker checks ran; expected one per statement"
    assert probe.peak >= 2, f"peak in-flight speaker checks was {probe.peak}"


def test_one_worker_stays_serial(tmp_path: Path) -> None:
    work, outputs, units = _prepare(tmp_path, count=8)
    probe = SpeakerProbe()
    _run_validate(work, outputs, units, workers=1, probe=probe)
    assert probe.peak == 1


def test_worker_threads_are_named_for_the_stage(tmp_path: Path) -> None:
    work, outputs, units = _prepare(tmp_path, count=8)
    probe = SpeakerProbe()
    _run_validate(work, outputs, units, workers=4, probe=probe)
    assert any(name.startswith("cbb-validate") for name in probe.threads), sorted(probe.threads)


# ══ the dedupe contract ═════════════════════════════════════════════════
def test_a_repeated_quote_is_asked_once_even_in_parallel(tmp_path: Path) -> None:
    """The (quote, speaker) memo must survive the move to threads."""
    work, outputs, units = _prepare(tmp_path, count=8, repeat_quote=True)
    probe = SpeakerProbe()
    _run_validate(work, outputs, units, workers=4, probe=probe)
    assert probe.calls == 1, (
        f"the same (quote, speaker) pair was asked {probe.calls} times; the memo is raced"
    )


def test_distinct_quotes_are_each_asked_once(tmp_path: Path) -> None:
    work, outputs, units = _prepare(tmp_path, count=6)
    probe = SpeakerProbe()
    _run_validate(work, outputs, units, workers=4, probe=probe)
    assert probe.calls == 6, f"expected 6 distinct checks, got {probe.calls}"


# ══ what must not change ════════════════════════════════════════════════
def test_the_assertions_file_is_identical_to_the_serial_run(tmp_path: Path) -> None:
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()

    serial_work, serial_outputs, serial_units = _prepare(serial_dir, count=8)
    _run_validate(serial_work, serial_outputs, serial_units, workers=1, probe=SpeakerProbe())

    parallel_work, parallel_outputs, parallel_units = _prepare(parallel_dir, count=8)
    _run_validate(
        parallel_work, parallel_outputs, parallel_units, workers=4, probe=SpeakerProbe()
    )

    def verdicts(work: Path) -> list[dict]:
        path = work / "assemble" / "assertions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [row.get("verification") for row in rows]

    assert verdicts(parallel_work) == verdicts(serial_work), (
        "parallel run recorded different speaker verdicts"
    )
