"""The quote-verification verdict must actually reach the work table.

The unit tests cover `verify_speaker_quote` in isolation. This one drives the validate
stage and checks the assertion record, because the failure mode being guarded against is
a verdict computed and then dropped — the project has produced that exact shape before
(a confidence score computed for 59 calls and never written back).
"""

from __future__ import annotations

import json
from pathlib import Path

from cbb_stage1.validate.integration import run as run_validate

UNIT = "u:t:v01:c001"
QUOTE = "「这家伙…… 就是母体……──这个终末的源头。」"
TEXT = (
    "「嗯？怎么了言万同学。你怕狗吗？」\n\n"
    + QUOTE
    + "\n\n（这么小只，这么柔弱的生物？）"
)


class FakeVerifier:
    """A verifier that says the quote was said by 言万."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:  # noqa: ANN001
        self.prompts.append(prompt)
        return '{"verdict": "support", "rationale": "上一句点名问了言万"}'


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _build_work(root: Path) -> None:
    start = TEXT.index(QUOTE)
    assertion = {
        "assertion_id": "a:test0000000001",
        "book_id": "t",
        "unit_id": UNIT,
        "predicate": "敌对",
        "library": "关系",
        "roles": {"施事": "e:t:person:0001", "受事": "e:t:person:0002"},
        "claim_layer": "statement",
        "speaker": {"kind": "entity", "entity_id": "e:t:person:0002", "name": "言万"},
        "evidence": [
            {
                "unit_id": UNIT,
                "start": start,
                "end": start + len(QUOTE),
                "quote": QUOTE,
                "occurrence": 1,
                "mode": "dialogue",
            }
        ],
        "confidence": 0.6,
        "confidence_method": "engine_self",
        "status": "asserted",
        "verification": {"level": "minor", "verdict": "none", "tier": "minor"},
    }
    _write(root / "resolve" / "assertions.jsonl", [assertion])
    _write(
        root / "identity" / "entities.jsonl",
        [
            {"entity_id": "e:t:person:0002", "canonical_name": "言万", "aliases": []},
            {"entity_id": "e:t:person:0001", "canonical_name": "露娜", "aliases": []},
        ],
    )
    texts = root / "plan" / "texts"
    texts.mkdir(parents=True, exist_ok=True)
    (texts / f"{UNIT.replace(':', '_')}.txt").write_text(TEXT, encoding="utf-8")


def test_the_verdict_is_written_back_to_the_assertion(tmp_path: Path) -> None:
    root = tmp_path / "work"
    _build_work(root)
    verifier = FakeVerifier()

    run_validate(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-20T00:00:00+08:00",
        upstream={"verify_speaker": True, "verifier_llm": verifier},
    )

    rows = [
        json.loads(line)
        for line in (root / "resolve" / "assertions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1, "the assertion must survive verification"
    verification = rows[0]["verification"]
    assert verification["verdict"] == "support"
    assert verification["speaker_check"] == "llm"
    assert "言万" in verification["rationale"] or verification["rationale"]
    assert verifier.prompts, "the verifier must actually be asked"


def test_the_speaker_is_never_removed_by_a_verdict(tmp_path: Path) -> None:
    """AVeriTeC records a judgement; it does not delete the attribution."""
    root = tmp_path / "work"
    _build_work(root)

    class Refuting:
        def generate(self, prompt: str) -> str:  # noqa: ANN001
            return '{"verdict": "against", "rationale": "是露娜说的"}'

    run_validate(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-20T00:00:00+08:00",
        upstream={"verify_speaker": True, "verifier_llm": Refuting()},
    )

    rows = [
        json.loads(line)
        for line in (root / "resolve" / "assertions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["verification"]["verdict"] == "against"
    assert rows[0]["speaker"]["name"] == "言万", "the verdict must not edit the record"


def test_without_the_flag_nothing_is_checked(tmp_path: Path) -> None:
    """The stage stays cheap unless a caller opts in."""
    root = tmp_path / "work"
    _build_work(root)
    verifier = FakeVerifier()

    run_validate(
        root,
        book_id="t",
        outputs=[],
        units=[],
        created_at="2026-10-20T00:00:00+08:00",
        upstream={"verifier_llm": verifier},
    )

    assert verifier.prompts == []
    rows = [
        json.loads(line)
        for line in (root / "resolve" / "assertions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["verification"]["verdict"] == "none"
