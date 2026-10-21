"""B11 — independent labelling and multi-annotator agreement.

The inventory asks for two things under B11:

* **多标注者不复用同一 claim** — several labellers judge the same claim without seeing
  each other's verdicts;
* **第三人证据充分性检查** — a third party checks the evidence, not just the claim.

The stage already has a verifier with its own context (C3), but that is one labeller. One
labeller cannot disagree with itself, so C3 answers "is the verifier independent of the
extractor" and B11 answers "would two competent readers reach the same verdict". They are
different questions and the second one needs a second reader.

The metric is Cohen's κ rather than raw agreement, because raw agreement is inflated by
class imbalance: with 90% `support` in the sample, two labellers agree 82% of the time by
chance alone and the number says nothing.
"""

from __future__ import annotations

import pytest

from cbb_stage1.qc import agreement

CLAIMS = ["cl:1", "cl:2", "cl:3", "cl:4", "cl:5", "cl:6"]


class RecordingLLM:
    """One model, several 'annotators' — but each call is a separate, clean context."""

    def __init__(self, verdicts_by_annotator: dict[str, list[str]]) -> None:
        self.prompts: list[str] = []
        self._verdicts = verdicts_by_annotator
        self._counters: dict[str, int] = {k: 0 for k in verdicts_by_annotator}

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        # the prompt names its annotator; each has its own reply sequence
        for name in self._verdicts:
            if f"标注者：{name}" in prompt:
                index = self._counters[name]
                self._counters[name] = index + 1
                sequence = self._verdicts[name]
                return sequence[index % len(sequence)]
        raise AssertionError(f"prompt does not name an annotator: {prompt[:80]!r}")


# ── Cohen's kappa ───────────────────────────────────────────────────────────
def test_perfect_agreement_is_kappa_one() -> None:
    a = ["support", "support", "against", "against"]
    b = ["support", "support", "against", "against"]
    assert agreement.cohen_kappa(a, b) == pytest.approx(1.0)


def test_systematic_disagreement_is_kappa_minus_one() -> None:
    """Every label conflicts. Chance agreement is 50%, observed is 0."""
    a = ["support", "against", "support", "against"]
    b = ["against", "support", "against", "support"]
    assert agreement.cohen_kappa(a, b) == pytest.approx(-1.0)


def test_chance_level_agreement_is_kappa_zero() -> None:
    # A: half support / half against. B independent of A but same marginals.
    # Po and Pe both 0.5 -> kappa 0.
    a = ["support", "support", "against", "against"]
    b = ["support", "against", "support", "against"]
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.0)


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        agreement.cohen_kappa(["support"], ["support", "against"])


def test_an_empty_sample_has_no_kappa() -> None:
    """No sample is not perfect agreement — it is no measurement."""
    assert agreement.cohen_kappa([], []) is None


def test_a_single_category_has_no_kappa() -> None:
    """When both labellers used only one label, Pe=1 and kappa is undefined."""
    assert agreement.cohen_kappa(["support", "support"], ["support", "support"]) is None


# ── independent labelling ──────────────────────────────────────────────────
def test_every_annotator_judges_every_sampled_claim() -> None:
    llm = RecordingLLM({
        "A": ["support"],
        "B": ["support"],
        "C": ["against"],
    })
    report = agreement.independent_label(
        CLAIMS, llm=llm, evidence_by_claim={"cl:1": ["原文"], "cl:2": ["原文"]},
        annotators=("A", "B", "C"),
    )
    assert report.annotators == 3
    assert len({label.annotator for label in report.labels}) == 3
    assert {label.claim_id for label in report.labels} == set(CLAIMS)


def test_a_labeller_never_sees_another_labellers_verdict() -> None:
    """The whole point of 'independent' — otherwise it is one opinion three times."""
    llm = RecordingLLM({"A": ["support"], "B": ["against"], "C": ["unsure"]})
    agreement.independent_label(CLAIMS[:2], llm=llm, annotators=("A", "B", "C"))
    for prompt in llm.prompts:
        named = [name for name in ("A", "B", "C") if f"标注者：{name}" in prompt]
        assert len(named) == 1
        # no other annotator's verdict word may appear
        for other in ("A", "B", "C"):
            if other in named:
                continue
            assert f"标注者：{other}" not in prompt


def test_disagreements_are_reported_not_averaged_away() -> None:
    llm = RecordingLLM({"A": ["support"], "B": ["against"]})
    report = agreement.independent_label(
        ["cl:1"], llm=llm, annotators=("A", "B")
    )
    assert report.disagreements == ("cl:1",)
    assert report.kappa is not None
    assert report.kappa < 1.0


def test_full_agreement_reports_no_disagreement() -> None:
    """Perfect agreement *across categories*.

    Both labellers giving the same single label everywhere leaves kappa undefined (0/0),
    not 1.0 — see `test_a_single_category_has_no_kappa`. Agreement is only measurable when
    there is something to agree about.
    """
    llm = RecordingLLM({"A": ["support", "against"], "B": ["support", "against"]})
    report = agreement.independent_label(
        ["cl:1", "cl:2", "cl:3", "cl:4"], llm=llm, annotators=("A", "B")
    )
    assert report.disagreements == ()
    assert report.kappa == pytest.approx(1.0)


def test_no_model_yields_unavailable_and_no_numbers() -> None:
    report = agreement.independent_label(CLAIMS, llm=None, annotators=("A", "B"))
    assert report.status == "unavailable"
    assert report.kappa is None
    assert report.labels == ()


def test_one_annotator_is_not_an_agreement_check() -> None:
    """κ needs two readers; one reader cannot disagree with itself."""
    llm = RecordingLLM({"A": ["support"]})
    report = agreement.independent_label(CLAIMS, llm=llm, annotators=("A",))
    assert report.status == "unavailable"
    assert report.kappa is None


def test_the_sample_size_is_capped() -> None:
    """Sampling, not a full pass: κ on thousands of claims costs more than it tells."""
    llm = RecordingLLM({"A": ["support"], "B": ["support"]})
    report = agreement.independent_label(
        CLAIMS, llm=llm, annotators=("A", "B"), sample_size=3
    )
    assert len({label.claim_id for label in report.labels}) == 3


def test_a_single_label_sample_reports_no_kappa_rather_than_a_bogus_one() -> None:
    """The report must say "not measurable", not print 0 or 1 in place of 0/0."""
    llm = RecordingLLM({"A": ["support"], "B": ["support"]})
    report = agreement.independent_label(CLAIMS[:3], llm=llm, annotators=("A", "B"))
    assert report.status == "measured"
    assert report.kappa is None, "kappa is undefined when both labellers used one label"
    assert report.disagreements == ()
    assert report.soft_gate()["annotator_agreement_kappa"] is None


def test_the_report_is_json_serialisable() -> None:
    import json

    llm = RecordingLLM({"A": ["support"], "B": ["support"]})
    report = agreement.independent_label(CLAIMS[:2], llm=llm, annotators=("A", "B"))
    payload = json.dumps(report.as_dict(), ensure_ascii=False)
    assert "independent_multi_annotator" in payload


def test_the_method_names_what_it_is() -> None:
    assert agreement.METHOD == "independent_multi_annotator"


def test_kappa_is_reported_as_a_soft_gate_not_a_hard_one() -> None:
    """A small sample makes κ unreliable; a hard gate on it would go red at random.

    This test documents the intended use: the number goes into ``soft_gates``, where a
    reader sees it, and never into ``hard_gates``, where a shaky statistic would block a
    build.
    """
    llm = RecordingLLM({"A": ["support", "against"], "B": ["support", "against"]})
    report = agreement.independent_label(CLAIMS[:4], llm=llm, annotators=("A", "B"))
    soft = report.soft_gate()
    assert soft["annotator_agreement_kappa"] == pytest.approx(1.0)
    assert soft["annotator_sample"] == 4
