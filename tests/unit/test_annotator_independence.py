"""Three labels from one model are not three annotators.

Measured substitution (B11)
---------------------------
`qc/agreement.py:227-236` loops `for name in names` and calls the **same** `llm` each time, changing
only the annotator name inside the prompt. The docstring says "each in a fresh context", which is true
of the prompt and false of the annotator: a single model asked the same question three times measures
its own self-consistency, not inter-annotator agreement.

That is not a small difference. κ between two copies of one model is inflated by the model's own
biases, which is exactly what a reader would want κ to detect. Reporting it as `annotator_agreement`
tells a reader that independent humans agreed when nothing independent was involved.

What the fix must do
--------------------
* allow each annotator to be served by a different model — that is the only thing that makes the
  labels independent;
* when they are not independent (one model for all names), say so in the report, because the number
  then means "self-consistency" and a reader must not read it as agreement;
* keep the single-model path working, since a soft gate should still run when only one model exists.

The distinction is recorded, not silently smoothed over: the report gains
`independence: "distinct_models" | "single_model"`.
"""

from __future__ import annotations

from typing import Any, Sequence

from cbb_stage1.qc.agreement import AgreementReport, independent_label


class CountingLLM:
    """One model. Records which prompt it saw."""

    def __init__(self, verdict: str = "support") -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.verdict


def _labels(report: AgreementReport) -> list[tuple[str, str]]:
    return [(label.annotator, label.verdict) for label in report.labels]


# ══ distinct models make the labels independent ═══════════════════════
def test_each_annotator_can_have_its_own_model() -> None:
    models = {"A": CountingLLM("support"), "B": CountingLLM("against"), "C": CountingLLM("support")}
    report = independent_label(["cl:1"], llm=models, claims={"cl:1": "某主张"})

    assert report.status == "measured"
    seen = {name: len(model.prompts) for name, model in models.items()}
    assert all(count == 1 for count in seen.values()), (
        f"not every annotator was asked exactly once: {seen}"
    )


def test_distinct_models_are_reported_as_independent() -> None:
    models = {"A": CountingLLM(), "B": CountingLLM(), "C": CountingLLM()}
    report = independent_label(["cl:1"], llm=models, claims={"cl:1": "某主张"})
    assert report.independence == "distinct_models", (
        f"the report does not say the labels came from independent models: {report.independence!r}"
    )


def test_a_shared_model_is_reported_as_self_consistency() -> None:
    """One model for all names must not be presented as inter-annotator agreement."""
    report = independent_label(["cl:1"], llm=CountingLLM(), claims={"cl:1": "某主张"})
    assert report.independence == "single_model", (
        f"three prompts to one model were reported as {report.independence!r}"
    )


def test_a_shared_model_still_produces_a_report() -> None:
    """It is self-consistency; with one constant model kappa is undefined, and None is the honest
    answer (chance agreement is 1, so kappa would be 0/0)."""
    report = independent_label(["cl:1", "cl:2"], llm=CountingLLM(), claims={"cl:1": "甲", "cl:2": "乙"})
    assert report.status == "measured"
    assert report.labels, "no labels were produced"
    assert report.kappa is None


# ══ the report carries the caveat into the box ═══════════════════════
def test_the_soft_gate_payload_states_the_independence() -> None:
    # DEFAULT_ANNOTATORS has three names; a mapping must cover all of them.
    models = {"A": CountingLLM(), "B": CountingLLM(), "C": CountingLLM()}
    report = independent_label(["cl:1"], llm=models, claims={"cl:1": "某主张"})
    payload = report.soft_gate()
    assert payload.get("annotator_independence") == "distinct_models", (
        f"the box payload drops the independence caveat: {payload}"
    )


def test_the_dict_payload_states_the_independence() -> None:
    report = independent_label(["cl:1"], llm=CountingLLM(), claims={"cl:1": "某主张"})
    assert report.as_dict().get("independence") == "single_model"


# ══ a partial mapping is an error, not a silent fallback ═════════════
def test_a_mapping_missing_an_annotator_is_rejected() -> None:
    """Silently reusing one model for the missing name would fake independence."""
    import pytest

    models = {"A": CountingLLM(), "B": CountingLLM()}
    with pytest.raises(ValueError, match="missing"):
        independent_label(
            ["cl:1"], llm=models, claims={"cl:1": "某主张"}, annotators=("A", "B", "C")
        )


# ══ the single-model path is unchanged ══════════════════════════════
def test_no_model_is_still_unavailable() -> None:
    report = independent_label(["cl:1"], llm=None)
    assert report.status == "unavailable"


def test_one_annotator_is_still_unavailable() -> None:
    report = independent_label(["cl:1"], llm=CountingLLM(), annotators=("A",))
    assert report.status == "unavailable"
