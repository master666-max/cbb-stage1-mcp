"""B11 — independent labelling and multi-annotator agreement.

The inventory asks for two things:

* **多标注者不复用同一 claim** — several labellers judge the same claim without seeing
  each other's verdicts;
* **第三人证据充分性检查** — a third party checks the evidence, not just the claim.

What this is not
----------------
It is not C3. C3 gives one verifier its own context, which answers "is the verifier
independent of the extractor". B11 answers a different question: "would two competent
readers reach the same verdict". One reader cannot disagree with itself, so C3 cannot
substitute for this no matter how clean its context is.

Why κ and not raw agreement
---------------------------
Raw agreement is inflated by class imbalance. A sample that is 90% `support` gives two
independent labellers 82% raw agreement by chance, and the number looks fine while carrying
no information. Cohen's κ subtracts chance agreement, so the reported number means
something.

Why it is a soft gate
---------------------
κ on a 20-claim sample is noisy: a handful of flips moves it by tenths. Putting it in
``hard_gates`` would fail builds at random, and a gate that fails at random gets switched
off — which is worse than not having it. :meth:`AgreementReport.soft_gate` exists so the
number lands where a reader sees it without gating on it.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_ANNOTATORS",
    "DEFAULT_SAMPLE_SIZE",
    "METHOD",
    "AgreementReport",
    "Label",
    "cohen_kappa",
    "independent_label",
]

logger = logging.getLogger(__name__)

#: Named so a reader can tell what produced the number.
METHOD = "independent_multi_annotator"

#: Three readers is the smallest number that can show one outlier disagreeing with a
#: pair; two readers only tell you they differ.
DEFAULT_ANNOTATORS: tuple[str, ...] = ("A", "B", "C")

#: Sampling, not a full pass. κ over thousands of claims costs far more than it tells.
DEFAULT_SAMPLE_SIZE = 20

LABEL_PROMPT = """\
你是标注者：{annotator}。请只依据下面的原文证据，判断这条主张是否成立。

判定只能取三值之一：
- support —— 证据支持这条主张
- against —— 证据与这条主张矛盾
- unsure  —— 证据不足或看不出

只输出一个词（support / against / unsure），不要解释。

主张（{claim_id}）：
{claim}

原文证据：
{evidence}
"""


@dataclass(frozen=True)
class Label:
    claim_id: str
    annotator: str
    verdict: str

    def as_dict(self) -> dict[str, str]:
        return {
            "claim_id": self.claim_id,
            "annotator": self.annotator,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class AgreementReport:
    status: str  # measured | unavailable
    kappa: float | None = None
    labels: tuple[Label, ...] = ()
    disagreements: tuple[str, ...] = ()
    annotators: int = 0
    sample: tuple[str, ...] = ()
    method: str = METHOD
    reason: str = ""
    #: ``distinct_models`` means the labels came from different models (real inter-annotator
    #: agreement); ``single_model`` means one model asked several times (self-consistency).
    independence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "kappa": self.kappa,
            "annotators": self.annotators,
            "sample": list(self.sample),
            "sample_size": len(self.sample),
            "disagreements": list(self.disagreements),
            "labels": [label.as_dict() for label in self.labels],
            "reason": self.reason,
            "independence": self.independence,
        }

    def soft_gate(self) -> dict[str, Any]:
        """The shape that goes into ``qc/report.json``'s ``soft_gates``.

        Soft on purpose — see the module docstring. A shaky statistic that blocks a build
        stops being run.
        """
        return {
            "annotator_agreement_kappa": self.kappa,
            "annotator_sample": len(self.sample),
            "annotators": self.annotators,
            "disagreements": len(self.disagreements),
            # The caveat travels with the number: a κ from one model is self-consistency.
            "annotator_independence": self.independence,
        }


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Cohen's κ for two label sequences.

    ``None`` when it is not defined — an empty sample, or a sample where both labellers
    used a single category (then chance agreement is 1 and κ would be 0/0). Returning a
    number there would be inventing one.
    """
    if len(first) != len(second):
        raise ValueError(
            f"label sequences differ in length: {len(first)} vs {len(second)}"
        )
    total = len(first)
    if total == 0:
        return None

    observed = sum(1 for a, b in zip(first, second) if a == b) / total
    first_counts = Counter(first)
    second_counts = Counter(second)
    categories = set(first_counts) | set(second_counts)
    chance = sum(
        (first_counts[c] / total) * (second_counts[c] / total) for c in categories
    )
    if chance >= 1.0:
        return None
    return round((observed - chance) / (1.0 - chance), 6)


def _evidence_block(items: Iterable[Any]) -> str:
    lines: list[str] = []
    for item in items or ():
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, Mapping):
            text = str(item.get("quote") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if lines else "（无）"


def _normalise_verdict(raw: str) -> str:
    lowered = str(raw or "").strip().lower()
    for candidate in ("support", "against", "unsure"):
        if candidate in lowered:
            return candidate
    return "unsure"


def _ask(llm: Any, prompt: str) -> str:
    generate = getattr(llm, "generate", None)
    if not callable(generate):
        return ""
    try:
        return str(generate(prompt) or "").strip()
    except Exception as exc:  # noqa: BLE001 - a failed label is unsure, not a crash
        logger.warning("agreement: label call failed: %s: %s", type(exc).__name__, exc)
        return ""


def _models_by_annotator(
    llm: Any, names: Sequence[str]
) -> dict[str, Any]:
    """Resolve ``llm`` into one model per annotator.

    A mapping must cover every requested name: silently reusing one model for a missing name would
    produce a number that looks independent and is not.
    """
    if not isinstance(llm, Mapping):
        return {name: llm for name in names}
    missing = [name for name in names if llm.get(name) is None]
    if missing:
        raise ValueError(
            f"llm mapping is missing annotator(s) {missing}; every annotator needs its own model or "
            "the labels are not independent"
        )
    return {name: llm[name] for name in names}


def independent_label(
    claim_ids: Sequence[str],
    *,
    llm: Any = None,
    claims: Mapping[str, str] | None = None,
    evidence_by_claim: Mapping[str, Sequence[Any]] | None = None,
    annotators: Sequence[str] = DEFAULT_ANNOTATORS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> AgreementReport:
    """Have several labellers judge the same claims, each in a fresh context.

    Each prompt names its own annotator and no other, and carries no previous verdict.

    ``llm`` may be one model or a ``{annotator: model}`` mapping. **Only the mapping gives
    independence**: one model asked three times measures its own self-consistency, and the report says
    which of the two happened (`independence`). Without that distinction a κ of 0.8 from a single model
    reads as "three readers agreed", which is the opposite of what it means.
    """
    names = tuple(str(name) for name in (annotators or ()) if str(name).strip())
    if llm is None or len(names) < 2:
        return AgreementReport(
            status="unavailable",
            annotators=len(names),
            reason=(
                "no model"
                if llm is None
                else "fewer than two annotators cannot disagree with each other"
            ),
        )

    limit = max(1, int(sample_size or DEFAULT_SAMPLE_SIZE))
    sample = tuple(str(cid) for cid in (claim_ids or ())[:limit] if str(cid).strip())
    if not sample:
        return AgreementReport(status="unavailable", annotators=len(names), reason="empty sample")

    texts = claims or {}
    evidence_map = evidence_by_claim or {}
    models = _models_by_annotator(llm, names)
    independence = (
        "distinct_models" if isinstance(llm, Mapping) else "single_model"
    )
    labels: list[Label] = []
    for claim_id in sample:
        claim_text = str(texts.get(claim_id) or "").strip()
        evidence = evidence_map.get(claim_id) or ()
        for name in names:
            prompt = LABEL_PROMPT.format(
                annotator=name,
                claim_id=claim_id,
                claim=claim_text or "（主张文本缺失）",
                evidence=_evidence_block(evidence),
            )
            labels.append(
                Label(
                    claim_id=claim_id,
                    annotator=name,
                    verdict=_normalise_verdict(_ask(models[name], prompt)),
                )
            )

    # pairwise κ against the first labeller, then average: the paper-level metric is
    # agreement between readers, and there is no single-pair convention for three.
    by_annotator: dict[str, dict[str, str]] = {name: {} for name in names}
    for label in labels:
        by_annotator[label.annotator][label.claim_id] = label.verdict

    scores: list[float] = []
    baseline = names[0]
    for name in names[1:]:
        first = [by_annotator[baseline][cid] for cid in sample]
        second = [by_annotator[name][cid] for cid in sample]
        value = cohen_kappa(first, second)
        if value is not None:
            scores.append(value)
    kappa = round(sum(scores) / len(scores), 6) if scores else None

    disagreements = tuple(
        cid
        for cid in sample
        if len({by_annotator[name][cid] for name in names}) > 1
    )

    return AgreementReport(
        status="measured",
        kappa=kappa,
        labels=tuple(labels),
        disagreements=disagreements,
        annotators=len(names),
        sample=sample,
        independence=independence,
    )
