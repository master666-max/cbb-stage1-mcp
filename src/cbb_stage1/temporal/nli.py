"""``nli_pairwise`` interface for timeline verification (design/01 §5.8, I25).

Design intent (TimeSET absorbed in v2.2): the solver does **not** call a model.
It offers a narrow pairwise NLI question interface and lets the caller inject an
implementation (mock in tests, cheap model in production). The default
implementation deliberately returns ``unknown`` and never performs a network
call -- stage 1 is runnable offline and a missing verifier must be visible as
``unknown``, not silently treated as confirmation.

Result contract: ``{method, question, result, verifier, model}`` with
``method="nli_pairwise"`` and ``result in {"yes","no","unknown"}`` (I25).
A contradictory result becomes a conflict record; it never overwrites the
constraint (design/01 §5.8).

Pure functions only: no I/O, no clock, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

VERIFY_METHODS = ("nli_pairwise", "explicit_expression", "topological")
VERIFY_RESULTS = ("yes", "no", "unknown")

DEFAULT_QUESTION_BY_KIND = {
    "before": "{left} 是否先于 {right}？",
    "after": "{left} 是否晚于 {right}？",
    "during": "{left} 是否发生在 {right} 期间？",
    "overlap": "{left} 与 {right} 在时间上是否有重叠？",
}


@dataclass(frozen=True)
class NliResult:
    """One pairwise verdict. ``method``/``result`` are closed-set checked."""

    question: str
    result: str = "unknown"
    method: str = "nli_pairwise"
    verifier: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if self.method not in VERIFY_METHODS:
            raise ValueError("verify.method 非法：%r（闭集 %s）" % (self.method, VERIFY_METHODS))
        if self.result not in VERIFY_RESULTS:
            raise ValueError("verify.result 非法：%r（闭集 %s）" % (self.result, VERIFY_RESULTS))

    def to_record(self) -> dict[str, Any]:
        return {"method": self.method, "question": self.question, "result": self.result,
                "verifier": self.verifier, "model": self.model}


# ``impl`` may return NliResult, a mapping, or a bare string in VERIFY_RESULTS.
NliImpl = Callable[..., Any]


def question_for(kind: str, left_id: str, right_id: str) -> str:
    """The fixed Question per constraint kind (keeps prompts comparable)."""
    template = DEFAULT_QUESTION_BY_KIND.get(kind)
    if template is None:
        raise ValueError("未知时序约束 kind：%r（闭集 %s）" % (kind, tuple(DEFAULT_QUESTION_BY_KIND)))
    return template.format(left=left_id, right=right_id)


def normalize_result(raw: Any, *, question: str, verifier: str | None = None,
                     model: str | None = None) -> NliResult:
    """Coerce an injected implementation's return value into :class:`NliResult`."""
    if isinstance(raw, NliResult):
        return raw
    if isinstance(raw, str):
        return NliResult(question=question, result=raw, verifier=verifier, model=model)
    if isinstance(raw, Mapping):
        return NliResult(
            question=str(raw.get("question") or question),
            result=str(raw.get("result", "unknown")),
            method=str(raw.get("method") or "nli_pairwise"),
            verifier=raw.get("verifier", verifier),
            model=raw.get("model", model),
        )
    raise TypeError("nli_pairwise 返回值必须是 NliResult／Mapping／字符串，现得 %r" % (raw,))


def nli_pairwise(left: Any, right: Any, question: str, *, impl: NliImpl | None = None,
                 verifier: str | None = None, model: str | None = None) -> NliResult:
    """Ask one pairwise question; ``left``/``right`` are node ids (or node dicts).

    With ``impl=None`` the verifier is *disabled*: result ``unknown`` and
    ``verifier="disabled"`` so the caller can record "未验证" explicitly.
    """
    left_id = _node_id(left)
    right_id = _node_id(right)
    if impl is None:
        return NliResult(question=question, result="unknown", verifier="disabled")
    raw = impl(left_id, right_id, question)
    return normalize_result(raw, question=question, verifier=verifier, model=model)


def _node_id(node: Any) -> str:
    if isinstance(node, Mapping):
        return str(node.get("id"))
    return str(node)


def expected_answer(kind: str) -> str:
    """Contradiction rule: for every supported kind, a ``yes`` is consistent."""
    if kind not in DEFAULT_QUESTION_BY_KIND:
        raise ValueError("未知时序约束 kind：%r" % (kind,))
    return "yes"


def verify_constraint(constraint: Any, impl: NliImpl | None = None) -> tuple[dict[str, Any], bool]:
    """Run pairwise verification for one constraint.

    Returns ``(verify_record, contradicted)``. ``contradicted`` is True only for
    a definite ``no``; ``unknown`` is recorded as unverified (never "passed").
    """
    kind = constraint.kind
    question = question_for(kind, constraint.left.id, constraint.right.id)
    result = nli_pairwise(constraint.left, constraint.right, question, impl=impl)
    contradicted = result.result == "no"
    return result.to_record(), contradicted
