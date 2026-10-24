"""CoVe — Chain-of-Verification (A09-CoVe-2024).

The paper's four steps, verbatim:

    1. Generate Baseline Response: Given a query, generate the response using the LLM.
    2. Plan Verifications: Given both query and baseline response, generate a list of
       verification questions that could help to self-analyze if there are any mistakes
       in the original response.
    3. Execute Verifications: Answer each verification question in turn, and hence check
       the answer against the original response to check for inconsistencies or mistakes.
    4. Generate Final Verified Response: Given the discovered inconsistencies (if any),
       generate a revised response incorporating the verification results.

The step that does the work
---------------------------
> The verification questions generated from planning are answered in the second step,
> where **crucially the context given to the LLM prompt only contains the questions, and
> not the original baseline response** and hence cannot repeat those answers directly.

Without that constraint the loop is "ask the model the same thing twice": it re-reads
its own claim in the context and agrees with it. So :func:`answer_independently` builds
its prompt from the question and the source evidence only — never from the claim.

Why the factored variant
------------------------
The paper compares `joint`, `2-step` and `factored` execution for step 3. Factored answers
each question as its own prompt, which the paper notes removes interference not only from
the baseline but **between answer contexts** too. It costs one call per question, so the
caller opts in and :data:`DEFAULT_MAX_QUESTIONS` stays small.

Where it sits
-------------
The extraction stage already adjudicates *evidence sufficiency* (B6, ``evidence_sufficiency``)
and *attribution* (AVeriTeC quote verification). CoVe answers a third question: given the
evidence, would a set of independent questions expose a mistake in the claim text? It is a
check on the claim's own wording, which neither of the other two covers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "DEFAULT_MAX_QUESTIONS",
    "METHOD",
    "CoVeResult",
    "VerificationQuestion",
    "answer_independently",
    "compare_with_claim",
    "cove_verify",
    "plan_verifications",
]

logger = logging.getLogger(__name__)

#: Named so a reader can tell what produced the verdict.
METHOD = "cove_4step"

#: The paper plans "a list"; 4 keeps the cost at N+1 calls per claim and is what the
#: tests pin. Callers may override.
DEFAULT_MAX_QUESTIONS = 4

PLAN_PROMPT = """\
下面是一条从小说里抽出的主张，以及支撑它的原文证据。请提出 {n} 个**验证问题**，
用来发现这条主张可能出错的地方。

要求：
- 每个问题要能被原文证据**直接回答**（问"有几条腿"而不是"作者想表达什么"）；
- 问题要针对主张里的具体事实（人物、地点、数量、因果），不要问笼统印象；
- 只输出问题本身，一行一个，不要编号以外的解释。

主张：
{claim}

原文证据：
{evidence}
"""

#: The answer prompt carries the question and the evidence — and deliberately NOT the
#: claim. The docstring above explains why; the test suite enforces it.
ANSWER_PROMPT = """\
只依据下面的原文证据回答这个问题。不要猜测，证据里没有就说"证据未提及"。

问题：{question}

原文证据：
{evidence}
"""

COMPARE_PROMPT = """\
下面是一条主张，以及一组**不看主张、只看原文证据**独立作答的验证问答。
请找出这些回答与主张**相矛盾**的地方。

规则：
- 回答只是"证据未提及"不算矛盾，那是证据不足，不是主张错；
- 只挑出主张说了、而回答显示**不是那样**的；
- 只输出矛盾项，一行一个，用「Q<编号>：<一句话说哪里矛盾>」；
- 没有矛盾就只输出 `无矛盾`。

主张：
{claim}

验证问答：
{answers}
"""

REVISE_PROMPT = """\
下面是一条主张、它依据的证据，以及一组独立作答的验证问答。请判断主张与这些回答
是否矛盾，并给出最终版本。

规则：
- 全部一致：原样返回主张，不要改写；
- 有矛盾：只改正矛盾的部分，其余保留；
- 只输出最终主张文本，不要解释。

主张：
{claim}

独立作答：
{answers}
"""


@dataclass(frozen=True)
class VerificationQuestion:
    """One verification question and the answer given without seeing the claim."""

    qid: str
    text: str
    answer: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"qid": self.qid, "text": self.text, "answer": self.answer}


@dataclass(frozen=True)
class CoVeResult:
    """Outcome of the four-step loop."""

    status: str  # verified | revised | unavailable
    method: str = METHOD
    questions: tuple[VerificationQuestion, ...] = ()
    inconsistencies: tuple[str, ...] = ()
    revised: str = ""
    verified_by: str = METHOD

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "questions": [item.as_dict() for item in self.questions],
            "inconsistencies": list(self.inconsistencies),
            "revised": self.revised,
            "verified_by": self.verified_by,
        }


def _ask(llm: Any, prompt: str) -> str:
    """One model call. A model without ``generate`` is treated as absent."""
    if llm is None:
        return ""
    generate = getattr(llm, "generate", None)
    if not callable(generate):
        return ""
    try:
        return str(generate(prompt) or "").strip()
    except Exception as exc:  # noqa: BLE001 - a failed step must not abort the stage
        logger.warning("cove: model call failed: %s: %s", type(exc).__name__, exc)
        return ""


def _evidence_block(evidence: Iterable[Any]) -> str:
    lines: list[str] = []
    for item in evidence or ():
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("quote") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if lines else "（无）"


def _clean_question(line: str) -> str:
    text = line.strip().lstrip("-•*· ").strip()
    # drop a leading "1." / "1)" / "（1）" numbering
    for marker in (".", ")", "）", "、"):
        head, sep, tail = text.partition(marker)
        if sep and head.isdigit():
            return tail.strip()
    return text


def plan_verifications(
    claim: str,
    *,
    llm: Any = None,
    evidence: Sequence[Any] = (),
    max_questions: int = DEFAULT_MAX_QUESTIONS,
) -> tuple[str, ...]:
    """Step 2: turn a claim into verification questions.

    Conditions on the claim on purpose — the questions have to target its weak points.
    """
    if not str(claim or "").strip():
        return ()
    count = max(1, int(max_questions or DEFAULT_MAX_QUESTIONS))
    prompt = PLAN_PROMPT.format(
        n=count, claim=claim.strip(), evidence=_evidence_block(evidence)
    )
    raw = _ask(llm, prompt)
    if not raw:
        return ()
    questions: list[str] = []
    for line in raw.splitlines():
        text = _clean_question(line)
        if text:
            questions.append(text)
        if len(questions) >= count:
            break
    return tuple(questions)


def answer_independently(
    question: str,
    *,
    llm: Any = None,
    evidence: Sequence[Any] = (),
) -> str:
    """Step 3: answer one question **without the claim in context**.

    This is the paper's central constraint and the reason the check is not circular.
    """
    if not str(question or "").strip():
        return ""
    prompt = ANSWER_PROMPT.format(
        question=question.strip(), evidence=_evidence_block(evidence)
    )
    return _ask(llm, prompt)


def compare_with_claim(
    claim: str,
    answers: Sequence[VerificationQuestion],
    *,
    llm: Any = None,
) -> tuple[str, ...]:
    """Step 3's comparison: which independent answers contradict the claim.

    One call for the whole set rather than one per pair: the answers are already
    independent of each other, so batching the comparison cannot let them interfere the
    way batching the *answers* would.

    A first version only looked for answers saying "证据未提及". That is an evidence-gap
    signal, not a contradiction — the paper asks whether the answer shows the baseline is
    *wrong*, so the comparison has to be made rather than pattern-matched.
    """
    usable = [item for item in answers if item.answer]
    if not usable or llm is None:
        return ()
    answers_block = "\n".join(
        f"Q{index}：{item.text}\nA{index}：{item.answer}"
        for index, item in enumerate(usable, 1)
    )
    raw = _ask(llm, COMPARE_PROMPT.format(claim=claim, answers=answers_block))
    if not raw or "无矛盾" in raw:
        return ()
    found: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or "无矛盾" in text:
            continue
        upper = text.upper()
        for index, item in enumerate(usable, 1):
            if f"Q{index}" in upper and item.qid not in found:
                found.append(item.qid)
                break
    return tuple(found)


def cove_verify(
    claim: str,
    *,
    llm: Any = None,
    evidence: Sequence[Any] = (),
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    workers: int = 1,
) -> CoVeResult:
    """Run the four steps on one claim.

    Step 1 is the caller's: the claim already exists by the time this is called. Steps 2
    and 3 run here; step 4 runs only when step 3 produced a signal worth acting on.

    Cost is ``2 + questions`` calls (plan, compare, one per question), plus one revision
    call when the comparison found a contradiction. Without a model the result is
    ``unavailable`` — never a fabricated ``verified``.

    The per-question answers are the part the paper says can be batched (A09, p.4: "they can be run
    in parallel, and hence be batched"). ``workers > 1`` answers them concurrently; the results still
    come back in question order, so the comparison step and the recorded questions are unaffected.
    The default stays ``1`` so every existing caller keeps identical behaviour.
    """
    text = str(claim or "").strip()
    if not text or llm is None:
        return CoVeResult(status="unavailable")

    raw_questions = plan_verifications(
        text, llm=llm, evidence=evidence, max_questions=max_questions
    )
    if not raw_questions:
        return CoVeResult(status="unavailable")

    def _answer_one(item: tuple[int, str]) -> VerificationQuestion:
        index, question = item
        answer = answer_independently(question, llm=llm, evidence=evidence)
        return VerificationQuestion(qid=f"q{index}", text=question, answer=answer)

    # Local import: the parallel helper lives under `orchestrator`, and importing it at module scope
    # would make the validate package depend on stage orchestration just for a thread pool.
    from cbb_stage1.orchestrator.parallel import run_ordered

    answered: list[VerificationQuestion] = run_ordered(
        list(enumerate(raw_questions, 1)),
        _answer_one,
        workers=int(workers or 1),
        thread_name_prefix="cove",
    )

    inconsistencies = compare_with_claim(text, answered, llm=llm)
    if not inconsistencies:
        return CoVeResult(
            status="verified",
            questions=tuple(answered),
            inconsistencies=(),
            revised=text,
        )

    answers_block = "\n".join(
        f"问：{item.text}\n答：{item.answer}" for item in answered
    )
    revised = _ask(
        llm,
        REVISE_PROMPT.format(claim=text, answers=answers_block),
    )
    return CoVeResult(
        status="revised",
        questions=tuple(answered),
        inconsistencies=inconsistencies,
        revised=revised or text,
    )
