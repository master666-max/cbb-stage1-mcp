"""Quote verification for speaker attribution (AVeriTeC's method, H04).

What the paper says
-------------------
AVeriTeC structures a claim as metadata plus evidence:

    Claim: ...    Date: ...    Speaker: Morgan Griffith    Type: ...
    Q1: ... A1: ...    Verdict: ...    Justification: ...

and names one task explicitly:

> Quote Verification. The primary task is to identify whether a quote was actually
> said by the supposed speaker.

It also separates two notions that this project had been conflating:

> The speaker of the original claim, e.g. the person or organization who made the
> claim. The source of the original claim, e.g. the person or organization who
> published the claim. This is not necessarily the same as the speaker; a person might
> make a comment in a newspaper, in which case the person is the speaker and the
> newspaper is the source.

What this module does
---------------------
Given a quote, the speaker claimed for it, and a little surrounding text, decide
whether the quote was said by that speaker. The decision is recorded as a verdict; it
never deletes an attribution or an assertion.

Why a cast miss is `unsure` and not `against`
---------------------------------------------
The entity roster available at verification time is derived from the resolved tables,
but a speaker can legitimately be introduced in an earlier unit and not appear as an
entity in the current one. Measured while auditing the removed cast check: it was
per-unit, so a correct cross-chapter attribution looked like an unknown name. Treating
that as `against` would delete a correct speaker. AVeriTeC's own treatment of
insufficient material is `unverifiable`, not `refuted`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "SPEAKER_VERDICTS",
    "SpeakerVerdict",
    "build_speaker_prompt",
    "parse_speaker_verdict",
    "verify_speaker_quote",
]

#: Closed verdict set; mirrors the box's ``verification.verdict``.
SPEAKER_VERDICTS = ("support", "against", "unsure")

#: Default context window, in characters, taken on each side of the quote.
CONTEXT_WINDOW = 200

PROMPT_TEMPLATE = """\
判断下面这段原文里的引文，是不是「{speaker}」这个人说的。

判断依据（按顺序考虑）：
1. 引文前后有没有「{speaker}说/道/问/答/喊」这类明确的言语归属；
2. 有没有明显的问答关系——前文有人问 {speaker}，紧接着是这句引文；
3. 引文里有没有自称或称呼能指认说话人。

只有当原文**明确**支持「{speaker}说的」时，才答 support。
如果原文明确显示是**别人**说的，答 against。
其余情况（没写谁说的、看不出、无法判断）一律答 unsure。

只输出 JSON，不要解释：
{{"verdict": "support|against|unsure", "rationale": "一句话理由"}}

原文上下文：
{context}

引文：
{quote}
"""


@dataclass(frozen=True)
class SpeakerVerdict:
    """One attribution check result."""

    verdict: str
    rationale: str
    method: str
    #: True when the check was made by the same model that produced the record, which
    #: §5.15 would not accept as independent verification.
    same_model: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "speaker_check": self.method,
            "speaker_check_same_model": self.same_model,
        }


def build_speaker_prompt(
    *,
    quote: str,
    speaker: str,
    context: str,
) -> str:
    return PROMPT_TEMPLATE.format(
        speaker=speaker,
        context=" ".join(str(context or "").split()),
        quote=" ".join(str(quote or "").split()),
    )


def parse_speaker_verdict(raw: Any) -> tuple[str, str]:
    """Extract ``(verdict, rationale)`` from a model answer.

    Accepts a JSON object (preferred) or a bare word. Anything unrecognisable becomes
    ``unsure`` — a verifier that rambles has not verified anything.
    """
    text = str(raw or "").strip()
    if not text:
        return "unsure", "empty answer"

    payload: Any = None
    try:
        import json

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(text[start: end + 1])
    except (ValueError, TypeError):
        payload = None

    if isinstance(payload, Mapping):
        verdict = str(payload.get("verdict") or "").strip().lower()
        rationale = str(payload.get("rationale") or "").strip()
        if verdict in SPEAKER_VERDICTS:
            return verdict, rationale
        return "unsure", rationale or f"unrecognised verdict {verdict!r}"

    lowered = text.lower()
    for candidate in SPEAKER_VERDICTS:
        if re.search(rf"\b{candidate}\b", lowered):
            return candidate, text[:120]
    return "unsure", text[:120]


def verify_speaker_quote(
    *,
    quote: str,
    speaker_name: str,
    context: str = "",
    cast: set[str] | None = None,
    llm: Any = None,
    cache: dict[tuple[str, str], SpeakerVerdict] | None = None,
    same_model: bool = False,
) -> SpeakerVerdict:
    """Decide whether ``quote`` was said by ``speaker_name``.

    The order of the checks is fixed and each short-circuit is deliberate:

    1. no speaker -> nothing to verify;
    2. speaker outside the cast -> ``unsure``. The roster is not authoritative for
       cross-unit attribution, so a miss is not evidence of a wrong attribution;
    3. no verifier -> ``unsure``. Never fabricate ``support``;
    4. otherwise ask the model and parse its answer.
    """
    name = str(speaker_name or "").strip()
    text = str(quote or "").strip()

    if not name or not text:
        return SpeakerVerdict(
            verdict="unsure",
            rationale="no speaker or no quote to check",
            method="no_speaker",
            same_model=same_model,
        )

    if cast is not None and name not in cast:
        return SpeakerVerdict(
            verdict="unsure",
            rationale=(
                f"speaker {name!r} is not in the resolved roster; the roster does not "
                "cover cross-unit speakers, so this is not evidence against"
            ),
            method="cast_miss",
            same_model=same_model,
        )

    key = (text, name)
    if cache is not None and key in cache:
        return cache[key]

    if llm is None:
        return SpeakerVerdict(
            verdict="unsure",
            rationale="no verifier available",
            method="no_verifier",
            same_model=same_model,
        )

    prompt = build_speaker_prompt(quote=text, speaker=name, context=context)
    try:
        raw = llm.generate(prompt)
    except Exception as exc:  # noqa: BLE001 - a failed check is unsure, not a verdict
        return SpeakerVerdict(
            verdict="unsure",
            rationale=f"verifier failed: {type(exc).__name__}: {exc}",
            method="llm_error",
            same_model=same_model,
        )

    verdict, rationale = parse_speaker_verdict(raw)
    result = SpeakerVerdict(
        verdict=verdict,
        rationale=rationale,
        method="llm",
        same_model=same_model,
    )
    if cache is not None:
        cache[key] = result
    return result
