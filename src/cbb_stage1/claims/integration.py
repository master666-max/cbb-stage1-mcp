"""P0-B integration for claim QA (``run`` contract).

Signature::

    run(work_root, *, book_id, outputs, units, created_at, upstream) -> dict

Reads the statements assembled by :mod:`cbb_stage1.assemble.integration`
(``assemble/assertions.jsonl``) and writes ``assemble/claims.jsonl``.

Every ``claim_layer=statement`` assertion becomes a claim:

* when ``upstream["claims_qa"]`` carries question/answer pairs for the statement
  (verifier output), they are validated and used;
* otherwise the claim is built with a single ``unanswerable`` question, which
  deterministically yields ``insufficient_evidence`` ("无 QA 用
  unanswerable/insufficient" -- never invent answers).

The fact-promotion gate from :mod:`cbb_stage1.claims.verdicts` is applied to
every claim: only a ``supported`` claim with >= 2 independent sources may
produce a derived fact row.  ``insufficient_evidence`` never promotes.  Promoted
fact rows go to ``assemble/promoted_facts.jsonl`` (this stage never mutates
``assertions.jsonl``; wiring the promoted facts into the knowledge table is the
Lead/box step).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import ValidationError

from cbb_stage1.models import Assertion
from cbb_stage1.orchestrator.parallel import run_ordered, stage_workers

from .qa import ClaimBuildError, build_claim, make_question, normalize_claim_text
from .verdicts import (
    DEFAULT_ABSTAIN_THRESHOLD,
    ClaimRuleError,
    PromotionRefused,
    abstain_decision,
    build_promoted_fact,
    evidence_sufficiency,
    promotion_decision,
)

__all__ = ["STAGE", "claims_path", "promoted_facts_path", "run"]

STAGE = "claims"

#: library (six-library) -> claim_type (§5.19 enum).
_LIBRARY_CLAIM_TYPE: dict[str, str] = {
    "实体": "attribute",
    "关系": "relation",
    "事件": "event",
    "埋线": "setting",
    "设定": "setting",
    "时序": "event",
}


# ── shape/path helpers ─────────────────────────────────────────


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    raise TypeError(f"expected mapping/model/unit record, got {type(value).__name__}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_text_atomic(path: Path, data: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    data = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    return _write_text_atomic(path, data)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover
        return path.as_posix()


def claims_path(work_root: str | Path) -> Path:
    return Path(work_root) / "assemble" / "claims.jsonl"


def promoted_facts_path(work_root: str | Path) -> Path:
    return Path(work_root) / "assemble" / "promoted_facts.jsonl"


# ── assert input / QA helpers ──────────────────────────────────


def _load_assertions(
    root: Path, upstream: Mapping[str, Any]
) -> tuple[list[Assertion], str, list[dict[str, Any]]]:
    """Load assembled assertions from the work table (upstream fallback)."""
    path = root / "assemble" / "assertions.jsonl"
    rows: list[dict[str, Any]] = []
    source = "none"
    if path.exists():
        rows = _read_jsonl(path)
        source = _rel(root, path)
    else:
        for key in ("assertions",):
            candidate = upstream.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                rows = [dict(row) for row in candidate if isinstance(row, Mapping)]
                if rows:
                    source = f"upstream:{key}"
                    break
        if not rows:
            assemble = upstream.get("assemble")
            if isinstance(assemble, Mapping) and isinstance(assemble.get("assertions"), Sequence):
                rows = [dict(row) for row in assemble["assertions"] if isinstance(row, Mapping)]
                if rows:
                    source = "upstream:assemble.assertions"
    statements: list[Assertion] = []
    refused: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if str(row.get("claim_layer") or "") != "statement":
            continue
        try:
            statements.append(Assertion.model_validate(row))
        except (ValidationError, TypeError, ValueError) as exc:
            refused.append(
                {
                    "statement_assertion_id": str(row.get("assertion_id") or ""),
                    "reason": "schema",
                    "detail": f"statement row rejected: {exc}",
                }
            )
    return statements, source, refused


def _load_unit_texts(root: Path, units: Sequence[Any]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for unit in units or ():
        try:
            record = _as_dict(unit)
        except TypeError:
            continue
        unit_id = str(record.get("unit_id") or "")
        if not unit_id:
            continue
        inline = record.get("text")
        if isinstance(inline, str) and inline:
            texts[unit_id] = inline
            continue
        text_path = record.get("text_path")
        if not text_path:
            continue
        candidate = root / str(text_path)
        try:
            if candidate.exists():
                texts[unit_id] = candidate.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - defensive
            continue
    return texts


def _qa_map(upstream: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = upstream.get("claims_qa")
    mapping: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw, Mapping):
        for statement_id, questions in raw.items():
            if isinstance(questions, Sequence) and not isinstance(questions, (str, bytes)):
                mapping[str(statement_id)] = [
                    dict(question) for question in questions if isinstance(question, Mapping)
                ]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            statement_id = str(entry.get("statement_assertion_id") or "")
            questions = entry.get("questions") or ()
            if statement_id and isinstance(questions, Sequence):
                mapping[statement_id] = [
                    dict(question) for question in questions if isinstance(question, Mapping)
                ]
    return mapping


def cove_worker_count(upstream: Mapping[str, Any], question_count: int) -> int:
    """How many threads CoVe may use for its independent answers.

    Derived from the stage budget, never hardcoded: a literal would ignore `Config.max_workers` and
    the pipeline's explicit `upstream["max_workers"]`.

    Capped at the budget for a reason that is not stylistic. CoVe runs *inside* a per-statement
    transform, which already occupies a worker thread when the stage is parallel. If each of those
    threads spawned `max_workers` more, the total in-flight model calls would exceed the gateway's
    `BoundedSemaphore` (sized to the stage budget), and the stage would block on itself.

    Returns 1 when the budget is 1, and never more than the budget.
    """
    try:
        budget = int(upstream.get("max_workers") or 1)
    except (TypeError, ValueError):
        budget = 1
    if budget <= 1:
        return 1
    # Leave the outer transform its share: half the budget, at least one.
    share = max(1, budget // 2)
    return max(1, min(share, int(question_count or 1), budget))


def _field(record: Any, name: str, default: Any = None) -> Any:
    """Read one field from a mapping or a model, whichever the caller has."""
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def build_claims_qa(
    statements: Sequence[Any],
    unit_texts: Mapping[str, str],
    names: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Ground each statement in its own evidence and answer it, when the evidence supports it.

    Returns ``{statement_assertion_id: [question]}`` in the shape ``_qa_map`` produces. Statements
    that cannot be grounded get **no entry**, so the caller's unanswerable fallback stays in charge.

    Only ``support`` is produced: ``refute`` requires contradiction analysis, which is the conflict
    detectors' and CoVe's job, and inventing one here would be a downgrade dressed as coverage.

    ``names`` is ``{entity_id: canonical_name}`` from the work layer. It is needed because
    `assemble` has already resolved a statement's roles to entity ids, so without the reverse map a
    grounded statement would look ungrounded.

    An answer must carry ``text, unit_id, start, end, quote`` (``qa.py:203``) — it is a grounded
    span, not a bare verdict word.
    """
    id_to_name = {key: value for key, value in (names or {}).items()}
    answered: dict[str, list[dict[str, Any]]] = {}
    for statement in statements or ():
        statement_id = str(_field(statement, "assertion_id") or "")
        if not statement_id:
            continue
        evidence = [item for item in (_field(statement, "evidence") or []) if item is not None]
        if not evidence:
            continue

        anchored: list[tuple[str, Any]] = []
        for item in evidence:
            unit_id = str(_field(item, "unit_id") or "")
            quote = str(_field(item, "quote") or "")
            text = unit_texts.get(unit_id) or ""
            if not quote or not text or quote not in text:
                # The quote is not in the unit it claims: nothing can be answered from it.
                anchored = []
                break
            anchored.append((text, item))
        if not anchored:
            continue

        roles = _field(statement, "roles") or {}
        surfaces: list[str] = []
        for value in roles.values() if isinstance(roles, Mapping) else ():
            if not isinstance(value, str) or not value.strip():
                continue
            text_value = value.strip()
            surfaces.append(text_value)
            if text_value in id_to_name:
                surfaces.append(str(id_to_name[text_value]))
        if not surfaces:
            continue
        # At least one entity the assertion is about must appear in the quoted text. This is a weak
        # check on purpose: it can only justify `support`, never `refute`.
        quotes = [str(_field(item, "quote") or "") for _text, item in anchored]
        if not any(any(surface in quote for quote in quotes) for surface in surfaces):
            continue

        answers: list[dict[str, Any]] = []
        for text, item in anchored:
            try:
                start = int(_field(item, "start"))
                end = int(_field(item, "end"))
            except (TypeError, ValueError):
                answers = []
                break
            answers.append(
                {
                    "text": text,
                    "unit_id": str(_field(item, "unit_id") or ""),
                    "start": start,
                    "end": end,
                    "quote": str(_field(item, "quote") or ""),
                    "source": "narration",
                }
            )
        if not answers:
            continue

        answered[statement_id] = [
            {
                "qid": f"q:{statement_id}",
                "question": f"原文是否支持这一断言（{_field(statement, 'predicate', '')}）？",
                "answers": answers,
                "stance": "support",
            }
        ]
    return answered


def _unresolved_map(upstream: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = upstream.get("unresolved_entities") or upstream.get("claims_unresolved_entities")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(statement_id): [str(item) for item in values if item]
        for statement_id, values in raw.items()
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
    }


def _question_text(statement: Assertion, names: Mapping[str, str]) -> str:
    text = _statement_text(statement, names).strip()
    if not text:
        text = statement.predicate
    return text if text.endswith("？") else f"{text}？"


def _load_names(root: Path) -> dict[str, str]:
    """``entity_id -> canonical_name`` from the resolve candidates (for readable claims)."""
    names: dict[str, str] = {}
    for row in _read_jsonl(root / "resolve" / "entity_candidates.jsonl"):
        entity_id = str(row.get("entity_id") or "")
        name = str(row.get("canonical_name") or "")
        if entity_id and name:
            names[entity_id] = name
    return names


def _statement_text(statement: Assertion, names: Mapping[str, str]) -> str:
    """Render ``predicate：role values`` using resolved entity names when possible."""
    parts: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            if value.startswith("e:"):
                name = names.get(value)
                if name:
                    parts.append(name)
            elif value:
                parts.append(value)
        elif isinstance(value, Mapping):
            literal = value.get("literal")
            if isinstance(literal, str) and literal:
                parts.append(literal)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                add(inner)

    for value in (statement.roles or {}).values():
        add(value)
    if parts:
        return f"{statement.predicate}：" + " / ".join(parts)
    return normalize_claim_text(statement)


def _claim_type_for(statement: Assertion) -> str:
    return _LIBRARY_CLAIM_TYPE.get(str(statement.library), "attribute")


# ── the stage entry point ─────────────────────────────────────


def _statement_quotes(statement: Any, *, limit: int = 8) -> list[str]:
    """The verbatim quotes supporting a statement assertion.

    Deduplicated and capped: the verifier only needs enough text to judge whether the
    atomic fact is supported, and a very long evidence block makes the prompt harder to
    answer, not easier.
    """
    quotes: list[str] = []
    for item in getattr(statement, "evidence", None) or ():
        quote = str(getattr(item, "quote", "") or "").strip()
        if quote and quote not in quotes:
            quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes


@dataclass
class _StatementOutcome:
    """Everything one statement produces, so the pool never touches shared state.

    ``refused`` and ``errors`` are per-statement lists rather than stage-level appends: their
    order is part of their meaning to a reader, and a thread appending into a shared list
    would interleave them arbitrarily.
    """

    claim_row: dict[str, Any] | None = None
    promoted_row: dict[str, Any] | None = None
    abstain_row: dict[str, Any] | None = None
    promotion: dict[str, Any] | None = None
    refused: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    degenerate: bool = False
    cove_revised: bool = False
    allowed: bool = False


def run(
    work_root: str | Path,
    *,
    book_id: str,
    outputs: list[dict],
    units: list[dict],
    created_at: str,
    upstream: dict,
) -> dict:
    """Build ``assemble/claims.jsonl`` (+ ``promoted_facts.jsonl``) from statements."""
    root = Path(work_root)
    upstream = upstream or {}
    statements, assertions_source, refused = _load_assertions(root, upstream)
    qa_map = _qa_map(upstream)
    unresolved_map = _unresolved_map(upstream)
    unit_texts = _load_unit_texts(root, units or [])
    names = _load_names(root)
    # B6: no producer supplied `claims_qa`, so every claim was built from a single unanswerable
    # question and the four-value rule only ever produced its fourth value. Ground what the
    # statement's own evidence can ground; leave the rest to the unanswerable fallback.
    if not qa_map:
        qa_map = build_claims_qa(statements, unit_texts, names)
    name_to_id = {name: entity_id for entity_id, name in names.items()}
    reasoning_llm = upstream.get("reasoning_llm")

    claim_rows: list[dict[str, Any]] = []
    promoted_rows: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    #: B12: claims that abstained, with the confidence and threshold behind it.
    abstain_rows: list[dict[str, Any]] = []
    #: B12 threshold. Configurable so a run can tighten or relax the gate; the
    #: decision records the value used, so a box never has to guess.
    abstain_threshold = float(
        upstream.get("abstain_threshold") or DEFAULT_ABSTAIN_THRESHOLD
    )
    allowed = 0
    afev_degenerate = 0
    #: Hooks that raised, recorded rather than swallowed: a claim that could not be
    #: verified is information, and it must not look like a claim that simply had no
    #: verdict.
    afev_errors: list[dict[str, Any]] = []
    #: B1 CoVe is opt-in and costs 2 + questions calls per claim.
    wants_cove = bool(
        upstream.get("cove") if isinstance(upstream, Mapping) else False
    )
    cove_llm = upstream.get("llm") if isinstance(upstream, Mapping) else None
    cove_revised = 0
    def _process_statement(statement: Assertion) -> _StatementOutcome:
        """Analyse one statement without touching this stage's shared state.

        Runs on a worker thread when ``workers > 1``, so it must not append to a
        stage-level list, increment a counter, or write a file: everything the
        statement produces comes back in the outcome and is consumed in input order
        below.

        A13-AFEV §4.4 is the reason this is legitimate: "each claim can be processed
        independently, this process is parallel and highly scalable during inference".
        """
        outcome = _StatementOutcome()
        statement_id = str(statement.assertion_id or "")
        questions = qa_map.get(statement_id)
        if questions:
            normalized_questions: list[dict[str, Any]] = []
            for index, question in enumerate(questions):
                item = dict(question)
                item.setdefault("qid", f"q{index + 1}")
                normalized_questions.append(item)
        else:
            normalized_questions = [
                make_question(
                    "q1", _question_text(statement, names), stance="unanswerable"
                ).model_dump()
            ]
        unresolved = unresolved_map.get(statement_id, [])
        try:
            claim = build_claim(
                statement,
                questions=normalized_questions,
                normalized_claim=_statement_text(statement, names),
                claim_type=_claim_type_for(statement),
                book_id=book_id,
                confidence=float(statement.confidence or 0.0) if questions else 0.0,
                confidence_method="claim_qa",
                verifier="integration:claims",
                context_mode="isolated_source" if questions else "isolated_claimed_only",
                unit_texts=unit_texts or None,
                unresolved_entities=unresolved,
            )
        except (ClaimBuildError, ClaimRuleError, ValidationError) as exc:
            outcome.refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": getattr(exc, "reason", "schema"),
                    "detail": str(exc),
                }
            )
            return outcome

        # B6 + B12: decide whether the evidence is sufficient and whether the confidence clears the
        # abstain threshold, BEFORE the verdict is used to promote anything.
        #
        # The confidence read here is `statement.confidence` — the **engine's own self-report**,
        # written by `normalize/facts.py` from what the extractor claimed about itself. It is not a
        # measured uncertainty score.
        #
        # It cannot be one at this point: semantic entropy is computed by the **validate** stage,
        # and `postprocess.py` runs claims before validate. So the window where a measured score
        # exists opens *after* this decision has already been made. This comment used to say the
        # semantic-entropy score "was computed and stored but never consulted here", which read as
        # though it were consulted now; it is not, and cannot be without moving the abstain
        # decision after validate. That ordering limitation is real and recorded rather than
        # papered over.
        sufficiency = evidence_sufficiency(list(statement.evidence or []))
        abstain = abstain_decision(
            statement.confidence,
            threshold=abstain_threshold,
            sufficiency=sufficiency,
        )
        if abstain.abstain:
            claim = claim.model_copy(update={"verdict": "insufficient_evidence"})
            outcome.abstain_row = (
                {
                    "statement_assertion_id": statement_id,
                    "claim_id": claim.claim_id,
                    "reason": abstain.reason,
                    "confidence": abstain.confidence,
                    "threshold": abstain.threshold,
                    "evidence_count": sufficiency.evidence_count,
                    "distinct_units": sufficiency.distinct_units,
                }
            )

        row = claim.model_dump(mode="json")
        row["created_at"] = created_at
        outcome.claim_row = row
        # B7/B9 runtime hooks: without a verifier LLM both are explicitly
        # degenerate (insufficient/unverifiable), never a fabricated verdict.
        # Local import avoids the claims <-> reasoning package init cycle.
        from cbb_stage1.reasoning.atomic import afev_verify
        from cbb_stage1.reasoning.vegraph import vegraph_verify

        # AFEV is asked to decide "does this evidence support this atomic fact", so it
        # needs the evidence. The call used to pass only the claim, and the prompt it
        # produced ended at "证据：" with nothing after it — measured in a real run: all
        # 7 fact-verification requests arrived with an empty evidence field, the verifier
        # correctly answered `unverifiable` for every one, the aggregate became
        # `insufficient_evidence`, and the reasoner then re-decomposed and re-verified
        # the same claim for another round. That loop is the shape of a stage that burns
        # time proving nothing.
        #
        # The statement's own quotes are the evidence: they are what the claim is built
        # from, and they are verbatim by the time they reach here.
        statement_evidence = _statement_quotes(statement)
        # Both hooks are optional analyses of one claim. Neither may abort the stage:
        # `afev_verify` raises AtomicRuleError when `decompose()` returns nothing (its
        # "no fabrication fallback" rule, which is correct at the module level), and a
        # single claim whose evidence cannot be atomised must not kill a whole run.
        # Measured: a real run died with `decompose() returned no atomic facts` at
        # claims, which sits BEFORE validate, so nothing downstream ever ran and the box
        # stayed empty. `promotion_decision` below already guards the same way.
        try:
            afev = afev_verify(
                str(row.get("normalized_claim") or ""),
                book_id=book_id,
                evidence=statement_evidence,
                llm=reasoning_llm,
            )
        except Exception as exc:  # noqa: BLE001 - one claim must not abort the stage
            afev = None
            outcome.errors.append(
                {
                    "claim_id": row.get("claim_id"),
                    "stage": "afev",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        # AFEV decomposed the claim into atomic facts and verified each one; the list used to be
        # dropped here (only `afev.verdict` fed a counter), so `claims.atomic_facts` was `[]` in
        # every box even though `build_claim` accepts the field.
        #
        # The facts go on the row only when their aggregate agrees with the claim's verdict —
        # the same invariant `build_claim` enforces at construction time (`qa.py:412-418`). Writing
        # them unconditionally produced rows that contradicted themselves, and the promotion gate
        # (which re-checks) then refused the claim.
        if afev is not None and afev.atomic_facts:
            from cbb_stage1.claims.verdicts import aggregate_atomic_facts

            aggregated = aggregate_atomic_facts(afev.atomic_facts)
            if aggregated == claim.verdict:
                row["atomic_facts"] = [
                    factor.model_dump(mode="json") for factor in afev.atomic_facts
                ]
            else:
                outcome.errors.append(
                    {
                        "claim_id": row.get("claim_id"),
                        "stage": "afev",
                        "error": (
                            "atomic_facts_inconsistent: aggregate "
                            f"{aggregated!r} != claim verdict {claim.verdict!r}"
                        ),
                    }
                )

        try:
            vegraph = vegraph_verify(
                str(row.get("normalized_claim") or ""),
                book_id=book_id,
                name_to_id=name_to_id,
                llm=reasoning_llm,
            )
        except Exception as exc:  # noqa: BLE001 - same reason
            vegraph = None
            outcome.errors.append(
                {
                    "claim_id": row.get("claim_id"),
                    "stage": "vegraph",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (
            afev is not None
            and vegraph is not None
            and afev.verdict == "insufficient_evidence"
            and vegraph.verdict == "insufficient_evidence"
        ):
            outcome.degenerate = True

        # B1 CoVe (§ the four-step loop). Opt-in like every other paid analysis: the
        # caller passes `upstream["cove"] = True` and an `llm`. Cost is 2 + questions
        # calls per claim (plan, compare, one per question) plus a revision when the
        # comparison finds a contradiction, so it is never on by default.
        #
        # The result is recorded on the claim, not acted on: a revision is a proposal a
        # reader can inspect, and rewriting a claim here would delete the evidence of
        # what the extractor actually produced.
        if wants_cove:
            from cbb_stage1.validate.cove import DEFAULT_MAX_QUESTIONS as cove_question_budget
            from cbb_stage1.validate.cove import cove_verify

            # The paper's factored CoVe exists so the per-question answers can be batched
            # (A09 p.4); without this the parallel path was unreachable from the only call site.
            cove_workers = cove_worker_count(upstream, cove_question_budget)
            try:
                cove_result = cove_verify(
                    str(row.get("normalized_claim") or ""),
                    llm=cove_llm,
                    evidence=statement_evidence,
                    workers=cove_workers,
                )
            except Exception as exc:  # noqa: BLE001 - one claim must not abort the stage
                cove_result = None
                outcome.errors.append(
                    {
                        "claim_id": row.get("claim_id"),
                        "stage": "cove",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if cove_result is not None:
                row["cove"] = cove_result.as_dict()
                if cove_result.status == "revised":
                    outcome.cove_revised = True

        try:
            decision = promotion_decision(claim, statement)
        except ClaimRuleError as exc:  # pragma: no cover - defensive
            decision = None
            outcome.refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": "claim_inconsistent",
                    "detail": str(exc),
                }
            )
        if decision is None:
            return outcome
        outcome.promotion = (
            {
                "claim_id": claim.claim_id,
                "statement_assertion_id": statement_id,
                "verdict": claim.verdict,
                "allowed": decision.allowed,
                "reason": decision.reason,
            }
        )
        if not decision.allowed:
            return outcome
        # §5.16/I23 layer rules on top of the AVeriTeC verdict: only a supported,
        # corroborated claim may become fact. Checked HERE rather than before the
        # claim was built, because the claim row is the evidence of what QA said —
        # suppressing it would erase the trail instead of refusing the promotion.
        from cbb_stage1.validate.confidence import validate_claim_promotion

        layer_violations = validate_claim_promotion(
            claim_layer=statement.claim_layer,
            verdict=claim.verdict,
            source_units=[
                str(answer.get("unit_id") or "")
                for question in claim.questions
                for answer in (question.answers or [])
                if isinstance(answer, Mapping)
            ],
            against=sum(
                1
                for question in claim.questions
                if question.stance == "refute"
                for _ in (question.answers or [])
            ),
            target_layer="fact",
        )
        if layer_violations:
            outcome.refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": layer_violations[0].code,
                    "detail": layer_violations[0].detail,
                }
            )
            return outcome
        try:
            fact = build_promoted_fact(claim, statement)
        except (PromotionRefused, ClaimRuleError) as exc:
            outcome.refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": getattr(getattr(exc, "decision", None), "reason", "promotion_refused"),
                    "detail": str(exc),
                }
            )
            return outcome
        outcome.allowed = True
        fact_row = fact.model_dump(mode="json")
        fact_row["created_at"] = created_at
        outcome.promoted_row = fact_row
        return outcome

    workers = stage_workers(upstream, len(statements))
    outcomes = run_ordered(
        statements,
        _process_statement,
        workers=workers,
        thread_name_prefix="cbb-claims",
    )
    # Single writer, input order: the pool returned outcomes in statement order, and every
    # side effect the serial loop used to perform happens here, in the same sequence.
    for outcome in outcomes:
        if outcome.abstain_row is not None:
            abstain_rows.append(outcome.abstain_row)
        if outcome.claim_row is not None:
            claim_rows.append(outcome.claim_row)
        if outcome.errors:
            afev_errors.extend(outcome.errors)
        if outcome.refused:
            refused.extend(outcome.refused)
        if outcome.degenerate:
            afev_degenerate += 1
        if outcome.cove_revised:
            cove_revised += 1
        if outcome.promotion is not None:
            promotions.append(outcome.promotion)
        if outcome.allowed:
            allowed += 1
        if outcome.promoted_row is not None:
            promoted_rows.append(outcome.promoted_row)

    claims_file = _write_jsonl_atomic(claims_path(root), claim_rows)
    promoted_file = _write_jsonl_atomic(promoted_facts_path(root), promoted_rows)

    by_verdict: dict[str, int] = {}
    for row in claim_rows:
        verdict = str(row.get("verdict"))
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1

    notes: list[str] = []
    if assertions_source == "none":
        notes.append(
            "claims: no assemble/assertions.jsonl; run assemble first "
            "(empty claims table written)"
        )
    if not qa_map:
        notes.append(
            "claims: no upstream claims_qa; every claim uses an unanswerable question "
            "-> insufficient_evidence (nothing promoted)"
        )
    notes.append(
        "claims: promoted facts are written to assemble/promoted_facts.jsonl; "
        "assertions.jsonl is never mutated by this stage"
    )
    if abstain_rows:
        notes.append(
            "B12 拒答：%d 条主张因证据不足或置信度 < %.2f 判为 insufficient_evidence"
            "（分数与阈值见 counts.abstained / abstains）"
            % (len(abstain_rows), abstain_threshold)
        )

    return {
        "stage": STAGE,
        "status": "ok",
        "book_id": book_id,
        "counts": {
            "statements": len(statements),
            "claims": len(claim_rows),
            "refused": len(refused),
            "by_verdict": dict(sorted(by_verdict.items())),
            "promotions_allowed": allowed,
            "promotions_refused": len(promotions) - allowed,
            "promoted_facts": len(promoted_rows),
            "afev_vegraph_degenerate": afev_degenerate,
            # Hooks that raised on a claim. Recorded so a run can tell "this claim had
            # no verdict" from "this claim could not be analysed at all".
            "afev_errors": len(afev_errors),
            # B1: how many claims CoVe wanted to revise. Zero when it is off, which is
            # the honest reading of "not run" as distinct from "found nothing".
            "cove_revised": cove_revised,
            # B12: how many claims abstained, and the threshold that decided it.
            "abstained": len(abstain_rows),
            "abstain_threshold": abstain_threshold,
        },
        "promotions": promotions,
        "refused": refused,
        "abstains": abstain_rows,
        "afev_error_rows": afev_errors,
        "paths": {
            "claims": _rel(root, claims_file),
            "promoted_facts": _rel(root, promoted_file),
        },
        "notes": notes,
    }
