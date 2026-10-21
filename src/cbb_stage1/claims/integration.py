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
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import ValidationError

from cbb_stage1.models import Assertion

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
    for statement in statements:
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
            refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": getattr(exc, "reason", "schema"),
                    "detail": str(exc),
                }
            )
            continue

        # B6 + B12: decide whether the evidence is sufficient and whether the
        # confidence clears the abstain threshold, BEFORE the verdict is used to
        # promote anything. Until now the semantic-entropy score was computed and
        # stored but never consulted here, so a claim could be promoted on a
        # 0.2-confidence basis.
        sufficiency = evidence_sufficiency(list(statement.evidence or []))
        abstain = abstain_decision(
            statement.confidence,
            threshold=abstain_threshold,
            sufficiency=sufficiency,
        )
        if abstain.abstain:
            claim = claim.model_copy(update={"verdict": "insufficient_evidence"})
            abstain_rows.append(
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
        claim_rows.append(row)
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
            afev_errors.append(
                {
                    "claim_id": row.get("claim_id"),
                    "stage": "afev",
                    "error": f"{type(exc).__name__}: {exc}",
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
            afev_errors.append(
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
            afev_degenerate += 1

        try:
            decision = promotion_decision(claim, statement)
        except ClaimRuleError as exc:  # pragma: no cover - defensive
            decision = None
            refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": "claim_inconsistent",
                    "detail": str(exc),
                }
            )
        if decision is None:
            continue
        promotions.append(
            {
                "claim_id": claim.claim_id,
                "statement_assertion_id": statement_id,
                "verdict": claim.verdict,
                "allowed": decision.allowed,
                "reason": decision.reason,
            }
        )
        if not decision.allowed:
            continue
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
            refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": layer_violations[0].code,
                    "detail": layer_violations[0].detail,
                }
            )
            continue
        try:
            fact = build_promoted_fact(claim, statement)
        except (PromotionRefused, ClaimRuleError) as exc:
            refused.append(
                {
                    "statement_assertion_id": statement_id,
                    "reason": getattr(getattr(exc, "decision", None), "reason", "promotion_refused"),
                    "detail": str(exc),
                }
            )
            continue
        allowed += 1
        fact_row = fact.model_dump(mode="json")
        fact_row["created_at"] = created_at
        promoted_rows.append(fact_row)

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
