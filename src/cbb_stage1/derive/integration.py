"""Postprocess integration for the derive stage.

Unified interface (design P0-A/P0-D)::

    run(work_root, *, book_id, outputs, units, created_at, upstream) -> dict

Writes the four derived-layer work tables consumed by ``box/writer``:

* ``work/derive/summaries.jsonl`` -- the §5.14 summary tree rows built from
  ``units``. Text stays ``None`` unless an upstream synthesis mapping supplies
  it: this pass never invents a chapter summary.
* ``work/derive/forecasts.jsonl`` -- deterministic, theory-only chapter
  forecasts (``derive.forecasts.forecast_chapter``); never facts.
* ``work/derive/hypotheses.jsonl`` -- Abductive candidates derived from open
  conflicts in ``work/audit/conflicts.jsonl``; every row is
  ``evidence_sufficiency="insufficient"`` with explicit missing premises.
* ``work/derive/community.jsonl`` -- structural connected components over
  entity-role edges (``derive.community``); ``summary``/``title`` stay ``None``
  because no LLM synthesis is fabricated here.

Boundaries: no box imports, no LLM, derived layer only. Missing inputs produce
explicit empty tables (or the tree skeleton), never fabricated records.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from cbb_stage1 import ids
from cbb_stage1.derive.community import build_community_reports
from cbb_stage1.derive.forecasts import forecast_chapter
from cbb_stage1.derive.hypotheses import abduce
from cbb_stage1.derive.raptor import build_raptor_clusters
from cbb_stage1.derive.research import build_research
from cbb_stage1.derive.stylometry import build_stylometry
from cbb_stage1.derive.themes import build_themes
from cbb_stage1.derive.emotion import build_emotion_curves
from cbb_stage1.derive.motivation import build_motivation_chains
from cbb_stage1.derive.summarize import (
    synthesize_tree_texts,
    synthesize_unit_summaries,
)
from cbb_stage1.derive.summaries import build_summary_tree, summary_rows
from cbb_stage1.reasoning.abduction import abduce as reasoning_abduce

__all__ = ["run"]


def _row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump())
    raise TypeError(f"expected mapping/pydantic model, got {type(value).__name__}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: JSONL row must be an object")
        out.append(value)
    return out


def _atomic_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False) for row in rows]
    _atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _unit_text(work_root: Path, unit: Mapping[str, Any]) -> str | None:
    text_path = unit.get("text_path")
    if not text_path:
        return None
    candidate = work_root / str(text_path)
    if not candidate.exists():
        return None
    return candidate.read_text(encoding="utf-8")


def _summary_texts(
    upstream: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Optional unit_id -> summary text supplied by an upstream synthesis pass."""
    if not upstream:
        return {}
    for key in ("unit_summaries", "summaries"):
        value = upstream.get(key)
        if isinstance(value, Mapping):
            return {str(unit_id): str(text) for unit_id, text in value.items() if str(text).strip()}
    return {}


def _volume_titles(upstream: Mapping[str, Any] | None) -> dict[int, str]:
    if not upstream:
        return {}
    value = upstream.get("volume_titles")
    if not isinstance(value, Mapping):
        return {}
    out: dict[int, str] = {}
    for key, title in value.items():
        try:
            out[int(key)] = str(title)
        except (TypeError, ValueError):
            continue
    return out


def _conflict_hypotheses(
    work_root: Path,
    book_id: str,
    created_at: str,
    llm: Any = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rows = _read_jsonl(work_root / "audit" / "conflicts.jsonl")
    hypotheses: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status") or "open") == "resolved":
            continue
        conflict_id = str(row.get("conflict_id") or "")
        left = str(row.get("left") or "")
        right = str(row.get("right") or "")
        if not conflict_id or not left or not right:
            warnings.append(f"conflict row skipped (missing id/left/right): {conflict_id or row!r}")
            continue
        reasoning = reasoning_abduce(
            f"冲突 {conflict_id} 需要补充原文证据后才能裁决",
            book_id=book_id,
            scope="setting",
            llm=llm,
        )
        missing = (
            list(reasoning.hypothesis.missing_premises)
            if reasoning.hypothesis is not None
            else []
        )
        hypothesis = abduce(
            book_id,
            (
                reasoning.hypothesis.claim
                if reasoning.hypothesis is not None
                else f"冲突 {conflict_id} 需要补充原文证据后才能裁决"
            ),
            scope="setting",
            missing_premises=missing or ["额外章节/场景的原文证据"],
            derived_from=[left, right],
            created_at=created_at,
        )
        hypotheses.setdefault(hypothesis.hypothesis_id, hypothesis.to_record())
    return [hypotheses[key] for key in sorted(hypotheses)], warnings


def _entity_volumes(
    work_root: Path,
    units: Iterable[Mapping[str, Any]],
) -> dict[str, list[int]]:
    """Entity id -> volumes it appears in (resolve mentions + unit volumes)."""
    unit_volume: dict[str, int] = {}
    for unit in units:
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        try:
            unit_volume[unit_id] = int(unit.get("volume") or 1)
        except (TypeError, ValueError):
            unit_volume[unit_id] = 1
    found: dict[str, set[int]] = {}
    for row in _read_jsonl(work_root / "resolve" / "mentions.jsonl"):
        entity_id = str(row.get("entity_id") or "")
        unit_id = str(row.get("unit_id") or "")
        if not entity_id.startswith("e:") or unit_id not in unit_volume:
            continue
        found.setdefault(entity_id, set()).add(unit_volume[unit_id])
    return {entity_id: sorted(volumes) for entity_id, volumes in sorted(found.items())}


def run(
    work_root: str | Path,
    *,
    book_id: str,
    outputs: Iterable[Any] = (),
    units: Iterable[Any] = (),
    created_at: str = "",
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the derived work tables; returns a postprocess receipt dict."""
    work_root = Path(work_root)
    units = [_row(unit) for unit in units or ()]
    warnings: list[str] = []

    # ---- summaries (tree + optional LLM text) ------------------------------
    upstream = upstream or {}
    aux_llm = upstream.get("llm") if isinstance(upstream, Mapping) else None
    config = upstream.get("config") if isinstance(upstream, Mapping) else None
    unit_texts = {
        str(unit.get("unit_id") or ""): _unit_text(work_root, unit) or ""
        for unit in units
    }
    if units:
        unit_summaries = _summary_texts(upstream)
        if aux_llm is not None and bool(getattr(config, "summary_synthesis", True)):
            synthesized = synthesize_unit_summaries(units, unit_texts, aux_llm)
            synthesized.update(unit_summaries)  # explicit upstream wins
            unit_summaries = synthesized
            if not unit_summaries:
                warnings.append("summary synthesis returned no text; tree stays text=None")
        tree = build_summary_tree(
            units,
            book_id=book_id,
            unit_summaries=unit_summaries,
            volume_titles=_volume_titles(upstream),
            produced_by="derive:integration",
        )
        summary_records = summary_rows(tree)
        if aux_llm is not None and bool(getattr(config, "summary_synthesis", True)):
            summary_records = synthesize_tree_texts(summary_records, aux_llm)
    else:
        summary_records = []
        warnings.append("no units supplied; summaries table explicitly empty")
    summaries_path = work_root / "derive" / "summaries.jsonl"
    _write_jsonl(summaries_path, summary_records)

    # ---- forecasts (chapter-level proxy, theory only) ----------------------
    #
    # §5.20 asks for a chapter-scoped auxiliary reading marked theoretical — not for
    # the real 100-Endings method (sentences × 100 generations). Each row carries
    # ``method`` stating what was actually computed, so the proxy cannot be mistaken
    # for the paper's metric.
    forecasts: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: int(item.get("order") or 0)):
        unit_id = str(unit.get("unit_id") or "")
        if not unit_id:
            continue
        text = _unit_text(work_root, unit)
        if text is None:
            warnings.append(f"unit {unit_id}: text file missing; forecast skipped")
            continue
        forecast = forecast_chapter(text)
        forecasts.append(
            {
                "forecast_id": ids.forecast_id(book_id, unit_id),
                "book_id": book_id,
                "unit_id": unit_id,
                "scope": forecast.scope,
                "target": unit_id,
                "theory": forecast.theory,
                "no_rate": forecast.no_rate,
                "top_terms": list(forecast.top_terms),
                "note": forecast.note,
                "method": forecast.method,
                "produced_by": "derive:integration",
                "created_at": created_at,
            }
        )
    forecasts_path = work_root / "derive" / "forecasts.jsonl"
    _write_jsonl(forecasts_path, forecasts)

    # ---- RAPTOR recursive cluster tree (structural; embedding if supplied) --
    unit_texts = {
        str(unit.get("unit_id") or ""): _unit_text(work_root, unit) or ""
        for unit in units
    }
    embedder = upstream.get("embedder") if isinstance(upstream, Mapping) else None
    max_levels = 4
    if isinstance(upstream, Mapping):
        try:
            max_levels = max(1, int(upstream.get("raptor_max_levels") or 4))
        except (TypeError, ValueError):
            max_levels = 4
    raptor_rows = (
        build_raptor_clusters(
            book_id,
            units,
            unit_texts,
            embedder=embedder,
            max_levels=max_levels,
        )
        if units
        else []
    )
    raptor_path = work_root / "derive" / "raptor.jsonl"
    _write_jsonl(raptor_path, raptor_rows)
    if raptor_rows and not any(int(row.get("level") or 0) > 1 for row in raptor_rows):
        warnings.append(
            "raptor tree has level 1 only: no cross-unit similarity above threshold "
            "or max_levels=1"
        )

    # ---- hypotheses (Abductive, from open conflicts only) ------------------
    hypotheses, hypothesis_warnings = _conflict_hypotheses(
        work_root,
        book_id,
        created_at,
        llm=upstream.get("reasoning_llm") if isinstance(upstream, Mapping) else None,
    )
    warnings.extend(hypothesis_warnings)
    hypotheses_path = work_root / "derive" / "hypotheses.jsonl"
    _write_jsonl(hypotheses_path, hypotheses)

    # ---- textual research aggregate (cues + hypotheses + conflicts) --------
    research_records = build_research(
        book_id,
        cues=_read_jsonl(work_root / "assemble" / "cues.jsonl"),
        hypotheses=hypotheses,
        conflicts=_read_jsonl(work_root / "audit" / "conflicts.jsonl"),
    )
    research_path = work_root / "derive" / "research.jsonl"
    _write_jsonl(research_path, research_records)

    # ---- community reports (structural hierarchy; no LLM) ------------------
    community_path = work_root / "derive" / "community.jsonl"
    entity_volumes = _entity_volumes(work_root, units)
    aux_llm = upstream.get("llm") if isinstance(upstream, Mapping) else None
    community_summarizer = aux_llm.generate if aux_llm is not None else None
    community_records = build_community_reports(
        book_id,
        _read_jsonl(work_root / "assemble" / "assertions.jsonl"),
        entity_volumes=entity_volumes or None,
        summarizer=community_summarizer,
    )
    _write_jsonl(community_path, community_records)
    if not community_records:
        warnings.append("community table empty: no entity-role edge with >=2 entities")
    elif not entity_volumes:
        warnings.append(
            "community level 2 skipped: no entity->volume map "
            "(work/resolve/mentions.jsonl lacked linked entities)"
        )

    # ---- stylometry (§5.20, "低幻觉，最先做") -------------------------------
    #
    # Pure character statistics over each unit's own text: no model call, so nothing
    # here can be hallucinated. §5.20 marks this file as the first of the five optional
    # inference tables for exactly that reason.
    stylometry_rows = build_stylometry(
        book_id, units, texts=unit_texts, created_at=created_at
    )
    stylometry_path = work_root / "derive" / "stylometry.jsonl"
    _write_jsonl(stylometry_path, stylometry_rows)
    if units and not stylometry_rows:
        warnings.append(
            "stylometry table empty: no unit had readable text in plan/texts"
        )

    # ---- themes (§5.20, F02 word-frequency by segment) ---------------------
    #
    # Nouns counted per unit via jieba POS tagging; the per-unit split is what lets a
    # reader tell a book-wide motif from a one-chapter word.
    theme_rows = build_themes(book_id, unit_texts, created_at=created_at)
    themes_path = work_root / "derive" / "themes.jsonl"
    _write_jsonl(themes_path, theme_rows)
    if units and not theme_rows:
        warnings.append(
            "themes table empty: jieba unavailable or no noun survived filtering"
        )

    # ---- emotion curves (§5.20; per-speaker lanes) --------------------------
    #
    # One lane per speaker, plus the narration voice under the empty-string key. The
    # split is not cosmetic: the UED paper finds near-zero correlation between the
    # narration arc and a character's arc, so merging them would describe neither. The
    # attributed quotes come from the statement assertions, which is why this table
    # could not exist before speaker attribution worked.
    emotion_rows = build_emotion_curves(
        book_id,
        units,
        texts=unit_texts,
        statements=_read_jsonl(work_root / "assemble" / "assertions.jsonl"),
        created_at=created_at,
    )
    emotion_path = work_root / "derive" / "emotion_curves.jsonl"
    _write_jsonl(emotion_path, emotion_rows)
    if units and not emotion_rows:
        warnings.append("emotion table empty: no sentence matched the polarity lexicon")

    # ---- motivation chains (§5.20, Maslow closed set) ----------------------
    #
    # The only inference table that calls a model. Opt-in, like every other paid
    # analysis in this pipeline: a caller that wants it passes
    # `upstream["motivation_chains"] = True` and an `llm`. Without a model the table is
    # written empty and the receipt says why — an empty table is honest, a table of
    # invented chains is not.
    motivation_rows: list[dict[str, Any]] = []
    wants_motivation = bool(
        upstream.get("motivation_chains") if isinstance(upstream, Mapping) else False
    )
    if wants_motivation:
        motivation_llm = upstream.get("llm") if isinstance(upstream, Mapping) else None
        # `roles.主体` carries an entity ID; the chain is meant to be read by a person,
        # so resolve it through the candidates table the resolve stage already wrote.
        motivation_rows = build_motivation_chains(
            book_id,
            _read_jsonl(work_root / "assemble" / "assertions.jsonl"),
            llm=motivation_llm,
            entity_names={
                str(row.get("entity_id") or ""): str(row.get("canonical_name") or "")
                for row in _read_jsonl(work_root / "resolve" / "entity_candidates.jsonl")
                if row.get("entity_id") and row.get("canonical_name")
            },
            created_at=created_at,
        )
        if not motivation_rows:
            warnings.append(
                "motivation chains requested but produced nothing: no model, or no "
                "character had a quoted behaviour"
            )
    motivation_path = work_root / "derive" / "motivation_chains.jsonl"
    _write_jsonl(motivation_path, motivation_rows)

    return {
        "stage": "derive",
        "status": "ok",
        "counts": {
            "summaries": len(summary_records),
            "forecasts": len(forecasts),
            "raptor": len(raptor_rows),
            "raptor_levels": max(
                (int(row.get("level") or 0) for row in raptor_rows), default=0
            ),
            "hypotheses": len(hypotheses),
            "research": len(research_records),
            "community": len(community_records),
            "community_levels": max(
                (int(row.get("level") or 0) for row in community_records), default=0
            ),
            "stylometry": len(stylometry_rows),
            "themes": len(theme_rows),
            "emotion_curves": len(emotion_rows),
            "motivation_chains": len(motivation_rows),
        },
        "paths": {
            "summaries": summaries_path.relative_to(work_root).as_posix(),
            "forecasts": forecasts_path.relative_to(work_root).as_posix(),
            "raptor": raptor_path.relative_to(work_root).as_posix(),
            "hypotheses": hypotheses_path.relative_to(work_root).as_posix(),
            "research": research_path.relative_to(work_root).as_posix(),
            "community": community_path.relative_to(work_root).as_posix(),
            "stylometry": stylometry_path.relative_to(work_root).as_posix(),
            "themes": themes_path.relative_to(work_root).as_posix(),
            "emotion_curves": emotion_path.relative_to(work_root).as_posix(),
            "motivation_chains": motivation_path.relative_to(work_root).as_posix(),
        },
        "warnings": warnings,
    }
