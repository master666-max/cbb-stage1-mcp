"""9-stage pipeline orchestration shell (ported from B09 BookContentGenerator).

Preserved shape (design ``08-现成管线骨架调研`` §2.1):

1. ``LOADING_BOOK`` - parse EPUB / text into chapters;
2. ``ANALYZING_STRUCTURE`` - chapters -> units (big-first, chunk fallback);
3. ``EXTRACTING_CHARACTERS_AND_ENTITIES`` - one engine call per unit, with
   content-hash idempotency and interval checkpoints;
4. ``RESOLVING_DUPLICATE_NAMES`` - postprocess sub-steps: resolve -> assemble ->
   claims -> cues -> temporal -> validate -> dedupe -> derive integrations
   (``work/plan/wiring.json`` records the call chain);
5. ``GENERATING_DESCRIPTIONS`` - canonical card table from postprocess when
   present, otherwise the deterministic card snapshot (no LLM);
6. ``BUILDING_CONNECTIONS`` - deterministic unit co-occurrence;
7. ``WRITING_OUTPUT`` - engine index, units, tasks;
8. ``SAVING_STATE`` - trace / progress / telemetry / state / registry;
9. ``COMPLETE``.

Bindings replace the source's Ollama/FastAPI coupling:

* config/paths/models come from ``cbb_stage1``;
* the only extraction dependency is the ``Engine`` protocol
  (``engines/base.Engine``); offline runs use ``engines/mock.MockEngine``;
* no box writes: ``Config.out_dir`` is reserved for the v2.2 box builder task.

Work artifacts are disposable; every artifact write is atomic.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import queue
import time

logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterator, Sequence

from cbb_stage1.ids import sha256_hex_bytes, sha256_hex_text, sha256_text

from cbb_stage1.agents import build_subagent_pool
from cbb_stage1.config import Config
from cbb_stage1.model_gateway import build_model_gateway
from cbb_stage1.engines.base import Engine
from cbb_stage1.engines.mock import MockEngine
from cbb_stage1.models import EngineOutput
from cbb_stage1.paths import WorkPaths

from . import collectors, llm_bridge, role_runtime, security, telemetry
from .checkpoint import DEFAULT_CHECKPOINT_INTERVAL, CheckpointManager, CheckpointStore
from .files import atomic_write_json, read_json, read_jsonl, unit_file_stem, write_jsonl
from .idempotency import IdempotencyStore, generate_idempotency_key
from .postprocess import (
    PostprocessError,
    StageSpec,
    load_wiring,
    run_postprocess,
)
from .progress import (
    STAGE_ORDER,
    TOTAL_STAGES,
    ProgressStage,
    clear_interrupt,
    emit_progress,
    reset_progress,
    safe_point_action,
    set_control_dir,
    save_progress_log,
)
from .tracer import (
    SpanContext,
    end_span,
    end_trace,
    log_decision,
    save_trace,
    start_span,
    start_trace,
)

__all__ = [
    "PipelineResult",
    "run_pipeline",
    "build_engine",
    "engine_fingerprint",
]

#: C8: minimum pending units before spawning the extraction thread pool.
#: Smaller batches (and ``max_workers=1`` / ``fail_fast``) run serially: thread
#: overhead is not worth it and serial keeps checkpoint/failure attribution
#: deterministic for tiny fixtures.
_PARALLEL_MIN_UNITS = 4


@dataclass
class PipelineResult:
    """Summary of one orchestration run (also written to ``work/plan/run.json``)."""

    book_id: str
    book_title: str = ""
    book_format: str = ""
    work_dir: str = ""
    trace_id: str = ""
    engine: str = ""
    status: str = "running"  # running | complete | partial | failed
    units_total: int = 0
    units_processed: int = 0
    units_cached: int = 0
    units_failed: int = 0
    units_abstained: int = 0
    mentions: int = 0
    entities: int = 0
    assertions: int = 0
    events: int = 0
    cues: int = 0
    connections: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: str | None = None
    run_path: str | None = None
    #: Compact postprocess summary (wiring ok/counts/path); never the tables.
    postprocess: dict | None = None
    #: Model that served verifier calls, when one was used. D4 (分级验证矩阵)
    #: requires the verifier model to differ from the extractor model; recording
    #: it here is what makes that check measurable at box-build time instead of
    #: an unverifiable claim.
    verifier_model: str | None = None

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "book_title": self.book_title,
            "book_format": self.book_format,
            "work_dir": self.work_dir,
            "trace_id": self.trace_id,
            "engine": self.engine,
            "verifier_model": self.verifier_model,
            "status": self.status,
            "units_total": self.units_total,
            "units_processed": self.units_processed,
            "units_cached": self.units_cached,
            "units_failed": self.units_failed,
            "units_abstained": self.units_abstained,
            "mentions": self.mentions,
            "entities": self.entities,
            "assertions": self.assertions,
            "events": self.events,
            "cues": self.cues,
            "connections": self.connections,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "run_path": self.run_path,
            "postprocess": self.postprocess,
            "stages": [stage.name for stage in STAGE_ORDER],
        }


# =============================================================================
# ENGINE BINDING
# =============================================================================

def build_engine(cfg: Config, gateway: Any = None) -> Engine:
    """Select the extraction engine from config.

    ``mock`` / model ``mock`` is the offline default. Any other provider imports
    the AI Reader adapter lazily; a missing adapter is a hard error (never a
    silent fallback to mock - that would be a degraded run).
    """
    if cfg.provider == "mock" or cfg.model == "mock":
        return MockEngine()

    from cbb_stage1.engines.factory import build_engine as build_from_provider

    return build_from_provider(
        cfg.provider,
        llm=gateway.async_client if gateway is not None else None,
        model=cfg.model or None,
        # The engine is built once for the whole book, so the per-unit tier budget is passed at call
        # time (`extract(..., max_gleanings=...)`). Passing the global number here meant the three
        # tiered budgets in Config had no effect at all.
        max_gleanings=cfg.max_gleanings,
    )


def engine_fingerprint(engine: Engine, cfg: Config) -> tuple[str, str]:
    """Return ``(engine_name, prompt_hash)`` used for idempotency staleness.

    Engines may expose ``name`` and ``prompt_sha256`` / ``prompt_hash``; when a
    prompt hash is absent we derive a stable value from engine name + model so a
    model switch still invalidates the cache.
    """
    name = str(getattr(engine, "name", type(engine).__name__))
    prompt_hash = getattr(engine, "prompt_sha256", None) or getattr(
        engine, "prompt_hash", None
    )
    if not prompt_hash:
        prompt_hash = sha256_hex_text(f"{name}:{cfg.model}", 16)
    return name, str(prompt_hash)


# =============================================================================
# HELPERS
# =============================================================================

def _now_iso() -> str:
    return datetime.now().isoformat()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_hex_bytes(path.read_bytes())


def _load_engine_output(path: Path) -> EngineOutput | None:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    try:
        return EngineOutput.model_validate(payload)
    except Exception:
        return None


def _coerce_output(output: EngineOutput | dict, unit_id: str) -> EngineOutput:
    """Validate an engine result and pin ``unit_id`` to the requested unit."""
    if not isinstance(output, EngineOutput):
        output = EngineOutput.model_validate(output)
    if output.unit_id != unit_id:
        log_decision(
            "engine_contract",
            [output.unit_id, unit_id],
            unit_id,
            "engine returned a mismatched unit_id; pinned to the requested unit",
        )
        output = output.model_copy(update={"unit_id": unit_id})
    return output


def _counts(output: EngineOutput) -> dict:
    return {
        "mentions": len(output.mentions),
        "entities": len(output.entities),
        "assertions": len(output.assertions),
        "events": len(output.events),
        "cues": len(output.cues),
    }


@dataclass
class _UnitPlan:
    """One unit's pre-computed execution plan (order-preserving)."""

    index: int
    unit: object
    output_path: Path
    card: dict
    state: str = "pending"  # pending | cached | done
    output: EngineOutput | None = None


@dataclass
class _ExtractOutcome:
    """Result of one engine call; errors are captured, never raised in a worker."""

    unit_id: str
    output: EngineOutput | None = None
    error: BaseException | None = None
    duration_ms: float = 0.0


#: Fields the prompt builder reads off a known-entity entry. The scanner returns
#: dicts while ``build_known_entities_section`` reads attributes, so a pass-through would
#: render an empty section — silent, and indistinguishable from not wiring it at all.
class _KnownEntity:
    """One pre-scanned entity, in the shape the prompt builder expects."""

    __slots__ = ("canonical_name", "type", "aliases")

    def __init__(self, name: str, entity_type: str, aliases: list[str]) -> None:
        self.canonical_name = name
        self.type = entity_type
        self.aliases = aliases


def _normalise_known_entities(entries: Any) -> list[_KnownEntity]:
    """Turn whatever the scanner returned into prompt-ready entries.

    Accepts dicts (``{"name"/"canonical_name", "entity_type"/"type", "aliases"}``) and
    objects alike, because the two halves of this pipeline were written against different
    shapes and the mismatch is exactly what made the dictionary invisible.
    """
    resolved: list[_KnownEntity] = []
    seen: set[str] = set()
    for entry in entries or ():
        if isinstance(entry, Mapping):
            name = str(
                entry.get("canonical_name") or entry.get("name") or ""
            ).strip()
            etype = str(entry.get("entity_type") or entry.get("type") or "")
            aliases = [str(a) for a in (entry.get("aliases") or ()) if str(a).strip()]
        else:
            name = str(
                getattr(entry, "canonical_name", None) or getattr(entry, "name", "") or ""
            ).strip()
            etype = str(getattr(entry, "entity_type", None) or getattr(entry, "type", "") or "")
            aliases = [str(a) for a in (getattr(entry, "aliases", None) or ()) if str(a).strip()]
        if not name or name in seen:
            continue
        seen.add(name)
        resolved.append(_KnownEntity(name, etype, aliases))
    return resolved


def _prescan_for_extraction(
    engine: Any,
    *,
    book_id: str,
    chapters: list[str],
    titles: list[str] | None = None,
) -> list[_KnownEntity]:
    """Run the pre-scan and return its dictionary in a usable shape.

    This is the second half of the two-stage pattern (A1): the first half already ran
    before extraction, and its result was discarded, so extraction never saw the
    dictionary it had just paid for.

    A missing or failing scanner degrades to an empty dictionary. It must not abort the
    run: the dictionary is a hint, and a run without hints is a worse run, not a broken
    one.
    """
    scan = getattr(engine, "prescan_book", None)
    if not callable(scan):
        return []
    try:
        entries = scan(book_id, chapters, titles)
    except Exception as exc:  # noqa: BLE001 - hints are optional, the run is not
        logger.warning("pre-scan failed, extracting without dictionary hints: %s", exc)
        return []
    return _normalise_known_entities(entries)


def _extract_one(
    engine: Engine,
    unit,
    card: dict,
    known_entities: Sequence[Any] = (),
    config: Any = None,
) -> _ExtractOutcome:
    """Call ``engine.extract`` for one unit (the thread-safe unit of work).

    C8: workers never raise. Failures are captured so the main thread can apply
    per-unit failure isolation or fail-fast semantics **in unit order**, keeping
    checkpoint/idempotency/task-record writes single-writer.
    """
    started = time.monotonic()
    try:
        # The fourth argument is the pre-scanned dictionary. It used to be a hardcoded
        # empty list, which is why every extraction ran without hints even though the
        # pre-scan had just run and the prompt builder could render them.
        # The per-unit tier budget, resolved here because the engine is built once per book. Passing
        # only the global `max_gleanings` into `build_engine` meant the tiered budgets in Config were
        # never consulted: a canonical character got the same extra passes as a walk-on.
        gleanings = None
        tier = str(getattr(unit, "tier", "") or "")
        if config is not None and tier:
            gleanings = int(config.for_tier(tier))
        # Only engines that declare the parameter get it: `MockEngine.extract` does not, and passing
        # it anyway raised TypeError inside the worker, turning every mocked run into a partial one.
        # Detected from the signature rather than by catching TypeError, because a TypeError raised
        # inside the engine must not be mistaken for "this engine does not take the argument".
        accepts_gleanings = False
        try:
            accepts_gleanings = "max_gleanings" in inspect.signature(engine.extract).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            accepts_gleanings = False
        if accepts_gleanings:
            output = engine.extract(
                unit.unit_id,
                unit.text,
                card,
                known_entities or [],
                max_gleanings=gleanings,
            )
        else:
            output = engine.extract(
                unit.unit_id, unit.text, card, known_entities or []
            )
    except BaseException as exc:  # noqa: BLE001 - policy is applied by the caller
        return _ExtractOutcome(
            unit_id=unit.unit_id,
            error=exc,
            duration_ms=(time.monotonic() - started) * 1000,
        )
    return _ExtractOutcome(
        unit_id=unit.unit_id,
        output=output,
        duration_ms=(time.monotonic() - started) * 1000,
    )


def _extract_from_pool(
    pool: "queue.Queue[Engine]",
    unit,
    card: dict,
    known_entities: Sequence[Any] = (),
    config: Any = None,
) -> _ExtractOutcome:
    """C8 live path: each worker takes a thread-confined engine instance."""
    engine = pool.get()
    try:
        return _extract_one(engine, unit, card, known_entities, config)
    finally:
        pool.put(engine)


def _plan_unit(
    index: int,
    unit,
    engine_dir: Path,
    card: dict,
    checkpoint_mgr,
    idempotency: IdempotencyStore,
    *,
    book_id: str,
    model_id: str,
    prompt_hash: str,
) -> _UnitPlan:
    """Resolve checkpoint/cache state for one unit before the parallel phase."""
    output_path = engine_dir / f"{unit_file_stem(unit.unit_id)}.json"

    # Resume path: checkpoint says done and the artifact is valid.
    if checkpoint_mgr.is_unit_done(unit.unit_id):
        output = _load_engine_output(output_path)
        if output is not None:
            return _UnitPlan(index, unit, output_path, card, "done", output)
        log_decision(
            "checkpoint",
            ["trust", "recompute"],
            "recompute",
            f"checkpoint recorded {unit.unit_id} but its engine artifact is "
            "missing/corrupt; recomputing",
        )

    key = generate_idempotency_key(
        book_id,
        unit.title,
        unit.seq,
        operation="extract",
        content_hash=unit.sha256,
    )
    cached_payload = idempotency.get_cached_result(key)
    if cached_payload is not None and not idempotency.is_stale(
        key, model_id, prompt_hash
    ):
        output = _coerce_output(cached_payload, unit.unit_id)
        log_decision(
            "idempotency",
            ["reuse", "reextract"],
            "reuse",
            f"cache hit for {unit.unit_id} ({key})",
        )
        return _UnitPlan(index, unit, output_path, card, "cached", output)

    return _UnitPlan(index, unit, output_path, card, "pending")


#: The one card field that cannot be part of its own preimage.
_CARD_HASH_FIELD = "card_sha256"


def _card_sha256(card: dict) -> str:
    """Content hash of a dispatch card, excluding the hash field itself.

    `sort_keys=True` because key order is not content: a hash that changes when a dict is rebuilt
    would make "the same card" unprovable. This is what lets a reader verify that the card recorded
    in the box is the card the candidate was extracted from.
    """
    payload = {key: value for key, value in card.items() if key != _CARD_HASH_FIELD}
    return sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


#: How often a paused run re-checks its control flags. Short enough that an interrupt during a pause
#: is honoured promptly; long enough not to spin a core.
_PAUSE_POLL_SECONDS = 0.5


def _resume_view(doc_status) -> dict[str, list[str]]:
    """The resume grouping for the current table (A22): what needs work, and whether it already ran."""
    from cbb_stage1.ingest.doc_status import resume_work

    return resume_work(doc_status.rows)


def _doc_status_table(work_root: Path):
    """The per-unit document-state table under ``plan/`` (A22).

    A thin wrapper so the loop reads as one line: the table is loaded once per run and saved after each
    transition, which keeps the file usable if the process dies mid-book.
    """
    from cbb_stage1.ingest.doc_status import DocStatusTable

    return DocStatusTable.load(work_root / "plan" / "doc-status.jsonl")


def _card_for(unit, cfg: Config) -> dict:
    """Build the minimal Card v2 input passed to the engine for one unit.

    The full Card v2 (glossary/rolling/few-shot retrieval) belongs to the
    extraction-task boundary; this card carries the unit text metadata and the
    v6 instruction block, and is deterministic (no timestamps).

    The card carries its own `card_sha256` so it can be identified after the fact: without it,
    `produced_by` could say which prompt was used but never which *inputs*.
    """
    card = {
        "contract": "cbb-card/2",
        "book": {"id": cfg.book_id, "title": "", "genre": cfg.genre},
        "unit": {
            "unit_id": unit.unit_id,
            "volume": unit.volume,
            "chapter": unit.chapter,
            "title": unit.title,
            "order": unit.order,
            "narrative_mode": unit.narrative_mode,
            "narrator": unit.narrator,
            "narrator_reliability": unit.narrator_reliability,
            "text_path": unit.text_path,
            "sha256": unit.sha256,
            "char_count": unit.chars,
        },
        "profile": {},
        "glossary": {"entities": [], "predicates": []},
        "rolling": {"summary": "", "open_cues": [], "unresolved": [], "budget_chars": 4000},
        "few_shot": [],
        "instructions": {
            "evidence": "每条事实必须给原文 quote；quote 必须是原文连续子串；不确定就给 abstain。",
            "aliases": "泛称/亲属称谓/官职不得作别名；别名必须有文本依据。",
            "forbidden": "不许用外部知识；不许补写未出现的设定。",
        },
    }
    card[_CARD_HASH_FIELD] = _card_sha256(card)
    return card


@contextmanager
def _stage_scope(
    index: int,
    stage: ProgressStage,
    message: str | None = None,
) -> Iterator[object]:
    """Emit the stage, time it, and wrap it in a trace span."""
    emit_progress(stage, index + 1, TOTAL_STAGES, message=message)
    telemetry.start_timer(f"stage:{stage.name}")
    span = start_span(f"stage:{stage.name}")
    try:
        yield span
    except BaseException as exc:
        end_span(span, status="error", error=str(exc))
        raise
    else:
        end_span(span, status="ok")
    finally:
        telemetry.stop_timer(f"stage:{stage.name}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    cfg: Config,
    chapters: str | int | None = "1-3",
    engine: Engine | None = None,
    *,
    fail_fast: bool = False,
    checkpoint_interval: int | None = None,
    min_connection_weight: int = 2,
    postprocess_stages: Sequence[StageSpec] | None = None,
) -> PipelineResult:
    """Run the 9-stage extraction orchestration for one book.

    Args:
        cfg: Runtime configuration (book path, work dir, chunk/engine knobs).
        chapters: Chapter selector (``"1-3"``, ``"2"``, ``"all"``…).
        engine: Optional engine override (tests / adapters). Defaults to
            :func:`build_engine`.
        fail_fast: If True, an engine error aborts the run (used to exercise
            checkpoint resume). Default False: per-unit failure isolation, the
            unit is marked ``failed`` in ``plan/tasks.jsonl`` and the run ends
            ``partial``.
        checkpoint_interval: Save the checkpoint every N completed units
            (default 5, as in the source shell).
        min_connection_weight: Co-occurrence threshold for the connections shell.
        postprocess_stages: Override the postprocess stage list (tests). Defaults
            to the eight package integrations in their fixed order.

    Returns:
        :class:`PipelineResult` summarising counts and artifact locations.

    Raises:
        FileNotFoundError: If ``cfg.book_path`` does not exist.
        ValueError: For invalid chapter selectors / book ids.
        EngineError: When a non-mock provider has no wired adapter.
        PostprocessError: When a postprocess integration is unavailable or fails;
            the run is then marked ``failed`` in ``plan/state.json`` and
            ``plan/wiring.json`` records the failing stage (never silent).
    """
    work_root = Path(cfg.work_dir)
    work = WorkPaths(work_root).ensure()
    plan = work.plan
    engine_dir = work.engine
    resolve_dir = work.resolve
    assemble_dir = work.assemble
    from cbb_stage1.ingest.media import extract_media_assets

    media_rows = extract_media_assets(cfg.book_path, work_root, cfg.book_id)
    write_jsonl(plan / "media.jsonl", media_rows)
    progress_dir = plan / "progress"
    traces_dir = plan / "traces"
    telemetry_dir = plan / "telemetry"

    checkpoint_store = CheckpointStore(
        plan / "checkpoint.json",
        interval=checkpoint_interval or DEFAULT_CHECKPOINT_INTERVAL,
    )
    idempotency = IdempotencyStore(plan / "idempotency.json")

    reset_progress()
    # A5: a detached pipeline can only see a pause through a file.
    set_control_dir(work_root / "plan")
    clear_interrupt()
    telemetry.reset_telemetry()
    telemetry.start_timer("pipeline_total")

    result = PipelineResult(book_id=cfg.book_id, work_dir=str(work_root))
    trace_id = start_trace(
        metadata={
            "book_path": str(cfg.book_path),
            "book_id": cfg.book_id,
            "chapters": chapters,
            "genre": cfg.genre,
            "model": cfg.model,
            "provider": cfg.provider,
            "started_at": _now_iso(),
        }
    )
    result.trace_id = trace_id

    gateway = build_model_gateway(cfg, work_root)
    selected_engine = engine or build_engine(cfg, gateway=gateway)
    subagents = build_subagent_pool(cfg, gateway=gateway)
    aux_llm = subagents.get("assistant")
    if aux_llm is None and gateway is not None:
        aux_llm = gateway.sync_llm("assistant")
    reasoning_llm = llm_bridge.build_reasoning_llm(subagents.get("reasoner") or aux_llm)
    verifier_llm = subagents.verifier()
    #: True when no dedicated verifier subagent exists and the check will run on the
    #: same model that produced the record. §5.15 wants independence; when it cannot be
    #: had, the fact is reported beside the verdict rather than hidden.
    verifier_same_model = verifier_llm is None
    if verifier_llm is None:
        verifier_llm = llm_bridge.build_verifier_llm(aux_llm)
    # B4 judge / D4 decoupling: a judge must come from a DIFFERENT model than the
    # extractor, otherwise §5.15's independence requirement is not met. Built only
    # when a separate endpoint is configured; never silently reused from aux_llm.
    judge_llm = None
    judge_base = str(getattr(cfg, "judge_base_url", "") or os.environ.get("CBB_MCP_JUDGE_BASE_URL", "")).strip()
    judge_model = str(getattr(cfg, "judge_model", "") or os.environ.get("CBB_MCP_JUDGE_MODEL", "")).strip()
    if judge_base and judge_model:
        try:
            judge_llm = llm_bridge.build_judge_llm(
                llm_bridge.build_http_llm(judge_base, judge_model)
            )
        except Exception:  # noqa: BLE001 - a broken judge must not abort the run
            judge_llm = None
    embedder = llm_bridge.build_embedder(cfg, gateway=gateway)
    nli = None
    if gateway is not None:
        from cbb_stage1.temporal.nli_client import build_nli

        nli_provider = (
            "host" if getattr(gateway, "transport_kind", "http") in {"host", "mcp"} else None
        )
        nli = build_nli(cfg, client=gateway.async_client, provider=nli_provider)
    engine_name, prompt_hash = engine_fingerprint(selected_engine, cfg)
    model_id = f"{engine_name}:{cfg.model}"
    result.engine = model_id

    # Replayability needs the exact model behind each role, not just the engine
    # name. ``trace.model_versions`` was always ``{}`` because nothing recorded it.
    try:
        from cbb_stage1.orchestrator.tracer import record_model_version

        record_model_version("engine", model_id)
        if prompt_hash:
            record_model_version("prompt", str(prompt_hash))
        label = str(getattr(cfg, "judge_model", "") or "").strip()
        if label:
            record_model_version("judge", label)
        verifier_label = str(getattr(cfg, "verifier_model", "") or "").strip()
        if verifier_label:
            record_model_version("verifier", verifier_label)
    except Exception:  # noqa: BLE001 - tracing must never abort a run
        pass

    outputs: dict[str, EngineOutput] = {}
    task_records: list[dict] = []
    units: list = []

    try:
        # -- 1. LOADING_BOOK ------------------------------------------------
        with _stage_scope(0, ProgressStage.LOADING_BOOK, f"loading {Path(cfg.book_path).name}"):
            from cbb_stage1.ingest.loader import load_book

            book = load_book(cfg.book_path)
            result.book_title = book.title
            result.book_format = book.format
        log_decision(
            "load_book",
            ["epub", "text"],
            book.format,
            f"{len(book.chapters)} chapters, {book.total_chars} chars",
        )

        # -- 2. ANALYZING_STRUCTURE ----------------------------------------
        with _stage_scope(1, ProgressStage.ANALYZING_STRUCTURE, "building units"):
            from cbb_stage1.ingest.units import build_units

            units = build_units(
                book,
                work_root,
                book_id=cfg.book_id,
                chapters=chapters,
                chunk_target_tokens=cfg.chunk_target_tokens,
                chunk_overlap_tokens=cfg.chunk_overlap_tokens,
                chunk_strategy=cfg.chunk_strategy,
                narrator_reliability=cfg.narrator_reliability,
                chars_per_token=1.5,
            )
            write_jsonl(plan / "units.jsonl", [unit.to_record() for unit in units])
        result.units_total = len(units)
        split_units = sum(1 for unit in units if unit.split_reason != "big_first")
        log_decision(
            "build_units",
            ["big_first", "chunk_split"],
            "chunk_split" if split_units else "big_first",
            f"{len(units)} units from {len(book.chapters)} chapters "
            f"({split_units} chunked)",
        )
        emit_progress(
            ProgressStage.ANALYZING_STRUCTURE,
            2,
            TOTAL_STAGES,
            message=f"{len(units)} units ready",
            sub_current=len(units),
            sub_total=len(units),
        )

        # -- 2b. PRE-SCAN (A1/A6): dictionary before per-unit extraction -----
        suspicious_units = []
        for unit in units:
            scan = security.scan(unit.text)
            if scan.suspicious:
                suspicious_units.append({"unit_id": unit.unit_id, "flags": scan.flags})
        if suspicious_units:
            log_decision(
                "security",
                ["trust", "untrusted"],
                "untrusted",
                f"{len(suspicious_units)} units contain injection-shaped phrases; "
                "text is treated as untrusted source, never instructions",
            )
        # A1 second half: the pre-scan dictionary has to reach the extraction prompt.
        # It used to be built and dropped — the call returned entries, nothing kept them,
        # and `_extract_one` passed a hardcoded `[]` as the fourth argument. Extraction ran
        # without hints while the run reported "pre-scan completed", which is why the gap
        # was invisible for so long.
        known_entities: list = []
        if hasattr(selected_engine, "prescan_book"):
            with SpanContext("prescan", unit=None):
                known_entities = _prescan_for_extraction(
                    selected_engine,
                    book_id=cfg.book_id,
                    chapters=[unit.text for unit in units],
                    titles=[unit.title for unit in units],
                )
            if known_entities:
                log_decision(
                    "prescan",
                    ["dictionary", "degraded"],
                    "dictionary",
                    f"pre-scan produced {len(known_entities)} entities for {len(units)} units",
                )
            else:
                # Named degradation, not silence: a run without hints is worse, not broken.
                log_decision(
                    "prescan",
                    ["dictionary", "degraded"],
                    "degraded",
                    "pre-scan produced no usable entries; extracting without hints",
                )

        # -- 2d. INCREMENTAL PLAN (A20/D6) ----------------------------------
        inc_plan = None
        if cfg.incremental:
            from cbb_stage1.incremental.runner import plan_incremental

            previous = cfg.out_dir if (cfg.out_dir / "manifest.json").exists() else None
            inc_plan = plan_incremental(
                work_root,
                [unit.to_record() for unit in units],
                previous_state=previous,
            )
            inc_plan.write(plan / "incremental-plan.json")
            # The plan's cleanup list was computed and never read, so a shrunk book kept the
            # removed units' engine artifacts forever. The replace half stays advisory — the
            # postprocess stages rewrite those tables whole — and the plan file says so.
            from cbb_stage1.incremental.runner import apply_cleanup_files

            apply_cleanup_files(
                work_root,
                inc_plan,
                live_unit_ids={unit.unit_id for unit in units},
            )
            log_decision(
                "incremental",
                ["full", "incremental"],
                "full" if inc_plan.full_run else "incremental",
                f"changed={len(inc_plan.changed_units)} "
                f"reuse={len(inc_plan.reuse_engine_outputs)} "
                f"dirty_tables={len(inc_plan.replace)}",
            )

        # -- 3. EXTRACTING_CHARACTERS_AND_ENTITIES --------------------------
        with _stage_scope(
            2,
            ProgressStage.EXTRACTING_CHARACTERS_AND_ENTITIES,
            f"{len(units)} units via {model_id}",
        ):
            with CheckpointManager(
                checkpoint_store, cfg.book_id, book.title, len(units)
            ) as checkpoint_mgr:
                # Phase 1 (main thread): order-preserving planning. Checkpoint
                # resume and idempotency cache hits are resolved before any
                # engine work and before anything is written.
                plans: list[_UnitPlan] = []
                for index, unit in enumerate(units):
                    # A5: the safe point is three-valued. Interrupt abandons the run; pause stops it
                    # here and stays resumable. Only asking `check_interrupt()` meant a paused run kept
                    # extracting, which is why stopping a run required killing the process.
                    action = safe_point_action()
                    if action == "wait":
                        log_decision(
                            "pause",
                            ["continue", "wait"],
                            "wait",
                            "pause requested; waiting at the unit safe point",
                        )
                        # Re-check on an interval rather than sleeping on a bare flag: a pause whose
                        # release never comes must still be interruptible, or the run hangs with no
                        # way out but a kill.
                        while safe_point_action() == "wait":
                            time.sleep(_PAUSE_POLL_SECONDS)
                        action = safe_point_action()
                    if action == "stop":
                        log_decision(
                            "interrupt",
                            ["continue", "stop"],
                            "stop",
                            "interrupt requested; stopped at the unit safe point",
                        )
                        break
                    plans.append(
                        _plan_unit(
                            index,
                            unit,
                            engine_dir,
                            _card_for(unit, cfg),
                            checkpoint_mgr,
                            idempotency,
                            book_id=cfg.book_id,
                            model_id=model_id,
                            prompt_hash=prompt_hash,
                        )
                    )

                pending = [
                    unit_plan for unit_plan in plans if unit_plan.state == "pending"
                ]
                if cfg.incremental and inc_plan is not None:
                    reuse = {
                        str(item[0]) if isinstance(item, (tuple, list)) else str(item)
                        for item in inc_plan.reuse_engine_outputs
                    }
                    for unit_plan in pending:
                        if unit_plan.unit.unit_id not in reuse:
                            continue
                        reused = _load_engine_output(unit_plan.output_path)
                        if reused is not None:
                            unit_plan.state = "cached"
                            unit_plan.output = reused
                    pending = [
                        unit_plan for unit_plan in plans if unit_plan.state == "pending"
                    ]
                outcomes: dict[int, _ExtractOutcome] = {}
                requested_workers = max(1, int(cfg.max_workers))
                # C8: threads only pay off for batches at least this large; tiny
                # batches and fail-fast stay serial so checkpoint/failure
                # semantics are deterministic. max_workers=1 is always serial.
                parallel = (
                    not fail_fast
                    and requested_workers > 1
                    and len(pending) >= _PARALLEL_MIN_UNITS
                )
                workers = min(requested_workers, len(pending)) if parallel else 1
                engine_pool: queue.Queue[Engine] | None = None
                if parallel and engine is None and (cfg.provider or "").lower() not in {"", "mock"}:
                    try:
                        engine_pool = queue.Queue()
                        for _ in range(workers):
                            engine_pool.put(build_engine(cfg, gateway=gateway))
                        log_decision(
                            "engine_pool",
                            ["shared", "per_worker"],
                            "per_worker",
                            f"{workers} live engine instances for thread confinement",
                        )
                    except Exception as exc:  # noqa: BLE001 - fallback is explicit
                        engine_pool = None
                        log_decision(
                            "engine_pool",
                            ["shared", "per_worker"],
                            "shared",
                            f"per-worker engine build failed: {type(exc).__name__}: {exc}",
                        )
                if pending:
                    if parallel:
                        log_decision(
                            "concurrency",
                            ["parallel", "serial"],
                            "parallel",
                            f"{len(pending)} units across {workers} workers "
                            f"(max_workers={requested_workers})",
                        )
                    elif requested_workers > 1 and not fail_fast:
                        log_decision(
                            "concurrency",
                            ["parallel", "serial"],
                            "serial",
                            f"{len(pending)} pending units < {_PARALLEL_MIN_UNITS}; "
                            "serial avoids thread overhead on tiny batches",
                        )
                    with SpanContext(
                        "extract:parallel", units=len(pending), workers=workers
                    ):
                        if not parallel:
                            for unit_plan in pending:
                                outcomes[unit_plan.index] = _extract_one(
                                    selected_engine,
                                    unit_plan.unit,
                                    unit_plan.card,
                                    known_entities,
                                )
                                if fail_fast and outcomes[unit_plan.index].error is not None:
                                    break
                        else:
                            with ThreadPoolExecutor(
                                max_workers=workers,
                                thread_name_prefix="cbb-extract",
                            ) as executor:
                                futures = {
                                    unit_plan.index: executor.submit(
                                        _extract_from_pool if engine_pool is not None else _extract_one,
                                        engine_pool if engine_pool is not None else selected_engine,
                                        unit_plan.unit,
                                        unit_plan.card,
                                        known_entities,
                                    )
                                    for unit_plan in pending
                                }
                                # Collect in unit order; persistence stays ordered.
                                for unit_plan in pending:
                                    outcomes[unit_plan.index] = futures[unit_plan.index].result()

                # A22: record the whole batch as `pending` before anything runs. Without this a run that
                # dies mid-book leaves no trace of the units it never reached, so a resume cannot tell
                # "not started" from "failed".
                doc_status = _doc_status_table(work_root)
                # Collected here rather than appended to `warnings` (defined much later in the
                # function): referencing a name that is not yet bound raises UnboundLocalError at the
                # moment the exception fires, which buries the real failure.
                doc_status_errors: list[str] = []
                for unit_plan in plans:
                    current = doc_status.state_of(unit_plan.unit.unit_id)
                    if current is None:
                        doc_status.transition(
                            unit_plan.unit.unit_id, "pending", at=_now_iso()
                        )
                    elif current in ("processed", "failed"):
                        # This run is a new attempt on the document; record that rather than appending a
                        # second terminal row for the same attempt.
                        doc_status.transition(
                            unit_plan.unit.unit_id,
                            "pending",
                            detail=f"re-armed from {current}",
                            at=_now_iso(),
                        )
                doc_status.save()
                # The four-way split, in the log where an operator looks: unstarted / failed /
                # in_flight / done. This is what `resume_work` is for.
                log_decision(
                    "doc_status",
                    ["unstarted", "failed", "in_flight", "done"],
                    "recorded",
                    str(
                        {
                            key: len(value)
                            for key, value in _resume_view(doc_status).items()
                        }
                    ),
                )

                # Phase 2 (main thread): persist / checkpoint / idempotency /
                # task records strictly in unit order (single writer).
                for unit_plan in plans:
                    position = unit_plan.index + 1
                    unit = unit_plan.unit
                    output = unit_plan.output
                    cached = unit_plan.state in ("cached", "done")
                    error: str | None = None

                    if unit_plan.state == "pending":
                        outcome = outcomes.get(unit_plan.index)
                        if outcome is None:
                            break  # fail-fast stopped before this unit ran
                        if outcome.error is not None:
                            exc = outcome.error
                            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                                raise exc
                            error = f"{type(exc).__name__}: {exc}"
                            telemetry.record_error(f"extract:{unit.unit_id}", error)
                            result.units_failed += 1
                            task_records.append(
                                {
                                    "unit_id": unit.unit_id,
                                    "order": unit.order,
                                    "state": "failed",
                                    "output_path": None,
                                    "cached": False,
                                    "error": error,
                                    "updated_at": _now_iso(),
                                }
                            )
                            emit_progress(
                                ProgressStage.EXTRACTING_CHARACTERS_AND_ENTITIES,
                                3,
                                TOTAL_STAGES,
                                message=f"unit {unit.unit_id} failed",
                                sub_current=position,
                                sub_total=len(units),
                            )
                            if fail_fast:
                                raise exc
                            continue

                        output = _coerce_output(outcome.output, unit.unit_id)  # type: ignore[arg-type]
                        atomic_write_json(unit_plan.output_path, output.model_dump())
                        key = generate_idempotency_key(
                            cfg.book_id,
                            unit.title,
                            unit.seq,
                            operation="extract",
                            content_hash=unit.sha256,
                        )
                        warnings = idempotency.safe_mark_complete(
                            key,
                            result=output.model_dump(),
                            model_name=model_id,
                            prompt_hash=prompt_hash,
                        )
                        if warnings:
                            log_decision(
                                "idempotency",
                                ["accept", "recompute"],
                                "accept",
                                warnings[0],
                            )
                        result.units_processed += 1
                        if output.abstain:
                            result.units_abstained += 1
                        log_decision(
                            "engine_call",
                            ["extract"],
                            "extract",
                            f"{unit.unit_id} in {outcome.duration_ms:.1f}ms",
                            unit_id=unit.unit_id,
                            duration_ms=round(outcome.duration_ms, 2),
                        )
                    elif unit_plan.state == "cached" and unit_plan.output is not None:
                        # Re-materialize the artifact from the cache; the write
                        # stays on the main thread (single writer).
                        atomic_write_json(unit_plan.output_path, unit_plan.output.model_dump())

                    if output is not None:
                        outputs[unit.unit_id] = output
                        if cached:
                            result.units_cached += 1
                        counts = _counts(output)
                        output_rel = str(unit_plan.output_path.relative_to(work_root)).replace(
                            "\\", "/"
                        )
                        task_records.append(
                            {
                                "unit_id": unit.unit_id,
                                "order": unit.order,
                                "state": "abstained" if output.abstain else "extracted",
                                "output_path": output_rel,
                                "output_sha256": _sha256_file(unit_plan.output_path),
                                "cached": cached,
                                "counts": counts,
                                "error": error,
                                "updated_at": _now_iso(),
                            }
                        )
                        # C1-C4: extractor receipt + verifier input (original
                        # text + spans) and verdicts (no LLM => unsure only).
                        role_runtime.record_extraction(
                            work_root,
                            unit,
                            output,
                            output_path=output_rel,
                            counts=counts,
                            cached=cached,
                        )
                        verification = role_runtime.record_verification(
                            work_root, unit, output, llm=verifier_llm
                        )
                        for verdict_name, verdict_count in verification["counts"].items():
                            telemetry.increment(
                                f"verifier_{verdict_name}", verdict_count
                            )
                        checkpoint_mgr.mark_complete(unit.unit_id)

                    # A22: the unit's terminal state, at the same single-writer point as the task record
                    # so the two cannot disagree about what happened.
                    try:
                        doc_status.transition(
                            unit.unit_id,
                            "failed" if error else "processed",
                            detail=error or "",
                            at=_now_iso(),
                        )
                    except Exception as exc:  # noqa: BLE001 - status must not fail a run
                        doc_status_errors.append(
                            f"{unit.unit_id}: {type(exc).__name__}: {exc}"
                        )
                    doc_status.save()
                    emit_progress(
                        ProgressStage.EXTRACTING_CHARACTERS_AND_ENTITIES,
                        3,
                        TOTAL_STAGES,
                        message=f"unit {unit.unit_id}",
                        sub_current=position,
                        sub_total=len(units),
                    )
        if doc_status_errors:
            log_decision(
                "doc_status",
                ["recorded", "errors"],
                "errors",
                "; ".join(doc_status_errors[:3]),
            )
        log_decision(
            "roles",
            list(role_runtime.FIVE_ROLES),
            "recorded",
            "extractor tasks/receipts and verifier inputs/verdicts written "
            "under work/ (no LLM: uncertain items stay unsure)",
        )

        write_jsonl(plan / "tasks.jsonl", task_records)
        if result.units_failed:
            telemetry.increment("units_failed", result.units_failed)

        # A7: cleaning + conflict checks through the ONE documented entry point.
        #
        # Previously the pipeline called detect_input_conflicts directly and never
        # ran the ten cleaning classes at all. ``validate_mapped_outputs`` is the
        # function that does both in the right order (clean, then check), and it had
        # no caller. Using it also means the cleaning rules cannot drift away from
        # the validation that depends on them.
        from cbb_stage1.normalize.facts import validate_mapped_outputs

        input_records = [
            (unit.order, outputs[unit.unit_id])
            for unit in units
            if unit.unit_id in outputs
        ]
        unit_texts_for_cleaning: dict[int, str] = {}
        for unit in units:
            try:
                unit_texts_for_cleaning[unit.order] = (
                    work_root / "plan" / "texts" / f"{unit.unit_id.replace(':', '_')}.txt"
                ).read_text(encoding="utf-8")
            except OSError:
                continue
        try:
            report = validate_mapped_outputs(
                input_records, unit_texts=unit_texts_for_cleaning or None
            )
        except Exception as exc:  # noqa: BLE001 - a broken validator must not abort
            log_decision(
                "normalize_validation",
                ["clean", "input_conflicts"],
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
            report = None
        if report is not None:
            # ValidationReport.outputs is list[EngineOutput] in unit order (not
            # (order, output) pairs); map them back onto the unit ids.
            cleaned = list(getattr(report, "outputs", None) or ())
            ordered_units = [u for u in units if u.unit_id in outputs]
            for unit, output in zip(ordered_units, cleaned):
                outputs[unit.unit_id] = output
            rules = getattr(report, "rules", None) or {}
            fired = {name: count for name, count in rules.items() if count}
            conflicts = getattr(report, "conflicts", None) or ()
            log_decision(
                "normalize_validation",
                ["clean", "input_conflicts"],
                "cleaned" if fired else "clean",
                f"cleaning={fired or 'no-op'}, conflicts={len(conflicts)}",
            )
            telemetry.increment("a7_validated_units", len(input_records))
            if conflicts:
                telemetry.increment("a7_input_conflicts", len(conflicts))

        # A7 second half: six conflict input checks over each unit's EngineOutput.
        from cbb_stage1.fact_validator import conflicts_as_jsonl, detect_input_conflicts

        input_records = [
            (unit.order, outputs[unit.unit_id])
            for unit in units
            if unit.unit_id in outputs
        ]
        input_conflicts = detect_input_conflicts(input_records)
        (plan / "input-conflicts.jsonl").write_text(
            conflicts_as_jsonl(input_conflicts, cfg.book_id),
            encoding="utf-8",
        )
        # (kept for the file artifact; the report above already ran the checks)
        log_decision(
            "fact_validator",
            ["clean", "input_conflicts"],
            "input_conflicts",
            f"{len(input_conflicts)} input conflict(s) written to "
            "plan/input-conflicts.jsonl",
        )

        # -- 4. RESOLVING_DUPLICATE_NAMES -----------------------------------
        with _stage_scope(
            3,
            ProgressStage.RESOLVING_DUPLICATE_NAMES,
            f"{len(outputs)} engine outputs",
        ):
            # 4a. Legacy deterministic candidate snapshot. It is kept as a
            # work-table view for consumers/tests written before wiring; the
            # postprocess integration tables below are authoritative and may
            # overwrite entity_candidates.jsonl.
            candidates = collectors.collect_candidates(units, outputs)
            collectors.write_candidate_files(candidates, resolve_dir)
            result.mentions = len(candidates["mentions"])
            result.entities = len(candidates["entities"])
            result.assertions = len(candidates["assertions"])
            result.events = len(candidates["events"])
            result.cues = len(candidates["cues"])

            # 4b. Postprocess: resolve -> assemble -> claims -> cues -> temporal
            # -> validate -> dedupe -> derive integrations, executed as sub-steps
            # of this stage so the 9-stage contract stays unchanged.
            engine_output_payloads = [
                outputs[unit.unit_id].model_dump()
                for unit in units
                if unit.unit_id in outputs
            ]
            unit_records = [unit.to_record() for unit in units]

            def _postprocess_progress(index: int, total: int, name: str) -> None:
                emit_progress(
                    ProgressStage.RESOLVING_DUPLICATE_NAMES,
                    4,
                    TOTAL_STAGES,
                    message=f"postprocess {index}/{total}: {name}",
                    sub_current=index,
                    sub_total=total,
                )

            postprocess_result = run_postprocess(
                work_root,
                book_id=cfg.book_id,
                outputs=engine_output_payloads,
                units=unit_records,
                created_at=result.started_at,
                upstream={
                    # How many independent items a postprocess stage may work on at
                    # once. Stated explicitly so a stage reads one key instead of
                    # introspecting Config, and so the gateway's max_concurrency can
                    # be set to at least this value.
                    "max_workers": int(getattr(cfg, "max_workers", 1) or 1),
                    # B12: without this the abstain gate could only ever see its own default.
                    "abstain_threshold": float(
                        getattr(cfg, "abstain_threshold", 0.5) or 0.5
                    ),
                    "nli": nli,
                    "nli_cfg": cfg,
                    "config": cfg,
                    "llm": aux_llm,
                    "reasoning_llm": reasoning_llm,
                    "verifier_llm": verifier_llm,
                    "judge_llm": judge_llm,
                    # AVeriTeC quote verification (H04): ask the verifier whether a
                    # statement's quote was actually said by its speaker. One call per
                    # statement — measured on a real chapter: 2 statements. The verdict
                    # is recorded on the assertion; nothing is deleted.
                    "verify_speaker": True,
                    # The verifier falls back to `aux_llm` when no separate endpoint is
                    # configured, so the check can run on the same model that produced
                    # the record. §5.15 would not accept that as independent, so the
                    # fact is reported beside the verdict instead of hidden.
                    "verifier_same_model": verifier_same_model,
                    # Optional embedder carried into every postprocess stage
                    # (``upstream["embedder"]``): ``derive`` uses it for RAPTOR
                    # and ``dedupe`` uses it for cosine KNN entity/description
                    # clustering; both keep their deterministic path when None.
                    "embedder": embedder,
                },
                stages=postprocess_stages,
                progress_cb=_postprocess_progress,
            )
            result.postprocess = {
                "ok": postprocess_result["ok"],
                "stages_total": postprocess_result["stages_total"],
                "stages_ok": postprocess_result["stages_ok"],
                "counts": postprocess_result["counts"],
                "paths": postprocess_result["paths"],
                "wiring_path": postprocess_result["wiring_path"],
            }
            telemetry.increment("postprocess_stages", postprocess_result["stages_ok"])
            # C1-C4: one task/receipt pair per postprocess stage
            # (resolver / arbiter / abducer / verifier role mapping).
            role_records = role_runtime.record_postprocess_chain(
                work_root, postprocess_result["stages"], book_id=cfg.book_id
            )
            telemetry.increment("role_postprocess_tasks", len(role_records))
            log_decision(
                "roles",
                list(role_runtime.FIVE_ROLES),
                "recorded",
                f"postprocess chain recorded {len(role_records)} task/receipt "
                "pairs (resolver/arbiter/abducer/verifier)",
            )
        log_decision(
            "resolve",
            ["shell_candidates", "postprocess_integrations"],
            "postprocess_integrations",
            "postprocess ran resolve/assemble/claims/cues/temporal/validate/"
            "dedupe/derive integrations in order; the exact-name candidate "
            "snapshot is superseded by the resolve integration table",
        )

        # -- 5. GENERATING_DESCRIPTIONS ------------------------------------
        with _stage_scope(
            4, ProgressStage.GENERATING_DESCRIPTIONS, f"{result.entities} entity candidates"
        ):
            cards_path = assemble_dir / "entity_cards.jsonl"
            if cards_path.exists():
                # The dedupe/derive integration already produced the canonical
                # card table; do not overwrite it with the deterministic shell.
                cards = read_jsonl(cards_path)
                log_decision(
                    "descriptions",
                    ["postprocess_table", "deterministic_shell"],
                    "postprocess_table",
                    "assemble/entity_cards.jsonl came from a postprocess "
                    "integration; the shell does not overwrite it",
                )
            else:
                cards = collectors.build_entity_cards(candidates["entities"], outputs)
                write_jsonl(cards_path, cards)
                log_decision(
                    "descriptions",
                    ["postprocess_table", "deterministic_shell"],
                    "deterministic_shell",
                    "no postprocess entity_cards table; wrote the deterministic "
                    "card snapshot (no synthesized descriptions)",
                )

        # -- 6. BUILDING_CONNECTIONS ---------------------------------------
        with _stage_scope(5, ProgressStage.BUILDING_CONNECTIONS, "unit co-occurrence"):
            connections = collectors.build_connections(
                units, outputs, min_weight=min_connection_weight
            )
            write_jsonl(assemble_dir / "connections.jsonl", connections)
            result.connections = len(connections)
        log_decision(
            "connections",
            ["cooccurrence"],
            "cooccurrence",
            f"{len(connections)} pairs with weight >= {min_connection_weight}",
        )

        # -- 7. WRITING_OUTPUT ---------------------------------------------
        with _stage_scope(6, ProgressStage.WRITING_OUTPUT, "work artifacts"):
            engine_index = collectors.build_engine_index(
                units, outputs, engine_dir, work_root
            )
            write_jsonl(engine_dir / "index.jsonl", engine_index)
            write_jsonl(plan / "units.jsonl", [unit.to_record() for unit in units])
            write_jsonl(plan / "tasks.jsonl", task_records)
        log_decision(
            "write_output",
            ["work_only"],
            "work_only",
            "box writing is a separate task; this stage writes work artifacts only",
        )

        # -- 8. SAVING_STATE -----------------------------------------------
        with _stage_scope(7, ProgressStage.SAVING_STATE, "state/run/registry"):
            result.ended_at = _now_iso()
            result.status = "partial" if result.units_failed else "complete"
            telemetry.increment("units_processed", result.units_processed)
            telemetry.increment("units_cached", result.units_cached)
            telemetry.increment("units_abstained", result.units_abstained)
            for name in ("mentions", "entities", "assertions", "events", "cues"):
                telemetry.increment(name, getattr(result, name))

            from cbb_stage1.ingest.registry import (
                create_book_record,
                load_registry,
                save_registry,
            )

            registry = load_registry(plan / "ingestion_log.json")
            registry.add_record(
                create_book_record(
                    book_id=cfg.book_id,
                    file_path=Path(cfg.book_path),
                    content_hash=book.content_hash,
                    title=book.title,
                    author=book.author,
                    chapter_count=len(book.chapters),
                    word_count=book.total_words,
                )
            )
            registry.update_entity_count(cfg.book_id, result.entities)
            save_registry(registry, plan / "ingestion_log.json")

            state = {
                "book_id": cfg.book_id,
                "cursor": len(task_records),
                "units_total": result.units_total,
                "units_processed": result.units_processed,
                "units_cached": result.units_cached,
                "units_failed": result.units_failed,
                "status": result.status,
                "trace_id": trace_id,
                "postprocess": result.postprocess,
                "updated_at": _now_iso(),
            }
            atomic_write_json(plan / "state.json", state)
            atomic_write_json(plan / "run.json", result.to_dict())
            result.run_path = str(plan / "run.json")

        # -- 9. COMPLETE ---------------------------------------------------
        with _stage_scope(
            8,
            ProgressStage.COMPLETE,
            f"{result.units_total} units, status={result.status}",
        ):
            pass

        # Observability flush happens after the COMPLETE stage so the trace and
        # progress log contain all 9 stages (and the COMPLETE span).
        trace = end_trace()
        if trace is not None:
            trace.metadata["result"] = result.to_dict()
            save_trace(trace, traces_dir)
        save_progress_log(progress_dir, trace_id)
        telemetry.stop_timer("pipeline_total")
        telemetry.save_telemetry(telemetry_dir, trace_id)

        return result

    except Exception as exc:
        telemetry.record_error("pipeline", str(exc), fatal=True)
        result.status = "failed"
        result.ended_at = _now_iso()
        if isinstance(exc, PostprocessError):
            # Failure is auditable: stage name + partial wiring artifact.
            result.postprocess = {
                "ok": False,
                "stage": exc.stage,
                "wiring_path": exc.wiring_path,
                "error": str(exc),
            }
            log_decision(
                "postprocess_failed",
                ["abort", "skip_silently"],
                "abort",
                str(exc),
                stage=exc.stage,
                wiring_path=exc.wiring_path,
            )
            # Record the partial chain (completed + failing stage) as C1-C4
            # task/receipt pairs; a failure here must never mask the root cause.
            wiring_payload = load_wiring(work_root)
            if wiring_payload is not None:
                try:
                    role_runtime.record_postprocess_chain(
                        work_root,
                        wiring_payload.get("stages", []),
                        book_id=cfg.book_id,
                    )
                except Exception as role_exc:  # noqa: BLE001 - audit only
                    telemetry.record_error(
                        "roles", f"{type(role_exc).__name__}: {role_exc}"
                    )
        trace = end_trace()
        if trace is not None:
            trace.metadata["result"] = result.to_dict()
            try:
                save_trace(trace, traces_dir)
            except OSError:
                pass
        try:
            save_progress_log(progress_dir, trace_id)
        except OSError:
            pass
        try:
            telemetry.stop_timer("pipeline_total")
            telemetry.save_telemetry(telemetry_dir, trace_id)
        except OSError:
            pass
        try:
            atomic_write_json(
                plan / "state.json",
                {
                    "book_id": cfg.book_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "postprocess": result.postprocess,
                    "trace_id": trace_id,
                    "updated_at": _now_iso(),
                },
            )
            atomic_write_json(plan / "run.json", result.to_dict())
            result.run_path = str(plan / "run.json")
        except OSError:
            pass
        raise
