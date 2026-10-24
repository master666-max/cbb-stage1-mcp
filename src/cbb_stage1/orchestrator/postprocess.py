"""Postprocess orchestration - wire the deterministic module packages into the pipeline.

The 2026-10-18 wiring audit (``设计/抽取段-v6/12`` §3, ``13``) found that the
identity / assembly / claims / cue / temporal / validation / dedupe / derive
packages existed with unit tests but were **not on any runtime call path**: the
box writer had grown parallel simplified implementations instead. This module
is the single adapter that puts them on the pipeline's 9-stage path.

Contract (identical for every package ``<pkg>/integration.py``)::

    run(
        work_root: Path,
        *,
        book_id: str,
        outputs: list[dict],   # EngineOutput.model_dump() for this run
        units: list[dict],     # sources/units.jsonl records for this run
        created_at: str,
        upstream: dict,        # {completed_stage_name: returned dict}
    ) -> dict                  # {"counts": dict, "paths": list|dict, ...}

This orchestrator:

* calls the eight integrations **in the fixed order** resolve → assemble →
  claims → cues → temporal → validate → dedupe → derive;
* threads ``upstream`` so a later stage can consume an earlier stage's result;
* writes ``work/plan/wiring.json`` (contract ``cbb-wiring/1``) with per-stage
  status / counts / paths / duration and the aggregated totals;
* **fails loudly**: a missing ``integration.py``, a missing ``run`` callable, a
  non-dict result or an exception inside a stage writes the partial wiring file
  and then raises :class:`PostprocessError`. Nothing is skipped silently and no
  stage result is fabricated.

The module does not import box/ or schemas; it only calls integration ``run``
callables and writes the wiring artifact.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .files import atomic_write_json, read_json

__all__ = [
    "parse_stage_names",
    "WIRING_CONTRACT",
    "StageSpec",
    "DEFAULT_STAGES",
    "PostprocessError",
    "run_postprocess",
    "load_wiring",
]

#: Contract id recorded in ``work/plan/wiring.json``.
WIRING_CONTRACT = "cbb-wiring/1"

#: Relative path of the wiring artifact under the work root.
WIRING_RELATIVE_PATH = "plan/wiring.json"


@dataclass(frozen=True)
class StageSpec:
    """One integration stage: where it lives **and what it promises**.

    A bare `(name, module)` pair answers "what runs" but not "may I skip it" or "what proves it ran".
    The three added fields are what a gate needs without importing eight modules:

    * ``required_outputs`` — work-relative tables that must exist after the stage ran;
    * ``required`` — whether a box is valid without this stage (a box without `validate` is not);
    * ``criterion`` — the one sentence a reader needs to judge "did it work";
    * ``toggle`` — for an optional stage, the config field that turns it off. Required for optional
      stages: a switch nobody can find is not a switch.
    """

    name: str
    module: str
    required_outputs: tuple[str, ...] = ()
    required: bool = True
    criterion: str = ""
    toggle: str = ""


#: Fixed call order (task P0-A). Later stages may read earlier stages' tables
#: through ``upstream`` and the work directory.
DEFAULT_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "resolve",
        "cbb_stage1.resolve.integration",
        required_outputs=("resolve/entity_candidates.jsonl",),
        required=True,
        criterion="every accepted entity has a content-addressed id; merges are logged",
    ),
    StageSpec(
        "assemble",
        "cbb_stage1.assemble.integration",
        required_outputs=("assemble/assertions.jsonl",),
        required=True,
        criterion="every assertion has a registered predicate, typed roles and a claim layer",
    ),
    StageSpec(
        "claims",
        "cbb_stage1.claims.integration",
        required_outputs=("assemble/claims.jsonl",),
        required=False,
        criterion="every claim carries an AVeriTeC verdict and the evidence behind it",
        toggle="claims_enabled",
    ),
    StageSpec(
        "cues",
        "cbb_stage1.cues.integration",
        required_outputs=("assemble/cues.jsonl",),
        required=False,
        criterion="cue state transitions are recorded with the unit that caused them",
        toggle="cues_enabled",
    ),
    StageSpec(
        "temporal",
        "cbb_stage1.temporal.integration",
        required_outputs=("assemble/timeline.jsonl",),
        required=False,
        criterion="every constraint is either verified or explicitly unknown",
        toggle="temporal_enabled",
    ),
    StageSpec(
        "validate",
        "cbb_stage1.validate.integration",
        required_outputs=("audit/conflicts.jsonl",),
        required=True,
        criterion="conflicts, quarantine and decisions are written; nothing is silently dropped",
    ),
    StageSpec(
        "dedupe",
        "cbb_stage1.dedupe.integration",
        required_outputs=("dedupe/clusters.jsonl",),
        required=False,
        criterion="duplicate assertions are merged through a logged, reversible merge",
        toggle="description_synthesis",
    ),
    StageSpec(
        "derive",
        "cbb_stage1.derive.integration",
        required_outputs=("derive/hypotheses.jsonl",),
        required=False,
        criterion="derived rows say they are derived and carry no fabricated confidence",
        toggle="summary_synthesis",
    ),
)


#: Optional stage -> the config field that switches it off. Derived from the specs so the two cannot
#: disagree; a test pins the equality.
OPTIONAL_STAGE_TOGGLES: dict[str, str] = {
    spec.name: spec.toggle for spec in DEFAULT_STAGES if spec.toggle
}


def stage_registry() -> dict[str, StageSpec]:
    """name -> spec, for callers that look a stage up by name.

    One registry rather than a second list: `DEFAULT_STAGES` is the source and this is a view of it.
    """
    return {spec.name: spec for spec in DEFAULT_STAGES}


def parse_stage_names(names: str | None) -> tuple[StageSpec, ...]:
    """Turn ``"resolve,derive"`` into the matching :class:`StageSpec` tuple.

    The result follows :data:`DEFAULT_STAGES` order, not the order given: later stages read tables
    earlier stages write, so honouring a reversed list would run `derive` against stale input.

    Unknown names raise instead of being skipped. A typo that quietly dropped `validate` would
    produce a box of unvalidated assertions, which is exactly the failure this project keeps finding.
    Duplicates collapse; an empty selection raises, because "run nothing" is almost never meant.
    """
    wanted = [piece.strip() for piece in str(names or "").split(",") if piece.strip()]
    if not wanted:
        raise ValueError("no stage names given")
    known = {spec.name: spec for spec in DEFAULT_STAGES}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(
            f"unknown stage name(s): {unknown}; known are {sorted(known)}"
        )
    selected = set(wanted)
    return tuple(spec for spec in DEFAULT_STAGES if spec.name in selected)


class PostprocessError(RuntimeError):
    """Raised when a postprocess stage cannot run or fails.

    Attributes:
        stage: Name of the failing stage, when known.
        wiring_path: Path of the (partial) wiring artifact that was written.
        cause: Original exception, if the stage raised.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        wiring_path: str | None = None,
        cause: BaseException | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.wiring_path = wiring_path
        self.cause = cause


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_paths(raw: Any, work_root: Path | None = None) -> list[str]:
    """Normalize a stage's ``paths`` return into a flat list of strings.

    Absolute paths under ``work_root`` are rewritten relative to the work root
    (POSIX separators) so ``wiring.json`` is portable across machines/runs.
    """
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        values: list[Any] = [raw]
    elif isinstance(raw, dict):
        values = list(raw.values())
    elif isinstance(raw, Iterable):
        values = list(raw)
    else:
        values = [raw]

    paths: list[str] = []
    root: Path | None = None
    if work_root is not None:
        try:
            root = Path(work_root).resolve()
        except OSError:
            root = None
    for value in values:
        text = str(value)
        if root is not None:
            try:
                candidate = Path(text)
                if candidate.is_absolute():
                    text = candidate.resolve().relative_to(root).as_posix()
            except (OSError, ValueError):
                pass
        paths.append(text)
    return paths


def _normalize_counts(raw: Any) -> dict[str, Any]:
    """Split a stage's ``counts`` into numeric counters and structured details.

    Dropping every non-numeric value hid real measurements: validate reports
    ``confidence_status`` as a mapping, and it was silently discarded, so a
    receipt could not distinguish "confidence was computed" from "confidence was
    never attempted" -- exactly the question the B2/B4/B5 audit needed answered.
    Structured readings are now kept under ``details``.
    """
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if isinstance(value, bool):
            details[name] = value
        elif isinstance(value, (int, float)):
            counts[name] = value
        elif value not in (None, {}, []):
            details[name] = value
    if details:
        counts["details"] = details
    return counts


def _write_wiring(
    work_root: Path,
    *,
    book_id: str,
    created_at: str,
    stages: Sequence[StageSpec],
    records: list[dict],
    ok: bool,
    error: str | None,
) -> Path:
    """Write ``work/plan/wiring.json`` (atomic) and return its path."""
    # Only numeric counters are summed. Structured readings (``details``) are
    # merged per stage instead of added: a mapping cannot be summed, and before
    # they were preserved they were simply dropped — which is why a receipt could
    # not show whether confidence had been computed.
    aggregated: dict[str, int | float] = {}
    detail_blocks: dict[str, dict[str, Any]] = {}
    for record in records:
        stage_name = str(record.get("stage") or record.get("name") or "")
        for key, value in record.get("counts", {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                if stage_name and value not in (None, {}, []):
                    detail_blocks.setdefault(stage_name, {})[str(key)] = value
                continue
            aggregated[key] = aggregated.get(key, 0) + value
    paths: list[str] = []
    for record in records:
        for path in record.get("paths", []):
            if path not in paths:
                paths.append(path)

    payload = {
        "contract": WIRING_CONTRACT,
        "book_id": book_id,
        "created_at": created_at,
        "details": detail_blocks,
        "ok": ok,
        "error": error,
        "order": [spec.name for spec in stages],
        "stages": records,
        "stages_total": len(stages),
        "stages_ok": sum(1 for record in records if record.get("status") == "ok"),
        "counts": aggregated,
        "paths": paths,
    }
    wiring_path = work_root / WIRING_RELATIVE_PATH
    atomic_write_json(wiring_path, payload)
    return wiring_path


def _stage_record(
    spec: StageSpec,
    *,
    status: str,
    duration_ms: float,
    counts: dict | None = None,
    paths: list[str] | None = None,
    error: str | None = None,
) -> dict:
    return {
        "name": spec.name,
        "module": spec.module,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "counts": counts or {},
        "paths": paths or [],
        "error": error,
    }


def run_postprocess(
    work_root: str | Path,
    *,
    book_id: str,
    outputs: list[dict],
    units: list[dict],
    created_at: str | None = None,
    upstream: dict | None = None,
    stages: Sequence[StageSpec] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run the eight integrations in order and record the wiring artifact.

    Args:
        work_root: Work root (``Config.work_dir``).
        book_id: Book slug.
        outputs: ``EngineOutput.model_dump()`` dicts for this run.
        units: Unit records (``sources/units.jsonl`` shape) for this run.
        created_at: Run timestamp recorded in wiring; defaults to now.
        upstream: Initial upstream mapping (``{stage_name: result}``), e.g. a
            caller-provided pre-stage result. Completed stages are added to it.
        stages: Override the stage list (tests / partial rollouts). Defaults to
            :data:`DEFAULT_STAGES`.
        progress_cb: Optional ``(index, total, stage_name)`` callback invoked
            after each successful stage; the pipeline uses it to emit
            sub-step progress without adding a 10th pipeline stage.

    Returns:
        A dict with ``ok``, ``stages``, ``counts``, ``paths``, ``upstream`` and
        ``wiring_path``.

    Raises:
        PostprocessError: If a stage's module is unavailable, exposes no
            callable ``run``, raises, or returns a non-dict. The partial wiring
            file is written first so the failure is auditable.
    """
    work_root = Path(work_root)
    resolved_stages = tuple(stages) if stages is not None else DEFAULT_STAGES
    if not resolved_stages:
        raise ValueError("postprocess requires at least one stage")
    created_at = created_at or _now_iso()
    accumulated: dict[str, Any] = dict(upstream or {})
    records: list[dict] = []

    for index, spec in enumerate(resolved_stages, 1):
        started = time.perf_counter()
        try:
            module = importlib.import_module(spec.module)
        except ImportError as exc:
            records.append(
                _stage_record(
                    spec,
                    status="unavailable",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=f"integration module unavailable: {spec.module} ({exc})",
                )
            )
            wiring_path = _write_wiring(
                work_root,
                book_id=book_id,
                created_at=created_at,
                stages=resolved_stages,
                records=records,
                ok=False,
                error=f"stage {spec.name!r}: integration module unavailable: {spec.module}",
            )
            raise PostprocessError(
                f"postprocess stage {spec.name!r}: integration module "
                f"{spec.module!r} is unavailable; refusing to skip silently",
                stage=spec.name,
                wiring_path=str(wiring_path),
                cause=exc,
            ) from exc

        run = getattr(module, "run", None)
        if not callable(run):
            records.append(
                _stage_record(
                    spec,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=f"{spec.module} exposes no callable run(...)",
                )
            )
            wiring_path = _write_wiring(
                work_root,
                book_id=book_id,
                created_at=created_at,
                stages=resolved_stages,
                records=records,
                ok=False,
                error=f"stage {spec.name!r}: {spec.module}.run is missing",
            )
            raise PostprocessError(
                f"postprocess stage {spec.name!r}: {spec.module!r} exposes no "
                "callable run(...)",
                stage=spec.name,
                wiring_path=str(wiring_path),
            )

        try:
            result = run(
                work_root,
                book_id=book_id,
                outputs=outputs,
                units=units,
                created_at=created_at,
                upstream=accumulated,
            )
        except Exception as exc:
            records.append(
                _stage_record(
                    spec,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            wiring_path = _write_wiring(
                work_root,
                book_id=book_id,
                created_at=created_at,
                stages=resolved_stages,
                records=records,
                ok=False,
                error=f"stage {spec.name!r} failed: {type(exc).__name__}: {exc}",
            )
            raise PostprocessError(
                f"postprocess stage {spec.name!r} failed: {type(exc).__name__}: {exc}",
                stage=spec.name,
                wiring_path=str(wiring_path),
                cause=exc,
            ) from exc

        if not isinstance(result, dict):
            records.append(
                _stage_record(
                    spec,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error=f"run(...) returned {type(result).__name__}, expected dict",
                )
            )
            wiring_path = _write_wiring(
                work_root,
                book_id=book_id,
                created_at=created_at,
                stages=resolved_stages,
                records=records,
                ok=False,
                error=f"stage {spec.name!r}: non-dict result",
            )
            raise PostprocessError(
                f"postprocess stage {spec.name!r}: run(...) returned "
                f"{type(result).__name__}, expected dict",
                stage=spec.name,
                wiring_path=str(wiring_path),
            )

        counts = _normalize_counts(result.get("counts"))
        paths = _normalize_paths(result.get("paths"), work_root)
        record = _stage_record(
            spec,
            status="ok",
            duration_ms=(time.perf_counter() - started) * 1000,
            counts=counts,
            paths=paths,
        )
        records.append(record)
        accumulated[spec.name] = result
        if progress_cb is not None:
            progress_cb(index, len(resolved_stages), spec.name)

    wiring_path = _write_wiring(
        work_root,
        book_id=book_id,
        created_at=created_at,
        stages=resolved_stages,
        records=records,
        ok=True,
        error=None,
    )
    payload = read_json(wiring_path, default={}) or {}
    return {
        "ok": True,
        "book_id": book_id,
        "created_at": created_at,
        "stages": records,
        "stages_total": len(resolved_stages),
        "stages_ok": len(records),
        "counts": payload.get("counts", {}),
        "paths": payload.get("paths", []),
        "upstream": accumulated,
        "wiring_path": str(wiring_path),
    }


def load_wiring(work_root: str | Path) -> dict | None:
    """Load ``work/plan/wiring.json`` (``None`` when it has not been written)."""
    payload = read_json(Path(work_root) / WIRING_RELATIVE_PATH)
    if not isinstance(payload, dict):
        return None
    return payload
