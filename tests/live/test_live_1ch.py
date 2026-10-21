"""Live smoke harness: one real chapter through a real LLM engine (P3-D).

This module is **opt-in** and never fakes a live result:

* default suite: ``CBB_LIVE`` is unset -> the live test is skipped with an
  explicit ``BLOCKED: ...`` reason (pytest ``skip``, printed to stdout);
* to run it: export ``CBB_LIVE=1`` plus a real provider configuration
  (``CBB_LLM_PROVIDER`` / ``CBB_LLM_BASE_URL`` / ``CBB_LLM_API_KEY`` /
  ``CBB_LLM_MODEL``); optionally point ``CBB_LIVE_BOOK`` at a real text/epub
  (defaults to the 2-chapter fixture).  Then exactly one chapter is extracted
  through the configured engine and the real artifact + counts are asserted and
  printed as a receipt.

Missing credentials or an unreachable Ollama endpoint -> ``pytest.skip`` with a
``BLOCKED`` reason (environment absence is reported, never papered over).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cbb_stage1.config import Config
from cbb_stage1.engines.factory import build_engine
from cbb_stage1.models import EngineOutput
from cbb_stage1.orchestrator.pipeline import run_pipeline

pytestmark = pytest.mark.live

FIXTURE_BOOK = Path(__file__).resolve().parents[1] / "fixtures" / "orchestrator" / "book.txt"
_ENV_TRUE = {"1", "true", "yes", "on"}
_CLOUD_PROVIDERS = {"openai", "anthropic", "deepseek", "qwen", "kimi", "ai_reader"}


def _enabled() -> bool:
    return os.environ.get("CBB_LIVE", "").strip().lower() in _ENV_TRUE


def _blocked(reason: str):
    """Report a BLOCKED environment condition and skip (never fake results)."""
    print(f"BLOCKED: {reason}")
    pytest.skip(f"BLOCKED: {reason}")


def _live_target() -> tuple[str, str, Path]:
    """Return ``(provider, model, book_path)`` or skip with a BLOCKED reason."""
    if not _enabled():
        _blocked(
            "live harness disabled (set CBB_LIVE=1 with a real provider config); "
            "no live extraction result is fabricated"
        )

    provider = (
        os.environ.get("CBB_LIVE_PROVIDER")
        or os.environ.get("CBB_LLM_PROVIDER")
        or "openai"
    ).strip().lower()
    model = (os.environ.get("CBB_LLM_MODEL") or "").strip()
    base_url = (os.environ.get("CBB_LLM_BASE_URL") or "").strip()
    api_key = (os.environ.get("CBB_LLM_API_KEY") or "").strip()

    if provider in ("mock", ""):
        _blocked("provider is mock; a live chapter must use a real provider")

    if provider in _CLOUD_PROVIDERS:
        missing = [
            name
            for name, value in (
                ("CBB_LLM_API_KEY", api_key),
                ("CBB_LLM_BASE_URL", base_url),
            )
            if not value
        ]
        if missing:
            _blocked(
                f"missing live credentials: {', '.join(missing)} "
                f"(provider={provider}); no live results fabricated"
            )
    elif provider in ("ollama", "local"):
        try:
            import httpx

            response = httpx.get(
                (base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/tags",
                timeout=3.0,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - environment absence
            _blocked(f"Ollama endpoint unreachable ({type(exc).__name__}: {exc})")
    else:
        _blocked(f"unsupported live provider {provider!r}")

    book = Path(os.environ.get("CBB_LIVE_BOOK") or FIXTURE_BOOK)
    if not book.exists():
        _blocked(f"live book not found: {book}")
    return provider, model or provider, book


# ── gate self-check (runs even when live is disabled) ──────────


def test_live_gate_reports_blocked_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """No live opt-in -> an explicit BLOCKED skip, not a silent pass."""
    monkeypatch.delenv("CBB_LIVE", raising=False)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _live_target()
    assert "BLOCKED:" in str(excinfo.value)
    assert "disabled" in str(excinfo.value)


# ── the live run (skipped unless configured) ───────────────────


def test_live_one_chapter_runs_through_real_engine(tmp_path: Path) -> None:
    provider, model, book = _live_target()

    cfg = Config(
        book_id="livetest",
        book_path=book,
        out_dir=tmp_path / "box",
        work_dir=tmp_path / "work",
        provider=provider,
        model=model,
        live=True,
    )
    engine = build_engine(provider, model=model, max_gleanings=cfg.max_gleanings)

    result = run_pipeline(cfg, chapters="1", engine=engine)

    # A real run must produce an artifact for the one chapter and must not fail.
    engine_dir = tmp_path / "work" / "engine"
    artifacts = sorted(engine_dir.glob("*.json"))
    assert artifacts, "live run wrote no engine artifact"
    assert result.units_processed >= 1
    assert result.units_failed == 0

    output = EngineOutput.model_validate(
        json.loads(artifacts[0].read_text(encoding="utf-8"))
    )
    assert output.unit_id.startswith("u:")
    # Abstain is allowed for a real model, but it must carry an explicit reason.
    if output.abstain:
        assert output.notes and any("abstain" in note or "failed" in note for note in output.notes)

    receipt = {
        "provider": provider,
        "model": model,
        "book": str(book),
        "unit_id": output.unit_id,
        "abstain": output.abstain,
        "mentions": len(output.mentions),
        "entities": len(output.entities),
        "assertions": len(output.assertions),
        "events": len(output.events),
        "pipeline_status": result.status,
        "units_processed": result.units_processed,
    }
    print("LIVE RECEIPT: " + json.dumps(receipt, ensure_ascii=False))
