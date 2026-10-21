"""Idempotency - content-hash keyed result cache with atomic state writes.

Ported from B09 ``src/observability/idempotency.py`` (read-only source). Shape is
preserved (key generation, processed/cached lookups, staleness on model/prompt
change, double-run safety, model invalidation). Adaptations:

* state file path is explicit (:class:`IdempotencyStore`), not a repo-global
  constant, so runs/tests do not share cache state;
* no Ollama-specific wording: the engine/model identifiers come from
  ``cbb_stage1.config`` / the engine fingerprint;
* ``safe_mark_complete`` returns the non-determinism warnings it prints so the
  caller can record them in the trace.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from cbb_stage1.ids import sha256_hex_text

__all__ = [
    "IdempotencyStore",
    "generate_idempotency_key",
    "generate_content_hash",
]

_STATE_VERSION = 1


def generate_idempotency_key(
    book_id: str,
    chapter_title: str,
    chunk_idx: int,
    operation: str = "extraction",
    content_hash: str | None = None,
) -> str:
    """Generate a 16-hex idempotency key for one unit operation.

    Content-hash keys deduplicate re-chunked/renamed units; positional keys are
    the fallback when no hash is available.
    """
    if content_hash:
        key_parts = f"{book_id}:{operation}:{content_hash}"
    else:
        key_parts = f"{book_id}:{chapter_title}:{chunk_idx}:{operation}"
    return sha256_hex_text(key_parts, 16)


def generate_content_hash(content: str) -> str:
    """Return the first 16 hex chars of the SHA-256 digest of ``content``."""
    return sha256_hex_text(content, 16)


class IdempotencyStore:
    """A JSON-file backed cache of completed pipeline operations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -- state management (atomic read/write) --

    def _load_state(self) -> dict:
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(state, dict) and isinstance(state.get("processed"), dict):
                    return state
            except (json.JSONDecodeError, OSError):
                pass
        return {"processed": {}, "version": _STATE_VERSION}

    def _save_state_atomic(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # -- processing checks --

    def is_processed(self, key: str) -> bool:
        """True when ``key`` exists in the cache."""
        return key in self._load_state()["processed"]

    def get_cached_result(self, key: str) -> dict | None:
        """Return the cached result for ``key``, or ``None``."""
        entry = self._load_state()["processed"].get(key)
        return entry.get("result") if entry else None

    def is_stale(
        self,
        key: str,
        model_name: str | None = None,
        prompt_hash: str | None = None,
    ) -> bool:
        """True when the cached result should be recomputed.

        A missing entry is always stale; a changed model or prompt hash makes
        an existing entry stale.
        """
        entry = self._load_state()["processed"].get(key)
        if not entry:
            return True
        if model_name and entry.get("model_name") != model_name:
            return True
        if prompt_hash and entry.get("prompt_hash") != prompt_hash:
            return True
        return False

    # -- marking completion --

    def mark_complete(
        self,
        key: str,
        result: dict | None = None,
        model_name: str | None = None,
        prompt_hash: str | None = None,
    ) -> None:
        """Record an operation as completed and cache its result."""
        state = self._load_state()
        state["processed"][key] = {
            "completed_at": datetime.now().isoformat(),
            "result": result,
            "model_name": model_name,
            "prompt_hash": prompt_hash,
        }
        self._save_state_atomic(state)

    def safe_mark_complete(
        self,
        key: str,
        result: dict | None = None,
        model_name: str | None = None,
        prompt_hash: str | None = None,
    ) -> list[str]:
        """Mark complete with a double-run non-determinism check.

        Returns a list of warning strings (empty when the cached and new results
        agree). The entry is updated regardless of the warnings, matching the
        source behaviour: a double run must not crash, but non-determinism is
        surfaced for triage.
        """
        warnings: list[str] = []
        state = self._load_state()
        existing = state["processed"].get(key)
        if existing and result is not None:
            existing_hash = sha256_hex_text(
                json.dumps(existing.get("result"), sort_keys=True), 12
            )
            new_hash = sha256_hex_text(json.dumps(result, sort_keys=True), 12)
            if existing_hash != new_hash:
                warnings.append(
                    f"non-determinism for key {key[:8]}: "
                    f"existing={existing_hash}, new={new_hash}"
                )
                print(f"  ! {warnings[-1]}")

        state["processed"][key] = {
            "completed_at": datetime.now().isoformat(),
            "result": result,
            "model_name": model_name,
            "prompt_hash": prompt_hash,
        }
        self._save_state_atomic(state)
        return warnings

    # -- management --

    def clear_all(self) -> None:
        """Delete all idempotency state from disk."""
        if self.path.exists():
            self.path.unlink()

    def get_stats(self) -> dict:
        """Return ``total_processed`` and ``untracked_entries`` counts."""
        processed = self._load_state()["processed"]
        untracked = sum(1 for entry in processed.values() if entry.get("model_name") is None)
        return {"total_processed": len(processed), "untracked_entries": untracked}

    def invalidate_for_model(self, model_name: str) -> int:
        """Invalidate all cached results produced by ``model_name``."""
        state = self._load_state()
        to_remove = [
            key
            for key, entry in state["processed"].items()
            if entry.get("model_name") == model_name
        ]
        for key in to_remove:
            del state["processed"][key]
        if to_remove:
            self._save_state_atomic(state)
            print(f"[idempotency] invalidated {len(to_remove)} entries for {model_name}")
        return len(to_remove)
