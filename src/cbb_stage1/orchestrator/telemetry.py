"""Telemetry - lightweight structured metrics for pipeline monitoring.

Ported from B09 ``src/observability/telemetry.py`` (read-only source). Shape is
preserved (counters / histograms / gauges / timers / errors + module-level
singleton). Adaptation: :func:`save_telemetry` takes an explicit output
directory instead of reading a repo-global config.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "MetricEntry",
    "TelemetryCollector",
    "reset_telemetry",
    "increment",
    "record",
    "start_timer",
    "stop_timer",
    "record_error",
    "get_report",
    "get_summary",
    "save_telemetry",
]


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class MetricEntry:
    """A single metric sample."""

    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


class TelemetryCollector:
    """Singleton metrics collector.

    Supports counters (increment-only), histograms (value distributions), and
    gauges (current value).
    """

    _instance: "TelemetryCollector | None" = None

    def __new__(cls) -> "TelemetryCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.reset()

    def reset(self) -> None:
        """Reset all metrics. Called at the start of each pipeline run."""
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._timers: dict[str, float] = {}
        self._start_time = time.monotonic()
        self._errors: list[dict[str, Any]] = []

    # -- Counters --

    def increment(self, name: str, amount: float = 1.0) -> None:
        self._counters[name] += amount

    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    # -- Histograms --

    def record(self, name: str, value: float) -> None:
        self._histograms[name].append(value)

    def get_histogram(self, name: str) -> dict:
        values = self._histograms.get(name, [])
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0}
        values_sorted = sorted(values)
        count = len(values_sorted)
        return {
            "count": count,
            "min": round(values_sorted[0], 2),
            "max": round(values_sorted[-1], 2),
            "mean": round(sum(values_sorted) / count, 2),
            "p50": round(values_sorted[count // 2], 2),
            "p95": round(values_sorted[int(count * 0.95)], 2)
            if count > 1
            else round(values_sorted[0], 2),
            "sum": round(sum(values_sorted), 2),
        }

    # -- Gauges --

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    # -- Timers --

    def start_timer(self, name: str) -> None:
        self._timers[name] = time.monotonic()

    def stop_timer(self, name: str) -> float:
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
        duration_ms = (time.monotonic() - start) * 1000
        self.record(f"{name}_duration_ms", duration_ms)
        return duration_ms

    # -- Errors --

    def record_error(self, stage: str, error: str, fatal: bool = False) -> None:
        self._errors.append(
            {
                "stage": stage,
                "error": error,
                "fatal": fatal,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.increment("errors_total")
        if fatal:
            self.increment("errors_fatal")

    # -- Reports --

    def get_report(self) -> dict:
        return {
            "duration_s": round(time.monotonic() - self._start_time, 2),
            "counters": dict(self._counters),
            "histograms": {
                name: self.get_histogram(name) for name in self._histograms
            },
            "gauges": dict(self._gauges),
            "errors": self._errors,
            "error_count": len(self._errors),
        }

    def get_summary(self) -> dict:
        report = self.get_report()
        counters = report["counters"]
        return {
            "duration_s": report["duration_s"],
            "llm_calls": int(counters.get("llm_calls", 0)),
            "tokens_prompt": int(counters.get("tokens_prompt", 0)),
            "tokens_completion": int(counters.get("tokens_completion", 0)),
            "units_processed": int(counters.get("units_processed", 0)),
            "units_cached": int(counters.get("units_cached", 0)),
            "units_failed": int(counters.get("units_failed", 0)),
            "units_abstained": int(counters.get("units_abstained", 0)),
            "mentions": int(counters.get("mentions", 0)),
            "entities": int(counters.get("entities", 0)),
            "assertions": int(counters.get("assertions", 0)),
            "events": int(counters.get("events", 0)),
            "cues": int(counters.get("cues", 0)),
            "errors": report["error_count"],
        }


# =============================================================================
# CONVENIENCE - module-level functions bound to one collector
# =============================================================================

_collector = TelemetryCollector()


def reset_telemetry() -> None:
    """Reset all telemetry metrics for a new pipeline run."""
    _collector.reset()


def increment(name: str, amount: float = 1.0) -> None:
    _collector.increment(name, amount)


def record(name: str, value: float) -> None:
    _collector.record(name, value)


def start_timer(name: str) -> None:
    _collector.start_timer(name)


def stop_timer(name: str) -> float:
    return _collector.stop_timer(name)


def record_error(stage: str, error: str, fatal: bool = False) -> None:
    _collector.record_error(stage, error, fatal)


def get_report() -> dict:
    return _collector.get_report()


def get_summary() -> dict:
    return _collector.get_summary()


def save_telemetry(out_dir: str | Path, trace_id: str | None = None) -> Path:
    """Save the telemetry report as JSON under ``out_dir``."""
    telemetry_dir = Path(out_dir)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{trace_id}" if trace_id else ""
    report_path = telemetry_dir / f"telemetry{suffix}.json"
    report_path.write_text(
        json.dumps(_collector.get_report(), ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    print(f"[telemetry] saved: {report_path}")
    return report_path
