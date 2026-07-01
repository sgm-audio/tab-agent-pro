"""
Tab Agent — Production Monitoring & Health Checks

Structured JSON logging, pipeline metrics, and health endpoint support.
All output goes to stdout (Docker/HF Spaces compatible). No external services required.

Usage:
    from monitoring import get_logger, HealthTracker, PipelineMetrics

    log = get_logger(__name__)
    log.info("transcription_start", file="song.wav", instrument="Guitar")

    health = HealthTracker()
    health.set_status("ready")

    metrics = PipelineMetrics()
    with metrics.track_stage("demucs"):
        ...

    # Then expose via Gradio /health route
    app.get("/health")(lambda: health.as_dict())
"""

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

# ============================================================================
# Structured Logger
# ============================================================================


class PipelineLogger:
    """Structured JSON logger for pipeline events."""

    def __init__(self, module: str):
        self.module = module
        self._start_time = time.time()

    def _emit(self, level: str, event: str, **kwargs):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "module": self.module,
            "event": event,
            "uptime_s": round(time.time() - self._start_time, 3),
            **kwargs,
        }
        print(json.dumps(record, default=str), flush=True)

    def info(self, event: str, **kwargs):
        self._emit("INFO", event, **kwargs)

    def warn(self, event: str, **kwargs):
        self._emit("WARN", event, **kwargs)

    def warning(self, event: str, **kwargs):
        self._emit("WARN", event, **kwargs)

    def error(self, event: str, exc: Exception | None = None, **kwargs):
        data = kwargs
        if exc:
            data["error_type"] = type(exc).__name__
            data["error_msg"] = str(exc)
        self._emit("ERROR", event, **data)

    def metric(self, name: str, value: float, unit: str = "", **kwargs):
        self._emit("METRIC", name, value=value, unit=unit, **kwargs)


def get_logger(module: str = "TabAgent") -> PipelineLogger:
    return PipelineLogger(module)


# ============================================================================
# Health Tracker
# ============================================================================


class HealthTracker:
    """
    Tracks service health for /health endpoint.
    Thread-safe, no external dependencies.
    """

    STATUS_UP = "up"
    STATUS_DOWN = "down"
    STATUS_STARTING = "starting"
    STATUS_DEGRADED = "degraded"

    def __init__(self):
        self._status = self.STATUS_STARTING
        self._last_error: str | None = None
        self._last_success: str | None = None
        self._total_requests = 0
        self._total_errors = 0
        self._component_status: dict[str, str] = {
            "demucs": "unknown",
            "basic_pitch": "unknown",
            "yourmt3": "unknown",
            "suno_detector": "unknown",
        }
        self._start_time = time.time()

    def set_status(self, status: str):
        self._status = status

    def set_component(self, name: str, status: str):
        self._component_status[name] = status

    def record_request(self, success: bool, details: str = ""):
        self._total_requests += 1
        if success:
            self._last_success = details
        else:
            self._total_errors += 1
            self._last_error = details

    def as_dict(self) -> dict[str, Any]:
        error_rate = (
            float(self._total_errors) / self._total_requests if self._total_requests > 0 else 0.0
        )
        return {
            "status": self._status,
            "uptime_s": round(time.time() - self._start_time, 1),
            "components": dict(self._component_status),
            "requests": {
                "total": self._total_requests,
                "errors": self._total_errors,
                "error_rate": round(error_rate, 4),
            },
            "last_error": self._last_error,
            "last_success": self._last_success,
            "python_version": sys.version,
            "platform": sys.platform,
        }


# Global health instance
health = HealthTracker()
health.set_status(HealthTracker.STATUS_UP)


# ============================================================================
# Pipeline Metrics
# ============================================================================


class PipelineMetrics:
    """Tracks per-stage timing and success/failure counts."""

    def __init__(self, logger: PipelineLogger | None = None):
        self.log = logger or get_logger("metrics")
        self.stage_times: dict[str, list] = {}
        self.stage_counts: dict[str, int] = {}
        self.stage_errors: dict[str, int] = {}

    def track_stage(self, stage_name: str):
        """Context manager that times a pipeline stage and logs metrics."""
        return _StageTracker(self, stage_name)

    def record_stage(self, name: str, elapsed: float, success: bool):
        self.stage_times.setdefault(name, []).append(elapsed)
        self.stage_counts[name] = self.stage_counts.get(name, 0) + 1
        if not success:
            self.stage_errors[name] = self.stage_errors.get(name, 0) + 1

        error_rate = (
            float(self.stage_errors.get(name, 0)) / self.stage_counts[name]
            if self.stage_counts[name] > 0
            else 0.0
        )

        self.log.metric(
            f"stage.{name}.elapsed_s",
            round(elapsed, 3),
            unit="s",
            success=success,
            error_rate=round(error_rate, 4),
        )

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "count": self.stage_counts.get(name, 0),
                "total_s": round(sum(times), 3),
                "avg_s": round(sum(times) / len(times), 3) if times else 0,
                "error_rate": (
                    round(self.stage_errors.get(name, 0) / self.stage_counts[name], 4)
                    if self.stage_counts.get(name, 0) > 0
                    else 0
                ),
            }
            for name, times in self.stage_times.items()
        }


class _StageTracker:
    def __init__(self, parent: PipelineMetrics, name: str):
        self.parent = parent
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        success = exc_type is None
        self.parent.record_stage(self.name, elapsed, success)
        # Don't suppress exceptions
        return False


# ============================================================================
# Default instances
# ============================================================================

default_metrics = PipelineMetrics()
default_log = get_logger("TabAgent")
