"""
Tab Agent — Production Monitoring & Health Checks.

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
import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

# ============================================================================
# Structured Logger
# ============================================================================


class PipelineLogger:
    """Structured JSON logger for pipeline events."""

    def __init__(self, module: str) -> None:
        self.module = module
        self._start_time = time.time()

    def _emit(self, level: str, event: str, **kwargs) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "module": self.module,
            "event": event,
            "uptime_s": round(time.time() - self._start_time, 3),
            **kwargs,
        }
        print(json.dumps(record, default=str), flush=True)

    def info(self, event: str, **kwargs) -> None:
        self._emit("INFO", event, **kwargs)

    def warn(self, event: str, **kwargs) -> None:
        self._emit("WARN", event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        self._emit("WARN", event, **kwargs)

    def error(self, event: str, exc: Exception | None = None, **kwargs) -> None:
        data = kwargs
        if exc:
            data["error_type"] = type(exc).__name__
            data["error_msg"] = str(exc)
        self._emit("ERROR", event, **data)

    def metric(self, name: str, value: float, unit: str = "", **kwargs) -> None:
        self._emit("METRIC", name, value=value, unit=unit, **kwargs)


def get_logger(module: str = "TabAgent") -> PipelineLogger:
    return PipelineLogger(module)


# ============================================================================
# Health Tracker
# ============================================================================


class HealthTracker:
    """
    Tracks service health for the /health and /ready endpoints.

    Thread-safe (single re-entrant lock guarding all mutable state) and free of
    external dependencies.

    Status is *derived* from component health rather than set by hand: a
    deployment whose transcription backends all failed must not report "up",
    otherwise a container health probe cannot tell a working image from one
    that cannot transcribe anything (docs-audit P0-2).
    """

    STATUS_UP = "up"
    STATUS_DOWN = "down"
    STATUS_STARTING = "starting"
    STATUS_DEGRADED = "degraded"

    # A component is hard-required when requests cannot succeed without it.
    # Basic Pitch is the always-available transcription backend, so without it
    # the instance cannot do its job at all.
    #
    # Demucs is *not* in this list on purpose: stem separation has a documented
    # passthrough fallback (`Splitter._raw_audio_fallback`), so an instance with
    # Basic Pitch but no Demucs still transcribes whole-mix audio. Marking it
    # required made a working deployment pull itself out of rotation (readiness
    # 503 while it happily served requests) — the opposite of the docs-audit
    # P0-2 goal, which was to catch *dead* instances, not degraded ones.
    REQUIRED_FOR_TRANSCRIPTION = ("basic_pitch",)
    # Backends whose absence degrades quality but not availability.
    DEGRADABLE_COMPONENTS = ("demucs",)
    OPTIONAL_COMPONENTS = ("suno_detector",)

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._status = self.STATUS_STARTING
        self._last_error: str | None = None
        self._last_success: str | None = None
        self._total_requests = 0
        self._total_errors = 0
        self._component_status: dict[str, str] = {
            "demucs": "unknown",
            "basic_pitch": "unknown",
            "suno_detector": "unknown",
        }
        self._start_time = time.time()

    def set_status(self, status: str) -> None:
        """Force an explicit status (also clears a derived 'down')."""
        with self._lock:
            self._status = status

    def set_component(self, name: str, status: str) -> None:
        with self._lock:
            self._component_status[name] = status

    def record_request(self, success: bool, details: str = "") -> None:
        with self._lock:
            self._total_requests += 1
            if success:
                self._last_success = details
            else:
                self._total_errors += 1
                self._last_error = details

    @staticmethod
    def _component_ok(status: str) -> bool:
        """A component is usable when it imported or loaded (degraded counts)."""
        if not status or status == "unknown":
            return False
        return status.startswith(("loaded", "available"))

    def component_health(self) -> dict[str, bool]:
        """Per-required-component availability, as booleans."""
        with self._lock:
            return {
                name: self._component_ok(self._component_status.get(name, "unknown"))
                for name in self.REQUIRED_FOR_TRANSCRIPTION
            }

    def degraded_components(self) -> list[str]:
        """Quality-degrading components that are currently unusable."""
        with self._lock:
            return [
                name
                for name in self.DEGRADABLE_COMPONENTS
                if not self._component_ok(self._component_status.get(name, "unknown"))
            ]

    def is_ready(self) -> bool:
        """
        Readiness: every required backend is loaded.

        Used by /ready and by the container healthcheck. Independent of
        request-level error rate so a single bad upload does not take an
        otherwise healthy instance out of rotation.
        """
        return all(self.component_health().values())

    def effective_status(self) -> str:
        """
        Derive the reported status from component health and error rate.

        * an explicit ``down``/``starting`` set by a caller is honoured;
        * otherwise the required components decide: none usable -> ``down``,
          some usable -> ``degraded``, all usable -> ``up``;
        * if no component has registered yet (bare ``monitoring`` import) the
          baseline is kept, because "unknown" is not the same as "broken".
          ``ready`` still reports False in that case.
        """
        with self._lock:
            explicit = self._status
            components = dict(self._component_status)
            requests, errors = self._total_requests, self._total_errors

        if explicit in (self.STATUS_STARTING, self.STATUS_DOWN):
            return explicit

        required = list(self.REQUIRED_FOR_TRANSCRIPTION)
        required_ok = [self._component_ok(components.get(c, "unknown")) for c in required]
        required_known = [
            components.get(c, "unknown") not in (None, "", "unknown") for c in required
        ]

        if not any(required_known):
            return explicit  # nothing registered yet: keep the baseline
        if not any(required_ok):
            # No usable backend: this instance cannot serve its one job.
            return self.STATUS_DOWN
        if not all(required_ok):
            return self.STATUS_DEGRADED
        if errors and requests and (errors / requests) >= 0.5:
            return self.STATUS_DEGRADED
        # Required backends are fine; a quality-only backend being gone makes
        # the instance degraded (serving, with reduced output) but not unready.
        degraded = [
            name
            for name in self.DEGRADABLE_COMPONENTS
            if not self._component_ok(components.get(name, "unknown"))
        ]
        if degraded:
            return self.STATUS_DEGRADED
        return self.STATUS_UP

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            total_requests, total_errors = self._total_requests, self._total_errors
            components = dict(self._component_status)
            last_error, last_success = self._last_error, self._last_success
            start_time = self._start_time

        error_rate = float(total_errors) / total_requests if total_requests > 0 else 0.0
        return {
            "status": self.effective_status(),
            "ready": self.is_ready(),
            "uptime_s": round(time.time() - start_time, 1),
            "components": components,
            "degraded_components": self.degraded_components(),
            "requests": {
                "total": total_requests,
                "errors": total_errors,
                "error_rate": round(error_rate, 4),
            },
            "last_error": last_error,
            "last_success": last_success,
            "python_version": sys.version,
            "platform": sys.platform,
        }


# Global health instance.
# Status is derived from component health (see effective_status): it is never
# hard-coded to "up", so a container with no working backend reports "down"
# instead of passing its healthcheck.
health = HealthTracker()


# ============================================================================
# Pipeline Metrics
# ============================================================================


class PipelineMetrics:
    """
    Tracks per-stage timing and success/failure counts.

    Only a bounded window of recent samples per stage is retained (``window``,
    default 500) so a long-lived server cannot grow without bound; totals are
    accumulated separately, so the all-time count and mean stay exact.
    Thread-safe: the Gradio app records stages from request worker threads.
    """

    def __init__(self, logger: PipelineLogger | None = None, window: int = 500) -> None:
        self.log = logger or get_logger("metrics")
        self.window = max(1, int(window))
        self._lock = threading.RLock()
        # Recent samples only (bounded); _total_s / _counts stay exact forever.
        self.stage_times: dict[str, deque] = {}
        self.stage_total_s: dict[str, float] = {}
        self.stage_counts: dict[str, int] = {}
        self.stage_errors: dict[str, int] = {}

    def track_stage(self, stage_name: str):
        """Context manager that times a pipeline stage and logs metrics."""
        return _StageTracker(self, stage_name)

    def record_stage(self, name: str, elapsed: float, success: bool) -> None:
        with self._lock:
            samples = self.stage_times.setdefault(name, deque(maxlen=self.window))
            samples.append(elapsed)
            self.stage_total_s[name] = self.stage_total_s.get(name, 0.0) + elapsed
            self.stage_counts[name] = self.stage_counts.get(name, 0) + 1
            if not success:
                self.stage_errors[name] = self.stage_errors.get(name, 0) + 1

            count = self.stage_counts[name]
            errors = self.stage_errors.get(name, 0)

        error_rate = float(errors) / count if count > 0 else 0.0

        self.log.metric(
            f"stage.{name}.elapsed_s",
            round(elapsed, 3),
            unit="s",
            success=success,
            error_rate=round(error_rate, 4),
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            names = set(self.stage_times) | set(self.stage_counts)
            out: dict[str, Any] = {}
            for name in names:
                count = self.stage_counts.get(name, 0)
                total = self.stage_total_s.get(name, 0.0)
                samples = list(self.stage_times.get(name, ()))
                errors = self.stage_errors.get(name, 0)
                out[name] = {
                    "count": count,
                    "total_s": round(total, 3),
                    "avg_s": round(total / count, 3) if count else 0,
                    # Windowed statistics: bounded memory, still representative.
                    "window_count": len(samples),
                    "window_avg_s": round(sum(samples) / len(samples), 3) if samples else 0,
                    "error_rate": round(errors / count, 4) if count else 0,
                }
            return out


class _StageTracker:
    def __init__(self, parent: PipelineMetrics, name: str) -> None:
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
