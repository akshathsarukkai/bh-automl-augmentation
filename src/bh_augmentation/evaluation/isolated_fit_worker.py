"""Spawn-isolated execution and observational fit-resource measurements.

The returned timing and memory measurements describe execution conditions.
They are intentionally kept beside, rather than inside, the scientific result
so callers do not accidentally include nondeterministic resource observations
in scientific hashes.
"""

from __future__ import annotations

import math
import multiprocessing
import pickle
import resource
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any


@dataclass(frozen=True, slots=True)
class IsolatedResourceUsage:
    """Observational resource measurements from one fresh child process."""

    elapsed_seconds: float
    rss_baseline_bytes: int
    rss_peak_bytes: int
    rss_increment_bytes: int
    timing_scope: str = "task_callable_only"
    rss_scope: str = "isolated_child_process_ru_maxrss"

    def __post_init__(self) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        if self.rss_baseline_bytes < 0:
            raise ValueError("rss_baseline_bytes must be non-negative.")
        if self.rss_peak_bytes < self.rss_baseline_bytes:
            raise ValueError("rss_peak_bytes cannot be below the RSS baseline.")
        if self.rss_increment_bytes != (
            self.rss_peak_bytes - self.rss_baseline_bytes
        ):
            raise ValueError("rss_increment_bytes must equal peak minus baseline.")

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a manifest-ready observational record."""
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "rss_baseline_bytes": self.rss_baseline_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "rss_increment_bytes": self.rss_increment_bytes,
            "timing_scope": self.timing_scope,
            "rss_scope": self.rss_scope,
        }


@dataclass(frozen=True, slots=True)
class IsolatedFitExecution:
    """A deterministic scientific result plus nondeterministic observations."""

    scientific_result: Any
    resource_usage: IsolatedResourceUsage


class IsolatedFitWorkerError(RuntimeError):
    """An exception raised by, or while communicating with, the child worker."""

    def __init__(
        self,
        message: str,
        *,
        remote_exception_type: str | None = None,
        remote_traceback: str | None = None,
        resource_usage: IsolatedResourceUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.remote_exception_type = remote_exception_type
        self.remote_traceback = remote_traceback
        self.resource_usage = resource_usage


class IsolatedFitTimeoutError(TimeoutError):
    """The isolated task exceeded its declared wall-clock timeout."""


def normalize_ru_maxrss_bytes(
    ru_maxrss: int | float,
    *,
    platform: str | None = None,
) -> int:
    """Normalize ``resource.ru_maxrss`` to bytes on Darwin and Linux.

    Darwin reports bytes while Linux reports KiB. Unsupported platforms are
    rejected instead of silently assigning an incorrect unit.
    """
    value = float(ru_maxrss)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("ru_maxrss must be finite and non-negative.")
    resolved_platform = sys.platform if platform is None else platform
    if resolved_platform == "darwin":
        multiplier = 1
    elif resolved_platform.startswith("linux"):
        multiplier = 1024
    else:
        raise ValueError(
            f"Unsupported ru_maxrss unit convention on platform {resolved_platform!r}."
        )
    return int(round(value * multiplier))


def run_isolated_fit(
    task: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    timeout_seconds: float = 600.0,
) -> IsolatedFitExecution:
    """Execute a spawn-pickleable top-level task in a fresh child process.

    ``task`` may be the low-complexity runner's top-level ``fit_predict``
    callable, but the helper remains generic so it can be tested and reused
    without importing a runner under construction.
    """
    if not callable(task):
        raise TypeError("task must be callable.")
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("timeout_seconds must be finite and positive.")
    resolved_args = tuple(args)
    resolved_kwargs = dict(kwargs or {})
    try:
        pickle.dumps((task, resolved_args, resolved_kwargs))
    except Exception as exc:
        raise TypeError(
            "The isolated task, args, and kwargs must be spawn-pickleable; "
            "use a top-level callable."
        ) from exc

    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_child_entrypoint,
        args=(child_connection, task, resolved_args, resolved_kwargs),
        name="bh-isolated-fit-worker",
    )
    try:
        process.start()
    except Exception as exc:
        parent_connection.close()
        child_connection.close()
        raise IsolatedFitWorkerError("Failed to start isolated fit worker.") from exc
    child_connection.close()

    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        parent_connection.close()
        raise IsolatedFitTimeoutError(
            f"Isolated fit worker exceeded timeout of {float(timeout_seconds):g} seconds."
        )

    exit_code = process.exitcode
    if not parent_connection.poll(1.0):
        parent_connection.close()
        raise IsolatedFitWorkerError(
            "Isolated fit worker exited without returning a result "
            f"(exit_code={exit_code})."
        )
    try:
        payload = pickle.loads(parent_connection.recv_bytes())
    except (EOFError, OSError, pickle.PickleError) as exc:
        raise IsolatedFitWorkerError(
            "Failed to decode isolated fit worker result "
            f"(exit_code={exit_code})."
        ) from exc
    finally:
        parent_connection.close()

    if not isinstance(payload, dict) or payload.get("status") not in {"ok", "error"}:
        raise IsolatedFitWorkerError("Isolated fit worker returned an invalid payload.")
    usage = _usage_from_payload(payload)
    if payload["status"] == "error":
        exception_type = str(payload.get("exception_type", "unknown"))
        message = str(payload.get("message", ""))
        raise IsolatedFitWorkerError(
            f"Isolated fit task raised {exception_type}: {message}",
            remote_exception_type=exception_type,
            remote_traceback=str(payload.get("traceback", "")),
            resource_usage=usage,
        )
    if exit_code != 0:
        raise IsolatedFitWorkerError(
            f"Isolated fit worker returned a result but exited with code {exit_code}."
        )
    return IsolatedFitExecution(
        scientific_result=payload["scientific_result"],
        resource_usage=usage,
    )


def _isolated_child_entrypoint(
    connection: Connection,
    task: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Run one task and send a fully pickled success or error payload."""
    baseline = _current_ru_maxrss_bytes()
    started = time.perf_counter()
    try:
        result = task(*args, **kwargs)
        elapsed = time.perf_counter() - started
        peak = max(baseline, _current_ru_maxrss_bytes())
        payload: dict[str, Any] = {
            "status": "ok",
            "scientific_result": result,
            **_resource_payload(elapsed, baseline, peak),
        }
        try:
            encoded = pickle.dumps(payload)
        except Exception as exc:
            payload = {
                "status": "error",
                "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": "Scientific result is not pickleable.",
                "traceback": traceback.format_exc(),
                **_resource_payload(elapsed, baseline, peak),
            }
            encoded = pickle.dumps(payload)
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        peak = max(baseline, _current_ru_maxrss_bytes())
        payload = {
            "status": "error",
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
            "traceback": traceback.format_exc(),
            **_resource_payload(elapsed, baseline, peak),
        }
        encoded = pickle.dumps(payload)
    try:
        connection.send_bytes(encoded)
    finally:
        connection.close()


def _current_ru_maxrss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return normalize_ru_maxrss_bytes(usage.ru_maxrss)


def _resource_payload(elapsed: float, baseline: int, peak: int) -> dict[str, Any]:
    return {
        "elapsed_seconds": float(elapsed),
        "rss_baseline_bytes": int(baseline),
        "rss_peak_bytes": int(peak),
        "rss_increment_bytes": int(peak - baseline),
        "timing_scope": "task_callable_only",
        "rss_scope": "isolated_child_process_ru_maxrss",
    }


def _usage_from_payload(payload: Mapping[str, Any]) -> IsolatedResourceUsage:
    try:
        return IsolatedResourceUsage(
            elapsed_seconds=float(payload["elapsed_seconds"]),
            rss_baseline_bytes=int(payload["rss_baseline_bytes"]),
            rss_peak_bytes=int(payload["rss_peak_bytes"]),
            rss_increment_bytes=int(payload["rss_increment_bytes"]),
            timing_scope=str(payload["timing_scope"]),
            rss_scope=str(payload["rss_scope"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IsolatedFitWorkerError(
            "Isolated fit worker returned invalid resource measurements."
        ) from exc
