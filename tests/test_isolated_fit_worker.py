"""Tests for spawn-isolated fit execution and resource observations."""

from __future__ import annotations

import time

import pytest

from bh_augmentation.evaluation.isolated_fit_worker import (
    IsolatedFitTimeoutError,
    IsolatedFitWorkerError,
    normalize_ru_maxrss_bytes,
    run_isolated_fit,
)


def _arithmetic_task(left: int, right: int, *, delay_seconds: float) -> dict[str, int]:
    time.sleep(delay_seconds)
    return {"sum": left + right}


def _failing_task() -> None:
    raise RuntimeError("deliberate child failure")


def _unpickleable_result_task() -> object:
    return lambda: None


def _slow_task() -> None:
    time.sleep(5.0)


def test_ru_maxrss_normalization_is_correct_for_darwin_and_linux() -> None:
    assert normalize_ru_maxrss_bytes(1234, platform="darwin") == 1234
    assert normalize_ru_maxrss_bytes(1234, platform="linux") == 1234 * 1024
    assert normalize_ru_maxrss_bytes(1.5, platform="linux2") == 1536
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_ru_maxrss_bytes(1234, platform="win32")
    with pytest.raises(ValueError, match="finite and non-negative"):
        normalize_ru_maxrss_bytes(-1, platform="darwin")


def test_worker_returns_scientific_result_and_separate_resource_usage() -> None:
    execution = run_isolated_fit(
        _arithmetic_task,
        args=(2, 5),
        kwargs={"delay_seconds": 0.03},
        timeout_seconds=10.0,
    )

    assert execution.scientific_result == {"sum": 7}
    usage = execution.resource_usage
    assert usage.elapsed_seconds >= 0.02
    assert usage.rss_peak_bytes >= usage.rss_baseline_bytes >= 0
    assert usage.rss_increment_bytes == (
        usage.rss_peak_bytes - usage.rss_baseline_bytes
    )
    assert usage.timing_scope == "task_callable_only"
    assert usage.rss_scope == "isolated_child_process_ru_maxrss"
    assert set(usage.to_dict()) == {
        "elapsed_seconds",
        "rss_baseline_bytes",
        "rss_peak_bytes",
        "rss_increment_bytes",
        "timing_scope",
        "rss_scope",
    }


def test_worker_propagates_remote_exception_and_diagnostics() -> None:
    with pytest.raises(IsolatedFitWorkerError) as captured:
        run_isolated_fit(_failing_task, timeout_seconds=10.0)

    error = captured.value
    assert error.remote_exception_type == "builtins.RuntimeError"
    assert "deliberate child failure" in str(error)
    assert error.remote_traceback is not None
    assert "_failing_task" in error.remote_traceback
    assert error.resource_usage is not None
    assert error.resource_usage.rss_peak_bytes >= error.resource_usage.rss_baseline_bytes


def test_worker_reports_unpickleable_scientific_result_as_remote_error() -> None:
    with pytest.raises(IsolatedFitWorkerError, match="not pickleable") as captured:
        run_isolated_fit(_unpickleable_result_task, timeout_seconds=10.0)

    assert captured.value.remote_exception_type is not None
    assert captured.value.resource_usage is not None


def test_worker_times_out_and_terminates_child() -> None:
    started = time.perf_counter()
    with pytest.raises(IsolatedFitTimeoutError, match="exceeded timeout"):
        run_isolated_fit(_slow_task, timeout_seconds=0.1)
    assert time.perf_counter() - started < 3.0


def test_worker_rejects_non_spawn_pickleable_callable_before_start() -> None:
    with pytest.raises(TypeError, match="top-level callable"):
        run_isolated_fit(lambda: None)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan"), True])
def test_worker_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        run_isolated_fit(_failing_task, timeout_seconds=timeout)
