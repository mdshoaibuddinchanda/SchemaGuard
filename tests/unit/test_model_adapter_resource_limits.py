from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import Timeout

from schemaguard.models.adapters.base import _classify_runtime_exception
from schemaguard.models.adapters.contracts import FailureCategory
from schemaguard.models.adapters.resources import (
    AdapterResourceTracker,
    GpuExecutionLock,
    ensure_gpu_budget,
)
from scripts import validate_model_adapters as adapter_runner


class _FakeProcess:
    pid = 99999999

    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode


def test_resource_tracker_marks_hard_ram_limit_breach(monkeypatch) -> None:
    monkeypatch.setattr(
        "schemaguard.models.adapters.resources.process_tree_memory_mib", lambda: 256.0
    )
    tracker = AdapterResourceTracker("cpu", hard_ram_limit_mib=128.0)
    try:
        tracker.capture()
        assert tracker.limit_exceeded is True
        assert "RAM" in (tracker.limit_failure_reason or "")
    finally:
        tracker.stop()


def test_missing_ram_telemetry_is_incomplete_not_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        "schemaguard.models.adapters.resources.process_tree_memory_mib", lambda: None
    )
    tracker = AdapterResourceTracker("cpu")
    try:
        # Per-process RSS is still useful, but process-tree telemetry is mandatory
        # for a passing resource record and must remain explicitly unavailable.
        assert tracker.peak_tree_ram_mib is None
        assert tracker.peak_ram_mib is not None
        assert tracker.telemetry_complete is False
    finally:
        tracker.stop()


def test_missing_gpu_telemetry_is_incomplete_not_zero(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(
        "schemaguard.models.adapters.resources.gpu_memory_state",
        lambda: {"free_mib": None, "total_mib": None},
    )
    tracker = AdapterResourceTracker("cuda")
    try:
        assert tracker.free_vram_before_mib is None
        assert tracker.peak_vram_reserved_mib is None
        assert tracker.telemetry_complete is False
    finally:
        tracker.stop()


def test_vram_soft_limit_breach_blocks_resources(monkeypatch) -> None:
    mib = 1024**2
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        max_memory_allocated=lambda _device: 128 * mib,
        max_memory_reserved=lambda _device: 3700 * mib,
        synchronize=lambda _device: None,
        memory_allocated=lambda _device: 0,
        memory_reserved=lambda _device: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(
        "schemaguard.models.adapters.resources.gpu_memory_state",
        lambda: {"free_mib": 5000.0, "total_mib": 8192.0},
    )
    tracker = AdapterResourceTracker("cuda", gpu_soft_limit_mib=3600.0)
    try:
        tracker.capture()
        assert tracker.peak_vram_reserved_mib == 3700.0
        assert tracker.limit_exceeded is True
        assert tracker.resource_limits_passed is False
    finally:
        tracker.stop()


def test_insufficient_prelaunch_gpu_headroom_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "schemaguard.models.adapters.resources.gpu_memory_state",
        lambda: {"free_mib": 800.0, "total_mib": 4096.0},
    )
    with pytest.raises(RuntimeError, match="insufficient conservative free VRAM"):
        ensure_gpu_budget(headroom_mib=512.0, expected_peak_mib=376.0)


def test_cuda_cleanup_failure_is_not_reported_as_complete(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        max_memory_allocated=lambda _device: 1,
        max_memory_reserved=lambda _device: 1,
        synchronize=lambda _device: None,
        memory_allocated=lambda _device: 32 * 1024**2,
        memory_reserved=lambda _device: 128 * 1024**2,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(
        "schemaguard.models.adapters.resources.gpu_memory_state",
        lambda: {"free_mib": 3000.0, "total_mib": 4096.0},
    )
    tracker = AdapterResourceTracker("cuda")
    try:
        tracker.verify_cleanup()
        assert tracker.cleanup_verified is False
        assert tracker.telemetry_complete is False
    finally:
        tracker.stop()


def test_gpu_lock_is_released_even_after_an_exception(tmp_path) -> None:
    first = GpuExecutionLock(tmp_path, timeout=0.1)
    first.acquire()
    try:
        with pytest.raises(Timeout):
            GpuExecutionLock(tmp_path, timeout=0.05).acquire()
    finally:
        first.release()
    second = GpuExecutionLock(tmp_path, timeout=0.1)
    second.acquire()
    second.release()


def test_cuda_out_of_memory_has_an_explicit_failure_category() -> None:
    assert (
        _classify_runtime_exception(RuntimeError("CUDA out of memory"), "cuda")
        == FailureCategory.FAIL_CUDA_OOM
    )


def test_worker_timeout_terminates_the_worker(monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(adapter_runner, "_worker_memory_mib", lambda _pid: 100.0)
    monkeypatch.setattr(adapter_runner.time, "monotonic", lambda: 10.0)
    failure = adapter_runner._monitor_worker(process, timeout=0.0)
    assert failure is not None
    assert failure["failure_category"] == FailureCategory.FAIL_TIMEOUT.value
    assert process.poll() is not None


def test_worker_ram_limit_breach_terminates_the_worker(monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(adapter_runner, "_worker_memory_mib", lambda _pid: 28672.1)
    failure = adapter_runner._monitor_worker(process, timeout=60.0)
    assert failure is not None
    assert failure["failure_category"] == FailureCategory.FAIL_RESOURCE_LIMIT.value
    assert process.poll() is not None


def test_worker_ram_telemetry_failure_terminates_the_worker(monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(adapter_runner, "_worker_memory_mib", lambda _pid: None)
    failure = adapter_runner._monitor_worker(process, timeout=60.0)
    assert failure is not None
    assert failure["failure_category"] == FailureCategory.FAIL_RESOURCE_LIMIT.value
    assert process.poll() is not None


def test_crashed_worker_without_result_is_reported_as_failure(monkeypatch) -> None:
    process = _FakeProcess(returncode=17)
    monkeypatch.setattr(adapter_runner.subprocess, "Popen", lambda *args, **kwargs: process)
    with tempfile.TemporaryDirectory(dir=adapter_runner.ROOT) as directory:
        outcome = adapter_runner._worker_subprocess(
            "TPFN3-8.5",
            "binary_numerical",
            "cpu",
            "normal",
            0,
            Path(directory),
        )
    assert outcome["status"] == "FAIL"
    assert outcome["failure_category"] == FailureCategory.FAIL_MODEL_RUNTIME.value
    assert "without a valid result" in outcome["failure_reason"]
