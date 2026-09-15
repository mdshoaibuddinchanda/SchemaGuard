import sys
from types import SimpleNamespace

from schemaguard.utils.resource_monitor import ResourceMonitor


def test_resource_runtime_and_ram_are_nonnegative() -> None:
    with ResourceMonitor("LR-1.9", "cpu") as monitor:
        result = monitor.finish()
    assert result.runtime_seconds >= 0
    assert result.cpu_time_seconds >= 0
    assert result.peak_ram_mib is None or result.peak_ram_mib >= 0


def test_gpu_missing_telemetry_is_null_not_zero(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with ResourceMonitor("TPFN3-8.5", "cuda") as monitor:
        result = monitor.finish()
    assert result.gpu_allocated_mib is None
    assert result.gpu_reserved_mib is None


def test_oom_and_timeout_are_explicit() -> None:
    with ResourceMonitor("TPFN3-8.5", "cuda") as monitor:
        result = monitor.finish(timed_out=True, termination_signal="SIGTERM", oom=True)
    assert result.timed_out is True
    assert result.oom is True
