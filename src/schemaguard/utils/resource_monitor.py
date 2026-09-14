"""Best-effort process and GPU resource measurement."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ..compatibility.contracts import ResourceRecord


@dataclass
class ResourceMonitor:
    model_id: str
    device: str
    started: float = 0.0
    cpu_started: float = 0.0
    peak_ram_mib: float | None = None
    gpu_total_before: float | None = None
    gpu_free_before: float | None = None
    gpu_allocated: float | None = None
    gpu_reserved: float | None = None
    monitoring_error: str | None = None

    def __enter__(self) -> ResourceMonitor:
        self.started = time.perf_counter()
        try:
            import psutil  # type: ignore[import-not-found]

            process = psutil.Process(os.getpid())
            self.cpu_started = float(process.cpu_times().user + process.cpu_times().system)
            self.peak_ram_mib = float(process.memory_info().rss / (1024**2))
        except Exception as exc:
            self.monitoring_error = f"RAM telemetry unavailable: {type(exc).__name__}: {exc}"
        if self.device == "cuda":
            self._capture_gpu_before()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Leave finalization to the probe so failed paths retain their category."""
        return None

    def _capture_gpu_before(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                self.gpu_free_before = float(free / (1024**2))
                self.gpu_total_before = float(total / (1024**2))
        except Exception as exc:
            self.monitoring_error = f"GPU telemetry unavailable: {type(exc).__name__}: {exc}"

    def finish(
        self, *, timed_out: bool = False, termination_signal: str | None = None, oom: bool = False
    ) -> ResourceRecord:
        runtime = max(0.0, time.perf_counter() - self.started)
        cpu_time = 0.0
        try:
            import psutil  # type: ignore[import-not-found]

            process = psutil.Process(os.getpid())
            times = process.cpu_times()
            cpu_time = max(0.0, float(times.user + times.system - self.cpu_started))
            self.peak_ram_mib = max(
                float(process.memory_info().rss / (1024**2)), self.peak_ram_mib or 0.0
            )
        except Exception as exc:
            if self.monitoring_error is None:
                self.monitoring_error = f"RAM telemetry unavailable: {type(exc).__name__}: {exc}"
        if self.device == "cuda":
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    self.gpu_allocated = float(torch.cuda.max_memory_allocated(0) / (1024**2))
                    self.gpu_reserved = float(torch.cuda.max_memory_reserved(0) / (1024**2))
            except Exception as exc:
                if self.monitoring_error is None:
                    self.monitoring_error = (
                        f"GPU telemetry unavailable: {type(exc).__name__}: {exc}"
                    )
        return ResourceRecord(
            model_id=self.model_id,
            device=self.device,  # type: ignore[arg-type]
            runtime_seconds=runtime,
            cpu_time_seconds=cpu_time,
            peak_ram_mib=self.peak_ram_mib,
            child_peak_ram_mib=None,
            gpu_allocated_mib=self.gpu_allocated,
            gpu_reserved_mib=self.gpu_reserved,
            gpu_total_mib_before=self.gpu_total_before,
            gpu_free_mib_before=self.gpu_free_before,
            gpu_total_mib_after=None,
            gpu_free_mib_after=None,
            timed_out=timed_out,
            termination_signal=termination_signal,
            oom=oom,
            monitoring_error=self.monitoring_error,
        )
