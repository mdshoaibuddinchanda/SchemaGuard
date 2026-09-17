"""Thread, memory, GPU-safety, and existing GPU-lock policies for adapters."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...utils.process_lock import ProcessLock

_THREADPOOL_LIMITER: Any = None


def configure_thread_limits() -> None:
    """Set native and BLAS thread ceilings before numerical/model imports."""

    global _THREADPOOL_LIMITER
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "2"
    if _THREADPOOL_LIMITER is None:
        try:
            from threadpoolctl import threadpool_limits

            _THREADPOOL_LIMITER = threadpool_limits(limits=2)
        except ImportError:
            _THREADPOOL_LIMITER = False


def process_tree_memory_mib() -> float | None:
    """Return current parent-plus-child RSS when psutil telemetry is available."""

    try:
        import psutil

        parent = psutil.Process(os.getpid())
        processes = [parent, *parent.children(recursive=True)]
        total_bytes = 0
        observed = 0
        for process in processes:
            try:
                if process.is_running():
                    total_bytes += int(process.memory_info().rss)
                    observed += 1
            except psutil.NoSuchProcess:
                # A child may exit between enumeration and RSS sampling.
                continue
            except psutil.AccessDenied:
                return None
        return total_bytes / (1024**2) if observed else None
    except Exception:
        return None


def process_memory_mib() -> float | None:
    """Return current-process RSS when psutil telemetry is available."""

    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024**2)
    except Exception:
        return None


def ensure_ram_budget(*, hard_ram_gib: float, anticipated_additional_gib: float = 2.0) -> float:
    """Refuse a fit when measured process/system headroom cannot honor the hard cap."""

    try:
        import psutil

        process_mib = process_tree_memory_mib()
        if process_mib is None:
            raise RuntimeError("process-tree RAM telemetry is unavailable")
        process_gib = process_mib / 1024.0
        available_gib = float(psutil.virtual_memory().available / (1024**3))
        if process_gib + anticipated_additional_gib > hard_ram_gib:
            raise RuntimeError(
                f"process tree plus {anticipated_additional_gib:.1f} GiB estimate "
                "exceeds hard RAM cap"
            )
        if available_gib < anticipated_additional_gib:
            raise RuntimeError(
                f"system available RAM {available_gib:.2f} GiB is below the fit estimate"
            )
        return process_mib
    except ImportError as exc:
        raise RuntimeError("process-tree RAM telemetry is required for bounded fitting") from exc


def _nvidia_smi_state() -> tuple[float | None, float | None]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        total, free = output.splitlines()[0].split(",")
        return float(total.strip()), float(free.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


def gpu_memory_state() -> dict[str, float | str | None]:
    """Use the lower free-memory reading from PyTorch and nvidia-smi."""

    total_smi, free_smi = _nvidia_smi_state()
    total_torch: float | None = None
    free_torch: float | None = None
    name: str | None = None
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_torch = float(total_bytes / (1024**2))
            free_torch = float(free_bytes / (1024**2))
            name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    totals = [value for value in (total_smi, total_torch) if value is not None]
    frees = [value for value in (free_smi, free_torch) if value is not None]
    return {
        "name": name,
        "total_mib": min(totals) if totals else None,
        "free_mib": min(frees) if frees else None,
        "free_nvidia_smi_mib": free_smi,
        "free_torch_mib": free_torch,
    }


def ensure_gpu_budget(
    *, headroom_mib: float, expected_peak_mib: float
) -> dict[str, float | str | None]:
    """Refuse a launch unless conservative telemetry includes headroom + expected peak."""

    state = gpu_memory_state()
    free_mib = state["free_mib"]
    if free_mib is None:
        raise RuntimeError("BLOCKED_ENVIRONMENT: CUDA free-memory telemetry is unavailable")
    if float(free_mib) < headroom_mib + expected_peak_mib:
        raise RuntimeError(
            "BLOCKED_ENVIRONMENT: insufficient conservative free VRAM "
            f"({free_mib:.1f} MiB < {headroom_mib + expected_peak_mib:.1f} MiB required)"
        )
    return state


class GpuExecutionLock:
    """Hold the repository's accepted GPU lock for one foundation adapter lifecycle."""

    def __init__(self, root: str | Path, timeout: float = 30.0) -> None:
        self.path = Path(root) / "data" / "cache" / "locks" / "gpu-capacity.lock"
        self.lock = ProcessLock(self.path, timeout=timeout)
        self.acquired = False

    def acquire(self) -> None:
        if not self.acquired:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.lock.__enter__()
            self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.lock.__exit__(None, None, None)
            self.acquired = False


@dataclass
class AdapterResourceTracker:
    device: str
    hard_ram_limit_mib: float | None = None
    gpu_soft_limit_mib: float | None = None
    started: float = field(default_factory=time.perf_counter)
    cpu_started: float | None = None
    peak_ram_mib: float | None = None
    peak_tree_ram_mib: float | None = None
    free_vram_before_mib: float | None = None
    free_vram_after_mib: float | None = None
    peak_vram_allocated_mib: float | None = None
    peak_vram_reserved_mib: float | None = None
    gpu_baseline_allocated_mib: float | None = None
    gpu_baseline_reserved_mib: float | None = None
    telemetry_error: str | None = None
    limit_exceeded: bool = False
    limit_failure_reason: str | None = None
    cleanup_verified: bool = False
    cleanup_gpu_allocated_mib: float | None = None
    cleanup_gpu_reserved_mib: float | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _monitor_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import psutil

            process_times = psutil.Process(os.getpid()).cpu_times()
            self.cpu_started = float(process_times.user + process_times.system)
        except Exception as exc:
            self.telemetry_error = f"CPU/RAM telemetry unavailable: {type(exc).__name__}"
        self.capture()
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    self.gpu_baseline_allocated_mib = float(
                        torch.cuda.memory_allocated(0) / (1024**2)
                    )
                    self.gpu_baseline_reserved_mib = float(
                        torch.cuda.memory_reserved(0) / (1024**2)
                    )
                free = gpu_memory_state()["free_mib"]
                self.free_vram_before_mib = float(free) if free is not None else None
            except Exception as exc:
                self._note_error(exc)
        self._monitor_thread = threading.Thread(target=self._monitor_ram, daemon=True)
        self._monitor_thread.start()

    def _monitor_ram(self) -> None:
        while not self._stop_event.wait(0.5):
            self._capture_ram()

    def _capture_ram(self) -> None:
        tree_memory = process_tree_memory_mib()
        memory = process_memory_mib()
        if tree_memory is not None:
            self.peak_tree_ram_mib = max(tree_memory, self.peak_tree_ram_mib or 0.0)
            if self.hard_ram_limit_mib is not None and tree_memory > self.hard_ram_limit_mib:
                self.limit_exceeded = True
                self.limit_failure_reason = (
                    f"process-tree RAM {tree_memory:.1f} MiB exceeded hard limit "
                    f"{self.hard_ram_limit_mib:.1f} MiB"
                )
        else:
            # A transient child-process race must not make a later complete
            # telemetry sample permanently fail the probe.
            pass
        if memory is not None:
            self.peak_ram_mib = max(memory, self.peak_ram_mib or 0.0)
        else:
            pass

    def capture(self) -> None:
        self._capture_ram()
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = float(torch.cuda.max_memory_allocated(0) / (1024**2))
                    reserved = float(torch.cuda.max_memory_reserved(0) / (1024**2))
                    self.peak_vram_allocated_mib = max(
                        allocated, self.peak_vram_allocated_mib or 0.0
                    )
                    self.peak_vram_reserved_mib = max(reserved, self.peak_vram_reserved_mib or 0.0)
                    if self.gpu_soft_limit_mib is not None and reserved > self.gpu_soft_limit_mib:
                        self.limit_exceeded = True
                        self.limit_failure_reason = (
                            f"reserved VRAM {reserved:.1f} MiB exceeded soft limit "
                            f"{self.gpu_soft_limit_mib:.1f} MiB"
                        )
                    free = gpu_memory_state()["free_mib"]
                    self.free_vram_after_mib = float(free) if free is not None else None
                    if free is None:
                        self._note_error(RuntimeError("CUDA free-memory telemetry is unavailable"))
            except Exception as exc:
                self._note_error(exc)

    def cpu_time_seconds(self) -> float | None:
        if self.cpu_started is None:
            return None
        try:
            import psutil

            process = psutil.Process(os.getpid())
            current = process.cpu_times()
            return max(0.0, float(current.user + current.system - self.cpu_started))
        except Exception as exc:
            self._note_error(exc)
            return None

    def wall_time_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.started)

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2)

    def verify_cleanup(self) -> None:
        """Record cleanup state without inventing missing telemetry."""

        if self.device == "cpu":
            self.cleanup_verified = True
            return
        try:
            import torch

            if not torch.cuda.is_available():
                self._note_error(RuntimeError("CUDA disappeared before cleanup verification"))
                self.cleanup_verified = False
                return
            torch.cuda.synchronize(0)
            self.cleanup_gpu_allocated_mib = float(torch.cuda.memory_allocated(0) / (1024**2))
            self.cleanup_gpu_reserved_mib = float(torch.cuda.memory_reserved(0) / (1024**2))
            allowed_allocated = max(64.0, (self.gpu_baseline_allocated_mib or 0.0) + 16.0)
            allowed_reserved = max(128.0, (self.gpu_baseline_reserved_mib or 0.0) + 64.0)
            self.cleanup_verified = (
                self.cleanup_gpu_allocated_mib <= allowed_allocated
                and self.cleanup_gpu_reserved_mib <= allowed_reserved
            )
            if not self.cleanup_verified:
                self._note_error(RuntimeError("CUDA allocations remain after adapter cleanup"))
        except Exception as exc:
            self._note_error(exc)
            self.cleanup_verified = False

    @property
    def execution_telemetry_complete(self) -> bool:
        """Whether measurements needed during model execution are available."""

        return (
            self.telemetry_error is None
            and self.cpu_started is not None
            and self.peak_ram_mib is not None
            and self.peak_tree_ram_mib is not None
            and (
                self.device == "cpu"
                or (
                    self.peak_vram_reserved_mib is not None
                    and self.gpu_baseline_allocated_mib is not None
                    and self.gpu_baseline_reserved_mib is not None
                    and self.free_vram_before_mib is not None
                    and self.free_vram_after_mib is not None
                )
            )
        )

    @property
    def telemetry_complete(self) -> bool:
        """Whether execution telemetry and post-release cleanup telemetry are complete."""

        return self.execution_telemetry_complete and (
            self.device == "cpu"
            or (
                self.cleanup_gpu_allocated_mib is not None
                and self.cleanup_gpu_reserved_mib is not None
            )
        )

    @property
    def resource_limits_passed(self) -> bool:
        return not self.limit_exceeded and self.telemetry_complete

    def _note_error(self, error: Exception) -> None:
        self.telemetry_error = f"{type(error).__name__}: {error}"
