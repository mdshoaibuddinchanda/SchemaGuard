"""Fail-closed resource observation for isolated scheduler worker processes."""

from __future__ import annotations

import json
import multiprocessing
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import FailureCategory, ResourceRecord, TaskResult, TaskSpec
from .state import TaskStateStore
from .worker import worker_entry


@dataclass(frozen=True)
class WorkerExecution:
    passed: bool
    resource: ResourceRecord
    result: dict[str, Any] | None
    failure_category: FailureCategory | None
    failure_reason: str | None
    traceback: str | None = None


def _sample_process_tree(pid: int) -> tuple[float | None, float | None, float | None]:
    try:
        import psutil

        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
        rss_values: list[float] = []
        cpu_seconds = 0.0
        for process in processes:
            try:
                memory = process.memory_info().rss / (1024**2)
                times = process.cpu_times()
            except psutil.NoSuchProcess:
                continue
            rss_values.append(memory)
            cpu_seconds += times.user + times.system
        if not rss_values:
            return None, None, None
        return rss_values[0], sum(rss_values), cpu_seconds
    except Exception:
        return None, None, None


def classify_resource_limits(
    *,
    process_tree_rss_mib: float | None,
    gpu_reserved_mib: float | None,
    ram_hard_limit_mib: float,
    vram_soft_limit_mib: float | None,
) -> FailureCategory | None:
    """Classify measured hard RAM or VRAM breaches without substituting missing values."""

    if process_tree_rss_mib is not None and process_tree_rss_mib > ram_hard_limit_mib:
        return "FAIL_RESOURCE_LIMIT"
    if (
        gpu_reserved_mib is not None
        and vram_soft_limit_mib is not None
        and gpu_reserved_mib > vram_soft_limit_mib
    ):
        return "FAIL_RESOURCE_LIMIT"
    return None


def execute_isolated_worker(
    task: TaskSpec,
    work_directory: str | Path,
    *,
    hard_ram_limit_mib: int,
    soft_ram_limit_mib: int,
    heartbeat_interval_seconds: float,
    state_store: TaskStateStore | None = None,
    gpu_sampler: Callable[[], dict[str, float | None] | None] | None = None,
    gpu_soft_limit_mib: int | None = None,
) -> WorkerExecution:
    """Spawn one Windows-safe process, enforcing time and process-tree RAM limits."""

    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    payload_path = work / "payload.bin"
    result_path = work / "worker_result.json"
    started = datetime.now(UTC)
    wall_started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=worker_entry,
        args=(task.model_dump(mode="json"), str(payload_path), str(result_path)),
        name=f"schemaguard-{task.task_name}",
    )
    category: FailureCategory | None = None
    reason: str | None = None
    traceback_text: str | None = None
    peak_worker: float | None = None
    peak_tree: float | None = None
    cpu_peak: float | None = None
    telemetry_samples = 0
    timed_out = False
    soft_warning = False
    hard_passed = True
    cleanup_passed = True
    result: dict[str, Any] | None = None
    gpu_free_before: float | None = None
    gpu_allocated_peak: float | None = None
    gpu_reserved_peak: float | None = None
    gpu_samples = 0
    if task.cache_identity.device_policy == "cuda":
        snapshot = gpu_sampler() if gpu_sampler is not None else None
        if snapshot is None or any(
            snapshot.get(name) is None for name in ("free_mib", "allocated_mib", "reserved_mib")
        ):
            category = "FAIL_RESOURCE_MONITOR"
            reason = "mandatory GPU telemetry is unavailable before worker launch"
        else:
            free_mib = snapshot.get("free_mib")
            allocated_mib = snapshot.get("allocated_mib")
            reserved_mib = snapshot.get("reserved_mib")
            assert free_mib is not None and allocated_mib is not None and reserved_mib is not None
            gpu_free_before = float(free_mib)
            gpu_allocated_peak = float(allocated_mib)
            gpu_reserved_peak = float(reserved_mib)
            gpu_samples = 0
    try:
        if category is not None:
            raise RuntimeError(reason or "GPU telemetry preflight failed")
        process.start()
        worker_pid = process.pid
        if worker_pid is None:
            raise RuntimeError("spawned worker process has no PID")
        if state_store is not None:
            state_store.set_worker_pid(task.task_id, worker_pid)
        last_heartbeat = time.perf_counter()
        while process.is_alive():
            worker_rss, tree_rss, cpu_value = _sample_process_tree(worker_pid)
            if worker_rss is not None and tree_rss is not None:
                telemetry_samples += 1
                peak_worker = max(worker_rss, peak_worker or 0.0)
                peak_tree = max(tree_rss, peak_tree or 0.0)
                if tree_rss > soft_ram_limit_mib:
                    soft_warning = True
                if classify_resource_limits(
                    process_tree_rss_mib=tree_rss,
                    gpu_reserved_mib=None,
                    ram_hard_limit_mib=hard_ram_limit_mib,
                    vram_soft_limit_mib=None,
                ):
                    hard_passed = False
                    category = "FAIL_RESOURCE_LIMIT"
                    reason = (
                        f"process-tree RSS {tree_rss:.1f} MiB exceeded "
                        f"{hard_ram_limit_mib} MiB hard limit"
                    )
                    process.terminate()
                    break
            if cpu_value is not None:
                cpu_peak = max(cpu_value, cpu_peak or 0.0)
            if task.cache_identity.device_policy == "cuda" and gpu_sampler is not None:
                gpu_snapshot = gpu_sampler()
                if gpu_snapshot is None or any(
                    gpu_snapshot.get(name) is None
                    for name in ("free_mib", "allocated_mib", "reserved_mib")
                ):
                    category = "FAIL_RESOURCE_MONITOR"
                    reason = "GPU telemetry became unavailable during worker execution"
                    process.terminate()
                    break
                allocated_mib = gpu_snapshot.get("allocated_mib")
                reserved_mib = gpu_snapshot.get("reserved_mib")
                assert allocated_mib is not None and reserved_mib is not None
                gpu_samples += 1
                allocated = float(allocated_mib)
                reserved = float(reserved_mib)
                gpu_allocated_peak = max(allocated, gpu_allocated_peak or 0.0)
                gpu_reserved_peak = max(reserved, gpu_reserved_peak or 0.0)
                if classify_resource_limits(
                    process_tree_rss_mib=None,
                    gpu_reserved_mib=reserved,
                    ram_hard_limit_mib=hard_ram_limit_mib,
                    vram_soft_limit_mib=(
                        float(gpu_soft_limit_mib) if gpu_soft_limit_mib is not None else None
                    ),
                ):
                    hard_passed = False
                    category = "FAIL_RESOURCE_LIMIT"
                    reason = (
                        f"reserved VRAM {reserved:.1f} MiB exceeded "
                        f"{gpu_soft_limit_mib} MiB hard limit"
                    )
                    process.terminate()
                    break
            now = time.perf_counter()
            if state_store is not None and now - last_heartbeat >= heartbeat_interval_seconds:
                state_store.heartbeat(task.task_id)
                last_heartbeat = now
            if now - wall_started > task.timeout_seconds:
                timed_out = True
                category = "FAIL_TIMEOUT"
                reason = f"worker exceeded {task.timeout_seconds:.3f} second timeout"
                process.terminate()
                break
            process.join(timeout=0.5)
        process.join(timeout=10)
        cleanup_passed = process.pid is None or not process.is_alive()
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
            cleanup_passed = not process.is_alive()
            if category is None:
                category = "FAIL_RESOURCE_MONITOR"
                reason = "worker could not be terminated and reaped"
        if category is None and result_path.is_file():
            raw_result: Any = None
            try:
                raw_result = json.loads(result_path.read_text(encoding="utf-8"))
                validated_result = TaskResult.model_validate(raw_result)
                if (
                    validated_result.worker_pid != worker_pid
                    or validated_result.task_id != task.task_id
                ):
                    raise ValueError(
                        "worker result PID or task identity does not match its process"
                    )
                result = validated_result.model_dump(mode="json")
            except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                category = "FAIL_MODEL_RUNTIME"
                reason = f"worker result is truncated or invalid: {type(exc).__name__}"
                if isinstance(raw_result, dict):
                    possible_traceback = raw_result.get("traceback")
                    if isinstance(possible_traceback, str):
                        traceback_text = possible_traceback
        if result is not None:
            if result.get("cpu_seconds") is not None:
                cpu_peak = max(float(result["cpu_seconds"]), cpu_peak or 0.0)
            traceback_text = result.get("traceback")
            if result.get("ok") is not True:
                error_text = str(result.get("error", "worker reported failure"))
                if "out of memory" in error_text.casefold() and "cuda" in error_text.casefold():
                    category = "FAIL_CUDA_OOM"
                else:
                    category = "FAIL_MODEL_RUNTIME"
                reason = error_text
        elif category is None:
            category = "FAIL_MODEL_RUNTIME"
            reason = f"worker exited without a valid result (exit code {process.exitcode})"
        if category is None and process.exitcode != 0:
            error_text = str((result or {}).get("error", ""))
            if "out of memory" in error_text.casefold() and "cuda" in error_text.casefold():
                category = "FAIL_CUDA_OOM"
                reason = error_text or "CUDA worker exceeded available memory"
            else:
                category = "FAIL_MODEL_RUNTIME"
                reason = f"worker exited with code {process.exitcode}"
        if telemetry_samples == 0 and category is None:
            category = "FAIL_RESOURCE_MONITOR"
            reason = "mandatory worker RAM telemetry is unavailable"
    except Exception as exc:
        category = "FAIL_RESOURCE_MONITOR"
        reason = f"resource monitor failed: {type(exc).__name__}: {exc}"
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        cleanup_passed = not process.is_alive()
    finally:
        if process.pid is not None and process.is_alive():
            process.kill()
            process.join(timeout=5)
        cleanup_passed = cleanup_passed or process.pid is None or not process.is_alive()

    ended = datetime.now(UTC)
    wall_seconds = max(0.0, time.perf_counter() - wall_started)
    telemetry_complete = (
        telemetry_samples > 0
        and peak_worker is not None
        and peak_tree is not None
        and (
            task.cache_identity.device_policy == "cpu"
            or (
                gpu_samples > 0
                and gpu_free_before is not None
                and gpu_allocated_peak is not None
                and gpu_reserved_peak is not None
            )
        )
    )
    if not telemetry_complete and category is None:
        category = "FAIL_RESOURCE_MONITOR"
        reason = "mandatory worker resource telemetry is incomplete"
    resource = ResourceRecord(
        started_at=started,
        ended_at=ended,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_peak,
        worker_pid=process.pid,
        exit_code=process.exitcode,
        peak_worker_rss_mib=peak_worker,
        peak_process_tree_rss_mib=peak_tree,
        gpu_allocated_mib=gpu_allocated_peak,
        gpu_reserved_mib=gpu_reserved_peak,
        gpu_free_before_mib=gpu_free_before,
        timed_out=timed_out,
        soft_limit_warning=soft_warning,
        hard_limit_passed=hard_passed,
        cleanup_passed=cleanup_passed,
        telemetry_complete=telemetry_complete,
        failure_category=category,
        failure_reason=reason,
    )
    return WorkerExecution(
        passed=category is None and resource.telemetry_complete and resource.cleanup_passed,
        resource=resource,
        result=result,
        failure_category=category,
        failure_reason=reason,
        traceback=traceback_text,
    )


__all__ = ["WorkerExecution", "classify_resource_limits", "execute_isolated_worker"]
