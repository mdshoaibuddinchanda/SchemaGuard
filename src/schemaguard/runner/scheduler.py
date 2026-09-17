"""Bounded CPU process scheduling and exclusive, adapter-policy GPU execution."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..cache.contracts import CacheSchedulerConfig
from ..cache.index import CacheIndex
from ..cache.store import CacheArtifact, CacheIntegrityError, CacheStore
from ..models.adapters.resources import GpuExecutionLock, ensure_gpu_budget
from ..utils.io import atomic_write_text
from .contracts import (
    FailureCategory,
    FailureRecord,
    ResourceRecord,
    RunManifest,
    RunPlan,
    TaskSpec,
)
from .resources import WorkerExecution, execute_isolated_worker
from .state import TaskStateStore

GpuSampler = Callable[[], dict[str, float | None] | None]


@dataclass
class SchedulerCounters:
    planned: int = 0
    executed: int = 0
    validated_cache_hits: int = 0
    recovered_abandoned: int = 0
    failed: int = 0
    blocked: int = 0
    retried: int = 0
    cancelled: int = 0


@dataclass(frozen=True)
class SchedulerResult:
    manifest: RunManifest
    state_path: Path


class Scheduler:
    """Schedule synthetic or future serializable condition tasks without pool nesting."""

    def __init__(
        self,
        repository_root: str | Path,
        config: CacheSchedulerConfig,
        cache: CacheStore,
        index: CacheIndex,
        *,
        source_commit: str,
        gpu_sampler: GpuSampler | None = None,
        gpu_budget_check: Callable[[], None] | None = None,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.config = config
        self.cache = cache
        self.index = index
        self.source_commit = source_commit
        self.gpu_sampler = gpu_sampler or _nvidia_smi_gpu_sample
        self.gpu_budget_check = gpu_budget_check
        self._measure_lock = threading.Lock()
        self._active_cpu = 0
        self._max_cpu = 0
        self._active_gpu = 0
        self._max_gpu = 0
        self._offline_network_attempts = 0

    def run(
        self, plan: RunPlan, *, resume: bool = False, retry_failed: bool = False
    ) -> SchedulerResult:
        state = TaskStateStore(
            self.repository_root / self.config.state_root,
            plan,
            lock_timeout=self.config.lock_timeout_seconds,
        )
        state.initialize()
        recovered = state.recover_abandoned(self.config.abandoned_lease_seconds)
        counters = SchedulerCounters(planned=len(plan.tasks), recovered_abandoned=recovered)
        self.index.rebuild(self.cache)

        cpu_tasks: list[TaskSpec] = []
        gpu_tasks: list[TaskSpec] = []
        for task in plan.tasks:
            record = state.read().tasks[task.task_id]
            if record.state == "FAILED":
                if not retry_failed:
                    counters.failed += 1
                    continue
                state.transition(
                    task.task_id, "PENDING", reason="EXPLICIT_RETRY", retry_failed=True
                )
                counters.retried += 1
            elif record.state == "BLOCKED":
                counters.blocked += 1
                continue
            elif record.state == "CANCELLED":
                counters.cancelled += 1
                continue
            elif record.state == "COMPLETE":
                try:
                    artifact = self.cache.read_validated(task.cache_identity)
                except CacheIntegrityError:
                    artifact = None
                if artifact is not None:
                    self.index.add(artifact)
                    counters.validated_cache_hits += 1
                    continue
                state.transition(task.task_id, "PENDING", reason="INVALID_CACHE_ARTIFACT")
                self.cache.quarantine_invalid(task.cache_identity)

            record = state.read().tasks[task.task_id]
            if record.state != "PENDING":
                continue
            try:
                artifact = self.cache.read_validated(task.cache_identity)
            except CacheIntegrityError:
                artifact = None
                self.cache.quarantine_invalid(task.cache_identity)
            if artifact is not None:
                state.transition(task.task_id, "RUNNING", reason="VALIDATED_CACHE_HIT")
                state.transition(
                    task.task_id,
                    "COMPLETE",
                    reason="VALIDATED_CACHE_HIT",
                    artifact_path=self._relative_payload_path(artifact),
                    artifact_sha256=artifact.payload_sha256,
                    cache_hit=True,
                )
                self.index.add(artifact)
                counters.validated_cache_hits += 1
                continue
            state.transition(task.task_id, "RUNNING", reason="TASK_SCHEDULED")
            (gpu_tasks if task.cache_identity.device_policy == "cuda" else cpu_tasks).append(task)

        resources: list[ResourceRecord] = []
        failures: list[FailureRecord] = []
        if cpu_tasks:
            max_workers = min(self.config.cpu_workers, plan.cpu_workers)
            task_iterator = iter(cpu_tasks)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                pending: dict[
                    Future[tuple[WorkerExecution | None, CacheArtifact | None, bool]], TaskSpec
                ] = {}
                for _ in range(max_workers):
                    initial_task = next(task_iterator, None)
                    if initial_task is not None:
                        pending[pool.submit(self._run_cpu_task, initial_task, state)] = initial_task
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        task = pending.pop(future)
                        outcome, artifact, was_hit = future.result()
                        if was_hit:
                            counters.validated_cache_hits += 1
                        elif outcome is not None:
                            resources.append(outcome.resource)
                            if outcome.passed and artifact is not None:
                                counters.executed += 1
                            else:
                                counters.failed += 1
                                category = outcome.failure_category or "FAIL_MODEL_RUNTIME"
                                failures.append(
                                    self._failure_record(
                                        task,
                                        category,
                                        outcome.failure_reason,
                                        outcome.traceback,
                                    )
                                )
                        next_task = next(task_iterator, None)
                        if next_task is not None:
                            pending[pool.submit(self._run_cpu_task, next_task, state)] = next_task

        for task in gpu_tasks:
            outcome, artifact, was_hit = self._run_gpu_task(task, state)
            if was_hit:
                counters.validated_cache_hits += 1
            elif outcome is None:
                counters.blocked += 1
                current = state.read().tasks[task.task_id]
                category = current.failure_category or "BLOCKED_GPU_UNAVAILABLE"
                failures.append(
                    FailureRecord(
                        task_id=task.task_id,
                        category=category,
                        reason=current.failure_reason or "GPU task blocked",
                    )
                )
            else:
                resources.append(outcome.resource)
                if outcome.passed and artifact is not None:
                    counters.executed += 1
                else:
                    counters.failed += 1
                    category = outcome.failure_category or "FAIL_MODEL_RUNTIME"
                    failures.append(
                        self._failure_record(
                            task, category, outcome.failure_reason, outcome.traceback
                        )
                    )

        final_state = state.read()
        # Counts include already-terminal records so a resume report is complete and auditable.
        counters.failed = sum(item.state == "FAILED" for item in final_state.tasks.values())
        counters.blocked = sum(item.state == "BLOCKED" for item in final_state.tasks.values())
        counters.cancelled = sum(item.state == "CANCELLED" for item in final_state.tasks.values())
        for record in final_state.tasks.values():
            if record.attempts and record.attempts[-1].resource is not None:
                if record.attempts[-1].resource not in resources:
                    resources.append(record.attempts[-1].resource)
            if record.state in {"FAILED", "BLOCKED"} and not any(
                item.task_id == record.task.task_id for item in failures
            ):
                category = record.failure_category or "FAIL_MODEL_RUNTIME"
                failures.append(
                    self._failure_record(
                        record.task,
                        category,
                        record.failure_reason,
                        None,
                    )
                )
        manifest = RunManifest(
            schema_version=1,
            stage="cache_scheduler_run",
            plan_hash=plan.plan_hash,
            source_implementation_commit=self.source_commit,
            mode="resume" if resume else "cold",
            planned=counters.planned,
            executed=counters.executed,
            validated_cache_hits=counters.validated_cache_hits,
            recovered_abandoned=counters.recovered_abandoned,
            failed=counters.failed,
            blocked=counters.blocked,
            retried=counters.retried,
            cancelled=counters.cancelled,
            max_cpu_concurrency=self._max_cpu,
            max_gpu_concurrency=self._max_gpu,
            offline_network_attempt_count=self._offline_network_attempts,
            resources=resources,
            failures=failures,
        )
        return SchedulerResult(manifest=manifest, state_path=state.path)

    def _run_cpu_task(
        self, task: TaskSpec, state: TaskStateStore
    ) -> tuple[WorkerExecution | None, CacheArtifact | None, bool]:
        return self._execute_locked(task, state, gpu=False)

    def _execute_locked(
        self, task: TaskSpec, state: TaskStateStore, *, gpu: bool
    ) -> tuple[WorkerExecution | None, CacheArtifact | None, bool]:
        try:
            with self.cache.identity_lock(task.cache_identity):
                try:
                    existing = self.cache.read_validated(task.cache_identity)
                except CacheIntegrityError:
                    self.cache.quarantine_invalid_locked(task.cache_identity)
                    existing = None
                if existing is not None:
                    state.transition(
                        task.task_id,
                        "COMPLETE",
                        reason="VALIDATED_CACHE_HIT_AFTER_LOCK",
                        artifact_path=self._relative_payload_path(existing),
                        artifact_sha256=existing.payload_sha256,
                        cache_hit=True,
                    )
                    self.index.add(existing)
                    return None, existing, True

                work_root = self.repository_root / self.config.state_root / "workers"
                work_root.mkdir(parents=True, exist_ok=True)
                work_dir = Path(tempfile.mkdtemp(prefix=f"{task.task_id[:12]}-", dir=work_root))
                try:
                    if gpu:
                        sampler = self.gpu_sampler
                        snapshot = sampler() if sampler is not None else None
                        if snapshot is None or any(
                            snapshot.get(name) is None
                            for name in ("free_mib", "allocated_mib", "reserved_mib")
                        ):
                            state.transition(
                                task.task_id,
                                "BLOCKED",
                                reason="BLOCKED_GPU_UNAVAILABLE: required GPU telemetry missing",
                                failure_category="BLOCKED_GPU_UNAVAILABLE",
                            )
                            return None, None, False
                        try:
                            if self.gpu_budget_check is None:
                                ensure_gpu_budget(
                                    headroom_mib=self.config.gpu_headroom_mib,
                                    expected_peak_mib=0.0,
                                )
                            else:
                                self.gpu_budget_check()
                        except RuntimeError as exc:
                            category: FailureCategory = (
                                "BLOCKED_INSUFFICIENT_VRAM"
                                if "insufficient" in str(exc).casefold()
                                else "BLOCKED_GPU_UNAVAILABLE"
                            )
                            state.transition(
                                task.task_id,
                                "BLOCKED",
                                reason=f"{category}: {exc}",
                                failure_category=category,
                            )
                            return None, None, False
                    with self._measure_lock:
                        if gpu:
                            self._active_gpu += 1
                            self._max_gpu = max(self._max_gpu, self._active_gpu)
                        else:
                            self._active_cpu += 1
                            self._max_cpu = max(self._max_cpu, self._active_cpu)
                    try:
                        outcome = execute_isolated_worker(
                            task,
                            work_dir,
                            hard_ram_limit_mib=self.config.ram_hard_limit_mib,
                            soft_ram_limit_mib=self.config.ram_soft_limit_mib,
                            heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
                            state_store=state,
                            gpu_sampler=self.gpu_sampler if gpu else None,
                            gpu_soft_limit_mib=(self.config.vram_soft_limit_mib if gpu else None),
                        )
                        if outcome.result is not None:
                            with self._measure_lock:
                                self._offline_network_attempts += int(
                                    outcome.result.get("network_attempt_count", 0)
                                )
                    finally:
                        with self._measure_lock:
                            if gpu:
                                self._active_gpu -= 1
                            else:
                                self._active_cpu -= 1

                    if outcome.passed and outcome.result is not None:
                        payload_path = work_dir / "payload.bin"
                        if (
                            not payload_path.is_file()
                            or outcome.result.get("task_id") != task.task_id
                            or outcome.result.get("payload_sha256") != _file_sha256(payload_path)
                        ):
                            outcome = self._replace_with_integrity_failure(outcome)
                    if outcome.passed:
                        published = self.cache.publish_file_locked(
                            task.cache_identity,
                            work_dir / "payload.bin",
                            producing_task_identity=task.task_id,
                        )
                        artifact = published.artifact
                        self.index.add(artifact)
                        state.transition(
                            task.task_id,
                            "COMPLETE",
                            reason="ARTIFACT_VALIDATED_AND_PUBLISHED",
                            resource=outcome.resource,
                            artifact_path=self._relative_payload_path(artifact),
                            artifact_sha256=artifact.payload_sha256,
                        )
                        return outcome, artifact, False
                    category = outcome.failure_category or "FAIL_MODEL_RUNTIME"
                    traceback_path = self._failure_record(
                        task, category, outcome.failure_reason, outcome.traceback
                    ).traceback_relative_path
                    state.transition(
                        task.task_id,
                        "FAILED",
                        reason=outcome.failure_reason or category,
                        failure_category=category,
                        resource=outcome.resource,
                    )
                    if traceback_path:
                        failures_path = (
                            self.repository_root / self.config.state_root / traceback_path
                        )
                        failures_path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_text(
                            failures_path, outcome.traceback or outcome.failure_reason or category
                        )
                    return outcome, None, False
                finally:
                    shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as exc:
            failure_category: FailureCategory = (
                "FAIL_CACHE_INTEGRITY"
                if isinstance(exc, CacheIntegrityError)
                else "FAIL_MODEL_RUNTIME"
            )
            current = state.read().tasks[task.task_id]
            if current.state == "RUNNING":
                state.transition(
                    task.task_id,
                    "FAILED",
                    reason=f"{type(exc).__name__}: {exc}",
                    failure_category=failure_category,
                )
            resource = _failed_resource(failure_category, f"{type(exc).__name__}: {exc}")
            outcome = WorkerExecution(
                False, resource, None, failure_category, str(exc), traceback.format_exc()
            )
            return outcome, None, False

    def _run_gpu_task(
        self, task: TaskSpec, state: TaskStateStore
    ) -> tuple[WorkerExecution | None, CacheArtifact | None, bool]:
        try:
            lock = GpuExecutionLock(self.repository_root, timeout=self.config.lock_timeout_seconds)
            lock.acquire()
            try:
                return self._execute_locked(task, state, gpu=True)
            finally:
                lock.release()
        except Exception as exc:
            current = state.read().tasks[task.task_id]
            category: FailureCategory = "BLOCKED_GPU_UNAVAILABLE"
            if current.state == "RUNNING":
                state.transition(
                    task.task_id,
                    "BLOCKED",
                    reason=f"{category}: {type(exc).__name__}: {exc}",
                    failure_category=category,
                )
            return None, None, False

    def _replace_with_integrity_failure(self, outcome: WorkerExecution) -> WorkerExecution:
        resource = outcome.resource.model_copy(
            update={
                "failure_category": "FAIL_CACHE_INTEGRITY",
                "failure_reason": "worker output is missing or differs from its returned digest",
            }
        )
        return WorkerExecution(
            passed=False,
            resource=resource,
            result=outcome.result,
            failure_category="FAIL_CACHE_INTEGRITY",
            failure_reason="worker output is missing or differs from its returned digest",
        )

    def _failure_record(
        self,
        task: TaskSpec,
        category: FailureCategory,
        reason: str | None,
        traceback_text: str | None,
    ) -> FailureRecord:
        relative: str | None = None
        if traceback_text:
            relative = f"tracebacks/{task.task_id}.txt"
        return FailureRecord(
            task_id=task.task_id,
            category=category,
            reason=reason or category,
            traceback_relative_path=relative,
        )

    def _relative_payload_path(self, artifact: CacheArtifact) -> str:
        return artifact.payload_path.resolve().relative_to(self.cache.root.resolve()).as_posix()


def _nvidia_smi_gpu_sample() -> dict[str, float | None] | None:
    """Read device-wide telemetry conservatively; unavailable fields stay null."""

    try:
        output = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        total, used, free = next(csv.reader([output]))
        return {
            "total_mib": float(total.strip()),
            "allocated_mib": float(used.strip()),
            "reserved_mib": float(used.strip()),
            "free_mib": float(free.strip()),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _failed_resource(category: FailureCategory, reason: str) -> ResourceRecord:
    now = datetime.now(UTC)
    return ResourceRecord(
        started_at=now,
        ended_at=now,
        wall_seconds=0,
        cpu_seconds=None,
        worker_pid=None,
        exit_code=None,
        peak_worker_rss_mib=None,
        peak_process_tree_rss_mib=None,
        gpu_allocated_mib=None,
        gpu_reserved_mib=None,
        gpu_free_before_mib=None,
        timed_out=category == "FAIL_TIMEOUT",
        soft_limit_warning=False,
        hard_limit_passed=category != "FAIL_RESOURCE_LIMIT",
        cleanup_passed=True,
        telemetry_complete=False,
        failure_category=category,
        failure_reason=reason,
    )


__all__ = ["Scheduler", "SchedulerCounters", "SchedulerResult"]
