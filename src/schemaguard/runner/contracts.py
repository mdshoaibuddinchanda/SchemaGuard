"""Strict task, attempt, resource, and run-manifest contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..cache.contracts import CacheIdentity
from ..utils.hashing import sha256_canonical_json

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
TaskStatus = Literal["PENDING", "RUNNING", "COMPLETE", "FAILED", "BLOCKED", "CANCELLED"]
FailureCategory = Literal[
    "FAIL_RESOURCE_LIMIT",
    "FAIL_TIMEOUT",
    "FAIL_CUDA_OOM",
    "FAIL_RESOURCE_MONITOR",
    "FAIL_MODEL_RUNTIME",
    "BLOCKED_GPU_UNAVAILABLE",
    "BLOCKED_INSUFFICIENT_VRAM",
    "FAIL_CACHE_INTEGRITY",
]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TaskSpec(StrictContract):
    task_id: Sha256
    task_name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    cache_identity: CacheIdentity
    operation: Literal[
        "synthetic_payload",
        "sleep",
        "raise_error",
        "exit_without_result",
        "partial_exit",
        "network_attempt",
    ]
    payload_token: str = Field(min_length=1, max_length=128)
    delay_seconds: float = Field(default=0.0, ge=0, le=3600, allow_inf_nan=False)
    timeout_seconds: float = Field(default=300.0, gt=0, le=86400, allow_inf_nan=False)

    @model_validator(mode="after")
    def verify_task_identity(self) -> TaskSpec:
        expected = task_identity(
            task_name=self.task_name,
            cache_key=self.cache_identity.cache_key,
            operation=self.operation,
            payload_token=self.payload_token,
            delay_seconds=self.delay_seconds,
            timeout_seconds=self.timeout_seconds,
        )
        if self.task_id != expected:
            raise ValueError("task ID does not match its canonical specification")
        return self


class TaskResult(StrictContract):
    """Strict worker envelope; payloads travel by file path, never through queues."""

    schema_version: Literal[1]
    ok: bool
    task_id: Sha256
    payload_sha256: Sha256 | None = None
    payload_size_bytes: int | None = Field(default=None, ge=0)
    worker_pid: int = Field(gt=0)
    worker_rss_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cpu_seconds: float = Field(ge=0, allow_inf_nan=False)
    worker_seconds: float = Field(ge=0, allow_inf_nan=False)
    network_attempt_count: int = Field(ge=0)
    thread_environment: dict[str, str | None] = Field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> TaskResult:
        if self.ok:
            if self.payload_sha256 is None or self.payload_size_bytes is None:
                raise ValueError("successful worker result requires payload digest and size")
            if self.error is not None or self.traceback is not None:
                raise ValueError("successful worker result cannot contain failure details")
            expected_threads = {
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            }
            if set(self.thread_environment) != expected_threads or any(
                value != "2" for value in self.thread_environment.values()
            ):
                raise ValueError("successful workers must report all four two-thread limits")
        elif not self.error:
            raise ValueError("failed worker result requires an error description")
        return self


def task_identity(
    *,
    task_name: str,
    cache_key: str,
    operation: str,
    payload_token: str,
    delay_seconds: float,
    timeout_seconds: float,
) -> str:
    return sha256_canonical_json(
        {
            "task_name": task_name,
            "cache_key": cache_key,
            "operation": operation,
            "payload_token": payload_token,
            "delay_seconds": float(delay_seconds),
            "timeout_seconds": float(timeout_seconds),
        }
    )


def build_task(
    task_name: str,
    cache_identity: CacheIdentity,
    *,
    operation: Literal[
        "synthetic_payload",
        "sleep",
        "raise_error",
        "exit_without_result",
        "partial_exit",
        "network_attempt",
    ] = "synthetic_payload",
    payload_token: str = "cache-scheduler-probe",
    delay_seconds: float = 0.0,
    timeout_seconds: float = 300.0,
) -> TaskSpec:
    delay_value = float(delay_seconds)
    timeout_value = float(timeout_seconds)
    return TaskSpec(
        task_id=task_identity(
            task_name=task_name,
            cache_key=cache_identity.cache_key,
            operation=operation,
            payload_token=payload_token,
            delay_seconds=delay_value,
            timeout_seconds=timeout_value,
        ),
        task_name=task_name,
        cache_identity=cache_identity,
        operation=operation,
        payload_token=payload_token,
        delay_seconds=delay_value,
        timeout_seconds=timeout_value,
    )


class RunPlan(StrictContract):
    schema_version: Literal[1]
    tasks: list[TaskSpec]
    random_seed: int = Field(ge=0)
    cpu_workers: Literal[1, 2]
    gpu_workers: Literal[1]

    @model_validator(mode="after")
    def unique_and_ordered_tasks(self) -> RunPlan:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("run plan contains duplicate task identities")
        if ids != sorted(ids):
            raise ValueError("run plan tasks must be in canonical task-ID order")
        return self

    @property
    def plan_hash(self) -> str:
        return sha256_canonical_json(self.model_dump(mode="json"))


def build_plan(
    tasks: list[TaskSpec], *, random_seed: int, cpu_workers: Literal[1, 2] = 2
) -> RunPlan:
    return RunPlan(
        schema_version=1,
        tasks=sorted(tasks, key=lambda task: task.task_id),
        random_seed=random_seed,
        cpu_workers=cpu_workers,
        gpu_workers=1,
    )


class ResourceRecord(StrictContract):
    started_at: datetime
    ended_at: datetime
    wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    cpu_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    worker_pid: int | None = Field(default=None, gt=0)
    exit_code: int | None = None
    peak_worker_rss_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_process_tree_rss_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    gpu_allocated_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    gpu_reserved_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    gpu_free_before_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    timed_out: bool
    soft_limit_warning: bool
    hard_limit_passed: bool
    cleanup_passed: bool
    telemetry_complete: bool
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_measurement(self) -> ResourceRecord:
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("resource timestamps must include a timezone")
        if self.ended_at < self.started_at:
            raise ValueError("resource end time precedes start time")
        if self.telemetry_complete and self.worker_pid is not None:
            if self.peak_worker_rss_mib is None or self.peak_process_tree_rss_mib is None:
                raise ValueError("complete worker telemetry requires both peak RSS values")
        if self.timed_out and self.failure_category != "FAIL_TIMEOUT":
            raise ValueError("timeouts must use FAIL_TIMEOUT")
        if not self.hard_limit_passed and self.failure_category != "FAIL_RESOURCE_LIMIT":
            raise ValueError("hard resource-limit failure must be explicitly classified")
        return self


class TaskAttempt(StrictContract):
    attempt_number: int = Field(ge=1)
    state: TaskStatus
    started_at: datetime
    heartbeat_at: datetime
    ended_at: datetime | None = None
    owner_pid: int | None = Field(default=None, gt=0)
    worker_pid: int | None = Field(default=None, gt=0)
    resource: ResourceRecord | None = None
    failure_category: FailureCategory | None = None
    reason: str | None = None


class TaskTransition(StrictContract):
    from_state: TaskStatus
    to_state: TaskStatus
    timestamp: datetime
    reason: str | None = None


class TaskStateRecord(StrictContract):
    task: TaskSpec
    state: TaskStatus = "PENDING"
    attempts: list[TaskAttempt] = Field(default_factory=list)
    transitions: list[TaskTransition] = Field(default_factory=list)
    artifact_path: str | None = None
    artifact_sha256: Sha256 | None = None
    cache_hit: bool = False
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_artifact_state(self) -> TaskStateRecord:
        if self.state == "COMPLETE":
            if self.artifact_path is None or self.artifact_sha256 is None:
                raise ValueError("complete task state requires a relative artifact and digest")
            path = PurePosixPath(self.artifact_path)
            if (
                path.is_absolute()
                or PureWindowsPath(self.artifact_path).is_absolute()
                or ".." in path.parts
                or "\\" in self.artifact_path
            ):
                raise ValueError("artifact path must be portable and relative")
        elif self.artifact_path is not None or self.artifact_sha256 is not None:
            raise ValueError("non-complete task state cannot reference a completed artifact")
        return self


class SchedulerState(StrictContract):
    schema_version: Literal[1]
    plan_hash: Sha256
    tasks: dict[str, TaskStateRecord]


class FailureRecord(StrictContract):
    task_id: Sha256
    category: FailureCategory
    reason: str
    traceback_relative_path: str | None = None


class RunManifest(StrictContract):
    schema_version: Literal[1]
    stage: Literal["cache_scheduler_run"]
    plan_hash: Sha256
    source_implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    mode: Literal["cold", "resume"]
    planned: int = Field(ge=0)
    executed: int = Field(ge=0)
    validated_cache_hits: int = Field(ge=0)
    recovered_abandoned: int = Field(ge=0)
    failed: int = Field(ge=0)
    blocked: int = Field(ge=0)
    retried: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    max_cpu_concurrency: int = Field(ge=0, le=2)
    max_gpu_concurrency: int = Field(ge=0, le=1)
    offline_network_attempt_count: int = Field(ge=0)
    resources: list[ResourceRecord]
    failures: list[FailureRecord]


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "FailureCategory",
    "FailureRecord",
    "ResourceRecord",
    "RunManifest",
    "RunPlan",
    "SchedulerState",
    "TaskAttempt",
    "TaskResult",
    "TaskSpec",
    "TaskStateRecord",
    "TaskStatus",
    "TaskTransition",
    "build_plan",
    "build_task",
    "task_identity",
    "utc_now",
]
