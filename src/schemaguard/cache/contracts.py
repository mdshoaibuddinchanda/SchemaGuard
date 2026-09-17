"""Strict portable contracts for cache identities and scheduler evidence."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..utils.hashing import sha256_canonical_json
from .fault_policy import CANONICAL_FAULT_NAMES, FAULT_POLICY, fault_evidence_sha256

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
TaskId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)


class CacheIdentity(StrictContract):
    """All causal scientific and implementation inputs for one cached artifact."""

    schema_version: Literal[1]
    dataset_sha256: Sha256
    split_sha256: Sha256
    view_certificate_sha256: Sha256
    model_spec_sha256: Sha256
    model_parameters_sha256: Sha256
    checkpoint_sha256: Sha256 | None
    dependency_lock_sha256: Sha256
    source_implementation_sha256: Sha256
    seed: int = Field(ge=0)
    device_policy: Literal["cpu", "cuda"]
    artifact_kind: Literal["model", "prediction", "metric", "condition", "probe"]

    @model_validator(mode="after")
    def reject_placeholder_hashes(self) -> CacheIdentity:
        for field_name in (
            "dataset_sha256",
            "split_sha256",
            "view_certificate_sha256",
            "model_spec_sha256",
            "model_parameters_sha256",
            "dependency_lock_sha256",
            "source_implementation_sha256",
        ):
            if getattr(self, field_name) == "0" * 64:
                raise ValueError(f"{field_name} cannot be a placeholder hash")
        if self.checkpoint_sha256 == "0" * 64:
            raise ValueError("checkpoint_sha256 cannot be a placeholder hash")
        return self

    @property
    def cache_key(self) -> str:
        """Return a platform-independent digest of canonical identity JSON."""

        return sha256_canonical_json(self.model_dump(mode="json"))


class CacheArtifactManifest(StrictContract):
    schema_version: Literal[1]
    identity: CacheIdentity
    cache_key: Sha256
    artifact_kind: Literal["model", "prediction", "metric", "condition", "probe"]
    payload_file: Literal["payload.bin"] = "payload.bin"
    payload_sha256: Sha256
    payload_size_bytes: int = Field(ge=0)
    producing_task_identity: TaskId
    created_at: datetime
    source_implementation_sha256: Sha256
    dependency_lock_sha256: Sha256
    validation_status: Literal["PASS"]
    completed: Literal[True]

    @model_validator(mode="after")
    def identity_matches_manifest(self) -> CacheArtifactManifest:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != UTC.utcoffset(
            self.created_at
        ):
            raise ValueError("created_at must be a UTC timestamp")
        if self.cache_key != self.identity.cache_key:
            raise ValueError("manifest cache key does not match canonical identity")
        if self.artifact_kind != self.identity.artifact_kind:
            raise ValueError("manifest artifact kind differs from identity")
        if self.source_implementation_sha256 != self.identity.source_implementation_sha256:
            raise ValueError("manifest implementation hash differs from identity")
        if self.dependency_lock_sha256 != self.identity.dependency_lock_sha256:
            raise ValueError("manifest dependency-lock hash differs from identity")
        return self


class CompletionMarker(StrictContract):
    schema_version: Literal[1]
    manifest_sha256: Sha256
    payload_sha256: Sha256


class CacheSchedulerConfig(StrictContract):
    schema_version: Literal[1]
    cpu_workers: Literal[2]
    cpu_threads_per_worker: Literal[2]
    gpu_workers: Literal[1]
    ram_soft_limit_mib: int = Field(gt=0, le=24576)
    ram_hard_limit_mib: int = Field(gt=0, le=28672)
    vram_soft_limit_mib: int = Field(gt=0, le=3600)
    gpu_headroom_mib: int = Field(ge=512)
    lock_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    heartbeat_interval_seconds: float = Field(gt=0, allow_inf_nan=False)
    abandoned_lease_seconds: float = Field(gt=0, allow_inf_nan=False)
    cache_root: str = Field(min_length=1)
    state_root: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy(self) -> CacheSchedulerConfig:
        if self.ram_soft_limit_mib >= self.ram_hard_limit_mib:
            raise ValueError("soft RAM limit must be lower than hard RAM limit")
        if self.heartbeat_interval_seconds >= self.abandoned_lease_seconds:
            raise ValueError("heartbeat interval must be shorter than abandoned lease")
        for name in ("cache_root", "state_root"):
            value = getattr(self, name).replace("\\", "/")
            if value.startswith("/") or (len(value) >= 2 and value[1] == ":"):
                raise ValueError(f"{name} must be repository-relative")
            if any(part == ".." for part in value.split("/")):
                raise ValueError(f"{name} cannot escape the repository")
        return self


class ProtectedHashRecord(StrictContract):
    path: str = Field(min_length=1)
    before_sha256: Sha256
    after_sha256: Sha256
    baseline_git_sha256: Sha256
    implementation_git_sha256: Sha256
    before_size_bytes: int = Field(ge=0)
    after_size_bytes: int = Field(ge=0)
    unchanged: bool

    @model_validator(mode="after")
    def validate_protected_path_and_hashes(self) -> ProtectedHashRecord:
        posix = PurePosixPath(self.path)
        windows = PureWindowsPath(self.path)
        if posix.is_absolute() or windows.is_absolute() or "\\" in self.path or ".." in posix.parts:
            raise ValueError("protected file paths must be portable repository-relative paths")
        computed = (
            self.before_sha256 == self.after_sha256
            and self.before_size_bytes == self.after_size_bytes
            and self.baseline_git_sha256 == self.implementation_git_sha256
        )
        if self.unchanged is not computed:
            raise ValueError("protected-file comparison flag contradicts its hashes or sizes")
        return self


class ProtectedHashComparison(StrictContract):
    schema_version: Literal[1]
    baseline_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    expected_file_count: int = Field(gt=0)
    records: list[ProtectedHashRecord]
    status: Literal["PASS", "FAIL"]

    @model_validator(mode="after")
    def validate_complete_comparison(self) -> ProtectedHashComparison:
        paths = [record.path for record in self.records]
        if not paths or paths != sorted(set(paths)) or len(paths) != self.expected_file_count:
            raise ValueError("protected comparison paths must be nonempty, unique, and sorted")
        observed_status = "PASS" if all(record.unchanged for record in self.records) else "FAIL"
        if self.status != observed_status:
            raise ValueError("protected comparison status contradicts its file records")
        return self


class ProbeResourceSummary(StrictContract):
    wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    cpu_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_worker_rss_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_process_tree_rss_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    telemetry_complete: bool
    hard_limit_passed: bool
    cleanup_passed: bool
    failure_category: str | None = None

    @model_validator(mode="after")
    def require_successful_resource_evidence(self) -> ProbeResourceSummary:
        if (
            self.peak_worker_rss_mib is None
            or self.peak_process_tree_rss_mib is None
            or not self.telemetry_complete
            or not self.hard_limit_passed
            or not self.cleanup_passed
            or self.failure_category is not None
        ):
            raise ValueError(
                "successful synthetic probes require complete passing resource evidence"
            )
        return self


class ProbeRunSummary(StrictContract):
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
    resources: list[ProbeResourceSummary]

    @model_validator(mode="after")
    def validate_probe_counters(self) -> ProbeRunSummary:
        if any((self.failed, self.blocked, self.retried, self.cancelled)):
            raise ValueError("successful probe evidence cannot include failed or retried work")
        if self.offline_network_attempt_count != 0 or self.max_gpu_concurrency != 0:
            raise ValueError("CPU synthetic probe must remain offline and use no GPU workers")
        if self.mode == "cold":
            if (
                self.executed != self.planned
                or self.validated_cache_hits != 0
                or len(self.resources) != self.executed
                or not 1 <= self.max_cpu_concurrency <= 2
            ):
                raise ValueError("cold probe must execute and monitor every planned CPU task")
        elif (
            self.executed != 0
            or self.validated_cache_hits != self.planned
            or self.resources
            or self.max_cpu_concurrency != 0
        ):
            raise ValueError("resume probe must validate-hit every task without execution")
        return self


class SchedulerProbeEvidence(StrictContract):
    schema_version: Literal[1]
    stage: Literal["cache_scheduler_probe"]
    source_implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_implementation_sha256: Sha256
    dependency_lock_sha256: Sha256
    configuration_sha256: Sha256
    plan_sha256: Sha256
    cache_identity_hashes: list[Sha256]
    cold: ProbeRunSummary
    resume: ProbeRunSummary
    maximum_mocked_gpu_concurrency: Literal[1]
    mocked_gpu_offline_network_attempt_count: Literal[0]
    validated_cache_artifact_count: int = Field(ge=0)
    duplicate_validated_artifact_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_probe_identity_and_results(self) -> SchedulerProbeEvidence:
        if len(self.cache_identity_hashes) != len(set(self.cache_identity_hashes)):
            raise ValueError("probe cache identities must be unique")
        if self.cold.planned != len(self.cache_identity_hashes):
            raise ValueError("cold probe task count differs from cache identity count")
        if self.resume.planned != self.cold.planned:
            raise ValueError("cold and resume probes must use the same plan size")
        if self.validated_cache_artifact_count != len(self.cache_identity_hashes):
            raise ValueError("probe artifact count differs from planned cache identities")
        if self.duplicate_validated_artifact_count != 0:
            raise ValueError("probe evidence cannot contain duplicate validated artifacts")
        return self


class CacheSchedulerInventory(StrictContract):
    schema_version: Literal[1]
    stage: Literal["cache_scheduler"]
    status: Literal["PASS_PENDING_REVIEW", "REPAIR_REQUIRED", "BLOCKED"]
    source_implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    implementation_hashes: dict[str, Sha256]
    configuration_sha256: Sha256
    plan_sha256: Sha256
    cache_identity_hashes: list[Sha256]
    test_case_names: list[str]
    test_statuses: dict[str, Literal["PASS", "FAIL", "NOT_VERIFIED"]]
    counters: dict[str, int]
    maximum_observed_cpu_concurrency: int = Field(ge=0, le=2)
    maximum_observed_gpu_concurrency: int = Field(ge=0, le=1)
    resource_summaries: dict[str, float | int | None]
    fault_evidence_sha256: Sha256
    probe_evidence_sha256: Sha256
    resume_results: dict[str, int]
    offline_network_attempt_count: int = Field(ge=0)
    protected_artifacts_unchanged: bool
    protected_hash_comparison_sha256: Sha256

    @model_validator(mode="after")
    def validate_inventory_evidence(self) -> CacheSchedulerInventory:
        if len(self.cache_identity_hashes) != len(set(self.cache_identity_hashes)):
            raise ValueError("cache identity hashes must be unique")
        if len(self.test_case_names) != len(set(self.test_case_names)):
            raise ValueError("test case names must be unique")
        for path in self.implementation_hashes:
            normalized = path.replace("\\", "/")
            if (
                normalized.startswith("/")
                or re.match(r"^[A-Za-z]:", normalized)
                or any(part == ".." for part in normalized.split("/"))
            ):
                raise ValueError("implementation hash keys must be portable relative paths")
        if any(count < 0 for count in self.counters.values()):
            raise ValueError("inventory counters cannot be negative")
        if self.status == "PASS_PENDING_REVIEW":
            if not self.protected_artifacts_unchanged:
                raise ValueError("passing inventory requires unchanged protected artifacts")
            if self.offline_network_attempt_count != 0:
                raise ValueError("passing inventory requires zero probe network attempts")
            if not self.test_statuses or any(
                status != "PASS" for status in self.test_statuses.values()
            ):
                raise ValueError("passing inventory requires every recorded test group to pass")
        return self


class FaultInjectionRecord(StrictContract):
    fault_name: str
    injection_point: str
    expected_failure_category: str
    observed_failure_category: str
    expected_artifact_accepted: bool
    artifact_accepted: bool
    expected_resume_behavior: str
    lock_released: bool
    resume_behavior: str
    evidence_sha256: Sha256
    status: Literal["PASS", "FAIL"]

    @model_validator(mode="after")
    def validate_policy_hash_and_status(self) -> FaultInjectionRecord:
        expectation = FAULT_POLICY.get(self.fault_name)
        if expectation is None:
            raise ValueError("fault name is not part of the frozen fault policy")
        if (
            self.injection_point != expectation.injection_point
            or self.expected_failure_category != expectation.expected_failure_category
            or self.expected_artifact_accepted is not expectation.expected_artifact_accepted
            or self.expected_resume_behavior != expectation.expected_resume_behavior
        ):
            raise ValueError("fault record expected behavior differs from the frozen policy")
        computed_hash = fault_evidence_sha256(self.model_dump(mode="python"))
        if self.evidence_sha256 != computed_hash:
            raise ValueError("fault evidence hash does not match its canonical payload")
        derived_status = (
            "PASS"
            if self.observed_failure_category == self.expected_failure_category
            and self.artifact_accepted is self.expected_artifact_accepted
            and self.resume_behavior == self.expected_resume_behavior
            and self.lock_released is True
            else "FAIL"
        )
        if self.status != derived_status:
            raise ValueError("fault record status contradicts its observed behavior")
        return self


class FaultInjectionEvidence(StrictContract):
    schema_version: Literal[1]
    stage: Literal["cache_scheduler_faults"]
    source_implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    records: list[FaultInjectionRecord]
    expected_fault_count: Literal[30]
    offline_network_attempt_count: int = Field(ge=0)

    @model_validator(mode="after")
    def complete_fault_matrix(self) -> FaultInjectionEvidence:
        names = [record.fault_name for record in self.records]
        if names != list(CANONICAL_FAULT_NAMES):
            raise ValueError("fault evidence must use the exact canonical 30-case matrix")
        if any(record.status != "PASS" for record in self.records):
            raise ValueError("fault evidence manifest requires every case to derive PASS")
        if self.offline_network_attempt_count != 1:
            raise ValueError(
                "fault evidence requires exactly one intentionally denied network attempt"
            )
        return self


def validate_fault_evidence_binding(
    evidence: FaultInjectionEvidence,
    *,
    inventory_source_implementation_commit: str,
    inventory_fault_evidence_sha256: str,
    observed_fault_evidence_sha256: str,
) -> None:
    """Bind a valid fault matrix to the inventory's source commit and whole-file digest."""

    if evidence.source_implementation_commit != inventory_source_implementation_commit:
        raise ValueError("fault evidence source commit differs from the inventory")
    if inventory_fault_evidence_sha256 != observed_fault_evidence_sha256:
        raise ValueError("fault evidence whole-file digest differs from the inventory")


def finite_nonnegative(value: float) -> float:
    """Small shared guard for measurements populated from external telemetry."""

    if not math.isfinite(value) or value < 0:
        raise ValueError("resource values must be finite and nonnegative")
    return value
