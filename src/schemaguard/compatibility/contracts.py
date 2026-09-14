"""Strict evidence contracts used by the model compatibility gate."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FailureCategory = Literal[
    "PASS",
    "PASS_WITH_CPU_FALLBACK",
    "FAIL_VERSION_MISMATCH",
    "FAIL_IMPORT",
    "FAIL_CONSTRUCTOR",
    "BLOCKED_API_MISMATCH",
    "BLOCKED_CHECKPOINT_AUTHORIZATION",
    "FAIL_CHECKPOINT_DOWNLOAD",
    "FAIL_CHECKPOINT_HASH",
    "FAIL_CACHE_CORRUPTION",
    "FAIL_CPU_INFERENCE",
    "FAIL_GPU_INFERENCE",
    "FAIL_GPU_OOM",
    "FAIL_TIMEOUT",
    "FAIL_INVALID_PROBABILITY",
    "FAIL_CLASS_ORDER",
    "FAIL_NONDETERMINISM",
    "FAIL_RESOURCE_MONITOR",
    "FAIL_DATA_FOUNDATION_MUTATION",
    "FAIL_TEST",
    "BLOCKED_UNEXPECTED_WORKTREE",
    "NOT_EXECUTED_NO_CUDA",
    "NOT_EXECUTED_UNSAFE_VRAM",
    "NOT_EXECUTED_OFFLINE_CACHE_MISS",
]

ProbeStatus = Literal["PASS", "PASS_WITH_CPU_FALLBACK", "FAIL", "BLOCKED", "NOT_EXECUTED"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PackageRecord(StrictModel):
    name: str
    expected_version: str
    observed_version: str | None = None
    import_name: str
    installed: bool
    status: Literal["PASS", "FAIL", "BLOCKED"]
    error: str | None = None


class DeviceRecord(StrictModel):
    device: Literal["cpu", "cuda"]
    available: bool
    name: str | None = None
    total_memory_mib: float | None = None
    free_memory_mib: float | None = None
    driver: str | None = None
    cuda_visible: bool | None = None
    reason: str | None = None


class CheckpointRecord(StrictModel):
    model_id: str
    identifier: str | None = None
    resolved_path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    cache_key: str | None = None
    cache_status: str
    identity_valid: bool
    authorization_status: str
    license_status: str
    error: str | None = None


class LicenseRecord(StrictModel):
    model_id: str
    package: str
    checkpoint: str | None = None
    authorization_status: str
    license_status: str
    terms_accepted_automatically: bool = False
    evidence: str | None = None


class ProbeRequest(StrictModel):
    schema_version: int = 1
    model_id: str
    device: Literal["cpu", "cuda"]
    config_path: str
    allow_network: bool = False
    offline: bool = True
    seed: int
    output_path: str
    timeout_seconds: int = Field(gt=0)


class ResourceRecord(StrictModel):
    model_id: str
    device: Literal["cpu", "cuda"]
    runtime_seconds: float = Field(ge=0)
    cpu_time_seconds: float = Field(ge=0)
    peak_ram_mib: float | None = Field(default=None, ge=0)
    child_peak_ram_mib: float | None = Field(default=None, ge=0)
    gpu_allocated_mib: float | None = Field(default=None, ge=0)
    gpu_reserved_mib: float | None = Field(default=None, ge=0)
    gpu_total_mib_before: float | None = Field(default=None, ge=0)
    gpu_free_mib_before: float | None = Field(default=None, ge=0)
    gpu_total_mib_after: float | None = Field(default=None, ge=0)
    gpu_free_mib_after: float | None = Field(default=None, ge=0)
    timed_out: bool = False
    termination_signal: str | None = None
    oom: bool = False
    monitoring_error: str | None = None


class FailureRecord(StrictModel):
    category: FailureCategory
    message: str
    traceback_path: str | None = None
    exception_type: str | None = None
    retryable: bool = False


class EnvironmentReport(StrictModel):
    schema_version: int = 1
    python_executable: str
    python_version: str
    conda_environment: str | None = None
    operating_system: str
    cpu: str
    logical_cpu_count: int = Field(ge=1)
    physical_cpu_count: int | None = Field(default=None, ge=1)
    installed_ram_mib: float | None = Field(default=None, ge=0)
    available_ram_mib: float | None = Field(default=None, ge=0)
    gpu: DeviceRecord
    pytorch_version: str | None = None
    pytorch_cuda_build: str | None = None
    cuda_visible: bool
    relevant_environment: dict[str, str]
    packages: list[PackageRecord]


class ProbeResult(StrictModel):
    schema_version: int = 1
    model_id: str
    package_name: str
    expected_version: str
    observed_version: str | None = None
    device: Literal["cpu", "cuda"]
    seed: int
    fixture_hash: str
    parameter_hash: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    git_commit: str
    start_time: str
    end_time: str
    runtime_seconds: float = Field(ge=0)
    peak_ram_mib: float | None = Field(default=None, ge=0)
    peak_vram_mib: float | None = Field(default=None, ge=0)
    prediction_shape: list[int]
    class_order: list[int | str]
    maximum_probability_sum_error: float | None = Field(default=None, ge=0)
    maximum_repeated_run_difference: float | None = Field(default=None, ge=0)
    status: ProbeStatus
    failure_category: FailureCategory
    full_traceback_path: str | None = None
    checkpoint_record: CheckpointRecord | None = None
    resource_record: ResourceRecord | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class PhaseResult(StrictModel):
    schema_version: int = 2
    stage: Literal["model_compatibility"] = "model_compatibility"
    status: Literal["PASS", "FAIL", "BLOCKED"]
    starting_commit: str
    branch: str
    environment: EnvironmentReport
    package_records: list[PackageRecord]
    checkpoint_records: list[CheckpointRecord]
    license_records: list[LicenseRecord]
    probes: list[ProbeResult]
    resources: list[ResourceRecord]
    acceptance_gates: dict[str, str]
    data_foundation_hashes_unchanged: bool
    failures: list[FailureRecord]
