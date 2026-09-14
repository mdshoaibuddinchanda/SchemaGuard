"""Authoritative strict contracts for machine-readable repository evidence."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .compatibility.contracts import (
    CheckpointRecord,
    EnvironmentReport,
    FailureRecord,
    LicenseRecord,
    PackageRecord,
    ProbeResult,
    ResourceRecord,
)
from .data.contracts import (
    DatasetInventoryContract,
    DatasetValidationRecordContract,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ArtifactContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConditionRecordContract(ArtifactContract):
    dataset_id: int = Field(gt=0)
    seed: int
    model_id: str
    view_id: str
    status: Literal["scheduled", "completed", "failed"] = "scheduled"
    reason: str | None = None


class ConditionManifestContract(ArtifactContract):
    schema_version: Literal[2] = 2
    stage: Literal["condition_manifest"] = "condition_manifest"
    scope: Literal["pilot", "main"]
    benchmark: Literal["SchemaOrbit-14"]
    config_checksum: Sha256
    scheduled_count: int = Field(gt=0)
    records: list[ConditionRecordContract]

    @model_validator(mode="after")
    def validate_count(self) -> ConditionManifestContract:
        if self.scheduled_count != len(self.records):
            raise ValueError("scheduled_count must equal the number of records")
        return self


class ModelCompatibilityReportContract(ArtifactContract):
    schema_version: Literal[2] = 2
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


class DatasetRegistryReportContract(DatasetInventoryContract):
    stage: Literal["dataset_registry"] = "dataset_registry"
    status: Literal["PASS", "FAIL", "BLOCKED"]
    validation_records: list[DatasetValidationRecordContract] = Field(default_factory=list)


class GpuStateContract(ArtifactContract):
    available: bool
    name: str | None = None
    total_mib: float | None = Field(default=None, ge=0)
    free_mib: float | None = Field(default=None, ge=0)
    torch_version: str | None = None
    cuda_build: str | None = None
    capability: list[int] | None = None
    reason: str | None = None
    driver: str | None = None


class GpuProfileSpecContract(ArtifactContract):
    profile_id: str
    source_dataset_id: int = Field(gt=0)
    rows: int = Field(gt=0)
    predictors: int = Field(gt=0)
    classes: int = Field(ge=2)
    train_rows: int = Field(gt=0)
    test_rows: int = Field(gt=0)


class GpuProfileRecordContract(ArtifactContract):
    schema_version: int = Field(ge=2)
    model_id: str
    profile_id: str
    device: Literal["cpu", "cuda"]
    strategy: Literal[
        "repeated_inference_single_worker",
        "fresh_worker_per_inference",
        "cpu_fallback",
    ]
    seed: int
    fixture_hash: Sha256 | None = None
    parameter_hash: Sha256 | None = None
    checkpoint_sha256: Sha256 | None = None
    cycles: int = Field(gt=0)
    status: Literal["PASS", "PASS_WITH_CPU_FALLBACK", "FAIL", "BLOCKED", "NOT_EXECUTED"]
    failure_category: str | None = None
    start_time: str = ""
    end_time: str | None = None
    runtime_seconds: float = Field(ge=0)
    load_seconds: float | None = Field(default=None, ge=0)
    fit_seconds: float | None = Field(default=None, ge=0)
    predict_seconds: list[float] = Field(default_factory=list)
    prediction_shape: list[int]
    class_order: list[int | str]
    probability_sum_error: float | None = Field(default=None, ge=0)
    repeat_max_abs_diff: float | None = Field(default=None, ge=0)
    prediction_hash: Sha256 | None = None
    peak_ram_mib: float | None = Field(default=None, ge=0)
    child_peak_ram_mib: float | None = Field(default=None, ge=0)
    peak_vram_allocated_mib: float | None = Field(default=None, ge=0)
    peak_vram_reserved_mib: float | None = Field(default=None, ge=0)
    gpu_free_before_mib: float | None = Field(default=None, ge=0)
    gpu_free_after_mib: float | None = Field(default=None, ge=0)
    monitoring_complete: bool | None = None
    monitoring_error: str | None = None
    timed_out: bool = False
    oom: bool = False
    execution_policy: str | None = None
    fallback_status: str | None = None
    gpu_soft_limit_mib: float | None = Field(default=None, ge=0)
    error: str | None = None
    traceback: str | None = None
    post_cleanup_ram_mib: float | None = Field(default=None, ge=0)
    post_cleanup_gpu_allocated_mib: float | None = Field(default=None, ge=0)
    post_cleanup_gpu_reserved_mib: float | None = Field(default=None, ge=0)


class OptimizationRecordContract(ArtifactContract):
    schema_version: int = Field(ge=2)
    model_id: str
    profile_id: str
    baseline_strategy: str
    candidate_strategy: str
    baseline_prediction_hash: Sha256 | None
    candidate_prediction_hashes: list[Sha256]
    prediction_equivalent: bool
    accepted: bool
    decision: str
    selected_strategy: str = "fresh_worker_per_inference"


class GpuCapacityReportContract(ArtifactContract):
    schema_version: Literal[2] = 2
    stage: Literal["gpu_capacity"] = "gpu_capacity"
    status: Literal["PASS", "FAIL", "BLOCKED"]
    gpu_state: GpuStateContract
    gpu_headroom_mib: float = Field(ge=0)
    gpu_soft_limit_mib: float = Field(gt=0)
    foundation_models_serialized: Literal[True] = True
    profiles: list[GpuProfileSpecContract]
    checkpoint_sha256: dict[str, Sha256]
    registry_sha256: Sha256
    runtime_config_sha256: Sha256
    profile_count: int = Field(ge=1)
    rows: list[GpuProfileRecordContract]
    optimization: list[OptimizationRecordContract]
    fallback_policy: str
    requested_profiles: int = Field(ge=1)
    accounted_profiles: int = Field(ge=0)
    monitoring_complete: bool


class GateCheckContract(ArtifactContract):
    gate_id: str = Field(pattern=r"^R[0-9]{2}$")
    requirement: str
    result: Literal["PASS", "FAIL", "NOT_VERIFIED"]
    evidence: str


class RepositoryValidationReportContract(ArtifactContract):
    schema_version: Literal[1] = 1
    stage: Literal["repository_repair"] = "repository_repair"
    status: Literal["PASS", "FAIL", "BLOCKED"]
    checks: list[GateCheckContract]


SCHEMA_CONTRACTS: dict[str, type[BaseModel]] = {
    "condition_manifest.schema.json": ConditionManifestContract,
    "model_compatibility.schema.json": ModelCompatibilityReportContract,
    "dataset_inventory.schema.json": DatasetRegistryReportContract,
    "gpu_capacity_report.schema.json": GpuCapacityReportContract,
    "repository_validation.schema.json": RepositoryValidationReportContract,
}


def schema_documents() -> dict[str, dict[str, Any]]:
    """Return deterministic JSON-schema documents for all tracked contracts."""

    return {
        filename: contract.model_json_schema(by_alias=True)
        for filename, contract in SCHEMA_CONTRACTS.items()
    }


__all__ = [
    "ConditionManifestContract",
    "DatasetRegistryReportContract",
    "GpuCapacityReportContract",
    "ModelCompatibilityReportContract",
    "RepositoryValidationReportContract",
    "SCHEMA_CONTRACTS",
    "schema_documents",
]
