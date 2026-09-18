"""Authoritative strict contracts for machine-readable repository evidence."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .cache.contracts import (
    CacheArtifactManifest,
    CacheIdentity,
    CacheSchedulerConfig,
    CacheSchedulerInventory,
    CompletionMarker,
    FaultInjectionEvidence,
    FaultInjectionRecord,
    ProbeResourceSummary,
    ProbeRunSummary,
    ProtectedHashComparison,
    ProtectedHashRecord,
    SchedulerProbeEvidence,
)
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
from .experiments.contracts import (
    ConditionResource,
    MetricRecord,
    PairedMetricRecord,
    PredictionFileRecord,
    ProtectedFoundationHashComparison,
    ProtectedSplitValidation,
    ResumeVerification,
    RuntimeEstimate,
    SmokeCondition,
    SmokeEvidenceInventory,
    SmokePlan,
    SmokeRunReport,
    SmokeValidationReport,
)
from .experiments.pilot_contracts import (
    DecisionPolicy,
    MetricPolicy,
    PilotConditionInventory,
    PilotDatasetSelection,
    PilotProtocolArtifact,
    PilotProtocolValidation,
    PilotStagedSchedule,
    PlannedCondition,
)
from .experiments.planning import SmokeConfig
from .models.adapters.contracts import (
    AdapterLeakageEvidenceManifest,
    ModelAdapterInventory,
    PredictionResult,
)
from .runner.contracts import (
    FailureRecord as SchedulerFailureRecord,
)
from .runner.contracts import (
    ResourceRecord as SchedulerResourceRecord,
)
from .runner.contracts import (
    RunManifest,
    RunPlan,
    SchedulerState,
    TaskAttempt,
    TaskResult,
    TaskSpec,
    TaskStateRecord,
    TaskTransition,
)
from .splits.contracts import SplitGenerationInventoryContract
from .transformations.contracts import (
    TransformationCacheManifest,
    TransformationCertificate,
    TransformationInventory,
    TransformationManifest,
    TransformationPropertyEvidence,
    TransformationValidationReport,
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


class DataFoundationBaselineDatasetContract(ArtifactContract):
    openml_data_id: int = Field(gt=0)
    openml_file_id: int = Field(gt=0)
    raw_source_sha256: Sha256
    processed_artifact_sha256: dict[str, Sha256]
    row_count: int = Field(gt=0)
    predictor_count: int = Field(gt=0)
    target_name: str
    accepted_class_values: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_artifact_set(self) -> DataFoundationBaselineDatasetContract:
        expected = {"features.parquet", "targets.parquet", "schema.json", "label_mapping.json"}
        if set(self.processed_artifact_sha256) != expected:
            raise ValueError("baseline must identify the four frozen processed artifacts")
        return self


class DataFoundationBaselineSplitContract(ArtifactContract):
    name: Literal["stratified_group_5fold_v1"]
    group_by: Literal["predictors"]
    no_cross_split_duplicate_invariant: Literal[True] = True
    cross_split_duplicate_group_count: int = Field(ge=0)
    duplicate_predictor_group_count: int = Field(ge=0)
    conflicting_target_group_count: int = Field(ge=0)
    split_sizes: dict[str, int]


class DataFoundationBaselineContract(ArtifactContract):
    schema_version: Literal[1] = 1
    baseline_name: Literal["data_foundation"] = "data_foundation"
    dataset: DataFoundationBaselineDatasetContract
    split_protocol: DataFoundationBaselineSplitContract


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
    process_exit: Literal["success", "exception", "timeout", "oom", "not_executed"] = "success"
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

    @model_validator(mode="after")
    def validate_report_consistency(self) -> GpuCapacityReportContract:
        if self.status != "PASS":
            return self
        if self.requested_profiles <= 0:
            raise ValueError("a passing GPU report requires requested profiles")
        if self.accounted_profiles != self.requested_profiles:
            raise ValueError("GPU profile accounting is incomplete")
        required_models = {"TPFN3-8.5", "TICL2-2.2"}
        covered_models = {row.model_id for row in self.rows}
        if not required_models.issubset(covered_models):
            raise ValueError("every foundation model requires GPU evidence")
        if any(row.status in {"FAIL", "BLOCKED", "NOT_EXECUTED"} for row in self.rows):
            raise ValueError("a passing GPU report cannot contain an unsuccessful profile")
        launched = [row for row in self.rows if row.status != "NOT_EXECUTED"]
        if not launched or any(row.monitoring_complete is not True for row in launched):
            raise ValueError("every launched GPU profile requires completed monitoring")
        for row in self.rows:
            if row.process_exit == "success" and row.status in {"FAIL", "BLOCKED"}:
                raise ValueError("failed profiles must classify their process exit")
            if row.status == "NOT_EXECUTED" and row.process_exit != "not_executed":
                raise ValueError("unexecuted profiles must classify their process exit")
            if row.timed_out and row.process_exit != "timeout":
                raise ValueError("timed-out profiles must classify their process exit")
            if row.strategy == "cpu_fallback" and not row.fallback_status:
                raise ValueError("CPU fallback profiles require an explicit fallback status")
            if row.device == "cuda" and row.status == "PASS":
                if row.peak_vram_reserved_mib is None:
                    raise ValueError("successful GPU profiles require peak reserved VRAM")
                if row.peak_vram_reserved_mib > self.gpu_soft_limit_mib:
                    raise ValueError("FAIL_GPU_SOFT_LIMIT: GPU soft limit exceeded")
            if row.failure_category == "FAIL_GPU_SOFT_LIMIT" and row.status == "PASS":
                raise ValueError("a soft-limit failure cannot be marked PASS")
        return self


class GateCheckContract(ArtifactContract):
    gate_id: str = Field(pattern=r"^R[0-9]{2}$")
    requirement: str
    result: Literal["PASS", "FAIL", "NOT_VERIFIED", "NOT_APPLICABLE_LOCAL_ARTIFACTS_ABSENT"]
    evidence: str


class RepositoryValidationReportContract(ArtifactContract):
    schema_version: Literal[1] = 1
    stage: Literal["repository_repair"] = "repository_repair"
    status: Literal["PASS", "FAIL", "BLOCKED"]
    checks: list[GateCheckContract]


SCHEMA_CONTRACTS: dict[str, type[BaseModel]] = {
    "condition_manifest.schema.json": ConditionManifestContract,
    "smoke_condition.schema.json": SmokeCondition,
    "smoke_config.schema.json": SmokeConfig,
    "smoke_plan.schema.json": SmokePlan,
    "smoke_prediction_file.schema.json": PredictionFileRecord,
    "smoke_metric.schema.json": MetricRecord,
    "smoke_runtime_estimate.schema.json": RuntimeEstimate,
    "smoke_paired_metric.schema.json": PairedMetricRecord,
    "smoke_condition_resource.schema.json": ConditionResource,
    "smoke_protected_foundation_hash_comparison.schema.json": ProtectedFoundationHashComparison,
    "smoke_protected_split_validation.schema.json": ProtectedSplitValidation,
    "smoke_resume_verification.schema.json": ResumeVerification,
    "smoke_run_report.schema.json": SmokeRunReport,
    "smoke_evidence_inventory.schema.json": SmokeEvidenceInventory,
    "smoke_validation_report.schema.json": SmokeValidationReport,
    "pilot_protocol.schema.json": PilotProtocolArtifact,
    "pilot_dataset_selection.schema.json": PilotDatasetSelection,
    "pilot_condition.schema.json": PlannedCondition,
    "pilot_condition_inventory.schema.json": PilotConditionInventory,
    "pilot_staged_schedule.schema.json": PilotStagedSchedule,
    "pilot_metric_policy.schema.json": MetricPolicy,
    "pilot_decision_policy.schema.json": DecisionPolicy,
    "pilot_protocol_validation.schema.json": PilotProtocolValidation,
    "cache_artifact_manifest.schema.json": CacheArtifactManifest,
    "cache_completion_marker.schema.json": CompletionMarker,
    "cache_identity.schema.json": CacheIdentity,
    "cache_scheduler_config.schema.json": CacheSchedulerConfig,
    "cache_scheduler_inventory.schema.json": CacheSchedulerInventory,
    "cache_scheduler_fault_evidence.schema.json": FaultInjectionEvidence,
    "cache_scheduler_fault_record.schema.json": FaultInjectionRecord,
    "cache_scheduler_protected_hash_comparison.schema.json": ProtectedHashComparison,
    "cache_scheduler_protected_hash_record.schema.json": ProtectedHashRecord,
    "cache_scheduler_probe_evidence.schema.json": SchedulerProbeEvidence,
    "cache_scheduler_probe_run.schema.json": ProbeRunSummary,
    "cache_scheduler_probe_resource.schema.json": ProbeResourceSummary,
    "data_foundation_baseline.schema.json": DataFoundationBaselineContract,
    "model_compatibility.schema.json": ModelCompatibilityReportContract,
    "model_adapter_inventory.schema.json": ModelAdapterInventory,
    "model_adapter_leakage_evidence.schema.json": AdapterLeakageEvidenceManifest,
    "model_adapter_result.schema.json": PredictionResult,
    "dataset_inventory.schema.json": DatasetRegistryReportContract,
    "gpu_capacity_report.schema.json": GpuCapacityReportContract,
    "repository_validation.schema.json": RepositoryValidationReportContract,
    "split_generation_inventory.schema.json": SplitGenerationInventoryContract,
    "scheduler_run_manifest.schema.json": RunManifest,
    "scheduler_run_plan.schema.json": RunPlan,
    "scheduler_state.schema.json": SchedulerState,
    "scheduler_failure_record.schema.json": SchedulerFailureRecord,
    "scheduler_resource_record.schema.json": SchedulerResourceRecord,
    "scheduler_task_attempt.schema.json": TaskAttempt,
    "scheduler_task_result.schema.json": TaskResult,
    "scheduler_task_spec.schema.json": TaskSpec,
    "scheduler_task_transition.schema.json": TaskTransition,
    "scheduler_task_record.schema.json": TaskStateRecord,
    "transformation_certificate.schema.json": TransformationCertificate,
    "transformation_cache_manifest.schema.json": TransformationCacheManifest,
    "transformation_manifest.schema.json": TransformationManifest,
    "transformation_property_evidence.schema.json": TransformationPropertyEvidence,
    "transformation_inventory.schema.json": TransformationInventory,
    "transformation_validation.schema.json": TransformationValidationReport,
}


def schema_documents() -> dict[str, dict[str, Any]]:
    """Return deterministic JSON-schema documents for all tracked contracts."""

    return {
        filename: contract.model_json_schema(by_alias=True)
        for filename, contract in SCHEMA_CONTRACTS.items()
    }


__all__ = [
    "ConditionManifestContract",
    "SmokeCondition",
    "SmokePlan",
    "SmokeEvidenceInventory",
    "SmokeRunReport",
    "SmokeValidationReport",
    "DataFoundationBaselineContract",
    "DatasetRegistryReportContract",
    "GpuCapacityReportContract",
    "ModelCompatibilityReportContract",
    "ModelAdapterInventory",
    "PredictionResult",
    "RepositoryValidationReportContract",
    "SplitGenerationInventoryContract",
    "SCHEMA_CONTRACTS",
    "schema_documents",
]
