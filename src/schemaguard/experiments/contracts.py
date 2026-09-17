"""Strict contracts for the ten-condition SchemaGuard smoke experiment."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..utils.hashing import sha256_canonical_json

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
ModelId = Literal["LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2"]
ViewId = Literal["V00", "V01"]
Device = Literal["cpu", "cuda"]
Partition = Literal["calibration", "test"]
AllPartition = Literal["train", "calibration", "test"]
SmokeRunStatus = Literal["PASS_PENDING_REVIEW", "PARTIAL_VALID", "REPAIR_REQUIRED", "BLOCKED"]
EvaluationSource = Literal["computed_after_predictions", "reused_cold_run", "not_performed"]
SmokeConditionStatus = Literal["PASS", "FAIL", "BLOCKED"]
GateResult = Literal["PASS", "FAIL", "NOT_VERIFIED"]
MetricName = Literal[
    "brier_score",
    "log_loss",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "expected_calibration_error",
]
RegistryParameterDecision = Literal[
    "catboost_yaml_no_scalar", "tabicl_checkpoint_version_from_checkpoint_field"
]

MODEL_ORDER: tuple[ModelId, ...] = (
    "LR-1.9",
    "CAT-1.2",
    "XGB-3.4",
    "TPFN3-8.5",
    "TICL2-2.2",
)
VIEW_ORDER: tuple[ViewId, ...] = ("V00", "V01")
FOUNDATION_MODELS = frozenset({"TPFN3-8.5", "TICL2-2.2"})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)


class SmokeEnvironment(StrictModel):
    os_platform: str
    python_version: str
    conda_environment: str | None
    cpu_identifier: str
    cpu_physical_cores: int | None = Field(default=None, ge=1)
    cpu_logical_cores: int | None = Field(default=None, ge=1)
    ram_total_mib: float | None = Field(default=None, gt=0)
    ram_available_mib: float | None = Field(default=None, ge=0)
    gpu_name: str | None
    gpu_total_mib: float | None = Field(default=None, gt=0)
    gpu_free_mib: float | None = Field(default=None, ge=0)
    gpu_driver: str | None
    torch_version: str | None
    torch_cuda_build: str | None
    cuda_visible: bool
    runtime_package_versions: dict[str, str]
    telemetry_warnings: list[str]


class SmokeGateRecord(StrictModel):
    gate_id: str = Field(pattern=r"^S[0-9]{2}$")
    result: GateResult
    evidence_sha256: Sha256
    detail: str


class ProtectedFoundationArtifactHash(StrictModel):
    path: str = Field(pattern=r"^(data/raw|data/processed|data/splits)/[A-Za-z0-9_./-]+$")
    before_sha256: Sha256
    after_sha256: Sha256
    before_size_bytes: int = Field(ge=0)
    after_size_bytes: int = Field(ge=0)
    unchanged: bool

    @model_validator(mode="after")
    def validate_comparison(self) -> ProtectedFoundationArtifactHash:
        posix = PurePosixPath(self.path)
        windows = PureWindowsPath(self.path)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
            raise ValueError("protected foundation artifact path must remain repository-relative")
        observed = (
            self.before_sha256 == self.after_sha256
            and self.before_size_bytes == self.after_size_bytes
        )
        if self.unchanged is not observed:
            raise ValueError("protected foundation status contradicts its hashes and sizes")
        return self


class ProtectedFoundationHashComparison(StrictModel):
    schema_version: Literal[1] = 1
    plan_sha256: Sha256
    baseline_snapshot_sha256: Sha256
    status: Literal["PASS", "FAIL"]
    expected_artifact_count: Literal[13] = 13
    artifacts: list[ProtectedFoundationArtifactHash] = Field(min_length=13, max_length=13)

    @model_validator(mode="after")
    def validate_artifact_set(self) -> ProtectedFoundationHashComparison:
        paths = [item.path for item in self.artifacts]
        if paths != sorted(set(paths)):
            raise ValueError("protected foundation hash artifacts must be unique and sorted")
        observed_status = "PASS" if all(item.unchanged for item in self.artifacts) else "FAIL"
        if self.status != observed_status:
            raise ValueError("protected foundation comparison status contradicts artifact records")
        return self


class ProtectedSplitValidation(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["PASS", "FAIL", "NOT_RUN"]
    strategy: Literal["stratified_group_5fold_v1"]
    assignment_sha256: Sha256
    row_count: int | None = Field(default=None, ge=0)
    partition_counts: dict[str, int] | None = None
    class_counts_by_partition: dict[str, dict[str, int]] | None = None
    predictor_group_count: int | None = Field(default=None, ge=0)
    duplicate_predictor_group_count: int | None = Field(default=None, ge=0)
    conflicting_target_group_count: int | None = Field(default=None, ge=0)
    crossing_predictor_group_count: int | None = Field(default=None, ge=0)
    deprecated_row_stratified_split_selected: Literal[False] = False
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_protected_split(self) -> ProtectedSplitValidation:
        if self.status == "NOT_RUN":
            if not self.failure_reason:
                raise ValueError("unrun protected split validation requires an explicit reason")
            return self
        if self.status == "FAIL":
            if not self.failure_reason:
                raise ValueError("failed protected split validation requires an explicit reason")
            return self
        if self.failure_reason is not None:
            raise ValueError("passing protected split validation cannot contain a failure reason")
        if self.status != "PASS":
            return self
        expected: dict[str, Any] = {
            "row_count": 748,
            "partition_counts": {"train": 449, "calibration": 150, "test": 149},
            "predictor_group_count": 502,
            "duplicate_predictor_group_count": 69,
            "conflicting_target_group_count": 31,
            "crossing_predictor_group_count": 0,
        }
        observed = {name: getattr(self, name) for name in expected}
        if observed != expected:
            raise ValueError("protected grouped-split evidence differs from the frozen baseline")
        if self.class_counts_by_partition is None or set(self.class_counts_by_partition) != {
            "train",
            "calibration",
            "test",
        }:
            raise ValueError("Phase 01 class counts must cover all three partitions")
        if any(
            set(counts) != {"0", "1"}
            or any(count <= 0 for count in counts.values())
            or sum(counts.values()) != expected["partition_counts"][part]
            for part, counts in self.class_counts_by_partition.items()
        ):
            raise ValueError("Phase 01 partition class counts are incomplete")
        return self


class ResumeVerification(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["PASS", "FAIL"]
    plan_sha256: Sha256
    before_hashes: dict[str, Sha256]
    after_hashes: dict[str, Sha256]
    before_mtime_ns: dict[str, int] = Field()
    after_mtime_ns: dict[str, int] = Field()
    cold_run_report_sha256: Sha256
    planned_conditions: Literal[10] = 10
    executed_conditions: int = Field(ge=0, le=10)
    validated_cache_hits: int = Field(ge=0, le=10)
    failed_conditions: int = Field(ge=0, le=10)
    blocked_conditions: int = Field(ge=0, le=10)
    network_attempt_count: int = Field(ge=0)
    duplicate_prediction_artifacts: int = Field(ge=0)
    result_rewrites: int = Field(ge=0)
    prediction_hashes_unchanged: bool
    prediction_mtimes_unchanged: bool

    @model_validator(mode="after")
    def validate_no_rewrites(self) -> ResumeVerification:
        if set(self.before_hashes) != set(self.after_hashes):
            raise ValueError("resume hash manifests must identify the same file paths")
        if self.prediction_hashes_unchanged is not (self.before_hashes == self.after_hashes):
            raise ValueError("resume hash status contradicts the before/after manifests")
        if set(self.before_mtime_ns) != set(self.after_mtime_ns):
            raise ValueError("resume mtime manifests must identify the same files")
        if self.prediction_mtimes_unchanged is not (self.before_mtime_ns == self.after_mtime_ns):
            raise ValueError("resume mtime status contradicts the before/after manifests")
        if (
            self.executed_conditions
            + self.validated_cache_hits
            + self.failed_conditions
            + self.blocked_conditions
            != 10
        ):
            raise ValueError("resume result must account for all ten tasks")
        is_pass = (
            self.executed_conditions == 0
            and self.validated_cache_hits == 10
            and self.failed_conditions == 0
            and self.blocked_conditions == 0
            and self.network_attempt_count == 0
            and self.duplicate_prediction_artifacts == 0
            and self.result_rewrites == 0
            and self.prediction_hashes_unchanged
            and self.prediction_mtimes_unchanged
        )
        if (self.status == "PASS") is not is_pass:
            raise ValueError("resume status contradicts cache, network, or output integrity")
        return self


def condition_identity(
    *,
    dataset_features_sha256: str,
    target_artifact_sha256: str,
    model_id: str,
    package_version: str,
    view_id: str,
    device: str,
    precision_policy: str,
    seed: int,
    model_spec_sha256: str,
    parameters_sha256: str,
    view_certificate_sha256: str,
    view_features_sha256: str,
    split_sha256: str,
    dependency_lock_sha256: str,
    source_implementation_sha256: str,
    checkpoint_sha256: str | None,
) -> str:
    return sha256_canonical_json(
        {
            "schema_version": 1,
            "artifact_kind": "prediction",
            "dataset_features_sha256": dataset_features_sha256,
            "target_artifact_sha256": target_artifact_sha256,
            "model_id": model_id,
            "package_version": package_version,
            "view_id": view_id,
            "device": device,
            "precision_policy": precision_policy,
            "seed": seed,
            "model_spec_sha256": model_spec_sha256,
            "parameters_sha256": parameters_sha256,
            "view_certificate_sha256": view_certificate_sha256,
            "view_features_sha256": view_features_sha256,
            "split_sha256": split_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
            "source_implementation_sha256": source_implementation_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )


class SmokeView(StrictModel):
    view_id: ViewId
    view_name: Literal["identity", "numeric_affine_units"]
    source_hash: Sha256
    output_hash: Sha256
    certificate_sha256: Sha256
    certificate_ids: list[Sha256] = Field(min_length=3, max_length=3)
    view_features_sha256: Sha256
    partition_feature_sha256: dict[AllPartition, Sha256]
    partition_feature_file_sha256: dict[AllPartition, Sha256]
    partition_certificate_file_sha256: dict[AllPartition, Sha256 | None]
    manifest_sha256: Sha256 | None
    partition_row_id_sha256: dict[AllPartition, Sha256]
    partition_rows: dict[AllPartition, int]

    @model_validator(mode="after")
    def validate_view(self) -> SmokeView:
        expected_name = {"V00": "identity", "V01": "numeric_affine_units"}[self.view_id]
        if self.view_name != expected_name:
            raise ValueError("view ID and frozen semantic name disagree")
        if set(self.partition_row_id_sha256) != {"train", "calibration", "test"}:
            raise ValueError("view must identify all train/calibration/test row sets")
        if set(self.partition_rows) != {"train", "calibration", "test"}:
            raise ValueError("view must record all train/calibration/test row counts")
        if set(self.partition_feature_sha256) != {"train", "calibration", "test"}:
            raise ValueError("view must hash all train/calibration/test feature matrices")
        if set(self.partition_feature_file_sha256) != {"train", "calibration", "test"}:
            raise ValueError("view must identify all train/calibration/test feature files")
        if set(self.partition_certificate_file_sha256) != {"train", "calibration", "test"}:
            raise ValueError("view must identify every partition certificate artifact")
        return self


class SmokeCondition(StrictModel):
    condition_id: Sha256
    task_name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    model_id: ModelId
    package_name: str
    package_version: str
    dataset_features_sha256: Sha256
    target_artifact_sha256: Sha256
    view_id: ViewId
    device: Device
    precision_policy: Literal["native", "auto"]
    seed: int = Field(ge=0)
    model_spec_sha256: Sha256
    parameters_sha256: Sha256
    model_parameters: dict[str, Any]
    preprocessing: str
    checkpoint_identifier: str | None
    checkpoint_sha256: Sha256 | None
    view_certificate_sha256: Sha256
    view_features_sha256: Sha256
    split_sha256: Sha256
    dependency_lock_sha256: Sha256
    source_implementation_sha256: Sha256

    @model_validator(mode="after")
    def validate_condition_identity(self) -> SmokeCondition:
        expected_device: Device = "cuda" if self.model_id in FOUNDATION_MODELS else "cpu"
        if self.device != expected_device:
            raise ValueError("device policy is not the frozen CPU/CUDA matrix")
        if self.parameters_sha256 != sha256_canonical_json(self.model_parameters):
            raise ValueError("parameter hash does not match frozen model parameters")
        expected_id = condition_identity(
            dataset_features_sha256=self.dataset_features_sha256,
            target_artifact_sha256=self.target_artifact_sha256,
            model_id=self.model_id,
            package_version=self.package_version,
            view_id=self.view_id,
            device=self.device,
            precision_policy=self.precision_policy,
            seed=self.seed,
            model_spec_sha256=self.model_spec_sha256,
            parameters_sha256=self.parameters_sha256,
            view_certificate_sha256=self.view_certificate_sha256,
            view_features_sha256=self.view_features_sha256,
            split_sha256=self.split_sha256,
            dependency_lock_sha256=self.dependency_lock_sha256,
            source_implementation_sha256=self.source_implementation_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
        )
        if self.condition_id != expected_id:
            raise ValueError("condition identity is not canonical")
        if self.task_name != f"{self.model_id}__{self.view_id}":
            raise ValueError("task name must identify exactly one frozen model/view pair")
        if (self.model_id in FOUNDATION_MODELS) != (self.checkpoint_sha256 is not None):
            raise ValueError("checkpoint identity presence differs from model family")
        return self


class SmokePlan(StrictModel):
    schema_version: Literal[1] = 1
    protocol_version: Literal["schema_guard_smoke_v1"]
    plan_name: Literal["schema_guard_smoke"] = "schema_guard_smoke"
    dataset_id: Literal[1464] = 1464
    dataset_version: Literal["openml_file_1586225"] = "openml_file_1586225"
    seed: Literal[1729] = 1729
    split_strategy: Literal["stratified_group_5fold_v1"] = "stratified_group_5fold_v1"
    source_commit: CommitSha
    code_identity_sha256: Sha256
    dependency_lock_sha256: Sha256
    dataset_features_sha256: Sha256
    dataset_source_sha256: Sha256
    target_artifact_sha256: Sha256
    assignment_sha256: Sha256
    assignment_logical_sha256: Sha256
    transformation_inventory_sha256: Sha256
    configuration_identity_sha256: Sha256
    output_path_policy_sha256: Sha256
    registry_parameter_decisions: list[RegistryParameterDecision] = Field(
        min_length=2, max_length=2
    )
    views: list[SmokeView] = Field(min_length=2, max_length=2)
    conditions: list[SmokeCondition] = Field(min_length=10, max_length=10)
    cpu_workers: Literal[2] = 2
    gpu_workers: Literal[1] = 1
    network_enabled: Literal[False] = False
    test_label_access_boundary: Literal["after_all_predictions_complete"]
    primary_metric: Literal["brier_score"]
    ece_binning_rule: Literal["equal_width_max_probability_confidence"]
    log_loss_epsilon: float = Field(gt=0, lt=0.01, allow_inf_nan=False)
    probability_sum_tolerance: float = Field(gt=0, le=1.0e-4, allow_inf_nan=False)
    reconstruction_rtol: float = Field(ge=0, le=1.0e-6, allow_inf_nan=False)
    reconstruction_atol: float = Field(ge=0, le=1.0e-6, allow_inf_nan=False)
    ece_bins: int = Field(ge=2, le=100)
    maximum_ram_mib: int = Field(gt=0, le=28672)
    maximum_vram_mib: int = Field(gt=0, le=3600)
    minimum_gpu_headroom_mib: int = Field(ge=512)
    cpu_timeout_seconds: int = Field(gt=0)
    cuda_timeout_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_frozen_matrix(self) -> SmokePlan:
        if self.registry_parameter_decisions != [
            "catboost_yaml_no_scalar",
            "tabicl_checkpoint_version_from_checkpoint_field",
        ]:
            raise ValueError("registry interpretation decisions differ from the reviewed exact set")
        if [view.view_id for view in self.views] != list(VIEW_ORDER):
            raise ValueError("smoke views must be ordered V00 then V01")
        expected_pairs = [(model, view) for model in MODEL_ORDER for view in VIEW_ORDER]
        observed_pairs = [(item.model_id, item.view_id) for item in self.conditions]
        if observed_pairs != expected_pairs:
            raise ValueError("smoke condition matrix/order must be exactly ten frozen pairs")
        if len({item.condition_id for item in self.conditions}) != 10:
            raise ValueError("smoke condition identities must be unique")
        if any(item.seed != self.seed for item in self.conditions):
            raise ValueError("condition seed differs from the frozen plan seed")
        if any(item.split_sha256 != self.assignment_sha256 for item in self.conditions):
            raise ValueError("condition split identity differs from the plan")
        if any(
            item.dataset_features_sha256 != self.dataset_features_sha256 for item in self.conditions
        ):
            raise ValueError("condition dataset identity differs from the plan")
        if any(
            item.target_artifact_sha256 != self.target_artifact_sha256 for item in self.conditions
        ):
            raise ValueError("condition target identity differs from the plan")
        if any(
            item.dependency_lock_sha256 != self.dependency_lock_sha256 for item in self.conditions
        ):
            raise ValueError("condition dependency identity differs from the plan")
        if any(
            item.source_implementation_sha256 != self.code_identity_sha256
            for item in self.conditions
        ):
            raise ValueError("condition source implementation differs from the plan")
        view_hashes = {item.view_id: item.certificate_sha256 for item in self.views}
        view_feature_hashes = {item.view_id: item.view_features_sha256 for item in self.views}
        if any(
            item.view_certificate_sha256 != view_hashes[item.view_id] for item in self.conditions
        ):
            raise ValueError("condition references an unplanned view certificate")
        if any(
            item.view_features_sha256 != view_feature_hashes[item.view_id]
            for item in self.conditions
        ):
            raise ValueError("condition references unplanned view feature content")
        return self

    @property
    def plan_sha256(self) -> str:
        return sha256_canonical_json(self.model_dump(mode="json"))


class PlanEnvelope(StrictModel):
    plan_sha256: Sha256
    plan: SmokePlan

    @model_validator(mode="after")
    def validate_plan_hash(self) -> PlanEnvelope:
        if self.plan_sha256 != self.plan.plan_sha256:
            raise ValueError("plan envelope hash does not match canonical plan")
        return self


class PredictionFileRecord(StrictModel):
    partition: Partition
    relative_path: str = Field(pattern=r"^results/smoke/predictions/[A-Za-z0-9_.-]+\.parquet$")
    sha256: Sha256
    row_count: int = Field(gt=0)
    row_id_sha256: Sha256
    probability_sha256: Sha256


class ConditionResource(StrictModel):
    wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    cpu_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    fit_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    prediction_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    peak_worker_rss_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_process_tree_rss_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_vram_allocated_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_vram_reserved_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    free_vram_before_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    telemetry_complete: bool
    cleanup_passed: bool
    adapter_resource_sha256: Sha256


class SmokeEvidenceCondition(StrictModel):
    condition_id: Sha256
    cache_identity_sha256: Sha256
    model_id: ModelId
    view_id: ViewId
    device: Device
    status: SmokeConditionStatus
    cache_status: Literal["created", "validated_cache_hit", "not_published"]
    prediction_hashes: list[Sha256]
    failure_category: str | None
    resource: ConditionResource | None


class AdapterResourceSummary(StrictModel):
    model_load_seconds: float = Field(ge=0, allow_inf_nan=False)
    preprocessing_fit_seconds: float = Field(ge=0, allow_inf_nan=False)
    fit_seconds: float = Field(ge=0, allow_inf_nan=False)
    prediction_seconds: float = Field(ge=0, allow_inf_nan=False)
    wall_time_seconds: float = Field(ge=0, allow_inf_nan=False)
    cpu_time_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_ram_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_process_tree_ram_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_vram_allocated_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_vram_reserved_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    free_vram_before_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    free_vram_after_mib: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    telemetry_complete: bool
    resource_limits_passed: bool
    cleanup_verified: bool


class PredictionOutputSummary(StrictModel):
    partition: Partition
    row_count: int = Field(gt=0)
    row_id_sha256: Sha256
    probability_sha256: Sha256
    parquet_sha256: Sha256
    class_order: list[int] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_classes(self) -> PredictionOutputSummary:
        if self.class_order != list(range(len(self.class_order))):
            raise ValueError("prediction output must use contiguous canonical class order")
        return self


class WorkerConditionResult(StrictModel):
    schema_version: Literal[1] = 1
    plan_sha256: Sha256
    condition_id: Sha256
    source_commit: CommitSha
    observed_package_version: str
    observed_parameters_sha256: Sha256
    checkpoint_identifier: str | None
    checkpoint_sha256: Sha256 | None
    training_row_id_sha256: Sha256
    training_target_sha256: Sha256
    adapter_resource: AdapterResourceSummary
    outputs: list[PredictionOutputSummary] = Field(min_length=2, max_length=2)
    network_attempt_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_output_set(self) -> WorkerConditionResult:
        if {item.partition for item in self.outputs} != {"calibration", "test"}:
            raise ValueError("worker must return predictions for calibration and test")
        if (self.checkpoint_identifier is None) != (self.checkpoint_sha256 is None):
            raise ValueError("checkpoint name and checksum must be present together")
        return self


class ConditionRun(StrictModel):
    condition_id: Sha256
    cache_identity_sha256: Sha256
    model_id: ModelId
    view_id: ViewId
    device: Device
    status: Literal["PASS", "FAIL", "BLOCKED"]
    cache_status: Literal["created", "validated_cache_hit", "not_published"]
    resource: ConditionResource | None
    prediction_files: list[PredictionFileRecord] = Field(default_factory=list)
    failure_category: str | None = None
    failure_reason: str | None = None
    traceback_relative_path: str | None = None

    @model_validator(mode="after")
    def validate_run_record(self) -> ConditionRun:
        if self.status == "PASS":
            if self.resource is None or len(self.prediction_files) != 2:
                raise ValueError("passing condition requires resources and both partition outputs")
            if self.failure_category is not None or self.failure_reason is not None:
                raise ValueError("passing condition cannot contain failure details")
        elif not self.failure_category or not self.failure_reason:
            raise ValueError("unsuccessful condition must retain a failure category and reason")
        return self


class MetricRecord(StrictModel):
    model_id: ModelId
    view_id: ViewId
    device: Device
    partition: Partition
    row_count: int = Field(gt=0)
    brier_score: float = Field(ge=0, le=2, allow_inf_nan=False)
    log_loss: float = Field(ge=0, allow_inf_nan=False)
    accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    balanced_accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    roc_auc: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    roc_auc_status: Literal["defined", "undefined_single_class"]
    expected_calibration_error: float = Field(ge=0, le=1, allow_inf_nan=False)
    prediction_sha256: Sha256
    target_artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_auc_state(self) -> MetricRecord:
        if (self.roc_auc is None) != (self.roc_auc_status == "undefined_single_class"):
            raise ValueError("AUROC value and explicit undefined status disagree")
        return self


class RuntimeEstimate(StrictModel):
    schema_version: Literal[1] = 1
    source_inventory_sha256: Sha256
    method: Literal["prior_binary_probe_max_list_schedule_with_safety_factor"]
    condition_seconds: dict[Sha256, Annotated[float, Field(gt=0, allow_inf_nan=False)]]
    cpu_condition_ids: list[Sha256] = Field(min_length=6, max_length=6)
    gpu_condition_ids: list[Sha256] = Field(min_length=4, max_length=4)
    expected_cpu_seconds: float = Field(ge=0, allow_inf_nan=False)
    expected_gpu_seconds: float = Field(ge=0, allow_inf_nan=False)
    safety_factor: float = Field(default=1.5, ge=1.5, le=1.5, allow_inf_nan=False)
    coordination_overhead_seconds: float = Field(
        default=60.0, ge=60.0, le=60.0, allow_inf_nan=False
    )
    estimated_total_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_estimate(self) -> RuntimeEstimate:
        cpu_ids = self.cpu_condition_ids
        gpu_ids = self.gpu_condition_ids
        if len(set(cpu_ids)) != 6 or len(set(gpu_ids)) != 4 or set(cpu_ids) & set(gpu_ids):
            raise ValueError("runtime estimate must identify six CPU and four GPU conditions")
        if set(self.condition_seconds) != set(cpu_ids) | set(gpu_ids):
            raise ValueError("runtime estimate must cover the exact ten planned condition IDs")
        expected_total = (
            self.expected_cpu_seconds + self.expected_gpu_seconds
        ) * self.safety_factor + self.coordination_overhead_seconds
        if abs(self.estimated_total_seconds - expected_total) > 1e-9:
            raise ValueError("estimated total runtime does not match its components")
        return self


class PairedMetricRecord(StrictModel):
    model_id: ModelId
    device: Device
    partition: Partition
    row_count: int = Field(gt=0)
    sii_mean_js_divergence: float = Field(ge=0, le=1, allow_inf_nan=False)
    sii_median_js_divergence: float = Field(ge=0, le=1, allow_inf_nan=False)
    sii_p90_js_divergence: float = Field(ge=0, le=1, allow_inf_nan=False)
    sii_max_js_divergence: float = Field(ge=0, le=1, allow_inf_nan=False)
    label_flip_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    maximum_absolute_probability_difference: float = Field(ge=0, le=1, allow_inf_nan=False)
    mean_absolute_probability_difference: float = Field(ge=0, le=1, allow_inf_nan=False)
    metric_deltas_v01_minus_v00: dict[MetricName, float | None]
    v00_prediction_sha256: Sha256
    v01_prediction_sha256: Sha256


class SmokeRunReport(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["smoke_experiment"] = "smoke_experiment"
    status: SmokeRunStatus
    source_commit: CommitSha
    plan_sha256: Sha256
    scheduler_plan_sha256: Sha256
    run_mode: Literal["cold", "resume"]
    environment: SmokeEnvironment
    started_at: str
    ended_at: str
    runtime_estimate: RuntimeEstimate
    cold_run_wall_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    planned_conditions: Literal[10] = 10
    executed_conditions: int = Field(ge=0, le=10)
    validated_cache_hits: int = Field(ge=0, le=10)
    failed_conditions: int = Field(ge=0, le=10)
    blocked_conditions: int = Field(ge=0, le=10)
    max_cpu_concurrency: int = Field(ge=0, le=2)
    max_gpu_concurrency: int = Field(ge=0, le=1)
    duplicate_prediction_artifacts: int = Field(ge=0)
    result_rewrites: int = Field(ge=0)
    evaluation_source: EvaluationSource
    conditions: list[ConditionRun] = Field(min_length=10, max_length=10)
    metrics: list[MetricRecord] = Field(default_factory=list)
    paired_metrics: list[PairedMetricRecord] = Field(default_factory=list)
    metric_table_sha256: Sha256 | None = None
    paired_metric_table_sha256: Sha256 | None = None
    cold_run_report_sha256: Sha256 | None = None
    resume_prediction_hashes_before_sha256: Sha256 | None = None
    resume_prediction_hashes_after_sha256: Sha256 | None = None
    resume_verification_sha256: Sha256 | None = None
    resume_prediction_hashes_unchanged: bool = False
    resume_prediction_mtimes_unchanged: bool = False
    protected_foundation_unchanged: bool
    protected_foundation_comparison_sha256: Sha256 | None
    protected_split_validation_sha256: Sha256 | None = None
    test_labels_opened_after_predictions: bool
    labels_evaluated: bool
    network_attempt_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_report(self) -> SmokeRunReport:
        if len({item.condition_id for item in self.conditions}) != 10:
            raise ValueError("run report must account for ten unique conditions")
        complete = all(item.status == "PASS" for item in self.conditions)
        if self.status == "PASS_PENDING_REVIEW":
            if not complete or not self.metrics or not self.paired_metrics:
                raise ValueError("passing smoke report requires all conditions and metrics")
            if (
                not self.protected_foundation_unchanged
                or not self.test_labels_opened_after_predictions
            ):
                raise ValueError("smoke acceptance requires preservation and sealed-label evidence")
            if not self.labels_evaluated:
                raise ValueError("passing smoke report must record final-boundary evaluation")
            if self.metric_table_sha256 is None:
                raise ValueError("passing smoke report requires a metric artifact hash")
            if self.paired_metric_table_sha256 is None:
                raise ValueError("passing smoke report requires a paired-metric artifact hash")
            if self.failed_conditions or self.blocked_conditions or self.network_attempt_count:
                raise ValueError("passing smoke report cannot include failures or network attempts")
            if self.run_mode == "cold" and (
                self.executed_conditions != 10 or self.validated_cache_hits != 0
            ):
                raise ValueError("cold pass must execute all ten conditions without cache hits")
            if self.run_mode == "resume" and (
                self.executed_conditions != 0 or self.validated_cache_hits != 10
            ):
                raise ValueError("resume pass must reuse ten conditions and execute none")
            if self.duplicate_prediction_artifacts or self.result_rewrites:
                raise ValueError("passing smoke run has duplicate artifacts or result rewrites")
            if self.protected_foundation_comparison_sha256 is None:
                raise ValueError("passing smoke run requires a protected-hash comparison")
            if self.protected_split_validation_sha256 is None:
                raise ValueError("passing smoke run requires protected split validation")
            if self.run_mode == "resume" and (
                self.cold_run_report_sha256 is None
                or self.resume_prediction_hashes_before_sha256 is None
                or self.resume_prediction_hashes_after_sha256 is None
                or self.resume_verification_sha256 is None
                or not self.resume_prediction_hashes_unchanged
                or not self.resume_prediction_mtimes_unchanged
            ):
                raise ValueError("passing resume report lacks proof of unchanged cold outputs")
            expected_cache_status = "created" if self.run_mode == "cold" else "validated_cache_hit"
            if any(item.cache_status != expected_cache_status for item in self.conditions):
                raise ValueError("condition cache outcomes disagree with the run mode")
            if self.evaluation_source == "not_performed":
                raise ValueError("passing smoke run cannot omit its metric evaluation")
            if self.run_mode == "resume" and self.cold_run_wall_seconds is None:
                raise ValueError("passing resume report must retain observed cold-run duration")
        if self.failed_conditions != sum(item.status == "FAIL" for item in self.conditions):
            raise ValueError("failed-condition count differs from condition records")
        if self.blocked_conditions != sum(item.status == "BLOCKED" for item in self.conditions):
            raise ValueError("blocked-condition count differs from condition records")
        if self.status == "PARTIAL_VALID" and not any(
            item.status != "PASS" for item in self.conditions
        ):
            raise ValueError("PARTIAL_VALID requires at least one recorded condition failure")
        return self


class SmokeEvidenceInventory(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["smoke_experiment_evidence"] = "smoke_experiment_evidence"
    status: SmokeRunStatus
    source_commit: CommitSha
    plan_sha256: Sha256
    run_report_sha256: Sha256
    resume_verification_sha256: Sha256 | None = None
    protected_split_validation_sha256: Sha256 | None = None
    condition_count: Literal[10] = 10
    passed_conditions: int = Field(ge=0, le=10)
    failed_conditions: int = Field(ge=0, le=10)
    blocked_conditions: int = Field(ge=0, le=10)
    prediction_artifact_count: int = Field(ge=0, le=20)
    metric_record_count: int = Field(ge=0, le=20)
    paired_metric_record_count: int = Field(ge=0, le=10)
    protected_foundation_unchanged: bool
    test_labels_opened_after_predictions: bool
    network_attempt_count: int = Field(ge=0)
    dataset_features_sha256: Sha256
    dataset_source_sha256: Sha256
    target_artifact_sha256: Sha256
    split_sha256: Sha256
    transformation_inventory_sha256: Sha256
    environment: SmokeEnvironment
    conditions: list[SmokeEvidenceCondition] = Field(min_length=10, max_length=10)
    metrics: list[MetricRecord] = Field(default_factory=list)
    paired_metrics: list[PairedMetricRecord] = Field(default_factory=list)
    acceptance_gates: list[SmokeGateRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> SmokeEvidenceInventory:
        if self.passed_conditions + self.failed_conditions + self.blocked_conditions != 10:
            raise ValueError("evidence inventory must account for every condition")
        if len({item.condition_id for item in self.conditions}) != 10:
            raise ValueError("sanitized inventory must retain ten unique condition identities")
        if sum(item.status == "PASS" for item in self.conditions) != self.passed_conditions:
            raise ValueError("inventory passed count differs from condition evidence")
        if sum(item.status == "FAIL" for item in self.conditions) != self.failed_conditions:
            raise ValueError("inventory failed count differs from condition evidence")
        if sum(item.status == "BLOCKED" for item in self.conditions) != self.blocked_conditions:
            raise ValueError("inventory blocked count differs from condition evidence")
        if (
            any(gate.result != "PASS" for gate in self.acceptance_gates)
            and self.status == "PASS_PENDING_REVIEW"
        ):
            raise ValueError("passing inventory cannot contain a failed or unverified gate")
        if self.status == "PASS_PENDING_REVIEW" and (
            self.passed_conditions != 10
            or self.failed_conditions
            or self.blocked_conditions
            or self.prediction_artifact_count != 20
            or not self.protected_foundation_unchanged
            or not self.test_labels_opened_after_predictions
            or self.network_attempt_count
        ):
            raise ValueError("passing smoke evidence does not satisfy frozen gates")
        return self


class SmokeValidationReport(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["smoke_experiment_validation"] = "smoke_experiment_validation"
    status: Literal["PASS", "FAIL"]
    plan_sha256: Sha256
    run_report_sha256: Sha256
    validated_prediction_count: int = Field(ge=0, le=20)
    validated_metric_record_count: int = Field(ge=0, le=20)
    validated_paired_metric_count: int = Field(ge=0, le=10)
    gates: list[SmokeGateRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_gate_results(self) -> SmokeValidationReport:
        if self.status == "PASS" and any(gate.result != "PASS" for gate in self.gates):
            raise ValueError("passing validation report cannot contain failed/unverified gates")
        return self


__all__ = [
    "AdapterResourceSummary",
    "ConditionResource",
    "ConditionRun",
    "MetricRecord",
    "ProtectedFoundationArtifactHash",
    "ProtectedFoundationHashComparison",
    "ProtectedSplitValidation",
    "PairedMetricRecord",
    "PlanEnvelope",
    "PredictionFileRecord",
    "PredictionOutputSummary",
    "SmokeCondition",
    "SmokeEvidenceInventory",
    "SmokeEvidenceCondition",
    "SmokeEnvironment",
    "SmokeGateRecord",
    "SmokePlan",
    "SmokeRunReport",
    "SmokeValidationReport",
    "ResumeVerification",
    "SmokeView",
    "WorkerConditionResult",
    "condition_identity",
]
