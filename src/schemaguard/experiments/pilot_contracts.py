"""Strict, immutable contracts for the plan-only representation-sensitivity pilot."""

from __future__ import annotations

import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..utils.hashing import sha256_canonical_json

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
ModelId = Literal["LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2"]
ViewId = Literal["V00", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10"]
Applicability = Literal["APPLICABLE", "NOT_APPLICABLE"]
DecisionOutcome = Literal[
    "ADVANCE_BROAD_METHOD",
    "ADVANCE_TFM_FOCUSED",
    "NARROW_CASE_STUDY",
    "STOP_OR_REDESIGN",
    "PARTIAL_CAPABILITY",
    "REPAIR_REQUIRED",
]
PilotStatus = Literal[
    "PASS_PENDING_REVIEW",
    "RESOURCE_REVIEW_REQUIRED",
    "REPAIR_REQUIRED",
    "BLOCKED",
]
GateResult = Literal["PASS", "FAIL", "NOT_VERIFIED"]

MODEL_ORDER: tuple[ModelId, ...] = (
    "LR-1.9",
    "CAT-1.2",
    "XGB-3.4",
    "TPFN3-8.5",
    "TICL2-2.2",
)
VIEW_ORDER: tuple[ViewId, ...] = (
    "V00",
    "V01",
    "V02",
    "V03",
    "V04",
    "V05",
    "V06",
    "V07",
    "V08",
    "V09",
    "V10",
)
SCHEMAORBIT_IDS = (3, 23, 29, 31, 36, 37, 38, 44, 46, 50, 54, 1067, 1464, 1489)
PILOT_OUTCOMES: tuple[DecisionOutcome, ...] = (
    "ADVANCE_BROAD_METHOD",
    "ADVANCE_TFM_FOCUSED",
    "NARROW_CASE_STUDY",
    "STOP_OR_REDESIGN",
    "PARTIAL_CAPABILITY",
    "REPAIR_REQUIRED",
)


class StrictPilotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


def validate_relative_path(value: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or "\\" in value
        or posix.as_posix() != value
    ):
        raise ValueError("path must be a portable repository-relative path")
    return value


class ProtocolIdentityConfig(StrictPilotModel):
    protocol_id: Literal["schemaguard_representation_sensitivity_pilot"]
    protocol_name: Literal["SchemaGuard Representation-Sensitivity Pilot"]
    protocol_version: Literal["schema_guard_representation_sensitivity_v1"]
    purpose: str
    primary_question: str
    secondary_question: str
    execution_authorized: Literal[False]
    offline: Literal[True]
    source_commit: CommitSha
    registry_config: str
    experiment_registry: str
    split_config: str
    transformation_config: str
    adapter_config: str
    model_runtime_config: str = Field(alias="model_config")
    dependency_lock: str
    dataset_report: str
    split_inventory: str
    transformation_inventory: str
    adapter_inventory: str
    smoke_review: str
    smoke_runtime_report: str
    selected_dataset_count: Literal[8]
    protected_anchor_dataset_id: Literal[1464]
    selected_seed_count: Literal[3]

    @model_validator(mode="after")
    def validate_paths(self) -> ProtocolIdentityConfig:
        for name in (
            "registry_config",
            "experiment_registry",
            "split_config",
            "transformation_config",
            "adapter_config",
            "model_runtime_config",
            "dependency_lock",
            "dataset_report",
            "split_inventory",
            "transformation_inventory",
            "adapter_inventory",
            "smoke_review",
            "smoke_runtime_report",
        ):
            validate_relative_path(getattr(self, name))
        return self


class SelectionPolicy(StrictPilotModel):
    max_rows: int = Field(gt=0)
    max_features: int = Field(gt=0)
    max_classes: int = Field(ge=2)
    foundation_max_train_rows: int = Field(gt=0)
    balanced_max_class_fraction: float = Field(gt=0.5, lt=1, allow_inf_nan=False)
    categorical_substantial_fraction: float = Field(gt=0, le=1, allow_inf_nan=False)
    small_dataset_max_rows: int = Field(gt=0)
    required_coverage: tuple[
        Literal[
            "numeric_only",
            "mixed_numeric_categorical",
            "substantial_categorical",
            "high_dimensional",
            "high_cardinality_categorical",
            "missing_values",
            "binary",
            "multiclass",
            "small",
            "medium",
            "balanced",
            "imbalanced",
        ],
        ...,
    ]
    duplicate_profile_fields: tuple[
        Literal["task_type", "schema_family", "size_band", "balance_band"], ...
    ]
    tie_break: Literal["ascending_canonical_dataset_id"]
    outcomes_prohibited: Literal[True]

    @model_validator(mode="after")
    def validate_coverage(self) -> SelectionPolicy:
        if len(set(self.required_coverage)) != len(self.required_coverage):
            raise ValueError("required coverage tags must be unique")
        if self.small_dataset_max_rows >= self.max_rows:
            raise ValueError("small and maximum row limits must be distinct")
        return self


class SeedPolicy(StrictPilotModel):
    source_config: str
    selection_rule: Literal["first_three_in_canonical_configured_order"]
    required_anchor_seed: Literal[1729]

    @model_validator(mode="after")
    def validate_source_path(self) -> SeedPolicy:
        validate_relative_path(self.source_config)
        return self


class ViewPolicy(StrictPilotModel):
    source_config: str
    required_count: Literal[11]
    source_inventory: str
    preserve_applicability_status: Literal[True]
    require_v00_applicable: Literal[True]
    do_not_materialize: Literal[True]

    @model_validator(mode="after")
    def validate_source_paths(self) -> ViewPolicy:
        validate_relative_path(self.source_config)
        validate_relative_path(self.source_inventory)
        return self


class ModelPolicy(StrictPilotModel):
    source_registry: str
    source_runtime_config: str
    source_adapter_config: str
    required_ids: tuple[ModelId, ...]
    devices: dict[ModelId, Literal["cpu", "cuda"]]
    checkpoint_hashes: dict[ModelId, Sha256]
    precision_policy: dict[ModelId, str]
    batch_policy: dict[ModelId, str]

    @model_validator(mode="after")
    def validate_matrix(self) -> ModelPolicy:
        if self.required_ids != MODEL_ORDER:
            raise ValueError("pilot model order must match the five accepted model IDs")
        if tuple(self.devices) != MODEL_ORDER or tuple(self.precision_policy) != MODEL_ORDER:
            raise ValueError("model device and precision maps must cover models in canonical order")
        if tuple(self.batch_policy) != MODEL_ORDER:
            raise ValueError("model batch map must cover models in canonical order")
        expected_devices = {
            "LR-1.9": "cpu",
            "CAT-1.2": "cpu",
            "XGB-3.4": "cpu",
            "TPFN3-8.5": "cuda",
            "TICL2-2.2": "cuda",
        }
        if self.devices != expected_devices:
            raise ValueError("frozen CPU/CUDA model assignments changed")
        if set(self.checkpoint_hashes) != {"TPFN3-8.5", "TICL2-2.2"}:
            raise ValueError("only the two frozen foundation models have checkpoints")
        for name in ("source_registry", "source_runtime_config", "source_adapter_config"):
            validate_relative_path(getattr(self, name))
        return self


class MetricPolicy(StrictPilotModel):
    primary_endpoint: Literal["worst_view_brier_degradation"]
    brier_aggregation: Literal["sum_squared_class_probability_error_per_row_then_mean"]
    relative_degradation_epsilon: float = Field(gt=0, allow_inf_nan=False)
    seed_aggregation: Literal["median"]
    dataset_aggregation: Literal["median"]
    cross_dataset_unit: Literal["dataset"]
    row_level_role: Literal["metric_calculation_only"]
    seed_level_role: Literal["repeatability"]
    dataset_model_level_aggregation: Literal["median_across_seeds"]
    dataset_level_summaries: tuple[
        Literal[
            "median",
            "interquartile_range",
            "minimum",
            "maximum",
            "win_tie_loss_counts",
            "transformation_family_summaries",
            "model_family_summaries",
        ],
        ...,
    ]
    effective_sample_size: Literal["number_of_datasets"]
    confirmatory_significance_claims: Literal[False]
    confidence_intervals: Literal["exploratory_only"]
    sii: SiiPolicy
    label_flip: LabelFlipPolicy
    secondary_metrics: tuple[
        Literal[
            "log_loss",
            "accuracy",
            "balanced_accuracy",
            "raw_auroc",
            "tolerance_aware_auroc",
            "expected_calibration_error",
            "maximum_absolute_probability_difference",
            "mean_absolute_probability_difference",
            "runtime",
            "fit_time",
            "prediction_time",
            "peak_process_tree_ram",
            "peak_cuda_allocated_memory",
            "peak_cuda_reserved_memory",
            "cache_status",
            "failure_category",
        ],
        ...,
    ]
    undefined_metric_status: Literal["explicit_undefined_with_reason"]
    non_rank_equivalence: NonRankEquivalencePolicy
    auroc: AurocPolicy
    ece: EcePolicy
    probability: ProbabilityPolicy

    @model_validator(mode="after")
    def validate_secondary_metrics(self) -> MetricPolicy:
        expected = (
            "log_loss",
            "accuracy",
            "balanced_accuracy",
            "raw_auroc",
            "tolerance_aware_auroc",
            "expected_calibration_error",
            "maximum_absolute_probability_difference",
            "mean_absolute_probability_difference",
            "runtime",
            "fit_time",
            "prediction_time",
            "peak_process_tree_ram",
            "peak_cuda_allocated_memory",
            "peak_cuda_reserved_memory",
            "cache_status",
            "failure_category",
        )
        if self.secondary_metrics != expected:
            raise ValueError("the complete canonical secondary-metric set and order are frozen")
        if self.relative_degradation_epsilon != 1.0e-12:
            raise ValueError("relative Brier degradation epsilon is frozen at 1e-12")
        expected_summaries = (
            "median",
            "interquartile_range",
            "minimum",
            "maximum",
            "win_tie_loss_counts",
            "transformation_family_summaries",
            "model_family_summaries",
        )
        if self.dataset_level_summaries != expected_summaries:
            raise ValueError("dataset-level summary fields and order are frozen")
        return self


class SiiPolicy(StrictPilotModel):
    name: Literal["semantic_instability_index"]
    per_row: Literal["maximum_pairwise_jsd_across_applicable_views"]
    logarithm_base: Literal[2]
    probability_clip: float = Field(gt=0, lt=0.01, allow_inf_nan=False)
    clipping_rule: Literal["clip_then_renormalize_rows"]
    summaries: tuple[Literal["mean", "median", "p90", "p95", "maximum"], ...]
    maximum_mean_pair_recorded: Literal[True]
    maximum_row_pair_recorded: Literal[True]

    @model_validator(mode="after")
    def validate_frozen_clipping(self) -> SiiPolicy:
        if self.probability_clip != 1.0e-15:
            raise ValueError("SII probability clipping is frozen at 1e-15")
        if self.summaries != ("mean", "median", "p90", "p95", "maximum"):
            raise ValueError("the complete ordered SII summary set is frozen")
        return self


class LabelFlipPolicy(StrictPilotModel):
    rule: Literal["any_argmax_class_difference_across_applicable_views"]
    include_pairwise_against_v00: Literal[True]
    include_worst_view: Literal[True]
    include_responsible_transformation_family: Literal[True]


class NonRankEquivalencePolicy(StrictPilotModel):
    absolute_atol: float = Field(ge=0, allow_inf_nan=False)
    relative_rtol: float = Field(ge=0, allow_inf_nan=False)


class AurocPolicy(StrictPilotModel):
    binary_only: Literal[True]
    numerical_tie_tolerance: float = Field(gt=0, allow_inf_nan=False)
    chunk_size: int = Field(gt=0)
    tie_credit: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    report_raw_and_tolerance_aware: Literal[True]
    record_tolerance_tie_pair_count: Literal[True]
    tie_only_changes_excluded_from_scientific_instability: Literal[True]
    tie_sensitivity_probability_max_abs_diff: float = Field(gt=0, allow_inf_nan=False)
    tie_sensitivity_label_flip_count: Literal[0]
    tie_sensitivity_non_rank_metrics: tuple[Literal["brier", "log_loss"], ...]
    tie_sensitivity_tolerance_aware_auroc_atol: float = Field(ge=0, allow_inf_nan=False)
    tie_sensitivity_classification: Literal["NUMERICAL_TIE_SENSITIVITY"]

    @model_validator(mode="after")
    def validate_tie_credit(self) -> AurocPolicy:
        if self.tie_credit != 0.5:
            raise ValueError("AUROC exact ties must receive half credit")
        if self.numerical_tie_tolerance != 1.0e-15 or self.chunk_size != 256:
            raise ValueError("AUROC tie tolerance and deterministic chunk size are frozen")
        if self.tie_sensitivity_probability_max_abs_diff != 1.0e-15:
            raise ValueError("tie sensitivity probability threshold must be 1e-15")
        if self.tie_sensitivity_non_rank_metrics != ("brier", "log_loss"):
            raise ValueError("Brier and log loss are both required for tie sensitivity")
        if self.tie_sensitivity_tolerance_aware_auroc_atol != 1.0e-12:
            raise ValueError("tolerance-aware AUROC comparison tolerance is frozen at 1e-12")
        return self


class EcePolicy(StrictPilotModel):
    bins: int = Field(ge=2)
    rule: Literal["equal_width_max_probability_confidence"]


class ProbabilityPolicy(StrictPilotModel):
    row_sum_atol: float = Field(gt=0, allow_inf_nan=False)
    non_rank_metric_tolerance_is_separate: Literal[True]


class LeakagePolicy(StrictPilotModel):
    split_before_views: Literal[True]
    transformation_fit_partition: Literal["train"]
    preprocessing_fit_partition: Literal["train"]
    model_fit_partition: Literal["train"]
    calibration_use: Literal["predeclared_calibration_or_evaluation_only"]
    test_features: Literal["final_frozen_inference_only"]
    seal_test_labels_until_all_predictions_and_structure_valid: Literal[True]
    stable_row_id_alignment_required: Literal[True]
    label_open_event_order: tuple[
        Literal[
            "plan_frozen",
            "test_labels_sealed",
            "training_completed",
            "calibration_predictions_completed",
            "test_predictions_completed",
            "prediction_structure_validated",
            "test_labels_opened",
            "metrics_generated",
            "results_sealed",
        ],
        ...,
    ]
    labels_must_not_influence: tuple[str, ...]

    @model_validator(mode="after")
    def validate_event_sequence(self) -> LeakagePolicy:
        expected = (
            "plan_frozen",
            "test_labels_sealed",
            "training_completed",
            "calibration_predictions_completed",
            "test_predictions_completed",
            "prediction_structure_validated",
            "test_labels_opened",
            "metrics_generated",
            "results_sealed",
        )
        if self.label_open_event_order != expected:
            raise ValueError("label opening must follow prediction structural validation")
        if len(set(self.labels_must_not_influence)) != len(self.labels_must_not_influence):
            raise ValueError("prohibited label-influence decisions must be unique")
        return self


class DecisionPolicy(StrictPilotModel):
    integrity_failure: Literal["REPAIR_REQUIRED"]
    foundation_model_ids: tuple[Literal["TPFN3-8.5", "TICL2-2.2"], ...]
    nontrivial_absolute_wbd: float = Field(ge=0, allow_inf_nan=False)
    nontrivial_relative_wbd: float = Field(ge=0, allow_inf_nan=False)
    require_all_v00_conditions: Literal[True]
    minimum_transformed_condition_completion_rate: float = Field(gt=0, le=1, allow_inf_nan=False)
    require_every_failure_classified: Literal[True]
    require_frozen_model_configuration: Literal[True]
    require_resume_cache_integrity: Literal[True]
    require_resource_ceilings: Literal[True]
    require_frozen_preprocessing_residual: Literal[True]
    exploratory_only: Literal[True]
    breadth_dataset_count: int = Field(gt=0)
    required_transformation_families: int = Field(gt=0)
    required_seed_direction_count: int = Field(gt=0)
    required_seed_count: int = Field(gt=0)
    exclude_numerical_tie_only: Literal[True]
    outcomes: tuple[DecisionOutcome, ...]

    @model_validator(mode="after")
    def validate_outcomes(self) -> DecisionPolicy:
        if self.outcomes != PILOT_OUTCOMES:
            raise ValueError("pilot decision vocabulary is frozen")
        if self.foundation_model_ids != ("TPFN3-8.5", "TICL2-2.2"):
            raise ValueError("the two frozen foundation models define the primary effect gate")
        if self.required_seed_direction_count > self.required_seed_count:
            raise ValueError("breadth direction requirement exceeds the seed count")
        if self.minimum_transformed_condition_completion_rate != 0.95:
            raise ValueError("at least 95% of applicable transformed conditions must complete")
        if (
            self.nontrivial_absolute_wbd != 0.03
            or self.nontrivial_relative_wbd != 0.10
            or self.breadth_dataset_count != 3
            or self.required_transformation_families != 2
            or self.required_seed_direction_count != 2
            or self.required_seed_count != 3
        ):
            raise ValueError("pilot effect and breadth thresholds are frozen")
        return self


class StagingPolicy(StrictPilotModel):
    stage_grouping: Literal["contiguous_pairs_by_ascending_dataset_id"]
    datasets_per_stage: Literal[2]
    stage_boundary: Literal["whole_dataset"]
    preserve_complete_condition_matrix: Literal[True]


class ResourcePolicy(StrictPilotModel):
    cpu_workers: Literal[2]
    gpu_workers: Literal[1]
    cpu_threads_per_worker: Literal[2]
    cpu_thread_environment: dict[str, Literal["2"]]
    foundation_models_sequential: Literal[True]
    vram_ceiling_mib: int = Field(gt=0, le=3600)
    ram_hard_limit_mib: int = Field(gt=0)
    timeout_seconds: dict[Literal["cpu_model_condition", "cuda_model_condition"], int]
    preferred_wall_seconds: int = Field(gt=0)
    hard_wall_seconds: int = Field(gt=0)
    preferred_storage_bytes: int = Field(gt=0)
    prediction_artifact_overhead_bytes: int = Field(gt=0)
    storage_bytes_per_row_class: int = Field(gt=0)
    storage_bytes_per_row_id: int = Field(gt=0)
    per_row_metadata_overhead_bytes: int = Field(gt=0)
    estimate_safety_factor: float = Field(ge=1, allow_inf_nan=False)
    coordination_seconds: int = Field(ge=0)
    no_silent_fallback: Literal[True]
    no_silent_precision_or_batch_change: Literal[True]
    offline_execution: Literal[True]
    atomic_incremental_publication: Literal[True]
    validated_resume: Literal[True]
    staging: StagingPolicy

    @model_validator(mode="after")
    def validate_bounds(self) -> ResourcePolicy:
        if self.hard_wall_seconds < self.preferred_wall_seconds:
            raise ValueError("hard runtime ceiling must not be below preferred target")
        if set(self.cpu_thread_environment) != {
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        }:
            raise ValueError("all four controlled numerical thread variables are required")
        if set(self.timeout_seconds) != {"cpu_model_condition", "cuda_model_condition"}:
            raise ValueError("CPU and CUDA condition timeouts are required")
        if any(value <= 0 for value in self.timeout_seconds.values()):
            raise ValueError("condition timeouts must be positive")
        return self


class RetryPolicy(StrictPilotModel):
    max_attempts_per_condition: Literal[2]
    retryable_categories: tuple[Literal["WORKER_CRASH", "TRANSIENT_IO", "TIMEOUT"], ...]
    non_retryable_categories: tuple[
        Literal[
            "GPU_OOM",
            "INVALID_PROBABILITY",
            "MISSING_ROW",
            "CHECKPOINT_FAILURE",
            "PACKAGE_OR_API_MISMATCH",
            "OFFLINE_NETWORK_ATTEMPT",
            "IDENTITY_MISMATCH",
        ],
        ...,
    ]
    failed_artifact_cache_hit: Literal[False]
    partial_artifact_promotion: Literal[False]
    retry_identity_must_match: Literal[True]
    test_outcome_controls_retry: Literal[False]

    @model_validator(mode="after")
    def validate_categories(self) -> RetryPolicy:
        if set(self.retryable_categories) & set(self.non_retryable_categories):
            raise ValueError("retry failure categories cannot overlap")
        if not self.retryable_categories or not self.non_retryable_categories:
            raise ValueError("retry and non-retryable failure categories must be explicit")
        if self.retryable_categories != ("WORKER_CRASH", "TRANSIENT_IO", "TIMEOUT"):
            raise ValueError("retryable failure categories are frozen")
        if self.non_retryable_categories != (
            "GPU_OOM",
            "INVALID_PROBABILITY",
            "MISSING_ROW",
            "CHECKPOINT_FAILURE",
            "PACKAGE_OR_API_MISMATCH",
            "OFFLINE_NETWORK_ATTEMPT",
            "IDENTITY_MISMATCH",
        ):
            raise ValueError("non-retryable failure categories are frozen")
        return self


class PilotConfig(StrictPilotModel):
    schema_version: Literal[1]
    protocol: ProtocolIdentityConfig
    selection: SelectionPolicy
    seeds: SeedPolicy
    views: ViewPolicy
    models: ModelPolicy
    metrics: MetricPolicy
    leakage: LeakagePolicy
    decisions: DecisionPolicy
    resources: ResourcePolicy
    retry: RetryPolicy


class SplitIdentity(StrictPilotModel):
    seed: int = Field(ge=0)
    strategy: Literal["stratified_group_5fold_v1"]
    assignment_sha256: Sha256
    logical_assignment_sha256: Sha256
    manifest_sha256: Sha256
    cross_partition_group_count: Literal[0]
    row_count: int = Field(gt=0)
    partition_counts: dict[Literal["train", "calibration", "test"], int]


class PilotDatasetSelection(StrictPilotModel):
    dataset_id: int = Field(gt=0)
    provider: Literal["openml"]
    provider_dataset_version: str
    provider_file_id: int = Field(gt=0)
    dataset_name: str
    target_column: str
    task_type: Literal["binary", "multiclass"]
    row_count: int = Field(gt=0)
    feature_count: int = Field(gt=0)
    numeric_feature_count: int = Field(ge=0)
    categorical_feature_count: int = Field(ge=0)
    class_count: int = Field(ge=2)
    class_distribution: dict[str, int]
    predictor_group_count: int = Field(ge=1)
    duplicate_group_count: int = Field(ge=0)
    license: str
    raw_source_sha256: Sha256
    processed_feature_sha256: Sha256
    target_artifact_sha256: Sha256
    quality_report_sha256: Sha256
    metadata_evidence_sha256: Sha256
    five_split_identities: tuple[SplitIdentity, ...]
    coverage_tags: tuple[str, ...]
    selection_rationale: str
    v00_model_compatibility: Literal["PASS_REGISTRY_CAPABILITY"]

    @model_validator(mode="after")
    def validate_selection(self) -> PilotDatasetSelection:
        if self.numeric_feature_count + self.categorical_feature_count != self.feature_count:
            raise ValueError("feature-type counts must add up to the registered feature count")
        if self.task_type != ("binary" if self.class_count == 2 else "multiclass"):
            raise ValueError("task type disagrees with class count")
        if sum(self.class_distribution.values()) != self.row_count:
            raise ValueError("class-distribution counts do not sum to rows")
        if len(self.five_split_identities) != 5:
            raise ValueError("selection evidence must preserve all five accepted split identities")
        if len({item.seed for item in self.five_split_identities}) != 5:
            raise ValueError("five split identities must use five distinct seeds")
        if any(
            sum(item.partition_counts.values()) != self.row_count
            or item.partition_counts.get("train", 0) <= 0
            or item.partition_counts.get("calibration", 0) <= 0
            or item.partition_counts.get("test", 0) <= 0
            or item.cross_partition_group_count != 0
            for item in self.five_split_identities
        ):
            raise ValueError("split identities must cover rows with positive, disjoint partitions")
        return self


class ExcludedDataset(StrictPilotModel):
    dataset_id: int = Field(gt=0)
    dataset_name: str
    eligible: Literal[True]
    exclusion_reason: str
    competing_selected_dataset_id: int = Field(gt=0)
    metadata_evidence_sha256: Sha256


class PilotSeedSelection(StrictPilotModel):
    schema_version: Literal[1] = 1
    seeds: tuple[int, int, int]
    source_config_sha256: Sha256
    canonical_order_preserved: Literal[True]
    dataset_seed_count: Literal[24]
    split_identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_seed_order(self) -> PilotSeedSelection:
        if self.seeds != (1729, 2718, 31415):
            raise ValueError("pilot seeds must be the first three canonical split-generation seeds")
        return self


class PilotViewRegistryEntry(StrictPilotModel):
    view_id: ViewId
    view_name: str
    transformation_family: str
    certificate_type: Literal["BIJECTION", "PROJECTION", "PERMUTATION", "COMPOSITION"]
    scientific_role: Literal["CONTROL", "PRIMARY_MIGRATION"]


class ViewApplicabilityRecord(StrictPilotModel):
    dataset_id: int = Field(gt=0)
    seed: int = Field(ge=0)
    view_id: ViewId
    view_name: str
    applicability: Applicability
    reason_code: str | None
    reason: str | None
    transformation_inventory_record_sha256: Sha256
    transformation_config_sha256: Sha256
    transformation_implementation_sha256: Sha256
    certificate_type: Literal["BIJECTION", "PROJECTION", "PERMUTATION", "COMPOSITION"]
    certificate_identities: tuple[Sha256, ...]
    source_feature_sha256: Sha256 | None
    target_artifact_sha256: Sha256
    reconstruction_rtol: float = Field(ge=0, allow_inf_nan=False)
    reconstruction_atol: float = Field(ge=0, allow_inf_nan=False)
    reconstruction_max_abs_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_applicability(self) -> ViewApplicabilityRecord:
        if self.view_id == "V00" and self.applicability != "APPLICABLE":
            raise ValueError("V00 identity view must be applicable")
        if self.applicability == "APPLICABLE":
            if self.reason is not None or self.reason_code is not None:
                raise ValueError("applicable views cannot have exclusion reasons")
            if len(self.certificate_identities) != 3 or self.source_feature_sha256 is None:
                raise ValueError(
                    "applicable view requires three certificate identities and source hash"
                )
            if self.reconstruction_max_abs_error is None:
                raise ValueError("applicable view must preserve reconstruction evidence")
        elif not self.reason or not self.reason_code or self.certificate_identities:
            raise ValueError("not-applicable view requires a reason and no certificates")
        return self


class PilotModelEntry(StrictPilotModel):
    model_id: ModelId
    model_class: str
    package: str
    version: str
    preprocessing: str
    parameters: dict[str, object]
    model_config_sha256: Sha256
    preprocessing_sha256: Sha256
    checkpoint_identifier: str | None
    checkpoint_sha256: Sha256 | None
    device: Literal["cpu", "cuda"]
    precision_policy: str
    batch_policy: str
    compatibility_evidence_sha256: Sha256


class RuntimeEstimate(StrictPilotModel):
    schema_version: Literal[1] = 1
    source_measurement_sha256: Sha256
    model_condition_counts: dict[ModelId, int]
    estimated_work_seconds_by_dataset: dict[int, dict[Literal["cpu", "cuda"], float]]
    estimated_prediction_storage_by_dataset_bytes: dict[int, int]
    estimated_cpu_seconds: float = Field(ge=0, allow_inf_nan=False)
    estimated_cuda_seconds: float = Field(ge=0, allow_inf_nan=False)
    conservative_wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    estimated_prediction_storage_bytes: int = Field(ge=0)
    estimated_cache_storage_bytes: int = Field(ge=0)
    peak_ram_mib: float = Field(ge=0, allow_inf_nan=False)
    peak_vram_mib: float = Field(ge=0, allow_inf_nan=False)
    estimate_method: str
    preferred_wall_limit_passed: bool
    hard_wall_limit_passed: bool
    preferred_storage_limit_passed: bool
    staged_schedule_required: bool

    @model_validator(mode="after")
    def validate_dataset_breakdown(self) -> RuntimeEstimate:
        dataset_ids = set(self.estimated_work_seconds_by_dataset)
        if dataset_ids != set(self.estimated_prediction_storage_by_dataset_bytes):
            raise ValueError("runtime work and storage breakdowns must cover the same datasets")
        if any(
            set(device_work) != {"cpu", "cuda"}
            or any(not math.isfinite(value) or value < 0 for value in device_work.values())
            for device_work in self.estimated_work_seconds_by_dataset.values()
        ):
            raise ValueError("dataset runtime breakdowns require finite nonnegative CPU/CUDA work")
        if any(value < 0 for value in self.estimated_prediction_storage_by_dataset_bytes.values()):
            raise ValueError("dataset storage estimates must be nonnegative")
        if (
            sum(self.estimated_prediction_storage_by_dataset_bytes.values())
            != self.estimated_prediction_storage_bytes
        ):
            raise ValueError("dataset storage breakdown must sum to the total prediction estimate")
        return self


class PilotProtocolArtifact(StrictPilotModel):
    schema_version: Literal[1] = 1
    protocol_id: Literal["schemaguard_representation_sensitivity_pilot"]
    protocol_name: Literal["SchemaGuard Representation-Sensitivity Pilot"]
    protocol_version: Literal["schema_guard_representation_sensitivity_v1"]
    execution_authorized: Literal[False] = False
    offline: Literal[True] = True
    protocol_sha256: Sha256
    source_commit: CommitSha
    source_hashes: dict[str, Sha256]
    dataset_selection_policy: SelectionPolicy
    selected_datasets: tuple[PilotDatasetSelection, ...]
    excluded_datasets: tuple[ExcludedDataset, ...]
    seed_selection: PilotSeedSelection
    view_registry: tuple[PilotViewRegistryEntry, ...]
    model_matrix: tuple[PilotModelEntry, ...]
    registry_parameter_decisions: tuple[
        Literal["catboost_yaml_no_scalar", "tabicl_checkpoint_version_from_checkpoint_field"], ...
    ]
    metric_policy: MetricPolicy
    leakage_policy: LeakagePolicy
    decision_policy: DecisionPolicy
    resource_policy: ResourcePolicy
    retry_policy: RetryPolicy
    runtime_estimate: RuntimeEstimate
    condition_inventory_sha256: Sha256 | None = None
    summary: dict[str, int]

    @model_validator(mode="after")
    def validate_protocol_matrix(self) -> PilotProtocolArtifact:
        if len(self.selected_datasets) != 8:
            raise ValueError("pilot protocol must freeze exactly eight datasets")
        ids = [item.dataset_id for item in self.selected_datasets]
        if 1464 not in ids or ids != sorted(set(ids)):
            raise ValueError(
                "selected datasets must uniquely include anchor 1464 in canonical order"
            )
        if len(self.excluded_datasets) != 6:
            raise ValueError("all six unselected registry datasets must be explained")
        excluded_ids = [item.dataset_id for item in self.excluded_datasets]
        if (
            excluded_ids != sorted(set(excluded_ids))
            or set(ids).union(excluded_ids) != set(SCHEMAORBIT_IDS)
            or set(ids).intersection(excluded_ids)
        ):
            raise ValueError(
                "selected and excluded IDs must partition the frozen SchemaOrbit-14 registry"
            )
        covered = set().union(*(set(item.coverage_tags) for item in self.selected_datasets))
        if not set(self.dataset_selection_policy.required_coverage).issubset(covered):
            raise ValueError(
                "selected dataset cohort does not satisfy the frozen diversity coverage"
            )
        if self.seed_selection.seeds != (1729, 2718, 31415):
            raise ValueError("protocol must preserve the first three canonical seeds")
        if len(self.seed_selection.seeds) != 3 or len(self.view_registry) != 11:
            raise ValueError("protocol must freeze three seeds and all eleven views")
        if tuple(item.view_id for item in self.view_registry) != VIEW_ORDER:
            raise ValueError("protocol must preserve all eleven registered view IDs in order")
        if tuple(item.model_id for item in self.model_matrix) != MODEL_ORDER:
            raise ValueError("protocol model matrix must retain canonical model order")
        if set(self.runtime_estimate.estimated_work_seconds_by_dataset) != set(ids):
            raise ValueError("runtime work estimates must cover exactly the selected datasets")
        expected_staging = (
            self.runtime_estimate.conservative_wall_seconds
            > self.resource_policy.preferred_wall_seconds
            or self.runtime_estimate.estimated_cache_storage_bytes
            > self.resource_policy.preferred_storage_bytes
        )
        if self.runtime_estimate.staged_schedule_required != expected_staging:
            raise ValueError(
                "staging requirement must follow the frozen runtime and storage limits"
            )
        if set(self.registry_parameter_decisions) != {
            "catboost_yaml_no_scalar",
            "tabicl_checkpoint_version_from_checkpoint_field",
        }:
            raise ValueError(
                "both documented frozen registry representation decisions are required"
            )
        if self.protocol_sha256 != sha256_canonical_json(
            self.model_dump(mode="json", exclude={"protocol_sha256", "condition_inventory_sha256"})
        ):
            raise ValueError("pilot protocol hash is not canonical")
        for relative in self.source_hashes:
            validate_relative_path(relative)
        if not self.source_hashes:
            raise ValueError("pilot protocol requires path-keyed source hashes")
        return self


class ConditionIdentityComponents(StrictPilotModel):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["pilot_prediction"] = "pilot_prediction"
    protocol_version: str
    pilot_protocol_sha256: Sha256
    dataset_id: int = Field(gt=0)
    raw_source_sha256: Sha256
    processed_feature_sha256: Sha256
    target_artifact_sha256: Sha256
    split_strategy: Literal["stratified_group_5fold_v1"]
    seed: int = Field(ge=0)
    split_assignment_sha256: Sha256
    split_logical_sha256: Sha256
    view_id: ViewId
    transformation_config_sha256: Sha256
    transformation_implementation_sha256: Sha256
    certificate_identity_sha256: Sha256
    model_id: ModelId
    package_name: str
    package_version: str
    model_config_sha256: Sha256
    complete_parameters_sha256: Sha256
    preprocessing_sha256: Sha256
    checkpoint_identifier: str | None
    checkpoint_sha256: Sha256 | None
    device_policy: Literal["cpu", "cuda"]
    precision_policy_sha256: Sha256
    batch_policy_sha256: Sha256
    dependency_lock_sha256: Sha256
    source_implementation_sha256: Sha256
    metric_policy_sha256: Sha256
    numerical_tolerance_policy_sha256: Sha256
    offline_policy: Literal[True]
    schema_version_identity: Literal[1] = 1

    @model_validator(mode="after")
    def validate_checkpoint_identity(self) -> ConditionIdentityComponents:
        if (self.checkpoint_identifier is None) != (self.checkpoint_sha256 is None):
            raise ValueError("checkpoint identifier and hash must be present or absent together")
        return self


class PlannedCondition(StrictPilotModel):
    condition_id: Sha256
    identity: ConditionIdentityComponents
    dataset_id: int = Field(gt=0)
    seed: int = Field(ge=0)
    view_id: ViewId
    model_id: ModelId
    device: Literal["cpu", "cuda"]
    applicability: Literal["APPLICABLE"] = "APPLICABLE"

    @model_validator(mode="after")
    def validate_identity(self) -> PlannedCondition:
        if (
            self.identity.dataset_id != self.dataset_id
            or self.identity.seed != self.seed
            or self.identity.view_id != self.view_id
            or self.identity.model_id != self.model_id
            or self.identity.device_policy != self.device
        ):
            raise ValueError("condition fields disagree with their identity payload")
        if self.condition_id != sha256_canonical_json(self.identity.model_dump(mode="json")):
            raise ValueError("condition identity does not match canonical payload")
        return self


class NotApplicableConditionTuple(StrictPilotModel):
    dataset_id: int = Field(gt=0)
    seed: int = Field(ge=0)
    view_id: ViewId
    reason_code: str = Field(min_length=1)
    evidence_sha256: Sha256
    transformation_inventory_record_sha256: Sha256


class PilotConditionInventory(StrictPilotModel):
    schema_version: Literal[1] = 1
    protocol_sha256: Sha256
    records: tuple[PlannedCondition, ...]
    not_applicable_views: tuple[NotApplicableConditionTuple, ...]
    maximum_possible_view_tuples: Literal[264] = 264
    maximum_possible_model_conditions: Literal[1320] = 1320
    selected_dataset_ids: tuple[int, ...] = Field(min_length=8, max_length=8)
    selected_seeds: tuple[int, int, int]
    actual_view_tuple_count: int = Field(ge=0, le=264)
    applicable_view_tuple_count: int = Field(ge=0, le=264)
    not_applicable_view_count: int = Field(ge=0, le=264)
    actual_condition_count: int = Field(ge=0, le=1320)
    cpu_condition_count: int = Field(ge=0, le=1320)
    cuda_condition_count: int = Field(ge=0, le=1320)
    counts_by_dataset: dict[str, int]
    counts_by_seed: dict[str, int]
    counts_by_view: dict[str, int]
    counts_by_model: dict[ModelId, int]
    source_hashes: dict[str, Sha256]
    inventory_sha256: Sha256

    @model_validator(mode="after")
    def validate_counts(self) -> PilotConditionInventory:
        conditions = self.records
        identities = [item.condition_id for item in conditions]
        if len(identities) != len(set(identities)):
            raise ValueError("condition identities must be unique")
        if len(conditions) != self.actual_condition_count:
            raise ValueError("condition count disagrees with records")
        if self.cpu_condition_count + self.cuda_condition_count != self.actual_condition_count:
            raise ValueError("CPU and CUDA counts must account for all conditions")
        if sum(item.device == "cpu" for item in conditions) != self.cpu_condition_count:
            raise ValueError("CPU condition count is not derived from records")
        if sum(item.device == "cuda" for item in conditions) != self.cuda_condition_count:
            raise ValueError("CUDA condition count is not derived from records")
        if self.selected_dataset_ids != tuple(sorted(set(self.selected_dataset_ids))):
            raise ValueError("selected dataset IDs must be sorted and unique")
        if self.selected_seeds != (1729, 2718, 31415):
            raise ValueError("inventory must use the first three canonical split-generation seeds")
        for relative in self.source_hashes:
            validate_relative_path(relative)
        if not self.source_hashes:
            raise ValueError("condition inventory requires path-keyed source hashes")
        expected_v00 = {
            (dataset_id, seed, model_id)
            for dataset_id in self.selected_dataset_ids
            for seed in self.selected_seeds
            for model_id in MODEL_ORDER
        }
        observed_v00 = {
            (item.dataset_id, item.seed, item.model_id)
            for item in conditions
            if item.view_id == "V00"
        }
        if observed_v00 != expected_v00:
            raise ValueError("V00 must retain all five model conditions for every selected pair")
        na_keys = [(item.dataset_id, item.seed, item.view_id) for item in self.not_applicable_views]
        if len(na_keys) != len(set(na_keys)):
            raise ValueError("NOT_APPLICABLE view tuples must be unique")
        applicable_keys = {(item.dataset_id, item.seed, item.view_id) for item in conditions}
        if applicable_keys.intersection(na_keys):
            raise ValueError("a view tuple cannot be both applicable and not applicable")
        if (
            self.actual_view_tuple_count
            != len(self.selected_dataset_ids) * len(self.selected_seeds) * 11
        ):
            raise ValueError("the complete dataset-seed-view matrix must be represented")
        expected_applicable = {(item.dataset_id, item.seed, item.view_id) for item in conditions}
        if len(expected_applicable) != self.applicable_view_tuple_count:
            raise ValueError("applicable view tuple count must be derived from conditions")
        models_by_view: dict[tuple[int, int, str], set[str]] = {}
        for item in conditions:
            models_by_view.setdefault((item.dataset_id, item.seed, item.view_id), set()).add(
                item.model_id
            )
        if any(models != set(MODEL_ORDER) for models in models_by_view.values()):
            raise ValueError("every applicable view tuple must retain all five model conditions")
        if (
            self.actual_view_tuple_count
            != self.applicable_view_tuple_count + self.not_applicable_view_count
        ):
            raise ValueError("view applicability totals do not reconcile")
        if self.not_applicable_view_count != len(na_keys):
            raise ValueError("NOT_APPLICABLE count must be derived from reason-coded records")
        expected_counts = {
            "counts_by_dataset": {
                str(key): sum(item.dataset_id == key for item in conditions)
                for key in self.selected_dataset_ids
            },
            "counts_by_seed": {
                str(key): sum(item.seed == key for item in conditions)
                for key in self.selected_seeds
            },
            "counts_by_view": {
                view: sum(item.view_id == view for item in conditions) for view in VIEW_ORDER
            },
            "counts_by_model": {
                model: sum(item.model_id == model for item in conditions) for model in MODEL_ORDER
            },
        }
        if any(getattr(self, key) != value for key, value in expected_counts.items()):
            raise ValueError("condition inventory summary counts are not derived from records")
        if self.inventory_sha256 != sha256_canonical_json(
            self.model_dump(mode="json", exclude={"inventory_sha256"})
        ):
            raise ValueError("condition inventory checksum does not match its canonical payload")
        return self


class PilotScheduleStage(StrictPilotModel):
    stage_id: str = Field(pattern=r"^dataset_group_[0-9]+_[0-9]+$")
    dataset_ids: tuple[int, int]
    condition_ids: tuple[Sha256, ...] = Field(min_length=1)
    model_condition_counts: dict[ModelId, int]
    cpu_condition_count: int = Field(ge=0)
    cuda_condition_count: int = Field(ge=0)
    estimated_cpu_seconds: float = Field(ge=0, allow_inf_nan=False)
    estimated_cuda_seconds: float = Field(ge=0, allow_inf_nan=False)
    conservative_wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    estimated_prediction_storage_bytes: int = Field(ge=0)
    estimated_cache_storage_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_stage_counts(self) -> PilotScheduleStage:
        if self.dataset_ids != tuple(sorted(set(self.dataset_ids))):
            raise ValueError("stage dataset IDs must be ascending and unique")
        if set(self.model_condition_counts) != set(MODEL_ORDER):
            raise ValueError("stage must report counts for exactly the five frozen models")
        if len(self.condition_ids) != len(set(self.condition_ids)):
            raise ValueError("condition IDs must be unique within a stage")
        if sum(self.model_condition_counts.values()) != len(self.condition_ids):
            raise ValueError("stage model counts must account for every condition identity")
        cpu_models: tuple[ModelId, ...] = ("LR-1.9", "CAT-1.2", "XGB-3.4")
        cuda_models: tuple[ModelId, ...] = ("TPFN3-8.5", "TICL2-2.2")
        if (
            sum(self.model_condition_counts[item] for item in cpu_models)
            != self.cpu_condition_count
        ):
            raise ValueError("stage CPU count disagrees with its model counts")
        if (
            sum(self.model_condition_counts[item] for item in cuda_models)
            != self.cuda_condition_count
        ):
            raise ValueError("stage CUDA count disagrees with its model counts")
        return self


class PilotStagedSchedule(StrictPilotModel):
    schema_version: Literal[1] = 1
    schedule_sha256: Sha256
    protocol_sha256: Sha256
    inventory_sha256: Sha256
    schedule_policy: Literal["contiguous_pairs_by_ascending_dataset_id"]
    dataset_ids: tuple[int, ...] = Field(min_length=8, max_length=8)
    stages: tuple[PilotScheduleStage, ...] = Field(min_length=1)
    total_condition_count: int = Field(ge=1)
    not_applicable_view_count: int = Field(ge=0)
    total_estimated_wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    hard_wall_limit_seconds: int = Field(gt=0)
    all_stages_within_hard_wall_limit: bool
    estimated_prediction_storage_bytes: int = Field(ge=0)
    estimated_cache_storage_bytes: int = Field(ge=0)
    full_condition_matrix_preserved: Literal[True]

    @model_validator(mode="after")
    def validate_schedule(self) -> PilotStagedSchedule:
        if self.dataset_ids != tuple(sorted(set(self.dataset_ids))):
            raise ValueError("schedule dataset IDs must be ascending and unique")
        staged_datasets = tuple(
            dataset_id for stage in self.stages for dataset_id in stage.dataset_ids
        )
        expected_groups = tuple(
            self.dataset_ids[index : index + 2] for index in range(0, len(self.dataset_ids), 2)
        )
        if staged_datasets != self.dataset_ids or tuple(
            stage.dataset_ids for stage in self.stages
        ) != expected_groups:
            raise ValueError("stages must preserve contiguous canonical dataset pairs")
        condition_ids = [
            condition_id for stage in self.stages for condition_id in stage.condition_ids
        ]
        if len(condition_ids) != self.total_condition_count or len(condition_ids) != len(
            set(condition_ids)
        ):
            raise ValueError("stages must account for every unique planned condition exactly once")
        if sum(stage.estimated_prediction_storage_bytes for stage in self.stages) != (
            self.estimated_prediction_storage_bytes
        ):
            raise ValueError("stage prediction storage must sum to the schedule total")
        if sum(stage.estimated_cache_storage_bytes for stage in self.stages) != (
            self.estimated_cache_storage_bytes
        ):
            raise ValueError("stage cache storage must sum to the schedule total")
        if not math.isclose(
            sum(stage.conservative_wall_seconds for stage in self.stages),
            self.total_estimated_wall_seconds,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("stage wall times must sum to the schedule total")
        actual_hard_limit_status = all(
            stage.conservative_wall_seconds <= self.hard_wall_limit_seconds
            for stage in self.stages
        )
        if self.all_stages_within_hard_wall_limit != actual_hard_limit_status:
            raise ValueError("stage hard-limit status disagrees with the stage estimates")
        expected_hash = sha256_canonical_json(
            self.model_dump(mode="json", exclude={"schedule_sha256"})
        )
        if self.schedule_sha256 != expected_hash:
            raise ValueError("staged schedule hash does not match its canonical payload")
        return self


class MetricEvaluationRecord(StrictPilotModel):
    value: float | None = Field(default=None, allow_inf_nan=False)
    status: Literal["DEFINED", "UNDEFINED"]
    reason: str | None = None

    @model_validator(mode="after")
    def validate_defined(self) -> MetricEvaluationRecord:
        if (self.status == "DEFINED") != (self.value is not None):
            raise ValueError("metric value must be present if and only if it is defined")
        if self.status == "UNDEFINED" and not self.reason:
            raise ValueError("undefined metric requires an explicit reason")
        if self.status == "DEFINED" and self.reason is not None:
            raise ValueError("defined metric cannot contain an undefined reason")
        return self


class LeakageEvent(StrictPilotModel):
    event: Literal[
        "plan_frozen",
        "test_labels_sealed",
        "training_completed",
        "calibration_predictions_completed",
        "test_predictions_completed",
        "prediction_structure_validated",
        "test_labels_opened",
        "metrics_generated",
        "results_sealed",
    ]
    sequence: int = Field(ge=0)
    evidence_sha256: Sha256


class ProtocolValidationGate(StrictPilotModel):
    gate_id: str = Field(pattern=r"^PF[0-9]{2}$")
    result: GateResult
    evidence_sha256: Sha256
    detail: str


class PilotProtocolValidation(StrictPilotModel):
    schema_version: Literal[1] = 1
    status: PilotStatus
    source_commit: CommitSha
    protocol_sha256: Sha256
    condition_inventory_sha256: Sha256
    gate_count: Literal[40] = 40
    passed_count: int = Field(ge=0, le=40)
    failed_count: int = Field(ge=0, le=40)
    not_verified_count: int = Field(ge=0, le=40)
    gates: tuple[ProtocolValidationGate, ...] = Field(min_length=40, max_length=40)
    pilot_execution_performed: Literal[False] = False
    test_labels_accessed: Literal[False] = False

    @model_validator(mode="after")
    def validate_gate_summary(self) -> PilotProtocolValidation:
        expected = tuple(f"PF{index:02d}" for index in range(1, 41))
        if tuple(item.gate_id for item in self.gates) != expected:
            raise ValueError("all forty gates must be present in canonical order")
        counts = {
            "passed_count": sum(item.result == "PASS" for item in self.gates),
            "failed_count": sum(item.result == "FAIL" for item in self.gates),
            "not_verified_count": sum(item.result == "NOT_VERIFIED" for item in self.gates),
        }
        for name, value in counts.items():
            if getattr(self, name) != value:
                raise ValueError("gate summary counts disagree with gate records")
        return self


def canonical_identity_hash(identity: ConditionIdentityComponents) -> str:
    return sha256_canonical_json(identity.model_dump(mode="json"))


__all__ = [
    "MODEL_ORDER",
    "PILOT_OUTCOMES",
    "VIEW_ORDER",
    "ConditionIdentityComponents",
    "ExcludedDataset",
    "LeakageEvent",
    "MetricEvaluationRecord",
    "MetricPolicy",
    "NotApplicableConditionTuple",
    "PilotConditionInventory",
    "PilotConfig",
    "PilotDatasetSelection",
    "PilotModelEntry",
    "PilotProtocolArtifact",
    "PilotProtocolValidation",
    "PilotScheduleStage",
    "PilotSeedSelection",
    "PilotStagedSchedule",
    "StagingPolicy",
    "PilotViewRegistryEntry",
    "PlannedCondition",
    "ProtocolValidationGate",
    "RuntimeEstimate",
    "SplitIdentity",
    "canonical_identity_hash",
]
