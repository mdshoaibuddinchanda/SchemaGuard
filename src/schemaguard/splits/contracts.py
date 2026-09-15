"""Closed-world contracts for the deterministic split workstream."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from schemaguard.utils.hashing import sha256_canonical_json

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictSplitContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: Literal[1] = 1

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)


class SplitGenerationConfig(StrictSplitContract):
    strategy: Literal["stratified_group_5fold_v1"]
    grouping_method: Literal["typed_predictor_sha256_v1"]
    group_by: Literal["predictors"]
    group_folds: Literal[5]
    train_fraction: float = Field(gt=0, lt=1)
    calibration_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)
    minimum_class_count_per_partition: int = Field(ge=1)
    seeds: list[int] = Field(min_length=1)
    datasets: list[int] = Field(min_length=1)
    max_workers: Literal[1, 2] = 2

    @model_validator(mode="after")
    def validate_frozen_values(self) -> SplitGenerationConfig:
        if abs(self.train_fraction + self.calibration_fraction + self.test_fraction - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to one")
        if self.seeds != [1729, 2718, 31415, 57721, 161803]:
            raise ValueError("the frozen seed set must not be changed")
        if self.datasets != [3, 23, 29, 31, 36, 37, 38, 44, 46, 50, 54, 1067, 1464, 1489]:
            raise ValueError("the frozen SchemaOrbit-14 dataset set must not be changed")
        if len(set(self.seeds)) != len(self.seeds) or len(set(self.datasets)) != len(self.datasets):
            raise ValueError("seeds and datasets must be unique")
        return self

    @property
    def configuration_hash(self) -> str:
        return sha256_canonical_json(self.canonical_dict())


class SplitManifestContract(StrictSplitContract):
    dataset_id: int = Field(gt=0)
    dataset_version: str
    dataset_name: str
    seed: int = Field(ge=0)
    strategy: Literal["stratified_group_5fold_v1"]
    strategy_version: Literal["v1"]
    grouping_method: Literal["typed_predictor_sha256_v1"]
    train_fraction: float = Field(gt=0, lt=1)
    calibration_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)
    row_count: int = Field(gt=0)
    predictor_count: int = Field(gt=0)
    class_values: list[int] = Field(min_length=2)
    partition_counts: dict[str, int]
    partition_class_counts: dict[str, dict[str, int]]
    predictor_group_count: int = Field(gt=0)
    duplicate_group_count: int = Field(ge=0)
    conflicting_target_group_count: int = Field(ge=0)
    cross_partition_group_count: int = Field(ge=0)
    feature_artifact_hash: Sha256
    target_artifact_hash: Sha256
    dataset_manifest_hash: Sha256
    assignment_artifact_hash: Sha256
    logical_assignment_hash: Sha256
    configuration_hash: Sha256
    grouping_implementation_hash: Sha256
    split_implementation_hash: Sha256
    cache_identity_hash: Sha256
    source_commit: str
    created_at: str
    validation_status: Literal["PASS", "FAILED", "CONSTRAINT_INFEASIBLE"]
    fold_assignment: dict[str, list[int]]
    selection_score: dict[str, float]
    size_deviations: dict[str, float]
    class_proportion_deviations: dict[str, dict[str, float]]
    candidate_count: int = Field(ge=0)
    candidate_diagnostics: list[dict[str, Any]]
    determinism: dict[str, Any]
    protected_baseline: bool = False

    @model_validator(mode="after")
    def validate_partition_contract(self) -> SplitManifestContract:
        if len(set(self.class_values)) != len(self.class_values):
            raise ValueError("class_values must not contain duplicates")
        if sorted(self.class_values) != list(range(len(self.class_values))):
            raise ValueError("class_values must be contiguous target codes")
        if abs(self.train_fraction + self.calibration_fraction + self.test_fraction - 1.0) > 1e-12:
            raise ValueError("manifest fractions must sum to one")
        if set(self.partition_counts) != {"train", "calibration", "test"}:
            raise ValueError("manifest must contain exactly three partitions")
        if set(self.partition_class_counts) != set(self.partition_counts):
            raise ValueError("partition class counts do not cover all partitions")
        if sum(self.partition_counts.values()) != self.row_count:
            raise ValueError("partition counts do not sum to row count")
        return self


class PartitionStatistics(StrictSplitContract):
    partition: Literal["train", "calibration", "test"]
    row_count: int = Field(ge=0)
    class_counts: dict[str, int]


class PredictorGroupStatistics(StrictSplitContract):
    predictor_group_count: int = Field(ge=1)
    duplicate_group_count: int = Field(ge=0)
    conflicting_target_group_count: int = Field(ge=0)
    cross_partition_group_count: int = Field(ge=0)


class DeterminismResult(StrictSplitContract):
    same_seed_same_data: bool
    shuffled_input_same_logical_hash: bool
    distinct_seed_hashes: int = Field(ge=1)


class CacheIdentityContract(StrictSplitContract):
    dataset_id: int = Field(gt=0)
    dataset_version: str
    feature_artifact_hash: Sha256
    target_artifact_hash: Sha256
    dataset_manifest_hash: Sha256
    seed: int = Field(ge=0)
    strategy: str
    strategy_version: str
    grouping_implementation_hash: Sha256
    split_configuration_hash: Sha256
    split_implementation_hash: Sha256
    source_commit: str
    artifact_schema_version: int = Field(ge=1)


class ValidationResultContract(StrictSplitContract):
    status: Literal["PASS", "FAILED", "CONSTRAINT_INFEASIBLE"]
    dataset_id: int = Field(gt=0)
    seed: int = Field(ge=0)
    row_count: int = Field(ge=0)
    cross_partition_group_count: int = Field(ge=0)
    logical_assignment_hash: Sha256 | None = None


class FailureClassification(StrictSplitContract):
    category: Literal[
        "BLOCKED_DATASET_ARTIFACT",
        "BLOCKED_BASELINE_SPLIT_MISMATCH",
        "CONSTRAINT_INFEASIBLE",
        "FAIL_ASSIGNMENT",
        "FAIL_MANIFEST",
        "FAIL_CACHE",
        "FAIL_ATOMICITY",
    ]
    message: str


class SplitInventoryRecord(StrictSplitContract):
    dataset_id: int = Field(gt=0)
    seed: int = Field(ge=0)
    status: Literal["PASS", "FAILED", "CONSTRAINT_INFEASIBLE", "MISSING"]
    protected_baseline: bool
    assignments_path: str
    manifest_path: str
    assignment_artifact_hash: Sha256 | None = None
    manifest_artifact_hash: Sha256 | None = None
    logical_assignment_hash: Sha256 | None = None
    split_implementation_hash: Sha256 | None = None
    cache_identity_hash: Sha256 | None = None
    row_count: int | None = Field(default=None, ge=0)
    partition_counts: dict[str, int] | None = None
    partition_class_counts: dict[str, dict[str, int]] | None = None
    predictor_group_count: int | None = Field(default=None, ge=0)
    duplicate_group_count: int | None = Field(default=None, ge=0)
    conflicting_target_group_count: int | None = Field(default=None, ge=0)
    cross_partition_group_count: int | None = Field(default=None, ge=0)


class SplitGenerationInventoryContract(StrictSplitContract):
    generated_at: str
    configuration_hash: Sha256
    expected_dataset_ids: list[int]
    expected_seeds: list[int]
    records: list[SplitInventoryRecord]

    @model_validator(mode="after")
    def validate_inventory(self) -> SplitGenerationInventoryContract:
        expected = {
            (dataset_id, seed)
            for dataset_id in self.expected_dataset_ids
            for seed in self.expected_seeds
        }
        observed = {(record.dataset_id, record.seed) for record in self.records}
        if observed != expected:
            raise ValueError("inventory does not contain exactly every dataset-seed task")
        return self
