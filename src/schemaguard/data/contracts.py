"""Strict Pydantic contracts for configuration and generated data records."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictContract(BaseModel):
    """Base contract with deterministic, closed-world validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: int = Field(default=1, ge=1)

    def canonical_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible deterministic representation."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=False)


class DatasetSourceConfig(StrictContract):
    internal_id: str
    provider: Literal["openml"]
    openml_data_id: int = Field(gt=0)
    openml_file_id: int = Field(gt=0)
    expected_name: str
    source_page: str
    metadata_url: str
    download_url: str
    raw_filename: str

    @field_validator("raw_filename")
    @classmethod
    def validate_raw_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or path.name != value or value in {"", ".", ".."}:
            raise ValueError("raw_filename must be a relative file name")
        if not value.endswith(".arff"):
            raise ValueError("raw_filename must use the .arff extension")
        return value


class TaskConfig(StrictContract):
    type: Literal["binary_classification"]
    target_source: Literal["openml_default_target"]
    expected_target_name: str


class ExpectedDatasetProperties(StrictContract):
    rows: int = Field(gt=0)
    predictor_columns: int = Field(gt=0)
    total_columns: int = Field(gt=0)
    classes: int = Field(gt=1)
    numeric_predictors: int = Field(ge=0)
    categorical_predictors: int = Field(ge=0)
    missing_values: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> ExpectedDatasetProperties:
        if self.numeric_predictors + self.categorical_predictors != self.predictor_columns:
            raise ValueError(
                "numeric_predictors + categorical_predictors must equal predictor_columns"
            )
        if self.total_columns != self.predictor_columns + 1:
            raise ValueError("total_columns must equal predictor_columns + target")
        return self


class ProcessingConfig(StrictContract):
    preserve_feature_names: bool
    preserve_row_order: bool
    preserve_duplicate_rows: bool
    impute_missing: bool
    scale_numeric: bool
    normalize_rows: bool
    encode_categorical: bool
    remove_outliers: bool
    balance_classes: bool
    drop_constant_columns: bool
    parquet_compression: Literal["zstd"]
    parquet_compression_level: int = Field(ge=1, le=22)

    @model_validator(mode="after")
    def validate_conservative_policy(self) -> ProcessingConfig:
        forbidden = {
            "impute_missing": self.impute_missing,
            "scale_numeric": self.scale_numeric,
            "normalize_rows": self.normalize_rows,
            "encode_categorical": self.encode_categorical,
            "remove_outliers": self.remove_outliers,
            "balance_classes": self.balance_classes,
            "drop_constant_columns": self.drop_constant_columns,
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(f"Conservative processing forbids enabled operations: {enabled}")
        if not (
            self.preserve_feature_names and self.preserve_row_order and self.preserve_duplicate_rows
        ):
            raise ValueError("Feature names, row order, and duplicate rows must be preserved")
        return self


class SplitConfig(StrictContract):
    master_seed: int = Field(ge=0, le=2**32 - 1)
    train_fraction: float = Field(gt=0, lt=1)
    calibration_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)
    stratified: Literal[True]
    strategy: Literal["stratified_group_5fold_v1"]
    group_by: Literal["predictors"]
    group_folds: Literal[5]

    @model_validator(mode="after")
    def validate_fractions(self) -> SplitConfig:
        total = self.train_fraction + self.calibration_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to 1.0 within 1e-12")
        return self


class ResourceConfig(StrictContract):
    download_chunk_bytes: int = Field(gt=0)
    download_attempts: int = Field(ge=1, le=3)
    request_timeout_seconds: float = Field(gt=0)
    lock_timeout_seconds: float = Field(gt=0)


class ValidationConfig(StrictContract):
    reject_missing_target: bool
    reject_infinite_numeric_values: bool
    reject_duplicate_column_names: bool
    reject_unexpected_shape: bool
    reject_unexpected_target: bool
    reject_unexpected_class_count: bool
    report_duplicate_rows: bool
    probability_data_not_applicable: bool


class SmokeDatasetConfig(StrictContract):
    dataset: DatasetSourceConfig
    task: TaskConfig
    expected: ExpectedDatasetProperties
    processing: ProcessingConfig
    split: SplitConfig
    resources: ResourceConfig
    validation: ValidationConfig


class SourceManifest(StrictContract):
    internal_dataset_id: str
    provider: Literal["openml"]
    openml_data_id: int
    openml_file_id: int
    dataset_name: str
    dataset_version: str
    source_page: str
    requested_download_url: str
    resolved_download_url: str
    metadata_url: str
    data_format: Literal["ARFF"]
    default_target_attribute: str
    provider_md5: str | None
    computed_md5: str
    computed_sha256: str
    file_size_bytes: int = Field(ge=0)
    raw_relative_path: str
    retrieved_at_utc: datetime
    http_etag: str | None = None
    http_last_modified: str | None = None
    package_versions: dict[str, str]
    cache_status: Literal["downloaded", "hit"]

    @field_validator("retrieved_at_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("raw_relative_path")
    @classmethod
    def require_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "raw_relative_path must be relative and stay inside the data directory"
            )
        return value


class AttributeSchema(StrictContract):
    name: str
    raw_type: str
    kind: Literal["numeric", "categorical"]
    is_target: bool
    position: int = Field(ge=0)


class ProcessedSchema(StrictContract):
    internal_dataset_id: str
    raw_sha256: str
    row_id_column: Literal["__sg_row_id"]
    target_column: str
    feature_columns: list[str]
    attributes: list[AttributeSchema]
    feature_row_count: int = Field(ge=0)
    target_row_count: int = Field(ge=0)


class QualityCheck(StrictContract):
    name: str
    status: Literal["passed", "failed", "warning", "not_applicable"]
    blocking: bool
    details: dict[str, Any] = Field(default_factory=dict)


class QualityReport(StrictContract):
    internal_dataset_id: str
    checks: list[QualityCheck]
    final_status: Literal["PASS", "FAILED"]
    warnings: list[str] = Field(default_factory=list)


class SplitManifest(StrictContract):
    internal_dataset_id: str
    data_manifest_hash: str
    master_seed: int
    derived_seeds: dict[str, int]
    split_fractions: dict[str, float]
    split_algorithm: str
    sklearn_version: str
    row_counts: dict[str, int]
    class_counts_by_split: dict[str, dict[str, int]]
    strategy: Literal["stratified_group_5fold_v1"]
    group_by: Literal["predictors"]
    group_folds: Literal[5]
    total_predictor_groups: int = Field(ge=1)
    duplicate_predictor_groups: int = Field(ge=0)
    largest_group_size: int = Field(ge=1)
    conflicting_target_groups: int = Field(ge=0)
    predictor_duplicate_groups_crossing_splits: int = Field(ge=0)
    size_deviations: dict[str, float]
    class_proportion_deviations: dict[str, dict[str, float]]
    fold_assignment: dict[str, list[int]]
    selection_score: dict[str, float]
    assignment_file_sha256: str
    created_at_utc: datetime
    validation_status: Literal["PASS", "FAILED"]

    @field_validator("created_at_utc")
    @classmethod
    def require_utc_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value.astimezone(UTC)


class ValidationSummary(StrictContract):
    stage: str
    status: Literal["PASS", "FAILED"]
    dataset_identity: dict[str, Any]
    source_verification: dict[str, Any]
    raw_file_hashes: dict[str, str]
    parsed_dimensions: dict[str, int]
    feature_type_counts: dict[str, int]
    missing_value_counts: dict[str, int]
    target_labels: list[str]
    target_class_frequencies: dict[str, int]
    duplicate_row_counts: dict[str, int]
    processed_artifact_hashes: dict[str, str]
    row_id_checks: dict[str, bool]
    split_sizes: dict[str, int]
    class_counts_by_split: dict[str, dict[str, int]]
    split_overlap_checks: dict[str, bool]
    deterministic_rerun_check: bool
    offline_rerun_check: bool
    unit_test_result: dict[str, Any]
    network_test_result: dict[str, Any]
    ruff_result: dict[str, Any]
    mypy_result: dict[str, Any]
    blocking_failures: list[str]
    warnings: list[str]
    artifact_inventory: list[dict[str, Any]]


class PhaseResult(StrictContract):
    stage: str
    status: Literal["PASS", "FAILED"]
    dataset_id: str
    failed_stage: str | None = None
    message: str | None = None
    completed_stages: list[str] = Field(default_factory=list)
    created_at_utc: datetime

    @field_validator("created_at_utc")
    @classmethod
    def require_phase_time_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value.astimezone(UTC)


# These closed-world contracts describe the dataset-registry artifacts.  They
# deliberately live beside the frozen smoke-data contracts so that both data
# workstreams use Pydantic as their authoritative artifact schema.
class DatasetSpecContract(StrictContract):
    id: int = Field(gt=0)
    name: str
    file_id: int = Field(gt=0)
    version: str
    rows: int = Field(gt=0)
    predictors: int = Field(gt=0)
    target: str
    provider_md5: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    ignore_attributes: list[str] = Field(default_factory=list)


class AcquisitionRulesContract(StrictContract):
    seed: int
    offline_default: bool
    require_version: str
    require_public: bool
    require_active: bool
    require_default_target: bool
    require_stable_checksum: bool
    min_rows: int = Field(gt=0)
    max_rows: int = Field(gt=0)
    min_predictors: int = Field(gt=0)
    max_predictors: int = Field(gt=0)
    min_classes: int = Field(ge=2)
    max_classes: int = Field(ge=2)
    parquet_compression: Literal["zstd"]
    download_attempts: int = Field(ge=1, le=3)
    request_timeout_seconds: float = Field(gt=0)
    retry_backoff_seconds: list[float]

    @model_validator(mode="after")
    def validate_ranges(self) -> AcquisitionRulesContract:
        if self.min_rows > self.max_rows or self.min_predictors > self.max_predictors:
            raise ValueError("acquisition ranges must be ordered")
        if self.min_classes > self.max_classes:
            raise ValueError("class ranges must be ordered")
        if len(self.retry_backoff_seconds) != self.download_attempts:
            raise ValueError("one retry backoff value is required per download attempt")
        if any(value < 0 for value in self.retry_backoff_seconds):
            raise ValueError("retry backoff values must be nonnegative")
        return self


class SchemaOrbitConfigContract(StrictContract):
    benchmark: Literal["SchemaOrbit-14"]
    openml_api_base: str
    download_base: str
    acquisition: AcquisitionRulesContract
    datasets: list[DatasetSpecContract]

    @model_validator(mode="after")
    def validate_registry(self) -> SchemaOrbitConfigContract:
        identifiers = [item.id for item in self.datasets]
        if len(identifiers) != 14 or len(set(identifiers)) != 14:
            raise ValueError("SchemaOrbit-14 requires fourteen unique datasets")
        return self


class OpenMLMetadataContract(StrictContract):
    data_id: int = Field(gt=0)
    file_id: int = Field(gt=0)
    name: str
    version: str
    format: Literal["ARFF"]
    default_target_attribute: str | None
    status: str
    visibility: str
    licence: str | None
    md5_checksum: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")] | None
    metadata_url: str
    download_url: str
    raw: dict[str, Any]


class FeatureMetadataContract(StrictContract):
    name: str
    data_type: str
    is_target: bool
    is_ignore: bool
    is_row_identifier: bool


class RawSourceManifestContract(StrictContract):
    cache_status: Literal["downloaded", "hit"]
    provider: Literal["openml"]
    openml_data_id: int = Field(gt=0)
    openml_file_id: int = Field(gt=0)
    dataset_name: str
    dataset_version: str
    data_format: Literal["ARFF"]
    default_target_attribute: str | None
    metadata_url: str
    requested_download_url: str
    resolved_download_url: str
    source_page: str | None = None
    provider_md5: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    computed_md5: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    computed_sha256: Sha256
    file_size_bytes: int = Field(ge=0)
    raw_relative_path: str
    retrieved_at_utc: str
    http_status: int | None = Field(default=None, ge=100, le=599)
    content_length_bytes: int | None = Field(default=None, ge=0)
    http_etag: str | None = None
    http_last_modified: str | None = None
    package_versions: dict[str, str]
    attempts: int = Field(default=1, ge=1, le=3)

    @model_validator(mode="after")
    def validate_checksum(self) -> RawSourceManifestContract:
        if self.provider_md5 != self.computed_md5:
            raise ValueError("provider and computed MD5 must match")
        if (
            self.content_length_bytes is not None
            and self.content_length_bytes != self.file_size_bytes
        ):
            raise ValueError("content length and file size must match")
        return self


class DatasetAttributeContract(StrictContract):
    name: str
    raw_type: str
    kind: Literal["numeric", "categorical"]
    position: int = Field(ge=0)


class DatasetSchemaContract(StrictContract):
    internal_dataset_id: str
    openml_data_id: int = Field(gt=0)
    target_column: str
    row_id_column: Literal["__sg_row_id"]
    feature_columns: list[str]
    feature_row_count: int = Field(ge=0)
    target_row_count: int = Field(ge=0)
    raw_sha256: Sha256
    attributes: list[DatasetAttributeContract]

    @model_validator(mode="after")
    def validate_schema(self) -> DatasetSchemaContract:
        if [item.position for item in self.attributes] != list(range(len(self.attributes))):
            raise ValueError("attribute positions must be contiguous")
        if [item.name for item in self.attributes] != self.feature_columns:
            raise ValueError("attribute and feature ordering must match")
        if self.feature_row_count != self.target_row_count:
            raise ValueError("feature and target row counts must match")
        return self


class LabelMappingContract(StrictContract):
    raw_sha256: Sha256
    original_to_code: dict[str, int]
    code_to_original: dict[str, str]

    @model_validator(mode="after")
    def validate_mapping(self) -> LabelMappingContract:
        if set(self.code_to_original) != {str(value) for value in self.original_to_code.values()}:
            raise ValueError("label mapping codes are not reversible")
        if any(value < 0 for value in self.original_to_code.values()):
            raise ValueError("target codes must be nonnegative")
        for label, code in self.original_to_code.items():
            if self.code_to_original.get(str(code)) != label:
                raise ValueError("label mapping is not an inverse mapping")
        return self


class ProcessedManifestContract(StrictContract):
    internal_dataset_id: str
    openml_data_id: int = Field(gt=0)
    row_count: int = Field(ge=0)
    feature_columns: list[str]
    target_columns: list[str]
    raw_sha256: Sha256
    source_manifest_sha256: Sha256
    processing_config_sha256: Sha256
    row_id_formula: str
    artifact_hashes: dict[str, Sha256]
    compression: Literal["zstd"]
    compression_level: int = Field(ge=1, le=22)

    @model_validator(mode="after")
    def validate_artifacts(self) -> ProcessedManifestContract:
        required = {
            "features.parquet",
            "targets.parquet",
            "schema.json",
            "label_mapping.json",
            "quality_report.json",
        }
        if set(self.artifact_hashes) != required:
            raise ValueError("processed manifest must hash exactly the four data artifacts")
        return self


class DatasetQualityReportContract(StrictContract):
    row_count: int = Field(ge=0)
    predictor_count: int = Field(ge=1)
    class_count: int = Field(ge=2)
    class_counts: dict[str, int]
    missing_cells: int = Field(ge=0)
    missing_by_column: dict[str, int]
    duplicate_predictor_groups: int = Field(ge=0)
    conflicting_target_groups: int = Field(ge=0)
    exact_duplicate_rows: int = Field(ge=0)
    predictor_hash: Sha256
    grouping_algorithm: Literal["typed_predictor_sha256_v1"]


class DatasetInventoryRecordContract(StrictContract):
    openml_data_id: int = Field(gt=0)
    dataset_name: str
    openml_file_id: int = Field(gt=0)
    dataset_version: str
    provider_md5: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    raw_sha256: Sha256
    raw_size_bytes: int = Field(ge=0)
    metadata_url: str
    download_url: str
    default_target: str | None
    ignored_attributes: list[str]
    feature_metadata: list[FeatureMetadataContract]
    expected_rows: int = Field(gt=0)
    observed_rows: int = Field(gt=0)
    expected_predictors: int = Field(gt=0)
    observed_predictors: int = Field(gt=0)
    class_count: int = Field(ge=2)
    class_counts: dict[str, int]
    quality: DatasetQualityReportContract
    license: str | None
    visibility: str
    status: Literal["PASS", "FAIL", "BLOCKED"]


class DatasetInventoryContract(StrictContract):
    benchmark: Literal["SchemaOrbit-14"]
    dataset_count: int = Field(ge=0)
    datasets: list[DatasetInventoryRecordContract]
    data_foundation: dict[str, Any]

    @model_validator(mode="after")
    def validate_count(self) -> DatasetInventoryContract:
        if self.dataset_count != len(self.datasets):
            raise ValueError("dataset_count does not match datasets")
        return self


class DatasetValidationRecordContract(StrictContract):
    openml_data_id: int = Field(gt=0)
    dataset_name: str
    dataset_version: str
    rows_expected: int = Field(gt=0)
    rows_observed: int = Field(gt=0)
    predictors_expected: int = Field(gt=0)
    predictors_observed: int = Field(gt=0)
    classes: int = Field(ge=2)
    class_counts: str
    ignored_attributes: str
    raw_sha256: Sha256
    provider_md5: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    missing_cells: int = Field(ge=0)
    duplicate_predictor_groups: int = Field(ge=0)
    conflicting_target_groups: int = Field(ge=0)
    status: Literal["PASS", "FAIL", "BLOCKED"]
