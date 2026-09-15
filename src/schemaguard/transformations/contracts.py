"""Closed-world Pydantic contracts for transformation evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.hashing import sha256_canonical_json


class StrictTransformationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: int = Field(default=1, ge=1)

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)


def _hash(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError("hash fields must be lowercase SHA-256 values")
    return value.lower()


class FeatureSchemaContract(StrictTransformationContract):
    columns: list[str]
    dtypes: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    integer_columns: list[str]
    row_id_column: str

    @model_validator(mode="after")
    def validate_schema(self) -> FeatureSchemaContract:
        if len(self.columns) != len(self.dtypes) or len(set(self.columns)) != len(self.columns):
            raise ValueError("feature schema columns and dtypes must be aligned and unique")
        for subset in (self.numeric_columns, self.categorical_columns, self.integer_columns):
            if not set(subset).issubset(self.columns):
                raise ValueError("feature schema subset contains an unknown column")
        if set(self.numeric_columns) & set(self.categorical_columns):
            raise ValueError("numeric and categorical columns must be disjoint")
        if self.row_id_column not in self.columns:
            raise ValueError("row id must be present in the feature schema")
        return self


class TransformationConfig(StrictTransformationContract):
    engine_name: Literal["certified_lossless_transformations"]
    split_strategy: Literal["stratified_group_5fold_v1"]
    max_numeric_columns: int = Field(ge=1, le=3)
    category_minimum: int = Field(ge=2)
    quotient_modulus: int = Field(ge=2)
    numerical_rtol: float = Field(ge=0)
    numerical_atol: float = Field(ge=0)
    cpu_workers: int = Field(ge=1, le=2)
    gpu_enabled: Literal[False] = False
    schema_registry_hash: str
    views: list[dict[str, Any]]

    @field_validator("schema_registry_hash")
    @classmethod
    def validate_registry_hash(cls, value: str) -> str:
        return _hash(value) or value

    @model_validator(mode="after")
    def validate_views(self) -> TransformationConfig:
        ids = [str(view.get("id")) for view in self.views]
        if len(ids) != 11 or len(set(ids)) != 11:
            raise ValueError("the transformation registry must contain exactly eleven views")
        return self


class ApplicabilityRecord(StrictTransformationContract):
    dataset_id: int | str
    seed: int
    view_id: str
    view_name: str
    status: Literal["APPLICABLE", "NOT_APPLICABLE"]
    reason_code: str | None = None
    reason: str | None = None
    feature_schema_hash: str
    created_at: str

    @model_validator(mode="after")
    def validate_reason(self) -> ApplicabilityRecord:
        if self.status == "NOT_APPLICABLE" and (not self.reason_code or not self.reason):
            raise ValueError("not-applicable records require a controlled reason")
        if self.status == "APPLICABLE" and self.reason_code is not None:
            raise ValueError("applicable records cannot carry a not-applicable reason")
        return self


class TransformationCertificate(StrictTransformationContract):
    view_id: str
    view_name: str
    certificate_type: Literal["BIJECTION", "PROJECTION", "CONTROL", "COMPOSITION"]
    dataset_id: int | str
    dataset_version: str
    seed: int
    split_strategy: str
    partition: Literal["train", "calibration", "test"]
    fit_scope: Literal["train_only", "none"]
    source_schema_hash: str
    output_schema_hash: str
    source_artifact_hash: str
    output_artifact_hash: str
    source_row_id_hash: str
    output_row_id_hash: str
    source_target_hash: str | None = None
    output_target_hash: str | None = None
    selected_columns: list[str] = Field(default_factory=list)
    generated_columns: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    inverse_parameters: dict[str, Any] = Field(default_factory=dict)
    missing_mask_policy: str
    dtype_policy: str
    numerical_tolerance: dict[str, float]
    configuration_hash: str
    implementation_hash: str
    source_commit: str
    validation_status: Literal["PASS", "FAIL", "N/A"]
    validation_results: dict[str, Any]
    not_applicable_reason: str | None = None
    created_at: str

    @field_validator(
        "source_schema_hash",
        "output_schema_hash",
        "source_artifact_hash",
        "output_artifact_hash",
        "source_row_id_hash",
        "output_row_id_hash",
        "source_target_hash",
        "output_target_hash",
        "configuration_hash",
        "implementation_hash",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        return _hash(value)

    @field_validator("created_at")
    @classmethod
    def validate_utc_time(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def validate_status(self) -> TransformationCertificate:
        if self.validation_status == "N/A" and not self.not_applicable_reason:
            raise ValueError("N/A certificates require a controlled reason")
        if self.validation_status == "PASS" and self.not_applicable_reason:
            raise ValueError("passing certificates cannot carry an N/A reason")
        if self.view_id == "V10" and not self.parameters.get("components"):
            raise ValueError("composition certificates require ordered component evidence")
        return self


class TransformationManifest(StrictTransformationContract):
    dataset_id: int | str
    dataset_version: str
    seed: int
    split_strategy: str
    view_id: str
    view_name: str
    source_feature_path: str
    target_reference_path: str
    partition_feature_paths: dict[str, str]
    certificate_paths: dict[str, str]
    source_hashes: dict[str, str]
    target_hashes: dict[str, str]
    manifest_status: Literal["PASS"]
    created_at: str

    @model_validator(mode="after")
    def validate_partitions(self) -> TransformationManifest:
        if set(self.partition_feature_paths) != {"train", "calibration", "test"}:
            raise ValueError("manifest must reference all three partitions")
        if set(self.certificate_paths) != {"train", "calibration", "test"}:
            raise ValueError("manifest must reference all three certificates")
        return self


class TransformationInventoryRecord(StrictTransformationContract):
    dataset_id: int
    seed: int
    view_id: str
    view_name: str
    status: Literal["PASS", "N/A", "FAIL"]
    reason_code: str | None = None
    reason: str | None = None
    applicability: Literal["APPLICABLE", "NOT_APPLICABLE"]
    certificate_ids: list[str] = Field(default_factory=list)
    source_hash: str | None = None
    output_hash: str | None = None
    reconstruction_max_abs_error: float | None = Field(default=None, ge=0)
    reconstruction_exact: bool | None = None
    deterministic: bool | None = None
    runtime_seconds: float = Field(ge=0)
    created_at: str

    @model_validator(mode="after")
    def validate_record(self) -> TransformationInventoryRecord:
        if self.status == "N/A":
            if self.applicability != "NOT_APPLICABLE" or not self.reason_code or not self.reason:
                raise ValueError("N/A inventory records require controlled applicability evidence")
        elif self.applicability != "APPLICABLE":
            raise ValueError("applicable inventory result required for PASS/FAIL")
        if self.status == "PASS" and not self.certificate_ids:
            raise ValueError("passing inventory records require certificates")
        return self


class TransformationInventory(StrictTransformationContract):
    engine_name: Literal["certified_lossless_transformations"]
    status: Literal["PASS_PENDING_REVIEW", "FAIL", "BLOCKED"]
    dataset_ids: list[int]
    seeds: list[int]
    view_ids: list[str]
    records: list[TransformationInventoryRecord]
    expected_record_count: int = Field(gt=0)
    applicable_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)
    pass_count: int = Field(ge=0)
    fail_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    property_example_count: int = Field(ge=0)
    registry_hash: str
    split_inventory_hash: str
    protected_hash_comparison: dict[str, Any]
    reconstruction_max_abs_error: float = Field(ge=0)
    generated_at: str

    @field_validator("registry_hash", "split_inventory_hash")
    @classmethod
    def validate_inventory_hash(cls, value: str) -> str:
        return _hash(value) or value

    @model_validator(mode="after")
    def validate_counts(self) -> TransformationInventory:
        if self.expected_record_count != len(self.records):
            raise ValueError("expected_record_count must equal records length")
        if self.applicable_count + self.not_applicable_count != len(self.records):
            raise ValueError("applicability counts do not account for all records")
        if (
            self.pass_count + self.fail_count + self.missing_count + self.not_applicable_count
            != len(self.records)
        ):
            raise ValueError("validation counts do not account for all records")
        if self.status == "PASS_PENDING_REVIEW" and (
            self.fail_count or self.missing_count or self.expected_record_count != 770
        ):
            raise ValueError("a passing inventory requires the complete 770-record matrix")
        return self


class TransformationValidationRecord(StrictTransformationContract):
    dataset_id: int
    seed: int
    view_id: str
    status: Literal["PASS", "N/A", "FAIL"]
    applicability: Literal["APPLICABLE", "NOT_APPLICABLE"]
    reason_code: str | None = None
    reason: str | None = None
    certificate_hashes: list[str] = Field(default_factory=list)
    source_hash: str | None = None
    output_hash: str | None = None
    reconstruction_error: float | None = Field(default=None, ge=0)
    deterministic: bool | None = None
    runtime_seconds: float = Field(ge=0)


class TransformationValidationReport(StrictTransformationContract):
    engine_name: Literal["certified_lossless_transformations"]
    status: Literal["PASS_PENDING_REVIEW", "FAIL", "BLOCKED"]
    records: list[TransformationValidationRecord]
    expected_record_count: int = Field(gt=0)
    registry_hash: str
    split_inventory_hash: str
    property_example_count: int = Field(ge=0)
    protected_hash_comparison: dict[str, Any]
    generated_at: str


def contract_hash(value: Any) -> str:
    return sha256_canonical_json(value)


__all__ = [
    "ApplicabilityRecord",
    "FeatureSchemaContract",
    "StrictTransformationContract",
    "TransformationCertificate",
    "TransformationConfig",
    "TransformationInventory",
    "TransformationInventoryRecord",
    "TransformationManifest",
    "TransformationValidationRecord",
    "TransformationValidationReport",
    "contract_hash",
]
