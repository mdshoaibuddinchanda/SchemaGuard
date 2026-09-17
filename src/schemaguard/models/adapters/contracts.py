"""Closed contracts shared by the five frozen model adapters."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ...utils.hashing import sha256_canonical_json

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
ModelId = Literal["LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2"]


class FailureCategory(StrEnum):
    PASS = "PASS"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
    BLOCKED_CHECKPOINT = "BLOCKED_CHECKPOINT"
    FAIL_CONTRACT = "FAIL_CONTRACT"
    FAIL_INPUT = "FAIL_INPUT"
    FAIL_LEAKAGE = "FAIL_LEAKAGE"
    FAIL_CLASS_ORDER = "FAIL_CLASS_ORDER"
    FAIL_PROBABILITY = "FAIL_PROBABILITY"
    FAIL_DETERMINISM = "FAIL_DETERMINISM"
    FAIL_SERIALIZATION = "FAIL_SERIALIZATION"
    FAIL_CACHE_INTEGRITY = "FAIL_CACHE_INTEGRITY"
    FAIL_RESOURCE_LIMIT = "FAIL_RESOURCE_LIMIT"
    FAIL_TIMEOUT = "FAIL_TIMEOUT"
    FAIL_CUDA_OOM = "FAIL_CUDA_OOM"
    FAIL_MODEL_RUNTIME = "FAIL_MODEL_RUNTIME"
    FAIL_TEST = "FAIL_TEST"


class AdapterFailure(RuntimeError):
    """A model adapter failure with a stable, non-misleading category."""

    def __init__(
        self,
        category: FailureCategory,
        message: str,
        *,
        traceback_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.traceback_path = traceback_path


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AdapterResourceRecord(StrictModel):
    model_load_seconds: float = Field(ge=0)
    preprocessing_fit_seconds: float = Field(ge=0)
    fit_seconds: float = Field(ge=0)
    preprocessing_seconds: float = Field(ge=0)
    prediction_seconds: float = Field(ge=0)
    serialization_seconds: float = Field(ge=0)
    cache_seconds: float = Field(ge=0)
    wall_time_seconds: float = Field(ge=0)
    cpu_time_seconds: float = Field(ge=0)
    peak_ram_mib: float | None = Field(default=None, ge=0)
    peak_process_tree_ram_mib: float | None = Field(default=None, ge=0)
    peak_vram_allocated_mib: float | None = Field(default=None, ge=0)
    peak_vram_reserved_mib: float | None = Field(default=None, ge=0)
    free_vram_before_mib: float | None = Field(default=None, ge=0)
    free_vram_after_mib: float | None = Field(default=None, ge=0)
    telemetry_complete: bool
    telemetry_error: str | None = None


class PredictionResult(StrictModel):
    """Strict adapter output. Row IDs and probabilities remain in memory/local cache only."""

    schema_version: Literal[1] = 1
    model_id: ModelId
    package_name: str
    package_version: str
    model_spec_sha256: Sha256
    source_commit: CommitSha
    parameter_sha256: Sha256
    adapter_sha256: Sha256
    preprocessing_sha256: Sha256
    checkpoint_identifier: str | None
    checkpoint_sha256: Sha256 | None
    fixture_id: str
    fixture_sha256: Sha256
    source_data_sha256: Sha256
    split_identity: str
    transformation_identity: str | None
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    execution_policy: str
    partition: Literal["train", "calibration", "test"]
    row_ids: list[str | int] = Field(min_length=1)
    row_ids_sha256: Sha256
    class_order: list[int] = Field(min_length=2)
    probabilities: list[list[float]] = Field(min_length=1)
    probabilities_sha256: Sha256
    logical_identity_sha256: Sha256
    started_at: str
    ended_at: str
    wall_time_seconds: float = Field(ge=0)
    cpu_time_seconds: float = Field(ge=0)
    peak_ram_mib: float | None = Field(default=None, ge=0)
    peak_vram_mib: float | None = Field(default=None, ge=0)
    status: Literal["PASS", "NOT_APPLICABLE", "BLOCKED", "FAIL"]
    failure_category: FailureCategory
    failure_reason: str | None = None
    traceback_path: str | None = None

    @model_validator(mode="after")
    def validate_output_contract(self) -> PredictionResult:
        if len(self.row_ids) != len(set(self.row_ids)):
            raise ValueError("prediction result contains duplicate row IDs")
        if len(self.class_order) != len(set(self.class_order)):
            raise ValueError("prediction result contains duplicate classes")
        if len(self.probabilities) != len(self.row_ids):
            raise ValueError("probability row count does not match row IDs")
        if sha256_canonical_json(self.row_ids) != self.row_ids_sha256:
            raise ValueError("row ID content hash mismatch")
        raw = b"".join(
            __import__("struct").pack("<d", value) for row in self.probabilities for value in row
        )
        if hashlib.sha256(raw).hexdigest() != self.probabilities_sha256:
            raise ValueError("probability content hash mismatch")
        for row in self.probabilities:
            if len(row) != len(self.class_order):
                raise ValueError("probability column count does not match canonical classes")
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in row):
                raise ValueError("probabilities must be finite values in [0, 1]")
            if abs(sum(row) - 1.0) > 1.0e-6:
                raise ValueError("probability row sum exceeds 1e-6 tolerance")
        if self.status == "PASS" and self.failure_category != FailureCategory.PASS:
            raise ValueError("passing predictions must use failure_category=PASS")
        if self.status != "PASS" and self.failure_category == FailureCategory.PASS:
            raise ValueError("unsuccessful predictions require a failure category")
        return self


class AdapterCacheIdentity(StrictModel):
    schema_version: Literal[1] = 1
    source_data_sha256: Sha256
    fixture_sha256: Sha256
    row_ids_sha256: Sha256
    model_spec_sha256: Sha256
    parameter_sha256: Sha256
    adapter_sha256: Sha256
    preprocessing_sha256: Sha256
    checkpoint_sha256: Sha256 | None
    source_commit: CommitSha
    package_runtime: str
    seed: int = Field(ge=0)
    device_policy: Literal["cpu", "cuda"]
    partition: Literal["train", "calibration", "test"]
    split_identity: str
    transformation_identity: str | None

    @property
    def cache_key(self) -> str:
        return sha256_canonical_json(self.model_dump(mode="json"))


class AdapterInventoryRecord(StrictModel):
    logical_case_id: Sha256
    prediction_identity_sha256: Sha256
    probabilities_sha256: Sha256
    model_id: ModelId
    fixture_id: str
    fixture_sha256: Sha256
    device: Literal["cpu", "cuda"]
    roundtrip: Literal["model_serialization", "prediction_cache"]
    model_spec_sha256: Sha256
    parameter_sha256: Sha256
    adapter_sha256: Sha256
    preprocessing_sha256: Sha256
    checkpoint_sha256: Sha256 | None
    row_ids_sha256: Sha256
    class_order: list[int] = Field(min_length=2)
    probability_sum_error: float = Field(ge=0)
    repeat_max_abs_difference: float = Field(ge=0)
    preprocessing_uncached_seconds: float = Field(ge=0)
    preprocessing_cache_hit_seconds: float = Field(ge=0)
    offline_network_attempts: int = Field(ge=0)
    roundtrip_passed: bool
    leakage_test_passed: bool
    status: Literal["PASS", "NOT_APPLICABLE", "BLOCKED", "FAIL"]
    failure_category: FailureCategory
    runtime_seconds: float = Field(ge=0)
    peak_ram_mib: float | None = Field(default=None, ge=0)
    peak_vram_mib: float | None = Field(default=None, ge=0)
    source_commit: CommitSha

    @model_validator(mode="after")
    def validate_pass(self) -> AdapterInventoryRecord:
        if self.status == "PASS" and self.failure_category != FailureCategory.PASS:
            raise ValueError("passing inventory records require failure_category=PASS")
        if self.status != "PASS" and self.failure_category == FailureCategory.PASS:
            raise ValueError("unsuccessful inventory records require an explicit category")
        return self


class ModelAdapterInventory(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["model_adapter_inventory"] = "model_adapter_inventory"
    source_commit: CommitSha
    records: list[AdapterInventoryRecord]

    @model_validator(mode="after")
    def unique_cases(self) -> ModelAdapterInventory:
        case_ids = [record.logical_case_id for record in self.records]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate logical adapter case in inventory")
        return self

    def merge(self, other: ModelAdapterInventory) -> ModelAdapterInventory:
        """Combine device-specific runs only when they share one source commit."""

        if self.source_commit != other.source_commit:
            raise ValueError("adapter inventories from different source commits cannot be merged")
        return ModelAdapterInventory(
            source_commit=self.source_commit,
            records=[*self.records, *other.records],
        )


class CheckpointIdentity(StrictModel):
    identifier: str
    sha256: Sha256


class AdapterResourceLimits(StrictModel):
    cpu_workers: Literal[1, 2] = 1
    cpu_threads: Literal[2] = 2
    soft_ram_gib: float = Field(gt=0, le=24)
    hard_ram_gib: float = Field(gt=0, le=28)
    gpu_workers: Literal[1] = 1
    gpu_headroom_mib: float = Field(ge=512)
    gpu_soft_limit_mib: float = Field(gt=0, le=3600)

    @model_validator(mode="after")
    def coherent_ram_limits(self) -> AdapterResourceLimits:
        if self.soft_ram_gib >= self.hard_ram_gib:
            raise ValueError("soft RAM limit must be lower than hard RAM limit")
        return self


class AdapterTolerance(StrictModel):
    probability_sum_atol: float = Field(gt=0, le=1.0e-6)
    cpu_repeat_max_abs_diff: float = Field(ge=0, le=1.0e-10)
    gpu_repeat_max_abs_diff: float = Field(ge=0, le=1.0e-5)


class ModelAdapterConfig(StrictModel):
    schema_version: Literal[1] = 1
    registry_config: Literal["model_compatibility.yaml"]
    experiment_registry: Literal["../experiment_registry.yaml"]
    seed: int = Field(ge=0)
    fixtures: list[str] = Field(min_length=1)
    resources: AdapterResourceLimits
    tolerances: AdapterTolerance
    cache_directory: str
    checkpoints: dict[ModelId, CheckpointIdentity]

    @model_validator(mode="after")
    def frozen_checkpoint_set(self) -> ModelAdapterConfig:
        expected = {"TPFN3-8.5", "TICL2-2.2"}
        if set(self.checkpoints) != expected:
            raise ValueError("adapter config must identify both frozen foundation checkpoints")
        if self.resources.hard_ram_gib <= self.resources.soft_ram_gib:
            raise ValueError("invalid RAM caps")
        return self


def implementation_digest(source_text: str) -> str:
    """Hash semantic source text independently of path and line-ending convention."""

    normalized = source_text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_canonical_json({"source": normalized})


__all__ = [
    "AdapterCacheIdentity",
    "AdapterFailure",
    "AdapterInventoryRecord",
    "AdapterResourceRecord",
    "CheckpointIdentity",
    "FailureCategory",
    "ModelAdapterConfig",
    "ModelAdapterInventory",
    "PredictionResult",
    "implementation_digest",
]
