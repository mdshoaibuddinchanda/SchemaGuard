"""Closed contracts shared by the five frozen model adapters."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ...utils.hashing import sha256_canonical_json

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
ModelId = Literal["LR-1.9", "CAT-1.2", "XGB-3.4", "TPFN3-8.5", "TICL2-2.2"]

MODEL_PACKAGE_VERSIONS = {
    "LR-1.9": ("scikit-learn", "1.9.1"),
    "CAT-1.2": ("catboost", "1.2.10"),
    "XGB-3.4": ("xgboost", "3.4.1"),
    "TPFN3-8.5": ("tabpfn", "8.5.0"),
    "TICL2-2.2": ("tabicl", "2.2.0"),
}
FOUNDATION_CHECKPOINTS = {
    "TPFN3-8.5": (
        "tabpfn-v3-classifier-v3_default.ckpt",
        "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
    ),
    "TICL2-2.2": (
        "tabicl-classifier-v2-20260212.ckpt",
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
    ),
}
ADAPTER_FIXTURE_IDS = (
    "binary_numerical",
    "multiclass_numerical",
    "missing_numerical",
    "categorical_only",
    "mixed_categorical",
    "unseen_category",
    "missing_categorical",
)


def expected_adapter_case_matrix() -> set[tuple[str, str, str]]:
    cases = {
        (model_id, fixture_id, "cpu")
        for model_id in MODEL_PACKAGE_VERSIONS
        for fixture_id in ADAPTER_FIXTURE_IDS
    }
    cases.update((model_id, "binary_numerical", "cuda") for model_id in FOUNDATION_CHECKPOINTS)
    return cases


def adapter_logical_case_id(
    *,
    model_id: str,
    fixture_sha256: str,
    device: str,
    partition: str,
    seed: int,
    parameter_sha256: str,
    preprocessing_sha256: str,
    checkpoint_sha256: str | None,
    roundtrip: str,
) -> str:
    return sha256_canonical_json(
        {
            "model_id": model_id,
            "fixture_sha256": fixture_sha256,
            "device": device,
            "partition": partition,
            "seed": seed,
            "parameter_sha256": parameter_sha256,
            "preprocessing_sha256": preprocessing_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "roundtrip": roundtrip,
        }
    )


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
    cpu_time_seconds: float | None = Field(default=None, ge=0)
    peak_ram_mib: float | None = Field(default=None, ge=0)
    peak_process_tree_ram_mib: float | None = Field(default=None, ge=0)
    peak_vram_allocated_mib: float | None = Field(default=None, ge=0)
    peak_vram_reserved_mib: float | None = Field(default=None, ge=0)
    free_vram_before_mib: float | None = Field(default=None, ge=0)
    free_vram_after_mib: float | None = Field(default=None, ge=0)
    telemetry_complete: bool
    telemetry_error: str | None = None
    gpu_headroom_passed: bool | None
    resource_limits_passed: bool
    resource_limit_failure: str | None = None
    cleanup_verified: bool
    cleanup_gpu_allocated_mib: float | None = Field(default=None, ge=0)
    cleanup_gpu_reserved_mib: float | None = Field(default=None, ge=0)


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
    cpu_time_seconds: float | None = Field(default=None, ge=0)
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


class SemanticEquivalenceCertificate(StrictModel):
    """Identity-bound proof that two serialized models make equivalent predictions."""

    schema_version: Literal[1] = 1
    existing_payload_sha256: Sha256
    candidate_payload_sha256: Sha256
    model_spec_sha256: Sha256
    parameter_sha256: Sha256
    preprocessing_sha256: Sha256
    training_data_sha256: Sha256
    checkpoint_sha256: Sha256 | None
    reference_fixture_sha256: Sha256
    reference_row_ids_sha256: Sha256
    reference_features_sha256: Sha256
    reference_class_order: list[int] = Field(min_length=2)
    reference_prediction_sha256_existing: Sha256
    reference_prediction_sha256_candidate: Sha256
    maximum_probability_difference: float = Field(ge=0)
    tolerance: float = Field(gt=0, le=1.0e-5)
    equivalent: Literal[True]
    certificate_sha256: Sha256

    @model_validator(mode="after")
    def validate_equivalence(self) -> SemanticEquivalenceCertificate:
        if self.existing_payload_sha256 == self.candidate_payload_sha256:
            raise ValueError("semantic certificate is only for differing serialized bytes")
        if len(self.reference_class_order) != len(set(self.reference_class_order)):
            raise ValueError("semantic certificate class order must be unique")
        if self.reference_class_order != list(range(len(self.reference_class_order))):
            raise ValueError("semantic certificate class order must be canonical")
        hashes = (
            self.existing_payload_sha256,
            self.candidate_payload_sha256,
            self.model_spec_sha256,
            self.parameter_sha256,
            self.preprocessing_sha256,
            self.training_data_sha256,
            self.checkpoint_sha256,
            self.reference_fixture_sha256,
            self.reference_row_ids_sha256,
            self.reference_features_sha256,
            self.reference_prediction_sha256_existing,
            self.reference_prediction_sha256_candidate,
        )
        if any(value == "0" * 64 for value in hashes if value is not None):
            raise ValueError("semantic certificate cannot use placeholder hashes")
        if self.maximum_probability_difference > self.tolerance:
            raise ValueError("semantic certificate exceeds its prediction tolerance")
        expected = sha256_canonical_json(
            self.model_dump(mode="json", exclude={"certificate_sha256"})
        )
        if self.certificate_sha256 != expected:
            raise ValueError("semantic equivalence certificate checksum mismatch")
        return self

    def validate_identity(
        self,
        identity: AdapterCacheIdentity,
        *,
        existing_payload_sha256: str,
        candidate_payload_sha256: str | None = None,
    ) -> None:
        if self.existing_payload_sha256 != existing_payload_sha256:
            raise ValueError("semantic certificate does not bind the cached payload")
        if candidate_payload_sha256 is not None and (
            self.candidate_payload_sha256 != candidate_payload_sha256
        ):
            raise ValueError("semantic certificate does not bind the candidate payload")
        expected = (
            self.model_spec_sha256 == identity.model_spec_sha256
            and self.parameter_sha256 == identity.parameter_sha256
            and self.preprocessing_sha256 == identity.preprocessing_sha256
            and self.training_data_sha256 == identity.source_data_sha256
            and self.checkpoint_sha256 == identity.checkpoint_sha256
            and self.reference_fixture_sha256 == identity.fixture_sha256
        )
        if not expected:
            raise ValueError("semantic certificate scientific identity differs from cache key")
        tolerance_limit = 1.0e-10 if identity.device_policy == "cpu" else 1.0e-5
        if self.tolerance > tolerance_limit:
            raise ValueError("semantic certificate exceeds the frozen device tolerance")


class CanonicalParameterIdentity(StrictModel):
    schema_version: Literal[1] = 1
    model_id: ModelId
    package_name: str
    package_version: str
    python_major_minor: Literal["3.12"]
    pytorch_version: str
    seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"]
    checkpoint_identifier: str | None
    checkpoint_sha256: Sha256 | None
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def validate_frozen_parameters(self) -> CanonicalParameterIdentity:
        if (self.package_name, self.package_version) != MODEL_PACKAGE_VERSIONS[self.model_id]:
            raise ValueError("parameter identity package differs from the frozen registry")
        expected_checkpoint = FOUNDATION_CHECKPOINTS.get(self.model_id)
        if expected_checkpoint is None:
            if self.checkpoint_identifier is not None or self.checkpoint_sha256 is not None:
                raise ValueError("classical parameter identity cannot include a checkpoint")
            if "model_path" in self.parameters:
                raise ValueError("classical parameter identity cannot contain a local model path")
        else:
            if (self.checkpoint_identifier, self.checkpoint_sha256) != expected_checkpoint:
                raise ValueError("parameter identity checkpoint differs from the frozen registry")
            path_identity = self.parameters.get("model_path")
            if path_identity != {
                "checkpoint_identifier": self.checkpoint_identifier,
                "checkpoint_sha256": self.checkpoint_sha256,
            }:
                raise ValueError("model_path must resolve to the portable checkpoint identity")
        if self.model_id == "TICL2-2.2":
            if self.parameters.get("kv_cache") is not False:
                raise ValueError("TabICL parameter identity requires kv_cache=False")
            if self.parameters.get("allow_auto_download") is not False:
                raise ValueError("TabICL parameter identity requires allow_auto_download=False")
        if self.model_id == "TPFN3-8.5" and "fit_with_cache" in self.parameters:
            raise ValueError("TabPFN parameter identity cannot enable fit_with_cache")
        if "device" in self.parameters and self.parameters["device"] != self.device:
            raise ValueError("constructor device parameter differs from the execution device")
        for value in self.parameters.values():
            if isinstance(value, str) and (
                value.startswith("/")
                or (len(value) >= 3 and value[1:3] == ":\\")
                or value.startswith("\\\\")
            ):
                raise ValueError("canonical model parameters cannot contain machine-local paths")
        return self


class AdapterInventoryRecord(StrictModel):
    logical_case_id: Sha256
    prediction_identity_sha256: Sha256 | None
    probabilities_sha256: Sha256 | None
    model_id: ModelId
    package_name: str
    package_version: str | None
    fixture_id: str
    fixture_sha256: Sha256
    device: Literal["cpu", "cuda"]
    partition: Literal["test"]
    seed: int = Field(ge=0)
    roundtrip: Literal["model_serialization", "prediction_cache"]
    model_spec_sha256: Sha256 | None
    parameter_sha256: Sha256 | None
    parameter_identity: CanonicalParameterIdentity | None
    adapter_sha256: Sha256 | None
    preprocessing_implementation_sha256: Sha256 | None
    preprocessing_sha256: Sha256 | None
    checkpoint_identifier: str | None
    checkpoint_sha256: Sha256 | None
    row_ids_sha256: Sha256 | None
    class_order: list[int] = Field(default_factory=list)
    probability_sum_error: float | None = Field(default=None, ge=0)
    repeat_max_abs_difference: float | None = Field(default=None, ge=0)
    preprocessing_uncached_seconds: float | None = Field(default=None, ge=0)
    preprocessing_cache_hit_seconds: float | None = Field(default=None, ge=0)
    offline_network_attempts: int = Field(ge=0)
    roundtrip_passed: bool
    leakage_test_passed: bool | None
    leakage_evidence_sha256: Sha256 | None
    status: Literal["PASS", "NOT_APPLICABLE", "BLOCKED", "FAIL"]
    failure_category: FailureCategory
    failure_reason: str | None = None
    runtime_seconds: float | None = Field(default=None, ge=0)
    peak_ram_mib: float | None = Field(default=None, ge=0)
    peak_process_tree_ram_mib: float | None = Field(default=None, ge=0)
    peak_vram_mib: float | None = Field(default=None, ge=0)
    free_vram_before_mib: float | None = Field(default=None, ge=0)
    telemetry_complete: bool
    resource_limits_passed: bool
    cleanup_verified: bool
    gpu_headroom_passed: bool | None
    worker_isolation: Literal["sequential_parent", "subprocess"]
    worker_exit_code: int | None
    source_commit: CommitSha

    @model_validator(mode="after")
    def validate_record(self) -> AdapterInventoryRecord:
        expected_package, expected_version = MODEL_PACKAGE_VERSIONS[self.model_id]
        if self.package_name != expected_package:
            raise ValueError("model package identity differs from the frozen registry")
        if self.status == "PASS" and self.package_version != expected_version:
            raise ValueError("passing model package version differs from the frozen registry")
        expected_checkpoint = FOUNDATION_CHECKPOINTS.get(self.model_id)
        if expected_checkpoint is None:
            if self.checkpoint_identifier is not None or self.checkpoint_sha256 is not None:
                raise ValueError("classical model must not claim a foundation checkpoint")
        elif self.status == "PASS" and (
            self.checkpoint_identifier,
            self.checkpoint_sha256,
        ) != expected_checkpoint:
            raise ValueError("foundation checkpoint identity differs from the frozen registry")
        elif self.status != "PASS" and self.checkpoint_identifier not in {
            None,
            expected_checkpoint[0],
        }:
            raise ValueError("failed foundation record names an unexpected checkpoint")
        if self.fixture_id not in ADAPTER_FIXTURE_IDS:
            raise ValueError("inventory references an unknown frozen fixture")
        if self.model_id not in FOUNDATION_CHECKPOINTS and (
            self.device != "cpu" or self.roundtrip != "model_serialization"
        ):
            raise ValueError("classical adapters require CPU model-serialization round-trips")
        if self.model_id in FOUNDATION_CHECKPOINTS and self.roundtrip != "prediction_cache":
            raise ValueError("foundation adapters require prediction-cache round-trips")
        if self.status == "PASS" and self.failure_category != FailureCategory.PASS:
            raise ValueError("passing inventory records require failure_category=PASS")
        if self.status != "PASS" and self.failure_category == FailureCategory.PASS:
            raise ValueError("unsuccessful inventory records require an explicit category")
        if self.status != "PASS":
            if self.roundtrip_passed or self.leakage_test_passed is not None:
                raise ValueError("unsuccessful records cannot claim round-trip or leakage success")
            if self.leakage_evidence_sha256 is not None:
                raise ValueError("unsuccessful records cannot claim linked leakage evidence")
            if not self.failure_reason:
                raise ValueError("unsuccessful records require an explicit failure reason")
            if (self.failure_category == FailureCategory.NOT_APPLICABLE) != (
                self.status == "NOT_APPLICABLE"
            ):
                raise ValueError(
                    "NOT_APPLICABLE status and failure category must be explicit and consistent"
                )
            return self
        if self.failure_category != FailureCategory.PASS:
            raise ValueError("passing inventory records require failure_category=PASS")
        if not self.roundtrip_passed:
            raise ValueError("passing inventory records require a successful round-trip")
        if self.leakage_test_passed is not True or self.leakage_evidence_sha256 is None:
            raise ValueError("passing inventory records require linked leakage evidence")
        if self.offline_network_attempts != 0:
            raise ValueError("passing adapter evidence must have zero network attempts")
        if not self.telemetry_complete or not self.resource_limits_passed:
            raise ValueError("passing adapter evidence requires complete, passing resource checks")
        if not self.cleanup_verified:
            raise ValueError("passing adapter evidence requires verified cleanup")
        required = (
            self.prediction_identity_sha256,
            self.probabilities_sha256,
            self.model_spec_sha256,
            self.parameter_sha256,
            self.parameter_identity,
            self.adapter_sha256,
            self.preprocessing_implementation_sha256,
            self.preprocessing_sha256,
            self.row_ids_sha256,
            self.probability_sum_error,
            self.repeat_max_abs_difference,
            self.runtime_seconds,
            self.peak_ram_mib,
            self.peak_process_tree_ram_mib,
        )
        if any(value is None for value in required):
            raise ValueError("passing adapter evidence is missing an identity or resource value")
        if (
            self.parameter_identity is None
            or self.parameter_sha256 is None
            or self.preprocessing_sha256 is None
        ):
            raise ValueError("passing adapter evidence is missing canonical identity fields")
        hash_values = (
            self.logical_case_id,
            self.prediction_identity_sha256,
            self.probabilities_sha256,
            self.fixture_sha256,
            self.model_spec_sha256,
            self.parameter_sha256,
            self.adapter_sha256,
            self.preprocessing_implementation_sha256,
            self.preprocessing_sha256,
            self.row_ids_sha256,
            self.leakage_evidence_sha256,
            self.checkpoint_sha256,
        )
        if any(value == "0" * 64 for value in hash_values if value is not None):
            raise ValueError("passing adapter evidence cannot use placeholder hashes")
        if (
            sha256_canonical_json(self.parameter_identity.model_dump(mode="json"))
            != self.parameter_sha256
        ):
            raise ValueError("canonical parameter identity hash mismatch")
        parameter_identity = self.parameter_identity
        if (
            parameter_identity.model_id != self.model_id
            or parameter_identity.package_name != self.package_name
            or parameter_identity.package_version != self.package_version
            or parameter_identity.seed != self.seed
            or parameter_identity.device != self.device
        ):
            raise ValueError("inventory fields differ from canonical parameter identity")
        if self.probability_sum_error is None or self.probability_sum_error > 1.0e-6:
            raise ValueError("probability rows exceed the frozen simplex tolerance")
        if self.peak_process_tree_ram_mib is not None and self.peak_process_tree_ram_mib > 28672.0:
            raise ValueError("adapter evidence exceeds the 28 GiB process-tree RAM cap")
        tolerance = 1.0e-5 if self.device == "cuda" else 1.0e-10
        if self.repeat_max_abs_difference is None or self.repeat_max_abs_difference > tolerance:
            raise ValueError("repeated inference exceeds the frozen determinism tolerance")
        if self.device == "cpu":
            if self.peak_vram_mib is not None or self.free_vram_before_mib is not None:
                raise ValueError("CPU evidence cannot claim CUDA resource measurements")
            if self.gpu_headroom_passed is not None:
                raise ValueError("CPU evidence cannot claim a CUDA headroom check")
        else:
            if self.peak_vram_mib is None or self.free_vram_before_mib is None:
                raise ValueError("CUDA evidence requires measured peak and pre-run free VRAM")
            if self.worker_isolation != "subprocess" or self.worker_exit_code != 0:
                raise ValueError("CUDA evidence requires a successful isolated worker exit")
            expected_peak = 376.0 if self.model_id == "TPFN3-8.5" else 198.0
            if self.free_vram_before_mib < 512.0 + expected_peak:
                raise ValueError("CUDA evidence lacks the frozen VRAM headroom")
            if self.peak_vram_mib > 3600.0:
                raise ValueError("CUDA evidence exceeds the frozen VRAM soft limit")
            if self.gpu_headroom_passed is not True:
                raise ValueError("CUDA evidence requires a passing explicit VRAM-headroom check")
        if self.model_id in FOUNDATION_CHECKPOINTS and self.worker_isolation != "subprocess":
            raise ValueError("foundation evidence requires an isolated worker process")
        expected_class_order = [0, 1, 2] if self.fixture_id == "multiclass_numerical" else [0, 1]
        if self.class_order != expected_class_order:
            raise ValueError("class order differs from the fixture's canonical classes")
        expected_roundtrip = (
            "prediction_cache" if self.model_id in FOUNDATION_CHECKPOINTS else "model_serialization"
        )
        if self.roundtrip != expected_roundtrip:
            raise ValueError("round-trip mode differs from the frozen adapter policy")
        identity = adapter_logical_case_id(
            model_id=self.model_id,
            fixture_sha256=self.fixture_sha256,
            device=self.device,
            partition=self.partition,
            seed=self.seed,
            parameter_sha256=self.parameter_sha256,
            preprocessing_sha256=self.preprocessing_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            roundtrip=self.roundtrip,
        )
        if self.logical_case_id != identity:
            raise ValueError("logical case identity does not match its canonical fields")
        return self


class ModelAdapterInventory(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["model_adapter_inventory"] = "model_adapter_inventory"
    source_commit: CommitSha
    records: list[AdapterInventoryRecord]

    @model_validator(mode="after")
    def complete_matrix(self) -> ModelAdapterInventory:
        keys = [(record.model_id, record.fixture_id, record.device) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate logical adapter case in inventory")
        expected = expected_adapter_case_matrix()
        observed = set(keys)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"inventory must contain the exact 37-case matrix; missing={missing}, extra={extra}"
            )
        if any(record.status != "PASS" for record in self.records):
            raise ValueError("final adapter inventory cannot contain failed or unexecuted cases")
        if any(record.source_commit != self.source_commit for record in self.records):
            raise ValueError("inventory record source commits do not match the inventory")
        if {record.seed for record in self.records} != {1729}:
            raise ValueError("inventory must use the frozen seed 1729")
        return self


class AdapterInventoryBatch(StrictModel):
    """A local, incomplete run batch that is never mistaken for accepted evidence."""

    schema_version: Literal[1] = 1
    stage: Literal["model_adapter_batch"] = "model_adapter_batch"
    source_commit: CommitSha
    records: list[AdapterInventoryRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> AdapterInventoryBatch:
        keys = [(record.model_id, record.fixture_id, record.device) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate logical adapter case in batch")
        if any(record.source_commit != self.source_commit for record in self.records):
            raise ValueError("batch record source commits do not match the batch")
        return self

    def merge(self, other: AdapterInventoryBatch) -> AdapterInventoryBatch:
        """Combine partial runs only when their source commit and case keys are disjoint."""

        if self.source_commit != other.source_commit:
            raise ValueError("adapter inventories from different source commits cannot be merged")
        return AdapterInventoryBatch(
            source_commit=self.source_commit,
            records=[*self.records, *other.records],
        )


class AdapterLeakageEvidence(StrictModel):
    schema_version: Literal[1] = 1
    model_id: ModelId
    fixture_id: str
    fixture_sha256: Sha256
    device: Literal["cpu", "cuda"]
    seed: int = Field(ge=0)
    source_commit: CommitSha
    model_spec_sha256: Sha256
    parameter_sha256: Sha256
    adapter_implementation_sha256: Sha256
    preprocessing_implementation_sha256: Sha256
    preprocessing_state_sha256: Sha256
    logical_case_id: Sha256
    training_feature_sha256: Sha256
    fit_row_ids_sha256: Sha256
    training_target_sha256: Sha256
    inference_feature_sha256: Sha256
    inference_row_ids_sha256: Sha256
    preprocessing_state_before_sha256: Sha256
    preprocessing_state_after_sha256: Sha256
    fit_call_count: Literal[1]
    test_label_access_attempts: Literal[1]
    calibration_label_access_attempts: Literal[1]
    heldout_labels_available: Literal[False]
    prediction_interface_rejects_label_arguments: Literal[True]
    state_unchanged: Literal[True]
    sentinel_excluded_from_fit: Literal[True]
    reordered_inference_preserves_state: Literal[True]
    duplicate_row_ids_rejected: Literal[True]
    empty_inputs_rejected: Literal[True]
    single_class_targets_rejected: Literal[True]
    noncontiguous_target_codes_rejected: Literal[True]
    preprocessing_fit_receives_features_only: Literal[True]
    forbidden_predictor_checks: dict[
        Literal["target_code", "target_label", "__sg_group_id"], Literal[True]
    ]
    passed: Literal[True]
    evidence_hash: Sha256

    @model_validator(mode="after")
    def verify_evidence_hash(self) -> AdapterLeakageEvidence:
        if set(self.forbidden_predictor_checks) != {
            "target_code",
            "target_label",
            "__sg_group_id",
        }:
            raise ValueError("all target and split metadata rejection checks are required")
        if self.preprocessing_state_before_sha256 != self.preprocessing_state_after_sha256:
            raise ValueError("inference changed fitted preprocessing state")
        if self.preprocessing_state_sha256 != self.preprocessing_state_before_sha256:
            raise ValueError("leakage evidence does not bind the fitted preprocessing state")
        if not self.state_unchanged or not self.passed:
            raise ValueError("leakage evidence has not passed its state and lifecycle checks")
        expected = sha256_canonical_json(self.model_dump(mode="json", exclude={"evidence_hash"}))
        if self.evidence_hash != expected:
            raise ValueError("leakage evidence checksum mismatch")
        return self


class AdapterLeakageEvidenceManifest(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["model_adapter_leakage_evidence"] = "model_adapter_leakage_evidence"
    source_commit: CommitSha
    records: list[AdapterLeakageEvidence] = Field(min_length=37, max_length=37)

    @model_validator(mode="after")
    def complete_matrix(self) -> AdapterLeakageEvidenceManifest:
        keys = [(record.model_id, record.fixture_id, record.device) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate leakage evidence case")
        if set(keys) != expected_adapter_case_matrix():
            raise ValueError("leakage evidence must cover the exact 37-case matrix")
        if any(record.source_commit != self.source_commit for record in self.records):
            raise ValueError("leakage evidence source commits do not match the manifest")
        return self


class AdapterLeakageEvidenceBatch(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["model_adapter_leakage_batch"] = "model_adapter_leakage_batch"
    source_commit: CommitSha
    records: list[AdapterLeakageEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> AdapterLeakageEvidenceBatch:
        keys = [record.logical_case_id for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate leakage evidence identity in batch")
        if any(record.source_commit != self.source_commit for record in self.records):
            raise ValueError("leakage batch source commits do not match the batch")
        return self

    def merge(self, other: AdapterLeakageEvidenceBatch) -> AdapterLeakageEvidenceBatch:
        if self.source_commit != other.source_commit:
            raise ValueError("leakage batches from different source commits cannot be merged")
        return AdapterLeakageEvidenceBatch(
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
    "AdapterInventoryBatch",
    "AdapterLeakageEvidence",
    "AdapterLeakageEvidenceBatch",
    "AdapterLeakageEvidenceManifest",
    "adapter_logical_case_id",
    "AdapterCacheIdentity",
    "AdapterFailure",
    "AdapterInventoryRecord",
    "AdapterResourceRecord",
    "CheckpointIdentity",
    "FailureCategory",
    "ModelAdapterConfig",
    "ModelAdapterInventory",
    "MODEL_PACKAGE_VERSIONS",
    "FOUNDATION_CHECKPOINTS",
    "expected_adapter_case_matrix",
    "PredictionResult",
    "implementation_digest",
]
