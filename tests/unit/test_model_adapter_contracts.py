from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.models.adapters.contracts import (
    AdapterCacheIdentity,
    FailureCategory,
    ModelAdapterInventory,
    PredictionResult,
)


def _identity(**changes: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": 1,
        "source_data_sha256": "a" * 64,
        "fixture_sha256": "a" * 64,
        "row_ids_sha256": "b" * 64,
        "model_spec_sha256": "c" * 64,
        "parameter_sha256": "d" * 64,
        "adapter_sha256": "e" * 64,
        "preprocessing_sha256": "f" * 64,
        "checkpoint_sha256": None,
        "source_commit": "1" * 40,
        "package_runtime": "scikit-learn==1.9.1;python=3.12",
        "seed": 1729,
        "device_policy": "cpu",
        "partition": "test",
        "split_identity": "split-1",
        "transformation_identity": None,
    }
    identity.update(changes)
    return identity


def _inventory_record(**changes: object) -> dict[str, object]:
    record: dict[str, object] = {
        "logical_case_id": "9" * 64,
        "prediction_identity_sha256": "8" * 64,
        "probabilities_sha256": "7" * 64,
        "model_id": "LR-1.9",
        "fixture_id": "binary_numerical",
        "fixture_sha256": "a" * 64,
        "device": "cpu",
        "roundtrip": "model_serialization",
        "model_spec_sha256": "b" * 64,
        "parameter_sha256": "c" * 64,
        "adapter_sha256": "d" * 64,
        "preprocessing_sha256": "e" * 64,
        "checkpoint_sha256": None,
        "row_ids_sha256": "f" * 64,
        "class_order": [0, 1],
        "probability_sum_error": 0.0,
        "repeat_max_abs_difference": 0.0,
        "preprocessing_uncached_seconds": 0.02,
        "preprocessing_cache_hit_seconds": 0.001,
        "offline_network_attempts": 0,
        "roundtrip_passed": True,
        "leakage_test_passed": True,
        "status": "PASS",
        "failure_category": "PASS",
        "runtime_seconds": 0.1,
        "peak_ram_mib": 100.0,
        "peak_vram_mib": None,
        "source_commit": "1" * 40,
    }
    record.update(changes)
    return record


def test_cache_identity_is_strict_and_sensitive_to_each_source() -> None:
    identity = AdapterCacheIdentity.model_validate(_identity())
    assert identity.cache_key == AdapterCacheIdentity.model_validate(_identity()).cache_key
    assert (
        identity.cache_key
        != AdapterCacheIdentity.model_validate(_identity(parameter_sha256="9" * 64)).cache_key
    )
    with pytest.raises(ValidationError):
        AdapterCacheIdentity.model_validate(_identity(unexpected="no"))


def test_prediction_result_rejects_duplicate_rows_and_invalid_simplex() -> None:
    common: dict[str, object] = {
        "schema_version": 1,
        "model_id": "LR-1.9",
        "package_name": "scikit-learn",
        "package_version": "1.9.1",
        "model_spec_sha256": "3" * 64,
        "source_commit": "1" * 40,
        "parameter_sha256": "a" * 64,
        "adapter_sha256": "b" * 64,
        "preprocessing_sha256": "c" * 64,
        "checkpoint_identifier": None,
        "checkpoint_sha256": None,
        "fixture_id": "binary_numerical",
        "fixture_sha256": "d" * 64,
        "source_data_sha256": "2" * 64,
        "split_identity": "fixture-split",
        "transformation_identity": None,
        "seed": 1729,
        "device": "cpu",
        "execution_policy": "single_process_cpu",
        "partition": "test",
        "row_ids": ["row-a", "row-b"],
        "row_ids_sha256": "UPDATE",
        "class_order": [0, 1],
        "probabilities": [[0.8, 0.2], [0.1, 0.9]],
        "probabilities_sha256": "UPDATE",
        "logical_identity_sha256": "1" * 64,
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "wall_time_seconds": 1.0,
        "cpu_time_seconds": 0.8,
        "peak_ram_mib": 100.0,
        "peak_vram_mib": None,
        "status": "PASS",
        "failure_category": "PASS",
        "failure_reason": None,
        "traceback_path": None,
    }
    import hashlib
    import struct

    from schemaguard.utils.hashing import sha256_canonical_json

    common["row_ids_sha256"] = sha256_canonical_json(common["row_ids"])
    payload = b"".join(struct.pack("<d", value) for row in common["probabilities"] for value in row)
    common["probabilities_sha256"] = hashlib.sha256(payload).hexdigest()
    assert PredictionResult.model_validate(common).status == "PASS"
    with pytest.raises(ValidationError):
        PredictionResult.model_validate({**common, "row_ids": ["same", "same"]})
    with pytest.raises(ValidationError):
        PredictionResult.model_validate({**common, "probabilities": [[0.8, 0.8], [0.1, 0.9]]})
    with pytest.raises(ValidationError):
        PredictionResult.model_validate({**common, "unexpected": True})


def test_inventory_contract_is_closed_and_case_ids_are_unique() -> None:
    payload = {
        "schema_version": 1,
        "stage": "model_adapter_inventory",
        "source_commit": "1" * 40,
        "records": [_inventory_record()],
    }
    assert ModelAdapterInventory.model_validate(payload).records[0].status == "PASS"
    duplicate = _inventory_record(logical_case_id="9" * 64)
    with pytest.raises(ValidationError):
        ModelAdapterInventory.model_validate(
            {**payload, "records": [payload["records"][0], duplicate]}
        )
    with pytest.raises(ValidationError):
        ModelAdapterInventory.model_validate({**payload, "unexpected": True})


def test_inventory_merge_requires_same_source_commit_and_unique_cases() -> None:
    source = ModelAdapterInventory.model_validate(
        {
            "schema_version": 1,
            "stage": "model_adapter_inventory",
            "source_commit": "1" * 40,
            "records": [_inventory_record()],
        }
    )
    gpu_record = _inventory_record(
        logical_case_id="6" * 64,
        device="cuda",
        roundtrip="prediction_cache",
    )
    gpu = ModelAdapterInventory.model_validate(
        {
            "schema_version": 1,
            "stage": "model_adapter_inventory",
            "source_commit": "1" * 40,
            "records": [gpu_record],
        }
    )
    assert len(source.merge(gpu).records) == 2
    with pytest.raises(ValueError, match="different source commits"):
        source.merge(gpu.model_copy(update={"source_commit": "2" * 40}))
    with pytest.raises(ValidationError, match="duplicate logical adapter case"):
        source.merge(source)


def test_failure_taxonomy_is_closed() -> None:
    assert FailureCategory.FAIL_LEAKAGE.value == "FAIL_LEAKAGE"
    with pytest.raises(ValueError):
        FailureCategory("FAIL_SOME_OTHER_THING")
