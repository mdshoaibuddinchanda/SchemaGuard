from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.models.adapters.contracts import (
    AdapterCacheIdentity,
    AdapterInventoryBatch,
    AdapterInventoryRecord,
    FailureCategory,
    ModelAdapterInventory,
    PredictionResult,
    adapter_logical_case_id,
    expected_adapter_case_matrix,
)
from schemaguard.utils.hashing import sha256_canonical_json


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
    model_id = str(changes.get("model_id", "LR-1.9"))
    device = str(changes.get("device", "cpu"))
    package_name, package_version = {
        "LR-1.9": ("scikit-learn", "1.9.1"),
        "CAT-1.2": ("catboost", "1.2.10"),
        "XGB-3.4": ("xgboost", "3.4.1"),
        "TPFN3-8.5": ("tabpfn", "8.5.0"),
        "TICL2-2.2": ("tabicl", "2.2.0"),
    }[model_id]
    is_foundation = model_id in {"TPFN3-8.5", "TICL2-2.2"}
    checkpoint = {
        "TPFN3-8.5": (
            "tabpfn-v3-classifier-v3_default.ckpt",
            "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
        ),
        "TICL2-2.2": (
            "tabicl-classifier-v2-20260212.ckpt",
            "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0",
        ),
    }.get(model_id)
    constructor_parameters = (
        {"n_estimators": 8, "device": device}
        if model_id == "TPFN3-8.5"
        else {"kv_cache": False, "allow_auto_download": False, "device": device}
        if model_id == "TICL2-2.2"
        else {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 1e-6,
            "class_weight": None,
            "random_state": 1729,
        }
    )
    if checkpoint:
        constructor_parameters["model_path"] = {
            "checkpoint_identifier": checkpoint[0],
            "checkpoint_sha256": checkpoint[1],
        }
    parameter_identity = {
        "schema_version": 1,
        "model_id": model_id,
        "package_name": package_name,
        "package_version": package_version,
        "python_major_minor": "3.12",
        "pytorch_version": "2.6.0+cu124" if is_foundation else "none",
        "seed": 1729,
        "device": device,
        "checkpoint_identifier": checkpoint[0] if checkpoint else None,
        "checkpoint_sha256": checkpoint[1] if checkpoint else None,
        "parameters": constructor_parameters,
    }
    parameter_hash = sha256_canonical_json(parameter_identity)
    roundtrip = "prediction_cache" if is_foundation else "model_serialization"
    checkpoint_identifier = checkpoint[0] if checkpoint else None
    checkpoint_hash = checkpoint[1] if checkpoint else None
    record: dict[str, object] = {
        "logical_case_id": "9" * 64,
        "prediction_identity_sha256": "8" * 64,
        "probabilities_sha256": "7" * 64,
        "model_id": model_id,
        "package_name": package_name,
        "package_version": package_version,
        "fixture_id": "binary_numerical",
        "fixture_sha256": "a" * 64,
        "device": device,
        "partition": "test",
        "seed": 1729,
        "roundtrip": roundtrip,
        "model_spec_sha256": "b" * 64,
        "parameter_sha256": parameter_hash,
        "parameter_identity": parameter_identity,
        "adapter_sha256": "d" * 64,
        "preprocessing_implementation_sha256": "5" * 64,
        "preprocessing_sha256": "e" * 64,
        "checkpoint_identifier": checkpoint_identifier,
        "checkpoint_sha256": checkpoint_hash,
        "row_ids_sha256": "f" * 64,
        "class_order": [0, 1],
        "probability_sum_error": 0.0,
        "repeat_max_abs_difference": 0.0,
        "preprocessing_uncached_seconds": 0.02,
        "preprocessing_cache_hit_seconds": 0.001,
        "offline_network_attempts": 0,
        "roundtrip_passed": True,
        "leakage_test_passed": True,
        "leakage_evidence_sha256": "6" * 64,
        "status": "PASS",
        "failure_category": "PASS",
        "failure_reason": None,
        "runtime_seconds": 0.1,
        "peak_ram_mib": 100.0,
        "peak_process_tree_ram_mib": 100.0,
        "peak_vram_mib": 100.0 if device == "cuda" else None,
        "free_vram_before_mib": 1200.0 if device == "cuda" else None,
        "telemetry_complete": True,
        "resource_limits_passed": True,
        "cleanup_verified": True,
        "gpu_headroom_passed": True if device == "cuda" else None,
        "worker_isolation": "subprocess" if is_foundation else "sequential_parent",
        "worker_exit_code": 0 if is_foundation else None,
        "source_commit": "1" * 40,
    }
    record.update(changes)
    if "logical_case_id" not in changes:
        record["logical_case_id"] = adapter_logical_case_id(
            model_id=str(record["model_id"]),
            fixture_sha256=str(record["fixture_sha256"]),
            device=str(record["device"]),
            partition=str(record["partition"]),
            seed=int(record["seed"]),
            parameter_sha256=str(record["parameter_sha256"]),
            preprocessing_sha256=str(record["preprocessing_sha256"]),
            checkpoint_sha256=record["checkpoint_sha256"],
            roundtrip=str(record["roundtrip"]),
        )
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


def test_final_inventory_rejects_partial_matrix_and_duplicate_case_ids() -> None:
    payload = {
        "schema_version": 1,
        "stage": "model_adapter_inventory",
        "source_commit": "1" * 40,
        "records": [_inventory_record()],
    }
    with pytest.raises(ValidationError, match="37-case matrix"):
        ModelAdapterInventory.model_validate(payload)
    with pytest.raises(ValidationError):
        ModelAdapterInventory.model_validate({**payload, "unexpected": True})


def test_final_inventory_accepts_only_the_frozen_37_case_matrix() -> None:
    records = [
        _inventory_record(
            model_id=model_id,
            fixture_id=fixture_id,
            fixture_sha256=sha256_canonical_json(fixture_id),
            device=device,
            class_order=[0, 1, 2] if fixture_id == "multiclass_numerical" else [0, 1],
        )
        for model_id, fixture_id, device in sorted(expected_adapter_case_matrix())
    ]
    inventory = ModelAdapterInventory(source_commit="1" * 40, records=records)
    assert len(inventory.records) == 37
    with pytest.raises(ValidationError, match="exact 37-case matrix"):
        ModelAdapterInventory(source_commit="1" * 40, records=records[:-1])


def test_inventory_batch_merge_requires_same_source_commit_and_unique_cases() -> None:
    source = AdapterInventoryBatch.model_validate(
        {
            "schema_version": 1,
            "stage": "model_adapter_batch",
            "source_commit": "1" * 40,
            "records": [_inventory_record()],
        }
    )
    gpu_record = _inventory_record(
        model_id="TPFN3-8.5",
        device="cuda",
        roundtrip="prediction_cache",
    )
    gpu = AdapterInventoryBatch.model_validate(
        {
            "schema_version": 1,
            "stage": "model_adapter_batch",
            "source_commit": "1" * 40,
            "records": [gpu_record],
        }
    )
    assert len(source.merge(gpu).records) == 2
    with pytest.raises(ValueError, match="different source commits"):
        source.merge(gpu.model_copy(update={"source_commit": "2" * 40}))
    with pytest.raises(ValidationError, match="duplicate logical adapter case"):
        source.merge(source)


def test_passing_inventory_record_requires_real_leakage_and_resource_evidence() -> None:
    with pytest.raises(ValidationError):
        AdapterInventoryRecord.model_validate(
            _inventory_record(leakage_test_passed=None, leakage_evidence_sha256=None)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"roundtrip_passed": False},
        {"leakage_test_passed": False},
        {"offline_network_attempts": 1},
        {"probability_sum_error": 1.1e-6},
        {"repeat_max_abs_difference": 1.1e-10},
        {"telemetry_complete": False},
        {"resource_limits_passed": False},
        {"runtime_seconds": None},
        {"peak_ram_mib": None},
        {"peak_process_tree_ram_mib": 28672.1},
        {"adapter_sha256": "0" * 64},
        {"class_order": [1, 0]},
    ],
)
def test_invalid_passing_cpu_record_fails_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AdapterInventoryRecord.model_validate(_inventory_record(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"roundtrip_passed": False},
        {"leakage_test_passed": False},
        {"offline_network_attempts": 1},
        {"probability_sum_error": 1.1e-6},
        {"repeat_max_abs_difference": 1.1e-5},
        {"telemetry_complete": False},
        {"resource_limits_passed": False},
        {"peak_process_tree_ram_mib": 28672.1},
        {"peak_vram_mib": 3600.1},
        {"free_vram_before_mib": 700.0},
        {"gpu_headroom_passed": False},
        {"cleanup_verified": False},
        {"worker_isolation": "sequential_parent"},
        {"worker_exit_code": 1},
        {"checkpoint_sha256": "0" * 64},
    ],
)
def test_invalid_passing_cuda_record_fails_closed(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AdapterInventoryRecord.model_validate(
            _inventory_record(model_id="TPFN3-8.5", device="cuda", **changes)
        )


def _complete_inventory_records() -> list[dict[str, object]]:
    return [
        _inventory_record(
            model_id=model_id,
            fixture_id=fixture_id,
            fixture_sha256=sha256_canonical_json(fixture_id),
            device=device,
            class_order=[0, 1, 2] if fixture_id == "multiclass_numerical" else [0, 1],
        )
        for model_id, fixture_id, device in sorted(expected_adapter_case_matrix())
    ]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mixed_commit", "missing_cuda"])
def test_final_matrix_rejects_incomplete_or_inconsistent_cases(mutation: str) -> None:
    records = _complete_inventory_records()
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(dict(records[0]))
    elif mutation == "mixed_commit":
        records[0] = {**records[0], "source_commit": "2" * 40}
    else:
        records = [
            row
            for row in records
            if not (row["device"] == "cuda" and row["model_id"] == "TICL2-2.2")
        ]
    with pytest.raises(ValidationError):
        ModelAdapterInventory(source_commit="1" * 40, records=records)


def test_final_matrix_rejects_classical_cuda_and_unexpected_fixture() -> None:
    with pytest.raises(ValidationError, match="classical adapters require CPU"):
        AdapterInventoryRecord.model_validate(
            _inventory_record(model_id="LR-1.9", device="cuda")
        )
    with pytest.raises(ValidationError):
        AdapterInventoryRecord.model_validate(_inventory_record(fixture_id="unexpected_fixture"))


def test_failed_record_cannot_claim_leakage_or_roundtrip_success() -> None:
    for changes in (
        {"status": "FAIL", "failure_category": "FAIL_TEST", "failure_reason": "test failed"},
        {
            "status": "FAIL",
            "failure_category": "FAIL_TEST",
            "failure_reason": "test failed",
            "roundtrip_passed": False,
            "leakage_test_passed": True,
        },
        {
            "status": "FAIL",
            "failure_category": "FAIL_TEST",
            "failure_reason": None,
            "roundtrip_passed": False,
            "leakage_test_passed": None,
        },
    ):
        with pytest.raises(ValidationError):
            AdapterInventoryRecord.model_validate(_inventory_record(**changes))


def test_failure_taxonomy_is_closed() -> None:
    assert FailureCategory.FAIL_LEAKAGE.value == "FAIL_LEAKAGE"
    with pytest.raises(ValueError):
        FailureCategory("FAIL_SOME_OTHER_THING")
