"""Executable leakage proofs and independent validation for adapter evidence."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np

from ...utils.hashing import hash_dataframe_logically, sha256_canonical_json
from .contracts import (
    FOUNDATION_CHECKPOINTS,
    MODEL_PACKAGE_VERSIONS,
    AdapterFailure,
    AdapterLeakageEvidence,
    AdapterLeakageEvidenceManifest,
    FailureCategory,
    ModelAdapterInventory,
    adapter_logical_case_id,
)


def build_leakage_evidence(
    adapter: Any, fixture: Any, prediction: Any, roundtrip: str
) -> dict[str, Any]:
    """Exercise the predictor-only boundary and return hashes, never source rows or labels."""

    from .base import _row_identifiers
    from .factory import create_adapter

    training_ids, training_features = _row_identifiers(fixture.train)
    inference_ids, inference_features = _row_identifiers(fixture.test)
    training_feature_hash = hash_dataframe_logically(training_features)
    training_target_hash = sha256_canonical_json(np.asarray(fixture.target).tolist())
    fit_signature = inspect.signature(adapter.preprocessor.fit)
    if (
        training_feature_hash != adapter.training_data_sha256
        or training_feature_hash != adapter.preprocessor.fit_input_sha256
        or adapter.preprocessor.fit_call_count != 1
        or sha256_canonical_json(training_ids) != adapter.training_row_ids_sha256
        or list(fit_signature.parameters) != ["training_features"]
    ):
        raise AdapterFailure(
            FailureCategory.FAIL_LEAKAGE,
            "preprocessing fit did not receive exactly the training features and row identity",
        )

    before = adapter.preprocessor.fitted_state_sha256
    sentinel = fixture.test.copy(deep=True)
    _, sentinel_predictors = _row_identifiers(sentinel)
    numeric_columns = [
        name
        for name in sentinel_predictors.columns
        if np.issubdtype(sentinel_predictors[name].dtype, np.number)
    ]
    if numeric_columns:
        sentinel.loc[:, numeric_columns[0]] = 987654321.0
    else:
        sentinel.loc[:, sentinel_predictors.columns[0]] = "__SCHEMAGUARD_INFERENCE_ONLY_SENTINEL__"
    adapter.predict_proba(
        sentinel,
        partition="test",
        fixture_id=fixture.name,
        fixture_sha256=fixture.sha256,
        split_identity=f"fixture:{fixture.name}:seed={adapter.seed}",
    )
    sentinel_state = adapter.preprocessor.fitted_state_sha256
    reordered = fixture.test.iloc[::-1].reset_index(drop=True)
    adapter.predict_proba(
        reordered,
        partition="test",
        fixture_id=fixture.name,
        fixture_sha256=fixture.sha256,
        split_identity=f"fixture:{fixture.name}:seed={adapter.seed}",
    )
    after = adapter.preprocessor.fitted_state_sha256
    sentinel_excluded = (
        before == sentinel_state == after
        and adapter.preprocessor.fit_call_count == 1
        and adapter.preprocessor.fit_input_sha256 == training_feature_hash
    )
    if not sentinel_excluded:
        raise AdapterFailure(
            FailureCategory.FAIL_LEAKAGE,
            "inference-only sentinel changed or entered fitted preprocessing state",
        )

    forbidden_checks: dict[str, bool] = {}
    for column in ("target_code", "target_label", "__sg_group_id"):
        invalid = fixture.train.copy(deep=True)
        invalid[column] = 0
        try:
            _row_identifiers(invalid)
        except AdapterFailure as exc:
            forbidden_checks[column] = exc.category == FailureCategory.FAIL_LEAKAGE
        else:
            forbidden_checks[column] = False
    if not all(forbidden_checks.values()):
        raise AdapterFailure(
            FailureCategory.FAIL_LEAKAGE,
            "a target or split-metadata predictor was not rejected",
        )

    prediction_signature = inspect.signature(adapter.predict_proba)
    rejected_label_arguments = True
    for name in ("test_targets", "calibration_targets"):
        if name in prediction_signature.parameters:
            rejected_label_arguments = False
            break
        try:
            adapter.predict_proba(
                fixture.test,
                partition="test",
                fixture_id=fixture.name,
                fixture_sha256=fixture.sha256,
                split_identity=f"fixture:{fixture.name}:seed={adapter.seed}",
                **{name: np.asarray([], dtype=np.int64)},
            )
        except TypeError:
            continue
        else:
            rejected_label_arguments = False
            break
    if not rejected_label_arguments:
        raise AdapterFailure(
            FailureCategory.FAIL_LEAKAGE,
            "prediction interface accepted a calibration or test target argument",
        )

    row_id_column = next(
        name for name in fixture.train.columns if str(name).startswith("__sg_")
    )
    duplicate_rows = fixture.train.copy(deep=True)
    duplicate_rows.loc[1, row_id_column] = duplicate_rows.loc[0, row_id_column]
    noncontiguous_targets = np.where(np.asarray(fixture.target) == 0, 0, 2).astype(np.int64)

    def rejected_fit(features: Any, targets: np.ndarray) -> bool:
        probe = create_adapter(adapter.model_id, seed=adapter.seed, device=adapter.device)
        try:
            probe.fit(
                features,
                targets,
                fixture_id=fixture.name,
                fixture_sha256=fixture.sha256,
                split_identity=f"fixture:{fixture.name}:seed={adapter.seed}",
            )
        except AdapterFailure as exc:
            return exc.category == FailureCategory.FAIL_INPUT
        finally:
            probe.release()
        return False

    duplicate_rejected = rejected_fit(duplicate_rows, fixture.target)
    empty_rejected = rejected_fit(fixture.train.iloc[:0].copy(), np.asarray([], dtype=np.int64))
    single_class_rejected = rejected_fit(
        fixture.train,
        np.zeros(len(fixture.target), dtype=np.int64),
    )
    noncontiguous_rejected = rejected_fit(fixture.train, noncontiguous_targets)
    if not all((duplicate_rejected, empty_rejected, single_class_rejected, noncontiguous_rejected)):
        raise AdapterFailure(
            FailureCategory.FAIL_LEAKAGE,
            "a malformed row/target input was not rejected by the adapter lifecycle",
        )

    logical_case_id = adapter_logical_case_id(
        model_id=adapter.model_id,
        fixture_sha256=fixture.sha256,
        device=adapter.device,
        partition=prediction.partition,
        seed=adapter.seed,
        parameter_sha256=adapter.parameter_sha256,
        preprocessing_sha256=prediction.preprocessing_sha256,
        checkpoint_sha256=prediction.checkpoint_sha256,
        roundtrip=roundtrip,
    )
    evidence_payload = {
        "schema_version": 1,
        "model_id": adapter.model_id,
        "fixture_id": fixture.name,
        "fixture_sha256": fixture.sha256,
        "device": adapter.device,
        "seed": adapter.seed,
        "source_commit": adapter.source_commit,
        "model_spec_sha256": adapter.model_spec_sha256,
        "parameter_sha256": adapter.parameter_sha256,
        "adapter_implementation_sha256": adapter.adapter_sha256,
        "preprocessing_implementation_sha256": adapter.preprocessing_implementation_sha256,
        "preprocessing_state_sha256": before,
        "logical_case_id": logical_case_id,
        "training_feature_sha256": training_feature_hash,
        "fit_row_ids_sha256": sha256_canonical_json(training_ids),
        "training_target_sha256": training_target_hash,
        "inference_feature_sha256": hash_dataframe_logically(inference_features),
        "inference_row_ids_sha256": sha256_canonical_json(inference_ids),
        "preprocessing_state_before_sha256": before,
        "preprocessing_state_after_sha256": after,
        "fit_call_count": adapter.preprocessor.fit_call_count,
        "test_label_access_attempts": 1,
        "calibration_label_access_attempts": 1,
        "heldout_labels_available": False,
        "prediction_interface_rejects_label_arguments": rejected_label_arguments,
        "state_unchanged": before == after,
        "sentinel_excluded_from_fit": sentinel_excluded,
        "reordered_inference_preserves_state": sentinel_state == after,
        "duplicate_row_ids_rejected": duplicate_rejected,
        "empty_inputs_rejected": empty_rejected,
        "single_class_targets_rejected": single_class_rejected,
        "noncontiguous_target_codes_rejected": noncontiguous_rejected,
        "preprocessing_fit_receives_features_only": True,
        "forbidden_predictor_checks": forbidden_checks,
        "passed": True,
    }
    evidence_payload["evidence_hash"] = sha256_canonical_json(evidence_payload)
    return AdapterLeakageEvidence.model_validate(evidence_payload).model_dump(mode="json")


def _contains_machine_path(value: Any) -> bool:
    if isinstance(value, str):
        return bool(re.match(r"^(?:[A-Za-z]:\\|/|\\\\)", value))
    if isinstance(value, dict):
        return any(_contains_machine_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_machine_path(item) for item in value)
    return False


def validate_inventory_evidence(
    inventory_payload: dict[str, Any] | ModelAdapterInventory,
    evidence_payload: dict[str, Any] | AdapterLeakageEvidenceManifest,
    *,
    root: str | Path | None = None,
) -> tuple[ModelAdapterInventory, AdapterLeakageEvidenceManifest]:
    """Validate exact coverage, hashes, portable identities, fixtures, and leakage links."""

    inventory = (
        inventory_payload
        if isinstance(inventory_payload, ModelAdapterInventory)
        else ModelAdapterInventory.model_validate(inventory_payload)
    )
    evidence = (
        evidence_payload
        if isinstance(evidence_payload, AdapterLeakageEvidenceManifest)
        else AdapterLeakageEvidenceManifest.model_validate(evidence_payload)
    )
    if inventory.source_commit != evidence.source_commit:
        raise ValueError("inventory and leakage evidence source commits differ")
    if any(
        _contains_machine_path(row.parameter_identity.model_dump(mode="json"))
        for row in inventory.records
        if row.parameter_identity
    ):
        raise ValueError("canonical parameter identities cannot expose machine-local paths")

    repository_root = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parents[4]
    )
    from ...models.registry import build_parameters, load_model_registry
    from .base import (
        _implementation_hash,
        _preprocessing_implementation_hash,
        _row_identifiers,
    )
    from .catboost_adapter import CatBoostAdapter
    from .fixtures import adapter_fixture_catalog
    from .logistic_regression_adapter import LogisticRegressionAdapter
    from .tabicl_adapter import TabICLAdapter
    from .tabpfn_adapter import TabPFNAdapter
    from .xgboost_adapter import XGBoostAdapter

    adapter_types = {
        "LR-1.9": LogisticRegressionAdapter,
        "CAT-1.2": CatBoostAdapter,
        "XGB-3.4": XGBoostAdapter,
        "TPFN3-8.5": TabPFNAdapter,
        "TICL2-2.2": TabICLAdapter,
    }
    specs = load_model_registry(repository_root / "configs/runtime/model_compatibility.yaml")
    spec_by_id = {spec.id: spec for spec in specs}
    expected_adapter_hashes = {
        model_id: _implementation_hash(adapter_type)
        for model_id, adapter_type in adapter_types.items()
    }
    expected_preprocessing_hash = _preprocessing_implementation_hash()
    fixture_catalog = adapter_fixture_catalog(1729)
    proofs = {proof.logical_case_id: proof for proof in evidence.records}
    if len(proofs) != len(evidence.records):
        raise ValueError("leakage proof logical identities are not unique")

    for record in inventory.records:
        if record.parameter_identity is None:
            raise ValueError("passing inventory record is missing canonical parameters")
        proof = proofs.get(record.logical_case_id)
        if proof is None or record.leakage_evidence_sha256 != proof.evidence_hash:
            raise ValueError(f"missing or mismatched leakage evidence for {record.logical_case_id}")
        fixture = fixture_catalog.get(record.fixture_id)
        if fixture is None or fixture.sha256 != record.fixture_sha256:
            raise ValueError(f"fixture identity mismatch for {record.model_id}/{record.fixture_id}")
        spec = spec_by_id[record.model_id]
        spec_hash = sha256_canonical_json(spec.model_dump(mode="json"))
        if record.model_spec_sha256 != spec_hash or proof.model_spec_sha256 != spec_hash:
            raise ValueError(f"frozen model spec hash mismatch for {record.model_id}")
        if record.adapter_sha256 != expected_adapter_hashes[record.model_id]:
            raise ValueError(f"adapter implementation hash mismatch for {record.model_id}")
        if record.preprocessing_implementation_sha256 != expected_preprocessing_hash:
            raise ValueError("preprocessing implementation hash mismatch in inventory")
        if proof.preprocessing_implementation_sha256 != expected_preprocessing_hash:
            raise ValueError("preprocessing implementation hash mismatch in leakage proof")
        for field_name in (
            "model_id",
            "fixture_id",
            "fixture_sha256",
            "device",
            "seed",
            "source_commit",
            "model_spec_sha256",
            "parameter_sha256",
        ):
            if getattr(proof, field_name) != getattr(record, field_name):
                raise ValueError(
                    f"leakage proof {field_name} mismatch for {record.model_id}/{record.fixture_id}"
                )
        if proof.adapter_implementation_sha256 != record.adapter_sha256:
            raise ValueError("leakage proof adapter implementation hash mismatch")
        if proof.preprocessing_state_sha256 != record.preprocessing_sha256:
            raise ValueError("leakage proof does not bind the recorded fitted preprocessing state")
        training_ids, training_features = _row_identifiers(fixture.train)
        inference_ids, inference_features = _row_identifiers(fixture.test)
        if (
            proof.fit_row_ids_sha256 != sha256_canonical_json(training_ids)
            or proof.training_feature_sha256 != hash_dataframe_logically(training_features)
            or proof.training_target_sha256 != sha256_canonical_json(fixture.target.tolist())
            or proof.inference_row_ids_sha256 != sha256_canonical_json(inference_ids)
            or proof.inference_feature_sha256 != hash_dataframe_logically(inference_features)
        ):
            raise ValueError("leakage proof source hashes do not match deterministic fixtures")
        expected_parameters = build_parameters(
            spec,
            seed=record.seed,
            device=record.device,
            target_classes=len(record.class_order),
        )
        checkpoint = FOUNDATION_CHECKPOINTS.get(record.model_id)
        if checkpoint is not None:
            expected_parameters["model_path"] = checkpoint[0]
            if record.model_id == "TICL2-2.2":
                expected_parameters["allow_auto_download"] = False
        observed_parameters = dict(record.parameter_identity.parameters)
        if checkpoint is not None:
            observed_parameters["model_path"] = checkpoint[0]
        if observed_parameters != expected_parameters:
            raise ValueError(
                f"constructor parameters differ from the frozen registry for {record.model_id}"
            )
        if (record.package_name, record.package_version) != MODEL_PACKAGE_VERSIONS[record.model_id]:
            raise ValueError(f"model package identity mismatch for {record.model_id}")
        if (
            checkpoint is not None
            and (
                record.checkpoint_identifier,
                record.checkpoint_sha256,
            )
            != checkpoint
        ):
            raise ValueError(f"checkpoint identity mismatch for {record.model_id}")
    return inventory, evidence


__all__ = ["build_leakage_evidence", "validate_inventory_evidence"]
