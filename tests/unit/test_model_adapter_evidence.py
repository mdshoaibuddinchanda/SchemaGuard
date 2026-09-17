from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.models.adapters.contracts import AdapterLeakageEvidence
from schemaguard.models.adapters.evidence import validate_inventory_evidence
from schemaguard.utils.hashing import sha256_canonical_json

pytestmark = pytest.mark.evidence


def _leakage_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_id": "LR-1.9",
        "fixture_id": "binary_numerical",
        "fixture_sha256": "a" * 64,
        "device": "cpu",
        "seed": 1729,
        "source_commit": "1" * 40,
        "model_spec_sha256": "b" * 64,
        "parameter_sha256": "5" * 64,
        "adapter_implementation_sha256": "c" * 64,
        "preprocessing_implementation_sha256": "6" * 64,
        "preprocessing_state_sha256": "3" * 64,
        "logical_case_id": "d" * 64,
        "training_feature_sha256": "e" * 64,
        "fit_row_ids_sha256": "f" * 64,
        "training_target_sha256": "0" * 64,
        "inference_feature_sha256": "1" * 64,
        "inference_row_ids_sha256": "2" * 64,
        "preprocessing_state_before_sha256": "3" * 64,
        "preprocessing_state_after_sha256": "3" * 64,
        "fit_call_count": 1,
        "test_label_access_attempts": 1,
        "calibration_label_access_attempts": 1,
        "heldout_labels_available": False,
        "prediction_interface_rejects_label_arguments": True,
        "state_unchanged": True,
        "sentinel_excluded_from_fit": True,
        "reordered_inference_preserves_state": True,
        "duplicate_row_ids_rejected": True,
        "empty_inputs_rejected": True,
        "single_class_targets_rejected": True,
        "noncontiguous_target_codes_rejected": True,
        "preprocessing_fit_receives_features_only": True,
        "forbidden_predictor_checks": {
            "target_code": True,
            "target_label": True,
            "__sg_group_id": True,
        },
        "passed": True,
    }
    payload.update(changes)
    payload.pop("evidence_hash", None)
    payload["evidence_hash"] = sha256_canonical_json(payload)
    return payload


def test_leakage_evidence_is_content_addressed_and_semantically_strict() -> None:
    evidence = AdapterLeakageEvidence.model_validate(_leakage_payload())
    assert evidence.evidence_hash == sha256_canonical_json(
        evidence.model_dump(mode="json", exclude={"evidence_hash"})
    )
    with pytest.raises(ValidationError):
        AdapterLeakageEvidence.model_validate(
            _leakage_payload(sentinel_excluded_from_fit=False)
        )
    with pytest.raises(ValidationError):
        AdapterLeakageEvidence.model_validate(
            _leakage_payload(preprocessing_state_after_sha256="4" * 64)
        )
    with pytest.raises(ValidationError):
        AdapterLeakageEvidence.model_validate(_leakage_payload(heldout_labels_available=True))


def test_dedicated_evidence_validator_rejects_missing_case_proof() -> None:
    with pytest.raises((ValidationError, ValueError), match="leakage|matrix|37"):
        validate_inventory_evidence(
            {"source_commit": "1" * 40, "records": []},
            {"source_commit": "1" * 40, "records": []},
        )
