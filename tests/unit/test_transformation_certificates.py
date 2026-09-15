from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.transformations.contracts import TransformationCertificate


def test_certificate_rejects_n_a_without_reason_and_bad_hash() -> None:
    payload = {
        "view_id": "V00",
        "view_name": "identity",
        "certificate_type": "CONTROL",
        "dataset_id": 1,
        "dataset_version": "local",
        "seed": 1729,
        "split_strategy": "stratified_group_5fold_v1",
        "partition": "train",
        "fit_scope": "none",
        "source_schema_hash": "a" * 64,
        "output_schema_hash": "a" * 64,
        "source_artifact_hash": "a" * 64,
        "output_artifact_hash": "a" * 64,
        "source_row_id_hash": "a" * 64,
        "output_row_id_hash": "a" * 64,
        "missing_mask_policy": "exact",
        "dtype_policy": "record",
        "numerical_tolerance": {"rtol": 1e-10, "atol": 1e-12},
        "configuration_hash": "a" * 64,
        "implementation_hash": "a" * 64,
        "source_commit": "abc",
        "validation_status": "N/A",
        "validation_results": {},
        "created_at": "2026-01-01T00:00:00Z",
    }
    with pytest.raises(ValidationError):
        TransformationCertificate.model_validate(payload)
