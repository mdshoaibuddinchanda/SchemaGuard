from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.transformations.contracts import (
    TransformationConfig,
    TransformationInventoryRecord,
)


def test_strict_configuration_rejects_unknown_keys() -> None:
    payload = {
        "engine_name": "certified_lossless_transformations",
        "split_strategy": "stratified_group_5fold_v1",
        "max_numeric_columns": 3,
        "category_minimum": 2,
        "quotient_modulus": 10,
        "numerical_rtol": 1e-10,
        "numerical_atol": 1e-12,
        "cpu_workers": 2,
        "gpu_enabled": False,
        "schema_registry_hash": "a" * 64,
        "views": [{"id": f"V{index:02d}"} for index in range(11)],
        "unexpected": True,
    }
    with pytest.raises(ValidationError):
        TransformationConfig.model_validate(payload)


def test_inventory_record_requires_reason_for_not_applicable() -> None:
    with pytest.raises(ValidationError):
        TransformationInventoryRecord(
            dataset_id=1,
            seed=1729,
            view_id="V01",
            view_name="numeric_affine_units",
            status="N/A",
            applicability="NOT_APPLICABLE",
            runtime_seconds=0,
            created_at="2026-01-01T00:00:00Z",
        )
