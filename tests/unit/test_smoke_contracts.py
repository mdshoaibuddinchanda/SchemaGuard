from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from schemaguard.experiments.contracts import SmokePlan, condition_identity
from tests.integration.smoke_experiment_support import tiny_smoke_plan


def test_smoke_plan_is_strict_and_contains_ten_unique_conditions(tmp_path: Path) -> None:
    plan = tiny_smoke_plan(tmp_path)
    restored = SmokePlan.model_validate_json(plan.model_dump_json())
    assert restored.plan_sha256 == plan.plan_sha256
    assert len(plan.conditions) == len({item.condition_id for item in plan.conditions}) == 10
    assert sum(item.device == "cpu" for item in plan.conditions) == 6
    assert sum(item.device == "cuda" for item in plan.conditions) == 4

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SmokePlan.model_validate({**plan.model_dump(), "unexpected": True})

    duplicate = plan.model_dump(mode="json")
    duplicate["conditions"][9] = duplicate["conditions"][0]
    with pytest.raises(ValidationError):
        SmokePlan.model_validate(duplicate)


def test_condition_identity_changes_with_scientific_and_runtime_inputs() -> None:
    base = {
        "dataset_features_sha256": "1" * 64,
        "target_artifact_sha256": "2" * 64,
        "model_id": "LR-1.9",
        "package_version": "1.9.1",
        "view_id": "V00",
        "device": "cpu",
        "precision_policy": "native",
        "seed": 1729,
        "model_spec_sha256": "3" * 64,
        "parameters_sha256": "4" * 64,
        "view_certificate_sha256": "5" * 64,
        "view_features_sha256": "6" * 64,
        "split_sha256": "7" * 64,
        "dependency_lock_sha256": "8" * 64,
        "source_implementation_sha256": "9" * 64,
        "checkpoint_sha256": None,
    }
    baseline = condition_identity(**base)
    assert condition_identity(**{**base, "parameters_sha256": "a" * 64}) != baseline
    assert condition_identity(**{**base, "view_features_sha256": "b" * 64}) != baseline
    assert condition_identity(**{**base, "dependency_lock_sha256": "c" * 64}) != baseline
    assert condition_identity(**{**base, "device": "cuda"}) != baseline


def test_canonical_json_identity_is_independent_of_mapping_insertion_order() -> None:
    from schemaguard.utils.hashing import sha256_canonical_json

    assert sha256_canonical_json({"alpha": 1, "beta": 2}) == sha256_canonical_json(
        {"beta": 2, "alpha": 1}
    )
