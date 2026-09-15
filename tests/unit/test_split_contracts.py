import pytest
from pydantic import ValidationError

from schemaguard.splits.contracts import SplitGenerationConfig


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "strategy": "stratified_group_5fold_v1",
        "grouping_method": "typed_predictor_sha256_v1",
        "group_by": "predictors",
        "group_folds": 5,
        "train_fraction": 0.60,
        "calibration_fraction": 0.20,
        "test_fraction": 0.20,
        "minimum_class_count_per_partition": 5,
        "seeds": [1729, 2718, 31415, 57721, 161803],
        "datasets": [3, 23, 29, 31, 36, 37, 38, 44, 46, 50, 54, 1067, 1464, 1489],
        "max_workers": 2,
    }


def test_frozen_configuration_parses_and_hashes() -> None:
    config = SplitGenerationConfig.model_validate(valid_payload())
    assert len(config.configuration_hash) == 64
    assert config.group_folds == 5


def test_unknown_configuration_key_is_rejected() -> None:
    payload = valid_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        SplitGenerationConfig.model_validate(payload)


def test_wrong_frozen_dataset_or_seed_set_is_rejected() -> None:
    payload = valid_payload()
    payload["seeds"] = [1]
    with pytest.raises(ValidationError):
        SplitGenerationConfig.model_validate(payload)


def test_invalid_fraction_is_rejected() -> None:
    payload = valid_payload()
    payload["test_fraction"] = 0.30
    with pytest.raises(ValidationError):
        SplitGenerationConfig.model_validate(payload)
