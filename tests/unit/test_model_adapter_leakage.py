from __future__ import annotations

import inspect

import numpy as np
import pytest

from schemaguard.models.adapters.contracts import AdapterFailure, FailureCategory
from schemaguard.models.adapters.factory import create_adapter
from schemaguard.models.adapters.fixtures import adapter_fixture_catalog
from schemaguard.models.adapters.preprocessing import TrainingPreprocessor


def test_test_sentinels_never_reach_fit_or_change_fitted_state(monkeypatch) -> None:
    fixture = adapter_fixture_catalog()["binary_numerical"]
    test = fixture.test.copy(deep=True)
    test.loc[:, "numeric_0"] = 987654321.0
    fit_rows: list[int] = []
    original_fit = TrainingPreprocessor.fit

    def fit_spy(preprocessor, training_features):
        fit_rows.append(len(training_features))
        assert 987654321.0 not in training_features["numeric_0"].values
        return original_fit(preprocessor, training_features)

    monkeypatch.setattr(TrainingPreprocessor, "fit", fit_spy)
    adapter = create_adapter("LR-1.9")
    try:
        adapter.fit(
            fixture.train,
            fixture.target,
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity="leakage-sentinel-test",
        )
        fitted_hash = adapter.preprocessor.fitted_state_sha256
        prediction = adapter.predict_proba(
            test,
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity="leakage-sentinel-test",
        )
        assert fit_rows == [len(fixture.train)]
        assert adapter.preprocessor.fitted_state_sha256 == fitted_hash
        assert prediction.row_ids == fixture.test["__sg_row_id"].tolist()
        assert "training_targets" in inspect.signature(adapter.fit).parameters
        assert "test_targets" not in inspect.signature(adapter.predict_proba).parameters
    finally:
        adapter.release()


@pytest.mark.parametrize("forbidden_column", ["target_code", "target_label", "__sg_group_id"])
def test_target_and_split_metadata_are_rejected_as_predictors(forbidden_column: str) -> None:
    fixture = adapter_fixture_catalog()["binary_numerical"]
    invalid = fixture.train.copy()
    invalid[forbidden_column] = 123
    adapter = create_adapter("LR-1.9")
    try:
        with pytest.raises(AdapterFailure) as error:
            adapter.fit(
                invalid,
                fixture.target,
                fixture_id=fixture.name,
                fixture_sha256=fixture.sha256,
                split_identity="metadata-rejection",
            )
        assert error.value.category == FailureCategory.FAIL_LEAKAGE
    finally:
        adapter.release()


def test_duplicate_row_ids_and_empty_training_inputs_fail_closed() -> None:
    fixture = adapter_fixture_catalog()["binary_numerical"]
    adapter = create_adapter("LR-1.9")
    duplicate = fixture.train.copy()
    duplicate.loc[1, "__sg_row_id"] = duplicate.loc[0, "__sg_row_id"]
    try:
        with pytest.raises(AdapterFailure) as duplicate_error:
            adapter.fit(
                duplicate,
                fixture.target,
                fixture_id=fixture.name,
                fixture_sha256=fixture.sha256,
                split_identity="duplicate-row",
            )
        assert duplicate_error.value.category == FailureCategory.FAIL_INPUT
    finally:
        adapter.release()

    empty_adapter = create_adapter("LR-1.9")
    try:
        with pytest.raises(AdapterFailure) as empty_error:
            empty_adapter.fit(
                fixture.train.iloc[:0],
                np.asarray([], dtype=np.int64),
                fixture_id=fixture.name,
                fixture_sha256=fixture.sha256,
                split_identity="empty-training",
            )
        assert empty_error.value.category == FailureCategory.FAIL_INPUT
    finally:
        empty_adapter.release()
