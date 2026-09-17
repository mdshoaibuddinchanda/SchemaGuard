from __future__ import annotations

import numpy as np
import pytest

from schemaguard.models.adapters.contracts import AdapterFailure, FailureCategory
from tests.unit.adapter_test_support import FIXTURES, fit_predict, reordered


@pytest.mark.parametrize("fixture_id", ["binary_numerical", "multiclass_numerical"])
def test_logistic_regression_binary_and_multiclass(fixture_id: str) -> None:
    fixture = FIXTURES[fixture_id]
    adapter, first = fit_predict("LR-1.9", fixture)
    try:
        second = adapter.predict_proba(
            fixture.test,
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed=1729",
        )
        assert first.class_order == list(fixture.classes)
        assert first.row_ids == fixture.test["__sg_row_id"].tolist()
        assert np.max(np.abs(np.asarray(first.probabilities) - second.probabilities)) <= 1e-10
        reversed_result = adapter.predict_proba(
            reordered(fixture.test),
            partition="test",
            fixture_id=fixture.name,
            fixture_sha256=fixture.sha256,
            split_identity=f"fixture:{fixture.name}:seed=1729",
        )
        np.testing.assert_allclose(
            np.asarray(first.probabilities)[::-1], reversed_result.probabilities, rtol=0, atol=1e-10
        )
    finally:
        adapter.release()


def test_logistic_regression_rejects_single_class_without_fitting() -> None:
    fixture = FIXTURES["binary_numerical"]
    adapter = __import__(
        "schemaguard.models.adapters.factory", fromlist=["create_adapter"]
    ).create_adapter("LR-1.9")
    try:
        with pytest.raises(AdapterFailure) as error:
            adapter.fit(
                fixture.train,
                np.zeros(len(fixture.target), dtype=np.int64),
                fixture_id=fixture.name,
                fixture_sha256=fixture.sha256,
                split_identity="invalid-single-class",
            )
        assert error.value.category == FailureCategory.FAIL_INPUT
        assert not adapter.fitted
    finally:
        adapter.release()
