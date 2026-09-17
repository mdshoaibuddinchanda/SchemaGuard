from __future__ import annotations

from tests.unit.adapter_test_support import FIXTURES, fit_predict


def test_xgboost_binary_and_multiclass_probability_contract() -> None:
    for fixture_id in ("binary_numerical", "multiclass_numerical"):
        fixture = FIXTURES[fixture_id]
        adapter, prediction = fit_predict("XGB-3.4", fixture)
        try:
            assert prediction.class_order == list(fixture.classes)
            assert len(prediction.probabilities) == len(fixture.test)
            assert adapter.parameters["n_jobs"] == 2
        finally:
            adapter.release()


def test_xgboost_unseen_category_uses_training_onehot_vocabulary() -> None:
    fixture = FIXTURES["unseen_category"]
    adapter, prediction = fit_predict("XGB-3.4", fixture)
    try:
        assert prediction.status == "PASS"
        assert adapter.preprocessing_strategy == "common_onehot_no_scaling"
    finally:
        adapter.release()
