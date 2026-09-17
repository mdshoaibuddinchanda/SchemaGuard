from __future__ import annotations

from tests.unit.adapter_test_support import FIXTURES, fit_predict


def test_catboost_native_categorical_and_unseen_value_inference() -> None:
    fixture = FIXTURES["unseen_category"]
    adapter, prediction = fit_predict("CAT-1.2", fixture)
    try:
        assert adapter.parameters["loss_function"] == "Logloss"
        assert prediction.class_order == [0, 1]
        assert len(prediction.row_ids) == len(fixture.test)
        assert prediction.probabilities_sha256
    finally:
        adapter.release()


def test_catboost_multiclass_loss_is_resolved_from_training_classes() -> None:
    fixture = FIXTURES["multiclass_numerical"]
    adapter, prediction = fit_predict("CAT-1.2", fixture)
    try:
        assert adapter.parameters["loss_function"] == "MultiClass"
        assert prediction.class_order == [0, 1, 2]
    finally:
        adapter.release()
