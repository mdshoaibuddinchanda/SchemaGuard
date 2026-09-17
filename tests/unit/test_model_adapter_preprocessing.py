from __future__ import annotations

import pandas as pd
import pytest
from scipy import sparse

from schemaguard.models.adapters.preprocessing import TrainingPreprocessor


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric": [1.0, None, 3.0, 4.0],
            "category": ["red", "blue", "red", "blue"],
        }
    )


@pytest.mark.parametrize("strategy", ["common_onehot_standard", "common_onehot_no_scaling"])
def test_common_preprocessing_fits_training_only_and_handles_unseen(strategy: str) -> None:
    train = _training_frame()
    original = train.copy(deep=True)
    preprocessor = TrainingPreprocessor(strategy).fit(train)
    state_hash = preprocessor.fitted_state_sha256
    transformed = preprocessor.transform(
        pd.DataFrame({"numeric": [None, 5.0], "category": ["never-seen", "red"]}),
        row_ids=["test-b", "test-a"],
        partition="test",
    )
    assert transformed.shape[0] == 2
    assert sparse.issparse(transformed)
    assert state_hash == preprocessor.fitted_state_sha256
    pd.testing.assert_frame_equal(train, original)


@pytest.mark.parametrize("strategy", ["native_catboost", "native_tabpfn", "native_tabicl"])
def test_native_preprocessing_preserves_roles_and_unknown_category(strategy: str) -> None:
    train = _training_frame()
    preprocessor = TrainingPreprocessor(strategy).fit(train)
    transformed = preprocessor.transform(
        pd.DataFrame({"numeric": [None, 5.0], "category": ["new", "blue"]}),
        row_ids=["x", "y"],
        partition="calibration",
    )
    assert list(transformed.columns) == ["numeric", "category"]
    assert transformed["category"].notna().all()
    assert transformed.iloc[0]["category"] != transformed.iloc[1]["category"]


def test_preprocessor_rejects_schema_drift_and_duplicate_rows() -> None:
    preprocessor = TrainingPreprocessor("native_tabpfn").fit(_training_frame())
    with pytest.raises(ValueError):
        preprocessor.transform(
            _training_frame().assign(extra=1), row_ids=[1, 2, 3, 4], partition="test"
        )
    with pytest.raises(ValueError):
        preprocessor.transform(_training_frame(), row_ids=[1, 1, 2, 3], partition="test")


def test_feature_matrix_cache_is_identity_bound_and_preserves_row_order() -> None:
    preprocessor = TrainingPreprocessor("common_onehot_no_scaling").fit(_training_frame())
    test = pd.DataFrame({"numeric": [2.0, 6.0], "category": ["red", "blue"]})
    first = preprocessor.transform(test, row_ids=["a", "b"], partition="test")
    second = preprocessor.transform(test, row_ids=["a", "b"], partition="test")
    reversed_order = preprocessor.transform(
        test.iloc[::-1].reset_index(drop=True), row_ids=["b", "a"], partition="test"
    )
    assert (first != second).nnz == 0
    assert (first[::-1] != reversed_order).nnz == 0
    assert preprocessor.cache_hits >= 1
