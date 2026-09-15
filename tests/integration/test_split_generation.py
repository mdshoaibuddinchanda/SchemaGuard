import pandas as pd
import pytest

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.splits.contracts import SplitGenerationConfig
from schemaguard.splits.grouping import predictor_group_ids
from schemaguard.splits.selection import select_fold_assignment


def _config() -> SplitGenerationConfig:
    return SplitGenerationConfig.model_validate(
        {
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
    )


@pytest.mark.integration
def test_grouped_split_selection_is_a_complete_micro_integration() -> None:
    row_ids = [f"r-{index:03d}" for index in range(100)]
    features = pd.DataFrame({ROW_ID_COLUMN: row_ids, "x": range(100)})
    features["predictor_group_id"] = predictor_group_ids(features).tolist()
    targets = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, TARGET_CODE_COLUMN: [index % 2 for index in range(100)]}
    )
    selected = select_fold_assignment(features, targets, _config(), 1729)
    assert selected["status"] == "PASS"
    assert selected["candidate_count"] == 20
    assert predictor_group_ids(features.drop(columns=["predictor_group_id"])).notna().all()
