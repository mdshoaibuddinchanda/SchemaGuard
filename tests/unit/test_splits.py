from pathlib import Path

import pandas as pd

from schemaguard.constants import GROUP_ID_COLUMN, ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.data.pipeline import load_smoke_config
from schemaguard.data.splits import (
    generate_splits,
    predictor_group_ids,
    validate_split_assignments,
)

CONFIG = Path(__file__).parents[2] / "configs" / "datasets" / "smoke_blood_transfusion.yaml"


def split_config():
    return load_smoke_config(CONFIG)


def make_tables(row_count: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_ids = [f"row-{index:03d}" for index in range(row_count)]
    codes = [index % 2 for index in range(row_count)]
    features = pd.DataFrame({ROW_ID_COLUMN: row_ids, "x": range(row_count)})
    targets = pd.DataFrame(
        {
            ROW_ID_COLUMN: row_ids,
            "target_label": [str(code) for code in codes],
            TARGET_CODE_COLUMN: codes,
        }
    )
    return features, targets


def test_identical_predictors_receive_identical_group_ids() -> None:
    features = pd.DataFrame(
        {
            ROW_ID_COLUMN: ["a", "b", "c"],
            "integer_feature": pd.Series([7, 7, 8], dtype="int64"),
            "float_feature": pd.Series([1.5, 1.5, 1.5], dtype="float64"),
        }
    )
    group_ids = predictor_group_ids(features)
    assert group_ids.iloc[0] == group_ids.iloc[1]
    assert group_ids.iloc[0] != group_ids.iloc[2]
    assert group_ids.name == GROUP_ID_COLUMN


def test_group_ids_are_type_aware_and_independent_of_row_order() -> None:
    integer_features = pd.DataFrame({ROW_ID_COLUMN: ["a"], "x": pd.Series([1], dtype="int64")})
    float_features = pd.DataFrame({ROW_ID_COLUMN: ["a"], "x": pd.Series([1.0], dtype="float64")})
    assert (
        predictor_group_ids(integer_features).iloc[0] != predictor_group_ids(float_features).iloc[0]
    )

    features = pd.DataFrame(
        {ROW_ID_COLUMN: ["a", "b", "c"], "x": [1, 2, 1], "y": ["left", "right", "left"]}
    )
    original = dict(zip(features[ROW_ID_COLUMN], predictor_group_ids(features), strict=True))
    shuffled = features.sample(frac=1, random_state=7).reset_index(drop=True)
    observed = dict(zip(shuffled[ROW_ID_COLUMN], predictor_group_ids(shuffled), strict=True))
    assert observed == original


def test_grouped_splits_are_complete_disjoint_and_repeatable(tmp_path: Path) -> None:
    features, targets = make_tables()
    first = generate_splits(features, targets, split_config(), tmp_path / "first", "a" * 64)
    second = generate_splits(
        features.sample(frac=1, random_state=5).reset_index(drop=True),
        targets.sample(frac=1, random_state=7).reset_index(drop=True),
        split_config(),
        tmp_path / "second",
        "a" * 64,
    )
    assert first.assignments.equals(second.assignments)
    assert first.manifest.assignment_file_sha256 == second.manifest.assignment_file_sha256
    assert first.manifest.row_counts == {"train": 12, "calibration": 4, "test": 4}
    assert first.manifest.strategy == "stratified_group_5fold_v1"
    assert first.manifest.group_by == "predictors"
    assert first.manifest.group_folds == 5
    assert first.manifest.total_predictor_groups == 20
    assert first.manifest.duplicate_predictor_groups == 0
    assert first.manifest.predictor_duplicate_groups_crossing_splits == 0
    details = validate_split_assignments(
        first.assignments, targets, split_config(), first.group_ids
    )
    assert details["splits_disjoint"]
    assert details["row_id_union_complete"]
    assert details["both_classes_in_every_split"]
    assert details["group_statistics"]["predictor_duplicate_groups_crossing_splits"] == 0


def test_conflicting_label_duplicate_groups_are_detected(tmp_path: Path) -> None:
    predictor_values = [0, 0, *range(1, 35)]
    codes = [0, 1, *[index % 2 for index in range(34)]]
    row_ids = [f"row-{index:03d}" for index in range(len(predictor_values))]
    features = pd.DataFrame({ROW_ID_COLUMN: row_ids, "x": predictor_values})
    targets = pd.DataFrame(
        {
            ROW_ID_COLUMN: row_ids,
            "target_label": [str(code) for code in codes],
            TARGET_CODE_COLUMN: codes,
        }
    )
    result = generate_splits(features, targets, split_config(), tmp_path / "conflict", "b" * 64)
    assert result.manifest.conflicting_target_groups == 1
    assert result.manifest.predictor_duplicate_groups_crossing_splits == 0
    assert result.manifest.row_counts == {
        "train": sum(result.assignments["split"] == "train"),
        "calibration": sum(result.assignments["split"] == "calibration"),
        "test": sum(result.assignments["split"] == "test"),
    }
