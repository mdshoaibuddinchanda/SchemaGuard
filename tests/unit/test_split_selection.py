import pandas as pd

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.splits.contracts import SplitGenerationConfig
from schemaguard.splits.grouping import predictor_group_ids
from schemaguard.splits.selection import derive_fold_seed, select_fold_assignment

from .test_split_contracts import valid_payload


def _config() -> SplitGenerationConfig:
    return SplitGenerationConfig.model_validate(valid_payload())


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    row_ids = [f"r-{index:03d}" for index in range(100)]
    features = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, "x": range(100), "category": [index % 3 for index in range(100)]}
    )
    features["predictor_group_id"] = predictor_group_ids(features).tolist()
    targets = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, TARGET_CODE_COLUMN: [index % 2 for index in range(100)]}
    )
    return features, targets


def test_selection_is_deterministic_and_considers_all_fold_pairs() -> None:
    features, targets = _tables()
    first = select_fold_assignment(features, targets, _config(), 1729)
    second = select_fold_assignment(features, targets, _config(), 1729)
    assert first["status"] == "PASS"
    assert first["candidate_count"] == 20
    assert first["fold_assignment"] == second["fold_assignment"]
    assert derive_fold_seed(1729) == derive_fold_seed(1729)


def test_selection_changes_with_seed_when_fold_order_changes() -> None:
    features, targets = _tables()
    first = select_fold_assignment(features, targets, _config(), 1729)
    second = select_fold_assignment(features, targets, _config(), 2718)
    assert (
        first["fold_assignment"] != second["fold_assignment"] or first["labels"] != second["labels"]
    )


def test_impossible_minimum_class_constraint_is_explicit() -> None:
    row_ids = [f"r-{index:02d}" for index in range(20)]
    features = pd.DataFrame({ROW_ID_COLUMN: row_ids, "x": range(20)})
    features["predictor_group_id"] = predictor_group_ids(features).tolist()
    targets = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, TARGET_CODE_COLUMN: [0] * 16 + [1] * 4}
    )
    result = select_fold_assignment(features, targets, _config(), 1729)
    assert result["status"] == "CONSTRAINT_INFEASIBLE"


def test_shuffled_physical_rows_keep_the_same_logical_selection() -> None:
    features, targets = _tables()
    first = select_fold_assignment(features, targets, _config(), 1729)
    order = features.sample(frac=1, random_state=31).index
    shuffled_features = features.loc[order].reset_index(drop=True)
    shuffled_targets = targets.set_index(ROW_ID_COLUMN).loc[
        shuffled_features[ROW_ID_COLUMN].tolist()
    ].reset_index()
    second = select_fold_assignment(shuffled_features, shuffled_targets, _config(), 1729)
    assert first["labels"] == second["labels"]
