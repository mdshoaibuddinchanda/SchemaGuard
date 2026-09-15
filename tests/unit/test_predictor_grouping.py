import pandas as pd

from schemaguard.constants import ROW_ID_COLUMN
from schemaguard.splits.grouping import predictor_group_ids


def test_grouping_is_typed_and_row_order_independent() -> None:
    frame = pd.DataFrame(
        {
            ROW_ID_COLUMN: ["a", "b", "c"],
            "integer": pd.Series([1, 1, 2], dtype="int64"),
            "text": ["é", "é", "ß"],
        }
    )
    original = dict(zip(frame[ROW_ID_COLUMN], predictor_group_ids(frame), strict=True))
    shuffled = frame.sample(frac=1, random_state=17).reset_index(drop=True)
    observed = dict(zip(shuffled[ROW_ID_COLUMN], predictor_group_ids(shuffled), strict=True))
    assert original == observed
    float_frame = pd.DataFrame({ROW_ID_COLUMN: ["a"], "integer": [1.0], "text": ["é"]})
    assert original["a"] != predictor_group_ids(float_frame).iloc[0]


def test_missing_values_are_stable_and_equal() -> None:
    frame = pd.DataFrame({ROW_ID_COLUMN: ["a", "b"], "x": [pd.NA, pd.NA]})
    groups = predictor_group_ids(frame)
    assert groups.iloc[0] == groups.iloc[1]
