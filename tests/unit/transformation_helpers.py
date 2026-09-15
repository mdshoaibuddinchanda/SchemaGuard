from __future__ import annotations

import pandas as pd

from schemaguard.transformations.base import FeatureSchema, feature_schema_from_frame
from schemaguard.utils.hashing import hash_dataframe_logically


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__sg_row_id": [f"row-{index}" for index in range(12)],
            "amount": [0.5, -2.0, 3.25, 4.0, 5.5, 6.0, 7.25, 8.0, 9.5, 10.0, 11.5, 12.0],
            "count": list(range(12)),
            "category": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", None],
        }
    )


def sample_schema(frame: pd.DataFrame | None = None) -> FeatureSchema:
    return feature_schema_from_frame(frame if frame is not None else sample_frame())


def transformation_config() -> dict[str, object]:
    return {"max_numeric_columns": 3, "category_minimum": 2, "quotient_modulus": 10}


def target_hash(frame: pd.DataFrame) -> str:
    target = pd.DataFrame(
        {
            "__sg_row_id": frame["__sg_row_id"],
            "__sg_target_code": [index % 2 for index in range(len(frame))],
        }
    )
    return hash_dataframe_logically(target)
