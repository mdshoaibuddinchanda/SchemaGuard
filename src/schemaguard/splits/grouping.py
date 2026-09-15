"""Stable predictor grouping; the existing typed implementation is authoritative."""

from __future__ import annotations

import hashlib
import inspect

import pandas as pd

from schemaguard.data.splits import canonical_predictor_vector, predictor_group_ids

GROUPING_METHOD = "typed_predictor_sha256_v1"
GROUPING_IMPLEMENTATION_HASH = hashlib.sha256(
    inspect.getsource(canonical_predictor_vector).encode("utf-8")
    + inspect.getsource(predictor_group_ids).encode("utf-8")
).hexdigest()


def grouped_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return row IDs and stable group IDs in the input row order."""

    if "__sg_row_id" not in features:
        raise ValueError("features must contain __sg_row_id")
    grouping_input = features.drop(
        columns=["predictor_group_id", "fold_id", "partition", "split"], errors="ignore"
    )
    groups = predictor_group_ids(grouping_input)
    return pd.DataFrame(
        {"__sg_row_id": features["__sg_row_id"].tolist(), "predictor_group_id": groups.tolist()}
    )


__all__ = [
    "GROUPING_METHOD",
    "GROUPING_IMPLEMENTATION_HASH",
    "grouped_features",
    "predictor_group_ids",
]
