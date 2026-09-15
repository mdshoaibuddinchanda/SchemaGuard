"""Collision-safe, exactly reversible categorical one-hot expansion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import (
    BaseTransformation,
    FeatureSchema,
    collision_safe_name,
    schema_categories,
    selected_columns,
    source_dtype_map,
)
from .category_permutation import _restore_dtype, category_key
from .codec import decode_typed_key, encode_typed_value


class CategoricalOneHotTransformation(BaseTransformation):
    view_id = "V04"
    view_name = "categorical_onehot"
    certificate_type = "BIJECTION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        candidates = selected_columns(feature_schema.categorical_columns, dataset_id, seed, 1)
        if not candidates:
            raise ValueError("no categorical feature is available")
        column = candidates[0]
        schema_values = schema_categories(feature_schema, column)
        values = {
            category_key(value): value
            for value in (schema_values or X_train[column].tolist())
            if not pd.isna(value)
        }
        if len(values) < int(self.config.get("category_minimum", 2)):
            raise ValueError("categorical feature has insufficient training categories")
        names: list[str] = []
        existing = {str(item) for item in X_train.columns}
        for key in sorted(values):
            names.append(
                collision_safe_name(f"{column}__category_{len(names)}", existing | set(names))
            )
        missing_name = collision_safe_name(f"{column}__missing", existing | set(names))
        names.append(missing_name)
        return {
            "selected_columns": [column],
            "generated_columns": names,
            "categories": [encode_typed_value(values[key]) for key in sorted(values)],
            "category_keys": sorted(values),
            "source_columns": list(X_train.columns),
            "source_dtype": str(X_train[column].dtype),
            "source_dtypes": source_dtype_map(feature_schema),
            "missing_column": missing_name,
            "operation": "exactly_one_active_indicator_or_missing_indicator",
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = dict(self._fit_parameters)
        column = params["selected_columns"][0]
        names = params["generated_columns"]
        out = X.drop(columns=[column]).copy(deep=True)
        indicators = np.zeros((len(X), len(names)), dtype="uint8")
        for index, value in enumerate(X[column].tolist()):
            if pd.isna(value):
                indicators[index, -1] = 1
            else:
                key = category_key(value)
                if key not in params["category_keys"]:
                    raise ValueError(f"unseen category in {column}: {value!r}")
                indicators[index, params["category_keys"].index(key)] = 1
        for index, name in enumerate(names):
            out[name] = indicators[:, index]
        return out, params

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        params = certificate.parameters
        names = params["generated_columns"]
        if any(name not in X_transformed.columns for name in names):
            raise ValueError("one-hot reconstruction is missing an indicator column")
        indicators = X_transformed[names].to_numpy()
        if not np.isin(indicators, [0, 1]).all() or not (indicators.sum(axis=1) == 1).all():
            raise ValueError("one-hot input must have exactly one active indicator per row")
        values = []
        for row in indicators:
            index = int(np.argmax(row))
            if index == len(names) - 1:
                values.append(np.nan)
            else:
                values.append(decode_typed_key(params["category_keys"][index]))
        out = X_transformed.drop(columns=names).copy(deep=True)
        source_columns = list(params["source_columns"])
        insert_at = source_columns.index(params["selected_columns"][0])
        out.insert(
            insert_at,
            params["selected_columns"][0],
            _restore_dtype(values, out.index, params["source_dtype"]),
        )
        return out[source_columns]
