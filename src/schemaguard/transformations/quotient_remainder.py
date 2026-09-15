"""Lossless integer quotient/remainder decomposition."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import (
    BaseTransformation,
    FeatureSchema,
    collision_safe_name,
    restore_series_dtype,
    selected_columns,
    source_dtype_map,
)


class QuotientRemainderTransformation(BaseTransformation):
    view_id = "V07"
    view_name = "integer_quotient_remainder"
    certificate_type = "BIJECTION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        if not feature_schema.integer_columns:
            raise ValueError("no integer feature is available")
        selected = selected_columns(feature_schema.integer_columns, dataset_id, seed, 1)[0]
        existing = {str(c) for c in X_train.columns}
        quotient = collision_safe_name(f"{selected}__quotient", existing)
        remainder = collision_safe_name(f"{selected}__remainder", existing | {quotient})
        return {
            "selected_columns": [selected],
            "generated_columns": [quotient, remainder],
            "modulus": int(self.config.get("quotient_modulus", 10)),
            "source_columns": list(X_train.columns),
            "source_dtype": str(X_train[selected].dtype),
            "source_dtypes": source_dtype_map(feature_schema),
            "operation": "x=q*modulus+r; 0<=r<modulus",
            "inverse_operation": "reconstruct_original_dtype",
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = dict(self._fit_parameters)
        column = params["selected_columns"][0]
        modulus = params["modulus"]
        values = pd.to_numeric(X[column], errors="raise")
        if not pd.api.types.is_integer_dtype(values.dropna()):
            raise ValueError("quotient/remainder requires integer values")
        quotient = (values // modulus).astype("Int64")
        remainder = (values % modulus).astype("Int64")
        out = X.drop(columns=[column]).copy(deep=True)
        position = list(X.columns).index(column)
        out.insert(position, params["generated_columns"][0], quotient)
        out.insert(position + 1, params["generated_columns"][1], remainder)
        return out, params

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        params = certificate.parameters
        quotient, remainder = params["generated_columns"]
        restored = X_transformed[quotient] * params["modulus"] + X_transformed[remainder]
        restored = restore_series_dtype(
            restored, params["source_dtypes"][params["selected_columns"][0]]
        )
        out = X_transformed.drop(columns=[quotient, remainder]).copy(deep=True)
        position = list(params["source_columns"]).index(params["selected_columns"][0])
        out.insert(position, params["selected_columns"][0], restored)
        return out[params["source_columns"]]
