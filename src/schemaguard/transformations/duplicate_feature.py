"""Exact duplicate-feature projection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseTransformation, FeatureSchema, collision_safe_name, selected_columns


class DuplicateFeatureTransformation(BaseTransformation):
    view_id = "V05"
    view_name = "duplicate_feature"
    certificate_type = "PROJECTION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        candidates = feature_schema.numeric_columns or feature_schema.categorical_columns
        if not candidates:
            raise ValueError("no predictor is available")
        selected = selected_columns(candidates, dataset_id, seed, 1)[0]
        generated = collision_safe_name(f"{selected}__duplicate", {str(c) for c in X_train.columns})
        return {
            "selected_columns": [selected],
            "generated_columns": [generated],
            "source_columns": list(X_train.columns),
            "operation": "exact_duplicate",
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = dict(self._fit_parameters)
        out = X.copy(deep=True)
        out[params["generated_columns"][0]] = out[params["selected_columns"][0]].copy()
        return out, params

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        return X_transformed.drop(columns=certificate.parameters["generated_columns"])[
            certificate.parameters["source_columns"]
        ]
