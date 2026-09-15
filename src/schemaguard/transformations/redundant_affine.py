"""Redundant affine feature projection with z=3*x+7."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseTransformation, FeatureSchema, collision_safe_name, selected_columns


class RedundantAffineTransformation(BaseTransformation):
    view_id = "V06"
    view_name = "redundant_affine_feature"
    certificate_type = "PROJECTION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        if not feature_schema.numeric_columns:
            raise ValueError("no numerical feature is available")
        selected = selected_columns(feature_schema.numeric_columns, dataset_id, seed, 1)[0]
        generated = collision_safe_name(
            f"{selected}__redundant_affine", {str(c) for c in X_train.columns}
        )
        return {
            "selected_columns": [selected],
            "generated_columns": [generated],
            "source_columns": list(X_train.columns),
            "operation": "z=3*x+7",
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = dict(self._fit_parameters)
        out = X.copy(deep=True)
        values = pd.to_numeric(out[params["selected_columns"][0]], errors="raise").astype(float)
        if np.isinf(values).any():
            raise ValueError("non-finite numeric input is not transformable")
        out[params["generated_columns"][0]] = values * 3.0 + 7.0
        return out, params

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        return X_transformed.drop(columns=certificate.parameters["generated_columns"])[
            certificate.parameters["source_columns"]
        ]
