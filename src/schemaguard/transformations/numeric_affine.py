"""Reversible numeric affine-unit transformation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import (
    BaseTransformation,
    FeatureSchema,
    restore_series_dtype,
    selected_columns,
    source_dtype_map,
)


class NumericAffineTransformation(BaseTransformation):
    view_id = "V01"
    view_name = "numeric_affine_units"
    certificate_type = "BIJECTION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        columns = selected_columns(
            feature_schema.numeric_columns,
            dataset_id,
            seed,
            int(self.config.get("max_numeric_columns", 3)),
        )
        scales = [0.1, 10.0, 1000.0]
        offsets = [-7.0, 13.0]
        parameters = {
            "selected_columns": columns,
            "generated_columns": [],
            "scales": {column: scales[index % len(scales)] for index, column in enumerate(columns)},
            "offsets": {
                column: offsets[index % len(offsets)] for index, column in enumerate(columns)
            },
            "operation": "x_prime = scale * x + offset",
            "source_columns": list(X_train.columns),
            "source_dtypes": source_dtype_map(feature_schema),
            "inverse_operation": "reconstruct_original_dtype",
        }
        return parameters

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        parameters = self._fit_parameters
        out = X.copy(deep=True)
        for column in parameters["selected_columns"]:
            values = pd.to_numeric(out[column], errors="raise")
            if np.isinf(values.astype(float)).any():
                raise ValueError("non-finite numeric input is not transformable")
            out[column] = (
                values.astype(float) * parameters["scales"][column] + parameters["offsets"][column]
            )
        return out, dict(parameters)

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        out = X_transformed.copy(deep=True)
        for column in certificate.parameters["selected_columns"]:
            restored = (
                pd.to_numeric(out[column], errors="raise")
                - certificate.parameters["offsets"][column]
            ) / certificate.parameters["scales"][column]
            out[column] = restore_series_dtype(
                pd.Series(restored, index=out.index),
                certificate.parameters["source_dtypes"][column],
            )
        return out[list(certificate.inverse_parameters["source_columns"])]
