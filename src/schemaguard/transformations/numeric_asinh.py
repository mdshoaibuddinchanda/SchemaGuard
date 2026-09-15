"""Training-only fitted asinh numeric transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseTransformation, FeatureSchema, selected_columns


class NumericAsinhTransformation(BaseTransformation):
    view_id = "V02"
    view_name = "numeric_asinh"
    certificate_type = "BIJECTION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, object]:
        columns = selected_columns(
            feature_schema.numeric_columns,
            dataset_id,
            seed,
            int(self.config.get("max_numeric_columns", 3)),
        )
        scales: dict[str, float] = {}
        for column in columns:
            values = pd.to_numeric(X_train[column], errors="raise").astype(float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            centered = np.abs(finite - median)
            scales[column] = max(float(np.median(centered)) if len(centered) else 0.0, 1.0e-12)
        return {
            "selected_columns": columns,
            "generated_columns": [],
            "scales": scales,
            "operation": "asinh(x / training_median_absolute_deviation)",
            "source_columns": list(X_train.columns),
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, object]]:
        parameters = self._fit_parameters
        out = X.copy(deep=True)
        for column in parameters["selected_columns"]:
            values = pd.to_numeric(out[column], errors="raise").astype(float)
            if np.isinf(values).any():
                raise ValueError("non-finite numeric input is not transformable")
            out[column] = np.arcsinh(values / parameters["scales"][column])
        return out, dict(parameters)

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        out = X_transformed.copy(deep=True)
        for column in certificate.parameters["selected_columns"]:
            out[column] = (
                np.sinh(pd.to_numeric(out[column], errors="raise"))
                * certificate.parameters["scales"][column]
            )
        return out[list(certificate.inverse_parameters["source_columns"])]
