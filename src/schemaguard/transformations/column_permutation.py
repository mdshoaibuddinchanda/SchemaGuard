"""Deterministic column-order control view."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from ..constants import ROW_ID_COLUMN
from .base import BaseTransformation, FeatureSchema


class ColumnPermutationTransformation(BaseTransformation):
    view_id = "V08"
    view_name = "column_permutation_control"
    certificate_type = "CONTROL"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        row_id = [ROW_ID_COLUMN] if ROW_ID_COLUMN in X_train.columns else []
        predictors = [str(c) for c in X_train.columns if str(c) != ROW_ID_COLUMN]
        ordered = [
            column
            for _, column in sorted(
                (hashlib.sha256(f"{dataset_id}|{seed}|{column}|V08".encode()).hexdigest(), column)
                for column in predictors
            )
        ]
        return {
            "selected_columns": predictors,
            "generated_columns": [],
            "source_columns": list(X_train.columns),
            "permuted_columns": row_id + ordered,
            "inverse_order": list(X_train.columns),
            "operation": "column_order_permutation",
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        return X[self._fit_parameters["permuted_columns"]].copy(deep=True), dict(
            self._fit_parameters
        )

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        return X_transformed[certificate.parameters["source_columns"]].copy(deep=True)
