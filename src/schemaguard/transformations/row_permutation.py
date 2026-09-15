"""Training-only row-order control view."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from ..constants import ROW_ID_COLUMN
from .base import BaseTransformation, FeatureSchema


class RowPermutationTransformation(BaseTransformation):
    view_id = "V09"
    view_name = "row_permutation_control"
    certificate_type = "CONTROL"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        return {
            "selected_columns": [],
            "generated_columns": [],
            "source_columns": list(X_train.columns),
            "operation": "training_rows_only",
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = dict(self._fit_parameters)
        if partition != "train":
            params["source_row_ids"] = [str(value) for value in X[ROW_ID_COLUMN].tolist()]
            params["output_row_ids"] = list(params["source_row_ids"])
            return X.copy(deep=True), params
        ordered = sorted(
            range(len(X)),
            key=lambda index: hashlib.sha256(
                f"{self.dataset_id}|{self.seed}|{X.iloc[index][ROW_ID_COLUMN]}".encode()
            ).hexdigest(),
        )
        out = X.iloc[ordered].reset_index(drop=True)
        params["source_row_ids"] = [str(value) for value in X[ROW_ID_COLUMN].tolist()]
        params["output_row_ids"] = [str(value) for value in out[ROW_ID_COLUMN].tolist()]
        return out, params

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        desired = certificate.parameters["source_row_ids"]
        order = {str(row_id): index for index, row_id in enumerate(desired)}
        out = X_transformed.copy(deep=True)
        return (
            out.assign(__sg_restore_order=out[ROW_ID_COLUMN].map(order))
            .sort_values("__sg_restore_order")
            .drop(columns=["__sg_restore_order"])
            .reset_index(drop=True)
        )
