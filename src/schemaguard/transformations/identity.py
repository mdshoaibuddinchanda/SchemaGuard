"""Identity control view."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseTransformation, FeatureSchema


class IdentityTransformation(BaseTransformation):
    view_id = "V00"
    view_name = "identity"
    certificate_type = "CONTROL"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        return {"selected_columns": [], "generated_columns": [], "operation": "identity"}

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        return X, {"selected_columns": [], "generated_columns": [], "operation": "identity"}

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        return X_transformed
