"""Type-aware, reversible categorical vocabulary permutation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    BaseTransformation,
    FeatureSchema,
    schema_categories,
    selected_columns,
    typed_value,
)


def category_key(value: Any) -> str:
    if pd.isna(value):
        return "__missing__"
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _decode_typed(item: dict[str, Any]) -> Any:
    if item["type"] == "missing":
        return np.nan
    value = item["value"]
    type_name = item["type"].split(".")[-1]
    if type_name == "int":
        return int(value)
    if type_name == "float":
        return float(value)
    if type_name == "bool":
        return value.lower() == "true"
    return value


class CategoryPermutationTransformation(BaseTransformation):
    view_id = "V03"
    view_name = "category_permutation"
    certificate_type = "BIJECTION"

    def _fit(
        self, X_train: pd.DataFrame, dataset_id: int | str, seed: int, feature_schema: FeatureSchema
    ) -> dict[str, Any]:
        candidates = selected_columns(feature_schema.categorical_columns, dataset_id, seed, 1)
        if not candidates:
            raise ValueError("no categorical feature is available")
        column = candidates[0]
        schema_values = schema_categories(feature_schema, column)
        values = list(schema_values) or [
            value for value in X_train[column].tolist() if not pd.isna(value)
        ]
        unique: dict[str, Any] = {category_key(value): value for value in values}
        ordered_keys = sorted(unique, key=lambda key: key)
        if len(ordered_keys) < int(self.config.get("category_minimum", 2)):
            raise ValueError("categorical feature has insufficient training categories")
        digest = hashlib.sha256(f"{dataset_id}|{seed}|{column}".encode()).hexdigest()
        shift = int(digest[:8], 16) % (len(ordered_keys) - 1) + 1
        forward = {
            key: ordered_keys[(index + shift) % len(ordered_keys)]
            for index, key in enumerate(ordered_keys)
        }
        inverse = {value: key for key, value in forward.items()}
        return {
            "selected_columns": [column],
            "generated_columns": [],
            "categories": [typed_value(unique[key]) for key in ordered_keys],
            "forward_order": [ordered_keys.index(forward[key]) for key in ordered_keys],
            "inverse_order": [ordered_keys.index(inverse[key]) for key in ordered_keys],
            "forward_keys": forward,
            "inverse_keys": inverse,
            "shift": shift,
            "source_columns": list(X_train.columns),
            "source_dtype": str(X_train[column].dtype),
        }

    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        params = dict(self._fit_parameters)
        column = params["selected_columns"][0]
        out = X.copy(deep=True)
        allowed = params["forward_keys"]
        transformed = []
        for value in out[column].tolist():
            key = category_key(value)
            if key == "__missing__":
                transformed.append(np.nan)
            elif key not in allowed:
                raise ValueError(f"unseen category in {column}: {value!r}")
            else:
                transformed.append(_decode_key(allowed[key]))
        out[column] = _restore_dtype(transformed, out.index, params["source_dtype"])
        return out, params

    def _reconstruct(self, X_transformed: pd.DataFrame, certificate) -> pd.DataFrame:
        params = certificate.parameters
        column = params["selected_columns"][0]
        allowed = params["inverse_keys"]
        out = X_transformed.copy(deep=True)
        restored = []
        for value in out[column].tolist():
            key = category_key(value)
            if key == "__missing__":
                restored.append(np.nan)
            elif key not in allowed:
                raise ValueError(f"unknown permuted category in {column}: {value!r}")
            else:
                restored.append(_decode_key(allowed[key]))
        out[column] = _restore_dtype(restored, out.index, params["source_dtype"])
        return out[list(params["source_columns"])]


def _decode_key(key: str) -> Any:
    value = key.rsplit(":", 1)[-1]
    type_name = key.split(":", 1)[0].split(".")[-1]
    if type_name == "int":
        return int(value)
    if type_name == "float":
        return float(value)
    if type_name == "bool":
        return value.lower() == "true"
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _restore_dtype(values: list[Any], index: pd.Index, dtype: str) -> pd.Series:
    if dtype in {"str", "string"}:
        return pd.Series(values, index=index, dtype="string")
    return pd.Series(values, index=index, dtype=dtype if dtype != "object" else "object")
