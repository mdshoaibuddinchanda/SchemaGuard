"""Train-fitted, row-order-preserving preprocessing for the frozen model matrix."""

from __future__ import annotations

import json
from typing import Any

from .resources import configure_thread_limits

configure_thread_limits()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import MissingIndicator, SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from ...constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN, TARGET_LABEL_COLUMN  # noqa: E402
from ...utils.hashing import (  # noqa: E402
    hash_dataframe_logically,
    sha256_canonical_json,
)

_MISSING_TOKEN = "__SCHEMAGUARD_MISSING__"
_UNKNOWN_TOKEN = "__SCHEMAGUARD_UNKNOWN__"
_STRATEGIES = {
    "common_onehot_standard",
    "common_onehot_no_scaling",
    "native_catboost",
    "native_tabpfn",
    "native_tabicl",
}


def _as_json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _category_token(value: Any) -> str:
    if pd.isna(value):
        return _MISSING_TOKEN
    payload = {"type": type(value).__qualname__, "value": _as_json_scalar(value)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _row_hash(row_ids: list[str | int]) -> str:
    return sha256_canonical_json(row_ids)


class TrainingPreprocessor:
    """A preprocessing state that can only be fitted once, on training features."""

    def __init__(self, strategy: str) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"unknown frozen preprocessing strategy: {strategy}")
        self.strategy = strategy
        self.expected_columns: tuple[str, ...] = ()
        self.numeric_columns: tuple[str, ...] = ()
        self.categorical_columns: tuple[str, ...] = ()
        self.category_vocabulary: dict[str, tuple[str, ...]] = {}
        self._transformer: ColumnTransformer | None = None
        self._fitted = False
        self._state_hash: str | None = None
        self.fit_call_count = 0
        self.fit_input_sha256: str | None = None
        self._matrix_cache: dict[str, Any] = {}
        self.cache_hits = 0

    @property
    def fitted_state_sha256(self) -> str:
        if not self._fitted or self._state_hash is None:
            raise RuntimeError("preprocessing has not been fitted")
        return self._state_hash

    @property
    def categorical_feature_indices(self) -> list[int]:
        if not self._fitted:
            raise RuntimeError("preprocessing has not been fitted")
        index = {column: position for position, column in enumerate(self.expected_columns)}
        return [index[column] for column in self.categorical_columns]

    def fit(self, training_features: pd.DataFrame) -> TrainingPreprocessor:
        if self._fitted:
            raise RuntimeError("preprocessing state is immutable after its training fit")
        self.fit_call_count += 1
        self._validate_frame(training_features)
        if training_features.empty or training_features.shape[1] == 0:
            raise ValueError("training features must contain rows and predictors")
        self.expected_columns = tuple(training_features.columns)
        self.fit_input_sha256 = hash_dataframe_logically(training_features)
        self.numeric_columns = tuple(
            column
            for column in self.expected_columns
            if pd.api.types.is_numeric_dtype(training_features[column].dtype)
            and not pd.api.types.is_bool_dtype(training_features[column].dtype)
        )
        self.categorical_columns = tuple(
            column for column in self.expected_columns if column not in self.numeric_columns
        )
        self._validate_numeric_values(training_features)
        self.category_vocabulary = {
            column: tuple(
                sorted(
                    {_category_token(value) for value in training_features[column].tolist()}
                    | {_MISSING_TOKEN, _UNKNOWN_TOKEN}
                )
            )
            for column in self.categorical_columns
        }

        if self.strategy.startswith("common_onehot"):
            transformers: list[tuple[str, Any, list[str]]] = []
            if self.numeric_columns:
                numeric_pipeline: list[tuple[str, Any]] = [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
                ]
                if self.strategy == "common_onehot_standard":
                    numeric_pipeline.append(("scaler", StandardScaler()))
                transformers.extend(
                    [
                        ("numeric", Pipeline(numeric_pipeline), list(self.numeric_columns)),
                        (
                            "missing_mask",
                            MissingIndicator(features="all", error_on_new=False),
                            list(self.numeric_columns),
                        ),
                    ]
                )
            if self.categorical_columns:
                categories = [
                    list(self.category_vocabulary[name]) for name in self.categorical_columns
                ]
                transformers.append(
                    (
                        "categorical",
                        OneHotEncoder(
                            categories=categories,
                            handle_unknown="error",
                            sparse_output=True,
                            dtype=np.float64,
                        ),
                        list(self.categorical_columns),
                    )
                )
            self._transformer = ColumnTransformer(
                transformers,
                remainder="drop",
                sparse_threshold=1.0,
                verbose_feature_names_out=False,
            )
            self._transformer.fit(self._common_frame(training_features))
        else:
            # Native model preprocessing still learns the categorical vocabulary from train
            # only; no calibration/test data or labels enter this state.
            self._state_hash = self._make_state_hash(training_features)
        self._fitted = True
        self._state_hash = self._make_state_hash(training_features)
        return self

    def transform(
        self,
        features: pd.DataFrame,
        *,
        row_ids: list[str | int],
        partition: str,
        use_cache: bool = True,
    ) -> Any:
        self._ensure_fitted()
        if partition not in {"train", "calibration", "test"}:
            raise ValueError("partition must be train, calibration, or test")
        self._validate_frame(features)
        if tuple(features.columns) != self.expected_columns:
            raise ValueError("feature columns or order changed after training fit")
        if len(row_ids) != len(features) or len(row_ids) != len(set(row_ids)):
            raise ValueError("row IDs must be unique and align one-to-one with feature rows")
        if not row_ids:
            raise ValueError("prediction feature batch cannot be empty")
        self._validate_numeric_values(features)
        cache_key = sha256_canonical_json(
            {
                "preprocessing_sha256": self.fitted_state_sha256,
                "frame_sha256": hash_dataframe_logically(features),
                "row_ids_sha256": _row_hash(row_ids),
                "partition": partition,
            }
        )
        if use_cache and cache_key in self._matrix_cache:
            self.cache_hits += 1
            return self._matrix_cache[cache_key]
        if self.strategy.startswith("common_onehot"):
            if self._transformer is None:
                raise RuntimeError("common preprocessing transformer was not fitted")
            result = self._transformer.transform(self._common_frame(features))
        else:
            result = self._native_frame(features)
        if result.shape[0] != len(row_ids):
            raise ValueError("preprocessing changed the row count")
        if use_cache:
            self._matrix_cache[cache_key] = result
        return result

    def transform_native_frame(
        self, features: pd.DataFrame, *, row_ids: list[str | int], partition: str
    ) -> pd.DataFrame:
        """Return train-vocabulary native columns for CatBoost and foundation models."""

        self._ensure_fitted()
        if not self.strategy.startswith("native_"):
            raise ValueError("native-frame access is only valid for native preprocessing")
        self._validate_transform_schema(features, row_ids, partition)
        return self._native_frame(features)

    def _validate_transform_schema(
        self, features: pd.DataFrame, row_ids: list[str | int], partition: str
    ) -> None:
        if partition not in {"train", "calibration", "test"}:
            raise ValueError("invalid partition")
        self._validate_frame(features)
        if tuple(features.columns) != self.expected_columns:
            raise ValueError("feature columns or order changed after training fit")
        if len(row_ids) != len(features) or len(row_ids) != len(set(row_ids)) or not row_ids:
            raise ValueError("row IDs must be unique and align one-to-one with feature rows")
        self._validate_numeric_values(features)

    def _common_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        prepared = features.copy(deep=False)
        if self.categorical_columns:
            prepared = prepared.copy()
            for column in self.categorical_columns:
                allowed = set(self.category_vocabulary[column])
                prepared[column] = [
                    token if (token := _category_token(value)) in allowed else _UNKNOWN_TOKEN
                    for value in features[column].tolist()
                ]
        return prepared

    def _native_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        prepared = features.copy(deep=True)
        for column in self.numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="raise").astype("float64")
        for column in self.categorical_columns:
            allowed = set(self.category_vocabulary[column])
            tokens = [
                token if (token := _category_token(value)) in allowed else _UNKNOWN_TOKEN
                for value in prepared[column].tolist()
            ]
            if self.strategy in {"native_tabpfn", "native_tabicl"}:
                prepared[column] = pd.Categorical(
                    tokens, categories=self.category_vocabulary[column]
                )
            else:
                prepared[column] = tokens
        return prepared

    def _make_state_hash(self, training_features: pd.DataFrame) -> str:
        fitted_numeric: dict[str, Any] = {}
        if self._transformer is not None and hasattr(self._transformer, "transformers_"):
            for name, transformer, columns in self._transformer.transformers_:
                if name != "numeric" or transformer == "drop":
                    continue
                pipeline = transformer
                imputer = pipeline.named_steps["imputer"]
                fitted_numeric["imputer_statistics"] = np.asarray(
                    imputer.statistics_, dtype="float64"
                ).tolist()
                if "scaler" in pipeline.named_steps:
                    scaler = pipeline.named_steps["scaler"]
                    fitted_numeric["mean"] = np.asarray(scaler.mean_, dtype="float64").tolist()
                    fitted_numeric["scale"] = np.asarray(scaler.scale_, dtype="float64").tolist()
                fitted_numeric["columns"] = list(columns)
        return sha256_canonical_json(
            {
                "schema_version": 1,
                "strategy": self.strategy,
                "columns": self.expected_columns or tuple(training_features.columns),
                "dtypes": [str(training_features[column].dtype) for column in training_features],
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "categorical_vocabulary": self.category_vocabulary,
                "fitted_numeric_state": fitted_numeric,
            }
        )

    @staticmethod
    def _validate_frame(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("adapter features must be a pandas DataFrame")
        if features.columns.duplicated().any():
            raise ValueError("duplicate feature names are not permitted")
        if any(not isinstance(name, str) or not name for name in features.columns):
            raise ValueError("feature names must be non-empty strings")
        reserved = {ROW_ID_COLUMN, TARGET_CODE_COLUMN, TARGET_LABEL_COLUMN}
        if reserved.intersection(features.columns):
            raise ValueError("row IDs and targets must be passed separately from predictors")
        if any(name.startswith("__sg_") for name in features.columns):
            raise ValueError("split/group metadata is not permitted in feature matrices")
        if not features.columns.is_unique:
            raise ValueError("feature columns must be unique")

    def _validate_numeric_values(self, features: pd.DataFrame) -> None:
        for column in self.numeric_columns:
            values = pd.to_numeric(features[column], errors="raise").to_numpy(dtype="float64")
            if np.isinf(values).any():
                raise ValueError(f"non-finite numeric input in feature {column!r}")

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("preprocessing must be fitted on training features first")
