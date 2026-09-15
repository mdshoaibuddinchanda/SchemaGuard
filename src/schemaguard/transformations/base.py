"""Shared interface and deterministic helpers for transformation views."""

from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd

from ..constants import (
    ROW_ID_COLUMN,
    TARGET_CODE_COLUMN,
    TARGET_LABEL_COLUMN,
)
from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json
from .codec import encode_typed_value
from .contracts import TransformationCertificate


@dataclass(frozen=True)
class FeatureSchema:
    """The predictor-only schema seen by a transformation."""

    columns: tuple[str, ...]
    dtypes: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    integer_columns: tuple[str, ...]
    row_id_column: str = ROW_ID_COLUMN
    categorical_values: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    @property
    def schema_hash(self) -> str:
        return sha256_canonical_json(
            {
                "columns": self.columns,
                "dtypes": self.dtypes,
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "integer_columns": self.integer_columns,
                "row_id_column": self.row_id_column,
                "categorical_values": [
                    [column, [encode_typed_value(value) for value in values]]
                    for column, values in self.categorical_values
                ],
            }
        )


def feature_schema_from_frame(
    frame: pd.DataFrame, categorical_values: dict[str, list[Any]] | None = None
) -> FeatureSchema:
    """Infer a predictor schema without inspecting target labels."""

    if frame.columns.duplicated().any():
        raise ValueError("feature columns must be unique")
    forbidden = {TARGET_CODE_COLUMN, TARGET_LABEL_COLUMN}
    if forbidden.intersection(str(column) for column in frame.columns):
        raise ValueError("target columns are not permitted in transformation inputs")
    columns = tuple(str(column) for column in frame.columns)
    numeric = tuple(
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and str(column) != ROW_ID_COLUMN
    )
    categorical = tuple(
        str(column)
        for column in frame.columns
        if str(column) != ROW_ID_COLUMN and str(column) not in numeric
    )
    integer = tuple(
        str(column)
        for column in frame.columns
        if str(column) != ROW_ID_COLUMN and pd.api.types.is_integer_dtype(frame[column])
    )
    return FeatureSchema(
        columns=columns,
        dtypes=tuple(str(frame[column].dtype) for column in frame.columns),
        numeric_columns=numeric,
        categorical_columns=categorical,
        integer_columns=integer,
        categorical_values=tuple(
            (column, tuple(values)) for column, values in sorted((categorical_values or {}).items())
        ),
    )


def schema_categories(schema: FeatureSchema, column: str) -> tuple[Any, ...]:
    return dict(schema.categorical_values).get(column, ())


@dataclass(frozen=True)
class TransformOutput:
    """A transformed table and its independently generated certificate."""

    data: pd.DataFrame
    certificate: TransformationCertificate


class Transformation(Protocol):
    """Typed public interface implemented by every frozen transformation."""

    view_id: str
    view_name: str
    certificate_type: str

    def fit(
        self,
        X_train: pd.DataFrame,
        dataset_id: int | str,
        seed: int,
        feature_schema: FeatureSchema,
    ) -> Self: ...

    def transform(self, X: pd.DataFrame, partition: str) -> pd.DataFrame: ...

    def reconstruct(
        self, X_transformed: pd.DataFrame, certificate: TransformationCertificate
    ) -> pd.DataFrame: ...

    def configuration_hash(self) -> str: ...

    def implementation_hash(self) -> str: ...


class BaseTransformation(ABC):
    """Common fit/transform/certificate plumbing with strict input checks."""

    view_id = ""
    view_name = ""
    certificate_type = "BIJECTION"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.dataset_id: int | str | None = None
        self.seed: int | None = None
        self.feature_schema: FeatureSchema | None = None
        self._fit_parameters: dict[str, Any] = {}
        self._partition_parameters: dict[str, dict[str, Any]] = {}
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        dataset_id: int | str,
        seed: int,
        feature_schema: FeatureSchema,
    ) -> Self:
        _validate_input_frame(X_train)
        if ROW_ID_COLUMN not in X_train.columns:
            raise ValueError(f"{ROW_ID_COLUMN} is required")
        if tuple(map(str, X_train.columns)) != feature_schema.columns:
            raise ValueError("feature schema does not match training columns")
        self.dataset_id = dataset_id
        self.seed = seed
        self.feature_schema = feature_schema
        self._fit_parameters = self._fit(X_train.copy(deep=True), dataset_id, seed, feature_schema)
        self._fitted = True
        return self

    @abstractmethod
    def _fit(
        self,
        X_train: pd.DataFrame,
        dataset_id: int | str,
        seed: int,
        feature_schema: FeatureSchema,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def transform(self, X: pd.DataFrame, partition: str) -> pd.DataFrame:
        self._ensure_fitted()
        _validate_input_frame(X)
        if tuple(map(str, X.columns)) != self.feature_schema.columns:  # type: ignore[union-attr]
            raise ValueError("transform columns do not match the fitted feature schema")
        if partition not in {"train", "calibration", "test"}:
            raise ValueError("partition must be train, calibration, or test")
        output, partition_parameters = self._transform(X.copy(deep=True), partition)
        _validate_output_frame(output)
        self._partition_parameters[partition] = dict(partition_parameters)
        return output

    @abstractmethod
    def _transform(self, X: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        raise NotImplementedError

    def reconstruct(
        self, X_transformed: pd.DataFrame, certificate: TransformationCertificate
    ) -> pd.DataFrame:
        self._ensure_fitted()
        if certificate.view_id != self.view_id:
            raise ValueError("certificate view does not match transformation")
        _validate_output_frame(X_transformed)
        restored = self._reconstruct(X_transformed.copy(deep=True), certificate)
        _validate_output_frame(restored)
        expected = self.feature_schema.columns  # type: ignore[union-attr]
        if tuple(map(str, restored.columns)) != expected:
            raise ValueError("reconstruction did not restore the source schema")
        return restored

    @abstractmethod
    def _reconstruct(
        self, X_transformed: pd.DataFrame, certificate: TransformationCertificate
    ) -> pd.DataFrame:
        raise NotImplementedError

    def configuration_hash(self) -> str:
        return sha256_canonical_json({"view_id": self.view_id, "config": self.config})

    def implementation_hash(self) -> str:
        component_hashes: list[str] = getattr(
            self, "component_implementation_hashes", lambda: []
        )()
        return sha256_canonical_json(
            {
                "schema_version": 3,
                "concrete_hash": _concrete_implementation_hash(type(self)),
                "component_hashes": component_hashes,
            }
        )

    def parameters_for(self, partition: str) -> dict[str, Any]:
        values = dict(self._fit_parameters)
        values.update(self._partition_parameters.get(partition, {}))
        return values

    def certificate_for(
        self,
        source: pd.DataFrame,
        transformed: pd.DataFrame,
        partition: str,
        *,
        dataset_version: str = "local",
        split_strategy: str = "stratified_group_5fold_v1",
        source_target_hash: str | None = None,
        output_target_hash: str | None = None,
        validation_status: str = "PASS",
        validation_results: dict[str, Any] | None = None,
        not_applicable_reason: str | None = None,
    ) -> TransformationCertificate:
        if source_target_hash is None or output_target_hash is None:
            raise ValueError("source_target_hash and output_target_hash are required")
        from .certificates import build_certificate, validate_roundtrip

        provisional = build_certificate(
            transformation=self,
            source=source,
            transformed=transformed,
            partition=partition,
            dataset_version=dataset_version,
            split_strategy=split_strategy,
            source_target_hash=source_target_hash,
            output_target_hash=output_target_hash,
            validation_status="FAIL",
            validation_results={"provisional": True},
            not_applicable_reason=not_applicable_reason,
        )
        restored = self.reconstruct(transformed, provisional)
        evidence = validate_roundtrip(
            source,
            transformed,
            restored,
            provisional,
            rtol=1.0e-10,
            atol=1.0e-12,
            transformation=self,
        )
        final = build_certificate(
            transformation=self,
            source=source,
            transformed=transformed,
            partition=partition,
            dataset_version=dataset_version,
            split_strategy=split_strategy,
            source_target_hash=source_target_hash,
            output_target_hash=output_target_hash,
            validation_status="PASS",
            validation_results=evidence,
            not_applicable_reason=not_applicable_reason,
        )
        validate_roundtrip(
            source,
            transformed,
            restored,
            final,
            rtol=1.0e-10,
            atol=1.0e-12,
            transformation=self,
        )
        return final

    def _ensure_fitted(self) -> None:
        if not self._fitted or self.feature_schema is None:
            raise RuntimeError("transformation must be fitted on training predictors first")


def selected_columns(
    candidates: tuple[str, ...], dataset_id: int | str, seed: int, limit: int
) -> list[str]:
    """Select columns using the frozen dataset/seed/column hash rule."""

    return [
        column
        for _, column in sorted(
            (hashlib.sha256(f"{dataset_id}|{seed}|{column}".encode()).hexdigest(), column)
            for column in candidates
        )[:limit]
    ]


def collision_safe_name(base: str, existing: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in existing:
        candidate = f"{base}__sg_generated_{suffix}"
        suffix += 1
    return candidate


def typed_value(value: Any) -> dict[str, Any]:
    return encode_typed_value(value)


def source_dtype_map(schema: FeatureSchema) -> dict[str, str]:
    return dict(zip(schema.columns, schema.dtypes, strict=True))


def restore_series_dtype(values: pd.Series, dtype: str) -> pd.Series:
    """Restore a source dtype after checking values are representable."""

    missing = values.isna()
    numeric = pd.to_numeric(values, errors="raise")
    integer_dtype = dtype.startswith(("int", "uint", "Int", "UInt"))
    if integer_dtype:
        finite = numeric[~missing].astype(float)
        rounded = np.rint(finite)
        if (
            not np.isfinite(finite).all()
            or not np.isclose(finite, rounded, rtol=1.0e-12, atol=1.0e-10).all()
        ):
            raise ValueError(f"values cannot be safely restored to integer dtype {dtype}")
        numeric = numeric.where(missing, rounded)
        target = pd.api.types.pandas_dtype(dtype)
        info = np.iinfo(target.numpy_dtype if hasattr(target, "numpy_dtype") else target)
        if len(finite) and (finite.min() < info.min or finite.max() > info.max):
            raise ValueError(f"values overflow integer dtype {dtype}")
        return numeric.astype(dtype)
    if dtype in {"float16", "float32", "float64", "Float32", "Float64"}:
        return numeric.astype(dtype)
    if dtype == "object":
        return values.astype(object)
    return values.astype(dtype)


def _validate_input_frame(frame: pd.DataFrame) -> None:
    if frame.columns.duplicated().any():
        raise ValueError("duplicate column names are not permitted")
    if any(str(column).startswith(TARGET_CODE_COLUMN) for column in frame.columns):
        raise ValueError("target code column cannot be transformed")
    if TARGET_LABEL_COLUMN in frame.columns:
        raise ValueError("target label column cannot be transformed")


def _validate_output_frame(frame: pd.DataFrame) -> None:
    if frame.columns.duplicated().any():
        raise ValueError("transformation produced duplicate columns")
    if ROW_ID_COLUMN not in frame.columns:
        raise ValueError("transformation must preserve the row-id column")


def frame_hash(frame: pd.DataFrame) -> str:
    return hash_dataframe_logically(frame.reset_index(drop=True))


@cache
def _concrete_implementation_hash(transformation_type: type[BaseTransformation]) -> str:
    from . import base as base_module
    from . import certificates as certificate_module
    from . import codec as codec_module
    from . import validation as validation_module

    modules = {
        "base": inspect.getsource(base_module),
        "certificate": inspect.getsource(certificate_module),
        "codec": inspect.getsource(codec_module),
        "validation": inspect.getsource(validation_module),
        "concrete": inspect.getsource(transformation_type),
    }
    return sha256_canonical_json({"schema_version": 3, "modules": modules})
