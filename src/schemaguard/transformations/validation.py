"""Invariant checks used by tests, materialization, and inventory validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN, TARGET_LABEL_COLUMN
from .base import frame_hash
from .certificates import validate_roundtrip
from .contracts import TransformationCertificate


def validate_transformation(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    restored: pd.DataFrame,
    certificate: TransformationCertificate,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    if TARGET_CODE_COLUMN in transformed.columns or TARGET_LABEL_COLUMN in transformed.columns:
        raise ValueError("target columns entered the transformed predictor table")
    if len(source) != len(transformed) or len(source) != len(restored):
        raise ValueError("row count changed during transformation")
    if transformed[ROW_ID_COLUMN].duplicated().any():
        raise ValueError("transformation duplicated a row identifier")
    result = validate_roundtrip(source, transformed, restored, certificate, rtol=rtol, atol=atol)
    aligned_transformed = transformed.set_index(ROW_ID_COLUMN).loc[source[ROW_ID_COLUMN].tolist()]
    for column in source.columns:
        if column == ROW_ID_COLUMN:
            continue
        if column in transformed.columns and not source[column].isna().equals(
            aligned_transformed[column].isna().reset_index(drop=True)
        ):
            raise ValueError(f"missing mask changed for source column {column}")
    return result


def validate_deterministic_outputs(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    return frame_hash(first) == frame_hash(second)


def validate_certificates(certificates: list[TransformationCertificate]) -> None:
    if not certificates:
        raise ValueError("at least one certificate is required")
    for certificate in certificates:
        if certificate.validation_status != "PASS":
            raise ValueError("validation inventory cannot contain a failed certificate")
