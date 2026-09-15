"""Certificate construction and invariant checks."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Literal, cast

import pandas as pd

from ..constants import ROW_ID_COLUMN
from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json
from .base import BaseTransformation
from .contracts import TransformationCertificate


@lru_cache(maxsize=1)
def current_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def schema_hash(frame: pd.DataFrame) -> str:
    return sha256_canonical_json(
        {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
        }
    )


def row_id_hash(frame: pd.DataFrame) -> str:
    if ROW_ID_COLUMN not in frame.columns:
        raise ValueError(f"{ROW_ID_COLUMN} is required")
    return hash_dataframe_logically(frame[[ROW_ID_COLUMN]].reset_index(drop=True))


def build_certificate(
    *,
    transformation: BaseTransformation,
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    partition: str,
    dataset_version: str,
    split_strategy: str,
    source_target_hash: str | None,
    output_target_hash: str | None,
    validation_status: str,
    validation_results: dict[str, Any],
    not_applicable_reason: str | None,
) -> TransformationCertificate:
    if transformation.dataset_id is None or transformation.seed is None:
        raise RuntimeError("transformation must be fitted before certifying output")
    certificate_type = cast(
        Literal["BIJECTION", "PROJECTION", "CONTROL", "COMPOSITION"],
        transformation.certificate_type,
    )
    certificate_partition = cast(Literal["train", "calibration", "test"], partition)
    parameters = transformation.parameters_for(partition)
    selected = parameters.get("selected_columns", [])
    generated = parameters.get("generated_columns", [])
    return TransformationCertificate(
        schema_version=1,
        view_id=transformation.view_id,
        view_name=transformation.view_name,
        certificate_type=certificate_type,
        dataset_id=transformation.dataset_id,
        dataset_version=dataset_version,
        seed=transformation.seed,
        split_strategy=split_strategy,
        partition=certificate_partition,
        fit_scope="train_only" if transformation.view_id not in {"V00", "V08", "V09"} else "none",
        source_schema_hash=schema_hash(source),
        output_schema_hash=schema_hash(transformed),
        source_artifact_hash=hash_dataframe_logically(source.reset_index(drop=True)),
        output_artifact_hash=hash_dataframe_logically(transformed.reset_index(drop=True)),
        source_row_id_hash=row_id_hash(source),
        output_row_id_hash=row_id_hash(transformed),
        source_target_hash=source_target_hash,
        output_target_hash=output_target_hash,
        selected_columns=list(selected),
        generated_columns=list(generated),
        parameters=parameters,
        inverse_parameters={"operation": "reconstruct", "source_columns": list(source.columns)},
        missing_mask_policy="preserve_missing_masks_exactly",
        dtype_policy="preserve_unmodified_dtypes; record transformed dtypes",
        numerical_tolerance={"rtol": 1.0e-10, "atol": 1.0e-12},
        configuration_hash=transformation.configuration_hash(),
        implementation_hash=transformation.implementation_hash(),
        source_commit=current_commit(),
        validation_status=validation_status,  # type: ignore[arg-type]
        validation_results=validation_results,
        not_applicable_reason=not_applicable_reason,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def certificate_hash(certificate: TransformationCertificate) -> str:
    """Return the stable identity of a validated certificate."""

    def without_timestamps(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_timestamps(item)
                for key, item in value.items()
                if key != "created_at"
            }
        if isinstance(value, list):
            return [without_timestamps(item) for item in value]
        return value

    payload = without_timestamps(certificate.canonical_dict())
    return sha256_canonical_json(payload)


def validate_roundtrip(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    restored: pd.DataFrame,
    certificate: TransformationCertificate,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Validate schema, row identity, missingness, and lossless reconstruction."""

    if certificate.source_artifact_hash != hash_dataframe_logically(source.reset_index(drop=True)):
        raise ValueError("source artifact hash does not match certificate")
    if certificate.output_artifact_hash != hash_dataframe_logically(
        transformed.reset_index(drop=True)
    ):
        raise ValueError("output artifact hash does not match certificate")
    if set(source[ROW_ID_COLUMN]) != set(transformed[ROW_ID_COLUMN]):
        raise ValueError("row-id set changed during transformation")
    if set(source[ROW_ID_COLUMN]) != set(restored[ROW_ID_COLUMN]):
        raise ValueError("row-id set changed during reconstruction")
    if tuple(restored.columns) != tuple(source.columns):
        raise ValueError("reconstruction columns do not match source")
    exact = source.equals(restored)
    max_error = 0.0
    if not exact:
        for column in source.columns:
            left, right = source[column], restored[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                mask = left.notna() & right.notna()
                if mask.any():
                    difference = (left[mask].astype(float) - right[mask].astype(float)).abs()
                    max_error = max(max_error, float(difference.max()))
                if not left.isna().equals(right.isna()):
                    raise ValueError(f"missing mask changed for {column}")
                if not pd.Series(
                    (left[mask].astype(float)).to_numpy(), index=left[mask].index
                ).equals(pd.Series(right[mask].astype(float).to_numpy(), index=right[mask].index)):
                    if not (
                        (left[mask].astype(float) - right[mask].astype(float)).abs()
                        <= atol + rtol * left[mask].abs()
                    ).all():
                        raise ValueError(f"numeric reconstruction exceeded tolerance for {column}")
            else:
                left_values = left.astype("object").where(left.notna(), "__missing__").tolist()
                right_values = right.astype("object").where(right.notna(), "__missing__").tolist()
                if left_values != right_values:
                    raise ValueError(f"categorical reconstruction changed values for {column}")
    return {
        "status": "PASS",
        "source_hash": certificate.source_artifact_hash,
        "output_hash": certificate.output_artifact_hash,
        "restored_hash": hash_dataframe_logically(restored.reset_index(drop=True)),
        "reconstruction_exact": bool(exact),
        "reconstruction_max_abs_error": max_error,
        "row_id_preserved": True,
    }
