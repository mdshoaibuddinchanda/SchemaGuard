"""Certificate construction and complete invariant verification."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ..constants import ROW_ID_COLUMN
from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json
from .base import BaseTransformation
from .contracts import TransformationCertificate

VIEW_METADATA: dict[str, tuple[str, str, str]] = {
    "V00": ("identity", "BIJECTION", "CONTROL"),
    "V01": ("numeric_affine_units", "BIJECTION", "PRIMARY_MIGRATION"),
    "V02": ("numeric_asinh", "BIJECTION", "PRIMARY_MIGRATION"),
    "V03": ("category_permutation", "BIJECTION", "PRIMARY_MIGRATION"),
    "V04": ("categorical_onehot", "BIJECTION", "PRIMARY_MIGRATION"),
    "V05": ("duplicate_feature", "PROJECTION", "PRIMARY_MIGRATION"),
    "V06": ("redundant_affine_feature", "PROJECTION", "PRIMARY_MIGRATION"),
    "V07": ("integer_quotient_remainder", "BIJECTION", "PRIMARY_MIGRATION"),
    "V08": ("column_permutation_control", "PERMUTATION", "CONTROL"),
    "V09": ("row_permutation_control", "PERMUTATION", "CONTROL"),
    "V10": ("composite_migration", "COMPOSITION", "PRIMARY_MIGRATION"),
}


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
    if transformation.view_id not in VIEW_METADATA:
        raise ValueError("unknown transformation view")
    view_name, certificate_type, scientific_role = VIEW_METADATA[transformation.view_id]
    if transformation.view_name != view_name or transformation.certificate_type != certificate_type:
        raise ValueError("transformation identity does not match the frozen certificate registry")
    certificate_partition = cast(Literal["train", "calibration", "test"], partition)
    parameters = dict(transformation.parameters_for(partition))
    parameters["_configuration_hash"] = transformation.configuration_hash()
    parameters["_implementation_hash"] = transformation.implementation_hash()
    parameters["_view_id"] = transformation.view_id
    parameters["_view_name"] = transformation.view_name
    parameters["_certificate_type"] = certificate_type
    parameters["_scientific_role"] = scientific_role
    selected = parameters.get("selected_columns", [])
    generated = parameters.get("generated_columns", [])
    numerical_tolerance = {
        "rtol": float(transformation.config.get("numerical_rtol", 1.0e-10)),
        "atol": float(transformation.config.get("numerical_atol", 1.0e-12)),
    }
    return TransformationCertificate(
        schema_version=2,
        view_id=transformation.view_id,
        view_name=view_name,
        certificate_type=cast(
            Literal["BIJECTION", "PROJECTION", "PERMUTATION", "COMPOSITION"],
            certificate_type,
        ),
        scientific_role=cast(Literal["PRIMARY_MIGRATION", "CONTROL"], scientific_role),
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
        inverse_parameters={
            "operation": "reconstruct",
            "source_columns": list(source.columns),
            "source_dtypes": [str(dtype) for dtype in source.dtypes],
            "restored_schema_hash": schema_hash(source),
        },
        missing_mask_policy="preserve_missing_masks_exactly",
        dtype_policy="restore_exact_source_dtypes",
        numerical_tolerance=numerical_tolerance,
        configuration_hash=transformation.configuration_hash(),
        implementation_hash=transformation.implementation_hash(),
        source_commit=current_commit(),
        validation_status=cast(Literal["PASS", "FAIL", "N/A"], validation_status),
        validation_results=validation_results,
        not_applicable_reason=not_applicable_reason,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def _without_timestamps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timestamps(item) for key, item in value.items() if key != "created_at"
        }
    if isinstance(value, list):
        return [_without_timestamps(item) for item in value]
    return value


def certificate_hash(certificate: TransformationCertificate) -> str:
    """Return the stable identity of a validated certificate."""

    return sha256_canonical_json(_without_timestamps(certificate.canonical_dict()))


def _maximum_errors(source: pd.DataFrame, restored: pd.DataFrame) -> tuple[float, float]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for column in source.columns:
        left, right = source[column], restored[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            mask = left.notna() & right.notna()
            if mask.any():
                difference = (left[mask].astype(float) - right[mask].astype(float)).abs()
                maximum_absolute = max(maximum_absolute, float(difference.max()))
                denominator = left[mask].abs().astype(float).clip(lower=np.finfo(float).tiny)
                maximum_relative = max(maximum_relative, float((difference / denominator).max()))
    return maximum_absolute, maximum_relative


def _validate_projection_relationship(
    source: pd.DataFrame, transformed: pd.DataFrame, certificate: TransformationCertificate
) -> None:
    generated_columns = set(certificate.generated_columns)
    relationships = certificate.parameters.get("generated_relationship")
    if not isinstance(relationships, dict) or set(relationships) != generated_columns:
        raise ValueError("projection relationships must cover exactly the generated columns")
    if any(column not in transformed.columns for column in generated_columns):
        raise ValueError("projection output is missing a declared generated column")
    aligned = transformed.set_index(ROW_ID_COLUMN).loc[source[ROW_ID_COLUMN].tolist()]
    for generated, relationship in relationships.items():
        if not isinstance(relationship, dict):
            raise ValueError("projection relationship must be a mapping")
        parent = relationship["parent"]
        if parent not in source.columns:
            raise ValueError("projection relationship parent column is missing")
        parent_values = source[parent].reset_index(drop=True)
        generated_values = aligned[generated].reset_index(drop=True)
        if not parent_values.isna().equals(generated_values.isna()):
            raise ValueError("projection parent and generated missing masks differ")
        if relationship["relationship"] == "equal":
            if str(parent_values.dtype) != str(generated_values.dtype) or not parent_values.equals(
                generated_values
            ):
                raise ValueError("duplicate feature relationship is invalid")
            for parent_value, generated_value in zip(
                parent_values.tolist(), generated_values.tolist(), strict=True
            ):
                if pd.isna(parent_value):
                    continue
                if type(parent_value) is not type(generated_value):
                    raise ValueError("duplicate feature relationship changed value type")
        elif relationship["relationship"] == "3*x+7":
            valid = parent_values.notna() & generated_values.notna()
            if valid.any():
                parent_numeric = parent_values[valid].astype(float)
                generated_numeric = generated_values[valid].astype(float)
                if not np.isfinite(parent_numeric).all() or not np.isfinite(
                    generated_numeric
                ).all():
                    raise ValueError("affine projection contains non-finite values")
                expected = parent_numeric * 3.0 + 7.0
                actual = generated_numeric
                if not np.allclose(
                    expected.to_numpy(), actual.to_numpy(), rtol=0.0, atol=0.0
                ):
                    raise ValueError("redundant affine relationship is invalid")
        else:
            raise ValueError("unknown projection relationship")


def _validate_permutation(certificate: TransformationCertificate) -> None:
    forward = certificate.parameters["forward_order"]
    inverse = certificate.parameters["inverse_order"]
    if len(forward) != len(inverse) or sorted(forward) != list(range(len(forward))):
        raise ValueError("permutation forward order is not complete")
    if sorted(inverse) != list(range(len(inverse))):
        raise ValueError("permutation inverse order is not complete")
    if any(inverse[forward[index]] != index for index in range(len(forward))):
        raise ValueError("permutation inverse does not invert forward order")


def _validate_composition(certificate: TransformationCertificate) -> None:
    components = [
        TransformationCertificate.model_validate(item)
        for item in certificate.parameters["component_certificates"]
    ]
    if [item.view_id for item in components] != ["V01", "V03", "V08"]:
        raise ValueError("composition components are not V01, V03, V08 in order")
    if components[0].source_artifact_hash != certificate.source_artifact_hash:
        raise ValueError("composition source hash continuity failed")
    if components[0].output_artifact_hash != components[1].source_artifact_hash:
        raise ValueError("composition first-to-second hash continuity failed")
    if components[1].output_artifact_hash != components[2].source_artifact_hash:
        raise ValueError("composition second-to-third hash continuity failed")
    if components[2].output_artifact_hash != certificate.output_artifact_hash:
        raise ValueError("composition output hash continuity failed")


def validate_roundtrip(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    restored: pd.DataFrame,
    certificate: TransformationCertificate,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    transformation: BaseTransformation | None = None,
) -> dict[str, Any]:
    """Validate every certificate claim against the supplied artifacts."""

    TransformationCertificate.model_validate(certificate.canonical_dict())
    if transformation is None and certificate.validation_status == "PASS":
        raise ValueError(
            "configuration and implementation identity require the fitted transformation"
        )
    if certificate.view_name != certificate.parameters.get("_view_name"):
        raise ValueError("certificate reserved view name does not match the top-level view name")
    if certificate.numerical_tolerance.get("rtol") != rtol or certificate.numerical_tolerance.get(
        "atol"
    ) != atol:
        raise ValueError("certificate numerical tolerances do not match validation tolerances")
    if certificate.source_artifact_hash != hash_dataframe_logically(source.reset_index(drop=True)):
        raise ValueError("source artifact hash does not match certificate")
    if certificate.output_artifact_hash != hash_dataframe_logically(
        transformed.reset_index(drop=True)
    ):
        raise ValueError("output artifact hash does not match certificate")
    if certificate.source_schema_hash != schema_hash(source):
        raise ValueError("source schema hash does not match certificate")
    if certificate.output_schema_hash != schema_hash(transformed):
        raise ValueError("output schema hash does not match certificate")
    if certificate.source_row_id_hash != row_id_hash(source):
        raise ValueError("source row-ID hash does not match certificate")
    if certificate.output_row_id_hash != row_id_hash(transformed):
        raise ValueError("output row-ID hash does not match certificate")
    if certificate.source_target_hash is None or certificate.output_target_hash is None:
        raise ValueError("target hashes are required for certificate validation")
    if certificate.source_target_hash != certificate.output_target_hash:
        raise ValueError("target hash changed during transformation")
    if transformation is not None:
        if certificate.configuration_hash != transformation.configuration_hash():
            raise ValueError("configuration hash does not match fitted transformation")
        if certificate.implementation_hash != transformation.implementation_hash():
            raise ValueError("implementation hash does not match fitted transformation")
    if certificate.parameters.get("_configuration_hash") != certificate.configuration_hash:
        raise ValueError("certificate configuration hash self-check failed")
    if certificate.parameters.get("_implementation_hash") != certificate.implementation_hash:
        raise ValueError("certificate implementation hash self-check failed")
    if certificate.parameters.get("_view_id") != certificate.view_id:
        raise ValueError("certificate view self-check failed")
    if certificate.parameters.get("_certificate_type") != certificate.certificate_type:
        raise ValueError("certificate type self-check failed")
    if certificate.parameters.get("_scientific_role") != certificate.scientific_role:
        raise ValueError("certificate scientific role self-check failed")
    if len(source) != len(transformed) or len(source) != len(restored):
        raise ValueError("row count changed during transformation")
    if transformed[ROW_ID_COLUMN].duplicated().any():
        raise ValueError("transformation duplicated a row identifier")
    if set(source[ROW_ID_COLUMN]) != set(transformed[ROW_ID_COLUMN]):
        raise ValueError("row-id set changed during transformation")
    if set(source[ROW_ID_COLUMN]) != set(restored[ROW_ID_COLUMN]):
        raise ValueError("row-id set changed during reconstruction")
    if tuple(restored.columns) != tuple(source.columns):
        raise ValueError("reconstruction columns do not match source")
    if schema_hash(restored) != certificate.source_schema_hash:
        raise ValueError("restored schema hash does not match source schema hash")
    if (
        not source[ROW_ID_COLUMN]
        .reset_index(drop=True)
        .equals(restored[ROW_ID_COLUMN].reset_index(drop=True))
    ):
        raise ValueError("reconstruction row order does not restore source order")
    for column in source.columns:
        if not source[column].isna().equals(restored[column].isna()):
            raise ValueError(f"missing mask changed for {column}")
    exact = source.equals(restored)
    maximum_absolute, maximum_relative = _maximum_errors(source, restored)
    if not exact:
        for column in source.columns:
            left, right = source[column], restored[column]
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                mask = left.notna() & right.notna()
                if (
                    mask.any()
                    and not (
                        (left[mask].astype(float) - right[mask].astype(float)).abs()
                        <= atol + rtol * left[mask].abs()
                    ).all()
                ):
                    raise ValueError(f"numeric reconstruction exceeded tolerance for {column}")
            elif left.astype("object").tolist() != right.astype("object").tolist():
                raise ValueError(f"categorical reconstruction changed values for {column}")
    if certificate.certificate_type == "PROJECTION":
        _validate_projection_relationship(source, transformed, certificate)
    elif certificate.certificate_type == "PERMUTATION":
        _validate_permutation(certificate)
    elif certificate.certificate_type == "COMPOSITION":
        _validate_composition(certificate)
    evidence = {
        "status": "PASS",
        "row_count_preserved": len(source) == len(transformed) == len(restored),
        "row_ids_preserved": set(source[ROW_ID_COLUMN])
        == set(transformed[ROW_ID_COLUMN])
        == set(restored[ROW_ID_COLUMN]),
        "row_id_order_policy": certificate.parameters.get(
            "target_alignment_policy", "ROW_ID_SET_PRESERVED"
        ),
        "target_hash_preserved": certificate.source_target_hash == certificate.output_target_hash,
        "source_schema_hash": certificate.source_schema_hash,
        "output_schema_hash": certificate.output_schema_hash,
        "restored_schema_hash": schema_hash(restored),
        "missing_masks_preserved": True,
        "reconstruction_exact": bool(exact),
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "configuration_hash_verified": transformation is not None
        and certificate.configuration_hash == transformation.configuration_hash(),
        "implementation_hash_verified": transformation is not None
        and certificate.implementation_hash == transformation.implementation_hash(),
        "certificate_type_verified": certificate.view_id in VIEW_METADATA
        and VIEW_METADATA[certificate.view_id][1] == certificate.certificate_type,
    }
    if certificate.validation_status == "PASS" and certificate.validation_results != evidence:
        raise ValueError("recorded validation results do not match recomputed evidence")
    return evidence


__all__ = [
    "VIEW_METADATA",
    "build_certificate",
    "certificate_hash",
    "schema_hash",
    "validate_roundtrip",
]
