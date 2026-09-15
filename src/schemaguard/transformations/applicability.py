"""Controlled applicability decisions for every frozen view."""

from __future__ import annotations

from datetime import UTC, datetime

from .base import FeatureSchema
from .contracts import ApplicabilityRecord


def assess_applicability(
    view_id: str,
    view_name: str,
    dataset_id: int | str,
    seed: int,
    schema: FeatureSchema,
    *,
    category_minimum: int = 2,
) -> ApplicabilityRecord:
    numeric = bool(schema.numeric_columns)
    categorical = bool(schema.categorical_columns)
    integer = bool(schema.integer_columns)
    if view_id in {"V00", "V08", "V09"}:
        applicable, reason_code, reason = bool(schema.columns), None, None
    elif view_id in {"V01", "V02", "V06"}:
        applicable = numeric
        reason_code = None if applicable else "NO_NUMERICAL_FEATURE"
        reason = None if applicable else "No numerical predictor is available"
    elif view_id in {"V03", "V04"}:
        applicable = categorical
        reason_code = None if applicable else "NO_CATEGORICAL_FEATURE"
        reason = None if applicable else "No categorical predictor is available"
    elif view_id == "V05":
        applicable = numeric or categorical
        reason_code = None if applicable else "NO_PREDICTOR_FEATURE"
        reason = None if applicable else "No predictor is available"
    elif view_id == "V07":
        applicable = integer
        reason_code = None if applicable else "NO_INTEGER_FEATURE"
        reason = None if applicable else "No integer predictor is available"
    elif view_id == "V10":
        applicable = numeric and categorical
        reason_code = None if applicable else "COMPOSITE_REQUIREMENTS_UNMET"
        reason = (
            None
            if applicable
            else "Composite requires at least one numerical and one categorical predictor"
        )
    else:
        raise ValueError(f"unknown transformation view {view_id}")
    return ApplicabilityRecord(
        schema_version=1,
        dataset_id=dataset_id,
        seed=seed,
        view_id=view_id,
        view_name=view_name,
        status="APPLICABLE" if applicable else "NOT_APPLICABLE",
        reason_code=reason_code,
        reason=reason,
        feature_schema_hash=schema.schema_hash,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
