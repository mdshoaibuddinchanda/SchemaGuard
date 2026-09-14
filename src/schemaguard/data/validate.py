"""Blocking and non-blocking validation for raw and processed smoke data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN, TARGET_LABEL_COLUMN

from .arff_parser import ParsedArff
from .contracts import QualityCheck, QualityReport, SmokeDatasetConfig


@dataclass(frozen=True)
class ValidationResult:
    report: QualityReport
    details: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.report.final_status == "PASS"


def _check(
    name: str,
    status: Literal["passed", "failed", "warning", "not_applicable"],
    blocking: bool,
    details: dict[str, Any] | None = None,
) -> QualityCheck:
    return QualityCheck(name=name, status=status, blocking=blocking, details=details or {})


def _status(condition: bool) -> Literal["passed", "failed"]:
    return "passed" if condition else "failed"


def validate_source_table(parsed: ParsedArff, config: SmokeDatasetConfig) -> ValidationResult:
    """Validate source shape, schema, target, missingness, and numeric finiteness."""
    frame = parsed.frame
    attributes = list(parsed.attributes)
    source_columns = [item.name for item in attributes]
    target_name = config.task.expected_target_name
    feature_columns = [name for name in source_columns if name != target_name]
    checks: list[QualityCheck] = []

    duplicate_columns = len(source_columns) != len(set(source_columns))
    checks.append(
        _check(
            "unique_column_names",
            _status(not duplicate_columns),
            True,
            {"duplicates": duplicate_columns},
        )
    )
    checks.append(
        _check(
            "expected_shape",
            _status(frame.shape == (config.expected.rows, config.expected.total_columns + 1)),
            True,
            {"observed_rows": len(frame), "observed_source_columns": len(source_columns)},
        )
    )
    checks.append(
        _check(
            "expected_target",
            _status(target_name in source_columns and target_name == "Class"),
            True,
            {"observed_target": target_name if target_name in source_columns else None},
        )
    )
    target_missing = int(frame[target_name].isna().sum()) if target_name in frame else -1
    checks.append(
        _check("missing_target", _status(target_missing == 0), True, {"count": target_missing})
    )
    target_labels = []
    if target_name in frame:
        target_labels = sorted(str(value) for value in frame[target_name].dropna().unique())
    checks.append(
        _check(
            "expected_class_count",
            _status(len(target_labels) == config.expected.classes),
            True,
            {"observed": len(target_labels), "labels": target_labels},
        )
    )
    missing_features = int(frame[feature_columns].isna().sum().sum()) if feature_columns else -1
    checks.append(
        _check(
            "missing_feature_values",
            _status(missing_features == config.expected.missing_values),
            True,
            {"count": missing_features},
        )
    )
    numeric_columns = list(parsed.numeric_columns)
    infinite_values = 0
    if numeric_columns:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        infinite_values = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    checks.append(
        _check(
            "finite_numeric_values", _status(infinite_values == 0), True, {"count": infinite_values}
        )
    )
    duplicate_complete = int(frame[source_columns].duplicated(keep=False).sum())
    duplicate_predictors = (
        int(frame[feature_columns].duplicated(keep=False).sum()) if feature_columns else 0
    )
    checks.append(
        _check(
            "duplicate_complete_rows",
            "warning" if duplicate_complete else "passed",
            False,
            {"rows_in_duplicate_groups": duplicate_complete},
        )
    )
    checks.append(
        _check(
            "duplicate_predictor_rows",
            "warning" if duplicate_predictors else "passed",
            False,
            {"rows_in_duplicate_groups": duplicate_predictors},
        )
    )
    checks.append(
        _check(
            "source_rows_preserved",
            _status(len(parsed.source_row_positions) == len(frame)),
            True,
            {"source_positions": len(parsed.source_row_positions), "rows": len(frame)},
        )
    )

    blocking_failures = [
        check.name for check in checks if check.blocking and check.status == "failed"
    ]
    warnings = [check.name for check in checks if check.status == "warning"]
    report = QualityReport(
        internal_dataset_id=config.dataset.internal_id,
        checks=checks,
        final_status="FAILED" if blocking_failures else "PASS",
        warnings=warnings,
    )
    return ValidationResult(
        report=report,
        details={
            "source_columns": source_columns,
            "feature_columns": feature_columns,
            "target_labels": target_labels,
            "target_class_frequencies": {
                str(label): int(count)
                for label, count in frame[target_name].value_counts(dropna=False).items()
            }
            if target_name in frame
            else {},
            "numeric_columns": numeric_columns,
            "categorical_columns": list(parsed.categorical_columns),
            "missing_values": int(frame[source_columns].isna().sum().sum()),
            "duplicate_complete_rows": duplicate_complete,
            "duplicate_predictor_rows": duplicate_predictors,
        },
    )


def validate_processed_tables(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    parsed: ParsedArff,
    config: SmokeDatasetConfig,
    label_mapping: dict[str, int],
) -> ValidationResult:
    """Validate row-ID alignment, conservative values, and target mapping."""
    source_columns = [item.name for item in parsed.attributes]
    feature_columns = [name for name in source_columns if name != config.task.expected_target_name]
    feature_ids = features[ROW_ID_COLUMN].tolist() if ROW_ID_COLUMN in features else []
    target_ids = targets[ROW_ID_COLUMN].tolist() if ROW_ID_COLUMN in targets else []
    checks = [
        _check("unique_row_ids", _status(len(feature_ids) == len(set(feature_ids))), True),
        _check("feature_target_row_id_sets", _status(set(feature_ids) == set(target_ids)), True),
        _check("feature_target_row_id_order", _status(feature_ids == target_ids), True),
        _check(
            "row_ids_not_null",
            _status(all(value is not None and not pd.isna(value) for value in feature_ids)),
            True,
        ),
        _check("all_source_rows_preserved", _status(len(features) == len(parsed.frame)), True),
        _check(
            "feature_columns_preserved",
            _status(list(features.columns) == [ROW_ID_COLUMN, *feature_columns]),
            True,
        ),
        _check(
            "target_columns_present",
            _status(
                list(targets.columns) == [ROW_ID_COLUMN, TARGET_LABEL_COLUMN, TARGET_CODE_COLUMN]
            ),
            True,
        ),
        _check(
            "feature_values_not_missing",
            _status(int(features[feature_columns].isna().sum().sum()) == 0),
            True,
        ),
    ]
    label_values = targets[TARGET_LABEL_COLUMN].astype(str).tolist()
    expected_codes = [label_mapping[label] for label in label_values]
    observed_codes = targets[TARGET_CODE_COLUMN].astype(int).tolist()
    checks.append(
        _check("target_codes_match_mapping", _status(expected_codes == observed_codes), True)
    )
    checks.append(
        _check(
            "target_labels_preserved",
            _status(
                label_values == parsed.frame[config.task.expected_target_name].astype(str).tolist()
            ),
            True,
        )
    )

    blocking_failures = [
        check.name for check in checks if check.blocking and check.status == "failed"
    ]
    report = QualityReport(
        internal_dataset_id=config.dataset.internal_id,
        checks=checks,
        final_status="FAILED" if blocking_failures else "PASS",
    )
    return ValidationResult(
        report=report,
        details={"blocking_failures": blocking_failures},
    )
