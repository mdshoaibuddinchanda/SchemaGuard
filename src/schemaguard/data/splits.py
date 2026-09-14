"""Deterministic predictor-group-aware stratified split generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from schemaguard.constants import (
    GROUP_ID_COLUMN,
    ROW_ID_COLUMN,
    TARGET_CODE_COLUMN,
    TARGET_LABEL_COLUMN,
)
from schemaguard.utils.hashing import sha256_canonical_json, sha256_file
from schemaguard.utils.io import atomic_write_json, atomic_write_parquet, read_json_validated
from schemaguard.utils.seeds import derive_component_seed

from .contracts import SmokeDatasetConfig, SplitManifest


class SplitError(ValueError):
    """Raised when deterministic split generation or validation fails."""


@dataclass(frozen=True)
class SplitArtifacts:
    assignments: pd.DataFrame
    group_ids: pd.Series
    assignments_path: Path
    manifest_path: Path
    manifest: SplitManifest


def _canonical_scalar(value: Any, dtype: str) -> dict[str, Any]:
    """Serialize a scalar with its exact value type and pandas dtype."""
    if value is None or value is pd.NA:
        return {"dtype": dtype, "type": "missing", "value": None}
    try:
        if bool(pd.isna(value)):
            return {"dtype": dtype, "type": "missing", "value": None}
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return {"dtype": dtype, "type": "bool", "value": bool(value)}
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return {"dtype": dtype, "type": "integer", "value": str(int(value))}
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return {"dtype": dtype, "type": "missing", "value": None}
        if np.isposinf(number):
            return {"dtype": dtype, "type": "float", "value": "+inf"}
        if np.isneginf(number):
            return {"dtype": dtype, "type": "float", "value": "-inf"}
        return {"dtype": dtype, "type": "float", "value": number.hex()}
    if isinstance(value, pd.Timestamp):
        return {"dtype": dtype, "type": "timestamp", "value": value.isoformat()}
    if isinstance(value, bytes):
        return {"dtype": dtype, "type": "bytes", "value": value.hex()}
    if isinstance(value, str):
        return {"dtype": dtype, "type": "string", "value": value}
    return {
        "dtype": dtype,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": repr(value),
    }


def canonical_predictor_vector(
    row: Mapping[str, Any], predictor_columns: Sequence[str], dtypes: Mapping[str, str]
) -> str:
    """Return a SHA-256 group ID for one predictor vector."""
    payload = {
        "columns": [
            {
                "name": column,
                "value": _canonical_scalar(row[column], dtypes[column]),
            }
            for column in predictor_columns
        ]
    }
    return sha256_canonical_json(payload)


def predictor_group_ids(features: pd.DataFrame) -> pd.Series:
    """Return stable SHA-256 IDs for unique predictor vectors.

    Row IDs and target columns are excluded. The returned Series preserves the
    input index, while each identical predictor vector receives the same ID.
    """
    excluded = {ROW_ID_COLUMN, GROUP_ID_COLUMN, TARGET_LABEL_COLUMN, TARGET_CODE_COLUMN}
    predictor_columns = [str(column) for column in features.columns if str(column) not in excluded]
    if not predictor_columns:
        raise SplitError("At least one predictor column is required to derive groups")
    dtypes = {column: str(features[column].dtype) for column in predictor_columns}
    values = [
        canonical_predictor_vector(row, predictor_columns, dtypes)
        for row in features[predictor_columns].to_dict(orient="records")
    ]
    return pd.Series(values, index=features.index, name=GROUP_ID_COLUMN, dtype="string")


def _class_counts(
    assignments: pd.DataFrame, target_codes: pd.DataFrame
) -> dict[str, dict[str, int]]:
    joined = assignments.merge(
        target_codes[[ROW_ID_COLUMN, TARGET_CODE_COLUMN]],
        on=ROW_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    table = pd.crosstab(joined["split"], joined[TARGET_CODE_COLUMN])
    return {
        split: {str(code): int(table.loc[split, code]) for code in table.columns}
        if split in table.index
        else {}
        for split in ("train", "calibration", "test")
    }


def _proportion_deviations(
    counts: dict[str, dict[str, int]], row_counts: dict[str, int], total_counts: Mapping[Any, int]
) -> dict[str, dict[str, float]]:
    total_rows = sum(row_counts.values())
    overall = {
        str(code): count / total_rows
        for code, count in sorted(total_counts.items(), key=lambda x: x[0])
    }
    return {
        split: {
            code: abs(counts[split].get(code, 0) / row_counts[split] - frequency)
            for code, frequency in overall.items()
        }
        for split in ("train", "calibration", "test")
    }


def _group_statistics(
    group_ids: pd.Series, target_codes: pd.Series, splits: pd.Series | None = None
) -> dict[str, Any]:
    table = pd.DataFrame(
        {
            GROUP_ID_COLUMN: group_ids.astype("string").tolist(),
            TARGET_CODE_COLUMN: target_codes.tolist(),
        }
    )
    sizes = table.groupby(GROUP_ID_COLUMN, sort=True, dropna=False).size()
    conflicting = (
        table.groupby(GROUP_ID_COLUMN, sort=True, dropna=False)[TARGET_CODE_COLUMN]
        .nunique(dropna=False)
        .gt(1)
        .sum()
    )
    crossing = 0
    if splits is not None:
        table["split"] = splits.tolist()
        crossing = int(
            table.groupby(GROUP_ID_COLUMN, sort=True, dropna=False)["split"].nunique().gt(1).sum()
        )
    return {
        "total_predictor_groups": int(sizes.size),
        "duplicate_predictor_groups": int((sizes > 1).sum()),
        "largest_group_size": int(sizes.max()),
        "conflicting_target_groups": int(conflicting),
        "predictor_duplicate_groups_crossing_splits": crossing,
    }


def _select_fold_assignment(
    fold_by_row: dict[str, int],
    row_ids: list[str],
    target_codes: pd.Series,
    config: SmokeDatasetConfig,
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, float], dict[str, dict[str, float]]]:
    """Select the best 3-fold train, 1-fold calibration, 1-fold test mapping."""
    fold_numbers = tuple(range(config.split.group_folds))
    target_counts = {
        int(code): int(count) for code, count in target_codes.value_counts().sort_index().items()
    }
    target_fractions = {
        "train": config.split.train_fraction,
        "calibration": config.split.calibration_fraction,
        "test": config.split.test_fraction,
    }
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for calibration_fold in fold_numbers:
        for test_fold in fold_numbers:
            if calibration_fold == test_fold:
                continue
            train_folds = tuple(
                fold for fold in fold_numbers if fold not in {calibration_fold, test_fold}
            )
            labels_by_fold = {
                fold: "calibration"
                if fold == calibration_fold
                else "test"
                if fold == test_fold
                else "train"
                for fold in fold_numbers
            }
            labels = {row_id: labels_by_fold[fold_by_row[row_id]] for row_id in row_ids}
            assignments = pd.Series([labels[row_id] for row_id in row_ids])
            row_counts = {
                split: int((assignments == split).sum())
                for split in ("train", "calibration", "test")
            }
            if any(row_counts[split] == 0 for split in row_counts):
                continue
            counts = _class_counts(
                pd.DataFrame({ROW_ID_COLUMN: row_ids, "split": assignments}),
                pd.DataFrame({ROW_ID_COLUMN: row_ids, TARGET_CODE_COLUMN: target_codes.tolist()}),
            )
            if any(len(counts[split]) < 2 for split in counts):
                continue
            size_deviations = {
                split: abs(row_counts[split] / len(row_ids) - target_fractions[split])
                for split in row_counts
            }
            class_deviations = _proportion_deviations(counts, row_counts, target_counts)
            size_score = float(sum(size_deviations.values()))
            class_score = float(sum(sum(values.values()) for values in class_deviations.values()))
            score = (size_score, class_score, train_folds, calibration_fold, test_fold)
            candidate = {
                "labels": labels,
                "fold_assignment": {
                    "train": list(train_folds),
                    "calibration": [calibration_fold],
                    "test": [test_fold],
                },
                "size_deviations": size_deviations,
                "class_deviations": class_deviations,
                "selection_score": {
                    "size_deviation_sum": size_score,
                    "class_proportion_deviation_sum": class_score,
                },
            }
            if best is None or score < best[0]:
                best = (score, candidate)
    if best is None:
        raise SplitError("No valid grouped fold assignment contains both classes in every split")
    candidate = best[1]
    return (
        cast(dict[str, str], candidate["labels"]),
        cast(dict[str, list[int]], candidate["fold_assignment"]),
        cast(dict[str, float], candidate["size_deviations"]),
        cast(dict[str, dict[str, float]], candidate["class_deviations"]),
    )


def validate_split_assignments(
    assignments: pd.DataFrame,
    target_codes: pd.DataFrame,
    config: SmokeDatasetConfig,
    group_ids: pd.Series | None = None,
) -> dict[str, Any]:
    """Validate complete, disjoint, grouped, stratified assignments."""
    if list(assignments.columns) != [ROW_ID_COLUMN, "split"]:
        raise SplitError("Assignment columns must be __sg_row_id and split")
    if len(assignments) != len(target_codes):
        raise SplitError("Assignment count differs from the processed target count")
    if assignments[ROW_ID_COLUMN].duplicated().any():
        raise SplitError("A row ID appears more than once in split assignments")
    if set(assignments[ROW_ID_COLUMN]) != set(target_codes[ROW_ID_COLUMN]):
        raise SplitError("Split assignment row IDs do not equal the complete target row-ID set")
    if not set(assignments["split"]).issubset({"train", "calibration", "test"}):
        raise SplitError("Split assignments contain an unknown split value")
    if config.split.strategy != "stratified_group_5fold_v1":
        raise SplitError(f"Unsupported split strategy: {config.split.strategy}")
    if group_ids is None:
        raise SplitError("Predictor group IDs are required for grouped split validation")
    if group_ids.index.has_duplicates:
        raise SplitError("Predictor group IDs have duplicate row-ID index values")
    ordered_group_ids = group_ids.reindex(assignments[ROW_ID_COLUMN].tolist())
    if ordered_group_ids.isna().any():
        raise SplitError("Predictor group IDs do not cover every assigned row")

    observed_sizes = {
        split: int((assignments["split"] == split).sum())
        for split in ("train", "calibration", "test")
    }
    counts = _class_counts(assignments, target_codes)
    if any(len(counts[split]) < 2 for split in counts):
        raise SplitError("Both target classes must occur in every split")
    target_counts = {
        int(code): int(count)
        for code, count in target_codes[TARGET_CODE_COLUMN].value_counts().items()
    }
    class_deviations = _proportion_deviations(counts, observed_sizes, target_counts)
    group_frame = pd.DataFrame(
        {
            ROW_ID_COLUMN: assignments[ROW_ID_COLUMN].tolist(),
            "split": assignments["split"].tolist(),
            GROUP_ID_COLUMN: ordered_group_ids.tolist(),
        }
    )
    group_frame = group_frame.merge(
        target_codes[[ROW_ID_COLUMN, TARGET_CODE_COLUMN]],
        on=ROW_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    group_statistics = _group_statistics(
        group_frame[GROUP_ID_COLUMN],
        group_frame[TARGET_CODE_COLUMN],
        group_frame["split"],
    )
    if group_statistics["predictor_duplicate_groups_crossing_splits"] != 0:
        raise SplitError("Predictor duplicate groups cross split boundaries")
    return {
        "row_count": len(assignments),
        "row_ids_unique": True,
        "row_id_union_complete": True,
        "split_sizes": observed_sizes,
        "class_counts_by_split": counts,
        "size_deviations": {
            split: abs(
                observed_sizes[split] / len(assignments)
                - getattr(config.split, f"{split}_fraction")
            )
            for split in observed_sizes
        },
        "class_proportion_deviations": class_deviations,
        "splits_disjoint": True,
        "both_classes_in_every_split": True,
        "group_statistics": group_statistics,
    }


def generate_splits(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    config: SmokeDatasetConfig,
    output_directory: str | Path,
    data_manifest_hash: str,
) -> SplitArtifacts:
    """Generate or reuse the frozen predictor-group split assignment artifact."""
    if list(targets.columns) != [ROW_ID_COLUMN, TARGET_LABEL_COLUMN, TARGET_CODE_COLUMN]:
        raise SplitError("Targets do not have the expected processed columns")
    if config.split.strategy != "stratified_group_5fold_v1":
        raise SplitError(f"Unsupported split strategy: {config.split.strategy}")
    if config.split.group_by != "predictors" or config.split.group_folds != 5:
        raise SplitError("The active grouped strategy requires predictor groups and five folds")

    targets_sorted = (
        targets[[ROW_ID_COLUMN, TARGET_LABEL_COLUMN, TARGET_CODE_COLUMN]]
        .sort_values(ROW_ID_COLUMN)
        .reset_index(drop=True)
    )
    if targets_sorted[ROW_ID_COLUMN].duplicated().any():
        raise SplitError("Target row IDs must be unique")
    features_by_id = features.set_index(ROW_ID_COLUMN, drop=False)
    if features_by_id.index.has_duplicates:
        raise SplitError("Feature row IDs must be unique")
    row_ids = targets_sorted[ROW_ID_COLUMN].tolist()
    if set(features_by_id.index) != set(row_ids):
        raise SplitError("Features and targets do not describe the same rows")
    ordered_features = features_by_id.loc[row_ids].reset_index(drop=True)
    ordered_group_series = predictor_group_ids(ordered_features)
    ordered_group_values = ordered_group_series.tolist()
    group_ids_by_row = pd.Series(ordered_group_values, index=row_ids, name=GROUP_ID_COLUMN)
    target_series = targets_sorted[TARGET_CODE_COLUMN].reset_index(drop=True)
    group_statistics = _group_statistics(ordered_group_series, target_series)

    fold_seed = derive_component_seed(config.split.master_seed, config.split.strategy)
    splitter = StratifiedGroupKFold(
        n_splits=config.split.group_folds,
        shuffle=True,
        random_state=fold_seed,
    )
    predictors = ordered_features.drop(
        columns=[
            column
            for column in (ROW_ID_COLUMN, TARGET_LABEL_COLUMN, TARGET_CODE_COLUMN)
            if column in ordered_features
        ]
    )
    fold_by_row: dict[str, int] = {}
    for fold, (_train_indices, held_out_indices) in enumerate(
        splitter.split(predictors, target_series, ordered_group_values)
    ):
        for index in held_out_indices:
            row_id = row_ids[int(index)]
            if row_id in fold_by_row:
                raise SplitError("A row was assigned to multiple grouped folds")
            fold_by_row[row_id] = fold
    if set(fold_by_row) != set(row_ids):
        raise SplitError("Grouped folds do not cover every row")

    labels, fold_assignment, size_deviations, class_deviations = _select_fold_assignment(
        fold_by_row, row_ids, target_series, config
    )
    assignments = pd.DataFrame(
        {ROW_ID_COLUMN: row_ids, "split": [labels[row_id] for row_id in row_ids]}
    )
    validation = validate_split_assignments(assignments, targets_sorted, config, group_ids_by_row)
    actual_group_statistics = validation["group_statistics"]
    if actual_group_statistics["predictor_duplicate_groups_crossing_splits"] != 0:
        raise SplitError("Selected grouped assignment crosses predictor groups")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    assignments_path = output / "assignments.parquet"
    manifest_path = output / "split_manifest.json"
    if not assignments_path.exists():
        atomic_write_parquet(assignments_path, assignments, compression="zstd", compression_level=3)
    else:
        existing = pd.read_parquet(assignments_path)
        if not existing.equals(assignments):
            raise SplitError(
                "Existing grouped split assignments differ from the deterministic assignment"
            )

    assignment_hash = sha256_file(assignments_path)
    existing_manifest: SplitManifest | None = None
    if manifest_path.exists():
        try:
            existing_manifest = read_json_validated(manifest_path, SplitManifest)
        except Exception:
            existing_manifest = None
    selection_score = {
        "size_deviation_sum": float(sum(size_deviations.values())),
        "class_proportion_deviation_sum": float(
            sum(sum(values.values()) for values in class_deviations.values())
        ),
    }
    manifest = SplitManifest(
        internal_dataset_id=config.dataset.internal_id,
        data_manifest_hash=data_manifest_hash,
        master_seed=config.split.master_seed,
        derived_seeds={"stratified_group_5fold": fold_seed},
        split_fractions={
            "train": config.split.train_fraction,
            "calibration": config.split.calibration_fraction,
            "test": config.split.test_fraction,
        },
        split_algorithm=(
            "sklearn.StratifiedGroupKFold:n_splits=5:three_folds_train_one_calibration_one_test"
        ),
        sklearn_version=sklearn.__version__,
        row_counts=validation["split_sizes"],
        class_counts_by_split=validation["class_counts_by_split"],
        strategy=config.split.strategy,
        group_by=config.split.group_by,
        group_folds=config.split.group_folds,
        total_predictor_groups=group_statistics["total_predictor_groups"],
        duplicate_predictor_groups=group_statistics["duplicate_predictor_groups"],
        largest_group_size=group_statistics["largest_group_size"],
        conflicting_target_groups=group_statistics["conflicting_target_groups"],
        predictor_duplicate_groups_crossing_splits=actual_group_statistics[
            "predictor_duplicate_groups_crossing_splits"
        ],
        size_deviations=validation["size_deviations"],
        class_proportion_deviations=validation["class_proportion_deviations"],
        fold_assignment=fold_assignment,
        selection_score=selection_score,
        assignment_file_sha256=assignment_hash,
        created_at_utc=(
            existing_manifest.created_at_utc if existing_manifest else datetime.now(UTC)
        ),
        validation_status="PASS",
    )
    if existing_manifest is None or existing_manifest.canonical_dict() != manifest.canonical_dict():
        atomic_write_json(manifest_path, manifest.canonical_dict())
    return SplitArtifacts(
        assignments=assignments,
        group_ids=group_ids_by_row,
        assignments_path=assignments_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
