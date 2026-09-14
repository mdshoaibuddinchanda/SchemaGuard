"""Deterministic stratified train/calibration/test assignment generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.utils.hashing import sha256_file
from schemaguard.utils.io import atomic_write_json, atomic_write_parquet, read_json_validated
from schemaguard.utils.seeds import derive_component_seed

from .contracts import SmokeDatasetConfig, SplitManifest


class SplitError(ValueError):
    """Raised when deterministic split generation or validation fails."""


@dataclass(frozen=True)
class SplitArtifacts:
    assignments: pd.DataFrame
    assignments_path: Path
    manifest_path: Path
    manifest: SplitManifest


def _expected_sizes(row_count: int, config: SmokeDatasetConfig) -> dict[str, int]:
    train = int(row_count * config.split.train_fraction)
    temporary = row_count - train
    calibration_ratio = config.split.calibration_fraction / (
        config.split.calibration_fraction + config.split.test_fraction
    )
    calibration = int(round(temporary * calibration_ratio))
    test = temporary - calibration
    return {"train": train, "calibration": calibration, "test": test}


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


def validate_split_assignments(
    assignments: pd.DataFrame,
    target_codes: pd.DataFrame,
    config: SmokeDatasetConfig,
) -> dict[str, Any]:
    """Validate complete, disjoint, stratified assignments and return audit details."""
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

    observed_sizes = assignments["split"].value_counts().to_dict()
    expected_sizes = _expected_sizes(len(target_codes), config)
    if observed_sizes != expected_sizes:
        raise SplitError(
            f"Split sizes {observed_sizes} do not match expected sizes {expected_sizes}"
        )
    counts = _class_counts(assignments, target_codes)
    if any(len(counts[split]) < 2 for split in counts):
        raise SplitError("Both target classes must occur in every split")

    overall = target_codes[TARGET_CODE_COLUMN].value_counts(normalize=True).to_dict()
    proportion_deltas: dict[str, dict[str, float]] = {}
    for split in ("train", "calibration", "test"):
        subset = assignments.loc[assignments["split"] == split, ROW_ID_COLUMN]
        split_codes = target_codes[target_codes[ROW_ID_COLUMN].isin(subset)][TARGET_CODE_COLUMN]
        frequencies = split_codes.value_counts(normalize=True).to_dict()
        proportion_deltas[split] = {
            str(code): abs(float(frequencies.get(code, 0.0)) - float(overall.get(code, 0.0)))
            for code in overall
        }
    if any(delta > 0.20 for values in proportion_deltas.values() for delta in values.values()):
        raise SplitError("Class proportions are not reasonably preserved")

    return {
        "row_count": len(assignments),
        "row_ids_unique": True,
        "row_id_union_complete": True,
        "split_sizes": expected_sizes,
        "class_counts_by_split": counts,
        "class_proportion_deltas": proportion_deltas,
        "splits_disjoint": True,
        "both_classes_in_every_split": True,
    }


def generate_splits(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    config: SmokeDatasetConfig,
    output_directory: str | Path,
    data_manifest_hash: str,
) -> SplitArtifacts:
    """Generate or reuse the frozen split assignment artifact."""
    if list(targets.columns) != [ROW_ID_COLUMN, "target_label", TARGET_CODE_COLUMN]:
        raise SplitError("Targets do not have the expected processed columns")
    joined = (
        targets[[ROW_ID_COLUMN, TARGET_CODE_COLUMN]]
        .sort_values(ROW_ID_COLUMN)
        .reset_index(drop=True)
    )
    if set(features[ROW_ID_COLUMN]) != set(joined[ROW_ID_COLUMN]):
        raise SplitError("Features and targets do not describe the same rows")
    train_seed = derive_component_seed(config.split.master_seed, "train_temp_split")
    calibration_seed = derive_component_seed(config.split.master_seed, "calibration_test_split")
    row_ids = joined[ROW_ID_COLUMN].tolist()
    codes = joined[TARGET_CODE_COLUMN].tolist()
    train_ids, temporary_ids, _train_codes, temporary_codes = train_test_split(
        row_ids,
        codes,
        test_size=config.split.calibration_fraction + config.split.test_fraction,
        random_state=train_seed,
        stratify=codes,
    )
    temporary_frame = pd.DataFrame(
        {ROW_ID_COLUMN: temporary_ids, TARGET_CODE_COLUMN: temporary_codes}
    )
    temporary_frame = temporary_frame.sort_values(ROW_ID_COLUMN).reset_index(drop=True)
    calibration_ids, test_ids = train_test_split(
        temporary_frame[ROW_ID_COLUMN].tolist(),
        test_size=0.5,
        random_state=calibration_seed,
        stratify=temporary_frame[TARGET_CODE_COLUMN].tolist(),
    )
    labels = {
        **{row_id: "train" for row_id in train_ids},
        **{row_id: "calibration" for row_id in calibration_ids},
        **{row_id: "test" for row_id in test_ids},
    }
    assignments = (
        pd.DataFrame({ROW_ID_COLUMN: row_ids, "split": [labels[row_id] for row_id in row_ids]})
        .sort_values(ROW_ID_COLUMN)
        .reset_index(drop=True)
    )
    validation = validate_split_assignments(assignments, joined, config)

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
                "Existing split assignments differ from the deterministic frozen assignment"
            )

    assignment_hash = sha256_file(assignments_path)
    existing_manifest: SplitManifest | None = None
    if manifest_path.exists():
        try:
            existing_manifest = read_json_validated(manifest_path, SplitManifest)
        except Exception:
            existing_manifest = None
    manifest = SplitManifest(
        internal_dataset_id=config.dataset.internal_id,
        data_manifest_hash=data_manifest_hash,
        master_seed=config.split.master_seed,
        derived_seeds={
            "train_temp_split": train_seed,
            "calibration_test_split": calibration_seed,
        },
        split_fractions={
            "train": config.split.train_fraction,
            "calibration": config.split.calibration_fraction,
            "test": config.split.test_fraction,
        },
        split_algorithm="sklearn.train_test_split:sorted_row_ids:two_stage_stratified",
        sklearn_version=sklearn.__version__,
        row_counts=validation["split_sizes"],
        class_counts_by_split=validation["class_counts_by_split"],
        assignment_file_sha256=assignment_hash,
        created_at_utc=(
            existing_manifest.created_at_utc if existing_manifest else datetime.now(UTC)
        ),
        validation_status="PASS",
    )
    if existing_manifest is None or existing_manifest.canonical_dict() != manifest.canonical_dict():
        atomic_write_json(manifest_path, manifest.canonical_dict())
    return SplitArtifacts(assignments, assignments_path, manifest_path, manifest)
