"""Independent validation for split artifacts and the 14-by-5 inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.data.schemaorbit import SchemaOrbitError
from schemaguard.utils.hashing import sha256_file

from .caching import cache_identity, cache_key, logical_assignment_hash
from .contracts import SplitGenerationConfig, SplitManifestContract
from .generation import (
    PROTECTED_ASSIGNMENT_SHA256,
    PROTECTED_DATASET_SEED,
    PROTECTED_LOGICAL_ASSIGNMENT_SHA256,
    _read_local_inputs,
    split_directory,
)
from .grouping import GROUPING_IMPLEMENTATION_HASH, grouped_features
from .implementation import SPLIT_IMPLEMENTATION_HASH
from .selection import select_fold_assignment


class SplitValidationError(ValueError):
    """A generated or protected split failed an independent invariant."""


def _reject_temporary_artifacts(root: Path) -> None:
    split_root = root / "data" / "splits"
    if not split_root.is_dir():
        return
    temporary = [
        path
        for path in split_root.rglob("*")
        if path.name.startswith(".seed_")
        or path.name.endswith((".part", ".partial", ".tmp"))
        or path.name.startswith("split-incomplete-")
    ]
    if temporary:
        rendered = ", ".join(str(path.relative_to(root)) for path in temporary)
        raise SplitValidationError(f"temporary or partial split artifacts found: {rendered}")


def _class_counts(frame: pd.DataFrame, targets: pd.DataFrame) -> dict[str, dict[str, int]]:
    joined = frame[[ROW_ID_COLUMN, "partition"]].merge(
        targets[[ROW_ID_COLUMN, TARGET_CODE_COLUMN]],
        on=ROW_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    result: dict[str, dict[str, int]] = {}
    for partition in ("train", "calibration", "test"):
        values = joined.loc[joined["partition"] == partition, TARGET_CODE_COLUMN]
        result[partition] = {
            str(code): int(count) for code, count in values.value_counts().sort_index().items()
        }
    return result


def _validate_common_inputs(
    root: Path, dataset_id: int
) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    try:
        return _read_local_inputs(root, dataset_id)
    except (OSError, KeyError, ValueError, SchemaOrbitError) as exc:
        raise SplitValidationError(str(exc)) from exc


def _validate_protected(
    root: Path, config: SplitGenerationConfig, directory: Path
) -> dict[str, Any]:
    manifest_path = directory / "split_manifest.json"
    assignment_path = directory / "assignments.parquet"
    if sha256_file(assignment_path) != PROTECTED_ASSIGNMENT_SHA256:
        raise SplitValidationError("protected Phase 01 assignment hash changed")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("validation_status") != "PASS" or payload.get("strategy") != config.strategy:
        raise SplitValidationError("protected Phase 01 manifest is not a passing grouped artifact")
    schema_config, spec, features, targets, _quality, data_manifest = _validate_common_inputs(
        root, 1464
    )
    del schema_config
    grouped = grouped_features(features)
    try:
        assignments = pd.read_parquet(assignment_path)
    except Exception as exc:
        raise SplitValidationError("assignment Parquet cannot be read") from exc
    if assignments.columns.tolist() != [ROW_ID_COLUMN, "split"]:
        raise SplitValidationError("protected Phase 01 assignment columns changed")
    frame = assignments.rename(columns={"split": "partition"})
    if (
        set(frame[ROW_ID_COLUMN]) != set(targets[ROW_ID_COLUMN])
        or frame[ROW_ID_COLUMN].duplicated().any()
    ):
        raise SplitValidationError("protected assignment row coverage is invalid")
    by_row = grouped.set_index(ROW_ID_COLUMN).loc[frame[ROW_ID_COLUMN].tolist()]
    partition_groups = pd.DataFrame(
        {"group": by_row["predictor_group_id"].tolist(), "partition": frame["partition"].tolist()}
    )
    crossing = int(partition_groups.groupby("group")["partition"].nunique().gt(1).sum())
    if crossing != 0:
        raise SplitValidationError("protected predictor groups cross partitions")
    observed_counts = {
        part: int((frame["partition"] == part).sum()) for part in ("train", "calibration", "test")
    }
    observed_class_counts = _class_counts(frame, targets)
    if observed_counts != payload.get("row_counts") or observed_class_counts != payload.get(
        "class_counts_by_split"
    ):
        raise SplitValidationError("protected split counts differ from accepted manifest")
    expected_data_manifest = root / "data" / "processed" / "openml" / "1464" / "data_manifest.json"
    if payload.get("data_manifest_hash") != sha256_file(expected_data_manifest):
        raise SplitValidationError("protected split source manifest identity changed")
    if payload.get("internal_dataset_id") != spec.name or payload.get("master_seed") != 1729:
        raise SplitValidationError("protected split identity changed")
    observed_logical_hash = logical_assignment_hash(
        [
            {"__sg_row_id": row_id, "predictor_group_id": group_id, "partition": partition}
            for row_id, group_id, partition in zip(
                frame[ROW_ID_COLUMN], by_row["predictor_group_id"], frame["partition"], strict=True
            )
        ]
    )
    if observed_logical_hash != PROTECTED_LOGICAL_ASSIGNMENT_SHA256:
        raise SplitValidationError("protected logical assignment hash changed")
    return {
        "status": "PASS",
        "dataset_id": 1464,
        "seed": 1729,
        "protected_baseline": True,
        "row_count": len(frame),
        "partition_counts": observed_counts,
        "partition_class_counts": observed_class_counts,
        "cross_partition_group_count": crossing,
        "assignment_artifact_hash": PROTECTED_ASSIGNMENT_SHA256,
        "logical_assignment_hash": observed_logical_hash,
    }


def validate_split_directory(
    root: str | Path,
    config: SplitGenerationConfig,
    dataset_id: int,
    seed: int,
    directory: str | Path,
) -> dict[str, Any]:
    """Validate an artifact directory before it is promoted or reused."""

    project_root = Path(root)
    split_path = Path(directory)
    assignment_path = split_path / "assignments.parquet"
    manifest_path = split_path / "split_manifest.json"
    if not assignment_path.is_file() or not manifest_path.is_file():
        raise SplitValidationError(f"split artifact set is incomplete for {dataset_id}/{seed}")
    if (dataset_id, seed) == PROTECTED_DATASET_SEED:
        return _validate_protected(project_root, config, split_path)
    try:
        manifest = SplitManifestContract.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SplitValidationError(
            f"split manifest contract failed for {dataset_id}/{seed}"
        ) from exc
    if manifest.validation_status != "PASS":
        raise SplitValidationError(f"manifest is not passing: {manifest.validation_status}")
    if manifest.dataset_id != dataset_id or manifest.seed != seed:
        raise SplitValidationError("split manifest dataset/seed identity mismatch")
    if manifest.configuration_hash != config.configuration_hash:
        raise SplitValidationError("split manifest configuration hash mismatch")
    if manifest.grouping_implementation_hash != GROUPING_IMPLEMENTATION_HASH:
        raise SplitValidationError("split grouping implementation hash mismatch")
    if manifest.split_implementation_hash != SPLIT_IMPLEMENTATION_HASH:
        raise SplitValidationError("split implementation hash mismatch")
    _, spec, features, targets, _quality, data_manifest = _validate_common_inputs(
        project_root, dataset_id
    )
    grouping_input = features.drop(columns=list(spec.ignore_attributes), errors="ignore")
    grouped = grouped_features(grouping_input)
    try:
        assignments = pd.read_parquet(assignment_path)
    except Exception as exc:
        raise SplitValidationError("assignment Parquet cannot be read") from exc
    expected_columns = [ROW_ID_COLUMN, "predictor_group_id", "fold_id", "partition"]
    if assignments.columns.tolist() != expected_columns:
        raise SplitValidationError("assignment columns are not the required split metadata")
    if len(assignments) != len(targets) or assignments[ROW_ID_COLUMN].duplicated().any():
        raise SplitValidationError("assignment row count or uniqueness is invalid")
    if set(assignments[ROW_ID_COLUMN]) != set(targets[ROW_ID_COLUMN]):
        raise SplitValidationError("assignments do not cover the complete target row-ID set")
    if not assignments["partition"].isin(["train", "calibration", "test"]).all():
        raise SplitValidationError("unknown partition value")
    expected_groups = (
        grouped.set_index(ROW_ID_COLUMN)
        .loc[assignments[ROW_ID_COLUMN].tolist()]["predictor_group_id"]
        .astype(str)
        .tolist()
    )
    if assignments["predictor_group_id"].astype(str).tolist() != expected_groups:
        raise SplitValidationError("assignment predictor groups differ from recomputed groups")
    if assignments["fold_id"].isna().any() or not pd.api.types.is_integer_dtype(
        assignments["fold_id"]
    ):
        raise SplitValidationError("fold IDs are missing or non-integral")
    if set(assignments["fold_id"].astype(int)) != set(range(config.group_folds)):
        raise SplitValidationError("fold IDs must cover every integer fold from zero through four")
    selection = select_fold_assignment(grouped, targets, config, seed)
    if selection["status"] != "PASS":
        raise SplitValidationError("recomputed fold selection is not feasible")
    expected_folds = [
        int(selection["fold_by_row"][str(row_id)]) for row_id in assignments[ROW_ID_COLUMN].tolist()
    ]
    observed_folds = assignments["fold_id"].astype(int).tolist()
    if observed_folds != expected_folds:
        raise SplitValidationError("assignment fold provenance differs from recomputed folds")
    if manifest.fold_assignment != selection["fold_assignment"]:
        raise SplitValidationError("manifest fold assignment differs from recomputed selection")
    fold_to_partition = {
        fold: partition
        for partition, folds in selection["fold_assignment"].items()
        for fold in folds
    }
    if any(fold not in fold_to_partition for fold in observed_folds):
        raise SplitValidationError("a persisted fold is not assigned to a partition")
    if any(
        group_frame["fold_id"].nunique() != 1
        for _, group_frame in assignments.groupby("predictor_group_id")
    ):
        raise SplitValidationError("a predictor group is split across folds")
    if any(
        group_frame["partition"].nunique() != 1 for _, group_frame in assignments.groupby("fold_id")
    ):
        raise SplitValidationError("a fold is split across partitions")
    if any(
        partition != fold_to_partition[int(fold)]
        for partition, fold in zip(assignments["partition"].tolist(), observed_folds, strict=True)
    ):
        raise SplitValidationError("persisted fold and partition provenance disagree")
    crossing = int(assignments.groupby("predictor_group_id")["partition"].nunique().gt(1).sum())
    if crossing != 0:
        raise SplitValidationError("predictor groups cross split partitions")
    counts = {
        part: int((assignments["partition"] == part).sum())
        for part in ("train", "calibration", "test")
    }
    class_counts = _class_counts(assignments, targets)
    if counts != manifest.partition_counts or class_counts != manifest.partition_class_counts:
        raise SplitValidationError("manifest partition counts do not match assignments")
    if any(
        class_counts[part].get(str(code), 0) < config.minimum_class_count_per_partition
        for part in counts
        for code in manifest.class_values
    ):
        raise SplitValidationError("minimum class count constraint failed")
    if sha256_file(assignment_path) != manifest.assignment_artifact_hash:
        raise SplitValidationError("assignment artifact hash mismatch")
    logical_hash = logical_assignment_hash(assignments.to_dict(orient="records"))
    if logical_hash != manifest.logical_assignment_hash:
        raise SplitValidationError("logical assignment hash mismatch")
    processed = project_root / "data" / "processed" / "openml" / str(dataset_id)
    feature_hash = sha256_file(processed / "features.parquet")
    target_hash = sha256_file(processed / "targets.parquet")
    data_manifest_hash = sha256_file(processed / "data_manifest.json")
    if (
        feature_hash != manifest.feature_artifact_hash
        or target_hash != manifest.target_artifact_hash
    ):
        raise SplitValidationError("processed feature or target artifact hash mismatch")
    if data_manifest_hash != manifest.dataset_manifest_hash:
        raise SplitValidationError("processed data manifest hash mismatch")
    identity = cache_identity(
        manifest.dataset_id,
        manifest.dataset_version,
        feature_hash,
        target_hash,
        data_manifest_hash,
        manifest.seed,
        manifest.strategy,
        manifest.strategy_version,
        manifest.grouping_implementation_hash,
        manifest.configuration_hash,
        manifest.source_commit,
        artifact_schema_version=manifest.schema_version,
        split_implementation_hash=manifest.split_implementation_hash,
    )
    if cache_key(identity) != manifest.cache_identity_hash:
        raise SplitValidationError("cache identity hash mismatch")
    if int(manifest.predictor_group_count) != int(assignments["predictor_group_id"].nunique()):
        raise SplitValidationError("predictor group count mismatch")
    if manifest.cross_partition_group_count != crossing:
        raise SplitValidationError("cross-partition group count mismatch")
    return {
        "status": "PASS",
        "dataset_id": dataset_id,
        "seed": seed,
        "protected_baseline": False,
        "row_count": len(assignments),
        "partition_counts": counts,
        "partition_class_counts": class_counts,
        "cross_partition_group_count": crossing,
        "assignment_artifact_hash": manifest.assignment_artifact_hash,
        "logical_assignment_hash": logical_hash,
    }


def validate_split(
    root: str | Path, config: SplitGenerationConfig, dataset_id: int, seed: int
) -> dict[str, Any]:
    return validate_split_directory(
        root, config, dataset_id, seed, split_directory(root, dataset_id, seed, config.strategy)
    )


def validate_all(root: str | Path, config: SplitGenerationConfig) -> list[dict[str, Any]]:
    _reject_temporary_artifacts(Path(root))
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for dataset_id in config.datasets:
        for seed in config.seeds:
            try:
                results.append(validate_split(root, config, dataset_id, seed))
            except Exception as exc:
                errors.append(f"{dataset_id}/{seed}: {exc}")
    if errors:
        raise SplitValidationError("; ".join(errors))
    return results
