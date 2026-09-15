"""Generate one deterministic SchemaOrbit split without mutating source data."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from schemaguard.constants import ROW_ID_COLUMN, TARGET_CODE_COLUMN
from schemaguard.data.contracts import RawSourceManifestContract
from schemaguard.data.schemaorbit import (
    SchemaOrbitError,
    load_schemaorbit_config,
    validate_processed,
)
from schemaguard.utils.hashing import sha256_file
from schemaguard.utils.io import atomic_write_json, atomic_write_parquet

from .caching import logical_assignment_hash
from .contracts import SplitGenerationConfig, SplitManifestContract
from .grouping import GROUPING_IMPLEMENTATION_HASH, grouped_features
from .locking import split_lock
from .selection import select_fold_assignment

PROTECTED_DATASET_SEED = (1464, 1729)
PROTECTED_ASSIGNMENT_SHA256 = "e08b9d5c94578b317dcc844c2a6aa7a4f96ad8df699352d1e11d86a7dd452236"
PROTECTED_LOGICAL_ASSIGNMENT_SHA256 = (
    "f66b2ab092218813b82e49b6a3c19df98b06840d90b463bd9d09ba1bce385365"
)


class SplitGenerationError(RuntimeError):
    """A split cannot be generated or validated from local evidence."""


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SplitGenerationError("Cannot determine source commit") from exc


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def split_directory(root: str | Path, dataset_id: int, seed: int, strategy: str) -> Path:
    return Path(root) / "data" / "splits" / "openml" / str(dataset_id) / strategy / f"seed_{seed}"


def _quarantine(directory: Path, root: Path, reason: str) -> Path:
    destination = (
        root
        / "data"
        / "cache"
        / "quarantine"
        / "splits"
        / (f"{directory.parent.parent.name}-{directory.name}-{reason}-{time.time_ns()}")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(destination))
    return destination


def _read_local_inputs(
    root: Path, dataset_id: int
) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    schema_config = load_schemaorbit_config(root / "configs" / "datasets" / "schemaorbit14.yaml")
    spec = next((item for item in schema_config.datasets if item.id == dataset_id), None)
    if spec is None:
        raise SplitGenerationError(f"Unknown SchemaOrbit dataset ID: {dataset_id}")
    raw_directory = root / "data" / "raw" / "openml" / str(dataset_id)
    processed_directory = root / "data" / "processed" / "openml" / str(dataset_id)
    raw_manifest_path = raw_directory / "source_manifest.json"
    if not raw_manifest_path.is_file():
        raise SplitGenerationError(f"BLOCKED_DATASET_ARTIFACT: missing {raw_manifest_path}")
    try:
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        if dataset_id != 1464:
            RawSourceManifestContract.model_validate(raw_manifest)
        elif raw_manifest.get("openml_data_id") != 1464:
            raise ValueError("legacy smoke source manifest identity mismatch")
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise SplitGenerationError(f"Invalid raw source manifest for dataset {dataset_id}") from exc
    raw_sha = str(raw_manifest["computed_sha256"])
    raw_path = root / "data" / str(raw_manifest["raw_relative_path"])
    if not raw_path.is_file() or sha256_file(raw_path) != raw_sha:
        raise SplitGenerationError(f"Raw artifact hash mismatch for dataset {dataset_id}")
    try:
        quality = validate_processed(root, schema_config, spec, raw_sha)
    except SchemaOrbitError as exc:
        raise SplitGenerationError(str(exc)) from exc
    features_path = processed_directory / "features.parquet"
    targets_path = processed_directory / "targets.parquet"
    data_manifest_path = processed_directory / "data_manifest.json"
    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)
    if features[ROW_ID_COLUMN].tolist() != targets[ROW_ID_COLUMN].tolist():
        raise SplitGenerationError(f"Feature/target row order mismatch for dataset {dataset_id}")
    if set(targets[TARGET_CODE_COLUMN]) != set(range(targets[TARGET_CODE_COLUMN].nunique())):
        raise SplitGenerationError(f"Target codes are not contiguous for dataset {dataset_id}")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    return schema_config, spec, features, targets, quality, data_manifest


def _assignment_rows(
    features: pd.DataFrame, targets: pd.DataFrame, selection: dict[str, Any]
) -> tuple[pd.DataFrame, str]:
    groups = features[[ROW_ID_COLUMN, "predictor_group_id"]].copy()
    row_ids = targets[ROW_ID_COLUMN].tolist()
    labels = selection["labels"]
    frame = pd.DataFrame(
        {
            ROW_ID_COLUMN: pd.Series(row_ids, dtype="string"),
            "predictor_group_id": groups.set_index(ROW_ID_COLUMN)
            .loc[row_ids]["predictor_group_id"]
            .astype("string")
            .tolist(),
            "partition": pd.Series(
                [labels[row_id] for row_id in row_ids], dtype="string"
            ),
        }
    )
    # The exact fold ID is not needed to identify a partition, but retaining it
    # makes candidate selection and independent audits inspectable.
    fold_lookup = {}
    for partition, fold_ids in selection["fold_assignment"].items():
        for fold_id in fold_ids:
            fold_lookup[partition] = fold_id
    frame["fold_id"] = frame["partition"].map(fold_lookup).astype("int64")
    frame = frame[[ROW_ID_COLUMN, "predictor_group_id", "fold_id", "partition"]]
    return frame, logical_assignment_hash(frame.to_dict(orient="records"))


def _manifest(
    root: Path,
    spec: Any,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    data_manifest: dict[str, Any],
    data_manifest_hash: str,
    assignment_hash: str,
    logical_hash: str,
    selection: dict[str, Any],
    seed: int,
    source_commit: str,
    configuration_hash: str,
) -> SplitManifestContract:
    feature_path = root / "data" / "processed" / "openml" / str(spec.id) / "features.parquet"
    target_path = root / "data" / "processed" / "openml" / str(spec.id) / "targets.parquet"
    return SplitManifestContract(
        dataset_id=spec.id,
        dataset_version=spec.version,
        dataset_name=spec.name,
        seed=seed,
        strategy="stratified_group_5fold_v1",
        strategy_version="v1",
        grouping_method="typed_predictor_sha256_v1",
        train_fraction=0.60,
        calibration_fraction=0.20,
        test_fraction=0.20,
        row_count=len(targets),
        predictor_count=int(len(data_manifest.get("feature_columns", list(features.columns[1:])))),
        class_values=sorted(int(value) for value in targets[TARGET_CODE_COLUMN].unique()),
        partition_counts=selection["partition_counts"],
        partition_class_counts=selection["partition_class_counts"],
        predictor_group_count=int(features["predictor_group_id"].nunique()),
        duplicate_group_count=int(features["predictor_group_id"].value_counts().gt(1).sum()),
        conflicting_target_group_count=int(
            pd.DataFrame(
                {"group": features["predictor_group_id"], "target": targets[TARGET_CODE_COLUMN]}
            )
            .groupby("group")["target"]
            .nunique()
            .gt(1)
            .sum()
        ),
        cross_partition_group_count=0,
        feature_artifact_hash=sha256_file(feature_path),
        target_artifact_hash=sha256_file(target_path),
        dataset_manifest_hash=data_manifest_hash,
        assignment_artifact_hash=assignment_hash,
        logical_assignment_hash=logical_hash,
        configuration_hash=configuration_hash,
        grouping_implementation_hash=GROUPING_IMPLEMENTATION_HASH,
        source_commit=source_commit,
        created_at=_utc_now(),
        validation_status="PASS",
        fold_assignment=selection["fold_assignment"],
        selection_score=selection["selection_score"],
        size_deviations=selection["size_deviations"],
        class_proportion_deviations=selection["class_proportion_deviations"],
        candidate_count=selection["candidate_count"],
        candidate_diagnostics=selection["candidate_diagnostics"],
        determinism={
            "seed": seed,
            "fold_seed": selection.get("fold_seed"),
            "candidate_count": selection["candidate_count"],
        },
    )


def _existing_is_valid(
    path: Path, root: Path, config: SplitGenerationConfig, dataset_id: int, seed: int
) -> bool:
    from .validation import validate_split

    try:
        validate_split(root, config, dataset_id, seed)
        return True
    except Exception:
        if path.is_dir():
            _quarantine(path, root, "invalid")
        return False


def generate_split(
    root: str | Path,
    config: SplitGenerationConfig,
    dataset_id: int,
    seed: int,
    *,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Generate or reuse one dataset/seed split under a process lock."""

    project_root = Path(root)
    if dataset_id not in config.datasets or seed not in config.seeds:
        raise SplitGenerationError("dataset ID or seed is outside the frozen configuration")
    target_directory = split_directory(project_root, dataset_id, seed, config.strategy)
    with split_lock(project_root, dataset_id, seed):
        if (dataset_id, seed) == PROTECTED_DATASET_SEED:
            assignment_path = target_directory / "assignments.parquet"
            if (
                not assignment_path.is_file()
                or sha256_file(assignment_path) != PROTECTED_ASSIGNMENT_SHA256
            ):
                raise SplitGenerationError(
                    "FAIL_PROTECTED_DATASET_MUTATION: protected grouped split "
                    "is not the accepted artifact"
                )
            from .validation import validate_split

            result = validate_split(project_root, config, dataset_id, seed)
            return {
                **result,
                "dataset_id": dataset_id,
                "seed": seed,
                "status": "PROTECTED_BASELINE",
            }
        if (
            target_directory.exists()
            and not force_recompute
            and _existing_is_valid(target_directory, project_root, config, dataset_id, seed)
        ):
            from .validation import validate_split

            result = validate_split(project_root, config, dataset_id, seed)
            return {
                **result,
                "dataset_id": dataset_id,
                "seed": seed,
                "status": "CACHE_HIT",
            }
        _schema_config, spec, features, targets, quality, data_manifest = _read_local_inputs(
            project_root, dataset_id
        )
        del quality
        groups = grouped_features(
            features.drop(columns=list(spec.ignore_attributes), errors="ignore")
        )
        grouped = features.drop(columns=list(spec.ignore_attributes), errors="ignore").copy()
        grouped["predictor_group_id"] = groups["predictor_group_id"].tolist()
        selection = select_fold_assignment(grouped, targets, config, seed)
        if selection["status"] != "PASS":
            raise SplitGenerationError(f"CONSTRAINT_INFEASIBLE: dataset {dataset_id} seed {seed}")
        selection["fold_seed"] = selection.get("fold_seed")
        assignment_frame, logical_hash = _assignment_rows(grouped, targets, selection)
        parent = target_directory.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".seed_{seed}.{os.getpid()}.{time.time_ns()}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            temporary_assignment = temporary / "assignments.parquet"
            atomic_write_parquet(
                temporary_assignment, assignment_frame, compression="zstd", compression_level=3
            )
            assignment_hash = sha256_file(temporary_assignment)
            data_manifest_hash = sha256_file(
                project_root
                / "data"
                / "processed"
                / "openml"
                / str(dataset_id)
                / "data_manifest.json"
            )
            manifest = _manifest(
                project_root,
                spec,
                grouped,
                targets,
                data_manifest,
                data_manifest_hash,
                assignment_hash,
                logical_hash,
                selection,
                seed,
                _git_commit(project_root),
                config.configuration_hash,
            )
            atomic_write_json(temporary / "split_manifest.json", manifest.canonical_dict())
            # Validate the complete temporary artifact before publication.
            from .validation import validate_split_directory

            validate_split_directory(project_root, config, dataset_id, seed, temporary)
            if target_directory.exists():
                _quarantine(target_directory, project_root, "replaced")
            os.replace(temporary, target_directory)
        except Exception:
            if temporary.exists():
                shutil.move(
                    str(temporary),
                    str(
                        project_root
                        / "data"
                        / "cache"
                        / "quarantine"
                        / f"split-incomplete-{dataset_id}-{seed}-{time.time_ns()}"
                    ),
                )
            raise
        return {
            "dataset_id": dataset_id,
            "seed": seed,
            "status": "GENERATED",
            "assignment_artifact_hash": assignment_hash,
            "logical_assignment_hash": logical_hash,
        }
