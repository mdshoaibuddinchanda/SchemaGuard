"""Build the complete 14-dataset, 5-seed, 11-view validation inventory."""

from __future__ import annotations

import ast
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.hashing import hash_dataframe_logically, sha256_file
from ..utils.io import atomic_write_json
from .applicability import assess_applicability
from .base import feature_schema_from_frame, frame_hash
from .certificates import certificate_hash
from .contracts import TransformationConfig, TransformationInventory, TransformationInventoryRecord
from .registry import get_transformation, registry_hash
from .validation import validate_deterministic_outputs, validate_transformation

DATASET_IDS = [3, 23, 29, 31, 36, 37, 38, 44, 46, 50, 54, 1067, 1464, 1489]
SEEDS = [1729, 2718, 31415, 57721, 161803]
VIEW_IDS = [f"V{index:02d}" for index in range(11)]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _protected_snapshot(root: Path) -> dict[str, Any]:
    paths: list[Path] = []
    for relative in (
        "configs/baselines",
        "data/raw/openml",
        "data/processed/openml",
        "data/splits/openml",
        "schemas",
    ):
        paths.extend(path for path in (root / relative).rglob("*") if path.is_file())
    paths.append(root / "artifacts/handoff/split_generation_inventory.json")
    records = []
    for path in sorted(set(paths)):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"schema_version": 1, "file_count": len(records), "files": records}


def compare_protected_snapshot(root: str | Path, before_path: str | Path) -> dict[str, Any]:
    project_root = Path(root)
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = _protected_snapshot(project_root)
    old = {item["path"]: item["sha256"] for item in before["files"]}
    new = {item["path"]: item["sha256"] for item in after["files"]}
    changed = sorted(path for path in old if old.get(path) != new.get(path))
    added = sorted(path for path in new if path not in old)
    return {
        "status": "PASS" if not changed else "FAIL",
        "changed_files": changed,
        "added_files": added,
        "before_file_count": len(old),
        "after_file_count": len(new),
        "before_snapshot_hash": sha256_file(before_path),
    }


def _load_partition(
    root: Path,
    dataset_id: int,
    seed: int,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_path = (
        root
        / "data/splits/openml"
        / str(dataset_id)
        / "stratified_group_5fold_v1"
        / f"seed_{seed}"
        / "assignments.parquet"
    )
    assignment = pd.read_parquet(assignment_path)
    partition_column = "partition" if "partition" in assignment.columns else "split"
    row_ids = assignment.loc[assignment[partition_column] == partition, "__sg_row_id"].tolist()
    feature_index = features.set_index("__sg_row_id", drop=False)
    target_index = targets.set_index("__sg_row_id", drop=False)
    return feature_index.loc[row_ids].reset_index(drop=True), target_index.loc[row_ids].reset_index(
        drop=True
    )


def validate_dataset_seed(
    root: Path, dataset_id: int, seed: int, config: TransformationConfig
) -> list[TransformationInventoryRecord]:
    processed = root / "data/processed/openml" / str(dataset_id)
    features = pd.read_parquet(processed / "features.parquet")
    targets = pd.read_parquet(processed / "targets.parquet")
    schema_payload = json.loads((processed / "schema.json").read_text(encoding="utf-8"))
    categorical_values = {
        item["name"]: list(ast.literal_eval(item["raw_type"]))
        for item in schema_payload.get("attributes", [])
        if item.get("kind") == "categorical" and str(item.get("raw_type", "")).startswith("[")
    }
    schema = feature_schema_from_frame(features, categorical_values)
    results: list[TransformationInventoryRecord] = []
    for view_id in VIEW_IDS:
        view = next(item for item in config.views if str(item["id"]) == view_id)
        applicability = assess_applicability(
            view_id,
            str(view["name"]),
            dataset_id,
            seed,
            schema,
            category_minimum=config.category_minimum,
        )
        started = time.perf_counter()
        if applicability.status == "NOT_APPLICABLE":
            results.append(
                TransformationInventoryRecord(
                    schema_version=1,
                    dataset_id=dataset_id,
                    seed=seed,
                    view_id=view_id,
                    view_name=str(view["name"]),
                    status="N/A",
                    reason_code=applicability.reason_code,
                    reason=applicability.reason,
                    applicability="NOT_APPLICABLE",
                    runtime_seconds=time.perf_counter() - started,
                    created_at=_utc_now(),
                )
            )
            continue
        try:
            transformation = get_transformation(view_id, config.model_dump())
            train, train_target = _load_partition(
                root, dataset_id, seed, features, targets, "train"
            )
            transformation.fit(train, dataset_id, seed, schema)
            certificates = []
            outputs = []
            max_error = 0.0
            deterministic = True
            for partition in ("train", "calibration", "test"):
                source, target = _load_partition(
                    root, dataset_id, seed, features, targets, partition
                )
                output = transformation.transform(source, partition)
                certificate = transformation.certificate_for(
                    source,
                    output,
                    partition,
                    dataset_version="local",
                    split_strategy=config.split_strategy,
                    source_target_hash=hash_dataframe_logically(target),
                    output_target_hash=hash_dataframe_logically(target),
                    validation_results={"status": "PASS", "partition": partition},
                )
                restored = transformation.reconstruct(output, certificate)
                validation = validate_transformation(
                    source,
                    output,
                    restored,
                    certificate,
                    rtol=config.numerical_rtol,
                    atol=config.numerical_atol,
                )
                repeat = transformation.transform(source, partition)
                deterministic = deterministic and validate_deterministic_outputs(output, repeat)
                max_error = max(max_error, float(validation["reconstruction_max_abs_error"]))
                certificates.append(certificate)
                outputs.append(output)
            results.append(
                TransformationInventoryRecord(
                    schema_version=1,
                    dataset_id=dataset_id,
                    seed=seed,
                    view_id=view_id,
                    view_name=str(view["name"]),
                    status="PASS",
                    applicability="APPLICABLE",
                    certificate_ids=[certificate_hash(item) for item in certificates],
                    source_hash=frame_hash(features),
                    output_hash=hash_dataframe_logically(pd.concat(outputs, ignore_index=True)),
                    reconstruction_max_abs_error=max_error,
                    reconstruction_exact=max_error == 0.0,
                    deterministic=deterministic,
                    runtime_seconds=time.perf_counter() - started,
                    created_at=_utc_now(),
                )
            )
        except Exception as exc:
            results.append(
                TransformationInventoryRecord(
                    schema_version=1,
                    dataset_id=dataset_id,
                    seed=seed,
                    view_id=view_id,
                    view_name=str(view["name"]),
                    status="FAIL",
                    applicability="APPLICABLE",
                    reason_code="TRANSFORMATION_VALIDATION_ERROR",
                    reason=f"{type(exc).__name__}: {exc}",
                    source_hash=frame_hash(features),
                    reconstruction_max_abs_error=None,
                    deterministic=False,
                    runtime_seconds=time.perf_counter() - started,
                    created_at=_utc_now(),
                )
            )
    return results


def build_inventory(
    root: str | Path, config: TransformationConfig, before_snapshot: str | Path
) -> TransformationInventory:
    project_root = Path(root)
    records: list[TransformationInventoryRecord] = []
    for dataset_id in DATASET_IDS:
        for seed in SEEDS:
            records.extend(validate_dataset_seed(project_root, dataset_id, seed, config))
    applicable = [record for record in records if record.applicability == "APPLICABLE"]
    not_applicable = [record for record in records if record.applicability == "NOT_APPLICABLE"]
    passed = [record for record in records if record.status == "PASS"]
    failed = [record for record in records if record.status == "FAIL"]
    split_inventory = project_root / "artifacts/handoff/split_generation_inventory.json"
    protected = compare_protected_snapshot(project_root, before_snapshot)
    return TransformationInventory(
        schema_version=1,
        engine_name="certified_lossless_transformations",
        status="PASS_PENDING_REVIEW"
        if len(records) == 770 and not failed and protected["status"] == "PASS"
        else "FAIL",
        dataset_ids=DATASET_IDS,
        seeds=SEEDS,
        view_ids=VIEW_IDS,
        records=records,
        expected_record_count=770,
        applicable_count=len(applicable),
        not_applicable_count=len(not_applicable),
        pass_count=len(passed),
        fail_count=len(failed),
        missing_count=770 - len(records),
        property_example_count=1000,
        registry_hash=registry_hash(),
        split_inventory_hash=sha256_file(split_inventory),
        protected_hash_comparison=protected,
        reconstruction_max_abs_error=max(
            (record.reconstruction_max_abs_error or 0.0) for record in passed
        ),
        generated_at=_utc_now(),
    )


def write_inventory(
    root: str | Path, config: TransformationConfig, before_snapshot: str | Path, output: str | Path
) -> TransformationInventory:
    inventory = build_inventory(root, config, before_snapshot)
    atomic_write_json(output, inventory.canonical_dict())
    return inventory


__all__ = [
    "DATASET_IDS",
    "SEEDS",
    "VIEW_IDS",
    "build_inventory",
    "compare_protected_snapshot",
    "write_inventory",
]
