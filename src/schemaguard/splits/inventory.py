"""Inventory and protected-artifact hash snapshots for split generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from schemaguard.utils.hashing import sha256_file
from schemaguard.utils.io import atomic_write_json

from .contracts import SplitGenerationConfig, SplitGenerationInventoryContract, SplitInventoryRecord
from .generation import PROTECTED_LOGICAL_ASSIGNMENT_SHA256


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def protected_paths(root: str | Path, config: SplitGenerationConfig) -> list[Path]:
    base = Path(root)
    paths: list[Path] = [base / "configs" / "baselines" / "data_foundation.json"]
    for dataset_id in config.datasets:
        raw = base / "data" / "raw" / "openml" / str(dataset_id)
        processed = base / "data" / "processed" / "openml" / str(dataset_id)
        if raw.is_dir():
            paths.extend(path for path in raw.iterdir() if path.is_file())
        if processed.exists():
            paths.extend(path for path in processed.iterdir() if path.is_file())
    existing = (
        base / "data" / "splits" / "openml" / "1464" / "stratified_group_5fold_v1" / "seed_1729"
    )
    if existing.exists():
        paths.extend(path for path in existing.iterdir() if path.is_file())
    return sorted(set(paths))


def snapshot_paths(root: str | Path, paths: list[Path]) -> dict[str, Any]:
    base = Path(root)
    return {
        "schema_version": 1,
        "captured_at": _utc_now(),
        "files": {
            str(path.relative_to(base)).replace("\\", "/"): sha256_file(path)
            for path in paths
            if path.is_file()
        },
    }


def write_protected_snapshot(
    root: str | Path, config: SplitGenerationConfig, output: str | Path
) -> dict[str, Any]:
    snapshot = snapshot_paths(root, protected_paths(root, config))
    atomic_write_json(output, snapshot)
    return snapshot


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_files = before.get("files", {})
    after_files = after.get("files", {})
    changed = sorted(
        path
        for path in set(before_files) | set(after_files)
        if before_files.get(path) != after_files.get(path)
    )
    return {
        "schema_version": 1,
        "status": "PASS" if not changed else "FAIL",
        "changed_files": changed,
        "file_count_before": len(before_files),
        "file_count_after": len(after_files),
    }


def build_inventory(
    root: str | Path, config: SplitGenerationConfig
) -> SplitGenerationInventoryContract:
    base = Path(root)
    records: list[SplitInventoryRecord] = []
    for dataset_id in config.datasets:
        for seed in config.seeds:
            directory = (
                base
                / "data"
                / "splits"
                / "openml"
                / str(dataset_id)
                / config.strategy
                / f"seed_{seed}"
            )
            assignments_path = directory / "assignments.parquet"
            manifest_path = directory / "split_manifest.json"
            if not assignments_path.is_file() or not manifest_path.is_file():
                records.append(
                    SplitInventoryRecord(
                        dataset_id=dataset_id,
                        seed=seed,
                        status="MISSING",
                        protected_baseline=dataset_id == 1464 and seed == 1729,
                        assignments_path=str(assignments_path.relative_to(base)),
                        manifest_path=str(manifest_path.relative_to(base)),
                    )
                )
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            protected = dataset_id == 1464 and seed == 1729 and "internal_dataset_id" in payload
            if protected:
                frame = pd.read_parquet(assignments_path)
                status: Literal["PASS", "FAILED"] = (
                    "PASS" if payload.get("validation_status") == "PASS" else "FAILED"
                )
                records.append(
                    SplitInventoryRecord(
                        dataset_id=dataset_id,
                        seed=seed,
                        status=status,
                        protected_baseline=True,
                        assignments_path=str(assignments_path.relative_to(base)),
                        manifest_path=str(manifest_path.relative_to(base)),
                        assignment_artifact_hash=sha256_file(assignments_path),
                        manifest_artifact_hash=sha256_file(manifest_path),
                        logical_assignment_hash=PROTECTED_LOGICAL_ASSIGNMENT_SHA256,
                        row_count=len(frame),
                        cross_partition_group_count=int(
                            payload.get("predictor_duplicate_groups_crossing_splits", 0)
                        ),
                    )
                )
            else:
                from .contracts import SplitManifestContract

                manifest = SplitManifestContract.model_validate(payload)
                records.append(
                    SplitInventoryRecord(
                        dataset_id=dataset_id,
                        seed=seed,
                        status=manifest.validation_status,
                        protected_baseline=False,
                        assignments_path=str(assignments_path.relative_to(base)),
                        manifest_path=str(manifest_path.relative_to(base)),
                        assignment_artifact_hash=sha256_file(assignments_path),
                        manifest_artifact_hash=sha256_file(manifest_path),
                        logical_assignment_hash=manifest.logical_assignment_hash,
                        row_count=manifest.row_count,
                        cross_partition_group_count=manifest.cross_partition_group_count,
                    )
                )
    return SplitGenerationInventoryContract(
        generated_at=_utc_now(),
        configuration_hash=config.configuration_hash,
        expected_dataset_ids=config.datasets,
        expected_seeds=config.seeds,
        records=records,
    )


def write_inventory(
    root: str | Path, config: SplitGenerationConfig, output: str | Path
) -> SplitGenerationInventoryContract:
    inventory = build_inventory(root, config)
    atomic_write_json(output, inventory.canonical_dict())
    return inventory
