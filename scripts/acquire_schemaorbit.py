"""Acquire and validate SchemaOrbit-14 without creating new split artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.artifact_contracts import DatasetRegistryReportContract  # noqa: E402
from schemaguard.data.schemaorbit import (  # noqa: E402
    SchemaOrbitError,
    acquire_raw,
    dataset_inventory_record,
    load_schemaorbit_config,
    process_dataset,
    validate_processed,
)
from schemaguard.data.splits import predictor_group_ids  # noqa: E402
from schemaguard.utils.hashing import sha256_file  # noqa: E402
from schemaguard.utils.io import atomic_write_json, atomic_write_parquet  # noqa: E402


def _tree_hashes(root: Path, directories: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory in directories:
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            result[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return result


def _data_foundation_summary(root: Path) -> dict[str, Any]:
    base = root / "data" / "processed" / "openml" / "1464"
    features = pd.read_parquet(base / "features.parquet")
    targets = pd.read_parquet(base / "targets.parquet")
    split_dir = (
        root / "data" / "splits" / "openml" / "1464" / "stratified_group_5fold_v1" / "seed_1729"
    )
    assignments = pd.read_parquet(split_dir / "assignments.parquet")
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    joined = assignments.merge(targets, on="__sg_row_id", validate="one_to_one")
    group_ids = predictor_group_ids(features)
    group_frame = pd.DataFrame({"group": group_ids, "row_id": features["__sg_row_id"]}).merge(
        assignments, left_on="row_id", right_on="__sg_row_id", validate="one_to_one"
    )
    crossing = int(group_frame.groupby("group")["split"].nunique().gt(1).sum())
    group_sizes = group_ids.value_counts()
    conflicts = (
        pd.DataFrame({"group": group_ids, "target": targets["target_code"]})
        .groupby("group")["target"]
        .nunique()
    )
    sizes = joined["split"].value_counts().to_dict()
    class_counts = {
        split: {
            str(k): int(v)
            for k, v in joined.loc[joined["split"] == split, "target_code"]
            .value_counts()
            .sort_index()
            .items()
        }
        for split in ("train", "calibration", "test")
    }
    if sizes != {"train": 449, "calibration": 150, "test": 149}:
        raise SchemaOrbitError(f"Data-foundation split sizes changed: {sizes}")
    if crossing != 0:
        raise SchemaOrbitError(f"Data-foundation grouped split crosses {crossing} predictor groups")
    if len(features) != 748 or len(group_sizes) != 502 or int((group_sizes > 1).sum()) != 69:
        raise SchemaOrbitError("Data-foundation row/group counts changed")
    if int(conflicts.gt(1).sum()) != 31:
        raise SchemaOrbitError("Data-foundation conflicting-target group count changed")
    return {
        "rows": len(features),
        "split_sizes": {k: int(v) for k, v in sizes.items()},
        "class_counts": class_counts,
        "predictor_groups": int(len(group_sizes)),
        "duplicate_predictor_groups": int((group_sizes > 1).sum()),
        "conflicting_target_groups": int(conflicts.gt(1).sum()),
        "predictor_groups_crossing_splits": crossing,
        "strategy": manifest["strategy"],
        "deprecated_row_split_not_selected": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets/schemaorbit14.yaml")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "results/validation")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.offline and args.allow_network:
        raise SystemExit("--offline and --allow-network are mutually exclusive")
    config = load_schemaorbit_config(args.config)
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    review = ROOT / "artifacts" / "data_foundation" / "review"
    review.mkdir(parents=True, exist_ok=True)
    data_foundation_dirs = [
        "data/raw/openml/1464",
        "data/processed/openml/1464",
        "data/splits/openml/1464/stratified_group_5fold_v1/seed_1729",
    ]
    before_path = review / "data_foundation_hashes_before.json"
    if not before_path.exists():
        atomic_write_json(before_path, _tree_hashes(ROOT, data_foundation_dirs))
    before = json.loads(before_path.read_text(encoding="utf-8"))
    if before != _tree_hashes(ROOT, data_foundation_dirs):
        raise SystemExit(
            "FAIL_DATA_FOUNDATION_MUTATION: data foundation changed before acquisition"
        )
    data_foundation = _data_foundation_summary(ROOT)
    inventory: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    client: httpx.Client | None = None
    if not args.offline and args.allow_network:
        client = httpx.Client(
            timeout=config.acquisition.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "SchemaGuard/SchemaOrbit"},
        )
    try:
        for spec in config.datasets:
            raw_path, metadata, features, raw_manifest = acquire_raw(
                client,
                ROOT,
                config,
                spec,
                offline=args.offline,
                allow_network=args.allow_network,
                refresh=args.refresh,
            )
            quality = process_dataset(ROOT, config, spec, raw_path, features, raw_manifest)
            # Run the read-only validation a second time after processing. This proves the
            # generated/reused artifacts are independently readable and source-bound.
            quality = (
                validate_processed(ROOT, config, spec, str(raw_manifest["computed_sha256"]))
                | quality
            )
            inventory.append(
                dataset_inventory_record(spec, metadata, features, raw_manifest, quality)
            )
            validations.append(
                {
                    "schema_version": 1,
                    "openml_data_id": spec.id,
                    "dataset_name": spec.name,
                    "dataset_version": metadata.version,
                    "rows_expected": spec.rows,
                    "rows_observed": quality["row_count"],
                    "predictors_expected": spec.predictors,
                    "predictors_observed": quality["predictor_count"],
                    "classes": quality["class_count"],
                    "class_counts": json.dumps(quality["class_counts"], sort_keys=True),
                    "ignored_attributes": json.dumps(spec.ignore_attributes),
                    "raw_sha256": raw_manifest["computed_sha256"],
                    "provider_md5": spec.provider_md5,
                    "missing_cells": quality["missing_cells"],
                    "duplicate_predictor_groups": quality["duplicate_predictor_groups"],
                    "conflicting_target_groups": quality["conflicting_target_groups"],
                    "status": "PASS",
                }
            )
    finally:
        if client is not None:
            client.close()
    after = _tree_hashes(ROOT, data_foundation_dirs)
    atomic_write_json(review / "data_foundation_hashes_after.json", after)
    comparison = {
        "unchanged": before == after,
        "changed_paths": sorted(
            set(before) ^ set(after)
            | {path for path in set(before) & set(after) if before[path] != after[path]}
        ),
    }
    atomic_write_json(review / "data_foundation_hash_comparison.json", comparison)
    if not comparison["unchanged"]:
        raise SystemExit(
            "FAIL_DATA_FOUNDATION_MUTATION: data foundation changed during acquisition"
        )
    report = {
        "schema_version": 2,
        "stage": "dataset_registry",
        "benchmark": "SchemaOrbit-14",
        "status": "PASS",
        "dataset_count": len(inventory),
        "datasets": inventory,
        "data_foundation": data_foundation,
        "validation_records": validations,
    }
    DatasetRegistryReportContract.model_validate(report)
    atomic_write_json(output / "dataset_registry_report.json", report)
    atomic_write_parquet(output / "dataset_registry_validation.parquet", pd.DataFrame(validations))
    print(
        json.dumps(
            {"status": "PASS", "datasets": len(inventory), "data_foundation": data_foundation},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SchemaOrbitError, httpx.HTTPError) as exc:
        print(f"BLOCKED_OR_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
