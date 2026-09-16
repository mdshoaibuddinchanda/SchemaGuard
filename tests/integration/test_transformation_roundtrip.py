from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from schemaguard.transformations.base import feature_schema_from_frame
from schemaguard.transformations.caching import (
    TransformationCache,
    certificate_parameter_hash,
    transformation_cache_key,
)
from schemaguard.transformations.contracts import TransformationManifest
from schemaguard.transformations.inventory import VIEW_IDS
from schemaguard.transformations.registry import get_transformation, load_transformation_config
from schemaguard.transformations.serialization import (
    validate_manifest_directory,
    write_certificate,
    write_manifest_last,
    write_partition,
)
from schemaguard.transformations.validation import validate_transformation
from schemaguard.utils.hashing import hash_dataframe_logically, sha256_canonical_json, sha256_file


def _source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "__sg_row_id": [f"e2e-{index}" for index in range(12)],
            "amount": [float(index - 6) for index in range(12)],
            "count": list(range(-6, 6)),
            "category": (["a:b", "quoted'\"", r"slash\\", "", "Δ"] * 2) + ["a:b", "quoted'\""],
        }
    )
    targets = pd.DataFrame(
        {
            "__sg_row_id": features["__sg_row_id"],
            "__sg_target_code": [index % 2 for index in range(12)],
        }
    )
    assignment = pd.DataFrame(
        {
            "__sg_row_id": features["__sg_row_id"],
            "partition": ["train"] * 6 + ["calibration"] * 3 + ["test"] * 3,
        }
    )
    return features, targets, assignment


@pytest.mark.integration
def test_end_to_end_materialization_cache_and_promotion_for_all_views(tmp_path: Path) -> None:
    config = load_transformation_config("configs/runtime/transformation_engine.yaml")
    features, targets, assignment = _source()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_partition(source_dir / "features.parquet", features)
    write_partition(source_dir / "targets.parquet", targets)
    write_partition(source_dir / "assignments.parquet", assignment)
    schema = feature_schema_from_frame(
        features,
        {"category": ["a:b", "quoted'\"", r"slash\\", "", "Δ"]},
    )
    cache = TransformationCache(tmp_path / "cache")
    for view_id in VIEW_IDS:
        transformation = get_transformation(view_id, config.model_dump())
        train = features.iloc[:6].reset_index(drop=True)
        transformation.fit(train, "e2e", 1729, schema)
        staging = tmp_path / f"staging-{view_id}"
        promoted = tmp_path / f"promoted-{view_id}"
        staging.mkdir()
        partition_feature_paths: dict[str, str] = {}
        certificate_paths: dict[str, str] = {}
        source_hashes: dict[str, str] = {}
        target_hashes: dict[str, str] = {}
        transformed_hashes: dict[str, str] = {}
        certificate_hashes: dict[str, str] = {}
        row_order_hashes: dict[str, str] = {}
        certificates = {}
        for partition in ("train", "calibration", "test"):
            ids = assignment.loc[assignment["partition"] == partition, "__sg_row_id"]
            source = features.set_index("__sg_row_id").loc[ids].reset_index()
            target = targets.set_index("__sg_row_id").loc[ids].reset_index()
            transformed = transformation.transform(source, partition)
            target_hash = hash_dataframe_logically(target)
            certificate = transformation.certificate_for(
                source,
                transformed,
                partition,
                dataset_version="e2e",
                source_target_hash=target_hash,
                output_target_hash=target_hash,
            )
            restored = transformation.reconstruct(transformed, certificate)
            validate_transformation(
                source, transformed, restored, certificate, transformation=transformation
            )
            feature_name = f"{partition}.parquet"
            certificate_name = f"{partition}.json"
            write_partition(staging / feature_name, transformed)
            write_certificate(staging / certificate_name, certificate)
            partition_feature_paths[partition] = feature_name
            certificate_paths[partition] = certificate_name
            source_hashes[partition] = hash_dataframe_logically(source)
            target_hashes[partition] = target_hash
            transformed_hashes[partition] = sha256_file(staging / feature_name)
            certificate_hashes[partition] = sha256_file(staging / certificate_name)
            row_order_hashes[partition] = hash_dataframe_logically(transformed[["__sg_row_id"]])
            certificates[partition] = certificate
            key = transformation_cache_key(
                dataset_id="e2e",
                dataset_version="e2e",
                feature_hash=source_hashes[partition],
                target_hash=target_hash,
                split_logical_hash=hash_dataframe_logically(assignment),
                partition=partition,
                view_id=view_id,
                view_config={"view_id": view_id, "config": config.model_dump()},
                implementation_hash=transformation.implementation_hash(),
                fit_parameter_hash=certificate_parameter_hash(certificate),
                certificate_schema_version=certificate.schema_version,
                python_major_minor="3.12",
            )
            cache.publish(
                key,
                staging / feature_name,
                staging / certificate_name,
                {
                    "dataset_id": "e2e",
                    "dataset_version": "e2e",
                    "seed": 1729,
                    "source_feature_hash": source_hashes[partition],
                    "target_hash": target_hash,
                    "split_logical_hash": hash_dataframe_logically(assignment),
                    "partition": partition,
                    "view_id": view_id,
                    "view_configuration_hash": transformation.configuration_hash(),
                    "implementation_hash": transformation.implementation_hash(),
                    "fit_parameter_hash": certificate_parameter_hash(certificate),
                    "certificate_schema_version": certificate.schema_version,
                    "python_major_minor": "3.12",
                },
            )
            assert cache.read_validated(key) is not None
            changed_key = transformation_cache_key(
                dataset_id="e2e",
                dataset_version="e2e",
                feature_hash=source_hashes[partition],
                target_hash=target_hash,
                split_logical_hash=hash_dataframe_logically(assignment),
                partition=partition,
                view_id=view_id,
                view_config={"view_id": view_id, "config": config.model_dump()},
                implementation_hash="f" * 64,
                fit_parameter_hash=certificate_parameter_hash(certificate),
                certificate_schema_version=certificate.schema_version,
                python_major_minor="3.12",
            )
            assert cache.read_validated(changed_key) is None
        manifest = TransformationManifest(
            schema_version=2,
            dataset_id="e2e",
            dataset_version="e2e",
            seed=1729,
            split_strategy=config.split_strategy,
            view_id=view_id,
            view_name=next(item["name"] for item in config.views if item["id"] == view_id),
            source_feature_path=str((source_dir / "features.parquet").resolve()),
            target_reference_path=str((source_dir / "targets.parquet").resolve()),
            split_assignment_path=str((source_dir / "assignments.parquet").resolve()),
            partition_feature_paths=partition_feature_paths,
            certificate_paths=certificate_paths,
            source_hashes=source_hashes,
            target_hashes=target_hashes,
            source_feature_hash=hash_dataframe_logically(features),
            source_target_hash=hash_dataframe_logically(targets),
            split_assignment_hash=hash_dataframe_logically(assignment),
            transformed_feature_hashes=transformed_hashes,
            certificate_hashes=certificate_hashes,
            partition_row_order_hashes=row_order_hashes,
            configuration_hash=transformation.configuration_hash(),
            implementation_hash=transformation.implementation_hash(),
            cache_identity=sha256_canonical_json(
                {
                    "dataset_id": "e2e",
                    "seed": 1729,
                    "view_id": view_id,
                    "source_feature_hash": hash_dataframe_logically(features),
                    "source_target_hash": hash_dataframe_logically(targets),
                    "split_assignment_hash": hash_dataframe_logically(assignment),
                    "configuration_hash": transformation.configuration_hash(),
                    "implementation_hash": transformation.implementation_hash(),
                }
            ),
            manifest_status="PASS",
            created_at=certificates["train"].created_at,
        )
        write_manifest_last(staging / "manifest.json", manifest)
        validate_manifest_directory(staging, project_root=Path.cwd())
        os.replace(staging, promoted)
        validate_manifest_directory(promoted, project_root=Path.cwd())
