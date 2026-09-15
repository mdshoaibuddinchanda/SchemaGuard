"""Atomic transformation artifact serialization and content validation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_json, atomic_write_parquet, read_json_validated
from .base import feature_schema_from_frame
from .contracts import TransformationCertificate, TransformationManifest
from .registry import get_transformation, load_transformation_config
from .validation import validate_transformation


def write_certificate(path: str | Path, certificate: TransformationCertificate) -> None:
    atomic_write_json(path, certificate.canonical_dict())


def read_certificate(path: str | Path) -> TransformationCertificate:
    return read_json_validated(path, TransformationCertificate)


def write_partition(path: str | Path, frame: pd.DataFrame) -> None:
    atomic_write_parquet(path, frame, compression="zstd", compression_level=3)


def write_manifest_last(path: str | Path, manifest: TransformationManifest) -> None:
    """Manifest publication is last so a visible directory is never half-certified."""

    atomic_write_json(path, manifest.canonical_dict())


def read_manifest(path: str | Path) -> TransformationManifest:
    return read_json_validated(path, TransformationManifest)


def _project_root(directory: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    candidates = [Path.cwd(), *directory.resolve().parents]
    for candidate in candidates:
        if (candidate / "configs/runtime/transformation_engine.yaml").is_file():
            return candidate
    return Path.cwd()


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _schema_categories(path: Path) -> dict[str, list[object]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["name"]: list(ast.literal_eval(item["raw_type"]))
        for item in payload.get("attributes", [])
        if item.get("kind") == "categorical" and str(item.get("raw_type", "")).startswith("[")
    }


def _partition_sources(
    root: Path, manifest: TransformationManifest
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    features = pd.read_parquet(_resolve(root, manifest.source_feature_path))
    targets = pd.read_parquet(_resolve(root, manifest.target_reference_path))
    assignment = pd.read_parquet(_resolve(root, manifest.split_assignment_path))
    partition_column = "partition" if "partition" in assignment.columns else "split"
    feature_index = features.set_index("__sg_row_id", drop=False)
    target_index = targets.set_index("__sg_row_id", drop=False)
    return {
        partition: (
            feature_index.loc[
                assignment.loc[assignment[partition_column] == partition, "__sg_row_id"].tolist()
            ].reset_index(drop=True),
            target_index.loc[
                assignment.loc[assignment[partition_column] == partition, "__sg_row_id"].tolist()
            ].reset_index(drop=True),
        )
        for partition in ("train", "calibration", "test")
    }


def validate_manifest_directory(
    directory: str | Path, *, project_root: str | Path | None = None
) -> TransformationManifest:
    """Reload every materialized file and prove the manifest relationships."""

    root_directory = Path(directory)
    manifest = read_manifest(root_directory / "manifest.json")
    expected_files = {
        "manifest.json",
        *manifest.partition_feature_paths.values(),
        *manifest.certificate_paths.values(),
    }
    actual_files = {
        path.relative_to(root_directory).as_posix()
        for path in root_directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(f"manifest directory contains unexpected or missing files: {actual_files}")
    project = _project_root(root_directory, project_root)
    sources = _partition_sources(project, manifest)
    source_features = pd.read_parquet(_resolve(project, manifest.source_feature_path))
    source_targets = pd.read_parquet(_resolve(project, manifest.target_reference_path))
    assignment = pd.read_parquet(_resolve(project, manifest.split_assignment_path))
    if hash_dataframe_logically(source_features) != manifest.source_feature_hash:
        raise ValueError("manifest source feature hash is invalid")
    if hash_dataframe_logically(source_targets) != manifest.source_target_hash:
        raise ValueError("manifest source target hash is invalid")
    if hash_dataframe_logically(assignment) != manifest.split_assignment_hash:
        raise ValueError("manifest split assignment hash is invalid")
    expected_cache_identity = sha256_canonical_json(
        {
            "dataset_id": manifest.dataset_id,
            "seed": manifest.seed,
            "view_id": manifest.view_id,
            "source_feature_hash": manifest.source_feature_hash,
            "source_target_hash": manifest.source_target_hash,
            "split_assignment_hash": manifest.split_assignment_hash,
            "configuration_hash": manifest.configuration_hash,
            "implementation_hash": manifest.implementation_hash,
        }
    )
    if manifest.cache_identity != expected_cache_identity:
        raise ValueError("manifest cache identity does not match its source identity")
    config = load_transformation_config(project / "configs/runtime/transformation_engine.yaml")
    schema = feature_schema_from_frame(
        source_features,
        _schema_categories(
            _resolve(project, manifest.source_feature_path).with_name("schema.json")
        ),
    )
    transformation = get_transformation(manifest.view_id, config.model_dump())
    train, _ = sources["train"]
    transformation.fit(train, manifest.dataset_id, manifest.seed, schema)
    for partition, (source, target) in sources.items():
        feature_path = root_directory / manifest.partition_feature_paths[partition]
        certificate_path = root_directory / manifest.certificate_paths[partition]
        transformed = pd.read_parquet(feature_path)
        certificate = read_certificate(certificate_path)
        if sha256_file(feature_path) != manifest.transformed_feature_hashes[partition]:
            raise ValueError(f"transformed feature file hash is invalid for {partition}")
        if sha256_file(certificate_path) != manifest.certificate_hashes[partition]:
            raise ValueError(f"certificate file hash is invalid for {partition}")
        if manifest.source_hashes[partition] != hash_dataframe_logically(source):
            raise ValueError(f"source hash is invalid for {partition}")
        if manifest.target_hashes[partition] != hash_dataframe_logically(target):
            raise ValueError(f"target hash is invalid for {partition}")
        if manifest.partition_row_order_hashes[partition] != hash_dataframe_logically(
            transformed[["__sg_row_id"]]
        ):
            raise ValueError(f"partition row-order hash is invalid for {partition}")
        if certificate.configuration_hash != manifest.configuration_hash:
            raise ValueError("manifest configuration hash does not match certificate")
        if certificate.implementation_hash != manifest.implementation_hash:
            raise ValueError("manifest implementation hash does not match certificate")
        restored = transformation.reconstruct(transformed, certificate)
        validate_transformation(
            source,
            transformed,
            restored,
            certificate,
            transformation=transformation,
        )
    return manifest


__all__ = [
    "read_certificate",
    "read_manifest",
    "validate_manifest_directory",
    "write_certificate",
    "write_manifest_last",
    "write_partition",
]
