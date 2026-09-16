from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from schemaguard.transformations.caching import (
    TransformationCache,
    certificate_parameter_hash,
    transformation_cache_key,
)
from schemaguard.transformations.identity import IdentityTransformation
from schemaguard.transformations.serialization import write_certificate, write_partition
from schemaguard.utils.hashing import hash_dataframe_logically

from .transformation_helpers import sample_frame, sample_schema, target_hash, transformation_config


def _published(tmp_path: Path):
    source = sample_frame()
    transformation = IdentityTransformation(transformation_config()).fit(
        source, "cache", 1729, sample_schema(source)
    )
    output = transformation.transform(source, "test")
    certificate = transformation.certificate_for(
        source,
        output,
        "test",
        dataset_version="v1",
        source_target_hash=target_hash(source),
        output_target_hash=target_hash(source),
    )
    feature_path = tmp_path / "staging.parquet"
    certificate_path = tmp_path / "staging.json"
    write_partition(feature_path, output)
    write_certificate(certificate_path, certificate)
    source_hash = hash_dataframe_logically(source)
    split_hash = "c" * 64
    metadata = {
        "dataset_id": "cache",
        "dataset_version": "v1",
        "seed": 1729,
        "source_feature_hash": source_hash,
        "target_hash": target_hash(source),
        "split_logical_hash": split_hash,
        "partition": "test",
        "view_id": "V00",
        "view_configuration_hash": transformation.configuration_hash(),
        "implementation_hash": transformation.implementation_hash(),
        "fit_parameter_hash": certificate_parameter_hash(certificate),
        "certificate_schema_version": certificate.schema_version,
        "python_major_minor": "3.12",
    }
    key = transformation_cache_key(
        dataset_id=metadata["dataset_id"],
        dataset_version=metadata["dataset_version"],
        feature_hash=metadata["source_feature_hash"],
        target_hash=metadata["target_hash"],
        split_logical_hash=metadata["split_logical_hash"],
        partition=metadata["partition"],
        view_id=metadata["view_id"],
        view_configuration_hash=metadata["view_configuration_hash"],
        implementation_hash=metadata["implementation_hash"],
        fit_parameter_hash=metadata["fit_parameter_hash"],
        certificate_schema_version=metadata["certificate_schema_version"],
        python_major_minor=metadata["python_major_minor"],
    )
    cache = TransformationCache(tmp_path / "cache")
    cache.publish(key, feature_path, certificate_path, metadata)
    assert cache.read_validated(key) is not None
    return cache, key


def _manifest(cache: TransformationCache, key: str) -> tuple[Path, dict]:
    _, _, path = cache.paths(key)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_copy_under_different_key_and_manifest_key_tampering_are_rejected(tmp_path: Path) -> None:
    cache, key = _published(tmp_path)
    changed_key = "f" * 64
    source_directory = cache.paths(key)[0].parent
    target_directory = cache.paths(changed_key)[0].parent
    target_directory.mkdir(parents=True)
    shutil.copytree(source_directory, target_directory / changed_key)
    manifest_path = target_directory / changed_key / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["cache_key"] = changed_key
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.read_validated(changed_key) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split_logical_hash", "d" * 64),
        ("view_configuration_hash", "e" * 64),
        ("python_major_minor", "3.13"),
        ("fit_parameter_hash", "f" * 64),
        ("implementation_hash", "0" * 64),
        ("certificate_identity", "1" * 64),
        ("dataset_id", "different"),
    ],
)
def test_identity_field_tampering_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    cache, key = _published(tmp_path)
    path, payload = _manifest(cache, key)
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.read_validated(key) is None


def test_certificate_and_feature_bytes_are_revalidated(tmp_path: Path) -> None:
    cache, key = _published(tmp_path)
    features, certificate, _ = cache.paths(key)
    features.write_bytes(features.read_bytes() + b"tampered")
    assert cache.read_validated(key) is None
    cache, key = _published(tmp_path / "second")
    _, certificate, _ = cache.paths(key)
    certificate.write_text(certificate.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert cache.read_validated(key) is None


def test_unexpected_cache_file_is_rejected(tmp_path: Path) -> None:
    cache, key = _published(tmp_path)
    unexpected = cache.paths(key)[0].parent / "unexpected.tmp"
    unexpected.write_bytes(b"partial")
    assert cache.read_validated(key) is None
