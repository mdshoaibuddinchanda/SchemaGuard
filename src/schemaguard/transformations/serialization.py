"""Atomic transformation artifact serialization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.io import atomic_write_json, atomic_write_parquet, read_json_validated
from .contracts import TransformationCertificate, TransformationManifest


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


def validate_manifest_directory(directory: str | Path) -> TransformationManifest:
    root = Path(directory)
    manifest = read_manifest(root / "manifest.json")
    for relative in list(manifest.partition_feature_paths.values()) + list(
        manifest.certificate_paths.values()
    ):
        if not (root / relative).is_file():
            raise ValueError(f"manifest references missing artifact: {relative}")
    return manifest
