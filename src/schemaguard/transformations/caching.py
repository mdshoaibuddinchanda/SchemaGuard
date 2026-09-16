"""Content-addressed transformation cache with fail-closed identity checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..utils.hashing import sha256_canonical_json
from ..utils.io import atomic_write_bytes, atomic_write_json
from ..utils.process_lock import ProcessLock

CACHE_FILES = {"features.parquet", "certificate.json", "manifest.json"}


def canonical_view_configuration_hash(view_config: Mapping[str, Any]) -> str:
    """Hash the exact canonical transformation configuration used by a view."""

    return sha256_canonical_json(dict(view_config))


def transformation_cache_key(
    *,
    dataset_id: int | str,
    dataset_version: str,
    feature_hash: str,
    target_hash: str,
    split_logical_hash: str,
    partition: str,
    view_id: str,
    view_config: Mapping[str, Any] | None = None,
    view_configuration_hash: str | None = None,
    implementation_hash: str,
    fit_parameter_hash: str,
    certificate_schema_version: int,
    python_major_minor: str = "3.12",
) -> str:
    """Return the one canonical key used by cache publication and reads."""

    if view_config is not None:
        computed_view_hash = canonical_view_configuration_hash(view_config)
        if view_configuration_hash is not None and view_configuration_hash != computed_view_hash:
            raise ValueError("view configuration hash does not match the supplied configuration")
        view_configuration_hash = computed_view_hash
    if view_configuration_hash is None:
        raise ValueError("a canonical view configuration hash is required")
    return sha256_canonical_json(
        {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "source_feature_hash": feature_hash,
            "target_hash": target_hash,
            "split_logical_hash": split_logical_hash,
            "partition": partition,
            "view_id": view_id,
            "view_configuration_hash": view_configuration_hash,
            "implementation_hash": implementation_hash,
            "fit_parameter_hash": fit_parameter_hash,
            "certificate_schema_version": certificate_schema_version,
            "python_major_minor": python_major_minor,
        }
    )


def cache_key_from_manifest(manifest: Any) -> str:
    """Reconstruct a cache key from a strict persisted cache manifest."""

    return transformation_cache_key(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        feature_hash=manifest.source_feature_hash,
        target_hash=manifest.target_hash,
        split_logical_hash=manifest.split_logical_hash,
        partition=manifest.partition,
        view_id=manifest.view_id,
        view_configuration_hash=manifest.view_configuration_hash,
        implementation_hash=manifest.implementation_hash,
        fit_parameter_hash=manifest.fit_parameter_hash,
        certificate_schema_version=manifest.certificate_schema_version,
        python_major_minor=manifest.python_major_minor,
    )


class TransformationCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def paths(self, key: str) -> tuple[Path, Path, Path]:
        directory = self.root / key[:2] / key
        return (
            directory / "features.parquet",
            directory / "certificate.json",
            directory / "manifest.json",
        )

    def lock_path(self, key: str) -> Path:
        return self.root / "locks" / f"{key}.lock"

    @staticmethod
    def _actual_files(directory: Path) -> set[str]:
        return {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }

    def read_validated(self, key: str) -> dict[str, Any] | None:
        """Return a hit only after recomputing every persisted identity relationship."""

        features, certificate, manifest_path = self.paths(key)
        directory = manifest_path.parent
        if not directory.is_dir() or self._actual_files(directory) != CACHE_FILES:
            return None
        try:
            from .certificates import certificate_identity_hash
            from .contracts import TransformationCacheManifest, TransformationCertificate

            manifest = TransformationCacheManifest.model_validate(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if manifest.cache_key != key or directory.name != key:
                return None
            if cache_key_from_manifest(manifest) != key:
                return None
            if manifest.feature_sha256 != _file_sha256(features):
                return None
            if manifest.certificate_sha256 != _file_sha256(certificate):
                return None
            parsed = TransformationCertificate.model_validate(
                json.loads(certificate.read_text(encoding="utf-8"))
            )
            if parsed.validation_status != "PASS":
                return None
            if parsed.schema_version != manifest.certificate_schema_version:
                return None
            if manifest.certificate_identity != certificate_identity_hash(parsed):
                return None
            if manifest.output_artifact_hash != parsed.output_artifact_hash:
                return None
            if manifest.source_artifact_hash != parsed.source_artifact_hash:
                return None
            expected = {
                "dataset_id": parsed.dataset_id,
                "dataset_version": parsed.dataset_version,
                "seed": parsed.seed,
                "source_feature_hash": parsed.source_artifact_hash,
                "target_hash": parsed.source_target_hash,
                "partition": parsed.partition,
                "view_id": parsed.view_id,
                "view_configuration_hash": parsed.configuration_hash,
                "implementation_hash": parsed.implementation_hash,
                "fit_parameter_hash": certificate_parameter_hash(parsed),
            }
            if any(getattr(manifest, name) != value for name, value in expected.items()):
                return None
            import pandas as pd

            frame = pd.read_parquet(features)
            if parsed.output_artifact_hash != sha256_dataframe(frame):
                return None
            return manifest.canonical_dict()
        except Exception:
            return None

    def publish(
        self, key: str, feature_path: Path, certificate_path: Path, metadata: Mapping[str, Any]
    ) -> None:
        """Validate and atomically publish one cache entry under its identity lock."""

        if not feature_path.is_file() or not certificate_path.is_file():
            raise ValueError("cache publication requires complete feature and certificate files")
        with ProcessLock(self.lock_path(key), timeout=120):
            from .certificates import certificate_identity_hash
            from .contracts import TransformationCacheManifest, TransformationCertificate

            parsed = TransformationCertificate.model_validate(
                json.loads(certificate_path.read_text(encoding="utf-8"))
            )
            if parsed.validation_status != "PASS":
                raise ValueError("cache publication requires a passing certificate")
            import pandas as pd

            if parsed.output_artifact_hash != sha256_dataframe(pd.read_parquet(feature_path)):
                raise ValueError("cache publication feature hash does not match certificate")
            fit_parameter_hash = certificate_parameter_hash(parsed)
            manifest_data = {
                **dict(metadata),
                "schema_version": 1,
                "cache_key": key,
                "feature_sha256": _file_sha256(feature_path),
                "certificate_sha256": _file_sha256(certificate_path),
                "certificate_identity": certificate_identity_hash(parsed),
                "source_artifact_hash": parsed.source_artifact_hash,
                "output_artifact_hash": parsed.output_artifact_hash,
                "fit_parameter_hash": fit_parameter_hash,
            }
            manifest = TransformationCacheManifest.model_validate(manifest_data)
            if cache_key_from_manifest(manifest) != key:
                raise ValueError("supplied cache key differs from the recomputed cache key")
            expected = {
                "dataset_id": parsed.dataset_id,
                "dataset_version": parsed.dataset_version,
                "seed": parsed.seed,
                "source_feature_hash": parsed.source_artifact_hash,
                "target_hash": parsed.source_target_hash,
                "partition": parsed.partition,
                "view_id": parsed.view_id,
                "view_configuration_hash": parsed.configuration_hash,
                "implementation_hash": parsed.implementation_hash,
                "certificate_schema_version": parsed.schema_version,
            }
            if any(getattr(manifest, name) != value for name, value in expected.items()):
                raise ValueError("cache manifest identity does not match its certificate")
            destination = self.paths(key)[0].parent
            existing = self._actual_files(destination) if destination.is_dir() else set()
            if existing - CACHE_FILES:
                raise ValueError("cache directory contains unexpected files")
            destination.mkdir(parents=True, exist_ok=True)
            feature_destination, certificate_destination, manifest_destination = self.paths(key)
            if feature_path.resolve() != feature_destination.resolve():
                atomic_write_bytes(feature_destination, feature_path.read_bytes())
            if certificate_path.resolve() != certificate_destination.resolve():
                atomic_write_bytes(certificate_destination, certificate_path.read_bytes())
            # The manifest is written last and is itself an atomic replacement.
            atomic_write_json(manifest_destination, manifest.canonical_dict())


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe(frame: Any) -> str:
    from ..utils.hashing import hash_dataframe_logically

    return hash_dataframe_logically(frame.reset_index(drop=True))


def certificate_parameter_hash(certificate: Any) -> str:
    parameters = {
        key: value for key, value in certificate.parameters.items() if not key.startswith("_")
    }
    return sha256_canonical_json(parameters)


__all__ = [
    "TransformationCache",
    "cache_key_from_manifest",
    "canonical_view_configuration_hash",
    "certificate_parameter_hash",
    "sha256_dataframe",
    "transformation_cache_key",
]
