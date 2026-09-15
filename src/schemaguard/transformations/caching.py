"""Content-addressed transformation cache with validation-before-publish."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.hashing import sha256_canonical_json
from ..utils.io import atomic_write_bytes, atomic_write_json
from ..utils.process_lock import ProcessLock


def transformation_cache_key(
    *,
    dataset_id: int | str,
    dataset_version: str,
    feature_hash: str,
    target_hash: str,
    split_logical_hash: str,
    partition: str,
    view_id: str,
    view_config: dict[str, Any],
    implementation_hash: str,
    fit_parameter_hash: str,
    certificate_schema_version: int,
    python_major_minor: str = "3.12",
) -> str:
    return sha256_canonical_json(
        {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "feature_hash": feature_hash,
            "target_hash": target_hash,
            "split_logical_hash": split_logical_hash,
            "partition": partition,
            "view_id": view_id,
            "view_config": view_config,
            "implementation_hash": implementation_hash,
            "fit_parameter_hash": fit_parameter_hash,
            "certificate_schema_version": certificate_schema_version,
            "python_major_minor": python_major_minor,
        }
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

    def read_validated(self, key: str) -> dict[str, Any] | None:
        features, certificate, manifest = self.paths(key)
        if not (features.is_file() and certificate.is_file() and manifest.is_file()):
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("cache_key") != key:
                return None
            if payload.get("feature_sha256") != _file_sha256(features):
                return None
            if payload.get("certificate_sha256") != _file_sha256(certificate):
                return None
            from .certificates import certificate_hash
            from .contracts import TransformationCertificate

            parsed = TransformationCertificate.model_validate(
                json.loads(certificate.read_text(encoding="utf-8"))
            )
            import pandas as pd

            frame = pd.read_parquet(features)
            if parsed.validation_status != "PASS":
                return None
            if parsed.output_artifact_hash != sha256_dataframe(frame):
                return None
            if payload.get("certificate_identity") != certificate_hash(parsed):
                return None
            if payload.get("output_artifact_hash") != parsed.output_artifact_hash:
                return None
            if payload.get("source_artifact_hash") != parsed.source_artifact_hash:
                return None
            expected = {
                "dataset_id": parsed.dataset_id,
                "dataset_version": parsed.dataset_version,
                "seed": parsed.seed,
                "partition": parsed.partition,
                "view_id": parsed.view_id,
                "target_hash": parsed.source_target_hash,
                "feature_hash": parsed.source_artifact_hash,
                "configuration_hash": parsed.configuration_hash,
                "implementation_hash": parsed.implementation_hash,
                "certificate_schema_version": parsed.schema_version,
                "fit_parameter_hash": _certificate_parameter_hash(parsed),
            }
            if any(payload.get(name) != value for name, value in expected.items()):
                return None
            return payload
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def publish(
        self, key: str, feature_path: Path, certificate_path: Path, metadata: dict[str, Any]
    ) -> None:
        destination = self.paths(key)[0].parent
        destination.mkdir(parents=True, exist_ok=True)
        with ProcessLock(self.lock_path(key), timeout=120):
            from .certificates import certificate_hash
            from .contracts import TransformationCertificate

            parsed = TransformationCertificate.model_validate(
                json.loads(certificate_path.read_text(encoding="utf-8"))
            )
            import pandas as pd

            if parsed.output_artifact_hash != sha256_dataframe(pd.read_parquet(feature_path)):
                raise ValueError("cache publication feature hash does not match certificate")
            manifest = {
                **metadata,
                "schema_version": 1,
                "cache_key": key,
                "feature_sha256": _file_sha256(feature_path),
                "certificate_sha256": _file_sha256(certificate_path),
                "dataset_id": parsed.dataset_id,
                "dataset_version": parsed.dataset_version,
                "seed": parsed.seed,
                "partition": parsed.partition,
                "view_id": parsed.view_id,
                "target_hash": parsed.source_target_hash,
                "feature_hash": parsed.source_artifact_hash,
                "configuration_hash": parsed.configuration_hash,
                "implementation_hash": parsed.implementation_hash,
                "certificate_schema_version": parsed.schema_version,
                "fit_parameter_hash": _certificate_parameter_hash(parsed),
            }
            manifest["certificate_identity"] = certificate_hash(parsed)
            manifest["source_artifact_hash"] = parsed.source_artifact_hash
            manifest["output_artifact_hash"] = parsed.output_artifact_hash
            feature_destination, certificate_destination, manifest_destination = self.paths(key)
            if feature_path.resolve() != feature_destination.resolve():
                atomic_write_bytes(feature_destination, feature_path.read_bytes())
            if certificate_path.resolve() != certificate_destination.resolve():
                atomic_write_bytes(certificate_destination, certificate_path.read_bytes())
            atomic_write_json(manifest_destination, manifest)


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


def _certificate_parameter_hash(certificate: Any) -> str:
    parameters = {
        key: value
        for key, value in certificate.parameters.items()
        if not key.startswith("_")
    }
    return sha256_canonical_json(parameters)
