"""Restart-safe SchemaOrbit-14 acquisition and validation.

This module is deliberately separate from the frozen Phase 01 smoke pipeline.
It never writes the existing 1464 artifacts and does not create split files.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_json, atomic_write_parquet
from ..utils.process_lock import ProcessLock
from .arff_parser import ParsedArff, parse_arff
from .splits import predictor_group_ids

SCHEMA_VERSION = 1
SMOKE_DATASET_ID = 1464


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    file_id: int
    version: str
    rows: int = Field(gt=0)
    predictors: int = Field(gt=0)
    target: str
    provider_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    ignore_attributes: list[str] = Field(default_factory=list)


class AcquisitionRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    offline_default: bool
    require_version: str
    require_public: bool
    require_active: bool
    require_default_target: bool
    require_stable_checksum: bool
    min_rows: int = Field(gt=0)
    max_rows: int = Field(gt=0)
    min_predictors: int = Field(gt=0)
    max_predictors: int = Field(gt=0)
    min_classes: int = Field(gt=1)
    max_classes: int = Field(gt=1)
    parquet_compression: str
    download_attempts: int = Field(ge=1, le=3)
    request_timeout_seconds: float = Field(gt=0)
    retry_backoff_seconds: list[float]

    @staticmethod
    def _validate_backoff(values: list[float], attempts: int) -> list[float]:
        if len(values) != attempts or any(value < 0 for value in values):
            raise ValueError("retry_backoff_seconds must contain one nonnegative value per attempt")
        return values

    @model_validator(mode="after")
    def validate_backoff(self) -> AcquisitionRules:
        self._validate_backoff(self.retry_backoff_seconds, self.download_attempts)
        return self


class SchemaOrbitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    benchmark: str
    openml_api_base: str
    download_base: str
    acquisition: AcquisitionRules
    datasets: list[DatasetSpec]


class SchemaOrbitError(RuntimeError):
    """A dataset cannot be accepted as a reproducible SchemaOrbit input."""


class OfflineCacheMiss(SchemaOrbitError):
    """Offline execution found no validated local source."""


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_id: int
    file_id: int
    name: str
    version: str
    format: str
    default_target_attribute: str | None
    status: str
    visibility: str
    licence: str | None
    md5_checksum: str | None
    metadata_url: str
    download_url: str
    raw: dict[str, Any]


class FeatureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    data_type: str
    is_target: bool
    is_ignore: bool
    is_row_identifier: bool


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise SchemaOrbitError("SchemaOrbit configuration must be a mapping")
    return value


def load_schemaorbit_config(path: str | Path) -> SchemaOrbitConfig:
    """Load the closed SchemaOrbit registry and reject unknown keys."""

    try:
        config = SchemaOrbitConfig.model_validate(_load_yaml(Path(path)))
    except ValidationError as exc:
        raise SchemaOrbitError(str(exc)) from exc
    if config.schema_version != SCHEMA_VERSION or config.benchmark != "SchemaOrbit-14":
        raise SchemaOrbitError("Unsupported SchemaOrbit configuration")
    if len(config.datasets) != 14 or len({item.id for item in config.datasets}) != 14:
        raise SchemaOrbitError("SchemaOrbit-14 must contain exactly fourteen unique datasets")
    if not (config.acquisition.min_rows <= config.acquisition.max_rows):
        raise SchemaOrbitError("Invalid row constraint")
    if not (config.acquisition.min_predictors <= config.acquisition.max_predictors):
        raise SchemaOrbitError("Invalid predictor constraint")
    if not (config.acquisition.min_classes <= config.acquisition.max_classes):
        raise SchemaOrbitError("Invalid class constraint")
    return config


def _package_versions() -> dict[str, str]:
    names = ("httpx", "liac-arff", "pandas", "pyarrow", "pydantic", "numpy")
    return {name: importlib.metadata.version(name) for name in names if _installed(name)}


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _metadata_from_payload(
    payload: dict[str, Any], spec: DatasetSpec, api_url: str
) -> DatasetMetadata:
    raw = payload.get("data_set_description")
    if not isinstance(raw, dict):
        raise SchemaOrbitError(f"OpenML metadata for {spec.id} is not a data_set_description")
    try:
        return DatasetMetadata(
            data_id=int(raw["id"]),
            file_id=int(raw["file_id"]),
            name=str(raw["name"]),
            version=str(raw["version"]),
            format=str(raw["format"]),
            default_target_attribute=(
                str(raw["default_target_attribute"])
                if raw.get("default_target_attribute") is not None
                else None
            ),
            status=str(raw["status"]),
            visibility=str(raw["visibility"]),
            licence=str(raw["licence"]) if raw.get("licence") is not None else None,
            md5_checksum=(str(raw["md5_checksum"]).lower() if raw.get("md5_checksum") else None),
            metadata_url=api_url,
            download_url=str(
                raw.get("url")
                or f"https://www.openml.org/data/v1/download/{spec.file_id}/{spec.name}.arff"
            ),
            raw=raw,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaOrbitError(f"Malformed OpenML metadata for {spec.id}: {exc}") from exc


def fetch_metadata(
    client: httpx.Client, config: SchemaOrbitConfig, spec: DatasetSpec
) -> tuple[DatasetMetadata, list[FeatureInfo]]:
    metadata_url = f"{config.openml_api_base}/data/{spec.id}"
    response = _request_with_retry(client, "GET", metadata_url, config)
    response.raise_for_status()
    metadata = _metadata_from_payload(response.json(), spec, metadata_url)
    features_url = f"{config.openml_api_base}/data/features/{spec.id}"
    features_response = _request_with_retry(client, "GET", features_url, config)
    features_response.raise_for_status()
    feature_payload = features_response.json()
    raw_features = feature_payload.get("data_features", {}).get("feature", [])
    if isinstance(raw_features, dict):
        raw_features = [raw_features]
    features = [
        FeatureInfo(
            name=str(item["name"]),
            data_type=str(item.get("data_type", "unknown")),
            is_target=str(item.get("is_target", "false")).lower() == "true",
            is_ignore=str(item.get("is_ignore", "false")).lower() == "true",
            is_row_identifier=str(item.get("is_row_identifier", "false")).lower() == "true",
        )
        for item in raw_features
    ]
    _validate_metadata(config, spec, metadata, features)
    return metadata, features


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _request_with_retry(
    client: httpx.Client, method: str, url: str, config: SchemaOrbitConfig
) -> httpx.Response:
    """Retry only transient transport and server failures with bounded backoff."""

    rules = config.acquisition
    last_error: BaseException | None = None
    for attempt in range(rules.download_attempts):
        try:
            try:
                response = client.request(method, url, timeout=rules.request_timeout_seconds)
            except AttributeError:
                response = client.get(url)
            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt + 1 < rules.download_attempts
            ):
                time.sleep(rules.retry_backoff_seconds[attempt])
                continue
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt + 1 >= rules.download_attempts:
                raise
            time.sleep(rules.retry_backoff_seconds[attempt])
    raise SchemaOrbitError(f"Request failed after bounded retries: {url}") from last_error


def _validate_metadata(
    config: SchemaOrbitConfig,
    spec: DatasetSpec,
    metadata: DatasetMetadata,
    features: list[FeatureInfo],
) -> None:
    rules = config.acquisition
    if metadata.data_id != spec.id or metadata.file_id != spec.file_id:
        raise SchemaOrbitError(f"Dataset identity mismatch for {spec.id}")
    if metadata.name != spec.name or metadata.version != rules.require_version:
        raise SchemaOrbitError(f"Dataset name/version mismatch for {spec.id}")
    if metadata.format.upper() != "ARFF" or metadata.status.lower() != "active":
        raise SchemaOrbitError(f"Dataset {spec.id} is not an active ARFF source")
    if rules.require_public and metadata.visibility.lower() != "public":
        raise SchemaOrbitError(f"Dataset {spec.id} is not public")
    if rules.require_default_target and metadata.default_target_attribute != spec.target:
        raise SchemaOrbitError(f"Default target mismatch for {spec.id}")
    if not metadata.md5_checksum or metadata.md5_checksum != spec.provider_md5:
        raise SchemaOrbitError(f"Provider checksum mismatch for {spec.id}")
    target_names = [item.name for item in features if item.is_target]
    if spec.target not in target_names:
        raise SchemaOrbitError(
            f"Target {spec.target!r} is absent from feature metadata for {spec.id}"
        )
    metadata_ignored = [item.name for item in features if item.is_ignore or item.is_row_identifier]
    if sorted(metadata_ignored) != sorted(spec.ignore_attributes):
        raise SchemaOrbitError(
            f"Ignore-attribute mismatch for {spec.id}: expected {spec.ignore_attributes}, "
            f"observed {metadata_ignored}"
        )
    predictor_count = len(features) - len(target_names) - len(metadata_ignored)
    if predictor_count != spec.predictors:
        raise SchemaOrbitError(f"Predictor count mismatch for {spec.id}: {predictor_count}")
    if not rules.min_predictors <= predictor_count <= rules.max_predictors:
        raise SchemaOrbitError(f"Predictor constraint failed for {spec.id}")


def _stream_digest(path: Path) -> tuple[str, str, int]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def _quarantine(path: Path, root: Path, reason: str) -> str | None:
    if not path.exists():
        return None
    destination = (
        root / "quarantine" / f"{path.name}.{reason}.{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return str(destination)


def _raw_paths(root: Path, spec: DatasetSpec) -> tuple[Path, Path, Path]:
    directory = root / "data" / "raw" / "openml" / str(spec.id)
    return (
        directory / f"{spec.name}.arff",
        directory / "openml_metadata.json",
        directory / "source_manifest.json",
    )


def acquire_raw(
    client: httpx.Client | None,
    root: str | Path,
    config: SchemaOrbitConfig,
    spec: DatasetSpec,
    *,
    offline: bool,
    allow_network: bool,
    refresh: bool = False,
) -> tuple[Path, DatasetMetadata, list[FeatureInfo], dict[str, Any]]:
    """Acquire one source under a per-dataset lock and return validated identity."""

    project_root = Path(root)
    raw_path, metadata_path, manifest_path = _raw_paths(project_root, spec)
    lock_path = project_root / "data" / "cache" / "locks" / f"openml-{spec.id}.lock"
    with ProcessLock(lock_path, timeout=300):
        if spec.id == SMOKE_DATASET_ID:
            metadata = _read_smoke_dataset_metadata(metadata_path, spec)
            features = _read_features_if_present(project_root / "data" / "cache" / "unused.json")
            if not features:
                features = []
            # The smoke dataset is validated independently and is never rewritten here.
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_source_manifest(manifest_path, manifest)
            expected_sha = manifest["computed_sha256"]
            if not raw_path.is_file() or sha256_file(raw_path) != expected_sha:
                raise SchemaOrbitError("Smoke data foundation raw artifact hash changed")
            return raw_path, metadata, features, manifest
        if offline:
            if not raw_path.is_file() or not metadata_path.is_file() or not manifest_path.is_file():
                raise OfflineCacheMiss(f"Offline cache miss for dataset {spec.id}")
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = _metadata_from_payload(
                {"data_set_description": raw_metadata},
                spec,
                f"{config.openml_api_base}/data/{spec.id}",
            )
            feature_payload = json.loads(
                (raw_path.parent / "openml_features.json").read_text(encoding="utf-8")
            )
            features = [FeatureInfo.model_validate(item) for item in feature_payload]
            _validate_metadata(config, spec, metadata, features)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_raw_manifest(raw_path, manifest, spec)
            _validate_source_manifest(manifest_path, manifest)
            return raw_path, metadata, features, manifest
        if client is None or not allow_network:
            raise OfflineCacheMiss(f"Network is disabled for dataset {spec.id}")
        metadata, features = fetch_metadata(client, config, spec)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.is_file() and manifest_path.is_file() and not refresh:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                _validate_raw_manifest(raw_path, manifest, spec)
                _validate_source_manifest(manifest_path, manifest)
                return raw_path, metadata, features, manifest
            except SchemaOrbitError:
                _quarantine(raw_path, project_root / "data" / "cache", "stale")
        partial = raw_path.with_suffix(raw_path.suffix + ".part")
        _quarantine(partial, project_root / "data" / "cache", "partial")
        headers: dict[str, str] = {}
        response_status: int | None = None
        download_attempts = 0
        for attempt in range(config.acquisition.download_attempts):
            download_attempts = attempt + 1
            try:
                with client.stream(
                    "GET", metadata.download_url, timeout=config.acquisition.request_timeout_seconds
                ) as response:
                    response_status = response.status_code
                    if (
                        response.status_code in _RETRYABLE_STATUS_CODES
                        and attempt + 1 < config.acquisition.download_attempts
                    ):
                        time.sleep(config.acquisition.retry_backoff_seconds[attempt])
                        continue
                    response.raise_for_status()
                    headers = dict(response.headers)
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes(1024 * 1024):
                            handle.write(chunk)
                    break
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                if attempt + 1 >= config.acquisition.download_attempts:
                    raise
                time.sleep(config.acquisition.retry_backoff_seconds[attempt])
        else:
            raise SchemaOrbitError(f"Download failed after bounded retries for dataset {spec.id}")
        observed_sha, observed_md5, size = _stream_digest(partial)
        if observed_md5 != spec.provider_md5 or observed_md5 != (metadata.md5_checksum or ""):
            _quarantine(partial, project_root / "data" / "cache", "checksum")
            raise SchemaOrbitError(f"Raw checksum mismatch for dataset {spec.id}")
        os.replace(partial, raw_path)
        manifest = {
            "schema_version": 1,
            "cache_status": "downloaded",
            "provider": "openml",
            "openml_data_id": spec.id,
            "openml_file_id": spec.file_id,
            "dataset_name": spec.name,
            "dataset_version": metadata.version,
            "data_format": metadata.format,
            "default_target_attribute": metadata.default_target_attribute,
            "metadata_url": metadata.metadata_url,
            "requested_download_url": metadata.download_url,
            "resolved_download_url": str(getattr(response, "url", metadata.download_url)),
            "provider_md5": spec.provider_md5,
            "computed_md5": observed_md5,
            "computed_sha256": observed_sha,
            "file_size_bytes": size,
            "http_status": response_status,
            "content_length_bytes": int(headers["content-length"])
            if headers.get("content-length", "").isdigit()
            else size,
            "http_etag": headers.get("etag"),
            "http_last_modified": headers.get("last-modified"),
            "retrieved_at_utc": _utc_now(),
            "raw_relative_path": str(raw_path.relative_to(project_root / "data")),
            "package_versions": _package_versions(),
            "attempts": download_attempts,
        }
        atomic_write_json(metadata_path, metadata.raw)
        atomic_write_json(
            raw_path.parent / "openml_features.json", [item.model_dump() for item in features]
        )
        atomic_write_json(manifest_path, manifest)
        return raw_path, metadata, features, manifest


def _read_smoke_dataset_metadata(path: Path, spec: DatasetSpec) -> DatasetMetadata:
    raw = json.loads(path.read_text(encoding="utf-8"))["data_set_description"]
    return DatasetMetadata(
        data_id=int(raw["id"]),
        file_id=int(raw["file_id"]),
        name=str(raw["name"]),
        version=str(raw["version"]),
        format=str(raw["format"]),
        default_target_attribute=str(raw["default_target_attribute"]),
        status=str(raw["status"]),
        visibility=str(raw["visibility"]),
        licence=str(raw.get("licence")),
        md5_checksum=str(raw.get("md5_checksum")),
        metadata_url=f"https://www.openml.org/api/v1/json/data/{spec.id}",
        download_url=str(raw["url"]),
        raw=raw,
    )


def _read_features_if_present(path: Path) -> list[FeatureInfo]:
    del path
    return []


def _validate_raw_manifest(path: Path, manifest: dict[str, Any], spec: DatasetSpec) -> None:
    if (
        manifest.get("openml_data_id") != spec.id
        or manifest.get("openml_file_id") != spec.file_id
    ):
        raise SchemaOrbitError(f"Raw manifest identity mismatch for {spec.id}")
    if (
        manifest.get("dataset_version") != spec.version
        or manifest.get("computed_md5") != spec.provider_md5
    ):
        raise SchemaOrbitError(f"Raw manifest version/checksum mismatch for {spec.id}")
    expected_sha = manifest.get("computed_sha256")
    if (
        not isinstance(expected_sha, str)
        or not path.is_file()
        or sha256_file(path) != expected_sha
        or _stream_digest(path)[1] != manifest.get("computed_md5")
        or _stream_digest(path)[2] != manifest.get("file_size_bytes")
    ):
        raise SchemaOrbitError(f"Raw artifact checksum mismatch for {spec.id}")


def _validate_source_manifest(path: Path, manifest: dict[str, Any]) -> None:
    from .contracts import RawSourceManifestContract

    try:
        RawSourceManifestContract.model_validate(manifest)
    except ValidationError as exc:
        raise SchemaOrbitError(f"Raw source manifest contract failed: {path}: {exc}") from exc


def _row_id(dataset_id: int, raw_sha: str, position: int) -> str:
    return hashlib.sha256(
        f"openml:{dataset_id}:raw_file_sha256:{raw_sha}:{position}".encode()
    ).hexdigest()[:32]


def _canonical_label(value: Any) -> str:
    if pd.isna(value):
        raise SchemaOrbitError("Missing target value")
    return str(value)


def _normalise_frame(
    parsed: ParsedArff,
    spec: DatasetSpec,
    raw_sha: str,
    feature_info: list[FeatureInfo],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = parsed.frame.copy()
    if spec.target not in frame.columns:
        raise SchemaOrbitError(f"Target {spec.target!r} absent from ARFF for {spec.id}")
    ignored = list(spec.ignore_attributes)
    observed_ignored = [
        item.name for item in feature_info if item.is_ignore or item.is_row_identifier
    ]
    if sorted(ignored) != sorted(observed_ignored):
        raise SchemaOrbitError(f"Ignored-attribute metadata changed for {spec.id}")
    predictors = [
        column
        for column in frame.columns
        if column not in {spec.target, *ignored, "__sg_source_row_position"}
    ]
    if len(predictors) != spec.predictors:
        raise SchemaOrbitError(f"ARFF predictor count mismatch for {spec.id}")
    labels = [_canonical_label(value) for value in frame[spec.target].tolist()]
    classes = sorted(set(labels))
    if not 2 <= len(classes) <= 10:
        raise SchemaOrbitError(f"Class constraint failed for {spec.id}")
    codes = [classes.index(label) for label in labels]
    row_ids = [
        _row_id(spec.id, raw_sha, int(position)) for position in frame["__sg_source_row_position"]
    ]
    if len(row_ids) != len(set(row_ids)):
        raise SchemaOrbitError(f"Row IDs are not unique for {spec.id}")
    features = frame[predictors].copy()
    features.insert(0, "__sg_row_id", row_ids)
    targets = pd.DataFrame(
        {
            "__sg_row_id": row_ids,
            "target_label": pd.Series(labels, dtype="string"),
            "target_code": codes,
        }
    )
    schema_attributes = []
    for position, name in enumerate(predictors):
        attribute = next((item for item in parsed.attributes if item.name == name), None)
        if attribute is None:
            raise SchemaOrbitError(f"Missing parsed schema for {name}")
        schema_attributes.append(
            {
                "name": name,
                "raw_type": attribute.raw_type,
                "kind": attribute.kind,
                "position": position,
            }
        )
    schema = {
        "schema_version": 1,
        "internal_dataset_id": spec.name,
        "openml_data_id": spec.id,
        "target_column": spec.target,
        "row_id_column": "__sg_row_id",
        "feature_columns": predictors,
        "feature_row_count": len(features),
        "target_row_count": len(targets),
        "raw_sha256": raw_sha,
        "attributes": schema_attributes,
    }
    return features, targets, {"classes": classes, "schema": schema}


def _quality(features: pd.DataFrame, targets: pd.DataFrame) -> dict[str, Any]:
    predictors = features.drop(columns=["__sg_row_id"])
    group_ids = predictor_group_ids(features)
    grouped = pd.DataFrame({"group_id": group_ids, "target": targets["target_code"]})
    counts = grouped.groupby("group_id", dropna=False)["target"].agg(["size", "nunique"])
    return {
        "row_count": int(len(features)),
        "predictor_count": int(predictors.shape[1]),
        "class_count": int(targets["target_code"].nunique()),
        "class_counts": {
            str(k): int(v) for k, v in targets["target_code"].value_counts().sort_index().items()
        },
        "missing_cells": int(predictors.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in predictors.isna().sum().items() if v},
        "duplicate_predictor_groups": int((counts["size"] > 1).sum()),
        "conflicting_target_groups": int(((counts["size"] > 1) & (counts["nunique"] > 1)).sum()),
        "exact_duplicate_rows": int(predictors.duplicated(keep=False).sum()),
        "predictor_hash": hash_dataframe_logically(predictors),
        "grouping_algorithm": "typed_predictor_sha256_v1",
    }


def _read_processed(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(directory / "features.parquet"), pd.read_parquet(
        directory / "targets.parquet"
    )


def validate_processed(
    root: str | Path, config: SchemaOrbitConfig, spec: DatasetSpec, raw_sha: str
) -> dict[str, Any]:
    """Validate an existing processed artifact without changing it."""

    project_root = Path(root)
    directory = project_root / "data" / "processed" / "openml" / str(spec.id)
    if spec.id == SMOKE_DATASET_ID:
        # The smoke dataset is frozen evidence from the completed data
        # foundation.  Validate its existing content without rewriting it.
        return _validate_legacy_smoke_processed(directory, config, spec, raw_sha)
    return _validate_processed_directory(project_root, directory, config, spec, raw_sha)


def _validate_legacy_smoke_processed(
    directory: Path, config: SchemaOrbitConfig, spec: DatasetSpec, raw_sha: str
) -> dict[str, Any]:
    """Validate the original smoke artifact using the same grouping algorithm."""
    try:
        features, targets = _read_processed(directory)
    except Exception as exc:
        raise SchemaOrbitError(f"Cannot read processed data for {spec.id}: {exc}") from exc
    if len(features) != spec.rows or len(targets) != spec.rows:
        raise SchemaOrbitError(f"Row count mismatch for {spec.id}")
    if features.columns[0] != "__sg_row_id" or targets.columns.tolist() != [
        "__sg_row_id",
        "target_label",
        "target_code",
    ]:
        raise SchemaOrbitError(f"Processed column contract mismatch for {spec.id}")
    predictor_columns = list(features.columns[1:])
    if len(predictor_columns) != spec.predictors or spec.target in predictor_columns:
        raise SchemaOrbitError(f"Processed predictor contract mismatch for {spec.id}")
    if features["__sg_row_id"].duplicated().any() or targets["__sg_row_id"].duplicated().any():
        raise SchemaOrbitError(f"Duplicate internal row IDs for {spec.id}")
    if set(features["__sg_row_id"]) != set(targets["__sg_row_id"]):
        raise SchemaOrbitError(f"Feature/target row IDs are not a complete join for {spec.id}")
    if not targets["target_code"].isin(range(int(targets["target_code"].nunique()))).all():
        raise SchemaOrbitError(f"Target codes are not contiguous for {spec.id}")
    if targets["target_code"].nunique() < config.acquisition.min_classes:
        raise SchemaOrbitError(f"Class minimum failed for {spec.id}")
    manifest_path = directory / "data_manifest.json"
    if not manifest_path.is_file():
        raise SchemaOrbitError(f"Missing processed manifest for {spec.id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("raw_sha256") != raw_sha:
        raise SchemaOrbitError(f"Processed source hash mismatch for {spec.id}")
    return _quality(features, targets)


def _validate_processed_directory(
    root: Path,
    directory: Path,
    config: SchemaOrbitConfig,
    spec: DatasetSpec,
    raw_sha: str,
) -> dict[str, Any]:
    required = (
        "features.parquet",
        "targets.parquet",
        "schema.json",
        "label_mapping.json",
        "quality_report.json",
        "data_manifest.json",
    )
    if not directory.is_dir() or any(not (directory / name).is_file() for name in required):
        raise SchemaOrbitError(f"Processed artifact set is incomplete for {spec.id}")
    manifest = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
    from .contracts import (
        DatasetQualityReportContract,
        DatasetSchemaContract,
        LabelMappingContract,
        ProcessedManifestContract,
    )

    try:
        manifest_contract = ProcessedManifestContract.model_validate(manifest)
        schema_contract = DatasetSchemaContract.model_validate(
            json.loads((directory / "schema.json").read_text(encoding="utf-8"))
        )
        mapping_contract = LabelMappingContract.model_validate(
            json.loads((directory / "label_mapping.json").read_text(encoding="utf-8"))
        )
        quality_contract = DatasetQualityReportContract.model_validate(
            json.loads((directory / "quality_report.json").read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SchemaOrbitError(
            f"Processed contract validation failed for {spec.id}: {exc}"
        ) from exc
    if (
        manifest_contract.openml_data_id != spec.id
        or manifest_contract.internal_dataset_id != spec.name
    ):
        raise SchemaOrbitError(f"Processed dataset identity mismatch for {spec.id}")
    if manifest_contract.raw_sha256 != raw_sha or schema_contract.raw_sha256 != raw_sha:
        raise SchemaOrbitError(f"Processed source hash mismatch for {spec.id}")
    source_manifest_path = root / "data" / "raw" / "openml" / str(spec.id) / "source_manifest.json"
    expected_source_hash = sha256_file(source_manifest_path)
    if manifest_contract.source_manifest_sha256 != expected_source_hash:
        raise SchemaOrbitError(f"Processed source-manifest identity mismatch for {spec.id}")
    if manifest_contract.processing_config_sha256 != sha256_canonical_json(config.model_dump()):
        raise SchemaOrbitError(f"Processed configuration identity mismatch for {spec.id}")
    observed_hashes = {
        name: sha256_file(directory / name) for name in manifest_contract.artifact_hashes
    }
    if observed_hashes != manifest_contract.artifact_hashes:
        raise SchemaOrbitError(f"Processed artifact hash mismatch for {spec.id}")
    try:
        features, targets = _read_processed(directory)
    except Exception as exc:
        raise SchemaOrbitError(f"Cannot read processed data for {spec.id}: {exc}") from exc
    if len(features) != spec.rows or len(targets) != spec.rows:
        raise SchemaOrbitError(f"Row count mismatch for {spec.id}")
    if list(features.columns) != [
        "__sg_row_id",
        *spec.ignore_attributes,
        *manifest_contract.feature_columns,
    ]:
        if list(features.columns[1:]) != manifest_contract.feature_columns:
            raise SchemaOrbitError(f"Processed feature ordering mismatch for {spec.id}")
    if list(targets.columns) != manifest_contract.target_columns:
        raise SchemaOrbitError(f"Processed target columns mismatch for {spec.id}")
    row_ids = features["__sg_row_id"]
    if (
        row_ids.isna().any()
        or row_ids.duplicated().any()
        or not row_ids.astype(str).str.fullmatch(r"[0-9a-f]{32}").all()
    ):
        raise SchemaOrbitError(
            f"Processed row IDs are not unique, complete, and stable for {spec.id}"
        )
    expected_row_ids = pd.Series(
        [_row_id(spec.id, raw_sha, position) for position in range(len(features))],
        dtype="string",
    )
    if not row_ids.astype("string").reset_index(drop=True).equals(expected_row_ids):
        raise SchemaOrbitError(f"Processed row IDs are not reproducible for {spec.id}")
    if not features["__sg_row_id"].reset_index(drop=True).equals(
        targets["__sg_row_id"].reset_index(drop=True)
    ):
        raise SchemaOrbitError(f"Feature/target row ordering is not identical for {spec.id}")
    if set(targets["target_code"]) != set(range(targets["target_code"].nunique())):
        raise SchemaOrbitError(f"Target codes are not contiguous for {spec.id}")
    mapping = mapping_contract.original_to_code
    if (
        targets["target_label"].map(mapping).astype("Int64").tolist()
        != targets["target_code"].tolist()
    ):
        raise SchemaOrbitError(f"Target mapping is not reversible for {spec.id}")
    expected_quality = {
        "schema_version": quality_contract.schema_version,
        **_quality(features, targets),
    }
    if quality_contract.model_dump(mode="json") != expected_quality:
        raise SchemaOrbitError(f"Quality report does not match processed data for {spec.id}")
    try:
        import pyarrow.parquet as pq

        for name in ("features.parquet", "targets.parquet"):
            parquet = pq.ParquetFile(directory / name)
            compressions = {
                parquet.metadata.row_group(0).column(index).compression
                for index in range(parquet.metadata.num_columns)
            }
            if compressions != {config.acquisition.parquet_compression.upper()}:
                raise SchemaOrbitError(
                    f"Unexpected Parquet compression for {spec.id}: {compressions}"
                )
    except ImportError as exc:
        raise SchemaOrbitError("pyarrow is required for Parquet validation") from exc
    return _quality(features, targets)


def _process_dataset_unlocked(
    root: str | Path,
    config: SchemaOrbitConfig,
    spec: DatasetSpec,
    raw_path: Path,
    features_info: list[FeatureInfo],
    raw_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically create or reuse one processed dataset, without splits."""

    project_root = Path(root)
    raw_sha = str(raw_manifest["computed_sha256"])
    directory = project_root / "data" / "processed" / "openml" / str(spec.id)
    if spec.id == SMOKE_DATASET_ID:
        quality = validate_processed(project_root, config, spec, raw_sha)
        quality["reuse_status"] = "preserved_data_foundation"
        return quality
    try:
        existing = validate_processed(project_root, config, spec, raw_sha)
        existing["reuse_status"] = "validated_existing"
        return existing
    except SchemaOrbitError as exc:
        if directory.exists():
            observed_artifact_hashes = {
                str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
                for path in directory.rglob("*")
                if path.is_file()
            }
            quarantine_root = project_root / "data" / "cache" / "quarantine"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            destination = quarantine_root / (
                f"processed-{spec.id}-"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            shutil.move(str(directory), str(destination))
            atomic_write_json(
                destination / "quarantine_reason.json",
                {
                    "reason": str(exc),
                    "expected_raw_sha256": raw_sha,
                    "observed_path": str(directory),
                    "observed_artifact_hashes": observed_artifact_hashes,
                },
            )
    parsed = parse_arff(raw_path)
    if len(parsed.frame) != spec.rows:
        raise SchemaOrbitError(f"ARFF row count mismatch for {spec.id}")
    features, targets, details = _normalise_frame(parsed, spec, raw_sha, features_info)
    quality = _quality(features, targets)
    if quality["row_count"] != spec.rows:
        raise SchemaOrbitError(f"Processed row count mismatch for {spec.id}")
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".processed-{spec.id}-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir(parents=True, exist_ok=False)
    directory = temporary
    atomic_write_parquet(
        directory / "features.parquet", features, config.acquisition.parquet_compression
    )
    atomic_write_parquet(
        directory / "targets.parquet", targets, config.acquisition.parquet_compression
    )
    atomic_write_json(directory / "schema.json", details["schema"])
    mapping = {
        "schema_version": 1,
        "raw_sha256": raw_sha,
        "original_to_code": {label: i for i, label in enumerate(details["classes"])},
        "code_to_original": {str(i): label for i, label in enumerate(details["classes"])},
    }
    atomic_write_json(directory / "label_mapping.json", mapping)
    atomic_write_json(directory / "quality_report.json", {"schema_version": 2, **quality})
    artifact_hashes = {
        name: sha256_file(directory / name)
        for name in (
            "features.parquet",
            "targets.parquet",
            "schema.json",
            "label_mapping.json",
            "quality_report.json",
        )
    }
    source_manifest_path = (
        project_root
        / "data"
        / "raw"
        / "openml"
        / str(spec.id)
        / "source_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "openml_data_id": spec.id,
        "internal_dataset_id": spec.name,
        "row_count": len(features),
        "feature_columns": list(features.columns[1:]),
        "target_columns": list(targets.columns),
        "raw_sha256": raw_sha,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "processing_config_sha256": sha256_canonical_json(config.model_dump()),
        "compression": config.acquisition.parquet_compression,
        "compression_level": 3,
        "artifact_hashes": artifact_hashes,
        "row_id_formula": "sha256(openml:{id}:raw_file_sha256:source_row_position)[:32]",
    }
    atomic_write_json(directory / "data_manifest.json", manifest)
    try:
        _validate_processed_directory(project_root, directory, config, spec, raw_sha)
        os.replace(directory, parent / str(spec.id))
    except Exception as exc:
        quarantine_root = project_root / "data" / "cache" / "quarantine"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = quarantine_root / (
            f"processed-incomplete-{spec.id}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        if directory.exists():
            shutil.move(str(directory), str(destination))
            atomic_write_json(
                destination / "quarantine_reason.json",
                {"reason": str(exc), "expected_raw_sha256": raw_sha},
            )
        raise
    directory = parent / str(spec.id)
    return quality | {"reuse_status": "processed_new"}


def process_dataset(
    root: str | Path,
    config: SchemaOrbitConfig,
    spec: DatasetSpec,
    raw_path: Path,
    features_info: list[FeatureInfo],
    raw_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Create or reuse a processed set under a dataset-specific process lock."""

    project_root = Path(root)
    lock_path = project_root / "data" / "cache" / "locks" / f"openml-{spec.id}-processed.lock"
    with ProcessLock(lock_path, timeout=300):
        return _process_dataset_unlocked(
            project_root, config, spec, raw_path, features_info, raw_manifest
        )


def dataset_inventory_record(
    spec: DatasetSpec,
    metadata: DatasetMetadata,
    features: list[FeatureInfo],
    raw_manifest: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    quality_artifact = {
        key: value for key, value in quality.items() if key not in {"reuse_status"}
    }
    return {
        "schema_version": 2,
        "openml_data_id": spec.id,
        "dataset_name": spec.name,
        "openml_file_id": spec.file_id,
        "dataset_version": metadata.version,
        "provider_md5": spec.provider_md5,
        "raw_sha256": raw_manifest.get("computed_sha256"),
        "raw_size_bytes": raw_manifest.get("file_size_bytes"),
        "metadata_url": metadata.metadata_url,
        "download_url": metadata.download_url,
        "default_target": metadata.default_target_attribute,
        "ignored_attributes": spec.ignore_attributes,
        "feature_metadata": [item.model_dump() for item in features],
        "expected_rows": spec.rows,
        "observed_rows": quality["row_count"],
        "expected_predictors": spec.predictors,
        "observed_predictors": quality["predictor_count"],
        "class_count": quality["class_count"],
        "class_counts": quality["class_counts"],
        "quality": {"schema_version": 2, **quality_artifact},
        "license": metadata.licence,
        "visibility": metadata.visibility,
        "status": "PASS",
    }


__all__ = [
    "DatasetSpec",
    "SchemaOrbitConfig",
    "SchemaOrbitError",
    "OfflineCacheMiss",
    "load_schemaorbit_config",
    "fetch_metadata",
    "acquire_raw",
    "process_dataset",
    "validate_processed",
    "dataset_inventory_record",
]
