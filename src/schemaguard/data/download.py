"""OpenML metadata verification, streaming download, and immutable raw caching."""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
from filelock import FileLock

from schemaguard.constants import OPENML_DATA_ID, OPENML_FILE_ID
from schemaguard.utils.hashing import md5_file, sha256_file
from schemaguard.utils.io import atomic_write_json, read_json_validated

from .contracts import SmokeDatasetConfig, SourceManifest


class DownloadError(RuntimeError):
    """Base class for source-acquisition failures."""


class SourceIntegrityError(DownloadError):
    """Raised when a source identity or checksum differs from the frozen contract."""


class OfflineCacheUnavailable(DownloadError):
    """Raised when offline mode cannot find a valid cached source."""


@dataclass(frozen=True)
class DownloadResult:
    raw_path: Path
    manifest_path: Path
    source_manifest: SourceManifest
    cache_status: str


def _package_versions() -> dict[str, str]:
    packages = ["httpx", "pydantic", "pandas", "pyarrow", "liac-arff", "scikit-learn"]
    return {
        package: importlib.metadata.version(package)
        for package in packages
        if _is_installed(package)
    }


def _is_installed(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    description = metadata.get("data_set_description", metadata)
    return description.get(key, default) if isinstance(description, dict) else default


def _verify_metadata(metadata: dict[str, Any], config: SmokeDatasetConfig) -> dict[str, Any]:
    observed = {
        "data_id": str(_metadata_value(metadata, "id")),
        "file_id": str(_metadata_value(metadata, "file_id")),
        "name": _metadata_value(metadata, "name"),
        "target": _metadata_value(metadata, "default_target_attribute"),
        "status": _metadata_value(metadata, "status"),
        "format": str(_metadata_value(metadata, "format", "")).upper(),
        "provider_md5": _metadata_value(metadata, "md5_checksum"),
        "version": str(_metadata_value(metadata, "version", "")),
    }
    expected = config.dataset
    mismatches = []
    if observed["data_id"] != str(expected.openml_data_id):
        mismatches.append(f"data ID {observed['data_id']} != {expected.openml_data_id}")
    if observed["file_id"] != str(expected.openml_file_id):
        mismatches.append(f"file ID {observed['file_id']} != {expected.openml_file_id}")
    if observed["name"] != expected.expected_name:
        mismatches.append(f"dataset name {observed['name']!r} != {expected.expected_name!r}")
    if observed["target"] != config.task.expected_target_name:
        mismatches.append(f"target {observed['target']!r} != {config.task.expected_target_name!r}")
    if observed["status"] not in {"active", "in_preparation"}:
        mismatches.append(f"dataset status {observed['status']!r} is not acceptable")
    if observed["format"] != "ARFF":
        mismatches.append(f"source format {observed['format']!r} != 'ARFF'")
    if mismatches:
        raise SourceIntegrityError(
            "OpenML metadata differs from frozen contract: " + "; ".join(mismatches)
        )
    return observed


def _cache_manifest_is_valid(
    raw_path: Path, manifest_path: Path, config: SmokeDatasetConfig
) -> SourceManifest | None:
    if not raw_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = read_json_validated(manifest_path, SourceManifest)
    except Exception as exc:
        raise SourceIntegrityError(f"Cached source manifest is invalid: {exc}") from exc
    expected = config.dataset
    identity_matches = (
        manifest.internal_dataset_id == expected.internal_id
        and manifest.openml_data_id == expected.openml_data_id
        and manifest.openml_file_id == expected.openml_file_id
        and manifest.dataset_name == expected.expected_name
        and manifest.default_target_attribute == config.task.expected_target_name
        and manifest.data_format == "ARFF"
    )
    if not identity_matches:
        raise SourceIntegrityError("Cached source manifest identity differs from frozen contract")
    if (
        sha256_file(raw_path) != manifest.computed_sha256
        or md5_file(raw_path) != manifest.computed_md5
    ):
        raise SourceIntegrityError("Cached raw source bytes do not match source manifest")
    return manifest


def _request_metadata(client: Any, config: SmokeDatasetConfig) -> dict[str, Any]:
    response = client.get(config.dataset.metadata_url)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise DownloadError("OpenML metadata response is not a JSON object")
    return payload


def download_dataset(
    config: SmokeDatasetConfig,
    raw_directory: str | Path,
    *,
    offline: bool = False,
    lock_path: str | Path | None = None,
    client: Any | None = None,
) -> DownloadResult:
    """Acquire the authoritative ARFF, or return a validated immutable cache hit."""
    raw_dir = Path(raw_directory)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / config.dataset.raw_filename
    manifest_path = raw_dir / "source_manifest.json"
    metadata_path = raw_dir / "openml_metadata.json"
    lock = Path(lock_path) if lock_path is not None else raw_dir / "openml_1464.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock), timeout=config.resources.lock_timeout_seconds):
        cached = _cache_manifest_is_valid(raw_path, manifest_path, config)
        if cached is not None:
            return DownloadResult(raw_path, manifest_path, cached, "hit")
        if offline:
            raise OfflineCacheUnavailable(
                f"Offline cache unavailable or invalid at {raw_path}; no HTTP request was made"
            )

        owns_client = client is None
        http_client = client or httpx.Client(
            follow_redirects=True,
            timeout=config.resources.request_timeout_seconds,
        )
        try:
            metadata = _request_metadata(http_client, config)
            observed = _verify_metadata(metadata, config)
            if not metadata_path.exists():
                atomic_write_json(metadata_path, metadata)
            provider_md5 = observed["provider_md5"]
            last_error: Exception | None = None
            temporary: Path | None = None
            response_headers: dict[str, str] = {}
            resolved_url = config.dataset.download_url
            for attempt in range(config.resources.download_attempts):
                try:
                    with http_client.stream("GET", config.dataset.download_url) as response:
                        response.raise_for_status()
                        resolved_url = str(response.url)
                        response_headers = dict(response.headers)
                        with NamedTemporaryFile(
                            mode="wb",
                            prefix=f".{raw_path.name}.",
                            suffix=".part",
                            dir=raw_dir,
                            delete=False,
                        ) as handle:
                            temporary = Path(handle.name)
                            for chunk in response.iter_bytes(
                                chunk_size=config.resources.download_chunk_bytes
                            ):
                                if chunk:
                                    handle.write(chunk)
                            handle.flush()
                        computed_md5 = md5_file(temporary, config.resources.download_chunk_bytes)
                        computed_sha256 = sha256_file(
                            temporary, config.resources.download_chunk_bytes
                        )
                        if provider_md5 and computed_md5.lower() != str(provider_md5).lower():
                            raise SourceIntegrityError(
                                f"Downloaded MD5 {computed_md5} != OpenML MD5 {provider_md5}"
                            )
                        if raw_path.exists():
                            if sha256_file(raw_path) != computed_sha256:
                                raise SourceIntegrityError(
                                    "A different raw file already exists; refusing to overwrite it"
                                )
                            temporary.unlink(missing_ok=True)
                            temporary = None
                        else:
                            temporary.replace(raw_path)
                            temporary = None
                        break
                except SourceIntegrityError:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                    raise
                except Exception as exc:  # Retry transport errors only.
                    last_error = exc
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                        temporary = None
                    if attempt + 1 < config.resources.download_attempts:
                        time.sleep(min(2**attempt, 8))
            else:
                raise DownloadError(
                    f"OpenML download failed after retries: {last_error}"
                ) from last_error

            manifest = SourceManifest(
                internal_dataset_id=config.dataset.internal_id,
                provider="openml",
                openml_data_id=OPENML_DATA_ID,
                openml_file_id=OPENML_FILE_ID,
                dataset_name=str(observed["name"]),
                dataset_version=str(observed["version"]),
                source_page=config.dataset.source_page,
                requested_download_url=config.dataset.download_url,
                resolved_download_url=resolved_url,
                metadata_url=config.dataset.metadata_url,
                data_format="ARFF",
                default_target_attribute=str(observed["target"]),
                provider_md5=None if provider_md5 is None else str(provider_md5),
                computed_md5=md5_file(raw_path, config.resources.download_chunk_bytes),
                computed_sha256=sha256_file(raw_path, config.resources.download_chunk_bytes),
                file_size_bytes=raw_path.stat().st_size,
                raw_relative_path=str(raw_path.relative_to(raw_dir.parent.parent.parent)).replace(
                    "\\", "/"
                ),
                retrieved_at_utc=datetime.now(UTC),
                http_etag=response_headers.get("etag"),
                http_last_modified=response_headers.get("last-modified"),
                package_versions=_package_versions(),
                cache_status="downloaded",
            )
            atomic_write_json(manifest_path, manifest.canonical_dict())
            return DownloadResult(raw_path, manifest_path, manifest, "miss")
        finally:
            if owns_client:
                http_client.close()
