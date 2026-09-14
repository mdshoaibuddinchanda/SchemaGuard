"""Content-addressed checkpoint metadata and safe offline reuse."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..utils.process_lock import ProcessLock


class CacheError(RuntimeError):
    pass


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    cache_key: str
    model_id: str
    package_version: str
    checkpoint_identifier: str
    checkpoint_sha256: str
    size_bytes: int = Field(ge=0)
    path: str
    device_policy: str
    model_parameters: dict[str, Any]
    python_major_minor: str
    pytorch_version: str | None
    code_commit: str
    validated: bool


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_cache_key(
    *,
    model_id: str,
    package_version: str,
    checkpoint_identifier: str,
    checkpoint_sha256: str | None,
    device_policy: str,
    model_parameters: dict[str, Any],
    python_major_minor: str,
    pytorch_version: str | None,
    code_commit: str,
    fixture_hash: str | None = None,
) -> str:
    payload = {
        "model_id": model_id,
        "package_version": package_version,
        "checkpoint_identifier": checkpoint_identifier,
        "checkpoint_sha256": checkpoint_sha256,
        "device_policy": device_policy,
        "model_parameters": model_parameters,
        "python_major_minor": python_major_minor,
        "pytorch_version": pytorch_version,
        "code_commit": code_commit,
        "fixture_hash": fixture_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_write(path: str | Path, payload: Any) -> None:
    """Write JSON by replace, leaving the previous complete file intact on interruption."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def read_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return CheckpointMetadata.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CacheError(f"Invalid or truncated checkpoint manifest: {source}") from exc


def validate_checkpoint_file(metadata: CheckpointMetadata) -> bool:
    checkpoint = Path(metadata.path)
    if not checkpoint.is_file() or checkpoint.stat().st_size != metadata.size_bytes:
        raise CacheError("Checkpoint is missing or partial")
    observed = sha256_file(checkpoint)
    if observed != metadata.checkpoint_sha256:
        raise CacheError("Checkpoint SHA-256 does not match manifest")
    return True


def write_checkpoint_metadata(cache_dir: str | Path, metadata: CheckpointMetadata) -> Path:
    cache = Path(cache_dir)
    manifest = cache / f"{metadata.cache_key}.json"
    lock = cache / f"{metadata.cache_key}.lock"
    with ProcessLock(lock):
        atomic_json_write(manifest, metadata.model_dump(mode="json"))
    return manifest


def register_checkpoint(
    checkpoint_path: str | Path,
    *,
    cache_dir: str | Path,
    model_id: str,
    package_version: str,
    checkpoint_identifier: str,
    device_policy: str,
    model_parameters: dict[str, Any],
    python_major_minor: str,
    pytorch_version: str | None,
    code_commit: str,
    fixture_hash: str | None = None,
) -> CheckpointMetadata:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise CacheError(f"Checkpoint does not exist: {path}")
    digest = sha256_file(path)
    key = checkpoint_cache_key(
        model_id=model_id,
        package_version=package_version,
        checkpoint_identifier=checkpoint_identifier,
        checkpoint_sha256=digest,
        device_policy=device_policy,
        model_parameters=model_parameters,
        python_major_minor=python_major_minor,
        pytorch_version=pytorch_version,
        code_commit=code_commit,
        fixture_hash=fixture_hash,
    )
    metadata = CheckpointMetadata(
        cache_key=key,
        model_id=model_id,
        package_version=package_version,
        checkpoint_identifier=checkpoint_identifier,
        checkpoint_sha256=digest,
        size_bytes=path.stat().st_size,
        path=str(path.resolve()),
        device_policy=device_policy,
        model_parameters=model_parameters,
        python_major_minor=python_major_minor,
        pytorch_version=pytorch_version,
        code_commit=code_commit,
        validated=True,
    )
    write_checkpoint_metadata(cache_dir, metadata)
    return metadata


def resolve_offline_checkpoint(cache_dir: str | Path, cache_key: str) -> CheckpointMetadata:
    """Accept a cache hit only after strict manifest and content validation."""

    manifest = Path(cache_dir) / f"{cache_key}.json"
    metadata = read_checkpoint_metadata(manifest)
    if metadata.cache_key != cache_key or not metadata.validated:
        raise CacheError("Cache identity is invalid")
    validate_checkpoint_file(metadata)
    return metadata


def quarantine(path: str | Path) -> Path:
    source = Path(path)
    if not source.exists():
        return source
    target = source.with_name(f"{source.name}.quarantine.{uuid.uuid4().hex}")
    source.replace(target)
    return target
