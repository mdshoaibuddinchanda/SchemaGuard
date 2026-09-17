"""Immutable cache entries published as same-filesystem directory transactions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..utils.hashing import sha256_file
from ..utils.io import atomic_write_json
from .contracts import CacheArtifactManifest, CacheIdentity, CompletionMarker
from .locks import CacheKeyLock


class CacheIntegrityError(RuntimeError):
    """An artifact is incomplete, inconsistent, or conflicts with immutable identity."""


@dataclass(frozen=True)
class CacheArtifact:
    cache_key: str
    artifact_kind: str
    payload_path: Path
    manifest_path: Path
    payload_sha256: str
    payload_size_bytes: int
    producing_task_identity: str
    created_at: str
    source_implementation_sha256: str
    dependency_lock_sha256: str


@dataclass(frozen=True)
class PublishResult:
    artifact: CacheArtifact
    reused_existing: bool


class CacheStore:
    """Store completed immutable artifacts; manifests, not the SQLite index, are truth."""

    def __init__(self, root: str | Path, *, lock_timeout_seconds: float = 120.0) -> None:
        self.root = Path(root)
        self.lock_timeout_seconds = lock_timeout_seconds

    def entry_path(self, cache_key: str) -> Path:
        self._validate_key(cache_key)
        return self.root / cache_key[:2] / cache_key

    def artifact_path(self, identity: CacheIdentity) -> Path:
        return self.entry_path(identity.cache_key) / "payload.bin"

    def manifest_path(self, identity: CacheIdentity) -> Path:
        return self.entry_path(identity.cache_key) / "manifest.json"

    def read_validated(self, identity: CacheIdentity) -> CacheArtifact | None:
        """Validate every authoritative file; return None only for a true cache miss."""

        directory = self.entry_path(identity.cache_key)
        if not directory.exists():
            return None
        return self._validate_directory(directory, identity.cache_key, identity)

    def load_payload(self, identity: CacheIdentity) -> bytes:
        artifact = self.read_validated(identity)
        if artifact is None:
            raise CacheIntegrityError("cache entry is absent")
        return artifact.payload_path.read_bytes()

    def publish_bytes(
        self,
        identity: CacheIdentity,
        payload: bytes,
        *,
        producing_task_identity: str,
    ) -> PublishResult:
        """Publish compact in-memory payloads (primarily probes and tests)."""

        if not isinstance(payload, bytes):
            raise TypeError("cache payload must be immutable bytes")
        self.root.mkdir(parents=True, exist_ok=True)
        with CacheKeyLock(self.root, identity.cache_key, self.lock_timeout_seconds):
            return self._publish_stream(
                identity,
                producing_task_identity,
                lambda target: self._write_bytes(target, payload),
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )

    def publish_file(
        self,
        identity: CacheIdentity,
        source_path: str | Path,
        *,
        producing_task_identity: str,
    ) -> PublishResult:
        """Stream a worker artifact into a cache transaction using bounded memory."""

        source = Path(source_path)
        if not source.is_file():
            raise CacheIntegrityError("worker payload file is missing")
        self.root.mkdir(parents=True, exist_ok=True)
        with CacheKeyLock(self.root, identity.cache_key, self.lock_timeout_seconds):
            return self.publish_file_locked(
                identity, source, producing_task_identity=producing_task_identity
            )

    def identity_lock(self, identity: CacheIdentity) -> CacheKeyLock:
        """Expose the per-key lock for compute-once scheduling, never a global lock."""

        return CacheKeyLock(self.root, identity.cache_key, self.lock_timeout_seconds)

    def publish_file_locked(
        self,
        identity: CacheIdentity,
        source_path: str | Path,
        *,
        producing_task_identity: str,
    ) -> PublishResult:
        """Publish while the caller already holds ``identity_lock(identity)``."""

        source = Path(source_path)
        if not source.is_file():
            raise CacheIntegrityError("worker payload file is missing")
        return self._publish_stream(
            identity,
            producing_task_identity,
            lambda target: self._copy_stream(source, target),
            sha256_file(source),
            source.stat().st_size,
        )

    def _publish_stream(
        self,
        identity: CacheIdentity,
        task_identity: str,
        writer: Callable[[Path], None],
        expected_hash: str,
        expected_size: int,
    ) -> PublishResult:
        import re

        if re.fullmatch(r"[0-9a-f]{64}", task_identity) is None:
            raise ValueError("producing task identity must be a lowercase SHA-256 digest")
        key = identity.cache_key
        destination = self.entry_path(key)
        if destination.exists():
            existing = self._validate_directory(destination, key, identity)
            if (
                existing.payload_sha256 == expected_hash
                and existing.payload_size_bytes == expected_size
            ):
                return PublishResult(existing, reused_existing=True)
            raise CacheIntegrityError(
                "different payload bytes conflict with immutable cache identity"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        transaction = Path(tempfile.mkdtemp(prefix=f".{key}.txn-", dir=destination.parent))
        try:
            payload_path = transaction / "payload.bin"
            writer(payload_path)
            actual_size = payload_path.stat().st_size
            actual_hash = sha256_file(payload_path)
            if actual_size != expected_size or actual_hash != expected_hash:
                raise CacheIntegrityError("source payload changed while it was being copied")
            manifest = CacheArtifactManifest(
                schema_version=1,
                identity=identity,
                cache_key=key,
                artifact_kind=identity.artifact_kind,
                payload_sha256=actual_hash,
                payload_size_bytes=actual_size,
                producing_task_identity=task_identity,
                created_at=datetime.now(UTC),
                source_implementation_sha256=identity.source_implementation_sha256,
                dependency_lock_sha256=identity.dependency_lock_sha256,
                validation_status="PASS",
                completed=True,
            )
            manifest_path = transaction / "manifest.json"
            atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
            marker = CompletionMarker(
                schema_version=1,
                manifest_sha256=sha256_file(manifest_path),
                payload_sha256=actual_hash,
            )
            # The completion marker is deliberately the last transaction file.
            atomic_write_json(transaction / ".complete.json", marker.model_dump(mode="json"))
            self._validate_directory(transaction, key, identity, allow_transaction=True)
            try:
                os.replace(transaction, destination)
            except OSError as exc:
                if destination.exists():
                    winner = self._validate_directory(destination, key, identity)
                    if (
                        winner.payload_sha256 == actual_hash
                        and winner.payload_size_bytes == actual_size
                    ):
                        return PublishResult(winner, reused_existing=True)
                raise CacheIntegrityError("atomic cache directory publication failed") from exc
            artifact = self._validate_directory(destination, key, identity)
            return PublishResult(artifact, reused_existing=False)
        finally:
            if transaction.exists():
                shutil.rmtree(transaction, ignore_errors=True)

    def quarantine_invalid(self, identity: CacheIdentity) -> Path | None:
        """Move an invalid entry aside under lock; validated immutable entries are never moved."""

        key = identity.cache_key
        destination = self.entry_path(key)
        if not destination.exists():
            return None
        with CacheKeyLock(self.root, key, self.lock_timeout_seconds):
            return self.quarantine_invalid_locked(identity)

    def quarantine_invalid_locked(self, identity: CacheIdentity) -> Path | None:
        """Quarantine an invalid entry while the caller holds its identity lock."""

        key = identity.cache_key
        destination = self.entry_path(key)
        if not destination.exists():
            return None
        try:
            self._validate_directory(destination, key, identity)
        except CacheIntegrityError:
            quarantine_root = self.root / ".quarantine"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            quarantined = quarantine_root / f"{key}-{uuid.uuid4().hex}"
            os.replace(destination, quarantined)
            return quarantined
        raise CacheIntegrityError("refusing to quarantine a valid immutable cache entry")

    def iter_validated(self) -> list[CacheArtifact]:
        """Discover valid entries from manifests; transaction and quarantine dirs are ignored."""

        result: list[CacheArtifact] = []
        if not self.root.is_dir():
            return result
        for prefix in sorted(self.root.iterdir(), key=lambda item: item.name):
            if len(prefix.name) != 2 or any(char not in "0123456789abcdef" for char in prefix.name):
                continue
            for directory in sorted(prefix.iterdir(), key=lambda item: item.name):
                if len(directory.name) != 64 or directory.name[:2] != prefix.name:
                    continue
                try:
                    raw = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                    manifest = CacheArtifactManifest.model_validate(raw)
                    result.append(
                        self._validate_directory(directory, manifest.cache_key, manifest.identity)
                    )
                except (
                    OSError,
                    ValueError,
                    ValidationError,
                    CacheIntegrityError,
                    json.JSONDecodeError,
                ):
                    continue
        return result

    def _validate_directory(
        self,
        directory: Path,
        cache_key: str,
        identity: CacheIdentity,
        *,
        allow_transaction: bool = False,
    ) -> CacheArtifact:
        expected_files = {"payload.bin", "manifest.json", ".complete.json"}
        actual_files = {item.name for item in directory.iterdir() if item.is_file()}
        if actual_files != expected_files or any(item.is_dir() for item in directory.iterdir()):
            raise CacheIntegrityError("cache entry is partial or contains unexpected files")
        try:
            manifest_path = directory / "manifest.json"
            marker_path = directory / ".complete.json"
            manifest = CacheArtifactManifest.model_validate(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            marker = CompletionMarker.model_validate(
                json.loads(marker_path.read_text(encoding="utf-8"))
            )
            payload_path = directory / "payload.bin"
            payload_size = payload_path.stat().st_size
            payload_hash = sha256_file(payload_path)
            if (
                not allow_transaction and directory.name != cache_key
            ) or manifest.cache_key != cache_key:
                raise CacheIntegrityError("cache directory and manifest keys differ")
            if manifest.identity != identity or identity.cache_key != cache_key:
                raise CacheIntegrityError("cache identity does not match the request")
            if manifest.validation_status != "PASS" or not manifest.completed:
                raise CacheIntegrityError("failed or incomplete artifact cannot be reused")
            if (
                payload_size != manifest.payload_size_bytes
                or payload_hash != manifest.payload_sha256
            ):
                raise CacheIntegrityError("cached payload size or SHA-256 mismatch")
            if marker.manifest_sha256 != sha256_file(manifest_path):
                raise CacheIntegrityError("completion marker does not identify this manifest")
            if marker.payload_sha256 != payload_hash:
                raise CacheIntegrityError("completion marker does not identify this payload")
            return CacheArtifact(
                cache_key=cache_key,
                artifact_kind=manifest.artifact_kind,
                payload_path=payload_path,
                manifest_path=manifest_path,
                payload_sha256=payload_hash,
                payload_size_bytes=payload_size,
                producing_task_identity=manifest.producing_task_identity,
                created_at=manifest.created_at.isoformat(),
                source_implementation_sha256=manifest.source_implementation_sha256,
                dependency_lock_sha256=manifest.dependency_lock_sha256,
            )
        except CacheIntegrityError:
            raise
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise CacheIntegrityError("cache manifest, marker, or payload is invalid") from exc

    @staticmethod
    def _validate_key(cache_key: str) -> None:
        if len(cache_key) != 64 or any(char not in "0123456789abcdef" for char in cache_key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")

    @staticmethod
    def _write_bytes(target: Path, data: bytes) -> None:
        with target.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _copy_stream(source: Path, target: Path) -> None:
        with source.open("rb") as input_handle, target.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())


__all__ = ["CacheArtifact", "CacheIntegrityError", "CacheStore", "PublishResult"]
