"""Small content-addressed, atomic adapter prediction/model round-trip cache."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...compatibility.checkpoint_cache import quarantine
from ...utils.io import atomic_write_json
from ...utils.process_lock import ProcessLock
from .contracts import AdapterCacheIdentity


class CacheIntegrityError(RuntimeError):
    """A cache artifact is absent, truncated, mismatched, or tampered with."""


class _CacheEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    artifact_kind: Literal["model", "prediction"]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity: AdapterCacheIdentity
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_size_bytes: int = Field(ge=0)
    payload_base64: str


class AdapterArtifactCache:
    """Cache only complete adapter artifacts under a source-bound identity."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(
        self, kind: Literal["model", "prediction"], identity: AdapterCacheIdentity
    ) -> Path:
        return self.directory / f"{identity.cache_key}.{kind}.json"

    def store_json(
        self,
        kind: Literal["model", "prediction"],
        identity: AdapterCacheIdentity,
        payload: Any,
    ) -> Path:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CacheIntegrityError("cache payload is not valid strict JSON") from exc
        return self._store(kind, identity, encoded)

    def store_bytes(
        self,
        kind: Literal["model", "prediction"],
        identity: AdapterCacheIdentity,
        payload: bytes,
        *,
        reuse_validated_existing: bool = False,
    ) -> Path:
        return self._store(
            kind,
            identity,
            payload,
            reuse_validated_existing=reuse_validated_existing,
        )

    def _store(
        self,
        kind: Literal["model", "prediction"],
        identity: AdapterCacheIdentity,
        payload: bytes,
        *,
        reuse_validated_existing: bool = False,
    ) -> Path:
        destination = self.path_for(kind, identity)
        lock_path = self.directory / "locks" / f"{identity.cache_key}.lock"
        self.directory.mkdir(parents=True, exist_ok=True)
        with ProcessLock(lock_path, timeout=60):
            if destination.exists():
                try:
                    existing = self._load_bytes(kind, identity)
                except CacheIntegrityError:
                    quarantine(destination)
                else:
                    if existing == payload:
                        return destination
                    if kind == "model" and reuse_validated_existing:
                        return destination
                    raise CacheIntegrityError(
                        "a validated cache key maps to different payload bytes"
                    )
            envelope = _CacheEnvelope(
                artifact_kind=kind,
                cache_key=identity.cache_key,
                identity=identity,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                payload_size_bytes=len(payload),
                payload_base64=base64.b64encode(payload).decode("ascii"),
            )
            atomic_write_json(destination, envelope.model_dump(mode="json"))
        return destination

    def load_json(
        self,
        kind: Literal["model", "prediction"],
        identity: AdapterCacheIdentity,
    ) -> Any:
        encoded = self._load_bytes(kind, identity)
        try:
            return json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError("cached JSON payload is truncated or invalid") from exc

    def load_bytes(
        self,
        kind: Literal["model", "prediction"],
        identity: AdapterCacheIdentity,
    ) -> bytes:
        return self._load_bytes(kind, identity)

    def _load_bytes(
        self,
        kind: Literal["model", "prediction"],
        identity: AdapterCacheIdentity,
    ) -> bytes:
        source = self.path_for(kind, identity)
        try:
            with source.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            envelope = _CacheEnvelope.model_validate(raw)
            if envelope.artifact_kind != kind or envelope.cache_key != identity.cache_key:
                raise CacheIntegrityError("cache kind or key does not match the request")
            if envelope.identity != identity:
                raise CacheIntegrityError("cache identity does not match the request")
            payload = base64.b64decode(envelope.payload_base64, validate=True)
            if len(payload) != envelope.payload_size_bytes:
                raise CacheIntegrityError("cache payload is partial")
            if hashlib.sha256(payload).hexdigest() != envelope.payload_sha256:
                raise CacheIntegrityError("cache payload checksum mismatch")
            return payload
        except CacheIntegrityError:
            raise
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise CacheIntegrityError(
                f"invalid or truncated adapter cache artifact: {source.name}"
            ) from exc
