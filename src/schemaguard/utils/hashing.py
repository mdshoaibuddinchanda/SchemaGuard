"""Streaming and logical hashing helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CHUNK_SIZE = 1024 * 1024


def _file_digest(path: str | Path, algorithm: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the SHA-256 digest of a file using bounded reads."""
    return _file_digest(path, "sha256", chunk_size)


def md5_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the MD5 digest of a file using bounded reads."""
    return _file_digest(path, "md5", chunk_size)


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_source_hash(path: str | Path) -> str:
    """Hash source bytes after normalizing CRLF line endings to LF."""
    source = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(source)


def source_file_hashes(path: str | Path) -> set[str]:
    """Return accepted SHA-256 values for LF and equivalent CRLF source bytes."""
    source = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return {
        sha256_bytes(source),
        sha256_bytes(source.replace(b"\n", b"\r\n")),
    }


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_canonical_json(value: Any) -> str:
    """Hash JSON with sorted keys, UTF-8 encoding, and stable separators."""
    encoded = json.dumps(
        _canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def hash_dataframe_logically(frame: pd.DataFrame) -> str:
    """Hash ordered columns, dtypes, shape, and ordered cell values."""
    values = pd.util.hash_pandas_object(frame, index=False).astype("uint64").tolist()
    payload = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "shape": list(frame.shape),
        "row_hashes": values,
    }
    return sha256_canonical_json(payload)


def verify_file_hash(path: str | Path, expected_sha256: str) -> bool:
    """Return whether a file exists and matches the expected SHA-256 digest."""
    candidate = Path(path)
    return candidate.is_file() and sha256_file(candidate) == expected_sha256.lower()
