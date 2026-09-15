"""Content identity for the computational split-generation implementation."""

from __future__ import annotations

from pathlib import Path

from schemaguard.utils.hashing import sha256_bytes

IMPLEMENTATION_SOURCE_FILES = (
    "contracts.py",
    "grouping.py",
    "selection.py",
    "generation.py",
    "validation.py",
    "caching.py",
    "implementation.py",
)


def split_implementation_hash() -> str:
    """Hash computational source files while excluding documentation and Git metadata."""

    base = Path(__file__).parent
    payload = bytearray()
    for filename in IMPLEMENTATION_SOURCE_FILES:
        source = (base / filename).read_text(encoding="utf-8").replace("\r\n", "\n")
        payload.extend(filename.encode("utf-8"))
        payload.extend(b"\0")
        payload.extend(source.encode("utf-8"))
        payload.extend(b"\0")
    return sha256_bytes(bytes(payload))


SPLIT_IMPLEMENTATION_HASH = split_implementation_hash()

__all__ = ["IMPLEMENTATION_SOURCE_FILES", "SPLIT_IMPLEMENTATION_HASH", "split_implementation_hash"]
