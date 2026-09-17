"""Canonical cache-key and implementation-identity helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..utils.hashing import canonical_source_hash, sha256_canonical_json
from .contracts import CacheIdentity


def implementation_hash(root: str | Path, source_paths: Iterable[str]) -> str:
    """Hash a sorted, repository-relative set of relevant source files."""

    project = Path(root).resolve()
    identities: dict[str, str] = {}
    for item in sorted(set(source_paths)):
        relative = Path(item)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("implementation paths must be safe repository-relative paths")
        candidate = (project / relative).resolve()
        if project not in candidate.parents or not candidate.is_file():
            raise ValueError(f"implementation source is missing or outside the repository: {item}")
        portable = relative.as_posix()
        identities[portable] = canonical_source_hash(candidate)
    if not identities:
        raise ValueError("at least one implementation source is required")
    return sha256_canonical_json(identities)


def dependency_lock_hash(path: str | Path) -> str:
    """Hash the resolved text lock with platform-independent line endings."""

    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError("dependency lock file is missing")
    return canonical_source_hash(candidate)


def identity_with(identity: CacheIdentity, **changes: object) -> CacheIdentity:
    """Create a validated identity variant, useful for explicit retry decisions."""

    return CacheIdentity.model_validate({**identity.model_dump(mode="json"), **changes})
