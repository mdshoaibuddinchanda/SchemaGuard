"""Content identity for the transformation engine implementation."""

from __future__ import annotations

from pathlib import Path

from ..utils.hashing import canonical_source_hash, sha256_canonical_json


def transformation_engine_implementation_hash(
    package_root: str | Path | None = None,
) -> str:
    """Hash all transformation modules and the shared logical-hashing helper.

    Source line endings are normalized before hashing so evidence is portable
    between native LF and CRLF checkouts.  Any semantic edit to a view,
    certificate, codec, validation, reconstruction, or logical-hashing module
    changes the resulting identity.
    """

    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    transformations_root = root / "transformations"
    hashing_path = root / "utils" / "hashing.py"
    if not transformations_root.is_dir() or not hashing_path.is_file():
        raise FileNotFoundError("transformation engine source tree is incomplete")
    paths = sorted(transformations_root.rglob("*.py"))
    paths.append(hashing_path)
    payload = {
        path.relative_to(root).as_posix(): canonical_source_hash(path)
        for path in paths
    }
    return sha256_canonical_json(payload)


__all__ = ["transformation_engine_implementation_hash"]
