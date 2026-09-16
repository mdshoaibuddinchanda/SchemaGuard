from __future__ import annotations

from pathlib import Path

from schemaguard.transformations.implementation import transformation_engine_implementation_hash


def test_engine_identity_is_line_ending_stable_and_semantic_change_sensitive(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "schemaguard"
    transformations = package_root / "transformations"
    utilities = package_root / "utils"
    transformations.mkdir(parents=True)
    utilities.mkdir()
    transformation_module = transformations / "codec.py"
    hashing_module = utilities / "hashing.py"
    transformation_module.write_bytes(b"VALUE = 'original'\n")
    hashing_module.write_bytes(b"def logical_hash(value):\n    return value\n")

    lf_identity = transformation_engine_implementation_hash(package_root)
    transformation_module.write_bytes(b"VALUE = 'original'\r\n")
    hashing_module.write_bytes(b"def logical_hash(value):\r\n    return value\r\n")
    crlf_identity = transformation_engine_implementation_hash(package_root)
    assert crlf_identity == lf_identity

    transformation_module.write_bytes(b"VALUE = 'semantic change'\r\n")
    assert transformation_engine_implementation_hash(package_root) != lf_identity
