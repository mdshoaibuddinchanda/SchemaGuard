from pathlib import Path

import pandas as pd

from schemaguard.utils.hashing import (
    hash_dataframe_logically,
    md5_file,
    sha256_bytes,
    sha256_canonical_json,
    sha256_file,
    verify_file_hash,
)


def test_known_hash_vectors(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert (
        sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert sha256_file(path) == sha256_bytes(b"abc")
    assert md5_file(path) == "900150983cd24fb0d6963f7d28e17f72"
    assert verify_file_hash(path, sha256_file(path))


def test_changed_bytes_change_hash(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    before = sha256_file(path)
    path.write_bytes(b"abd")
    assert sha256_file(path) != before


def test_canonical_json_and_dataframe_hashes_are_order_sensitive() -> None:
    assert sha256_canonical_json({"b": 2, "a": 1}) == sha256_canonical_json({"a": 1, "b": 2})
    first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    second = first.iloc[::-1].reset_index(drop=True)
    assert hash_dataframe_logically(first) != hash_dataframe_logically(second)
