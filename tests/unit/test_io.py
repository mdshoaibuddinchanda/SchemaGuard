from pathlib import Path

import pandas as pd

from schemaguard.utils.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_parquet,
    read_json_validated,
)


def test_atomic_json_is_canonical_and_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"
    atomic_write_json(target, {"z": 1, "a": [2, 3]})
    assert target.read_text(encoding="utf-8") == '{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n'
    assert read_json_validated(target) == {"z": 1, "a": [2, 3]}


def test_atomic_bytes_does_not_leave_part_files(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    atomic_write_bytes(target, b"stable")
    assert target.read_bytes() == b"stable"
    assert list(tmp_path.glob("*.part")) == []


def test_atomic_parquet_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "table.parquet"
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    atomic_write_parquet(target, frame)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(target))
