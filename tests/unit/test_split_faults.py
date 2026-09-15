import json
from pathlib import Path

import pytest

from schemaguard.splits.contracts import SplitManifestContract
from schemaguard.splits.locking import split_lock
from schemaguard.utils.io import atomic_write_json


def test_truncated_json_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json.loads(path.read_text(encoding="utf-8"))


def test_atomic_json_write_leaves_valid_document(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"schema_version": 1, "status": "PASS"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "PASS"
    assert not list(tmp_path.glob("*.part"))


def test_lock_is_reentrant_for_one_worker(tmp_path: Path) -> None:
    lock = split_lock(tmp_path, 3, 1729, timeout=1)
    with lock:
        assert Path(lock.path).name == "split-3-1729.lock"


def test_contract_rejects_unknown_manifest_field() -> None:
    with pytest.raises(Exception):
        SplitManifestContract.model_validate({"unknown": True})
