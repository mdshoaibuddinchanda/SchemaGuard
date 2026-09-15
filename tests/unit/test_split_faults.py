import json
import threading
from pathlib import Path

import pytest
from filelock import Timeout

from schemaguard.splits.contracts import SplitGenerationConfig, SplitManifestContract
from schemaguard.splits.inventory import compare_snapshots, verify_protected_snapshot
from schemaguard.splits.locking import split_lock
from schemaguard.utils.io import atomic_write_json

from .test_split_contracts import valid_payload


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


def test_protected_snapshot_mutation_is_not_a_pass() -> None:
    comparison = compare_snapshots(
        {"files": {"data/processed/features.parquet": "a"}},
        {"files": {"data/processed/features.parquet": "b"}},
    )
    assert comparison["status"] == "FAIL"
    assert comparison["changed_files"] == ["data/processed/features.parquet"]


def test_missing_protected_snapshot_is_blocked_when_local_evidence_exists(tmp_path: Path) -> None:
    source = tmp_path / "data" / "processed" / "openml" / "3"
    source.mkdir(parents=True)
    (source / "features.parquet").write_bytes(b"local evidence")
    config = SplitGenerationConfig.model_validate(valid_payload())
    with pytest.raises(ValueError, match="BLOCKED_PROTECTED_SNAPSHOT_MISSING"):
        verify_protected_snapshot(
            tmp_path,
            config,
            tmp_path / "before.json",
            tmp_path / "after.json",
            tmp_path / "comparison.json",
        )


def test_empty_transaction_directory_is_detected(tmp_path: Path) -> None:
    transaction = tmp_path / "data" / "splits" / "openml" / "3" / "stratified_group_5fold_v1"
    (transaction / ".seed_1729.empty").mkdir(parents=True)
    from schemaguard.splits.validation import _reject_temporary_artifacts

    with pytest.raises(ValueError, match="temporary or partial"):
        _reject_temporary_artifacts(tmp_path)


def test_stale_lock_is_recoverable(tmp_path: Path) -> None:
    lock = split_lock(tmp_path, 3, 1729, timeout=1)
    Path(lock.path).parent.mkdir(parents=True, exist_ok=True)
    Path(lock.path).write_text("stale owner", encoding="utf-8")
    with lock:
        assert Path(lock.path).is_file()


def test_active_lock_times_out(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with split_lock(tmp_path, 3, 1729, timeout=1):
            entered.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=hold_lock)
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(Timeout):
            with split_lock(tmp_path, 3, 1729, timeout=0.1):
                pass
    finally:
        release.set()
        worker.join(timeout=5)
