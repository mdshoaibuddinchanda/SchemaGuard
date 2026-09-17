from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemaguard.cache.store import CacheIntegrityError, CacheStore
from tests.unit.cache_scheduler_helpers import cache_identity


def _publish(store: CacheStore, payload: bytes = b"stable-payload"):
    return store.publish_bytes(cache_identity(), payload, producing_task_identity="a" * 64).artifact


def test_atomic_cache_roundtrip_and_immutable_identical_reuse(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    first = _publish(store)
    reused = store.publish_bytes(
        cache_identity(), b"stable-payload", producing_task_identity="b" * 64
    )
    assert reused.reused_existing
    assert reused.artifact.payload_sha256 == first.payload_sha256
    assert store.load_payload(cache_identity()) == b"stable-payload"
    assert {item.name for item in first.payload_path.parent.iterdir()} == {
        "payload.bin",
        "manifest.json",
        ".complete.json",
    }


def test_same_identity_with_different_bytes_is_an_integrity_failure(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    _publish(store)
    with pytest.raises(CacheIntegrityError, match="different payload"):
        store.publish_bytes(cache_identity(), b"different", producing_task_identity="a" * 64)


@pytest.mark.parametrize(
    "fault",
    [
        "manifest_truncated",
        "payload_truncated",
        "marker_missing",
        "identity_wrong",
        "key_wrong",
        "schema_wrong",
    ],
)
def test_corrupt_entry_never_validates_and_can_be_quarantined(tmp_path: Path, fault: str) -> None:
    store = CacheStore(tmp_path / "cache")
    artifact = _publish(store)
    directory = artifact.payload_path.parent
    if fault == "manifest_truncated":
        (directory / "manifest.json").write_text("{", encoding="utf-8")
    elif fault == "payload_truncated":
        (directory / "payload.bin").write_bytes(b"bad")
    elif fault == "marker_missing":
        (directory / ".complete.json").unlink()
    else:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if fault == "identity_wrong":
            manifest["identity"]["dataset_sha256"] = "9" * 64
        elif fault == "key_wrong":
            manifest["cache_key"] = "9" * 64
        elif fault == "schema_wrong":
            manifest["schema_version"] = 9
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CacheIntegrityError):
        store.read_validated(cache_identity())
    quarantined = store.quarantine_invalid(cache_identity())
    assert quarantined is not None and quarantined.exists()
    assert store.read_validated(cache_identity()) is None


@pytest.mark.parametrize("files", ["payload_only", "payload_manifest_no_marker"])
def test_incomplete_transaction_directory_is_never_discovered(tmp_path: Path, files: str) -> None:
    store = CacheStore(tmp_path / "cache")
    identity = cache_identity()
    entry_parent = store.entry_path(identity.cache_key).parent
    entry_parent.mkdir(parents=True)
    transaction = entry_parent / f".{identity.cache_key}.txn-interrupted"
    transaction.mkdir()
    (transaction / "payload.bin").write_bytes(b"partial")
    if files == "payload_manifest_no_marker":
        (transaction / "manifest.json").write_text("{}", encoding="utf-8")
    assert store.read_validated(identity) is None
    assert store.iter_validated() == []


def test_worker_payload_is_streamed_and_manifest_binds_source_identity(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    source = tmp_path / "worker.bin"
    source.write_bytes(b"worker-payload")
    artifact = store.publish_file(
        cache_identity(), source, producing_task_identity="c" * 64
    ).artifact
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_implementation_sha256"] == cache_identity().source_implementation_sha256
    assert manifest["dependency_lock_sha256"] == cache_identity().dependency_lock_sha256
    assert artifact.payload_path.read_bytes() == b"worker-payload"


def test_interrupted_write_cleans_transaction_and_never_publishes(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    source = tmp_path / "source.bin"
    source.write_bytes(b"complete source")
    original = store._copy_stream

    def interrupt(source_path: Path, target_path: Path) -> None:
        target_path.write_bytes(b"partial")
        raise OSError("injected interruption")

    store._copy_stream = interrupt  # type: ignore[method-assign]
    with pytest.raises(OSError):
        store.publish_file(cache_identity(), source, producing_task_identity="d" * 64)
    store._copy_stream = original  # type: ignore[method-assign]
    assert store.read_validated(cache_identity()) is None
    assert not list((tmp_path / "cache").glob("**/*.txn-*"))
