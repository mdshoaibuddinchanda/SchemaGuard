from __future__ import annotations

from contextlib import closing
from pathlib import Path

from schemaguard.cache.index import CacheIndex
from schemaguard.cache.store import CacheStore
from tests.unit.cache_scheduler_helpers import cache_identity


def test_index_supports_filters_and_stores_portable_relative_paths(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    store = CacheStore(cache_root)
    artifact = store.publish_bytes(
        cache_identity(), b"index-me", producing_task_identity="a" * 64
    ).artifact
    index = CacheIndex(tmp_path / "runtime" / "index.sqlite", cache_root)
    index.add(artifact)
    rows = index.lookup(
        cache_key=artifact.cache_key,
        artifact_kind="probe",
        task_identity="a" * 64,
        status="COMPLETE",
    )
    assert len(rows) == 1
    assert rows[0].payload_path == f"{artifact.cache_key[:2]}/{artifact.cache_key}/payload.bin"
    assert not Path(rows[0].payload_path).is_absolute()


def test_missing_or_stale_index_rebuilds_from_valid_manifests(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    store = CacheStore(cache_root)
    artifacts = [
        store.publish_bytes(
            cache_identity(seed=1729 + index),
            f"artifact-{index}".encode(),
            producing_task_identity=str(index + 1) * 64,
        ).artifact
        for index in range(2)
    ]
    index = CacheIndex(tmp_path / "runtime" / "missing.sqlite", cache_root)
    assert index.rebuild(store) == 2
    assert len(index.lookup(status="COMPLETE")) == 2
    # A stale extra row is discarded; the filesystem remains authoritative.
    with closing(index._connect()) as connection, connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "f" * 64,
                "probe",
                "e" * 64,
                "COMPLETE",
                "bad",
                "bad",
                "f" * 64,
                1,
                "now",
                "now",
                "f" * 64,
            ),
        )
    assert index.rebuild(store) == len(artifacts)
    assert len(index.lookup()) == 2


def test_corrupt_sqlite_index_is_quarantined_and_rebuilt(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    store = CacheStore(cache_root)
    artifact = store.publish_bytes(
        cache_identity(), b"authoritative", producing_task_identity="a" * 64
    ).artifact
    database = tmp_path / "runtime" / "index.sqlite"
    database.parent.mkdir()
    database.write_bytes(b"not a sqlite database")
    index = CacheIndex(database, cache_root)
    assert index.rebuild(store) == 1
    assert len(index.lookup(cache_key=artifact.cache_key)) == 1
    assert list(database.parent.glob("index.sqlite.corrupt-*"))


def test_readonly_lookup_does_not_create_a_missing_index(tmp_path: Path) -> None:
    database = tmp_path / "runtime" / "missing.sqlite"
    index = CacheIndex(database, tmp_path / "cache")
    assert index.lookup(cache_key="a" * 64) == []
    assert not database.exists()
