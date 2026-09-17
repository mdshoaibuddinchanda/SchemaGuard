from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from schemaguard.cache.contracts import CacheIdentity
from schemaguard.cache.locks import CacheKeyLock
from schemaguard.cache.store import CacheIntegrityError, CacheStore
from tests.unit.cache_scheduler_helpers import cache_identity

pytestmark = pytest.mark.integration


def _publish_in_process(
    cache_path: str,
    identity_data: dict[str, object],
    barrier: object,
    payload: bytes,
    queue: object,
) -> None:
    identity = CacheIdentity.model_validate(identity_data)
    barrier.wait(timeout=20)  # type: ignore[attr-defined]
    try:
        result = CacheStore(cache_path).publish_bytes(
            identity,
            payload,
            producing_task_identity="a" * 64,
        )
        queue.put(("PASS", result.artifact.payload_sha256))  # type: ignore[attr-defined]
    except CacheIntegrityError as exc:
        queue.put(("FAIL_CACHE_INTEGRITY", str(exc)))  # type: ignore[attr-defined]


def _acquire_then_exit(cache_path: str, cache_key: str, acquired: object) -> None:
    lock = CacheKeyLock(cache_path, cache_key, timeout=20)
    lock.__enter__()
    acquired.set()  # type: ignore[attr-defined]
    os._exit(0)


def _run_two_publishers(tmp_path: Path, payloads: tuple[bytes, bytes]) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    cache_path = str(tmp_path / "cache")
    identity = cache_identity()
    workers = [
        context.Process(
            target=_publish_in_process,
            args=(cache_path, identity.model_dump(mode="json"), barrier, payload, queue),
        )
        for payload in payloads
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert worker.exitcode == 0
    return [queue.get(timeout=5) for _ in workers]


def test_two_processes_publish_one_validated_artifact(tmp_path: Path) -> None:
    results = _run_two_publishers(tmp_path, (b"same", b"same"))
    assert {item[0] for item in results} == {"PASS"}
    store = CacheStore(tmp_path / "cache")
    assert store.load_payload(cache_identity()) == b"same"
    assert len(store.iter_validated()) == 1


def test_conflicting_same_key_writers_cannot_publish_two_payloads(tmp_path: Path) -> None:
    results = _run_two_publishers(tmp_path, (b"left", b"right"))
    assert sorted(item[0] for item in results) == ["FAIL_CACHE_INTEGRITY", "PASS"]
    assert len(CacheStore(tmp_path / "cache").iter_validated()) == 1


def test_crashed_lock_holder_releases_operating_system_lock(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    identity = cache_identity()
    worker = context.Process(
        target=_acquire_then_exit,
        args=(str(tmp_path / "cache"), identity.cache_key, acquired),
    )
    worker.start()
    assert acquired.wait(timeout=20)
    worker.join(timeout=20)
    assert worker.exitcode == 0
    # The lock file intentionally remains. FileLock's OS lock, not age/deletion, controls ownership.
    with CacheKeyLock(tmp_path / "cache", identity.cache_key, timeout=2):
        assert True


def test_stale_lock_file_without_a_holder_does_not_block(tmp_path: Path) -> None:
    identity = cache_identity()
    lock = CacheKeyLock(tmp_path / "cache", identity.cache_key, timeout=2)
    lock.path.parent.mkdir(parents=True)
    lock.path.write_text("stale metadata is not ownership", encoding="utf-8")
    with lock:
        assert lock.path.exists()
