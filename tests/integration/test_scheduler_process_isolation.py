from __future__ import annotations

from pathlib import Path

import pytest

from schemaguard.cache.index import CacheIndex
from schemaguard.cache.store import CacheStore
from schemaguard.runner.contracts import build_plan, build_task
from schemaguard.runner.scheduler import Scheduler
from tests.unit.cache_scheduler_helpers import cache_identity, scheduler_config

pytestmark = pytest.mark.integration


def _scheduler(tmp_path: Path, *, gpu: bool = False) -> Scheduler:
    config = scheduler_config(cache_root="cache", state_root="runtime")
    root = tmp_path
    cache = CacheStore(root / "cache", lock_timeout_seconds=10)
    index = CacheIndex(root / "runtime" / "index.sqlite", cache.root)

    def sampler() -> dict[str, float]:
        return {
            "total_mib": 4096.0,
            "free_mib": 3900.0,
            "allocated_mib": 128.0,
            "reserved_mib": 128.0,
        }

    return Scheduler(
        root,
        config,
        cache,
        index,
        source_commit="b" * 40,
        gpu_sampler=sampler if gpu else None,
        gpu_budget_check=(lambda: None) if gpu else None,
    )


def test_timeout_and_worker_exit_without_result_fail_closed(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    timeout_task = build_task(
        "timed-worker",
        cache_identity(seed=1984, model_parameters_sha256="8" * 64),
        operation="sleep",
        payload_token="timeout",
        delay_seconds=2,
        timeout_seconds=0.5,
    )
    timeout_result = scheduler.run(build_plan([timeout_task], random_seed=1729))
    assert timeout_result.manifest.failed == 1
    assert timeout_result.manifest.resources[0].timed_out
    assert timeout_result.manifest.resources[0].failure_category == "FAIL_TIMEOUT"
    assert not CacheStore(tmp_path / "cache").iter_validated()

    crash_task = build_task(
        "crashing-worker",
        cache_identity(seed=1985, model_parameters_sha256="9" * 64),
        operation="exit_without_result",
        payload_token="crash",
    )
    crash_result = scheduler.run(build_plan([crash_task], random_seed=1729))
    assert crash_result.manifest.failed == 1
    assert crash_result.manifest.failures[0].category == "FAIL_MODEL_RUNTIME"
    assert not CacheStore(tmp_path / "cache").read_validated(crash_task.cache_identity)


def test_cpu_resource_limit_failure_does_not_publish(monkeypatch, tmp_path: Path) -> None:
    import schemaguard.runner.resources as resources

    monkeypatch.setattr(resources, "_sample_process_tree", lambda _: (30000.0, 30000.0, 0.1))
    scheduler = _scheduler(tmp_path)
    task = build_task(
        "over-limit-worker",
        cache_identity(seed=1986, model_parameters_sha256="a" * 64),
    )
    result = scheduler.run(build_plan([task], random_seed=1729))
    assert result.manifest.failed == 1
    assert result.manifest.resources[0].failure_category == "FAIL_RESOURCE_LIMIT"
    assert not CacheStore(tmp_path / "cache").read_validated(task.cache_identity)


def test_gpu_queue_is_exclusive_and_releases_lock_on_failure(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path, gpu=True)
    tasks = [
        build_task(
            f"gpu-mock-{index}",
            cache_identity(
                seed=3000 + index,
                device_policy="cuda",
                model_parameters_sha256=f"{index + 1:x}" * 64,
            ),
            operation="sleep",
            payload_token=f"gpu-{index}",
            delay_seconds=0.15,
        )
        for index in range(2)
    ]
    result = scheduler.run(build_plan(tasks, random_seed=1729))
    assert result.manifest.executed == 2
    assert result.manifest.max_gpu_concurrency == 1
    assert all(resource.telemetry_complete for resource in result.manifest.resources)

    failed = build_task(
        "gpu-fault",
        cache_identity(seed=4000, device_policy="cuda", model_parameters_sha256="c" * 64),
        operation="raise_error",
        payload_token="gpu-error",
    )
    failed_result = scheduler.run(build_plan([failed], random_seed=1729))
    assert failed_result.manifest.failed == 1
    assert failed_result.manifest.failures[0].category == "FAIL_MODEL_RUNTIME"
    lock_path = tmp_path / "data/cache/locks/gpu-capacity.lock"
    from schemaguard.utils.process_lock import ProcessLock

    with ProcessLock(lock_path, timeout=2):
        assert True
