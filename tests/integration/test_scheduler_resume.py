from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from schemaguard.cache.index import CacheIndex
from schemaguard.cache.store import CacheStore
from schemaguard.runner.contracts import build_plan, build_task
from schemaguard.runner.scheduler import Scheduler
from tests.unit.cache_scheduler_helpers import cache_identity, scheduler_config

pytestmark = pytest.mark.integration


def _system(tmp_path: Path) -> tuple[Scheduler, CacheStore]:
    config = scheduler_config(cache_root="cache", state_root="runtime")
    cache = CacheStore(tmp_path / config.cache_root, lock_timeout_seconds=20)
    index = CacheIndex(tmp_path / config.state_root / "cache_index.sqlite", cache.root)
    return (
        Scheduler(tmp_path, config, cache, index, source_commit="a" * 40),
        cache,
    )


def _plan(count: int = 4, *, delay: float = 0.2):
    tasks = []
    for index in range(count):
        identity = cache_identity(
            seed=1729 + index,
            model_parameters_sha256=f"{index + 1:x}" * 64,
        )
        tasks.append(
            build_task(
                f"resume-task-{index}",
                identity,
                payload_token=f"value-{index}",
                operation="sleep" if delay else "synthetic_payload",
                delay_seconds=delay,
            )
        )
    return build_plan(tasks, random_seed=1729, cpu_workers=2)


def test_cold_execution_resume_and_corruption_recomputation(tmp_path: Path) -> None:
    scheduler, cache = _system(tmp_path)
    plan = _plan()
    cold = scheduler.run(plan)
    assert cold.manifest.executed == 4
    assert cold.manifest.validated_cache_hits == 0
    assert cold.manifest.failed == 0
    assert cold.manifest.max_cpu_concurrency == 2
    assert cold.manifest.offline_network_attempt_count == 0
    assert len(cache.iter_validated()) == 4

    state = scheduler.run(plan, resume=True)
    assert state.manifest.executed == 0
    assert state.manifest.validated_cache_hits == 4
    assert state.manifest.failed == 0
    for task in plan.tasks:
        record = json.loads(cache.load_payload(task.cache_identity))
        assert record["thread_environment"] == {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
        persisted = (
            __import__("schemaguard.runner.state", fromlist=["TaskStateStore"])
            .TaskStateStore(tmp_path / "runtime", plan)
            .read()
            .tasks[task.task_id]
        )
        assert persisted.attempts[0].worker_pid != os.getpid()

    corrupt_task = plan.tasks[0]
    cache.artifact_path(corrupt_task.cache_identity).write_bytes(b"truncated")
    repaired = scheduler.run(plan, resume=True)
    assert repaired.manifest.executed == 1
    assert repaired.manifest.validated_cache_hits == 3
    assert len(cache.iter_validated()) == 4
    assert len({item.cache_key for item in cache.iter_validated()}) == 4


def test_cache_hit_executes_no_worker_and_reordered_plan_has_same_identity(
    tmp_path: Path,
) -> None:
    scheduler, _ = _system(tmp_path)
    plan = _plan(2, delay=0)
    first = scheduler.run(plan)
    reordered = build_plan(list(reversed(plan.tasks)), random_seed=1729, cpu_workers=2)
    assert first.manifest.executed == 2
    assert reordered.plan_hash == plan.plan_hash
    second = scheduler.run(reordered, resume=True)
    assert second.manifest.executed == 0
    assert second.manifest.validated_cache_hits == 2
