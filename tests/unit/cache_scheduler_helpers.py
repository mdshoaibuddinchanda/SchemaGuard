from __future__ import annotations

from schemaguard.cache.contracts import CacheIdentity, CacheSchedulerConfig
from schemaguard.runner.contracts import RunPlan, TaskSpec, build_plan, build_task
from schemaguard.utils.hashing import sha256_canonical_json


def cache_identity(**changes: object) -> CacheIdentity:
    value: dict[str, object] = {
        "schema_version": 1,
        "dataset_sha256": "1" * 64,
        "split_sha256": "2" * 64,
        "view_certificate_sha256": "3" * 64,
        "model_spec_sha256": "4" * 64,
        "model_parameters_sha256": "5" * 64,
        "checkpoint_sha256": None,
        "dependency_lock_sha256": "6" * 64,
        "source_implementation_sha256": "7" * 64,
        "seed": 1729,
        "device_policy": "cpu",
        "artifact_kind": "probe",
    }
    value.update(changes)
    return CacheIdentity.model_validate(value)


def scheduler_config(**changes: object) -> CacheSchedulerConfig:
    value: dict[str, object] = {
        "schema_version": 1,
        "cpu_workers": 2,
        "cpu_threads_per_worker": 2,
        "gpu_workers": 1,
        "ram_soft_limit_mib": 24576,
        "ram_hard_limit_mib": 28672,
        "vram_soft_limit_mib": 3600,
        "gpu_headroom_mib": 512,
        "lock_timeout_seconds": 120,
        "heartbeat_interval_seconds": 5,
        "abandoned_lease_seconds": 60,
        "cache_root": "data/cache/conditions",
        "state_root": "artifacts/cache_scheduler/runtime",
    }
    value.update(changes)
    return CacheSchedulerConfig.model_validate(value)


def task(name: str = "synthetic-task", identity: CacheIdentity | None = None) -> TaskSpec:
    return build_task(name, identity or cache_identity(), payload_token=f"payload-{name}")


def plan(tasks: list[TaskSpec] | None = None) -> RunPlan:
    return build_plan(tasks or [task()], random_seed=1729, cpu_workers=2)


def digest(value: object) -> str:
    return sha256_canonical_json(value)
