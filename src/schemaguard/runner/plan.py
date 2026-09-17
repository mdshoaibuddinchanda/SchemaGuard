"""Deterministic synthetic scheduler plans and strict runtime configuration loading."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..cache.contracts import CacheIdentity, CacheSchedulerConfig
from ..cache.keys import dependency_lock_hash, implementation_hash
from ..utils.hashing import canonical_source_hash, sha256_canonical_json
from .contracts import RunPlan, TaskSpec, build_plan, build_task

IMPLEMENTATION_PATHS = (
    "src/schemaguard/cache/__init__.py",
    "src/schemaguard/cache/contracts.py",
    "src/schemaguard/cache/fault_policy.py",
    "src/schemaguard/cache/keys.py",
    "src/schemaguard/cache/locks.py",
    "src/schemaguard/cache/store.py",
    "src/schemaguard/cache/index.py",
    "src/schemaguard/runner/__init__.py",
    "src/schemaguard/runner/contracts.py",
    "src/schemaguard/runner/faults.py",
    "src/schemaguard/runner/plan.py",
    "src/schemaguard/runner/resources.py",
    "src/schemaguard/runner/scheduler.py",
    "src/schemaguard/runner/state.py",
    "src/schemaguard/runner/worker.py",
    "src/schemaguard/utils/hashing.py",
    "src/schemaguard/utils/io.py",
    "src/schemaguard/utils/process_lock.py",
    "src/schemaguard/models/adapters/resources.py",
    "scripts/run_cache_scheduler_probe.py",
    "scripts/validate_cache_scheduler.py",
)


def load_config(path: str | Path) -> CacheSchedulerConfig:
    """Read YAML once and reject every extra, missing, or invalid field."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("cache/scheduler configuration must be a YAML mapping")
    return CacheSchedulerConfig.model_validate(payload)


def runtime_configuration_hash(path: str | Path) -> str:
    """Hash the strict YAML configuration independently of checkout line endings."""

    return canonical_source_hash(path)


def current_commit(root: str | Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(root),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    value = completed.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Git returned an invalid source commit")
    return value


def make_probe_plan(root: str | Path, config: CacheSchedulerConfig) -> tuple[RunPlan, str, str]:
    """Build four fixed CPU-only synthetic condition tasks, independent of input ordering."""

    project = Path(root).resolve()
    code_hash = implementation_hash(project, IMPLEMENTATION_PATHS)
    lock_hash = dependency_lock_hash(project / "uv.lock")
    tasks: list[TaskSpec] = []
    for index, suffix in enumerate(("alpha", "bravo", "charlie", "delta")):
        identity = CacheIdentity(
            schema_version=1,
            dataset_sha256=sha256_canonical_json({"fixture": "synthetic-dataset-v1"}),
            split_sha256=sha256_canonical_json({"split": "synthetic-no-label-split-v1"}),
            view_certificate_sha256=sha256_canonical_json({"view": "identity-probe-v1"}),
            model_spec_sha256=sha256_canonical_json({"worker": "synthetic-cache-probe-v1"}),
            model_parameters_sha256=sha256_canonical_json(
                {"payload_token": f"payload-{suffix}", "seed": 1729 + index}
            ),
            checkpoint_sha256=None,
            dependency_lock_sha256=lock_hash,
            source_implementation_sha256=code_hash,
            seed=1729 + index,
            device_policy="cpu",
            artifact_kind="probe",
        )
        tasks.append(
            build_task(
                f"synthetic-cache-{suffix}",
                identity,
                payload_token=f"payload-{suffix}",
            )
        )
    return build_plan(tasks, random_seed=1729, cpu_workers=config.cpu_workers), code_hash, lock_hash


__all__ = [
    "IMPLEMENTATION_PATHS",
    "current_commit",
    "load_config",
    "make_probe_plan",
    "runtime_configuration_hash",
]
