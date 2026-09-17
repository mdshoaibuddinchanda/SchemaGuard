from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from schemaguard.runner.contracts import RunPlan, TaskResult, build_plan, build_task
from schemaguard.runner.plan import load_config, runtime_configuration_hash
from tests.unit.cache_scheduler_helpers import cache_identity, task


def test_plan_order_and_hash_do_not_depend_on_input_order() -> None:
    tasks = [task("task-c"), task("task-a"), task("task-b")]
    first = build_plan(tasks, random_seed=1729, cpu_workers=2)
    second = build_plan(list(reversed(tasks)), random_seed=1729, cpu_workers=2)
    assert [item.task_id for item in first.tasks] == [item.task_id for item in second.tasks]
    assert first.plan_hash == second.plan_hash


def test_plan_rejects_duplicate_ids_and_noncanonical_order() -> None:
    original = task("task-a")
    with pytest.raises(ValidationError):
        RunPlan(
            schema_version=1,
            tasks=[original, original],
            random_seed=1729,
            cpu_workers=2,
            gpu_workers=1,
        )
    reversed_tasks = [task("task-a"), task("task-b")]
    ordered = build_plan(reversed_tasks, random_seed=1729)
    with pytest.raises(ValidationError):
        RunPlan(
            schema_version=1,
            tasks=list(reversed(ordered.tasks)),
            random_seed=1729,
            cpu_workers=2,
            gpu_workers=1,
        )


def test_task_identity_binds_parameters_and_config_is_strict(tmp_path: Path) -> None:
    original = build_task("named-task", cache_identity(), payload_token="first")
    changed = build_task("named-task", cache_identity(), payload_token="second")
    assert original.task_id != changed.task_id
    config_path = Path(__file__).resolve().parents[2] / "configs/runtime/cache_scheduler.yaml"
    config = load_config(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["unrecognized"] = True
    unknown_config = tmp_path / "unknown.yaml"
    unknown_config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(unknown_config)
    assert config.gpu_workers == 1


def test_runtime_configuration_hash_is_stable_across_checkout_line_endings(
    tmp_path: Path,
) -> None:
    source_path = Path(__file__).resolve().parents[2] / "configs/runtime/cache_scheduler.yaml"
    normalized = source_path.read_bytes().replace(b"\r\n", b"\n")
    lf_path = tmp_path / "config-lf.yaml"
    crlf_path = tmp_path / "config-crlf.yaml"
    lf_path.write_bytes(normalized)
    crlf_path.write_bytes(normalized.replace(b"\n", b"\r\n"))

    assert load_config(lf_path) == load_config(crlf_path)
    assert runtime_configuration_hash(lf_path) == runtime_configuration_hash(crlf_path)


def test_worker_result_is_strict_and_requires_thread_limits() -> None:
    valid = {
        "schema_version": 1,
        "ok": True,
        "task_id": "a" * 64,
        "payload_sha256": "b" * 64,
        "payload_size_bytes": 12,
        "worker_pid": 123,
        "worker_rss_mib": 5.0,
        "cpu_seconds": 0.01,
        "worker_seconds": 0.02,
        "network_attempt_count": 0,
        "thread_environment": {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        },
    }
    assert TaskResult.model_validate(valid).ok
    with pytest.raises(ValidationError, match="two-thread limits"):
        TaskResult.model_validate({**valid, "thread_environment": {"OMP_NUM_THREADS": "2"}})
    with pytest.raises(ValidationError):
        TaskResult.model_validate({**valid, "unrecognized": True})
