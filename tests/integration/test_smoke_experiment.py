from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from schemaguard.cache.store import CacheStore
from schemaguard.experiments import execution, planning
from schemaguard.experiments.evaluation import evaluate_smoke, expected_prediction_paths
from schemaguard.experiments.planning import load_smoke_config, write_plan
from schemaguard.utils.hashing import sha256_file
from tests.integration.smoke_experiment_support import fake_smoke_worker_entry, tiny_smoke_plan

pytestmark = pytest.mark.integration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_tiny_smoke_plan_cold_execution_metrics_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler_path = tmp_path / "configs/runtime/cache_scheduler.yaml"
    scheduler_path.parent.mkdir(parents=True)
    scheduler_config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/runtime/cache_scheduler.yaml").read_text(encoding="utf-8")
    )
    scheduler_config["cache_root"] = "cache/conditions"
    scheduler_config["state_root"] = "runtime/scheduler"
    scheduler_path.write_text(yaml.safe_dump(scheduler_config), encoding="utf-8")

    config = load_smoke_config(REPOSITORY_ROOT / "configs/runtime/smoke_experiment.yaml")
    plan = tiny_smoke_plan(tmp_path)
    write_plan(tmp_path, plan)
    monkeypatch.setattr(execution, "smoke_worker_entry", fake_smoke_worker_entry)

    base_scheduler = execution._AdapterOwnedGpuScheduler

    class FixtureTelemetryScheduler(base_scheduler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                *args,
                gpu_sampler=lambda: {
                    "free_mib": 4096.0,
                    "allocated_mib": 0.0,
                    "reserved_mib": 0.0,
                },
                gpu_budget_check=lambda: None,
                **kwargs,
            )

    monkeypatch.setattr(execution, "_AdapterOwnedGpuScheduler", FixtureTelemetryScheduler)
    cold_runs, cold_result, cold_network = execution.run_conditions(
        tmp_path, plan, config, resume=False
    )
    assert cold_result.manifest.planned == 10
    assert cold_result.manifest.executed == 10
    assert cold_result.manifest.validated_cache_hits == 0
    assert cold_result.manifest.failed == cold_result.manifest.blocked == 0
    assert cold_result.manifest.max_cpu_concurrency <= 2
    assert cold_result.manifest.max_gpu_concurrency == 1
    assert cold_network == 0
    assert all(item.status == "PASS" and item.cache_status == "created" for item in cold_runs)
    assert len(expected_prediction_paths(tmp_path, cold_runs)) == 20

    before = {
        path.relative_to(tmp_path).as_posix(): (sha256_file(path), path.stat().st_mtime_ns)
        for path in expected_prediction_paths(tmp_path, cold_runs)
    }
    resume_runs, resume_result, resume_network = execution.run_conditions(
        tmp_path, plan, config, resume=True
    )
    after = {
        path.relative_to(tmp_path).as_posix(): (sha256_file(path), path.stat().st_mtime_ns)
        for path in expected_prediction_paths(tmp_path, resume_runs)
    }
    assert resume_result.manifest.executed == 0
    assert resume_result.manifest.validated_cache_hits == 10
    assert resume_result.manifest.failed == resume_result.manifest.blocked == 0
    assert resume_network == 0
    assert all(item.cache_status == "validated_cache_hit" for item in resume_runs)
    assert after == before

    monkeypatch.setattr(planning, "validate_views_at_evaluation_boundary", lambda _: None)
    metrics, paired, _, _ = evaluate_smoke(tmp_path, plan, resume_runs)
    assert len(metrics) == 20
    assert len(paired) == 10
    assert all(record.row_count == 2 for record in metrics)
    assert all(record.sii_max_js_divergence == 0.0 for record in paired)
    assert all(record.label_flip_rate == 0.0 for record in paired)

    _, identities = execution._scheduler_plan(plan)
    cache = CacheStore(tmp_path / "cache/conditions")
    first = plan.conditions[0]
    another = plan.conditions[1]
    first_artifact = cache.read_validated(identities[first.condition_id])
    assert first_artifact is not None
    with pytest.raises(ValueError, match="condition metadata"):
        execution._load_worker_archive(
            first_artifact, another, plan, identities[another.condition_id]
        )

    prediction = tmp_path / next(iter(before))
    prediction.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="missing or changed"):
        expected_prediction_paths(tmp_path, resume_runs)
