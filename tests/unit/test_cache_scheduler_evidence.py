from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemaguard.cache.contracts import (
    CacheSchedulerInventory,
    ProbeResourceSummary,
    ProbeRunSummary,
    ProtectedHashComparison,
    ProtectedHashRecord,
    SchedulerProbeEvidence,
)


def test_protected_hash_comparison_rejects_forged_status_and_private_paths() -> None:
    record = {
        "path": "configs/runtime/example.yaml",
        "before_sha256": "a" * 64,
        "after_sha256": "a" * 64,
        "baseline_git_sha256": "b" * 64,
        "implementation_git_sha256": "b" * 64,
        "before_size_bytes": 10,
        "after_size_bytes": 10,
        "unchanged": True,
    }
    comparison = ProtectedHashComparison.model_validate(
        {
            "schema_version": 1,
            "baseline_commit": "c" * 40,
            "implementation_commit": "d" * 40,
            "expected_file_count": 1,
            "records": [record],
            "status": "PASS",
        }
    )
    assert comparison.status == "PASS"
    with pytest.raises(ValidationError):
        ProtectedHashRecord.model_validate({**record, "path": "C:/Users/private/file.txt"})
    with pytest.raises(ValidationError):
        ProtectedHashRecord.model_validate({**record, "unchanged": False})


def test_probe_evidence_requires_complete_offline_cold_and_cache_hit_resume() -> None:
    resource = ProbeResourceSummary(
        wall_seconds=0.1,
        cpu_seconds=0.02,
        peak_worker_rss_mib=12,
        peak_process_tree_rss_mib=20,
        telemetry_complete=True,
        hard_limit_passed=True,
        cleanup_passed=True,
        failure_category=None,
    )
    cold = ProbeRunSummary(
        mode="cold",
        planned=1,
        executed=1,
        validated_cache_hits=0,
        recovered_abandoned=0,
        failed=0,
        blocked=0,
        retried=0,
        cancelled=0,
        max_cpu_concurrency=1,
        max_gpu_concurrency=0,
        offline_network_attempt_count=0,
        resources=[resource],
    )
    resume = ProbeRunSummary(
        mode="resume",
        planned=1,
        executed=0,
        validated_cache_hits=1,
        recovered_abandoned=0,
        failed=0,
        blocked=0,
        retried=0,
        cancelled=0,
        max_cpu_concurrency=0,
        max_gpu_concurrency=0,
        offline_network_attempt_count=0,
        resources=[],
    )
    evidence = SchedulerProbeEvidence(
        schema_version=1,
        stage="cache_scheduler_probe",
        source_implementation_commit="c" * 40,
        source_implementation_sha256="a" * 64,
        dependency_lock_sha256="b" * 64,
        configuration_sha256="d" * 64,
        plan_sha256="e" * 64,
        cache_identity_hashes=["f" * 64],
        cold=cold,
        resume=resume,
        maximum_mocked_gpu_concurrency=1,
        mocked_gpu_offline_network_attempt_count=0,
        validated_cache_artifact_count=1,
        duplicate_validated_artifact_count=0,
    )
    assert evidence.resume.validated_cache_hits == 1
    with pytest.raises(ValidationError):
        ProbeRunSummary.model_validate(
            {**resume.model_dump(), "executed": 1, "validated_cache_hits": 0}
        )


def test_passing_inventory_cannot_claim_protected_changes_or_network_attempts() -> None:
    valid = {
        "schema_version": 1,
        "stage": "cache_scheduler",
        "status": "PASS_PENDING_REVIEW",
        "source_implementation_commit": "a" * 40,
        "implementation_hashes": {"src/schemaguard/runner/worker.py": "b" * 64},
        "configuration_sha256": "c" * 64,
        "plan_sha256": "d" * 64,
        "cache_identity_hashes": ["e" * 64],
        "test_case_names": ["tests.unit.test_cache_keys::test_keys"],
        "test_statuses": {"unit": "PASS"},
        "counters": {"duplicate_artifacts": 0},
        "maximum_observed_cpu_concurrency": 1,
        "maximum_observed_gpu_concurrency": 1,
        "resource_summaries": {"peak_vram_reserved_mib": None},
        "fault_evidence_sha256": "f" * 64,
        "probe_evidence_sha256": "1" * 64,
        "resume_results": {"executed": 0, "validated_cache_hits": 1},
        "offline_network_attempt_count": 0,
        "protected_artifacts_unchanged": True,
        "protected_hash_comparison_sha256": "2" * 64,
    }
    assert CacheSchedulerInventory.model_validate(valid).status == "PASS_PENDING_REVIEW"
    with pytest.raises(ValidationError, match="zero probe network attempts"):
        CacheSchedulerInventory.model_validate({**valid, "offline_network_attempt_count": 1})
    with pytest.raises(ValidationError, match="unchanged protected artifacts"):
        CacheSchedulerInventory.model_validate({**valid, "protected_artifacts_unchanged": False})
