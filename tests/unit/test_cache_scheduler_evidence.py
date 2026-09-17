from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from schemaguard.cache.contracts import (
    CacheSchedulerInventory,
    FaultInjectionEvidence,
    FaultInjectionRecord,
    ProbeResourceSummary,
    ProbeRunSummary,
    ProtectedHashComparison,
    ProtectedHashRecord,
    SchedulerProbeEvidence,
    validate_fault_evidence_binding,
)
from schemaguard.cache.fault_policy import (
    CANONICAL_FAULT_NAMES,
    FAULT_POLICY,
    fault_evidence_sha256,
)
from schemaguard.utils.hashing import canonical_source_hash
from schemaguard.utils.io import atomic_write_json


def _valid_fault_evidence_payload() -> dict[str, object]:
    commit = "a" * 40
    records = []
    for name, expectation in FAULT_POLICY.items():
        record: dict[str, object] = {
            "fault_name": name,
            "injection_point": expectation.injection_point,
            "expected_failure_category": expectation.expected_failure_category,
            "observed_failure_category": expectation.expected_failure_category,
            "expected_artifact_accepted": expectation.expected_artifact_accepted,
            "artifact_accepted": expectation.expected_artifact_accepted,
            "expected_resume_behavior": expectation.expected_resume_behavior,
            "resume_behavior": expectation.expected_resume_behavior,
            "lock_released": True,
        }
        record["evidence_sha256"] = fault_evidence_sha256(record)
        record["status"] = "PASS"
        records.append(record)
    return {
        "schema_version": 1,
        "stage": "cache_scheduler_faults",
        "source_implementation_commit": commit,
        "records": records,
        "expected_fault_count": 30,
        "offline_network_attempt_count": 1,
    }


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


@pytest.mark.parametrize(
    "field",
    (
        "fault_name",
        "injection_point",
        "expected_failure_category",
        "observed_failure_category",
        "expected_artifact_accepted",
        "artifact_accepted",
        "expected_resume_behavior",
        "resume_behavior",
        "lock_released",
        "evidence_sha256",
        "status",
    ),
)
def test_fault_evidence_rejects_each_single_field_tamper(field: str) -> None:
    payload = _valid_fault_evidence_payload()
    tampered = copy.deepcopy(payload)
    record = tampered["records"][0]  # type: ignore[index]
    assert isinstance(record, dict)
    if field in {"expected_artifact_accepted", "artifact_accepted", "lock_released"}:
        record[field] = not record[field]
    elif field == "status":
        record[field] = "FAIL"
    elif field == "evidence_sha256":
        record[field] = "0" * 64
    else:
        record[field] = f"tampered-{field}"
    with pytest.raises(ValidationError):
        FaultInjectionEvidence.model_validate(tampered)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "unknown", "twenty_nine", "thirty_one", "reordered"),
)
def test_fault_evidence_rejects_incomplete_duplicate_or_noncanonical_matrix(
    mutation: str,
) -> None:
    payload = _valid_fault_evidence_payload()
    tampered = copy.deepcopy(payload)
    records = tampered["records"]  # type: ignore[index]
    assert isinstance(records, list)
    if mutation in {"missing", "twenty_nine"}:
        tampered["records"] = records[:-1]
    elif mutation == "duplicate":
        tampered["records"] = records[:-1] + [copy.deepcopy(records[0])]
    elif mutation == "unknown":
        unknown_record = copy.deepcopy(records[0])
        unknown_record["fault_name"] = "unknown_fault"
        tampered["records"] = [unknown_record, *records[1:]]
    elif mutation == "thirty_one":
        tampered["records"] = [*records, copy.deepcopy(records[0])]
    else:
        tampered["records"] = list(reversed(records))
    with pytest.raises(ValidationError):
        FaultInjectionEvidence.model_validate(tampered)


@pytest.mark.parametrize("attempt_count", (0, 2))
def test_fault_evidence_requires_exactly_one_denied_network_attempt(
    attempt_count: int,
) -> None:
    payload = _valid_fault_evidence_payload()
    with pytest.raises(ValidationError):
        FaultInjectionEvidence.model_validate(
            {**payload, "offline_network_attempt_count": attempt_count}
        )


def test_fault_evidence_policy_is_complete_and_canonical() -> None:
    payload = _valid_fault_evidence_payload()
    evidence = FaultInjectionEvidence.model_validate(payload)
    assert tuple(record.fault_name for record in evidence.records) == CANONICAL_FAULT_NAMES
    assert len(FAULT_POLICY) == evidence.expected_fault_count == 30
    assert all(record.status == "PASS" for record in evidence.records)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observed_failure_category", "FAIL_MODEL_RUNTIME"),
        ("artifact_accepted", True),
        ("resume_behavior", "UNSAFE_REUSE"),
        ("lock_released", False),
    ),
)
def test_fault_status_is_derived_even_when_tamper_hash_is_recomputed(
    field: str, value: object
) -> None:
    payload = _valid_fault_evidence_payload()
    record = copy.deepcopy(payload["records"][0])  # type: ignore[index]
    assert isinstance(record, dict)
    record[field] = value
    record["evidence_sha256"] = fault_evidence_sha256(record)
    with pytest.raises(ValidationError, match="status contradicts"):
        FaultInjectionRecord.model_validate(record)


def test_fault_evidence_binding_rejects_commit_and_whole_file_digest_tampering(
    tmp_path,
) -> None:
    payload = _valid_fault_evidence_payload()
    evidence = FaultInjectionEvidence.model_validate(payload)
    evidence_path = tmp_path / "fault-evidence.json"
    atomic_write_json(evidence_path, evidence.model_dump(mode="json"))
    digest = canonical_source_hash(evidence_path)
    validate_fault_evidence_binding(
        evidence,
        inventory_source_implementation_commit=evidence.source_implementation_commit,
        inventory_fault_evidence_sha256=digest,
        observed_fault_evidence_sha256=digest,
    )

    altered_commit = FaultInjectionEvidence.model_validate(
        {**payload, "source_implementation_commit": "b" * 40}
    )
    with pytest.raises(ValueError, match="source commit"):
        validate_fault_evidence_binding(
            altered_commit,
            inventory_source_implementation_commit=evidence.source_implementation_commit,
            inventory_fault_evidence_sha256=digest,
            observed_fault_evidence_sha256=digest,
        )
    with pytest.raises(ValueError, match="whole-file digest"):
        validate_fault_evidence_binding(
            evidence,
            inventory_source_implementation_commit=evidence.source_implementation_commit,
            inventory_fault_evidence_sha256="0" * 64,
            observed_fault_evidence_sha256=digest,
        )
