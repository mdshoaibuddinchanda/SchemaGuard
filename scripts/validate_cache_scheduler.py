"""Generate or validate sanitized cache/scheduler evidence without running research jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError  # noqa: E402

from schemaguard.cache.contracts import (  # noqa: E402
    CacheSchedulerInventory,
    FaultInjectionEvidence,
    ProbeResourceSummary,
    ProbeRunSummary,
    ProtectedHashComparison,
    ProtectedHashRecord,
    SchedulerProbeEvidence,
)
from schemaguard.cache.index import CacheIndex  # noqa: E402
from schemaguard.cache.keys import dependency_lock_hash, implementation_hash  # noqa: E402
from schemaguard.cache.store import CacheStore  # noqa: E402
from schemaguard.runner.contracts import RunManifest, build_plan, build_task  # noqa: E402
from schemaguard.runner.faults import run_fault_injection_suite  # noqa: E402
from schemaguard.runner.plan import (  # noqa: E402
    IMPLEMENTATION_PATHS,
    current_commit,
    load_config,
    make_probe_plan,
    runtime_configuration_hash,
)
from schemaguard.runner.scheduler import Scheduler  # noqa: E402
from schemaguard.utils.hashing import (  # noqa: E402
    canonical_source_hash,
    sha256_canonical_json,
    sha256_file,
)
from schemaguard.utils.io import atomic_write_json, atomic_write_text  # noqa: E402

STARTING_COMMIT = "cc8e2a53ceb73e833d54917dd195eb99bf01719c"
PROTECTED_BEFORE = Path("artifacts/cache_scheduler/runtime/protected_hashes_before.json")
FAULT_NAMES = {
    "payload_write_interrupted",
    "completion_interrupted",
    "manifest_truncated",
    "payload_truncated",
    "completion_marker_missing",
    "payload_checksum_wrong",
    "manifest_identity_wrong",
    "cache_key_wrong",
    "artifact_schema_wrong",
    "failed_artifact_candidate",
    "model_configuration_change",
    "implementation_change",
    "dependency_lock_change",
    "dataset_checksum_change",
    "split_checksum_change",
    "view_certificate_change",
    "checkpoint_checksum_change",
    "device_policy_change",
    "two_process_same_identity_publication",
    "same_key_different_payload",
    "lock_holder_process_crash",
    "stale_lock_file_without_owner",
    "worker_exceeds_hard_ram",
    "worker_timeout",
    "worker_exit_without_result",
    "restart_with_running_task",
    "missing_cache_index",
    "corrupt_cache_index",
    "complete_state_without_valid_payload",
    "network_attempt_during_offline_execution",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--fault-evidence", type=Path, required=True)
    parser.add_argument("--probe-evidence", type=Path)
    parser.add_argument("--protected-comparison", type=Path)
    parser.add_argument(
        "--record",
        action="store_true",
        help="execute bounded tests and probes, then write sanitized evidence",
    )
    return parser


def _inside_repository(path: Path, *, label: str) -> Path:
    candidate = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the repository") from exc
    return candidate


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=ROOT,
            capture_output=True,
            check=False,
            timeout=30,
        ).returncode
        == 0
    )


def _implementation_file_hashes() -> dict[str, str]:
    return {path: canonical_source_hash(ROOT / path) for path in sorted(IMPLEMENTATION_PATHS)}


def _compare_protected_files(
    implementation_commit: str, output_path: Path
) -> ProtectedHashComparison:
    before_path = ROOT / PROTECTED_BEFORE
    before = json.loads(before_path.read_text(encoding="utf-8"))
    if (
        before.get("schema_version") != 1
        or before.get("baseline_commit") != STARTING_COMMIT
        or before.get("protected_file_count") != len(before.get("files", []))
    ):
        raise ValueError("protected pre-implementation snapshot is malformed or from another base")
    records: list[ProtectedHashRecord] = []
    for entry in sorted(before["files"], key=lambda item: item["path"]):
        relative = entry["path"]
        if not isinstance(relative, str):
            raise ValueError("protected snapshot contains a non-string relative path")
        posix = PurePosixPath(relative)
        windows = PureWindowsPath(relative)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or "\\" in relative:
            raise ValueError("protected snapshot contains a non-portable path")
        current_path = _inside_repository(Path(relative), label="protected file")
        if not current_path.is_file():
            after_sha = "0" * 64
            after_size = 0
        else:
            after_sha = sha256_file(current_path)
            after_size = current_path.stat().st_size
        baseline_git = _git_blob_sha256(STARTING_COMMIT, relative)
        implementation_git = _git_blob_sha256(implementation_commit, relative)
        unchanged = (
            entry.get("sha256") == after_sha
            and entry.get("size_bytes") == after_size
            and baseline_git == implementation_git
        )
        records.append(
            ProtectedHashRecord(
                path=relative,
                before_sha256=entry["sha256"],
                after_sha256=after_sha,
                baseline_git_sha256=baseline_git,
                implementation_git_sha256=implementation_git,
                before_size_bytes=entry["size_bytes"],
                after_size_bytes=after_size,
                unchanged=unchanged,
            )
        )
    comparison = ProtectedHashComparison(
        schema_version=1,
        baseline_commit=STARTING_COMMIT,
        implementation_commit=implementation_commit,
        expected_file_count=before["protected_file_count"],
        records=records,
        status="PASS" if all(item.unchanged for item in records) else "FAIL",
    )
    atomic_write_json(output_path, comparison.model_dump(mode="json"))
    return comparison


def _validate_protected_comparison(
    path: Path, expected_implementation_commit: str
) -> ProtectedHashComparison:
    comparison = ProtectedHashComparison.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if comparison.baseline_commit != STARTING_COMMIT:
        raise ValueError("protected comparison does not use the required starting commit")
    if comparison.implementation_commit != expected_implementation_commit:
        raise ValueError("protected comparison implementation commit differs from the inventory")
    if not _git_ancestor(STARTING_COMMIT, expected_implementation_commit):
        raise ValueError("starting commit is not an ancestor of the implementation commit")
    current = current_commit(ROOT)
    if not _git_ancestor(expected_implementation_commit, current):
        raise ValueError("implementation commit is not an ancestor of the evidence checkout")
    for record in comparison.records:
        baseline_git = _git_blob_sha256(STARTING_COMMIT, record.path)
        implementation_git = _git_blob_sha256(expected_implementation_commit, record.path)
        current_git = _git_blob_sha256(current, record.path)
        if (
            baseline_git != record.baseline_git_sha256
            or implementation_git != record.implementation_git_sha256
            or current_git != implementation_git
        ):
            raise ValueError(f"protected Git-tree identity differs for {record.path}")
    return comparison


def _read_probe_receipt(
    path: Path,
    *,
    mode: str,
    implementation_commit: str,
    implementation_sha256: str,
    dependency_sha256: str,
    configuration_sha256: str,
    plan_hash: str,
) -> RunManifest:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "source_implementation_commit",
        "source_implementation_sha256",
        "dependency_lock_sha256",
        "configuration_sha256",
        "plan_hash",
        "mode",
        "run_manifest",
        "state_path",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError(f"{mode} probe receipt has unknown or missing fields")
    if receipt["schema_version"] != 1 or receipt["mode"] != mode:
        raise ValueError(f"{mode} probe receipt has an unsupported schema or mode")
    expected = {
        "source_implementation_commit": implementation_commit,
        "source_implementation_sha256": implementation_sha256,
        "dependency_lock_sha256": dependency_sha256,
        "configuration_sha256": configuration_sha256,
        "plan_hash": plan_hash,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{mode} probe receipt identity does not match the current sources")
    state_path = PurePosixPath(str(receipt["state_path"]))
    if (
        state_path.is_absolute()
        or PureWindowsPath(str(receipt["state_path"])).is_absolute()
        or ".." in state_path.parts
        or "\\" in str(receipt["state_path"])
    ):
        raise ValueError("probe receipt contains a non-portable state path")
    return RunManifest.model_validate(receipt["run_manifest"])


def _validate_probe_manifests(cold: RunManifest, resume: RunManifest, plan_size: int) -> None:
    if (
        cold.mode != "cold"
        or cold.planned != plan_size
        or cold.executed != plan_size
        or cold.validated_cache_hits != 0
        or cold.failed != 0
        or cold.blocked != 0
        or cold.cancelled != 0
        or cold.offline_network_attempt_count != 0
        or cold.max_cpu_concurrency < 1
        or cold.max_cpu_concurrency > 2
        or cold.max_gpu_concurrency != 0
        or len(cold.resources) != plan_size
    ):
        raise ValueError("cold probe did not execute every CPU task cleanly and within limits")
    if any(
        not item.telemetry_complete
        or not item.cleanup_passed
        or not item.hard_limit_passed
        or item.failure_category is not None
        for item in cold.resources
    ):
        raise ValueError("cold probe has missing resource telemetry or an unsuccessful attempt")
    if (
        resume.mode != "resume"
        or resume.planned != plan_size
        or resume.executed != 0
        or resume.validated_cache_hits != plan_size
        or resume.failed != 0
        or resume.blocked != 0
        or resume.offline_network_attempt_count != 0
    ):
        raise ValueError("resume probe did not validate-hit every task without execution")


def _run_summary(manifest: RunManifest, *, retain_resources: bool) -> ProbeRunSummary:
    resources = [
        ProbeResourceSummary(
            wall_seconds=record.wall_seconds,
            cpu_seconds=record.cpu_seconds,
            peak_worker_rss_mib=record.peak_worker_rss_mib,
            peak_process_tree_rss_mib=record.peak_process_tree_rss_mib,
            telemetry_complete=record.telemetry_complete,
            hard_limit_passed=record.hard_limit_passed,
            cleanup_passed=record.cleanup_passed,
            failure_category=record.failure_category,
        )
        for record in (manifest.resources if retain_resources else [])
    ]
    return ProbeRunSummary(
        mode=manifest.mode,
        planned=manifest.planned,
        executed=manifest.executed,
        validated_cache_hits=manifest.validated_cache_hits,
        recovered_abandoned=manifest.recovered_abandoned,
        failed=manifest.failed,
        blocked=manifest.blocked,
        retried=manifest.retried,
        cancelled=manifest.cancelled,
        max_cpu_concurrency=manifest.max_cpu_concurrency,
        max_gpu_concurrency=manifest.max_gpu_concurrency,
        offline_network_attempt_count=manifest.offline_network_attempt_count,
        resources=resources,
    )


def _run_gpu_policy_probe(config: Any, implementation_commit: str) -> tuple[int, int]:
    """Measure serialization with mocked telemetry and synthetic workers only."""

    import tempfile

    config_values = config.model_dump(mode="python")
    config_values["cache_root"] = "cache"
    config_values["state_root"] = "runtime"
    local_config = type(config).model_validate(config_values)
    with tempfile.TemporaryDirectory(prefix="schemaguard-gpu-policy-") as temporary:
        root = Path(temporary)
        tasks = []
        for number in range(2):
            identity_values = {
                "schema_version": 1,
                "dataset_sha256": sha256_canonical_json({"gpu-probe": number}),
                "split_sha256": sha256_canonical_json({"split": "isolated"}),
                "view_certificate_sha256": sha256_canonical_json({"view": "synthetic"}),
                "model_spec_sha256": sha256_canonical_json({"task": "gpu-policy-only"}),
                "model_parameters_sha256": sha256_canonical_json({"number": number}),
                "checkpoint_sha256": None,
                "dependency_lock_sha256": dependency_lock_hash(ROOT / "uv.lock"),
                "source_implementation_sha256": implementation_hash(ROOT, IMPLEMENTATION_PATHS),
                "seed": 9900 + number,
                "device_policy": "cuda",
                "artifact_kind": "probe",
            }
            from schemaguard.cache.contracts import CacheIdentity

            identity = CacheIdentity.model_validate(identity_values)
            tasks.append(
                build_task(
                    f"gpu-policy-{number}",
                    identity,
                    operation="sleep",
                    payload_token=f"synthetic-{number}",
                    delay_seconds=0.1,
                )
            )
        plan = build_plan(tasks, random_seed=1729)
        cache_root = root / local_config.cache_root
        cache = CacheStore(cache_root, lock_timeout_seconds=local_config.lock_timeout_seconds)
        index = CacheIndex(root / local_config.state_root / "index.sqlite", cache_root)

        def mock_gpu_sample() -> dict[str, float]:
            return {
                "total_mib": 4096.0,
                "free_mib": 3900.0,
                "allocated_mib": 128.0,
                "reserved_mib": 128.0,
            }

        scheduler = Scheduler(
            root,
            local_config,
            cache,
            index,
            source_commit=implementation_commit,
            gpu_sampler=mock_gpu_sample,
            gpu_budget_check=lambda: None,
        )
        result = scheduler.run(plan)
        if result.manifest.executed != 2 or result.manifest.failed or result.manifest.blocked:
            raise ValueError("mocked GPU queue policy probe did not execute both test workers")
        return result.manifest.max_gpu_concurrency, result.manifest.offline_network_attempt_count


def _junit_results(path: Path) -> tuple[list[str], int, int]:
    if not path.is_file():
        return [], 0, 1
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    names = sorted(
        {
            f"{case.attrib.get('classname', 'unknown')}::{case.attrib.get('name', 'unknown')}"
            for case in cases
        }
    )
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return names, len(cases) - failures - errors - skipped, failures + errors


def _run_command(
    name: str,
    command: list[str],
    log_directory: Path,
    statuses: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=2700,
    )
    atomic_write_text(log_directory / f"{name}.txt", result.stdout + result.stderr)
    statuses[name] = "PASS" if result.returncode == 0 else "FAIL"
    return result


def _private_document_remains_untracked() -> bool:
    """Check only Git metadata; never open or hash the user's reference document."""

    filename = "SchemaGuard_Complete_Research_and_Engineering_Plan.docx"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", filename],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return tracked.returncode != 0 and any(
        line == f"?? {filename}" for line in status.stdout.splitlines()
    )


def _record_local_checks(log_directory: Path) -> tuple[dict[str, str], list[str], dict[str, int]]:
    log_directory.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    all_names: set[str] = set()
    counts = {"unit_passed": 0, "unit_failed": 0, "integration_passed": 0, "integration_failed": 0}
    unit_xml = log_directory / "unit-junit.xml"
    integration_xml = log_directory / "integration-junit.xml"
    unit_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/unit",
        "-m",
        "not network and not gpu and not foundation_model and not evidence",
        "-rA",
        f"--junitxml={unit_xml}",
    ]
    unit_result = _run_command("unit_tests", unit_command, log_directory, statuses)
    unit_names, counts["unit_passed"], counts["unit_failed"] = _junit_results(unit_xml)
    all_names.update(unit_names)
    integration_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/integration/test_scheduler_resume.py",
        "tests/integration/test_scheduler_concurrency.py",
        "tests/integration/test_scheduler_process_isolation.py",
        "-m",
        "integration and not network and not gpu and not foundation_model",
        "-rA",
        f"--junitxml={integration_xml}",
    ]
    integration_result = _run_command(
        "scheduler_integration_tests", integration_command, log_directory, statuses
    )
    integration_names, counts["integration_passed"], counts["integration_failed"] = _junit_results(
        integration_xml
    )
    all_names.update(integration_names)
    if unit_result.returncode != 0 and counts["unit_failed"] == 0:
        counts["unit_failed"] = 1
    if integration_result.returncode != 0 and counts["integration_failed"] == 0:
        counts["integration_failed"] = 1

    commands = (
        ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        ("mypy", [sys.executable, "-m", "mypy", "src/schemaguard"]),
        ("schema_generation", [sys.executable, "scripts/generate_artifact_schemas.py"]),
        ("repository_naming", [sys.executable, "scripts/validate_repository_naming.py"]),
        ("repository_structure", [sys.executable, "scripts/validate_repository_repair.py"]),
        (
            "model_adapter_evidence",
            [sys.executable, "scripts/validate_model_adapter_evidence.py"],
        ),
        ("diff_check", ["git", "diff", "--check"]),
    )
    for name, command in commands:
        _run_command(name, command, log_directory, statuses)
    return statuses, sorted(all_names), counts


def _validate_schemas() -> None:
    from schemaguard.artifact_contracts import schema_documents

    for filename, expected in schema_documents().items():
        path = ROOT / "schemas" / filename
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError(f"generated schema is stale: {filename}")


def _validate_evidence(
    *,
    config_path: Path,
    inventory_path: Path,
    fault_path: Path,
    probe_path: Path,
    comparison_path: Path,
) -> CacheSchedulerInventory:
    config = load_config(config_path)
    implementation_commit = current_commit(ROOT)
    plan, code_hash, dependency_hash = make_probe_plan(ROOT, config)
    configuration_hash = runtime_configuration_hash(config_path)
    inventory = CacheSchedulerInventory.model_validate(
        json.loads(inventory_path.read_text(encoding="utf-8"))
    )
    fault = FaultInjectionEvidence.model_validate(
        json.loads(fault_path.read_text(encoding="utf-8"))
    )
    probe = SchedulerProbeEvidence.model_validate(
        json.loads(probe_path.read_text(encoding="utf-8"))
    )
    comparison = _validate_protected_comparison(
        comparison_path, inventory.source_implementation_commit
    )
    if inventory.source_implementation_commit != fault.source_implementation_commit:
        raise ValueError("inventory and fault evidence refer to different implementation commits")
    if inventory.source_implementation_commit != implementation_commit and not _git_ancestor(
        inventory.source_implementation_commit, implementation_commit
    ):
        raise ValueError("inventory implementation commit is not in the evidence history")
    if inventory.implementation_hashes != _implementation_file_hashes():
        raise ValueError("implementation source hashes differ from the sanitized inventory")
    if inventory.configuration_sha256 != configuration_hash:
        raise ValueError("runtime configuration hash differs from the inventory")
    if inventory.plan_sha256 != plan.plan_hash:
        raise ValueError("probe plan hash differs from the inventory")
    if inventory.cache_identity_hashes != sorted(
        task.cache_identity.cache_key for task in plan.tasks
    ):
        raise ValueError("probe cache identity hashes differ from the inventory")
    if inventory.fault_evidence_sha256 != canonical_source_hash(fault_path):
        raise ValueError("fault evidence file digest differs from the inventory")
    if inventory.probe_evidence_sha256 != canonical_source_hash(probe_path):
        raise ValueError("probe evidence file digest differs from the inventory")
    if inventory.protected_hash_comparison_sha256 != canonical_source_hash(comparison_path):
        raise ValueError("protected comparison digest differs from the inventory")
    if inventory.protected_artifacts_unchanged != (comparison.status == "PASS"):
        raise ValueError("inventory protected-artifact status contradicts its comparison")
    if fault.expected_fault_count != 30 or any(item.status != "PASS" for item in fault.records):
        raise ValueError("one or more required fault-injection cases did not pass")
    if fault.offline_network_attempt_count != 1:
        raise ValueError(
            "fault suite did not record exactly one intentionally denied network attempt"
        )
    if (
        probe.source_implementation_commit != inventory.source_implementation_commit
        or probe.source_implementation_sha256 != code_hash
        or probe.dependency_lock_sha256 != dependency_hash
        or probe.configuration_sha256 != configuration_hash
        or probe.plan_sha256 != plan.plan_hash
        or probe.cache_identity_hashes
        != sorted(task.cache_identity.cache_key for task in plan.tasks)
        or probe.cold.max_cpu_concurrency != inventory.maximum_observed_cpu_concurrency
        or probe.maximum_mocked_gpu_concurrency != inventory.maximum_observed_gpu_concurrency
        or probe.validated_cache_artifact_count != inventory.counters.get("cache_entries")
        or probe.duplicate_validated_artifact_count != inventory.counters.get("duplicate_artifacts")
    ):
        raise ValueError("sanitized probe evidence differs from the inventory or frozen sources")
    if probe.cold.offline_network_attempt_count or probe.resume.offline_network_attempt_count:
        raise ValueError("ordinary cache/scheduler probes must have zero network attempts")

    _validate_schemas()
    source_code_hash = implementation_hash(ROOT, IMPLEMENTATION_PATHS)
    receipts_root = ROOT / config.state_root / "probe_runs"
    if inventory.maximum_observed_gpu_concurrency != 1:
        raise ValueError("GPU policy probe did not prove a single exclusive worker")
    if inventory.offline_network_attempt_count != 0:
        raise ValueError("ordinary offline probes must have zero network attempts")
    if inventory.counters.get("cold_executed") != len(plan.tasks):
        raise ValueError("inventory cold execution count differs from sanitized probe evidence")
    if inventory.counters.get("resume_executed") != 0:
        raise ValueError("resume run unexpectedly executed completed work")
    if inventory.counters.get("resume_cache_hits") != len(plan.tasks):
        raise ValueError("resume run did not record every validated cache hit")
    expected_tree_peak = max(
        (item.peak_process_tree_rss_mib for item in probe.cold.resources), default=None
    )
    expected_worker_peak = max(
        (item.peak_worker_rss_mib for item in probe.cold.resources), default=None
    )
    if (
        inventory.resource_summaries.get("cold_peak_process_tree_rss_mib") != expected_tree_peak
        or inventory.resource_summaries.get("cold_peak_worker_rss_mib") != expected_worker_peak
        or inventory.resource_summaries.get("peak_vram_reserved_mib") is not None
    ):
        raise ValueError("resource summary differs from sanitized cold probe observations")
    if inventory.counters.get("duplicate_artifacts") != 0:
        raise ValueError("inventory reports duplicate validated artifacts")
    if inventory.status != "PASS_PENDING_REVIEW":
        raise ValueError("cache/scheduler evidence is not in PASS_PENDING_REVIEW state")

    cold_receipt_path = receipts_root / "cold_run.json"
    resume_receipt_path = receipts_root / "resume_run.json"
    if cold_receipt_path.exists() or resume_receipt_path.exists():
        if not cold_receipt_path.exists() or not resume_receipt_path.exists():
            raise ValueError(
                "local cold and resume receipts must either both exist or both be absent"
            )
        cold = _read_probe_receipt(
            cold_receipt_path,
            mode="cold",
            implementation_commit=inventory.source_implementation_commit,
            implementation_sha256=source_code_hash,
            dependency_sha256=dependency_hash,
            configuration_sha256=configuration_hash,
            plan_hash=plan.plan_hash,
        )
        resume = _read_probe_receipt(
            resume_receipt_path,
            mode="resume",
            implementation_commit=inventory.source_implementation_commit,
            implementation_sha256=source_code_hash,
            dependency_sha256=dependency_hash,
            configuration_sha256=configuration_hash,
            plan_hash=plan.plan_hash,
        )
        _validate_probe_manifests(cold, resume, len(plan.tasks))
        if (
            cold.executed != probe.cold.executed
            or cold.max_cpu_concurrency != probe.cold.max_cpu_concurrency
            or resume.executed != probe.resume.executed
            or resume.validated_cache_hits != probe.resume.validated_cache_hits
        ):
            raise ValueError("local probe receipts differ from sanitized probe evidence")
    cache_root = ROOT / config.cache_root
    if cache_root.is_dir():
        cache = CacheStore(cache_root, lock_timeout_seconds=config.lock_timeout_seconds)
        artifacts = cache.iter_validated()
        expected_keys = {task.cache_identity.cache_key for task in plan.tasks}
        actual_keys = {artifact.cache_key for artifact in artifacts}
        if len(artifacts) != len(actual_keys) or not expected_keys.issubset(actual_keys):
            raise ValueError("current cache entries are missing or a validated key is duplicated")
        index = CacheIndex(ROOT / config.state_root / "cache_index.sqlite", cache_root)
        index_rows = index.lookup(status="COMPLETE")
        if not expected_keys.issubset({row.cache_key for row in index_rows}):
            raise ValueError(
                "derivative index does not contain exactly the validated probe artifacts"
            )
    return inventory


def main() -> int:
    args = _parser().parse_args()
    config_path = _inside_repository(args.config, label="configuration")
    inventory_path = _inside_repository(args.inventory, label="inventory")
    fault_path = _inside_repository(args.fault_evidence, label="fault evidence")
    comparison_path = _inside_repository(
        args.protected_comparison
        if args.protected_comparison is not None
        else inventory_path.with_name("cache_scheduler_protected_hash_comparison.json"),
        label="protected comparison",
    )
    probe_path = _inside_repository(
        args.probe_evidence
        if args.probe_evidence is not None
        else inventory_path.with_name("cache_scheduler_probe_evidence.json"),
        label="probe evidence",
    )

    if args.record:
        if sys.version_info[:2] != (3, 12):
            raise SystemExit(
                "evidence recording requires Python 3.12 in the existing P12 environment"
            )
        environment = os.environ.get("CONDA_DEFAULT_ENV", "")
        if (
            Path(environment).name.casefold() != "p12"
            or Path(sys.executable).parent.name.casefold() != "p12"
        ):
            raise SystemExit("evidence recording must use the existing P12 Conda environment")
        config = load_config(config_path)
        implementation_commit = current_commit(ROOT)
        plan, source_hash, dependency_hash = make_probe_plan(ROOT, config)
        config_hash = runtime_configuration_hash(config_path)
        status_dir = ROOT / config.state_root / "validation_logs"
        test_statuses, test_names, test_counts = _record_local_checks(status_dir)
        test_statuses["private_docx_untracked"] = (
            "PASS" if _private_document_remains_untracked() else "FAIL"
        )
        _validate_schemas()
        gpu_concurrency, gpu_network_attempts = _run_gpu_policy_probe(config, implementation_commit)
        fault = run_fault_injection_suite(source_commit=implementation_commit)
        atomic_write_json(fault_path, fault.model_dump(mode="json"))
        comparison = _compare_protected_files(implementation_commit, comparison_path)
        receipts_root = ROOT / config.state_root / "probe_runs"
        cold = _read_probe_receipt(
            receipts_root / "cold_run.json",
            mode="cold",
            implementation_commit=implementation_commit,
            implementation_sha256=source_hash,
            dependency_sha256=dependency_hash,
            configuration_sha256=config_hash,
            plan_hash=plan.plan_hash,
        )
        resume = _read_probe_receipt(
            receipts_root / "resume_run.json",
            mode="resume",
            implementation_commit=implementation_commit,
            implementation_sha256=source_hash,
            dependency_sha256=dependency_hash,
            configuration_sha256=config_hash,
            plan_hash=plan.plan_hash,
        )
        _validate_probe_manifests(cold, resume, len(plan.tasks))
        cache_root = ROOT / config.cache_root
        cache = CacheStore(cache_root, lock_timeout_seconds=config.lock_timeout_seconds)
        artifacts = cache.iter_validated()
        expected_keys = {task.cache_identity.cache_key for task in plan.tasks}
        actual_keys = {artifact.cache_key for artifact in artifacts}
        cache_ok = len(artifacts) == len(actual_keys) and expected_keys.issubset(actual_keys)
        test_statuses["cache_validation"] = "PASS" if cache_ok else "FAIL"
        if cache_ok:
            index = CacheIndex(ROOT / config.state_root / "cache_index.sqlite", cache_root)
            test_statuses["cache_index"] = (
                "PASS"
                if expected_keys.issubset(
                    {row.cache_key for row in index.lookup(status="COMPLETE")}
                )
                else "FAIL"
            )
        else:
            test_statuses["cache_index"] = "FAIL"
        probe = SchedulerProbeEvidence(
            schema_version=1,
            stage="cache_scheduler_probe",
            source_implementation_commit=implementation_commit,
            source_implementation_sha256=source_hash,
            dependency_lock_sha256=dependency_hash,
            configuration_sha256=config_hash,
            plan_sha256=plan.plan_hash,
            cache_identity_hashes=sorted(task.cache_identity.cache_key for task in plan.tasks),
            cold=_run_summary(cold, retain_resources=True),
            resume=_run_summary(resume, retain_resources=False),
            maximum_mocked_gpu_concurrency=gpu_concurrency,
            mocked_gpu_offline_network_attempt_count=gpu_network_attempts,
            validated_cache_artifact_count=len(expected_keys & actual_keys),
            duplicate_validated_artifact_count=len(artifacts) - len(actual_keys),
        )
        atomic_write_json(probe_path, probe.model_dump(mode="json"))
        test_statuses["offline_probe"] = (
            "PASS"
            if cold.offline_network_attempt_count == 0
            and resume.offline_network_attempt_count == 0
            and gpu_network_attempts == 0
            else "FAIL"
        )
        test_statuses["gpu_exclusive_policy"] = "PASS" if gpu_concurrency == 1 else "FAIL"
        test_statuses["fault_injection"] = (
            "PASS"
            if len(fault.records) == len(FAULT_NAMES)
            and {item.fault_name for item in fault.records} == FAULT_NAMES
            and all(item.status == "PASS" for item in fault.records)
            else "FAIL"
        )
        test_statuses["protected_files"] = "PASS" if comparison.status == "PASS" else "FAIL"
        try:
            _validate_schemas()
            test_statuses["schema_freshness"] = "PASS"
        except Exception:
            test_statuses["schema_freshness"] = "FAIL"
        cache_root = ROOT / config.cache_root
        cache = CacheStore(cache_root, lock_timeout_seconds=config.lock_timeout_seconds)
        artifacts = cache.iter_validated()
        unique_keys = {item.cache_key for item in artifacts}
        resources = cold.resources
        maximum_process_tree = max(
            (
                item.peak_process_tree_rss_mib
                for item in resources
                if item.peak_process_tree_rss_mib is not None
            ),
            default=None,
        )
        maximum_worker_rss = max(
            (
                item.peak_worker_rss_mib
                for item in resources
                if item.peak_worker_rss_mib is not None
            ),
            default=None,
        )
        counters = {
            **test_counts,
            "planned": cold.planned,
            "cold_executed": cold.executed,
            "cold_cache_hits": cold.validated_cache_hits,
            "resume_executed": resume.executed,
            "resume_cache_hits": resume.validated_cache_hits,
            "cache_entries": len(expected_keys & actual_keys),
            "duplicate_artifacts": len(artifacts) - len(unique_keys),
            "fault_count": len(fault.records),
            "faults_passed": sum(item.status == "PASS" for item in fault.records),
            "protected_file_count": len(comparison.records),
        }
        all_names = sorted(set(test_names) | FAULT_NAMES)
        inventory = CacheSchedulerInventory(
            schema_version=1,
            stage="cache_scheduler",
            status=(
                "PASS_PENDING_REVIEW"
                if comparison.status == "PASS"
                and all(value == "PASS" for value in test_statuses.values())
                else "REPAIR_REQUIRED"
            ),
            source_implementation_commit=implementation_commit,
            implementation_hashes=_implementation_file_hashes(),
            configuration_sha256=config_hash,
            plan_sha256=plan.plan_hash,
            cache_identity_hashes=sorted(task.cache_identity.cache_key for task in plan.tasks),
            test_case_names=all_names,
            test_statuses=test_statuses,
            counters=counters,
            maximum_observed_cpu_concurrency=cold.max_cpu_concurrency,
            maximum_observed_gpu_concurrency=gpu_concurrency,
            resource_summaries={
                "cold_total_worker_seconds": sum(item.wall_seconds for item in resources),
                "cold_peak_worker_rss_mib": maximum_worker_rss,
                "cold_peak_process_tree_rss_mib": maximum_process_tree,
                "peak_vram_reserved_mib": None,
            },
            fault_evidence_sha256=canonical_source_hash(fault_path),
            probe_evidence_sha256=canonical_source_hash(probe_path),
            resume_results={
                "planned": resume.planned,
                "executed": resume.executed,
                "validated_cache_hits": resume.validated_cache_hits,
                "failed": resume.failed,
                "blocked": resume.blocked,
                "recovered_abandoned": resume.recovered_abandoned,
                "retried": resume.retried,
                "cancelled": resume.cancelled,
            },
            offline_network_attempt_count=cold.offline_network_attempt_count
            + resume.offline_network_attempt_count
            + gpu_network_attempts,
            protected_artifacts_unchanged=comparison.status == "PASS",
            protected_hash_comparison_sha256=canonical_source_hash(comparison_path),
        )
        atomic_write_json(inventory_path, inventory.model_dump(mode="json"))
        print(json.dumps(inventory.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if inventory.status == "PASS_PENDING_REVIEW" else 2

    try:
        inventory = _validate_evidence(
            config_path=config_path,
            inventory_path=inventory_path,
            fault_path=fault_path,
            probe_path=probe_path,
            comparison_path=comparison_path,
        )
    except (OSError, ValueError, ValidationError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": inventory.status,
                "implementation_commit": inventory.source_implementation_commit,
                "faults_passed": inventory.counters.get("faults_passed", 0),
                "protected_files_unchanged": inventory.protected_artifacts_unchanged,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
