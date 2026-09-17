"""Executable fault-injection matrix for cache publication and scheduler resume."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..cache.contracts import (
    CacheArtifactManifest,
    CacheIdentity,
    CacheSchedulerConfig,
    FaultInjectionEvidence,
    FaultInjectionRecord,
)
from ..cache.index import CacheIndex
from ..cache.locks import CacheKeyLock
from ..cache.store import CacheIntegrityError, CacheStore
from ..runner.contracts import build_plan, build_task
from ..runner.plan import current_commit
from ..runner.resources import classify_resource_limits, execute_isolated_worker
from ..runner.scheduler import Scheduler
from ..runner.state import TaskStateStore
from ..utils.hashing import sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_json


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(index: int, **changes: object) -> CacheIdentity:
    base: dict[str, object] = {
        "schema_version": 1,
        "dataset_sha256": _digest(f"dataset:{index}"),
        "split_sha256": _digest(f"split:{index}"),
        "view_certificate_sha256": _digest(f"view:{index}"),
        "model_spec_sha256": _digest(f"model:{index}"),
        "model_parameters_sha256": _digest(f"parameters:{index}"),
        "checkpoint_sha256": None,
        "dependency_lock_sha256": _digest(f"lock:{index}"),
        "source_implementation_sha256": _digest(f"implementation:{index}"),
        "seed": 10000 + index,
        "device_policy": "cpu",
        "artifact_kind": "probe",
    }
    base.update(changes)
    return CacheIdentity.model_validate(base)


def _publish(store: CacheStore, identity: CacheIdentity, payload: bytes = b"stable") -> None:
    store.publish_bytes(
        identity,
        payload,
        producing_task_identity=_digest(f"task:{identity.seed}"),
    )


def _refresh_completion_marker(store: CacheStore, identity: CacheIdentity) -> None:
    directory = store.entry_path(identity.cache_key)
    manifest_path = directory / "manifest.json"
    marker_path = directory / ".complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_json(marker_path, marker)


def _child_publish(
    root: str,
    identity_value: dict[str, Any],
    barrier: Any,
    queue: Any,
) -> None:
    identity = CacheIdentity.model_validate(identity_value)
    barrier.wait(timeout=20)
    try:
        artifact = (
            CacheStore(root)
            .publish_bytes(
                identity,
                b"concurrent-fault-payload",
                producing_task_identity=_digest("concurrent-task"),
            )
            .artifact
        )
        queue.put(("PASS", artifact.payload_sha256))
    except CacheIntegrityError as exc:
        queue.put(("FAIL_CACHE_INTEGRITY", str(exc)))


def _child_lock_crash(root: str, key: str, ready: Any) -> None:
    lock = CacheKeyLock(root, key, timeout=10)
    lock.__enter__()
    ready.set()
    os._exit(0)


def _child_interrupt_publication(
    root: str,
    identity_value: dict[str, Any],
    stage: str,
) -> None:
    """Crash while holding the identity lock, leaving only a private transaction dir."""

    identity = CacheIdentity.model_validate(identity_value)
    store = CacheStore(root)
    destination = store.entry_path(identity.cache_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with store.identity_lock(identity):
        transaction = Path(
            tempfile.mkdtemp(prefix=f".{identity.cache_key}.txn-crashed-", dir=destination.parent)
        )
        payload_path = transaction / "payload.bin"
        payload = b"partial" if stage == "payload" else b"complete-payload"
        with payload_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if stage == "manifest":
            manifest = CacheArtifactManifest(
                schema_version=1,
                identity=identity,
                cache_key=identity.cache_key,
                artifact_kind=identity.artifact_kind,
                payload_sha256=sha256_file(payload_path),
                payload_size_bytes=payload_path.stat().st_size,
                producing_task_identity=_digest("interrupted-publication"),
                created_at=datetime.now(UTC),
                source_implementation_sha256=identity.source_implementation_sha256,
                dependency_lock_sha256=identity.dependency_lock_sha256,
                validation_status="PASS",
                completed=True,
            )
            atomic_write_json(transaction / "manifest.json", manifest.model_dump(mode="json"))
        os._exit(23)


def _scheduler_system(root: Path, *, commit: str) -> tuple[Scheduler, CacheStore]:
    config = CacheSchedulerConfig.model_validate(
        {
            "schema_version": 1,
            "cpu_workers": 2,
            "cpu_threads_per_worker": 2,
            "gpu_workers": 1,
            "ram_soft_limit_mib": 24576,
            "ram_hard_limit_mib": 28672,
            "vram_soft_limit_mib": 3600,
            "gpu_headroom_mib": 512,
            "lock_timeout_seconds": 15,
            "heartbeat_interval_seconds": 1,
            "abandoned_lease_seconds": 10,
            "cache_root": "cache",
            "state_root": "runtime",
        }
    )
    cache = CacheStore(root / "cache", lock_timeout_seconds=15)
    index = CacheIndex(root / "runtime" / "index.sqlite", cache.root)
    return Scheduler(root, config, cache, index, source_commit=commit), cache


def _record(
    records: list[FaultInjectionRecord],
    *,
    name: str,
    injection: str,
    expected: str,
    observed: str,
    artifact_accepted: bool,
    expected_acceptance: bool,
    resume: str,
    expected_resume: str,
    lock_root: Path,
    cache_key: str,
) -> None:
    try:
        with CacheKeyLock(lock_root, cache_key, timeout=2):
            lock_released = True
    except TimeoutError:
        lock_released = False
    evidence = {
        "fault_name": name,
        "injection_point": injection,
        "expected_failure_category": expected,
        "observed_failure_category": observed,
        "artifact_accepted": artifact_accepted,
        "lock_released": lock_released,
        "resume_behavior": resume,
    }
    passed = (
        expected == observed
        and artifact_accepted == expected_acceptance
        and resume == expected_resume
        and lock_released
    )
    records.append(
        FaultInjectionRecord(
            fault_name=name,
            injection_point=injection,
            expected_failure_category=expected,
            observed_failure_category=observed,
            artifact_accepted=artifact_accepted,
            lock_released=lock_released,
            resume_behavior=resume,
            evidence_sha256=sha256_canonical_json(evidence),
            status="PASS" if passed else "FAIL",
        )
    )


def run_fault_injection_suite(*, source_commit: str | None = None) -> FaultInjectionEvidence:
    """Run each required injection in a disposable directory and return observed evidence."""

    commit = source_commit
    if commit is None:
        commit = current_commit(Path(__file__).resolve().parents[3])
    records: list[FaultInjectionRecord] = []
    network_attempts = 0

    with TemporaryDirectory(prefix="schemaguard-faults-") as temporary:
        root = Path(temporary)
        context = multiprocessing.get_context("spawn")

        invalid_faults = [
            ("payload_write_interrupted", "process stopped during payload write"),
            ("completion_interrupted", "payload durable before completion marker"),
            ("manifest_truncated", "manifest parser"),
            ("payload_truncated", "payload checksum validation"),
            ("completion_marker_missing", "completion marker validation"),
            ("payload_checksum_wrong", "payload checksum validation"),
            ("manifest_identity_wrong", "canonical identity validation"),
            ("cache_key_wrong", "key recomputation"),
            ("artifact_schema_wrong", "strict manifest schema validation"),
            ("failed_artifact_candidate", "failed-state cache candidate"),
        ]

        for index, (name, injection) in enumerate(invalid_faults, 1):
            case_root = root / name
            identity = _identity(index)
            store = CacheStore(case_root / "cache")
            if index == 1:
                stage = "payload"
            elif index == 2:
                stage = "manifest"
            else:
                stage = ""
            if index <= 2:
                crashed_writer = context.Process(
                    target=_child_interrupt_publication,
                    args=(str(store.root), identity.model_dump(mode="json"), stage),
                )
                crashed_writer.start()
                crashed_writer.join(timeout=20)
                if crashed_writer.is_alive():
                    crashed_writer.kill()
                    crashed_writer.join(timeout=5)
                partial_candidate = store.read_validated(identity)
                entries = store.iter_validated()
                observed = (
                    "CACHE_MISS"
                    if crashed_writer.exitcode == 23 and partial_candidate is None and not entries
                    else "UNSAFE_PARTIAL_PUBLICATION"
                )
                interrupted_republication = store.publish_bytes(
                    identity,
                    b"recomputed-after-interrupted-transaction",
                    producing_task_identity=_digest(f"recovered-task:{index}"),
                )
                accepted_recomputed = store.read_validated(identity) is not None
                _record(
                    records,
                    name=name,
                    injection=injection,
                    expected="CACHE_MISS",
                    observed=observed,
                    artifact_accepted=partial_candidate is not None,
                    expected_acceptance=False,
                    resume=(
                        "RECOMPUTED_AFTER_INTERRUPTED_TRANSACTION"
                        if accepted_recomputed and not interrupted_republication.reused_existing
                        else "RECOMPUTATION_FAILED"
                    ),
                    expected_resume="RECOMPUTED_AFTER_INTERRUPTED_TRANSACTION",
                    lock_root=store.root,
                    cache_key=identity.cache_key,
                )
                continue
            else:
                artifact = store.publish_bytes(
                    identity,
                    b"valid-before-fault",
                    producing_task_identity=_digest(f"task:{index}"),
                ).artifact
                directory = artifact.payload_path.parent
                manifest_path = directory / "manifest.json"
                marker_path = directory / ".complete.json"
                if name == "manifest_truncated":
                    manifest_path.write_text("{truncated", encoding="utf-8")
                elif name == "payload_truncated":
                    artifact.payload_path.write_bytes(b"short")
                elif name == "completion_marker_missing":
                    marker_path.unlink()
                elif name == "payload_checksum_wrong":
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    marker["payload_sha256"] = _digest("wrong-checksum")
                    atomic_write_json(marker_path, marker)
                else:
                    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if name == "manifest_identity_wrong":
                        manifest_payload["identity"]["dataset_sha256"] = _digest("wrong-data")
                    elif name == "cache_key_wrong":
                        manifest_payload["cache_key"] = _digest("wrong-key")
                    elif name == "artifact_schema_wrong":
                        manifest_payload["schema_version"] = 99
                    else:
                        manifest_payload["validation_status"] = "FAIL"
                    atomic_write_json(manifest_path, manifest_payload)
                    _refresh_completion_marker(store, identity)
            try:
                store.read_validated(identity)
                observed = "PASS"
            except CacheIntegrityError:
                observed = "FAIL_CACHE_INTEGRITY"
            accepted_corrupt = False
            quarantined = store.quarantine_invalid(identity)
            if quarantined is not None:
                store.publish_bytes(
                    identity,
                    b"recomputed-after-fault",
                    producing_task_identity=_digest(f"recovered-task:{index}"),
                )
                accepted_corrupt = store.read_validated(identity) is not None
            resume = "RECOMPUTED_AFTER_QUARANTINE" if accepted_corrupt else "RECOMPUTATION_FAILED"
            _record(
                records,
                name=name,
                injection=injection,
                expected="FAIL_CACHE_INTEGRITY",
                observed=observed,
                artifact_accepted=False,
                expected_acceptance=False,
                resume=resume,
                expected_resume="RECOMPUTED_AFTER_QUARANTINE",
                lock_root=store.root,
                cache_key=identity.cache_key,
            )

        identity_changes = [
            ("model_configuration_change", {"model_parameters_sha256": _digest("changed-params")}),
            ("implementation_change", {"source_implementation_sha256": _digest("changed-code")}),
            ("dependency_lock_change", {"dependency_lock_sha256": _digest("changed-lock")}),
            ("dataset_checksum_change", {"dataset_sha256": _digest("changed-data")}),
            ("split_checksum_change", {"split_sha256": _digest("changed-split")}),
            ("view_certificate_change", {"view_certificate_sha256": _digest("changed-view")}),
            ("checkpoint_checksum_change", {"checkpoint_sha256": _digest("changed-checkpoint")}),
            ("device_policy_change", {"device_policy": "cuda"}),
        ]
        for index, (name, change) in enumerate(identity_changes, 20):
            store = CacheStore(root / name / "cache")
            original = _identity(index)
            _publish(store, original)
            changed_identity = CacheIdentity.model_validate(
                {**original.model_dump(mode="json"), **change}
            )
            candidate_artifact = store.read_validated(changed_identity)
            observed = "PASS" if candidate_artifact is not None else "CACHE_MISS"
            _record(
                records,
                name=name,
                injection="change one scientific or implementation identity field",
                expected="CACHE_MISS",
                observed=observed,
                artifact_accepted=candidate_artifact is not None,
                expected_acceptance=False,
                resume=(
                    "NEW_IDENTITY_REQUIRES_COMPUTE"
                    if candidate_artifact is None
                    else "UNSAFE_REUSE"
                ),
                expected_resume="NEW_IDENTITY_REQUIRES_COMPUTE",
                lock_root=store.root,
                cache_key=changed_identity.cache_key,
            )

        concurrent_store = CacheStore(root / "concurrent" / "cache")
        concurrent_identity = _identity(40)
        barrier = context.Barrier(2)
        queue = context.Queue()
        workers = [
            context.Process(
                target=_child_publish,
                args=(
                    str(concurrent_store.root),
                    concurrent_identity.model_dump(mode="json"),
                    barrier,
                    queue,
                ),
            )
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
        child_results = [queue.get(timeout=5) for _ in workers]
        concurrent_count = len(concurrent_store.iter_validated())
        concurrent_ok = (
            all(worker.exitcode == 0 for worker in workers)
            and all(item[0] == "PASS" for item in child_results)
            and concurrent_count == 1
        )
        _record(
            records,
            name="two_process_same_identity_publication",
            injection="two spawned writers publish identical bytes simultaneously",
            expected="ONE_VALIDATED_ARTIFACT",
            observed="ONE_VALIDATED_ARTIFACT" if concurrent_ok else "DUPLICATE_OR_FAILED",
            artifact_accepted=concurrent_count == 1,
            expected_acceptance=True,
            resume="WINNING_ARTIFACT_REUSED",
            expected_resume="WINNING_ARTIFACT_REUSED",
            lock_root=concurrent_store.root,
            cache_key=concurrent_identity.cache_key,
        )

        conflict_store = CacheStore(root / "same_key_conflict" / "cache")
        conflict_identity = _identity(41)
        _publish(conflict_store, conflict_identity, b"original")
        try:
            conflict_store.publish_bytes(
                conflict_identity,
                b"different",
                producing_task_identity=_digest("conflicting-task"),
            )
            observed = "PASS"
        except CacheIntegrityError:
            observed = "FAIL_CACHE_INTEGRITY"
        original_retained = conflict_store.load_payload(conflict_identity) == b"original"
        _record(
            records,
            name="same_key_different_payload",
            injection="publish different bytes for an already validated identity",
            expected="FAIL_CACHE_INTEGRITY",
            observed=observed,
            artifact_accepted=original_retained,
            expected_acceptance=True,
            resume="ORIGINAL_IMMUTABLE_ENTRY_RETAINED",
            expected_resume="ORIGINAL_IMMUTABLE_ENTRY_RETAINED",
            lock_root=conflict_store.root,
            cache_key=conflict_identity.cache_key,
        )

        crash_store = CacheStore(root / "lock_crash" / "cache")
        crash_identity = _identity(42)
        event = context.Event()
        crash_worker = context.Process(
            target=_child_lock_crash,
            args=(str(crash_store.root), crash_identity.cache_key, event),
        )
        crash_worker.start()
        acquired = event.wait(timeout=15)
        crash_worker.join(timeout=15)
        crashed_lock_recovered = acquired and crash_worker.exitcode == 0
        _record(
            records,
            name="lock_holder_process_crash",
            injection="owner exits without executing lock-release cleanup",
            expected="PASS",
            observed="PASS" if crashed_lock_recovered else "FAIL_LOCK_RELEASE",
            artifact_accepted=False,
            expected_acceptance=False,
            resume="LOCK_REACQUIRED_AFTER_OWNER_EXIT",
            expected_resume="LOCK_REACQUIRED_AFTER_OWNER_EXIT",
            lock_root=crash_store.root,
            cache_key=crash_identity.cache_key,
        )

        stale_store = CacheStore(root / "stale_lock" / "cache")
        stale_identity = _identity(43)
        stale_lock = CacheKeyLock(stale_store.root, stale_identity.cache_key, timeout=2)
        stale_lock.path.parent.mkdir(parents=True, exist_ok=True)
        stale_lock.path.write_text("inactive lock-file remnant", encoding="utf-8")
        try:
            with stale_lock:
                stale_result = "PASS"
        except TimeoutError:
            stale_result = "FAIL_LOCK_TIMEOUT"
        _record(
            records,
            name="stale_lock_file_without_owner",
            injection="leave old lock-file bytes without an active OS lock",
            expected="PASS",
            observed=stale_result,
            artifact_accepted=False,
            expected_acceptance=False,
            resume="LOCK_ACQUIRED_WITHOUT_DELETING_STALE_FILE",
            expected_resume="LOCK_ACQUIRED_WITHOUT_DELETING_STALE_FILE",
            lock_root=stale_store.root,
            cache_key=stale_identity.cache_key,
        )

        limit_identity = _identity(44)
        limit_store = CacheStore(root / "ram_limit" / "cache")
        limit_category = classify_resource_limits(
            process_tree_rss_mib=28673,
            gpu_reserved_mib=None,
            ram_hard_limit_mib=28672,
            vram_soft_limit_mib=3600,
        )
        _record(
            records,
            name="worker_exceeds_hard_ram",
            injection="inject a measured process-tree RSS sample above configured hard cap",
            expected="FAIL_RESOURCE_LIMIT",
            observed=limit_category or "PASS",
            artifact_accepted=limit_store.read_validated(limit_identity) is not None,
            expected_acceptance=False,
            resume="NO_ARTIFACT_PUBLISHED",
            expected_resume="NO_ARTIFACT_PUBLISHED",
            lock_root=limit_store.root,
            cache_key=limit_identity.cache_key,
        )

        timeout_identity = _identity(45)
        timeout_task = build_task(
            "injected-timeout",
            timeout_identity,
            operation="sleep",
            payload_token="timeout",
            delay_seconds=2,
            timeout_seconds=0.5,
        )
        timeout_execution = execute_isolated_worker(
            timeout_task,
            root / "timeout-worker",
            hard_ram_limit_mib=28672,
            soft_ram_limit_mib=24576,
            heartbeat_interval_seconds=1,
        )
        timeout_store = CacheStore(root / "timeout-cache")
        _record(
            records,
            name="worker_timeout",
            injection="sleep worker exceeds its strict wall-time deadline",
            expected="FAIL_TIMEOUT",
            observed=timeout_execution.failure_category or "PASS",
            artifact_accepted=timeout_store.read_validated(timeout_identity) is not None,
            expected_acceptance=False,
            resume="NO_ARTIFACT_PUBLISHED",
            expected_resume="NO_ARTIFACT_PUBLISHED",
            lock_root=timeout_store.root,
            cache_key=timeout_identity.cache_key,
        )

        crash_identity = _identity(46)
        crash_task = build_task(
            "injected-worker-exit",
            crash_identity,
            operation="exit_without_result",
            payload_token="crash",
        )
        crash_execution = execute_isolated_worker(
            crash_task,
            root / "crash-worker",
            hard_ram_limit_mib=28672,
            soft_ram_limit_mib=24576,
            heartbeat_interval_seconds=1,
        )
        crash_artifact_store = CacheStore(root / "crash-cache")
        _record(
            records,
            name="worker_exit_without_result",
            injection="worker exits abruptly before writing a result envelope",
            expected="FAIL_MODEL_RUNTIME",
            observed=crash_execution.failure_category or "PASS",
            artifact_accepted=crash_artifact_store.read_validated(crash_identity) is not None,
            expected_acceptance=False,
            resume="NO_ARTIFACT_PUBLISHED",
            expected_resume="NO_ARTIFACT_PUBLISHED",
            lock_root=crash_artifact_store.root,
            cache_key=crash_identity.cache_key,
        )

        restart_identity = _identity(47)
        restart_plan = build_plan(
            [build_task("abandoned-task", restart_identity)], random_seed=1729
        )
        restart_state = TaskStateStore(root / "restart" / "runtime", restart_plan)
        restart_state.initialize()
        restart_state.transition(restart_plan.tasks[0].task_id, "RUNNING")
        snapshot = restart_state.read()
        current = snapshot.tasks[restart_plan.tasks[0].task_id]
        attempts = list(current.attempts)
        from datetime import timedelta

        attempts[-1] = attempts[-1].model_copy(
            update={
                "owner_pid": 2_000_000_000,
                "heartbeat_at": datetime.now(UTC) - timedelta(seconds=120),
            }
        )
        snapshot.tasks[restart_plan.tasks[0].task_id] = current.model_copy(
            update={"attempts": attempts}
        )
        atomic_write_json(restart_state.path, snapshot.model_dump(mode="json"))
        recovered_count = restart_state.recover_abandoned(60)
        recovered_state = restart_state.read().tasks[restart_plan.tasks[0].task_id].state
        restart_store = CacheStore(root / "restart" / "cache")
        _record(
            records,
            name="restart_with_running_task",
            injection="persist an expired lease whose worker and owner processes are gone",
            expected="PASS",
            observed="PASS" if recovered_count == 1 else "FAIL_RECOVERY",
            artifact_accepted=restart_store.read_validated(restart_identity) is not None,
            expected_acceptance=False,
            resume="PENDING_FOR_SAFE_RETRY" if recovered_state == "PENDING" else "NOT_RECOVERED",
            expected_resume="PENDING_FOR_SAFE_RETRY",
            lock_root=restart_store.root,
            cache_key=restart_identity.cache_key,
        )

        for index_name, corrupt in (("missing", False), ("corrupt", True)):
            index_root = root / f"index_{index_name}"
            index_identity = _identity(48 if index_name == "missing" else 49)
            index_store = CacheStore(index_root / "cache")
            _publish(index_store, index_identity)
            database = index_root / "runtime" / "index.sqlite"
            database.parent.mkdir(parents=True)
            if corrupt:
                database.write_bytes(b"not a sqlite database")
            cache_index = CacheIndex(database, index_store.root)
            rebuilt = cache_index.rebuild(index_store)
            found = len(cache_index.lookup(cache_key=index_identity.cache_key)) == 1
            _record(
                records,
                name=f"{index_name}_cache_index",
                injection="remove the derivative index"
                if not corrupt
                else "truncate the SQLite index",
                expected="INDEX_REBUILT",
                observed="INDEX_REBUILT" if rebuilt == 1 and found else "INDEX_INVALID",
                artifact_accepted=index_store.read_validated(index_identity) is not None,
                expected_acceptance=True,
                resume="MANIFEST_REDISCOVERED",
                expected_resume="MANIFEST_REDISCOVERED",
                lock_root=index_store.root,
                cache_key=index_identity.cache_key,
            )

        complete_identity = _identity(50)
        complete_task = build_task("complete-without-payload", complete_identity)
        complete_plan = build_plan([complete_task], random_seed=1729)
        complete_scheduler, complete_store = _scheduler_system(
            root / "complete-state", commit=commit
        )
        complete_state = TaskStateStore(root / "complete-state" / "runtime", complete_plan)
        complete_state.initialize()
        complete_state.transition(complete_task.task_id, "RUNNING")
        complete_state.transition(
            complete_task.task_id,
            "COMPLETE",
            reason="injected missing artifact",
            artifact_path="missing/payload.bin",
            artifact_sha256=_digest("missing"),
        )
        complete_resume_result = complete_scheduler.run(complete_plan, resume=True)
        complete_after = complete_store.read_validated(complete_identity) is not None
        _record(
            records,
            name="complete_state_without_valid_payload",
            injection="persist COMPLETE state without the content-addressed payload",
            expected="RECOMPUTE_INVALID_COMPLETE",
            observed="RECOMPUTE_INVALID_COMPLETE"
            if complete_resume_result.manifest.executed == 1 and complete_after
            else "UNSAFE_COMPLETE_SKIP",
            artifact_accepted=complete_after,
            expected_acceptance=True,
            resume="TASK_REEXECUTED_AND_VALIDATED"
            if complete_resume_result.manifest.executed == 1
            else "TASK_NOT_REEXECUTED",
            expected_resume="TASK_REEXECUTED_AND_VALIDATED",
            lock_root=complete_store.root,
            cache_key=complete_identity.cache_key,
        )

        network_identity = _identity(51)
        network_task = build_task(
            "offline-network-attempt",
            network_identity,
            operation="network_attempt",
            payload_token="must-be-denied",
        )
        network_plan = build_plan([network_task], random_seed=1729)
        network_scheduler, network_store = _scheduler_system(root / "offline", commit=commit)
        network_result = network_scheduler.run(network_plan)
        offline_attempts = network_result.manifest.offline_network_attempt_count
        network_attempts += offline_attempts
        network_not_cached = network_store.read_validated(network_identity) is None
        _record(
            records,
            name="network_attempt_during_offline_execution",
            injection="worker attempts loopback socket connection under Python audit deny hook",
            expected="FAIL_MODEL_RUNTIME",
            observed=network_result.manifest.failures[0].category
            if network_result.manifest.failures
            else "PASS",
            artifact_accepted=not network_not_cached,
            expected_acceptance=False,
            resume="NETWORK_DENIED_NO_ARTIFACT"
            if offline_attempts == 1 and network_not_cached
            else "NETWORK_GUARD_FAILED",
            expected_resume="NETWORK_DENIED_NO_ARTIFACT",
            lock_root=network_store.root,
            cache_key=network_identity.cache_key,
        )

    return FaultInjectionEvidence(
        schema_version=1,
        stage="cache_scheduler_faults",
        source_implementation_commit=commit,
        records=records,
        expected_fault_count=30,
        offline_network_attempt_count=network_attempts,
    )


__all__ = ["run_fault_injection_suite"]
