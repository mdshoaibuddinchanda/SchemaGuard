"""Isolated model conditions executed through the existing cache scheduler."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any

from ..cache.contracts import CacheIdentity, CacheSchedulerConfig
from ..cache.index import CacheIndex
from ..cache.store import CacheArtifact, CacheIntegrityError, CacheStore
from ..runner.contracts import RunPlan, TaskSpec, build_plan, build_task
from ..runner.scheduler import Scheduler, SchedulerResult
from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_bytes
from .contracts import (
    AdapterResourceSummary,
    ConditionResource,
    ConditionRun,
    Partition,
    PredictionFileRecord,
    PredictionOutputSummary,
    SmokeCondition,
    SmokeConditionStatus,
    SmokePlan,
    WorkerConditionResult,
)
from .planning import ALL_PARTITIONS, SmokeConfig, _read_json

PREDICTION_PARTITIONS: tuple[Partition, ...] = ("calibration", "test")

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PREDICTION_COLUMNS = (
    "schema_version",
    "run_identity_sha256",
    "plan_sha256",
    "condition_id",
    "dataset_id",
    "model_id",
    "view_id",
    "device",
    "seed",
    "partition",
    "row_id",
    "class_order",
    "p_0",
    "p_1",
    "predicted_class",
    "fit_time_seconds",
    "prediction_time_seconds",
    "peak_ram_mib",
    "peak_vram_mib",
    "dataset_features_sha256",
    "target_artifact_sha256",
    "assignment_sha256",
    "view_certificate_sha256",
    "view_features_sha256",
    "model_spec_sha256",
    "planned_parameters_sha256",
    "observed_parameters_sha256",
    "checkpoint_sha256",
    "source_commit",
    "dependency_lock_sha256",
    "code_identity_sha256",
    "cache_identity_sha256",
    "validation_status",
    "resource_sha256",
)


def _atomic_worker_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _worker_result(
    *,
    task_id: str,
    result_path: Path,
    output_path: Path,
    started: float,
    cpu_started: float,
    ok: bool,
    network_attempts: int,
    error: str | None = None,
    traceback_text: str | None = None,
) -> None:
    result: dict[str, Any] = {
        "schema_version": 1,
        "ok": ok,
        "task_id": task_id,
        "worker_pid": os.getpid(),
        "cpu_seconds": max(0.0, time.process_time() - cpu_started),
        "worker_seconds": max(0.0, time.perf_counter() - started),
        "network_attempt_count": network_attempts,
    }
    if ok:
        result.update(
            {
                "payload_sha256": sha256_file(output_path),
                "payload_size_bytes": output_path.stat().st_size,
                "worker_rss_mib": None,
                "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            }
        )
        try:
            import psutil

            result["worker_rss_mib"] = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except Exception:
            pass
    else:
        result.update({"error": error or "smoke condition failed", "traceback": traceback_text})
    _atomic_worker_json(result_path, result)


def _install_offline_network_guard() -> list[int]:
    """Count and block Python socket connection attempts in this worker process."""
    attempts = [0]

    def deny_network(event: str, _: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            attempts[0] += 1
            raise PermissionError("smoke experiment is frozen offline")

    sys.addaudithook(deny_network)
    return attempts


def _worker_error_text(exc: BaseException) -> str:
    category = getattr(exc, "category", None)
    category_name = getattr(category, "value", category)
    error = f"{type(exc).__name__}: {exc}"
    if isinstance(category_name, str) and category_name.startswith(("BLOCKED_", "FAIL_")):
        return f"{category_name}: {error}"
    return error


def _reported_failure_category(fallback: str | None, reason: str | None) -> str | None:
    if reason:
        candidate = reason.partition(":")[0]
        if candidate.startswith(("BLOCKED_", "FAIL_")):
            return candidate
    return fallback


def _partition_frames(root: Path, plan: SmokePlan, condition: SmokeCondition):
    import pandas as pd

    assignment_path = (
        root / "data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/assignments.parquet"
    )
    assignment = pd.read_parquet(assignment_path)
    if sha256_file(assignment_path) != plan.assignment_sha256:
        raise ValueError("worker split assignment bytes differ from the frozen plan")
    if hash_dataframe_logically(assignment) != plan.assignment_logical_sha256:
        raise ValueError("worker split assignment rows differ from the frozen plan")
    part_col = "partition" if "partition" in assignment.columns else "split"
    view = next(item for item in plan.views if item.view_id == condition.view_id)
    if condition.view_id == "V00":
        feature_path = root / "data/processed/openml/1464/features.parquet"
        if sha256_file(feature_path) != plan.dataset_features_sha256:
            raise ValueError("worker source feature bytes differ from the frozen plan")
        all_features = pd.read_parquet(feature_path)
        indexed = all_features.set_index("__sg_row_id", drop=False)
        frames = {
            partition: indexed.loc[
                assignment.loc[assignment[part_col] == partition, "__sg_row_id"].tolist()
            ].reset_index(drop=True)
            for partition in ALL_PARTITIONS
        }
    else:
        directory = (
            root
            / "data/transformed/openml/1464/stratified_group_5fold_v1/seed_1729"
            / view.view_name
        )
        if sha256_file(directory / "manifest.json") != view.manifest_sha256:
            raise ValueError("worker V01 manifest differs from the frozen plan")
        manifest = _read_json(directory / "manifest.json")
        frames = {
            partition: pd.read_parquet(directory / manifest["partition_feature_paths"][partition])
            for partition in ALL_PARTITIONS
        }
        for partition in ALL_PARTITIONS:
            feature_path = directory / manifest["partition_feature_paths"][partition]
            certificate_path = directory / manifest["certificate_paths"][partition]
            if sha256_file(feature_path) != view.partition_feature_file_sha256[partition]:
                raise ValueError(f"worker V01 {partition} feature file differs from the plan")
            if (
                hash_dataframe_logically(frames[partition])
                != view.partition_feature_sha256[partition]
            ):
                raise ValueError(f"worker V01 {partition} features differ from the frozen plan")
            if sha256_file(certificate_path) != view.partition_certificate_file_sha256[partition]:
                raise ValueError(f"worker V01 {partition} certificate differs from the plan")
    for partition, frame in frames.items():
        expected_ids = assignment.loc[assignment[part_col] == partition, "__sg_row_id"].tolist()
        if frame["__sg_row_id"].tolist() != expected_ids:
            raise ValueError(f"{condition.view_id} {partition} row alignment differs from the plan")
        if hash_dataframe_logically(frame) != view.partition_feature_sha256[partition]:
            raise ValueError(f"{condition.view_id} {partition} features differ from the plan")
        if (
            condition.view_id == "V00"
            and sha256_file(root / "data/processed/openml/1464/features.parquet")
            != view.partition_feature_file_sha256[partition]
        ):
            raise ValueError("worker V00 source feature bytes differ from the view plan")
    target_path = root / "data/processed/openml/1464/targets.parquet"
    if sha256_file(target_path) != plan.target_artifact_sha256:
        raise ValueError("worker target-artifact bytes differ from the frozen opaque hash")
    return frames, assignment


def _training_targets(root: Path, row_ids: list[str | int], assignment):
    import pandas as pd

    partition_column = "partition" if "partition" in assignment.columns else "split"
    assigned_training_ids = assignment.loc[
        assignment[partition_column] == "train", "__sg_row_id"
    ].tolist()
    if row_ids != assigned_training_ids:
        raise ValueError("model fitting may access target labels for training rows only")
    target_path = root / "data/processed/openml/1464/targets.parquet"
    targets = pd.read_parquet(target_path, filters=[("__sg_row_id", "in", row_ids)])
    if "target_code" not in targets.columns:
        raise ValueError("frozen target table has no target_code column")
    if targets["__sg_row_id"].duplicated().any() or set(targets["__sg_row_id"]) != set(row_ids):
        raise ValueError("filtered training-target read differs from the planned training rows")
    indexed = targets.set_index("__sg_row_id", drop=False)
    ordered = indexed.loc[row_ids].reset_index(drop=True)
    target_codes = ordered["target_code"].to_numpy(dtype="int64")
    if set(target_codes.tolist()) != {0, 1}:
        raise ValueError("smoke training target codes must contain both canonical classes")
    return target_codes, ordered


def _prediction_frame(
    prediction: Any,
    *,
    condition: SmokeCondition,
    plan: SmokePlan,
    resource_sha256: str,
    adapter_resource: AdapterResourceSummary,
    observed_parameters_sha256: str,
    cache_identity_sha256: str,
):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for row_id, probabilities in zip(prediction.row_ids, prediction.probabilities, strict=True):
        rows.append(
            {
                "schema_version": 1,
                "run_identity_sha256": plan.plan_sha256,
                "plan_sha256": plan.plan_sha256,
                "condition_id": condition.condition_id,
                "dataset_id": plan.dataset_id,
                "model_id": condition.model_id,
                "view_id": condition.view_id,
                "device": condition.device,
                "seed": condition.seed,
                "partition": prediction.partition,
                "row_id": row_id,
                "class_order": ",".join(str(value) for value in prediction.class_order),
                "p_0": float(probabilities[0]),
                "p_1": float(probabilities[1]),
                "predicted_class": int(
                    prediction.class_order[int(probabilities[1] > probabilities[0])]
                ),
                "fit_time_seconds": adapter_resource.fit_seconds,
                "prediction_time_seconds": adapter_resource.prediction_seconds,
                "peak_ram_mib": adapter_resource.peak_ram_mib,
                "peak_vram_mib": adapter_resource.peak_vram_reserved_mib,
                "dataset_features_sha256": plan.dataset_features_sha256,
                "target_artifact_sha256": plan.target_artifact_sha256,
                "assignment_sha256": plan.assignment_sha256,
                "view_certificate_sha256": condition.view_certificate_sha256,
                "view_features_sha256": condition.view_features_sha256,
                "model_spec_sha256": condition.model_spec_sha256,
                "planned_parameters_sha256": condition.parameters_sha256,
                "observed_parameters_sha256": observed_parameters_sha256,
                "checkpoint_sha256": condition.checkpoint_sha256,
                "source_commit": plan.source_commit,
                "dependency_lock_sha256": plan.dependency_lock_sha256,
                "code_identity_sha256": plan.code_identity_sha256,
                "cache_identity_sha256": cache_identity_sha256,
                "validation_status": "PASS",
                "resource_sha256": resource_sha256,
            }
        )
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    if frame["row_id"].duplicated().any():
        raise ValueError("adapter returned duplicate inference row IDs")
    return frame


def smoke_worker_entry(task: dict[str, Any], output_path: str, result_path: str) -> None:
    """Scheduler-compatible worker that fits one frozen condition and emits Parquet outputs."""
    from .planning import PlanEnvelope

    for name in THREAD_VARIABLES:
        os.environ[name] = "2"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=2)
    except ImportError:
        pass
    network_attempts = _install_offline_network_guard()
    started = time.perf_counter()
    cpu_started = time.process_time()
    target = Path(output_path)
    result_file = Path(result_path)
    adapter = None

    def observed_network_attempts() -> int:
        adapter_count = 0
        module = sys.modules.get("schemaguard.models.adapters.base")
        if module is not None:
            adapter_count = int(getattr(module, "NETWORK_ATTEMPT_COUNT", 0))
        return max(network_attempts[0], adapter_count)

    try:
        root = Path.cwd().resolve()
        plan_sha = str(task["payload_token"])
        envelope_path = root / "results/smoke/runtime/plans" / f"{plan_sha}.json"
        envelope = PlanEnvelope.model_validate(_read_json(envelope_path))
        if envelope.plan_sha256 != plan_sha:
            raise ValueError("worker received a mismatched plan token")
        plan = envelope.plan
        condition = next(item for item in plan.conditions if item.task_name == task["task_name"])
        if task["cache_identity"]["view_certificate_sha256"] != condition.view_certificate_sha256:
            raise ValueError("scheduler cache identity does not match the frozen view certificate")
        frames, assignment = _partition_frames(root, plan, condition)
        train_ids = frames["train"]["__sg_row_id"].tolist()
        train_targets, target_frame = _training_targets(root, train_ids, assignment)
        training_target_sha = hash_dataframe_logically(target_frame)
        fixture_sha = sha256_canonical_json(
            {
                "dataset_features_sha256": plan.dataset_features_sha256,
                "assignment_sha256": plan.assignment_sha256,
                "view_certificate_sha256": condition.view_certificate_sha256,
                "train_feature_sha256": hash_dataframe_logically(frames["train"]),
            }
        )
        from ..models.adapters.factory import create_adapter

        adapter = create_adapter(
            condition.model_id,
            seed=condition.seed,
            device=condition.device,
        )
        if adapter.preprocessing_strategy != condition.preprocessing:
            raise ValueError("adapter preprocessing differs from the frozen smoke plan")
        adapter.fit(
            frames["train"],
            train_targets,
            fixture_id=f"openml-1464-{condition.view_id}",
            fixture_sha256=fixture_sha,
            split_identity=condition.split_sha256,
            transformation_identity=condition.view_certificate_sha256,
        )
        normalized_parameters = dict(adapter.parameters)
        normalized_parameters.pop("model_path", None)
        normalized_parameters.pop("allow_auto_download", None)
        if normalized_parameters != condition.model_parameters:
            raise ValueError("adapter runtime parameters differ from the frozen smoke plan")
        observed_parameters_sha = sha256_canonical_json(normalized_parameters)
        if observed_parameters_sha != condition.parameters_sha256:
            raise ValueError("adapter parameter digest differs from the frozen plan")
        predictions = {
            partition: adapter.predict_proba(
                frames[partition],
                partition=partition,
                fixture_id=f"openml-1464-{condition.view_id}",
                fixture_sha256=fixture_sha,
                split_identity=condition.split_sha256,
                transformation_identity=condition.view_certificate_sha256,
            )
            for partition in PREDICTION_PARTITIONS
        }
        for partition, prediction in predictions.items():
            expected_ids = frames[partition]["__sg_row_id"].tolist()
            if prediction.row_ids != expected_ids or prediction.class_order != [0, 1]:
                raise ValueError(f"{partition} prediction alignment or class ordering changed")
        adapter.release()
        resource_payload = adapter.resource_record().model_dump(mode="json")
        adapter_resource = AdapterResourceSummary.model_validate(
            {key: resource_payload[key] for key in AdapterResourceSummary.model_fields}
        )
        if not adapter_resource.resource_limits_passed or not adapter_resource.cleanup_verified:
            raise ValueError("adapter resource or cleanup acceptance failed")
        resource_sha = sha256_canonical_json(adapter_resource.model_dump(mode="json"))
        condition_dir = Path(tempfile.mkdtemp(prefix="smoke-condition-", dir=target.parent))
        summaries: list[PredictionOutputSummary] = []
        try:
            archive_temp = target.with_name(f".{target.name}.{os.getpid()}.part")
            with zipfile.ZipFile(archive_temp, "w", compression=zipfile.ZIP_STORED) as archive:
                for partition, prediction in predictions.items():
                    frame = _prediction_frame(
                        prediction,
                        condition=condition,
                        plan=plan,
                        resource_sha256=resource_sha,
                        adapter_resource=adapter_resource,
                        observed_parameters_sha256=observed_parameters_sha,
                        cache_identity_sha256=sha256_canonical_json(task["cache_identity"]),
                    )
                    parquet_path = condition_dir / f"{partition}.parquet"
                    frame.to_parquet(parquet_path, index=False, compression="zstd")
                    partition_sha = sha256_file(parquet_path)
                    probability_sha = sha256_canonical_json(
                        frame[["row_id", "p_0", "p_1"]].to_dict(orient="records")
                    )
                    summaries.append(
                        PredictionOutputSummary(
                            partition=partition,
                            row_count=len(frame),
                            row_id_sha256=sha256_canonical_json(frame["row_id"].tolist()),
                            probability_sha256=probability_sha,
                            parquet_sha256=partition_sha,
                            class_order=prediction.class_order,
                        )
                    )
                    archive.write(parquet_path, arcname=f"{partition}_predictions.parquet")
                result = WorkerConditionResult(
                    plan_sha256=plan.plan_sha256,
                    condition_id=condition.condition_id,
                    source_commit=plan.source_commit,
                    observed_package_version=adapter.package_version,
                    observed_parameters_sha256=observed_parameters_sha,
                    checkpoint_identifier=adapter.checkpoint_identifier,
                    checkpoint_sha256=adapter.checkpoint_sha256,
                    training_row_id_sha256=sha256_canonical_json(train_ids),
                    training_target_sha256=training_target_sha,
                    adapter_resource=adapter_resource,
                    outputs=summaries,
                    network_attempt_count=observed_network_attempts(),
                )
                archive.writestr(
                    "condition_metadata.json",
                    json.dumps(
                        result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                    ),
                )
            os.replace(archive_temp, target)
        finally:
            import shutil

            shutil.rmtree(condition_dir, ignore_errors=True)
        _worker_result(
            task_id=task["task_id"],
            result_path=result_file,
            output_path=target,
            started=started,
            cpu_started=cpu_started,
            ok=True,
            network_attempts=observed_network_attempts(),
        )
    except BaseException as exc:
        if adapter is not None:
            try:
                adapter.release()
            except Exception:
                pass
        _worker_result(
            task_id=str(task.get("task_id", "")),
            result_path=result_file,
            output_path=target,
            started=started,
            cpu_started=cpu_started,
            ok=False,
            network_attempts=observed_network_attempts(),
            error=_worker_error_text(exc),
            traceback_text=traceback.format_exc(),
        )


class _AdapterOwnedGpuScheduler(Scheduler):
    """Keep GPU conditions sequential; the isolated adapter owns the existing GPU lock."""

    def _run_gpu_task(self, task: TaskSpec, state):  # type: ignore[no-untyped-def]
        # ModelAdapterBase acquires/releases this same lock inside the worker. Holding it here
        # would deadlock the spawned worker; Scheduler.run still serializes this queue.
        return self._execute_locked(task, state, gpu=True)


def _scheduler_plan(plan: SmokePlan) -> tuple[RunPlan, dict[str, CacheIdentity]]:
    tasks: list[TaskSpec] = []
    identities: dict[str, CacheIdentity] = {}
    for condition in plan.conditions:
        identity = CacheIdentity(
            schema_version=1,
            dataset_sha256=sha256_canonical_json(
                {
                    "features_sha256": plan.dataset_features_sha256,
                    "target_file_sha256": plan.target_artifact_sha256,
                }
            ),
            split_sha256=plan.assignment_sha256,
            view_certificate_sha256=condition.view_certificate_sha256,
            model_spec_sha256=sha256_canonical_json(
                {
                    "model_spec_sha256": condition.model_spec_sha256,
                    "view_features_sha256": condition.view_features_sha256,
                }
            ),
            model_parameters_sha256=condition.parameters_sha256,
            checkpoint_sha256=condition.checkpoint_sha256,
            dependency_lock_sha256=plan.dependency_lock_sha256,
            source_implementation_sha256=plan.code_identity_sha256,
            seed=condition.seed,
            device_policy=condition.device,
            artifact_kind="condition",
        )
        identities[condition.condition_id] = identity
        tasks.append(
            build_task(
                condition.task_name,
                identity,
                operation="synthetic_payload",
                payload_token=plan.plan_sha256,
                timeout_seconds=float(
                    plan.cpu_timeout_seconds
                    if condition.device == "cpu"
                    else plan.cuda_timeout_seconds
                ),
            )
        )
    return build_plan(tasks, random_seed=plan.seed, cpu_workers=2), identities


def _load_worker_archive(
    artifact: CacheArtifact,
    condition: SmokeCondition,
    plan: SmokePlan,
    identity: CacheIdentity,
):
    import pandas as pd

    with zipfile.ZipFile(artifact.payload_path, "r") as archive:
        names = set(archive.namelist())
        expected_names = {
            "condition_metadata.json",
            "calibration_predictions.parquet",
            "test_predictions.parquet",
        }
        if names != expected_names:
            raise ValueError("cached smoke condition archive is incomplete or contains extra files")
        metadata = WorkerConditionResult.model_validate(
            json.loads(archive.read("condition_metadata.json").decode("utf-8"))
        )
        if (
            metadata.plan_sha256 != plan.plan_sha256
            or metadata.condition_id != condition.condition_id
        ):
            raise ValueError("cached condition metadata does not match the frozen plan")
        if metadata.source_commit != plan.source_commit:
            raise ValueError("cached condition source commit differs from the plan")
        if metadata.observed_package_version != condition.package_version:
            raise ValueError("cached condition package version differs from the frozen model")
        if metadata.observed_parameters_sha256 != condition.parameters_sha256:
            raise ValueError("cached condition parameters differ from the frozen plan")
        if (metadata.checkpoint_identifier, metadata.checkpoint_sha256) != (
            condition.checkpoint_identifier,
            condition.checkpoint_sha256,
        ):
            raise ValueError("cached condition checkpoint identity differs from the plan")
        expected_cache_identity = sha256_canonical_json(identity.model_dump(mode="json"))
        prediction_files: list[PredictionFileRecord] = []
        adapter_resource = metadata.adapter_resource
        resource = ConditionResource(
            wall_seconds=adapter_resource.wall_time_seconds,
            cpu_seconds=adapter_resource.cpu_time_seconds,
            fit_seconds=adapter_resource.fit_seconds,
            prediction_seconds=adapter_resource.prediction_seconds,
            peak_worker_rss_mib=adapter_resource.peak_ram_mib,
            peak_process_tree_rss_mib=adapter_resource.peak_process_tree_ram_mib,
            peak_vram_allocated_mib=adapter_resource.peak_vram_allocated_mib,
            peak_vram_reserved_mib=adapter_resource.peak_vram_reserved_mib,
            free_vram_before_mib=adapter_resource.free_vram_before_mib,
            telemetry_complete=adapter_resource.telemetry_complete,
            cleanup_passed=adapter_resource.cleanup_verified,
            adapter_resource_sha256=sha256_canonical_json(adapter_resource.model_dump(mode="json")),
        )
        for output in metadata.outputs:
            entry = f"{output.partition}_predictions.parquet"
            payload = archive.read(entry)
            observed_sha = hashlib.sha256(payload).hexdigest()
            if observed_sha != output.parquet_sha256:
                raise ValueError("cached prediction Parquet checksum is invalid")
            frame = pd.read_parquet(__import__("io").BytesIO(payload))
            if len(frame) != output.row_count or tuple(frame.columns) != PREDICTION_COLUMNS:
                raise ValueError("cached prediction schema or row count is invalid")
            if frame["plan_sha256"].ne(plan.plan_sha256).any():
                raise ValueError("cached prediction plan lineage mismatch")
            if frame["condition_id"].ne(condition.condition_id).any():
                raise ValueError("cached prediction condition lineage mismatch")
            if frame["class_order"].ne(",".join(str(value) for value in output.class_order)).any():
                raise ValueError("cached prediction class ordering is inconsistent")
            if output.class_order != [0, 1]:
                raise ValueError("smoke output class order is not canonical binary order")
            expected_view = next(view for view in plan.views if view.view_id == condition.view_id)
            if len(frame) != expected_view.partition_rows[output.partition]:
                raise ValueError("cached prediction row count differs from the frozen split")
            if (
                sha256_canonical_json(frame["row_id"].tolist())
                != expected_view.partition_row_id_sha256[output.partition]
            ):
                raise ValueError("cached prediction row IDs differ from the frozen split")
            lineage = {
                "run_identity_sha256": plan.plan_sha256,
                "model_id": condition.model_id,
                "view_id": condition.view_id,
                "device": condition.device,
                "seed": condition.seed,
                "dataset_features_sha256": plan.dataset_features_sha256,
                "target_artifact_sha256": plan.target_artifact_sha256,
                "assignment_sha256": plan.assignment_sha256,
                "view_certificate_sha256": condition.view_certificate_sha256,
                "view_features_sha256": condition.view_features_sha256,
                "model_spec_sha256": condition.model_spec_sha256,
                "planned_parameters_sha256": condition.parameters_sha256,
                "source_commit": plan.source_commit,
                "dependency_lock_sha256": plan.dependency_lock_sha256,
                "code_identity_sha256": plan.code_identity_sha256,
                "dataset_id": plan.dataset_id,
                "cache_identity_sha256": expected_cache_identity,
                "validation_status": "PASS",
            }
            for field, expected_value in lineage.items():
                if frame[field].ne(expected_value).any():
                    raise ValueError(f"cached prediction lineage mismatch in {field}")
            if frame["partition"].ne(output.partition).any():
                raise ValueError("cached prediction partition identity changed")
            if frame["target_artifact_sha256"].isna().any():
                raise ValueError("cached prediction target source hash is missing")
            import numpy as np

            probabilities = frame[["p_0", "p_1"]].to_numpy(dtype="float64")
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
                or np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > plan.probability_sum_tolerance
            ):
                raise ValueError("cached prediction probabilities are invalid")
            if not np.array_equal(
                frame["predicted_class"].to_numpy(dtype="int64"),
                probabilities.argmax(axis=1),
            ):
                raise ValueError("cached predicted classes differ from canonical argmax")
            if sha256_canonical_json(frame["row_id"].tolist()) != output.row_id_sha256:
                raise ValueError("cached prediction row identity hash is invalid")
            probability_sha = sha256_canonical_json(
                frame[["row_id", "p_0", "p_1"]].to_dict(orient="records")
            )
            if probability_sha != output.probability_sha256:
                raise ValueError("cached prediction probability hash is invalid")
            relative = f"results/smoke/predictions/{condition.task_name}_{output.partition}.parquet"
            prediction_files.append(
                PredictionFileRecord(
                    partition=output.partition,
                    relative_path=relative,
                    sha256=observed_sha,
                    row_count=output.row_count,
                    row_id_sha256=output.row_id_sha256,
                    probability_sha256=output.probability_sha256,
                )
            )
        return metadata, resource, prediction_files


def _load_scheduler_state(state_path: Path) -> dict[str, Any]:
    from ..runner.contracts import SchedulerState

    value = SchedulerState.model_validate(_read_json(state_path))
    return {task_id: record.model_dump(mode="json") for task_id, record in value.tasks.items()}


def run_conditions(
    root: str | Path,
    plan: SmokePlan,
    config: SmokeConfig,
    *,
    resume: bool,
) -> tuple[list[ConditionRun], SchedulerResult, int]:
    """Run one cold/resume scheduler pass and validate/materialize all cache results."""
    from ..runner import resources as runner_resources

    project = Path(root).resolve()
    scheduler_config = CacheSchedulerConfig.model_validate(
        _read_yaml(project / config.scheduler_config)
    )
    if (
        scheduler_config.cpu_workers != config.execution.cpu_workers
        or scheduler_config.gpu_workers != config.execution.gpu_workers
        or scheduler_config.ram_hard_limit_mib != config.execution.maximum_ram_mib
        or scheduler_config.vram_soft_limit_mib != config.execution.maximum_vram_mib
        or scheduler_config.gpu_headroom_mib < config.execution.minimum_gpu_headroom_mib
    ):
        raise ValueError("smoke resource policy differs from the accepted scheduler policy")
    scheduler_plan, identities = _scheduler_plan(plan)
    cache = CacheStore(
        project / scheduler_config.cache_root,
        lock_timeout_seconds=scheduler_config.lock_timeout_seconds,
    )
    preexisting_valid_cache_ids: set[str] = set()
    if resume:
        for condition in plan.conditions:
            try:
                artifact = cache.read_validated(identities[condition.condition_id])
            except CacheIntegrityError:
                artifact = None
            if artifact is not None:
                preexisting_valid_cache_ids.add(condition.condition_id)
    index = CacheIndex(
        project / scheduler_config.state_root / "smoke_cache_index.sqlite", cache.root
    )
    scheduler = _AdapterOwnedGpuScheduler(
        project,
        scheduler_config,
        cache,
        index,
        source_commit=plan.source_commit,
    )
    prior_worker = runner_resources.worker_entry
    old_cwd = Path.cwd()
    total_network_attempts = 0
    try:
        os.chdir(project)
        runner_resources.worker_entry = smoke_worker_entry
        result = scheduler.run(scheduler_plan, resume=resume)
    finally:
        runner_resources.worker_entry = prior_worker
        os.chdir(old_cwd)
    task_state = _load_scheduler_state(result.state_path)
    if resume and len(preexisting_valid_cache_ids) > result.manifest.validated_cache_hits:
        raise ValueError(
            "resume scheduler report accounts for fewer hits than validated cache inputs"
        )
    task_by_name = {task.task_name: task for task in scheduler_plan.tasks}
    failures = {item.task_id: item for item in result.manifest.failures}
    runs: list[ConditionRun] = []
    for condition in plan.conditions:
        task = task_by_name[condition.task_name]
        record = task_state[task.task_id]
        cache_hit = condition.condition_id in preexisting_valid_cache_ids
        if record.get("state") != "COMPLETE":
            failure = failures.get(task.task_id)
            category = failure.category if failure is not None else record.get("failure_category")
            reason = failure.reason if failure is not None else record.get("failure_reason")
            reason_text = str(reason or "scheduler condition did not complete")
            category = _reported_failure_category(category, reason_text)
            status: SmokeConditionStatus = (
                "BLOCKED" if str(category).startswith("BLOCKED") else "FAIL"
            )
            runs.append(
                ConditionRun(
                    condition_id=condition.condition_id,
                    cache_identity_sha256=identities[condition.condition_id].cache_key,
                    model_id=condition.model_id,
                    view_id=condition.view_id,
                    device=condition.device,
                    status=status,
                    cache_status="not_published",
                    resource=None,
                    failure_category=str(category or "FAIL_MODEL_RUNTIME"),
                    failure_reason=reason_text,
                    traceback_relative_path=(
                        failure.traceback_relative_path if failure is not None else None
                    ),
                )
            )
            continue
        artifact = cache.read_validated(identities[condition.condition_id])
        if artifact is None:
            raise ValueError(
                f"completed scheduler task lacks a valid cached artifact: {condition.task_name}"
            )
        metadata, resource, prediction_files = _load_worker_archive(
            artifact, condition, plan, identities[condition.condition_id]
        )
        total_network_attempts += metadata.network_attempt_count
        for output in metadata.outputs:
            entry = f"{output.partition}_predictions.parquet"
            target = (
                project
                / "results/smoke/predictions"
                / f"{condition.task_name}_{output.partition}.parquet"
            )
            with zipfile.ZipFile(artifact.payload_path, "r") as archive:
                payload = archive.read(entry)
            if target.exists():
                if sha256_file(target) != output.parquet_sha256:
                    quarantine = target.with_name(
                        f"{target.name}.invalid-{sha256_file(target)}-{uuid.uuid4().hex}"
                    )
                    os.replace(target, quarantine)
                else:
                    continue
            atomic_write_bytes(target, payload)
        runs.append(
            ConditionRun(
                condition_id=condition.condition_id,
                cache_identity_sha256=identities[condition.condition_id].cache_key,
                model_id=condition.model_id,
                view_id=condition.view_id,
                device=condition.device,
                status="PASS",
                cache_status="validated_cache_hit" if cache_hit else "created",
                resource=resource,
                prediction_files=prediction_files,
            )
        )
    return runs, result, max(total_network_attempts, result.manifest.offline_network_attempt_count)


def _read_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


__all__ = ["PREDICTION_COLUMNS", "run_conditions", "smoke_worker_entry"]
