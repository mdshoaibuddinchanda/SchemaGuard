"""Tiny, deterministic smoke-plan and spawn-safe worker fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from schemaguard.experiments.contracts import (
    MODEL_ORDER,
    AdapterResourceSummary,
    PlanEnvelope,
    PredictionOutputSummary,
    SmokeCondition,
    SmokePlan,
    SmokeView,
    condition_identity,
)
from schemaguard.experiments.execution import PREDICTION_COLUMNS, _worker_result
from schemaguard.models.adapters.contracts import FOUNDATION_CHECKPOINTS
from schemaguard.utils.hashing import sha256_canonical_json, sha256_file

_MODEL_PACKAGES = {
    "LR-1.9": ("scikit-learn", "1.9.1"),
    "CAT-1.2": ("catboost", "1.2.10"),
    "XGB-3.4": ("xgboost", "3.4.1"),
    "TPFN3-8.5": ("tabpfn", "8.5.0"),
    "TICL2-2.2": ("tabicl", "2.2.0"),
}
_PARTITIONS = ("train", "calibration", "test")


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tiny_smoke_plan(root: Path) -> SmokePlan:
    """Create a strict ten-condition plan bound to a temporary target table."""
    target_path = root / "data/processed/openml/1464/targets.parquet"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    row_ids = [f"{part}-{index}" for part in _PARTITIONS for index in range(2)]
    pd.DataFrame({"__sg_row_id": row_ids, "target_code": [0, 1, 0, 1, 0, 1]}).to_parquet(
        target_path, index=False
    )
    target_sha = sha256_file(target_path)
    partitions = {name: [f"{name}-0", f"{name}-1"] for name in _PARTITIONS}
    row_hashes = {name: sha256_canonical_json(ids) for name, ids in partitions.items()}
    row_counts = {name: len(ids) for name, ids in partitions.items()}
    views: list[SmokeView] = []
    view_content: dict[str, tuple[str, str]] = {}
    for view_id, view_name in (("V00", "identity"), ("V01", "numeric_affine_units")):
        partition_hashes = {name: _digest(f"{view_id}:{name}:features") for name in _PARTITIONS}
        feature_hash = sha256_canonical_json(partition_hashes)
        cert_ids = [_digest(f"{view_id}:cert:{index}") for index in range(3)]
        cert_hash = sha256_canonical_json(cert_ids)
        views.append(
            SmokeView(
                view_id=view_id,
                view_name=view_name,
                source_hash=_digest("source"),
                output_hash=_digest(f"{view_id}:output"),
                certificate_sha256=cert_hash,
                certificate_ids=cert_ids,
                view_features_sha256=feature_hash,
                partition_feature_sha256=partition_hashes,
                partition_feature_file_sha256={
                    name: _digest(f"{view_id}:{name}:file") for name in _PARTITIONS
                },
                partition_certificate_file_sha256={
                    name: (_digest(f"{view_id}:{name}:certificate") if view_id == "V01" else None)
                    for name in _PARTITIONS
                },
                manifest_sha256=_digest("manifest") if view_id == "V01" else None,
                partition_row_id_sha256=row_hashes,
                partition_rows=row_counts,
            )
        )
        view_content[view_id] = (feature_hash, cert_hash)

    dataset_sha = _digest("features")
    source_sha = _digest("dataset-source")
    assignment_sha = _digest("assignment")
    lock_sha = _digest("dependency-lock")
    code_sha = _digest("implementation")
    conditions: list[SmokeCondition] = []
    for model_id in MODEL_ORDER:
        package, version = _MODEL_PACKAGES[model_id]
        device = "cuda" if model_id in FOUNDATION_CHECKPOINTS else "cpu"
        checkpoint_name, checkpoint_sha = FOUNDATION_CHECKPOINTS.get(model_id, (None, None))
        for view_id in ("V00", "V01"):
            parameters: dict[str, Any] = {}
            parameter_sha = sha256_canonical_json(parameters)
            spec_sha = _digest(f"model-spec:{model_id}")
            view_feature_sha, view_cert_sha = view_content[view_id]
            condition_id = condition_identity(
                dataset_features_sha256=dataset_sha,
                target_artifact_sha256=target_sha,
                model_id=model_id,
                package_version=version,
                view_id=view_id,
                device=device,
                precision_policy="auto" if device == "cuda" else "native",
                seed=1729,
                model_spec_sha256=spec_sha,
                parameters_sha256=parameter_sha,
                view_certificate_sha256=view_cert_sha,
                view_features_sha256=view_feature_sha,
                split_sha256=assignment_sha,
                dependency_lock_sha256=lock_sha,
                source_implementation_sha256=code_sha,
                checkpoint_sha256=checkpoint_sha,
            )
            conditions.append(
                SmokeCondition(
                    condition_id=condition_id,
                    task_name=f"{model_id}__{view_id}",
                    model_id=model_id,
                    package_name=package,
                    package_version=version,
                    dataset_features_sha256=dataset_sha,
                    target_artifact_sha256=target_sha,
                    view_id=view_id,
                    device=device,
                    precision_policy="auto" if device == "cuda" else "native",
                    seed=1729,
                    model_spec_sha256=spec_sha,
                    parameters_sha256=parameter_sha,
                    model_parameters=parameters,
                    preprocessing="fixture_only",
                    checkpoint_identifier=checkpoint_name,
                    checkpoint_sha256=checkpoint_sha,
                    view_certificate_sha256=view_cert_sha,
                    view_features_sha256=view_feature_sha,
                    split_sha256=assignment_sha,
                    dependency_lock_sha256=lock_sha,
                    source_implementation_sha256=code_sha,
                )
            )
    return SmokePlan(
        protocol_version="schema_guard_smoke_v1",
        source_commit="a" * 40,
        code_identity_sha256=code_sha,
        dependency_lock_sha256=lock_sha,
        dataset_features_sha256=dataset_sha,
        dataset_source_sha256=source_sha,
        target_artifact_sha256=target_sha,
        assignment_sha256=assignment_sha,
        assignment_logical_sha256=_digest("assignment-logical"),
        transformation_inventory_sha256=_digest("transformation-inventory"),
        configuration_identity_sha256=_digest("configuration"),
        output_path_policy_sha256=_digest("output-path-policy"),
        registry_parameter_decisions=[
            "catboost_yaml_no_scalar",
            "tabicl_checkpoint_version_from_checkpoint_field",
        ],
        views=views,
        conditions=conditions,
        test_label_access_boundary="after_all_predictions_complete",
        primary_metric="brier_score",
        ece_binning_rule="equal_width_max_probability_confidence",
        log_loss_epsilon=1e-15,
        probability_sum_tolerance=1e-6,
        reconstruction_rtol=1e-10,
        reconstruction_atol=1e-12,
        ece_bins=10,
        maximum_ram_mib=28672,
        maximum_vram_mib=3600,
        minimum_gpu_headroom_mib=512,
        cpu_timeout_seconds=30,
        cuda_timeout_seconds=30,
    )


def fake_smoke_worker_entry(task: dict[str, Any], output_path: str, result_path: str) -> None:
    """Spawn-safe synthetic worker that emits valid smoke-shaped records."""
    from schemaguard.experiments.contracts import PlanEnvelope

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "2"
    started = time.perf_counter()
    cpu_started = time.process_time()
    target = Path(output_path)
    result_file = Path(result_path)
    try:
        plan_path = Path.cwd() / "results/smoke/runtime/plans" / f"{task['payload_token']}.json"
        plan = PlanEnvelope.model_validate_json(plan_path.read_text(encoding="utf-8")).plan
        condition = next(item for item in plan.conditions if item.task_name == task["task_name"])
        resource = AdapterResourceSummary(
            model_load_seconds=0.0,
            preprocessing_fit_seconds=0.0,
            fit_seconds=0.01,
            prediction_seconds=0.01,
            wall_time_seconds=0.02,
            cpu_time_seconds=0.01,
            peak_ram_mib=64.0,
            peak_process_tree_ram_mib=128.0,
            peak_vram_allocated_mib=64.0 if condition.device == "cuda" else None,
            peak_vram_reserved_mib=96.0 if condition.device == "cuda" else None,
            free_vram_before_mib=3000.0 if condition.device == "cuda" else None,
            free_vram_after_mib=2900.0 if condition.device == "cuda" else None,
            telemetry_complete=True,
            resource_limits_passed=True,
            cleanup_verified=True,
        )
        resource_sha = sha256_canonical_json(resource.model_dump(mode="json"))
        temp_dir = Path(tempfile.mkdtemp(prefix="smoke-fixture-", dir=target.parent))
        summaries: list[PredictionOutputSummary] = []
        temporary_archive = target.with_name(f".{target.name}.{os.getpid()}.part")
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_STORED) as archive:
            for partition in ("calibration", "test"):
                row_ids = [f"{partition}-0", f"{partition}-1"]
                probabilities = [(0.8, 0.2), (0.2, 0.8)]
                rows = []
                cache_identity_sha = sha256_canonical_json(task["cache_identity"])
                for row_id, (p0, p1) in zip(row_ids, probabilities, strict=True):
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
                            "partition": partition,
                            "row_id": row_id,
                            "class_order": "0,1",
                            "p_0": p0,
                            "p_1": p1,
                            "predicted_class": int(p1 > p0),
                            "fit_time_seconds": 0.01,
                            "prediction_time_seconds": 0.01,
                            "peak_ram_mib": 64.0,
                            "peak_vram_mib": resource.peak_vram_reserved_mib,
                            "dataset_features_sha256": plan.dataset_features_sha256,
                            "target_artifact_sha256": plan.target_artifact_sha256,
                            "assignment_sha256": plan.assignment_sha256,
                            "view_certificate_sha256": condition.view_certificate_sha256,
                            "view_features_sha256": condition.view_features_sha256,
                            "model_spec_sha256": condition.model_spec_sha256,
                            "planned_parameters_sha256": condition.parameters_sha256,
                            "observed_parameters_sha256": condition.parameters_sha256,
                            "checkpoint_sha256": condition.checkpoint_sha256,
                            "source_commit": plan.source_commit,
                            "dependency_lock_sha256": plan.dependency_lock_sha256,
                            "code_identity_sha256": plan.code_identity_sha256,
                            "cache_identity_sha256": cache_identity_sha,
                            "validation_status": "PASS",
                            "resource_sha256": resource_sha,
                        }
                    )
                frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
                frame_path = temp_dir / f"{partition}.parquet"
                frame.to_parquet(frame_path, index=False, compression="zstd")
                relative = f"{partition}_predictions.parquet"
                archive.write(frame_path, arcname=relative)
                summaries.append(
                    PredictionOutputSummary(
                        partition=partition,
                        row_count=len(frame),
                        row_id_sha256=sha256_canonical_json(frame["row_id"].tolist()),
                        probability_sha256=sha256_canonical_json(
                            frame[["row_id", "p_0", "p_1"]].to_dict(orient="records")
                        ),
                        parquet_sha256=sha256_file(frame_path),
                        class_order=[0, 1],
                    )
                )
            result = {
                "schema_version": 1,
                "plan_sha256": plan.plan_sha256,
                "condition_id": condition.condition_id,
                "source_commit": plan.source_commit,
                "observed_package_version": condition.package_version,
                "observed_parameters_sha256": condition.parameters_sha256,
                "checkpoint_identifier": condition.checkpoint_identifier,
                "checkpoint_sha256": condition.checkpoint_sha256,
                "training_row_id_sha256": _digest("training-row-ids"),
                "training_target_sha256": _digest("training-targets"),
                "adapter_resource": resource.model_dump(mode="json"),
                "outputs": [item.model_dump(mode="json") for item in summaries],
                "network_attempt_count": 0,
            }
            archive.writestr(
                "condition_metadata.json",
                json.dumps(result, sort_keys=True, separators=(",", ":")),
            )
        os.replace(temporary_archive, target)
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        _worker_result(
            task_id=task["task_id"],
            result_path=result_file,
            output_path=target,
            started=started,
            cpu_started=cpu_started,
            ok=True,
            network_attempts=0,
        )
    except BaseException as exc:
        _worker_result(
            task_id=str(task.get("task_id", "")),
            result_path=result_file,
            output_path=target,
            started=started,
            cpu_started=cpu_started,
            ok=False,
            network_attempts=0,
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )


def write_test_plan(root: Path, plan: SmokePlan) -> None:
    envelope = PlanEnvelope(plan_sha256=plan.plan_sha256, plan=plan)
    path = root / "results/smoke/runtime/plans" / f"{plan.plan_sha256}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(envelope.model_dump_json(), encoding="utf-8")


__all__ = ["fake_smoke_worker_entry", "tiny_smoke_plan", "write_test_plan"]
