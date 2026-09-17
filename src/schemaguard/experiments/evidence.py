"""Sanitized run evidence, runtime capture, and independent artifact validation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..utils.hashing import sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_json
from .contracts import (
    ConditionRun,
    EvaluationSource,
    GateResult,
    MetricRecord,
    PairedMetricRecord,
    RuntimeEstimate,
    SmokeEnvironment,
    SmokeEvidenceCondition,
    SmokeEvidenceInventory,
    SmokeGateRecord,
    SmokePlan,
    SmokeRunReport,
    SmokeRunStatus,
    SmokeValidationReport,
)
from .execution import PREDICTION_COLUMNS


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def estimate_smoke_runtime(root: str | Path, plan: SmokePlan) -> RuntimeEstimate:
    """Estimate the frozen run from accepted binary-fixture adapter timings."""
    project = Path(root).resolve()
    inventory_path = project / "artifacts/handoff/model_adapter_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = inventory.get("records")
    if not isinstance(records, list):
        raise ValueError("accepted adapter inventory has no timing records")
    condition_seconds: dict[str, float] = {}
    for condition in plan.conditions:
        candidates = [
            float(record["runtime_seconds"])
            for record in records
            if isinstance(record, dict)
            and record.get("model_id") == condition.model_id
            and record.get("package_version") == condition.package_version
            and record.get("device") == condition.device
            and record.get("fixture_id") == "binary_numerical"
            and record.get("status") == "PASS"
            and isinstance(record.get("runtime_seconds"), (int, float))
            and 0 < float(record["runtime_seconds"]) < float("inf")
        ]
        if not candidates:
            raise ValueError(
                f"no accepted binary-fixture runtime supports estimate for {condition.model_id}"
            )
        condition_seconds[condition.condition_id] = max(candidates)

    from .execution import _scheduler_plan

    scheduler_plan, _ = _scheduler_plan(plan)
    by_task = {item.task_name: item for item in plan.conditions}
    cpu_loads = [0.0, 0.0]
    gpu_total = 0.0
    for task in scheduler_plan.tasks:
        condition = by_task[task.task_name]
        duration = condition_seconds[condition.condition_id]
        if condition.device == "cpu":
            worker = min(range(2), key=cpu_loads.__getitem__)
            cpu_loads[worker] += duration
        else:
            gpu_total += duration
    cpu_ids = [item.condition_id for item in plan.conditions if item.device == "cpu"]
    gpu_ids = [item.condition_id for item in plan.conditions if item.device == "cuda"]
    expected_cpu = max(cpu_loads)
    estimated_total = (expected_cpu + gpu_total) * 1.5 + 60.0
    return RuntimeEstimate(
        source_inventory_sha256=sha256_file(inventory_path),
        method="prior_binary_probe_max_list_schedule_with_safety_factor",
        condition_seconds=condition_seconds,
        cpu_condition_ids=cpu_ids,
        gpu_condition_ids=gpu_ids,
        expected_cpu_seconds=expected_cpu,
        expected_gpu_seconds=gpu_total,
        estimated_total_seconds=estimated_total,
    )


def capture_environment() -> SmokeEnvironment:
    """Capture only non-secret, non-user-specific runtime properties."""
    import importlib.metadata

    cpu_id = platform.processor() or platform.machine() or "unknown"
    physical: int | None = None
    logical: int | None = None
    ram_total: float | None = None
    ram_available: float | None = None
    runtime_versions: dict[str, str] = {}
    telemetry: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "scikit-learn",
        "catboost",
        "xgboost",
        "tabpfn",
        "tabicl",
        "torch",
        "psutil",
    ):
        try:
            runtime_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        ram_total = float(memory.total / (1024**2))
        ram_available = float(memory.available / (1024**2))
    except Exception as exc:
        telemetry["ram"] = type(exc).__name__
    gpu_name: str | None = None
    gpu_total: float | None = None
    gpu_free: float | None = None
    driver: str | None = None
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        fields = [item.strip() for item in output.split(",")]
        gpu_name, gpu_total, gpu_free, driver = (
            fields[0],
            float(fields[1]),
            float(fields[2]),
            fields[3],
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        telemetry["gpu"] = type(exc).__name__
    torch_version: str | None = None
    cuda_build: str | None = None
    cuda_visible = False
    try:
        torch_version = importlib.metadata.version("torch")
        import torch

        cuda_build = torch.version.cuda
        cuda_visible = bool(torch.cuda.is_available())
        if cuda_visible and gpu_name is None:
            gpu_name = str(torch.cuda.get_device_name(0))
        if cuda_visible and gpu_total is None:
            _, total_bytes = torch.cuda.mem_get_info(0)
            gpu_total = float(total_bytes / (1024**2))
        if cuda_visible:
            free_bytes, _ = torch.cuda.mem_get_info(0)
            gpu_free = float(free_bytes / (1024**2))
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError) as exc:
        telemetry["torch"] = type(exc).__name__
    return SmokeEnvironment(
        os_platform=platform.platform(),
        python_version=platform.python_version(),
        conda_environment=os.environ.get("CONDA_DEFAULT_ENV"),
        cpu_identifier=cpu_id,
        cpu_physical_cores=physical,
        cpu_logical_cores=logical,
        ram_total_mib=ram_total,
        ram_available_mib=ram_available,
        gpu_name=gpu_name,
        gpu_total_mib=gpu_total,
        gpu_free_mib=gpu_free,
        gpu_driver=driver,
        torch_version=torch_version,
        torch_cuda_build=cuda_build,
        cuda_visible=cuda_visible,
        runtime_package_versions=runtime_versions,
        telemetry_warnings=[f"{name}:{kind}" for name, kind in sorted(telemetry.items())],
    )


def build_run_report(
    *,
    plan: SmokePlan,
    scheduler_result: Any,
    conditions: list[ConditionRun],
    environment: SmokeEnvironment,
    runtime_estimate: RuntimeEstimate,
    started_at: str,
    ended_at: str,
    status: SmokeRunStatus,
    metrics: list[MetricRecord],
    paired_metrics: list[PairedMetricRecord],
    metric_sha256: str | None,
    paired_metric_sha256: str | None,
    protected_foundation_unchanged: bool,
    protected_foundation_comparison_sha256: str | None,
    protected_split_validation_sha256: str | None,
    labels_evaluated: bool,
    evaluation_source: EvaluationSource,
    duplicate_prediction_artifacts: int = 0,
    result_rewrites: int = 0,
    cold_run_report_sha256: str | None = None,
    resume_prediction_hashes_before_sha256: str | None = None,
    resume_prediction_hashes_after_sha256: str | None = None,
    resume_verification_sha256: str | None = None,
    resume_prediction_hashes_unchanged: bool = False,
    resume_prediction_mtimes_unchanged: bool = False,
    network_attempt_count: int | None = None,
    cold_run_wall_seconds: float | None = None,
) -> SmokeRunReport:
    manifest = scheduler_result.manifest
    failed = sum(item.status == "FAIL" for item in conditions)
    blocked = sum(item.status == "BLOCKED" for item in conditions)
    return SmokeRunReport(
        status=status,
        source_commit=plan.source_commit,
        plan_sha256=plan.plan_sha256,
        scheduler_plan_sha256=manifest.plan_hash,
        run_mode=manifest.mode,
        environment=environment,
        runtime_estimate=runtime_estimate,
        started_at=started_at,
        ended_at=ended_at,
        cold_run_wall_seconds=cold_run_wall_seconds,
        planned_conditions=manifest.planned,
        executed_conditions=manifest.executed,
        validated_cache_hits=manifest.validated_cache_hits,
        failed_conditions=failed,
        blocked_conditions=blocked,
        max_cpu_concurrency=manifest.max_cpu_concurrency,
        max_gpu_concurrency=manifest.max_gpu_concurrency,
        duplicate_prediction_artifacts=duplicate_prediction_artifacts,
        result_rewrites=result_rewrites,
        evaluation_source=evaluation_source,
        conditions=conditions,
        metrics=metrics,
        paired_metrics=paired_metrics,
        metric_table_sha256=metric_sha256,
        paired_metric_table_sha256=paired_metric_sha256,
        cold_run_report_sha256=cold_run_report_sha256,
        resume_prediction_hashes_before_sha256=resume_prediction_hashes_before_sha256,
        resume_prediction_hashes_after_sha256=resume_prediction_hashes_after_sha256,
        resume_verification_sha256=resume_verification_sha256,
        resume_prediction_hashes_unchanged=resume_prediction_hashes_unchanged,
        resume_prediction_mtimes_unchanged=resume_prediction_mtimes_unchanged,
        protected_foundation_unchanged=protected_foundation_unchanged,
        protected_foundation_comparison_sha256=protected_foundation_comparison_sha256,
        protected_split_validation_sha256=protected_split_validation_sha256,
        test_labels_opened_after_predictions=True,
        labels_evaluated=labels_evaluated,
        network_attempt_count=(
            manifest.offline_network_attempt_count
            if network_attempt_count is None
            else network_attempt_count
        ),
    )


def validate_prediction_tables(root: str | Path, plan: SmokePlan, report: SmokeRunReport) -> int:
    """Independently check all row-level prediction artifacts without opening labels."""
    import numpy as np
    import pandas as pd

    project = Path(root).resolve()
    condition_plan = {item.condition_id: item for item in plan.conditions}
    run_by_id = {item.condition_id: item for item in report.conditions}
    paths = [
        project / item.relative_path
        for condition in report.conditions
        for item in condition.prediction_files
    ]
    expected = 20 if all(item.status == "PASS" for item in report.conditions) else len(paths)
    if len(paths) != expected or len(set(paths)) != len(paths):
        raise ValueError("prediction artifact count is incomplete or contains duplicate paths")
    for path in paths:
        record = next(
            artifact
            for condition in report.conditions
            for artifact in condition.prediction_files
            if project / artifact.relative_path == path
        )
        if not path.is_file() or sha256_file(path) != record.sha256:
            raise ValueError(
                f"prediction output is absent or checksum-invalid: {record.relative_path}"
            )
        frame = pd.read_parquet(path)
        if tuple(frame.columns) != PREDICTION_COLUMNS or frame.empty:
            raise ValueError("prediction table has an invalid schema or no rows")
        condition_id = str(frame["condition_id"].iloc[0])
        planned = condition_plan.get(condition_id)
        run = run_by_id.get(condition_id)
        if planned is None or run is None or run.status != "PASS":
            raise ValueError("prediction table references an unknown condition")
        if frame["condition_id"].ne(condition_id).any():
            raise ValueError("prediction table mixes condition identities")
        if frame["row_id"].duplicated().any():
            raise ValueError("prediction table contains duplicate row IDs")
        if record.partition not in {"calibration", "test"}:
            raise ValueError("prediction manifest names an unsupported partition")
        view = next(item for item in plan.views if item.view_id == planned.view_id)
        if len(frame) != view.partition_rows[record.partition]:
            raise ValueError("prediction row count differs from the planned partition")
        if (
            sha256_canonical_json(frame["row_id"].tolist())
            != view.partition_row_id_sha256[record.partition]
        ):
            raise ValueError("prediction row IDs differ from the planned partition")
        if frame["partition"].ne(record.partition).any():
            raise ValueError("prediction rows do not match their manifest partition")
        lineage = {
            "run_identity_sha256": plan.plan_sha256,
            "plan_sha256": plan.plan_sha256,
            "condition_id": planned.condition_id,
            "dataset_id": plan.dataset_id,
            "model_id": planned.model_id,
            "view_id": planned.view_id,
            "device": planned.device,
            "seed": planned.seed,
            "dataset_features_sha256": plan.dataset_features_sha256,
            "target_artifact_sha256": plan.target_artifact_sha256,
            "assignment_sha256": plan.assignment_sha256,
            "view_certificate_sha256": planned.view_certificate_sha256,
            "view_features_sha256": planned.view_features_sha256,
            "model_spec_sha256": planned.model_spec_sha256,
            "planned_parameters_sha256": planned.parameters_sha256,
            "observed_parameters_sha256": planned.parameters_sha256,
            "checkpoint_sha256": planned.checkpoint_sha256,
            "source_commit": plan.source_commit,
            "dependency_lock_sha256": plan.dependency_lock_sha256,
            "code_identity_sha256": plan.code_identity_sha256,
            "cache_identity_sha256": run.cache_identity_sha256,
        }
        for column, value in lineage.items():
            if frame[column].ne(value).any():
                raise ValueError(f"prediction lineage differs from its plan in {column}")
        if frame["class_order"].ne("0,1").any():
            raise ValueError("prediction class order is not canonical binary order")
        probabilities = frame[["p_0", "p_1"]].to_numpy(dtype="float64")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError("prediction table contains invalid probabilities")
        if np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > plan.probability_sum_tolerance:
            raise ValueError("prediction table violates the frozen probability-sum tolerance")
        if frame["validation_status"].ne("PASS").any():
            raise ValueError("prediction table contains an unsuccessful validation status")
        if not np.array_equal(
            frame["predicted_class"].to_numpy(dtype="int64"),
            probabilities.argmax(axis=1),
        ):
            raise ValueError("persisted predicted classes differ from probability argmax")
        for column in ("fit_time_seconds", "prediction_time_seconds"):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"prediction resource field {column} is invalid")
        if len(frame) != record.row_count:
            raise ValueError("prediction table row count differs from its manifest")
    return len(paths)


def validate_metric_tables(root: str | Path, report: SmokeRunReport) -> tuple[int, int]:
    import numpy as np
    import pandas as pd

    project = Path(root).resolve()
    if not report.metrics:
        if report.metric_table_sha256 is not None or report.paired_metric_table_sha256 is not None:
            raise ValueError("metric checksums exist without metric records")
        return 0, 0
    metric_path = project / "results/smoke/metrics/model_metrics.parquet"
    paired_path = project / "results/smoke/metrics/paired_view_metrics.parquet"
    if sha256_file(metric_path) != report.metric_table_sha256:
        raise ValueError("metric table checksum differs from the run report")
    if sha256_file(paired_path) != report.paired_metric_table_sha256:
        raise ValueError("paired metric table checksum differs from the run report")
    metrics = pd.read_parquet(metric_path)
    paired = pd.read_parquet(paired_path)
    if len(metrics) != len(report.metrics) or len(paired) != len(report.paired_metrics):
        raise ValueError("metric table row counts differ from strict evidence contracts")
    if tuple(metrics.columns) != tuple(MetricRecord.model_fields) or tuple(paired.columns) != tuple(
        PairedMetricRecord.model_fields
    ):
        raise ValueError("metric Parquet columns differ from their strict contracts")
    if not set(metrics["roc_auc_status"]).issubset({"defined", "undefined_single_class"}):
        raise ValueError("AUROC undefined states are not represented explicitly")
    for column in (
        "brier_score",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "expected_calibration_error",
    ):
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"metric column {column} has non-finite or nonnumeric values")
    auc = pd.to_numeric(metrics["roc_auc"], errors="coerce").to_numpy(dtype="float64")
    auc_defined = metrics["roc_auc_status"].eq("defined").to_numpy()
    if not np.isfinite(auc[auc_defined]).all() or not np.isnan(auc[~auc_defined]).all():
        raise ValueError("AUROC values and undefined-state declarations disagree")
    for row, metric_expected in zip(metrics.to_dict(orient="records"), report.metrics, strict=True):
        if np.isnan(float(row["roc_auc"])):
            row["roc_auc"] = None
        observed = MetricRecord.model_validate(row)
        if observed.model_dump(mode="json") != metric_expected.model_dump(mode="json"):
            raise ValueError("metric table contents differ from their strict run-report records")
    for row, paired_expected in zip(
        paired.to_dict(orient="records"), report.paired_metrics, strict=True
    ):
        paired_observed = PairedMetricRecord.model_validate(row)
        if paired_observed.model_dump(mode="json") != paired_expected.model_dump(mode="json"):
            raise ValueError("paired-metric table differs from strict run-report records")
    return len(metrics), len(paired)


def make_inventory(
    plan: SmokePlan,
    report: SmokeRunReport,
    report_sha256: str,
    acceptance_gates: list[SmokeGateRecord],
) -> SmokeEvidenceInventory:
    return SmokeEvidenceInventory(
        status=report.status,
        source_commit=report.source_commit,
        plan_sha256=plan.plan_sha256,
        run_report_sha256=report_sha256,
        resume_verification_sha256=report.resume_verification_sha256,
        protected_split_validation_sha256=report.protected_split_validation_sha256,
        passed_conditions=sum(item.status == "PASS" for item in report.conditions),
        failed_conditions=sum(item.status == "FAIL" for item in report.conditions),
        blocked_conditions=sum(item.status == "BLOCKED" for item in report.conditions),
        prediction_artifact_count=sum(len(item.prediction_files) for item in report.conditions),
        metric_record_count=len(report.metrics),
        paired_metric_record_count=len(report.paired_metrics),
        protected_foundation_unchanged=report.protected_foundation_unchanged,
        test_labels_opened_after_predictions=report.test_labels_opened_after_predictions,
        network_attempt_count=report.network_attempt_count,
        dataset_features_sha256=plan.dataset_features_sha256,
        dataset_source_sha256=plan.dataset_source_sha256,
        target_artifact_sha256=plan.target_artifact_sha256,
        split_sha256=plan.assignment_sha256,
        transformation_inventory_sha256=plan.transformation_inventory_sha256,
        environment=report.environment,
        conditions=[
            SmokeEvidenceCondition(
                condition_id=item.condition_id,
                cache_identity_sha256=item.cache_identity_sha256,
                model_id=item.model_id,
                view_id=item.view_id,
                device=item.device,
                status=item.status,
                cache_status=item.cache_status,
                prediction_hashes=[output.sha256 for output in item.prediction_files],
                failure_category=item.failure_category,
                resource=item.resource,
            )
            for item in report.conditions
        ],
        metrics=report.metrics,
        paired_metrics=report.paired_metrics,
        acceptance_gates=acceptance_gates,
    )


def write_json(path: str | Path, value: Any) -> str:
    destination = Path(path)
    atomic_write_json(
        destination, value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    )
    return sha256_file(destination)


def validate_smoke_report(root: str | Path, plan: SmokePlan, report_path: str | Path):
    from .contracts import (
        ProtectedFoundationHashComparison,
        ProtectedSplitValidation,
        ResumeVerification,
    )
    from .execution import _scheduler_plan

    project = Path(root).resolve()
    report_file = Path(report_path)
    if not report_file.is_absolute():
        report_file = project / report_file
    value = json.loads(report_file.read_text(encoding="utf-8"))
    report = SmokeRunReport.model_validate(value)
    if report.plan_sha256 != plan.plan_sha256 or report.source_commit != plan.source_commit:
        raise ValueError("run report does not match the frozen plan/source identity")
    if report.runtime_estimate != estimate_smoke_runtime(project, plan):
        raise ValueError("run report runtime estimate differs from accepted probe evidence")
    if [item.condition_id for item in report.conditions] != [
        item.condition_id for item in plan.conditions
    ]:
        raise ValueError("run report condition identities differ from the plan")
    scheduler_plan, identities = _scheduler_plan(plan)
    if report.scheduler_plan_sha256 != scheduler_plan.plan_hash:
        raise ValueError("run report scheduler plan hash differs from the recomputed plan")
    for run, condition in zip(report.conditions, plan.conditions, strict=True):
        if run.cache_identity_sha256 != identities[condition.condition_id].cache_key:
            raise ValueError("run report cache identity differs from the frozen condition")
    prediction_count = validate_prediction_tables(project, plan, report)
    metric_count, paired_count = validate_metric_tables(project, report)
    if report.labels_evaluated:
        if prediction_count != 20 or metric_count != 20 or paired_count != 10:
            raise ValueError("label evaluation is recorded without a complete prediction matrix")
    elif metric_count or paired_count:
        raise ValueError("metrics exist although final-boundary label evaluation is not recorded")
    if not report.test_labels_opened_after_predictions:
        raise ValueError("label-access ordering is not attested")

    foundation_comparison_path = (
        project
        / "results/smoke/runtime"
        / f"protected_foundation_hash_comparison_{report.run_mode}.json"
    )
    comparison: ProtectedFoundationHashComparison | None = None
    if foundation_comparison_path.is_file():
        comparison = ProtectedFoundationHashComparison.model_validate(
            json.loads(foundation_comparison_path.read_text(encoding="utf-8"))
        )
        if comparison.plan_sha256 != plan.plan_sha256:
            raise ValueError("protected foundation comparison belongs to another smoke plan")
        if report.protected_foundation_comparison_sha256 != sha256_file(foundation_comparison_path):
            raise ValueError("run report protected foundation checksum is invalid")
        if report.protected_foundation_unchanged != (comparison.status == "PASS"):
            raise ValueError("run report protected foundation status differs from its comparison")
    elif report.protected_foundation_comparison_sha256 is not None:
        raise ValueError("run report references a missing protected foundation comparison")

    split_validation_path = project / "results/smoke/runtime/grouped_split_validation.json"
    split_validation: ProtectedSplitValidation | None = None
    if split_validation_path.is_file():
        split_validation = ProtectedSplitValidation.model_validate(
            json.loads(split_validation_path.read_text(encoding="utf-8"))
        )
        if report.protected_split_validation_sha256 != sha256_file(split_validation_path):
            raise ValueError("run report protected split-validation checksum is invalid")
        if split_validation.assignment_sha256 != plan.assignment_sha256:
            raise ValueError("protected split validation belongs to another assignment")
    elif report.protected_split_validation_sha256 is not None:
        raise ValueError("run report references a missing protected split-validation record")

    resume_verified = False
    if report.run_mode == "resume" and report.resume_verification_sha256 is not None:
        verification_path = project / "results/smoke/runtime/resume_verification.json"
        if (
            not verification_path.is_file()
            or sha256_file(verification_path) != report.resume_verification_sha256
        ):
            raise ValueError("resume verification artifact is absent or checksum-invalid")
        verification = ResumeVerification.model_validate(
            json.loads(verification_path.read_text(encoding="utf-8"))
        )
        if verification.plan_sha256 != plan.plan_sha256:
            raise ValueError("resume verification belongs to another plan")
        if verification.cold_run_report_sha256 != report.cold_run_report_sha256:
            raise ValueError("resume verification references another cold report")
        if (
            sha256_canonical_json(verification.before_hashes)
            != report.resume_prediction_hashes_before_sha256
            or sha256_canonical_json(verification.after_hashes)
            != report.resume_prediction_hashes_after_sha256
        ):
            raise ValueError("resume hash summaries differ from the verification artifact")
        cold_path = project / "results/smoke/runs" / f"{plan.plan_sha256}_cold.json"
        if not cold_path.is_file() or sha256_file(cold_path) != verification.cold_run_report_sha256:
            raise ValueError("resume verification does not identify the cold-run report")
        cold_report = SmokeRunReport.model_validate(
            json.loads(cold_path.read_text(encoding="utf-8"))
        )
        if cold_report.run_mode != "cold" or cold_report.plan_sha256 != plan.plan_sha256:
            raise ValueError("resume run has no matching cold-run source")
        cold_duration = (
            datetime.fromisoformat(cold_report.ended_at)
            - datetime.fromisoformat(cold_report.started_at)
        ).total_seconds()
        if report.cold_run_wall_seconds != cold_duration:
            raise ValueError("resume report cold-run duration differs from the cold report")
        if [item.status for item in cold_report.conditions] != [
            item.status for item in report.conditions
        ]:
            raise ValueError("resume changed one or more model-condition outcomes")
        if (
            verification.executed_conditions != report.executed_conditions
            or verification.validated_cache_hits != report.validated_cache_hits
            or verification.failed_conditions != report.failed_conditions
            or verification.blocked_conditions != report.blocked_conditions
            or verification.network_attempt_count != report.network_attempt_count
            or verification.duplicate_prediction_artifacts != report.duplicate_prediction_artifacts
            or verification.result_rewrites != report.result_rewrites
            or verification.prediction_hashes_unchanged != report.resume_prediction_hashes_unchanged
            or verification.prediction_mtimes_unchanged != report.resume_prediction_mtimes_unchanged
        ):
            raise ValueError("resume report differs from its strict verification record")
        resume_verified = verification.status == "PASS"

    split_status: GateResult = (
        "PASS"
        if split_validation is not None and split_validation.status == "PASS"
        else "NOT_VERIFIED"
    )
    network_status: GateResult = "PASS" if report.network_attempt_count == 0 else "FAIL"
    smoke_status: GateResult = "PASS" if report.status == "PASS_PENDING_REVIEW" else "FAIL"
    gates = [
        SmokeGateRecord(
            gate_id="S01",
            result="PASS",
            evidence_sha256=sha256_file(report_file),
            detail="strict plan, report, scheduler identity, and ten-condition order validated",
        ),
        SmokeGateRecord(
            gate_id="S02",
            result="PASS",
            evidence_sha256=plan.plan_sha256,
            detail="all ten conditions are accounted for with canonical cache identities",
        ),
        SmokeGateRecord(
            gate_id="S03",
            result="PASS",
            evidence_sha256=sha256_file(report_file),
            detail=(
                f"{prediction_count} prediction artifacts passed independent lineage "
                "and probability validation"
            ),
        ),
        SmokeGateRecord(
            gate_id="S04",
            result="PASS",
            evidence_sha256=report.metric_table_sha256 or plan.plan_sha256,
            detail=(
                f"{metric_count} metric and {paired_count} paired records match "
                "strict report values"
            ),
        ),
        SmokeGateRecord(
            gate_id="S05",
            result=("PASS" if comparison is not None and comparison.status == "PASS" else "FAIL"),
            evidence_sha256=report.protected_foundation_comparison_sha256 or plan.plan_sha256,
            detail=(
                "Phase 01 hashes independently validated"
                if comparison is not None and comparison.status == "PASS"
                else "Phase 01 comparison is absent or records changed protected artifacts"
            ),
        ),
        SmokeGateRecord(
            gate_id="S06",
            result=split_status,
            evidence_sha256=(report.protected_split_validation_sha256 or report.plan_sha256),
            detail=(
                "grouped split and duplicate-leakage baseline revalidated"
                if split_status == "PASS"
                else (
                    split_validation.failure_reason or "split validation did not pass"
                    if split_validation is not None
                    else "Phase 01 split validation record is absent"
                )
            ),
        ),
        SmokeGateRecord(
            gate_id="S07",
            result=network_status,
            evidence_sha256=report.plan_sha256,
            detail=(
                "network access remained unused"
                if network_status == "PASS"
                else "one or more prohibited network attempts were recorded"
            ),
        ),
        SmokeGateRecord(
            gate_id="S08",
            result=("PASS" if report.test_labels_opened_after_predictions else "FAIL"),
            evidence_sha256=report.plan_sha256,
            detail="test-label access occurred only after model predictions completed",
        ),
        SmokeGateRecord(
            gate_id="S09",
            result=("PASS" if report.run_mode == "cold" or resume_verified else "FAIL"),
            evidence_sha256=report.resume_verification_sha256 or report.plan_sha256,
            detail=(
                "cold/resume behavior matches the recorded execution mode"
                if report.run_mode == "cold" or resume_verified
                else "resume did not satisfy its no-work, no-network, no-rewrite acceptance"
            ),
        ),
        SmokeGateRecord(
            gate_id="S10",
            result=smoke_status,
            evidence_sha256=sha256_file(report_file),
            detail=f"overall smoke result: {report.status}",
        ),
    ]
    validation = SmokeValidationReport(
        status="PASS" if all(gate.result == "PASS" for gate in gates) else "FAIL",
        plan_sha256=plan.plan_sha256,
        run_report_sha256=sha256_file(report_file),
        validated_prediction_count=prediction_count,
        validated_metric_record_count=metric_count,
        validated_paired_metric_count=paired_count,
        gates=gates,
    )
    return report, validation


def render_review(
    *,
    plan: SmokePlan,
    report: SmokeRunReport,
    inventory: SmokeEvidenceInventory,
    validation: SmokeValidationReport,
    foundation_comparison: Any,
    split_validation: Any,
    commands: list[str],
) -> str:
    """Render a sanitized, row-free tracked handoff from validated evidence."""

    def table_row(values: list[Any]) -> str:
        return "| " + " | ".join(str(value) for value in values) + " |"

    lines = [
        "# SchemaGuard Smoke Experiment Review",
        "",
        f"**Status:** `{report.status}`",
        "",
        "`completed_stage`: `smoke_experiment`  ",
        "`next_stage`: `independent_smoke_review`  ",
        f"`source_commit`: `{report.source_commit}`  ",
        f"`plan_sha256`: `{plan.plan_sha256}`  ",
        f"`run_report_sha256`: `{inventory.run_report_sha256}`  ",
        f"`runtime_probe_inventory_sha256`: `{report.runtime_estimate.source_inventory_sha256}`  ",
        f"`validation_status`: `{validation.status}`",
        "",
        (
            "The run is a compatibility/consistency smoke only. It does not establish a scientific "
            "effect or authorize the pilot, SCNF, COSA, or the main experiment."
        ),
        "",
        "## Environment",
        "",
        f"- OS: {report.environment.os_platform}",
        (
            f"- Python: {report.environment.python_version}; Conda environment: "
            f"{report.environment.conda_environment}"
        ),
        (
            f"- CPU: {report.environment.cpu_identifier}; physical/logical cores: "
            f"{report.environment.cpu_physical_cores}/{report.environment.cpu_logical_cores}"
        ),
        (
            f"- RAM total/available at capture: {report.environment.ram_total_mib}/"
            f"{report.environment.ram_available_mib} MiB"
        ),
        (
            f"- GPU: {report.environment.gpu_name}; total/free: "
            f"{report.environment.gpu_total_mib}/{report.environment.gpu_free_mib} MiB; "
            f"driver: {report.environment.gpu_driver}"
        ),
        (
            f"- CUDA visible: {report.environment.cuda_visible}; PyTorch: "
            f"{report.environment.torch_version}; CUDA build: "
            f"{report.environment.torch_cuda_build}"
        ),
        (
            "- Runtime package versions: `"
            f"{json.dumps(report.environment.runtime_package_versions, sort_keys=True)}`"
        ),
        f"- Telemetry warnings: {report.environment.telemetry_warnings or 'None'}",
        "",
        "## Runtime estimate and observation",
        "",
        (
            f"- Estimated cold-run total: {report.runtime_estimate.estimated_total_seconds:.1f}s; "
            f"CPU critical path: {report.runtime_estimate.expected_cpu_seconds:.1f}s; "
            f"sequential GPU work: {report.runtime_estimate.expected_gpu_seconds:.1f}s."
        ),
        (
            f"- Observed cold-run wall time: {report.cold_run_wall_seconds:.1f}s."
            if report.cold_run_wall_seconds is not None
            else "- Observed cold-run wall time: not recorded."
        ),
        (
            "- Estimate basis: maximum prior successful binary-numerical adapter runtime per "
            "model/device, deterministic two-worker CPU list schedule, 1.5 safety factor, "
            "and 60 seconds coordination allowance."
        ),
        "",
        "## Condition outcomes",
        "",
        "| Model | View | Device | Outcome | Cache | CPU/GPU resource evidence | Failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition in report.conditions:
        resource = condition.resource
        resource_text = (
            "not recorded"
            if resource is None
            else (
                f"{resource.wall_seconds:.3f}s; RAM {resource.peak_process_tree_rss_mib} MiB; "
                f"VRAM reserved {resource.peak_vram_reserved_mib} MiB; "
                f"telemetry {'complete' if resource.telemetry_complete else 'incomplete'}"
            )
        )
        failure = condition.failure_category or "None"
        if condition.traceback_relative_path:
            failure = f"{failure}; {condition.traceback_relative_path}"
        lines.append(
            table_row(
                [
                    condition.model_id,
                    condition.view_id,
                    condition.device,
                    condition.status,
                    condition.cache_status,
                    resource_text,
                    failure,
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Metric summary",
            "",
            (
                "Metrics are reported only when every condition completed and the Phase 01 "
                "label-access boundary was satisfied."
            ),
            "",
            table_row(
                [
                    "Model",
                    "View",
                    "Partition",
                    "Brier",
                    "Log loss",
                    "Accuracy",
                    "Balanced accuracy",
                    "AUROC",
                    "ECE",
                ]
            ),
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in report.metrics:
        lines.append(
            table_row(
                [
                    metric.model_id,
                    metric.view_id,
                    metric.partition,
                    f"{metric.brier_score:.6f}",
                    f"{metric.log_loss:.6f}",
                    f"{metric.accuracy:.6f}",
                    f"{metric.balanced_accuracy:.6f}",
                    metric.roc_auc if metric.roc_auc is not None else metric.roc_auc_status,
                    f"{metric.expected_calibration_error:.6f}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Paired V00/V01 consistency summary",
            "",
            table_row(
                [
                    "Model",
                    "Partition",
                    "Mean JS",
                    "Median JS",
                    "P90 JS",
                    "Max JS",
                    "Label flips",
                    "Max/mean |ΔP|",
                    "ΔBrier",
                    "ΔLog loss",
                ]
            ),
            table_row(["---"] + ["---:"] * 9),
        ]
    )
    for pair in report.paired_metrics:
        lines.append(
            table_row(
                [
                    pair.model_id,
                    pair.partition,
                    f"{pair.sii_mean_js_divergence:.8f}",
                    f"{pair.sii_median_js_divergence:.8f}",
                    f"{pair.sii_p90_js_divergence:.8f}",
                    f"{pair.sii_max_js_divergence:.8f}",
                    f"{pair.label_flip_rate:.6f}",
                    f"{pair.maximum_absolute_probability_difference:.6f}/"
                    f"{pair.mean_absolute_probability_difference:.6f}",
                    pair.metric_deltas_v01_minus_v00["brier_score"],
                    pair.metric_deltas_v01_minus_v00["log_loss"],
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Cache and resume",
            "",
            (
                f"- Mode: {report.run_mode}; executed: {report.executed_conditions}; "
                f"validated cache hits: {report.validated_cache_hits}."
            ),
            (
                f"- Network attempts: {report.network_attempt_count}; duplicate artifacts: "
                f"{report.duplicate_prediction_artifacts}; "
                f"output rewrites: {report.result_rewrites}."
            ),
            (
                f"- Resume hashes unchanged: {report.resume_prediction_hashes_unchanged}; "
                f"mtimes unchanged: {report.resume_prediction_mtimes_unchanged}."
            ),
            f"- Cold report SHA-256: {report.cold_run_report_sha256 or 'Not applicable'}.",
            (
                f"- Resume verification SHA-256: "
                f"{report.resume_verification_sha256 or 'Not applicable'}."
            ),
            "",
            "## Phase 01 preservation",
            "",
            (
                f"- Protected artifact comparison: `{foundation_comparison.status}` "
                f"({len(foundation_comparison.artifacts)} files); unchanged: "
                f"{sum(item.unchanged for item in foundation_comparison.artifacts)}."
            ),
            (
                f"- Comparison source snapshot SHA-256: "
                f"`{foundation_comparison.baseline_snapshot_sha256}`."
            ),
            (
                f"- Grouped split verification: `{split_validation.status}`; strategy: "
                f"`{split_validation.strategy}`; rows: {split_validation.row_count}; "
                f"partitions: `{json.dumps(split_validation.partition_counts, sort_keys=True)}`."
            ),
            (
                f"- Predictor groups: {split_validation.predictor_group_count}; duplicates: "
                f"{split_validation.duplicate_predictor_group_count}; conflicting targets: "
                f"{split_validation.conflicting_target_group_count}; crossing groups: "
                f"{split_validation.crossing_predictor_group_count}."
            ),
            (
                "- Deprecated row-stratified split selected: "
                f"{split_validation.deprecated_row_stratified_split_selected}."
            ),
            "",
            "## Acceptance checks",
            "",
            "| Gate | Result | Detail |",
            "|---|---|---|",
        ]
    )
    for gate in inventory.acceptance_gates:
        lines.append(f"| {gate.gate_id} | {gate.result} | {gate.detail} |")
    lines.extend(
        [
            "",
            "## Commands executed",
            "",
            "```text",
            *commands,
            "```",
            "",
            "## Generated local evidence",
            "",
            (
                "Prediction rows, model cache payloads, raw worker tracebacks, and resource "
                "samples remain in ignored local outputs. This tracked handoff contains no "
                "row-level predictions or credentials."
            ),
            "",
            "## Deviations",
            "",
            "None recorded."
            if report.status == "PASS_PENDING_REVIEW"
            else "See per-condition failure categories and Phase 01 validation state above.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "build_run_report",
    "capture_environment",
    "make_inventory",
    "render_review",
    "utc_now",
    "validate_smoke_report",
    "write_json",
]
