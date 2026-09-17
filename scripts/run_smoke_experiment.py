"""Run the authorized, offline ten-condition SchemaGuard smoke experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from schemaguard.cache.contracts import CacheSchedulerConfig  # noqa: E402
from schemaguard.cache.store import CacheStore  # noqa: E402
from schemaguard.experiments.contracts import (  # noqa: E402
    EvaluationSource,
    MetricRecord,
    PairedMetricRecord,
    ProtectedFoundationArtifactHash,
    ProtectedFoundationHashComparison,
    ProtectedSplitValidation,
    ResumeVerification,
    SmokePlan,
    SmokeRunStatus,
)
from schemaguard.experiments.evaluation import evaluate_smoke  # noqa: E402
from schemaguard.experiments.evidence import (  # noqa: E402
    build_run_report,
    capture_environment,
    estimate_smoke_runtime,
    utc_now,
    write_json,
)
from schemaguard.experiments.execution import _scheduler_plan, run_conditions  # noqa: E402
from schemaguard.experiments.planning import (  # noqa: E402
    build_smoke_plan,
    load_smoke_config,
    verify_runtime_sources_after_plan,
    write_plan,
)
from schemaguard.utils.hashing import sha256_canonical_json, sha256_file  # noqa: E402
from schemaguard.utils.io import atomic_write_text  # noqa: E402

REQUIRED_START_COMMIT = "acf5c7fa1315bd8e8727f926b32f488f2d05111f"
P12_PYTHON = Path(r"D:\Conda\P12\python.exe")
BASELINE_SNAPSHOT = Path("results/smoke/runtime/protected_foundation_baseline.json")
PREDICTION_DIRECTORY = Path("results/smoke/predictions")
METRIC_FILES = (
    Path("results/smoke/metrics/model_metrics.parquet"),
    Path("results/smoke/metrics/paired_view_metrics.parquet"),
)


def _git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _assert_start_state() -> str:
    commit = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    remote = _git("rev-parse", "origin/main")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_START_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "BLOCKED_UNEXPECTED_WORKTREE: required starting commit is not an ancestor"
        )
    if branch != "main" or remote != REQUIRED_START_COMMIT:
        raise RuntimeError(
            "BLOCKED_UNEXPECTED_WORKTREE: branch or origin/main no longer matches "
            "the authorized base"
        )

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    for entry in (part for part in status.split(b"\0") if part):
        if entry[:2] != b"??":
            raise RuntimeError("BLOCKED_UNEXPECTED_WORKTREE: tracked or staged changes are present")
        relative = Path(os.fsdecode(entry[3:]))
        # The known private reference document is intentionally only listed by Git status;
        # this check never opens, hashes, stages, moves, or modifies it.
        if relative.parent != Path(".") or relative.suffix.casefold() != ".docx":
            raise RuntimeError("BLOCKED_UNEXPECTED_WORKTREE: unexpected untracked file is present")
    return commit


def _assert_p12() -> None:
    if Path(sys.executable).resolve() != P12_PYTHON.resolve():
        raise RuntimeError(f"smoke must run with the existing interpreter {P12_PYTHON}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("smoke requires Python 3.12 in Conda environment P12")
    environment = os.environ.get("CONDA_DEFAULT_ENV")
    if environment != "P12" and Path(sys.prefix).name.casefold() != "p12":
        raise RuntimeError("current Python process is not running in the existing P12 environment")


def _load_baseline_snapshot() -> tuple[dict[str, dict[str, Any]], str]:
    path = ROOT / BASELINE_SNAPSHOT
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 13:
        raise RuntimeError(
            "BLOCKED_FOUNDATION_BASELINE: prior 13-artifact snapshot is absent or invalid"
        )
    for relative, record in artifacts.items():
        if (
            not isinstance(relative, str)
            or not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or not isinstance(record.get("size_bytes"), int)
        ):
            raise RuntimeError("BLOCKED_FOUNDATION_BASELINE: snapshot record is malformed")
    return artifacts, sha256_file(path)


def _capture_protected_foundation(
    baseline: dict[str, dict[str, Any]], *, include_target_bytes: bool
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for relative in sorted(baseline):
        if relative == "data/processed/openml/1464/targets.parquet" and not include_target_bytes:
            continue
        path = ROOT / Path(relative)
        if not path.is_file():
            raise RuntimeError(
                f"FAIL_FOUNDATION_MUTATION: protected artifact is missing: {relative}"
            )
        observed[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return observed


def _compare_protected_foundation(
    *,
    plan: SmokePlan,
    baseline: dict[str, dict[str, Any]],
    baseline_sha256: str,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    run_mode: str,
) -> tuple[ProtectedFoundationHashComparison, str]:
    if set(before) != set(baseline) or set(after) != set(baseline):
        raise RuntimeError("FAIL_FOUNDATION_MUTATION: protected artifact set changed")
    records = []
    for relative in sorted(baseline):
        if before[relative] != baseline[relative]:
            raise RuntimeError(
                f"FAIL_FOUNDATION_MUTATION: baseline mismatch before smoke: {relative}"
            )
        previous = before[relative]
        current = after[relative]
        records.append(
            ProtectedFoundationArtifactHash(
                path=relative,
                before_sha256=previous["sha256"],
                after_sha256=current["sha256"],
                before_size_bytes=previous["size_bytes"],
                after_size_bytes=current["size_bytes"],
                unchanged=(previous == current),
            )
        )
    comparison = ProtectedFoundationHashComparison(
        plan_sha256=plan.plan_sha256,
        baseline_snapshot_sha256=baseline_sha256,
        status="PASS" if all(item.unchanged for item in records) else "FAIL",
        artifacts=records,
    )
    output_path = (
        ROOT / "results/smoke/runtime" / f"protected_foundation_hash_comparison_{run_mode}.json"
    )
    digest = write_json(output_path, comparison)
    return comparison, digest


def _validate_protected_split(plan: SmokePlan) -> ProtectedSplitValidation:
    assignment_hash = plan.assignment_sha256
    try:
        from audit_data_foundation import group_audit

        from schemaguard.splits.contracts import SplitGenerationConfig
        from schemaguard.splits.validation import validate_split

        split_config = SplitGenerationConfig.model_validate(
            yaml.safe_load(
                (ROOT / "configs/runtime/split_generation.yaml").read_text(encoding="utf-8")
            )
        )
        if split_config.strategy != "stratified_group_5fold_v1":
            raise ValueError("deprecated row-stratified split was selected")
        result = validate_split(ROOT, split_config, 1464, 1729)
        import pandas as pd

        assignment_path = (
            ROOT / "data/splits/openml/1464/stratified_group_5fold_v1/seed_1729/assignments.parquet"
        )
        assignment = pd.read_parquet(assignment_path)
        features = pd.read_parquet(ROOT / "data/processed/openml/1464/features.parquet")
        targets = pd.read_parquet(ROOT / "data/processed/openml/1464/targets.parquet")
        details = group_audit(features, targets, assignment)
        value = ProtectedSplitValidation(
            status="PASS",
            strategy=split_config.strategy,
            assignment_sha256=assignment_hash,
            row_count=int(result["row_count"]),
            partition_counts=result["partition_counts"],
            class_counts_by_partition=details["class_counts_by_split"],
            predictor_group_count=details["total_predictor_groups"],
            duplicate_predictor_group_count=details["duplicate_predictor_groups"],
            conflicting_target_group_count=details["conflicting_target_groups"],
            crossing_predictor_group_count=details["predictor_duplicate_groups_crossing_splits"],
            deprecated_row_stratified_split_selected=False,
        )
        if sha256_file(assignment_path) != assignment_hash:
            raise ValueError("grouped split assignment hash changed after planning")
        return value
    except Exception as exc:
        return ProtectedSplitValidation(
            status="FAIL",
            strategy="stratified_group_5fold_v1",
            assignment_sha256=assignment_hash,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )


def _not_run_split_validation(plan: SmokePlan, reason: str) -> ProtectedSplitValidation:
    return ProtectedSplitValidation(
        status="NOT_RUN",
        strategy="stratified_group_5fold_v1",
        assignment_sha256=plan.assignment_sha256,
        failure_reason=reason,
    )


def _write_split_validation(value: ProtectedSplitValidation) -> str:
    return write_json(ROOT / "results/smoke/runtime/grouped_split_validation.json", value)


def _output_state() -> tuple[dict[str, str], dict[str, int]]:
    paths = sorted((ROOT / PREDICTION_DIRECTORY).glob("*.parquet"))
    paths.extend(ROOT / relative for relative in METRIC_FILES if (ROOT / relative).is_file())
    hashes: dict[str, str] = {}
    mtimes: dict[str, int] = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        hashes[relative] = sha256_file(path)
        mtimes[relative] = path.stat().st_mtime_ns
    return hashes, mtimes


def _duplicate_artifact_count() -> int:
    directory = ROOT / PREDICTION_DIRECTORY
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    content_hashes = [sha256_file(path) for path in files]
    return max(0, len(content_hashes) - len(set(content_hashes)))


def _ensure_new_cold_run(plan: SmokePlan, config_path: Path) -> None:
    config = load_smoke_config(config_path)
    scheduler_config = CacheSchedulerConfig.model_validate(
        yaml.safe_load((ROOT / config.scheduler_config).read_text(encoding="utf-8"))
    )
    scheduler_plan, identities = _scheduler_plan(plan)
    state_path = ROOT / scheduler_config.state_root / "runs" / f"{scheduler_plan.plan_hash}.json"
    if state_path.exists():
        raise RuntimeError(
            "BLOCKED_EXISTING_SMOKE_STATE: refusing to mislabel a repeated run as cold"
        )
    if any((ROOT / PREDICTION_DIRECTORY).glob("*.parquet")):
        raise RuntimeError(
            "BLOCKED_EXISTING_SMOKE_OUTPUTS: use independent review before rerunning"
        )
    if any((ROOT / "results/smoke/metrics").glob("*.parquet")):
        raise RuntimeError(
            "BLOCKED_EXISTING_SMOKE_OUTPUTS: existing metrics must not be overwritten"
        )
    cache = CacheStore(ROOT / scheduler_config.cache_root)
    for identity in identities.values():
        if cache.entry_path(identity.cache_key).exists():
            raise RuntimeError("BLOCKED_EXISTING_SMOKE_CACHE: refusing to claim a cold execution")


def _condition_status(
    runs: list[Any], foundation_hashes_ok: bool, split_status: str, network_attempt_count: int
) -> SmokeRunStatus:
    if not foundation_hashes_ok or split_status == "FAIL":
        return "REPAIR_REQUIRED"
    # An attempted connection violates the frozen offline protocol even when blocked.
    if network_attempt_count:
        return "REPAIR_REQUIRED"
    capability_failures = {
        "FAIL_CUDA_OOM",
        "FAIL_GPU_INFERENCE",
        "FAIL_MODEL_RUNTIME",
        "FAIL_RESOURCE_LIMIT",
        "FAIL_TIMEOUT",
    }
    if any(
        item.status == "FAIL" and item.failure_category not in capability_failures
        for item in runs
    ):
        return "REPAIR_REQUIRED"
    if any(item.status == "BLOCKED" for item in runs):
        return "BLOCKED"
    if any(item.status == "FAIL" for item in runs):
        return "PARTIAL_VALID"
    if split_status != "PASS":
        return "REPAIR_REQUIRED"
    return "PASS_PENDING_REVIEW"


def _write_runner_exception(plan: SmokePlan | None, exc: BaseException) -> Path:
    destination = ROOT / "results/smoke/runtime/failures"
    destination.mkdir(parents=True, exist_ok=True)
    suffix = plan.plan_sha256 if plan is not None else "preflight"
    path = destination / f"{suffix}.txt"
    atomic_write_text(path, "".join(traceback.format_exception(exc)))
    return path


def execute(config_relative: str) -> int:
    plan: SmokePlan | None = None
    try:
        source_commit = _assert_start_state()
        _assert_p12()
        baseline, baseline_sha = _load_baseline_snapshot()
        before_without_targets = _capture_protected_foundation(baseline, include_target_bytes=False)
        for relative, value in before_without_targets.items():
            if value != baseline[relative]:
                raise RuntimeError(
                    f"FAIL_FOUNDATION_MUTATION: protected baseline mismatch: {relative}"
                )

        config_path = Path(config_relative)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = load_smoke_config(config_path)
        plan = build_smoke_plan(ROOT, config_path, source_commit_override=source_commit)
        if plan.source_commit != source_commit:
            raise RuntimeError("smoke plan does not identify the committed implementation")
        plan_path = write_plan(ROOT, plan)
        runtime_estimate = estimate_smoke_runtime(ROOT, plan)
        runtime_estimate_sha = write_json(
            ROOT / "results/smoke/runtime/runtime_estimate.json", runtime_estimate
        )

        # The plan is durable before target-artifact bytes are hashed. No test labels are decoded
        # during preflight, planning, configuration, or resource decisions.
        verify_runtime_sources_after_plan(ROOT, plan)
        before = _capture_protected_foundation(baseline, include_target_bytes=True)
        for relative, artifact_state in before.items():
            if artifact_state != baseline[relative]:
                raise RuntimeError(
                    f"FAIL_FOUNDATION_MUTATION: protected baseline mismatch: {relative}"
                )
        write_json(
            ROOT / "results/smoke/runtime/protected_foundation_hashes_before.json",
            {"plan_sha256": plan.plan_sha256, "artifacts": before},
        )

        for name, env_value in {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }.items():
            os.environ[name] = env_value

        _ensure_new_cold_run(plan, config_path)
        environment = capture_environment()
        cold_started = utc_now()
        cold_runs, cold_result, cold_network_attempts = run_conditions(
            ROOT, plan, config, resume=False
        )
        protected_split_validation = (
            _validate_protected_split(plan)
            if all(item.status == "PASS" for item in cold_runs)
            else _not_run_split_validation(
                plan, "not run because one or more model conditions failed before final evaluation"
            )
        )
        split_validation_sha = _write_split_validation(protected_split_validation)
        metrics: list[MetricRecord] = []
        paired_metrics: list[PairedMetricRecord] = []
        metric_sha = None
        paired_metric_sha = None
        labels_evaluated = False
        evaluation_source: EvaluationSource = "not_performed"
        if (
            all(item.status == "PASS" for item in cold_runs)
            and protected_split_validation.status == "PASS"
        ):
            metrics, paired_metrics, metric_sha, paired_metric_sha = evaluate_smoke(
                ROOT, plan, cold_runs
            )
            labels_evaluated = True
            evaluation_source = "computed_after_predictions"

        cold_after = _capture_protected_foundation(baseline, include_target_bytes=True)
        write_json(
            ROOT / "results/smoke/runtime/protected_foundation_hashes_after_cold.json",
            {"plan_sha256": plan.plan_sha256, "artifacts": cold_after},
        )
        cold_comparison, cold_comparison_sha = _compare_protected_foundation(
            plan=plan,
            baseline=baseline,
            baseline_sha256=baseline_sha,
            before=before,
            after=cold_after,
            run_mode="cold",
        )
        cold_status = _condition_status(
            cold_runs,
            cold_comparison.status == "PASS",
            protected_split_validation.status,
            cold_network_attempts,
        )
        cold_ended = utc_now()
        cold_run_wall_seconds = (
            datetime.fromisoformat(cold_ended) - datetime.fromisoformat(cold_started)
        ).total_seconds()
        cold_report = build_run_report(
            plan=plan,
            scheduler_result=cold_result,
            conditions=cold_runs,
            environment=environment,
            runtime_estimate=runtime_estimate,
            started_at=cold_started,
            ended_at=cold_ended,
            status=cold_status,
            metrics=metrics,
            paired_metrics=paired_metrics,
            metric_sha256=metric_sha,
            paired_metric_sha256=paired_metric_sha,
            protected_foundation_unchanged=cold_comparison.status == "PASS",
            protected_foundation_comparison_sha256=cold_comparison_sha,
            protected_split_validation_sha256=split_validation_sha,
            labels_evaluated=labels_evaluated,
            evaluation_source=evaluation_source,
            duplicate_prediction_artifacts=_duplicate_artifact_count(),
            network_attempt_count=cold_network_attempts,
        )
        cold_report_path = ROOT / "results/smoke/runs" / f"{plan.plan_sha256}_cold.json"
        cold_report_sha = write_json(cold_report_path, cold_report)

        before_resume_hashes, before_resume_mtimes = _output_state()
        resume_started = utc_now()
        resume_runs, resume_result, resume_network_attempts = run_conditions(
            ROOT, plan, config, resume=True
        )
        after_resume_hashes, after_resume_mtimes = _output_state()
        hashes_unchanged = before_resume_hashes == after_resume_hashes
        mtimes_unchanged = before_resume_mtimes == after_resume_mtimes
        rewrites = sum(
            before_resume_mtimes.get(path) != after_resume_mtimes.get(path)
            or before_resume_hashes.get(path) != after_resume_hashes.get(path)
            for path in set(before_resume_hashes) | set(after_resume_hashes)
        )
        duplicate_count = _duplicate_artifact_count()
        resume_failed_count = sum(item.status == "FAIL" for item in resume_runs)
        resume_blocked_count = sum(item.status == "BLOCKED" for item in resume_runs)
        verification_status: Literal["PASS", "FAIL"] = (
            "PASS"
            if (
                resume_result.manifest.executed == 0
                and resume_result.manifest.validated_cache_hits == 10
                and resume_failed_count == 0
                and resume_blocked_count == 0
                and resume_network_attempts == 0
                and duplicate_count == 0
                and rewrites == 0
                and hashes_unchanged
                and mtimes_unchanged
            )
            else "FAIL"
        )
        verification = ResumeVerification(
            status=verification_status,
            plan_sha256=plan.plan_sha256,
            before_hashes=before_resume_hashes,
            after_hashes=after_resume_hashes,
            before_mtime_ns=before_resume_mtimes,
            after_mtime_ns=after_resume_mtimes,
            cold_run_report_sha256=cold_report_sha,
            executed_conditions=resume_result.manifest.executed,
            validated_cache_hits=resume_result.manifest.validated_cache_hits,
            failed_conditions=resume_failed_count,
            blocked_conditions=resume_blocked_count,
            network_attempt_count=resume_network_attempts,
            duplicate_prediction_artifacts=duplicate_count,
            result_rewrites=rewrites,
            prediction_hashes_unchanged=hashes_unchanged,
            prediction_mtimes_unchanged=mtimes_unchanged,
        )
        verification_path = ROOT / "results/smoke/runtime/resume_verification.json"
        verification_sha = write_json(verification_path, verification)

        final_after = _capture_protected_foundation(baseline, include_target_bytes=True)
        write_json(
            ROOT / "results/smoke/runtime/protected_foundation_hashes_after_resume.json",
            {"plan_sha256": plan.plan_sha256, "artifacts": final_after},
        )
        final_comparison, final_comparison_sha = _compare_protected_foundation(
            plan=plan,
            baseline=baseline,
            baseline_sha256=baseline_sha,
            before=before,
            after=final_after,
            run_mode="resume",
        )
        final_status: SmokeRunStatus
        if [item.status for item in cold_runs] != [item.status for item in resume_runs]:
            final_status = "REPAIR_REQUIRED"
        elif final_comparison.status != "PASS":
            final_status = "REPAIR_REQUIRED"
        elif protected_split_validation.status == "FAIL":
            final_status = "REPAIR_REQUIRED"
        elif cold_status == "BLOCKED":
            final_status = "BLOCKED"
        elif cold_status != "PASS_PENDING_REVIEW":
            final_status = cold_status
        elif verification.status != "PASS":
            final_status = "REPAIR_REQUIRED"
        else:
            final_status = "PASS_PENDING_REVIEW"

        final_report = build_run_report(
            plan=plan,
            scheduler_result=resume_result,
            conditions=resume_runs,
            environment=environment,
            runtime_estimate=runtime_estimate,
            started_at=resume_started,
            ended_at=utc_now(),
            status=final_status,
            metrics=metrics,
            paired_metrics=paired_metrics,
            metric_sha256=metric_sha,
            paired_metric_sha256=paired_metric_sha,
            protected_foundation_unchanged=final_comparison.status == "PASS",
            protected_foundation_comparison_sha256=final_comparison_sha,
            protected_split_validation_sha256=split_validation_sha,
            labels_evaluated=labels_evaluated,
            evaluation_source=("reused_cold_run" if labels_evaluated else "not_performed"),
            duplicate_prediction_artifacts=duplicate_count,
            result_rewrites=rewrites,
            cold_run_report_sha256=cold_report_sha,
            resume_prediction_hashes_before_sha256=sha256_canonical_json(before_resume_hashes),
            resume_prediction_hashes_after_sha256=sha256_canonical_json(after_resume_hashes),
            resume_verification_sha256=verification_sha,
            resume_prediction_hashes_unchanged=hashes_unchanged,
            resume_prediction_mtimes_unchanged=mtimes_unchanged,
            network_attempt_count=resume_network_attempts,
            cold_run_wall_seconds=cold_run_wall_seconds,
        )
        final_report_path = ROOT / "results/smoke/runs" / f"{plan.plan_sha256}_resume.json"
        final_report_sha = write_json(final_report_path, final_report)
        write_json(
            ROOT / "results/smoke/runtime/runner_summary.json",
            {
                "schema_version": 1,
                "source_commit": source_commit,
                "plan_sha256": plan.plan_sha256,
                "plan_path": plan_path.relative_to(ROOT).as_posix(),
                "cold_report_path": cold_report_path.relative_to(ROOT).as_posix(),
                "cold_report_sha256": cold_report_sha,
                "resume_report_path": final_report_path.relative_to(ROOT).as_posix(),
                "resume_report_sha256": final_report_sha,
                "runtime_estimate_sha256": runtime_estimate_sha,
                "status": final_report.status,
            },
        )
        print(
            json.dumps(
                {
                    "status": final_report.status,
                    "plan_sha256": plan.plan_sha256,
                    "cold_report": cold_report_path.relative_to(ROOT).as_posix(),
                    "resume_report": final_report_path.relative_to(ROOT).as_posix(),
                    "executed": resume_result.manifest.executed,
                    "cache_hits": resume_result.manifest.validated_cache_hits,
                    "network_attempts": resume_network_attempts,
                    "estimated_runtime_seconds": runtime_estimate.estimated_total_seconds,
                    "observed_cold_runtime_seconds": cold_run_wall_seconds,
                    "protected_foundation_unchanged": final_comparison.status == "PASS",
                },
                sort_keys=True,
            )
        )
        return 0 if final_report.status == "PASS_PENDING_REVIEW" else 2
    except Exception as exc:
        failure_path = _write_runner_exception(plan, exc)
        print(
            json.dumps(
                {
                    "status": "REPAIR_REQUIRED",
                    "error_category": type(exc).__name__,
                    "traceback_path": failure_path.relative_to(ROOT).as_posix(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/runtime/smoke_experiment.yaml",
        help="strict smoke-experiment YAML config",
    )
    args = parser.parse_args()
    return execute(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
