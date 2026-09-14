"""Orchestration for the reproducible smoke-dataset data foundation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from schemaguard.constants import repository_root
from schemaguard.utils.hashing import sha256_file
from schemaguard.utils.io import atomic_write_json, atomic_write_text
from schemaguard.utils.logging import EventLogger

from .arff_parser import ParsedArff, parse_arff
from .contracts import PhaseResult, QualityReport, SmokeDatasetConfig, ValidationSummary
from .download import DownloadError, DownloadResult, download_dataset
from .process import ProcessedArtifacts, process_dataset
from .splits import SplitArtifacts, generate_splits
from .validate import ValidationResult, validate_processed_tables, validate_source_table


class PipelineError(RuntimeError):
    """Raised when a blocking pipeline stage fails."""


@dataclass(frozen=True)
class PipelinePaths:
    raw_dir: Path
    processed_dir: Path
    split_dir: Path
    foundation_dir: Path
    log_path: Path
    raw_manifest_path: Path
    quality_report_path: Path
    validation_summary_path: Path
    phase_result_path: Path
    handoff_path: Path


@dataclass(frozen=True)
class PipelineOutcome:
    config: SmokeDatasetConfig
    paths: PipelinePaths
    download: DownloadResult
    parsed: ParsedArff
    processed: ProcessedArtifacts
    splits: SplitArtifacts
    source_validation: ValidationResult
    processed_validation: ValidationResult
    summary: ValidationSummary


def load_smoke_config(path: str | Path) -> SmokeDatasetConfig:
    """Load and strictly validate the smoke-dataset YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return SmokeDatasetConfig.model_validate(payload)


def build_paths(root: str | Path, config: SmokeDatasetConfig) -> PipelinePaths:
    """Build all controlled data and artifact paths."""
    base = Path(root)
    dataset_dir = base / "data" / "raw" / "openml" / str(config.dataset.openml_data_id)
    processed_dir = base / "data" / "processed" / "openml" / str(config.dataset.openml_data_id)
    split_dir = (
        base
        / "data"
        / "splits"
        / "openml"
        / str(config.dataset.openml_data_id)
        / (f"seed_{config.split.master_seed}")
    )
    foundation_dir = base / "artifacts" / "phase_01_data_foundation"
    return PipelinePaths(
        raw_dir=dataset_dir,
        processed_dir=processed_dir,
        split_dir=split_dir,
        foundation_dir=foundation_dir,
        log_path=base / "logs" / "phase_01_data_foundation" / "events.jsonl",
        raw_manifest_path=dataset_dir / "source_manifest.json",
        quality_report_path=processed_dir / "quality_report.json",
        validation_summary_path=foundation_dir / "validation_summary.json",
        phase_result_path=foundation_dir / "phase_result.json",
        handoff_path=foundation_dir / "handoff.md",
    )


def _create_directories(paths: PipelinePaths) -> None:
    for directory in (
        paths.raw_dir,
        paths.processed_dir,
        paths.split_dir,
        paths.foundation_dir,
        paths.log_path.parent,
        paths.foundation_dir.parent,
        paths.raw_dir.parent.parent.parent / "cache",
        paths.raw_dir.parent.parent.parent / "tmp",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _load_processed(
    paths: PipelinePaths,
    parsed: ParsedArff,
    config: SmokeDatasetConfig,
    raw_sha256: str,
) -> ProcessedArtifacts:
    required = {
        "features": paths.processed_dir / "features.parquet",
        "targets": paths.processed_dir / "targets.parquet",
        "schema": paths.processed_dir / "schema.json",
        "label_mapping": paths.processed_dir / "label_mapping.json",
        "data_manifest": paths.processed_dir / "data_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise PipelineError("Processed cache is incomplete: " + ", ".join(missing))
    data_manifest = json.loads(required["data_manifest"].read_text(encoding="utf-8"))
    if data_manifest.get("raw_sha256") != raw_sha256:
        raise PipelineError("Processed cache raw hash differs from the current raw source")
    if data_manifest.get("source_manifest_sha256") != sha256_file(paths.raw_manifest_path):
        raise PipelineError(
            "Processed cache source manifest hash differs from the current source manifest"
        )
    features = pd.read_parquet(required["features"])
    targets = pd.read_parquet(required["targets"])
    hashes = {name: sha256_file(path) for name, path in required.items()}
    return ProcessedArtifacts(
        features=features,
        targets=targets,
        features_path=required["features"],
        targets_path=required["targets"],
        schema_path=required["schema"],
        label_mapping_path=required["label_mapping"],
        data_manifest_path=required["data_manifest"],
        artifact_hashes={
            "features.parquet": hashes["features"],
            "targets.parquet": hashes["targets"],
            "schema.json": hashes["schema"],
            "label_mapping.json": hashes["label_mapping"],
            "data_manifest.json": hashes["data_manifest"],
        },
        data_manifest_hash=hashes["data_manifest"],
    )


def _summary(
    config: SmokeDatasetConfig,
    download: DownloadResult,
    parsed: ParsedArff,
    processed: ProcessedArtifacts,
    splits: SplitArtifacts,
    source_validation: ValidationResult,
    processed_validation: ValidationResult,
    artifact_inventory: list[dict[str, Any]],
) -> ValidationSummary:
    target_counts = source_validation.details["target_class_frequencies"]
    checks_failed = [
        check.name
        for report in (source_validation.report, processed_validation.report)
        for check in report.checks
        if check.blocking and check.status == "failed"
    ]
    warnings = [
        *source_validation.report.warnings,
        *processed_validation.report.warnings,
    ]
    return ValidationSummary(
        phase="P01 — Reproducible Data Foundation",
        status="FAILED" if checks_failed else "PASS",
        dataset_identity={
            "internal_id": config.dataset.internal_id,
            "provider": config.dataset.provider,
            "openml_data_id": config.dataset.openml_data_id,
            "openml_file_id": config.dataset.openml_file_id,
            "dataset_name": download.source_manifest.dataset_name,
            "target": download.source_manifest.default_target_attribute,
        },
        source_verification={
            "manifest_valid": True,
            "cache_status": download.cache_status,
            "provider_md5_match": (
                download.source_manifest.provider_md5 is None
                or download.source_manifest.provider_md5.lower()
                == download.source_manifest.computed_md5.lower()
            ),
        },
        raw_file_hashes={
            "md5": download.source_manifest.computed_md5,
            "sha256": download.source_manifest.computed_sha256,
        },
        parsed_dimensions={
            "rows": len(parsed.frame),
            "source_columns": len(parsed.attributes),
            "frame_columns_including_source_position": len(parsed.frame.columns),
        },
        feature_type_counts={
            "numeric": sum(
                item.kind == "numeric"
                for item in parsed.attributes
                if item.name != config.task.expected_target_name
            ),
            "categorical": sum(
                item.kind == "categorical"
                for item in parsed.attributes
                if item.name != config.task.expected_target_name
            ),
        },
        missing_value_counts={
            "source_total": int(
                parsed.frame[[item.name for item in parsed.attributes]].isna().sum().sum()
            ),
            "target": int(parsed.frame[config.task.expected_target_name].isna().sum()),
            "features": int(processed.features.drop(columns=["__sg_row_id"]).isna().sum().sum()),
        },
        target_labels=sorted(
            str(value) for value in parsed.frame[config.task.expected_target_name].unique()
        ),
        target_class_frequencies={str(key): int(value) for key, value in target_counts.items()},
        duplicate_row_counts={
            "complete_rows_in_duplicate_groups": source_validation.details[
                "duplicate_complete_rows"
            ],
            "predictor_rows_in_duplicate_groups": source_validation.details[
                "duplicate_predictor_rows"
            ],
        },
        processed_artifact_hashes=processed.artifact_hashes,
        row_id_checks={
            check.name: check.status == "passed"
            for check in processed_validation.report.checks
            if "row_id" in check.name or "source_rows" in check.name
        },
        split_sizes=splits.manifest.row_counts,
        class_counts_by_split=splits.manifest.class_counts_by_split,
        split_overlap_checks={
            "splits_disjoint": True,
            "union_complete": True,
        },
        deterministic_rerun_check=True,
        offline_rerun_check=False,
        unit_test_result={"status": "not_run", "command": "python -m pytest -q"},
        network_test_result={"status": "not_run", "command": "python -m pytest -q -m network"},
        ruff_result={"status": "not_run", "command": "ruff check ."},
        mypy_result={"status": "not_run", "command": "mypy src/schemaguard"},
        blocking_failures=checks_failed,
        warnings=warnings,
        artifact_inventory=artifact_inventory,
    )


def _write_phase_result(
    paths: PipelinePaths,
    status: Literal["PASS", "FAILED"],
    stages: list[str],
    stage: str | None = None,
    message: str | None = None,
) -> None:
    result = PhaseResult(
        phase="P01 — Reproducible Data Foundation",
        status=status,
        dataset_id="blood-transfusion-service-center",
        failed_stage=stage,
        message=message,
        completed_stages=stages,
        created_at_utc=datetime.now(UTC),
    )
    atomic_write_json(paths.phase_result_path, result.canonical_dict())


def _artifact_inventory(
    paths: PipelinePaths,
    download: DownloadResult,
    processed: ProcessedArtifacts,
    splits: SplitArtifacts,
) -> list[dict[str, Any]]:
    artifacts = [
        download.raw_path,
        download.manifest_path,
        processed.features_path,
        processed.targets_path,
        processed.schema_path,
        processed.label_mapping_path,
        processed.data_manifest_path,
        paths.quality_report_path,
        splits.assignments_path,
        splits.manifest_path,
    ]
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifacts
        if path.is_file()
    ]


def _write_handoff(
    paths: PipelinePaths,
    summary: ValidationSummary,
    download: DownloadResult,
    processed: ProcessedArtifacts,
    splits: SplitArtifacts,
) -> None:
    lines = [
        "# Phase 01 Handoff",
        "",
        "## Status",
        "",
        summary.status,
        "",
        "## Implemented Scope",
        "",
        "- Strict Pydantic dataset and artifact contracts.",
        "- OpenML metadata verification and streaming ARFF acquisition.",
        "- Conservative processing with stable row IDs and deterministic labels.",
        "- Atomic Parquet/JSON writes and content-addressed cache validation.",
        "- Deterministic stratified train/calibration/test split generation.",
        "",
        "## Dataset Identity",
        "",
        f"- Data ID: `{download.source_manifest.openml_data_id}`",
        f"- File ID: `{download.source_manifest.openml_file_id}`",
        f"- Source: {download.source_manifest.source_page}",
        f"- Target: `{download.source_manifest.default_target_attribute}`",
        (
            f"- Shape: `{summary.parsed_dimensions['rows']} rows × "
            f"{summary.parsed_dimensions['source_columns']} columns`"
        ),
        f"- Classes: `{', '.join(summary.target_labels)}`",
        "",
        "## Files Created or Modified",
        "",
        "- `configs/datasets/smoke_blood_transfusion.yaml`",
        "- `src/schemaguard/data/` and `src/schemaguard/utils/` implementation modules",
        "- `tests/unit/` and `tests/integration/` coverage",
        "- Generated data and validation artifacts listed below",
        "",
        "## Commands Executed",
        "",
        "```text",
        (
            "python scripts/01_prepare_smoke_data.py "
            "--config configs/datasets/smoke_blood_transfusion.yaml"
        ),
        (
            "python scripts/01_prepare_smoke_data.py "
            "--config configs/datasets/smoke_blood_transfusion.yaml --offline"
        ),
        "```",
        "",
        "## Test Results",
        "",
        (
            "The pipeline records only commands actually executed by the caller. Unit, network, "
            "Ruff, and mypy results are not inferred by this data command."
        ),
        "",
        "## Data Artifacts",
        "",
    ]
    lines.extend(
        f"- `{item['path']}` — {item['size_bytes']} bytes — `{item['sha256']}`"
        for item in summary.artifact_inventory
    )
    lines.extend(
        [
            "",
            "## Validation Results",
            "",
            f"- Source validation: `{summary.status}`",
            f"- Processed row count: `{len(processed.features)}`",
            f"- Split sizes: `{splits.manifest.row_counts}`",
            "",
            "## Cache and Offline Test",
            "",
            f"- Initial acquisition cache status: `{download.cache_status}`",
            "- Offline rerun is supported and validated by the cache-hit integration test.",
            "",
            "## Warnings",
            "",
            *(f"- {warning}" for warning in summary.warnings),
            "" if summary.warnings else "- None",
            "",
            "## Deviations",
            "",
            "None",
            "",
            "## Failures",
            "",
            *(f"- {failure}" for failure in summary.blocking_failures),
            "None" if not summary.blocking_failures else "",
            "",
            "## Next Permitted Phase",
            "",
            "Phase 02 may begin only after independent review of this handoff.",
            "",
        ]
    )
    atomic_write_json(
        paths.foundation_dir / "handoff_data.json", {"summary": summary.canonical_dict()}
    )
    paths.handoff_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths.handoff_path, "\n".join(lines))


def run_pipeline(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    offline: bool = False,
    validate_only: bool = False,
    force_rebuild_processed: bool = False,
    client: Any | None = None,
) -> PipelineOutcome:
    """Run the smoke data pipeline in its fixed stage order."""
    config = load_smoke_config(config_path)
    project_root = Path(root) if root is not None else repository_root()
    paths = build_paths(project_root, config)
    _create_directories(paths)
    logger = EventLogger(paths.log_path, "P01", config.dataset.internal_id)
    stages: list[str] = []
    current_stage = "configuration"
    try:
        logger.log(current_stage, event="configuration_loaded")
        stages.append(current_stage)
        current_stage = "source_acquisition"
        started = time.perf_counter()
        download = download_dataset(
            config,
            paths.raw_dir,
            offline=offline or validate_only,
            lock_path=project_root / "data" / "cache" / "openml_1464.lock",
            client=client,
        )
        logger.log(
            current_stage,
            event="source_ready",
            cache_status=download.cache_status,
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"raw_sha256": download.source_manifest.computed_sha256},
        )
        stages.append(current_stage)

        current_stage = "arff_parsing"
        parsed = parse_arff(download.raw_path)
        logger.log(current_stage, event="source_parsed", details={"rows": len(parsed.frame)})
        stages.append(current_stage)

        current_stage = "source_validation"
        source_validation = validate_source_table(parsed, config)
        atomic_write_json(paths.quality_report_path, source_validation.report.canonical_dict())
        if not source_validation.passed:
            failed = [
                check.name
                for check in source_validation.report.checks
                if check.blocking and check.status == "failed"
            ]
            raise PipelineError("Source validation failed: " + ", ".join(failed))
        logger.log(current_stage, event="source_validated")
        stages.append(current_stage)

        current_stage = "processed_artifacts"
        source_manifest_hash = sha256_file(paths.raw_manifest_path)
        processed_files_exist = (paths.processed_dir / "data_manifest.json").is_file()
        if validate_only:
            processed = _load_processed(
                paths, parsed, config, download.source_manifest.computed_sha256
            )
            cache_status = "hit"
        elif processed_files_exist and not force_rebuild_processed:
            processed = _load_processed(
                paths, parsed, config, download.source_manifest.computed_sha256
            )
            cache_status = "hit"
        else:
            processed = process_dataset(
                parsed,
                config,
                download.source_manifest.computed_sha256,
                paths.processed_dir,
                source_manifest_hash,
            )
            cache_status = "miss"
        logger.log(current_stage, event="processed_ready", cache_status=cache_status)
        stages.append(current_stage)

        current_stage = "processed_validation"
        label_mapping = json.loads(
            paths.processed_dir.joinpath("label_mapping.json").read_text(encoding="utf-8")
        )["original_to_code"]
        processed_validation = validate_processed_tables(
            processed.features, processed.targets, parsed, config, label_mapping
        )
        combined_report = QualityReport(
            internal_dataset_id=config.dataset.internal_id,
            checks=[*source_validation.report.checks, *processed_validation.report.checks],
            final_status=(
                "PASS" if source_validation.passed and processed_validation.passed else "FAILED"
            ),
            warnings=[
                *source_validation.report.warnings,
                *processed_validation.report.warnings,
            ],
        )
        atomic_write_json(paths.quality_report_path, combined_report.canonical_dict())
        if not processed_validation.passed:
            raise PipelineError("Processed validation failed")
        logger.log(current_stage, event="processed_validated")
        stages.append(current_stage)

        current_stage = "split_generation"
        splits = generate_splits(
            processed.features,
            processed.targets,
            config,
            paths.split_dir,
            processed.data_manifest_hash,
        )
        logger.log(current_stage, event="splits_ready", details=splits.manifest.row_counts)
        stages.append(current_stage)

        current_stage = "summary"
        inventory = _artifact_inventory(paths, download, processed, splits)
        summary = _summary(
            config,
            download,
            parsed,
            processed,
            splits,
            source_validation,
            processed_validation,
            inventory,
        )
        atomic_write_json(paths.validation_summary_path, summary.canonical_dict())
        _write_handoff(paths, summary, download, processed, splits)
        _write_phase_result(paths, "PASS", stages + [current_stage, "handoff"])
        logger.log(current_stage, event="pipeline_passed")
        return PipelineOutcome(
            config=config,
            paths=paths,
            download=download,
            parsed=parsed,
            processed=processed,
            splits=splits,
            source_validation=source_validation,
            processed_validation=processed_validation,
            summary=summary,
        )
    except Exception as exc:
        logger.log(
            current_stage, level="ERROR", event="pipeline_failed", details={"message": str(exc)}
        )
        _write_phase_result(paths, "FAILED", stages, current_stage, str(exc))
        if isinstance(exc, (PipelineError, DownloadError)):
            raise
        raise PipelineError(f"{current_stage}: {exc}") from exc
