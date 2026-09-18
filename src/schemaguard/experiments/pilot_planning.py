"""Metadata-only freezing and deterministic condition planning for the pilot."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import ValidationError

from ..artifact_contracts import DatasetRegistryReportContract
from ..data.contracts import (
    DatasetQualityReportContract,
    DatasetSchemaContract,
    ProcessedManifestContract,
    ProcessedSchema,
    QualityReport,
    RawSourceManifestContract,
)
from ..data.schemaorbit import load_schemaorbit_config
from ..models.adapters.contracts import ModelAdapterInventory
from ..models.registry import ModelSpec, load_runtime_config, validate_registry
from ..splits.contracts import SplitGenerationConfig, SplitGenerationInventoryContract
from ..transformations.contracts import TransformationInventory
from ..transformations.implementation import transformation_engine_implementation_hash
from ..transformations.registry import (
    load_transformation_config,
    registered_views,
)
from ..utils.hashing import canonical_source_hash, sha256_canonical_json, sha256_file
from .pilot_contracts import (
    MODEL_ORDER,
    VIEW_ORDER,
    ConditionIdentityComponents,
    ExcludedDataset,
    NotApplicableConditionTuple,
    PilotConditionInventory,
    PilotConfig,
    PilotDatasetSelection,
    PilotModelEntry,
    PilotProtocolArtifact,
    PilotScheduleStage,
    PilotSeedSelection,
    PilotStagedSchedule,
    PilotViewRegistryEntry,
    PlannedCondition,
    RuntimeEstimate,
    SplitIdentity,
    ViewApplicabilityRecord,
    canonical_identity_hash,
)
from .planning import index_experiment_registry_rows, reconcile_registry_parameters

DATASET_IDS = (3, 23, 29, 31, 36, 37, 38, 44, 46, 50, 54, 1067, 1464, 1489)
EXPECTED_SPLIT_SEEDS = (1729, 2718, 31415, 57721, 161803)
SPLIT_STRATEGY: Literal["stratified_group_5fold_v1"] = "stratified_group_5fold_v1"
_DEVICE_BY_MODEL = {
    "LR-1.9": "cpu",
    "CAT-1.2": "cpu",
    "XGB-3.4": "cpu",
    "TPFN3-8.5": "cuda",
    "TICL2-2.2": "cuda",
}
_VIEW_FAMILIES = {
    "V00": "Identity control",
    "V01": "Numeric affine",
    "V02": "Strictly monotone piecewise numeric mapping",
    "V03": "Category permutation",
    "V04": "Reversible categorical one-hot representation",
    "V05": "Duplicate feature",
    "V06": "Redundant affine feature",
    "V07": "Reversible integer split",
    "V08": "Column permutation",
    "V09": "Row permutation",
    "V10": "Two-transform composition",
}
_ADAPTER_MODULES = {
    "LR-1.9": "src/schemaguard/models/adapters/logistic_regression_adapter.py",
    "CAT-1.2": "src/schemaguard/models/adapters/catboost_adapter.py",
    "XGB-3.4": "src/schemaguard/models/adapters/xgboost_adapter.py",
    "TPFN3-8.5": "src/schemaguard/models/adapters/tabpfn_adapter.py",
    "TICL2-2.2": "src/schemaguard/models/adapters/tabicl_adapter.py",
}
_PROTOCOL_SOURCE_PATHS = (
    "src/schemaguard/experiments/pilot_contracts.py",
    "src/schemaguard/experiments/pilot_planning.py",
    "src/schemaguard/experiments/pilot_validation.py",
    "scripts/freeze_pilot_protocol.py",
    "scripts/validate_pilot_protocol.py",
    "src/schemaguard/experiments/contracts.py",
    "src/schemaguard/experiments/planning.py",
    "src/schemaguard/experiments/execution.py",
    "src/schemaguard/experiments/evaluation.py",
    "src/schemaguard/experiments/evidence.py",
    "src/schemaguard/models/registry.py",
    "src/schemaguard/models/adapters/contracts.py",
    "src/schemaguard/models/adapters/preprocessing.py",
    "src/schemaguard/splits/contracts.py",
    "src/schemaguard/transformations/contracts.py",
    "src/schemaguard/transformations/implementation.py",
    "src/schemaguard/transformations/registry.py",
    "src/schemaguard/cache/contracts.py",
    "src/schemaguard/runner/contracts.py",
    "src/schemaguard/utils/hashing.py",
    "src/schemaguard/utils/io.py",
    "src/schemaguard/utils/resource_monitor.py",
    "src/schemaguard/utils/process_lock.py",
)


class PlanningError(ValueError):
    """Raised when accepted metadata cannot support an immutable pilot plan."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PlanningError(f"expected a JSON object: {path.as_posix()}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PlanningError(f"expected a YAML mapping: {path.as_posix()}")
    return value


def load_pilot_config(path: str | Path) -> PilotConfig:
    """Load the entire closed-world pilot configuration with strict contracts."""
    source = Path(path)
    raw = _read_yaml(source)
    try:
        config = PilotConfig.model_validate(raw)
    except ValidationError as exc:
        raise PlanningError(f"invalid pilot protocol configuration: {exc}") from exc
    if config.protocol.source_commit != "9a59e5a5dda320ec00f1915c80fb82d2531cb442":
        raise PlanningError("pilot protocol starting commit differs from the authorized base")
    if config.protocol.execution_authorized:
        raise PlanningError("pilot execution must remain unauthorized during protocol freezing")
    return config


def _relative_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    base = root.resolve()
    if candidate != base and base not in candidate.parents:
        raise PlanningError(f"input path escapes repository: {value}")
    return candidate


def _sha_file(root: Path, relative: str) -> str:
    path = _relative_path(root, relative)
    if not path.is_file():
        raise PlanningError(f"required frozen input is missing: {relative}")
    return sha256_file(path)


def _canonical_source_digest(root: Path, relative: str) -> str:
    path = _relative_path(root, relative)
    if not path.is_file():
        raise PlanningError(f"required implementation source is missing: {relative}")
    return canonical_source_hash(path)


def _coverage_tags(
    *,
    rows: int,
    features: int,
    numeric: int,
    categorical: int,
    classes: int,
    class_counts: dict[str, int],
    missing_cells: int,
    policy: Any,
) -> tuple[str, ...]:
    fraction = categorical / features
    schema_family = (
        "numeric_only"
        if categorical == 0
        else "categorical_only"
        if numeric == 0
        else "mixed_numeric_categorical"
    )
    max_class_fraction = max(class_counts.values()) / rows
    tags: set[str] = {schema_family, "binary" if classes == 2 else "multiclass"}
    tags.add("small" if rows <= policy.small_dataset_max_rows else "medium")
    tags.add(
        "balanced" if max_class_fraction <= policy.balanced_max_class_fraction else "imbalanced"
    )
    if fraction >= policy.categorical_substantial_fraction:
        tags.add("substantial_categorical")
    if features >= 50 and fraction >= policy.categorical_substantial_fraction:
        tags.add("high_cardinality_categorical")
    if features >= 50:
        tags.add("high_dimensional")
    if missing_cells > 0:
        tags.add("missing_values")
    return tuple(sorted(tags))


def _schema_family(candidate: dict[str, Any]) -> str:
    if candidate["categorical_feature_count"] == 0:
        return "numeric_only"
    if candidate["numeric_feature_count"] == 0:
        return "categorical_only"
    return "mixed"


def _duplicate_profile(candidate: dict[str, Any], policy: Any) -> tuple[str, ...]:
    values = {
        "task_type": "binary" if candidate["class_count"] == 2 else "multiclass",
        "schema_family": _schema_family(candidate),
        "size_band": "small"
        if candidate["row_count"] <= policy.small_dataset_max_rows
        else "medium",
        "balance_band": "balanced"
        if max(candidate["class_distribution"].values()) / candidate["row_count"]
        <= policy.balanced_max_class_fraction
        else "imbalanced",
    }
    return tuple(values[field] for field in policy.duplicate_profile_fields)


def select_candidate_ids(
    candidates: list[dict[str, Any]], policy: Any, *, anchor_id: int = 1464
) -> tuple[int, ...]:
    """Select eight metadata-only records by coverage, cross-strata, and profile diversity.

    No outcomes, predictions, test-row data, or transformation results are accepted by
    this function. Input ordering is ignored; lexical dataset-ID order resolves final ties.
    """
    ordered = sorted(candidates, key=lambda record: int(record["dataset_id"]))
    ids = [int(record["dataset_id"]) for record in ordered]
    required_fields = {
        "dataset_id",
        "row_count",
        "feature_count",
        "numeric_feature_count",
        "categorical_feature_count",
        "class_count",
        "class_distribution",
        "missing_cells",
        "duplicate_group_count",
        "coverage_tags",
        "source_artifacts_valid",
        "validated_split_seeds",
        "cross_partition_group_count",
        "v00_compatible_model_ids",
    }
    if len(ids) != len(set(ids)):
        raise PlanningError("dataset candidate IDs must be unique")
    if set(ids) - set(DATASET_IDS):
        raise PlanningError("dataset selection includes an unregistered SchemaOrbit identifier")
    if any(set(record) != required_fields for record in ordered):
        raise PlanningError(
            "selection candidates must provide only registered metadata and eligibility evidence"
        )
    for record in ordered:
        if (
            record["source_artifacts_valid"] is not True
            or tuple(record["validated_split_seeds"]) != EXPECTED_SPLIT_SEEDS
            or record["cross_partition_group_count"] != 0
            or tuple(record["v00_compatible_model_ids"]) != MODEL_ORDER
        ):
            raise PlanningError(
                "dataset lacks complete source, five grouped split, or V00 model evidence"
            )
    if anchor_id not in ids:
        raise PlanningError("protected anchor dataset 1464 is not an eligible candidate")
    if len(ordered) < 8:
        raise PlanningError(f"only {len(ordered)} datasets satisfy selection preconditions")

    best_ids: tuple[int, ...] | None = None
    best_score: tuple[int, int, int, int] | None = None
    anchor = next(record for record in ordered if int(record["dataset_id"]) == anchor_id)
    others = [record for record in ordered if int(record["dataset_id"]) != anchor_id]
    for remainder in itertools.combinations(others, 7):
        subset = (anchor, *remainder)
        tags = set().union(*(set(item["coverage_tags"]) for item in subset))
        required_covered = len(tags & set(policy.required_coverage))
        strata = {
            (
                _schema_family(item),
                "binary" if item["class_count"] == 2 else "multiclass",
            )
            for item in subset
        }
        profiles = Counter(_duplicate_profile(item, policy) for item in subset)
        profile_collisions = sum(count - 1 for count in profiles.values() if count > 1)
        unique_metadata_signatures = len(
            {
                (
                    item["feature_count"],
                    item["class_count"],
                    item["missing_cells"] > 0,
                    item["duplicate_group_count"],
                )
                for item in subset
            }
        )
        score = (required_covered, len(strata), -profile_collisions, unique_metadata_signatures)
        candidate_ids = tuple(sorted(int(item["dataset_id"]) for item in subset))
        if (
            best_score is None
            or score > best_score
            or (score == best_score and (best_ids is None or candidate_ids < best_ids))
        ):
            best_score = score
            best_ids = candidate_ids
    if best_ids is None:
        raise PlanningError("no deterministic eight-dataset selection could be constructed")
    return best_ids


def _validate_accepted_smoke(root: Path, config: PilotConfig) -> tuple[dict[str, Any], str]:
    path = _relative_path(root, config.protocol.smoke_review)
    schema_path = _relative_path(root, "artifacts/handoff/smoke_independent_review.schema.json")
    from jsonschema import Draft202012Validator

    evidence = _read_json(path)
    schema = _read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(evidence)
    if (
        evidence.get("verdict") != "VERIFIED_PASS"
        or evidence.get("gate_counts")
        != {"failed": 0, "not_applicable": 0, "passed": 32, "total": 32}
        or evidence.get("recomputation")
        != {
            "canonical_class_order": [0, 1],
            "metric_records_recomputed": 20,
            "paired_records_recomputed": 10,
            "prediction_artifacts_validated": 20,
            "probabilities_valid": True,
        }
        or evidence.get("leakage_result") != "PASS"
        or evidence.get("github_actions_run_id") != "35291513248"
        or evidence.get("pilot_started") is not False
    ):
        raise PlanningError("accepted smoke identity or required independent results differ")
    return evidence, sha256_file(path)


def _load_registries(root: Path, config: PilotConfig) -> tuple[Any, Any, Any, Any, Any]:
    dataset_registry = load_schemaorbit_config(
        _relative_path(root, config.protocol.registry_config)
    )
    if tuple(item.id for item in dataset_registry.datasets) != DATASET_IDS:
        raise PlanningError("SchemaOrbit-14 dataset IDs or order changed")
    experiment_raw = _read_yaml(_relative_path(root, config.protocol.experiment_registry))
    experiment_rows = index_experiment_registry_rows(experiment_raw.get("models", []))
    runtime_config = load_runtime_config(_relative_path(root, config.protocol.model_runtime_config))
    model_specs = validate_registry(runtime_config)
    split_config = SplitGenerationConfig.model_validate(
        _read_yaml(_relative_path(root, config.protocol.split_config))
    )
    if tuple(split_config.datasets) != DATASET_IDS or tuple(split_config.seeds) != (
        1729,
        2718,
        31415,
        57721,
        161803,
    ):
        raise PlanningError("split-generation dataset or seed order changed")
    transformation_config = load_transformation_config(
        _relative_path(root, config.protocol.transformation_config)
    )
    if tuple(item.view_id for item in registered_views()) != VIEW_ORDER:
        raise PlanningError("the accepted eleven-view registry changed")
    return dataset_registry, experiment_rows, model_specs, split_config, transformation_config


def _load_dataset_candidates(
    root: Path,
    config: PilotConfig,
    registry: Any,
    split_config: SplitGenerationConfig,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], str, str]:
    dataset_report_path = _relative_path(root, config.protocol.dataset_report)
    dataset_report = DatasetRegistryReportContract.model_validate(_read_json(dataset_report_path))
    if dataset_report.status != "PASS" or dataset_report.dataset_count != 14:
        raise PlanningError("SchemaOrbit-14 source and metadata validation did not pass")
    if {record.openml_data_id for record in dataset_report.validation_records} != set(DATASET_IDS):
        raise PlanningError("dataset validation evidence does not cover exactly SchemaOrbit-14")
    report_by_id = {record.openml_data_id: record for record in dataset_report.datasets}
    if set(report_by_id) != set(DATASET_IDS):
        raise PlanningError("dataset registry report has missing or unregistered records")

    split_inventory_path = _relative_path(root, config.protocol.split_inventory)
    split_inventory = SplitGenerationInventoryContract.model_validate(
        _read_json(split_inventory_path)
    )
    if (
        split_inventory.expected_dataset_ids != list(DATASET_IDS)
        or split_inventory.expected_seeds != list(split_config.seeds)
        or len(split_inventory.records) != 70
    ):
        raise PlanningError(
            "the accepted grouped split inventory is not the complete 70-record set"
        )
    split_records = {(item.dataset_id, item.seed): item for item in split_inventory.records}
    if len(split_records) != 70:
        raise PlanningError("duplicate dataset-seed records exist in split evidence")

    candidate_records: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for spec in registry.datasets:
        dataset_id = int(spec.id)
        record = report_by_id[dataset_id]
        raw_manifest_path = root / "data/raw/openml" / str(dataset_id) / "source_manifest.json"
        processed_root = root / "data/processed/openml" / str(dataset_id)
        if not raw_manifest_path.is_file() or not processed_root.is_dir():
            continue
        try:
            raw_manifest = RawSourceManifestContract.model_validate(_read_json(raw_manifest_path))
            processed_manifest_path = processed_root / "data_manifest.json"
            processed_manifest = ProcessedManifestContract.model_validate(
                _read_json(processed_manifest_path)
            )
            quality_path = processed_root / "quality_report.json"
            quality_payload = _read_json(quality_path)
            try:
                dataset_quality = DatasetQualityReportContract.model_validate(quality_payload)
                quality_matches_registry = dataset_quality.model_dump(
                    mode="json"
                ) == record.quality.model_dump(mode="json")
            except ValidationError:
                legacy_quality = QualityReport.model_validate(quality_payload)
                quality_matches_registry = (
                    legacy_quality.internal_dataset_id == spec.name
                    and legacy_quality.final_status == "PASS"
                )
            schema_payload = _read_json(processed_root / "schema.json")
            try:
                schema: DatasetSchemaContract | ProcessedSchema = (
                    DatasetSchemaContract.model_validate(schema_payload)
                )
            except ValidationError:
                schema = ProcessedSchema.model_validate(schema_payload)
        except (ValidationError, OSError, KeyError, ValueError):
            continue
        if (
            record.status != "PASS"
            or raw_manifest.openml_data_id != dataset_id
            or raw_manifest.openml_file_id != spec.file_id
            or raw_manifest.dataset_version != spec.version
            or raw_manifest.internal_dataset_id not in (None, spec.name)
            or raw_manifest.computed_sha256 != record.raw_sha256
            or raw_manifest.provider_md5 != spec.provider_md5
            or processed_manifest.openml_data_id != dataset_id
            or processed_manifest.internal_dataset_id != spec.name
            or processed_manifest.raw_sha256 != record.raw_sha256
            or processed_manifest.row_count != record.observed_rows
            or record.observed_rows != spec.rows
            or record.observed_predictors != spec.predictors
            or not quality_matches_registry
            or getattr(schema, "openml_data_id", dataset_id) != dataset_id
            or schema.internal_dataset_id != spec.name
            or schema.raw_sha256 != record.raw_sha256
        ):
            continue
        hashes = processed_manifest.artifact_hashes
        actual_hashes_match = True
        for filename, expected_hash in hashes.items():
            artifact = processed_root / filename
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                actual_hashes_match = False
                break
        raw_relative_path = raw_manifest.raw_relative_path.replace("\\", "/")
        if not raw_relative_path.startswith("raw/openml/"):
            continue
        raw_project_path = f"data/{raw_relative_path}"
        raw_path = _relative_path(root, raw_project_path)
        if not raw_path.is_file() or sha256_file(raw_path) != raw_manifest.computed_sha256:
            actual_hashes_match = False
        if not actual_hashes_match:
            continue
        attributes = [item for item in schema.attributes if not getattr(item, "is_target", False)]
        numeric_count = sum(item.kind == "numeric" for item in attributes)
        categorical_count = sum(item.kind == "categorical" for item in attributes)
        feature_count = len(schema.feature_columns)
        if (
            feature_count != record.observed_predictors
            or feature_count != numeric_count + categorical_count
        ):
            continue
        class_counts = {str(key): int(value) for key, value in record.class_counts.items()}
        if (
            sum(class_counts.values()) != record.observed_rows
            or len(class_counts) != record.class_count
        ):
            continue

        five_splits: list[SplitIdentity] = []
        split_ok = True
        for seed in split_config.seeds:
            split_record = split_records.get((dataset_id, int(seed)))
            if (
                split_record is None
                or split_record.status != "PASS"
                or split_record.cross_partition_group_count != 0
                or split_record.assignment_artifact_hash is None
                or split_record.manifest_artifact_hash is None
                or split_record.logical_assignment_hash is None
                or split_record.row_count is None
                or split_record.partition_counts is None
            ):
                split_ok = False
                break
            assignment_path = _relative_path(root, split_record.assignments_path)
            manifest_path = _relative_path(root, split_record.manifest_path)
            if (
                not assignment_path.is_file()
                or not manifest_path.is_file()
                or sha256_file(assignment_path) != split_record.assignment_artifact_hash
                or sha256_file(manifest_path) != split_record.manifest_artifact_hash
            ):
                split_ok = False
                break
            manifest = _read_json(manifest_path)
            if (
                manifest.get("strategy") != SPLIT_STRATEGY
                or manifest.get(
                    "cross_partition_group_count",
                    manifest.get("predictor_duplicate_groups_crossing_splits"),
                )
                != 0
                or manifest.get("assignment_artifact_hash", manifest.get("assignment_file_sha256"))
                != split_record.assignment_artifact_hash
                or (
                    manifest.get("logical_assignment_hash") is not None
                    and manifest.get("logical_assignment_hash")
                    != split_record.logical_assignment_hash
                )
                or manifest.get("validation_status", manifest.get("status")) != "PASS"
            ):
                split_ok = False
                break
            five_splits.append(
                SplitIdentity(
                    seed=int(seed),
                    strategy=SPLIT_STRATEGY,
                    assignment_sha256=split_record.assignment_artifact_hash,
                    logical_assignment_sha256=split_record.logical_assignment_hash,
                    manifest_sha256=split_record.manifest_artifact_hash,
                    cross_partition_group_count=0,
                    row_count=split_record.row_count,
                    partition_counts=cast(
                        dict[Literal["train", "calibration", "test"], int],
                        split_record.partition_counts,
                    ),
                )
            )
        if not split_ok or len(five_splits) != 5:
            continue
        if (
            record.observed_rows > config.selection.max_rows
            or feature_count > config.selection.max_features
            or record.class_count > config.selection.max_classes
            or any(
                item.partition_counts["train"] > config.selection.foundation_max_train_rows
                for item in five_splits
            )
        ):
            continue
        candidate = {
            "dataset_id": dataset_id,
            "dataset_name": record.dataset_name,
            "provider": "openml",
            "provider_dataset_version": record.dataset_version,
            "provider_file_id": record.openml_file_id,
            "target_column": schema.target_column,
            "task_type": "binary" if record.class_count == 2 else "multiclass",
            "row_count": record.observed_rows,
            "feature_count": feature_count,
            "numeric_feature_count": numeric_count,
            "categorical_feature_count": categorical_count,
            "class_count": record.class_count,
            "class_distribution": class_counts,
            "predictor_group_count": split_records[
                (dataset_id, int(split_config.seeds[0]))
            ].predictor_group_count,
            "duplicate_group_count": record.quality.duplicate_predictor_groups,
            "license": str(record.license or "UNSPECIFIED"),
            "raw_source_sha256": record.raw_sha256,
            "processed_feature_sha256": hashes["features.parquet"],
            "target_artifact_sha256": hashes["targets.parquet"],
            "quality_report_sha256": sha256_file(quality_path),
            "metadata_evidence_sha256": sha256_canonical_json(record.model_dump(mode="json")),
            "five_split_identities": tuple(five_splits),
            "missing_cells": record.quality.missing_cells,
        }
        candidate["coverage_tags"] = _coverage_tags(
            rows=record.observed_rows,
            features=feature_count,
            numeric=numeric_count,
            categorical=categorical_count,
            classes=record.class_count,
            class_counts=class_counts,
            missing_cells=record.quality.missing_cells,
            policy=config.selection,
        )
        by_id[dataset_id] = candidate
        candidate_records.append(candidate)
    return (
        candidate_records,
        by_id,
        sha256_file(dataset_report_path),
        sha256_file(split_inventory_path),
    )


def _validate_v00_compatibility(
    candidates: list[dict[str, Any]],
    model_specs: tuple[ModelSpec, ...],
    adapter_inventory: ModelAdapterInventory,
    gpu_report: dict[str, Any],
    adapter_config: dict[str, Any],
    model_compatibility: PilotConfig,
) -> tuple[dict[int, dict[str, str]], dict[str, str]]:
    adapter_models = {record.model_id for record in adapter_inventory.records}
    if adapter_models != set(MODEL_ORDER):
        raise PlanningError("adapter evidence does not cover all five accepted models")
    if gpu_report.get("status") != "PASS":
        raise PlanningError("accepted GPU capacity evidence is unavailable")
    gpu_rows = gpu_report.get("rows", [])
    gpu_pass = {
        row.get("model_id")
        for row in gpu_rows
        if row.get("device") == "cuda" and row.get("status") == "PASS"
    }
    if not {"TPFN3-8.5", "TICL2-2.2"}.issubset(gpu_pass):
        raise PlanningError("foundation-model CUDA profile evidence is incomplete")
    gpu_profiles = gpu_report.get("profiles", [])
    compat_by_dataset: dict[int, dict[str, str]] = {}
    adapter_hashes: dict[str, str] = {}
    for spec in model_specs:
        matching = [item for item in adapter_inventory.records if item.model_id == spec.id]
        if not matching or any(
            item.status != "PASS" or not item.roundtrip_passed for item in matching
        ):
            raise PlanningError(f"accepted adapter evidence is incomplete for {spec.id}")
        adapter_hashes[spec.id] = sha256_canonical_json(
            [item.model_dump(mode="json") for item in matching]
        )
    for dataset in candidates:
        dataset_id = int(dataset["dataset_id"])
        compat_by_dataset[dataset_id] = {}
        for spec in model_specs:
            if (
                dataset["row_count"] > 6000
                or dataset["feature_count"] > 100
                or dataset["class_count"] > 10
                or max(item.partition_counts["train"] for item in dataset["five_split_identities"])
                > 5000
            ):
                raise PlanningError(f"dataset {dataset_id} exceeds a frozen model input limit")
            if spec.id in {"TPFN3-8.5", "TICL2-2.2"}:
                envelope = any(
                    profile.get("rows", 0) >= dataset["row_count"]
                    and profile.get("predictors", 0) >= dataset["feature_count"]
                    and profile.get("classes", 0) >= dataset["class_count"]
                    for profile in gpu_profiles
                )
                if not envelope:
                    raise PlanningError(
                        f"foundation GPU capacity evidence does not envelope dataset {dataset_id}"
                    )
            compat_by_dataset[dataset_id][spec.id] = "PASS_REGISTRY_CAPABILITY"
    if adapter_config.get("resources", {}).get("gpu_soft_limit_mib") != 3600:
        raise PlanningError("adapter GPU ceiling differs from frozen 3,600 MiB policy")
    if model_compatibility.models.required_ids != MODEL_ORDER:
        raise PlanningError("configured model identifiers differ from the accepted five-model set")
    return compat_by_dataset, adapter_hashes


def _effective_parameters(model: ModelSpec, class_count: int, seed: int) -> dict[str, Any]:
    parameters = dict(model.parameters)
    if model.id == "CAT-1.2":
        parameters["loss_function"] = "Logloss" if class_count == 2 else "MultiClass"
        parameters["random_seed"] = seed
    else:
        parameters["random_state"] = seed
    return parameters


def _build_model_matrix(
    root: Path,
    config: PilotConfig,
    experiment_rows: dict[str, dict[str, Any]],
    model_specs: tuple[ModelSpec, ...],
    adapter_inventory: ModelAdapterInventory,
    adapter_hashes: dict[str, str],
    model_adapter_raw: dict[str, Any],
) -> tuple[tuple[PilotModelEntry, ...], tuple[str, ...]]:
    checkpoint_config = model_adapter_raw.get("checkpoints", {})
    specs_by_id = {item.id: item for item in model_specs}
    adapter_source_hash = _canonical_source_digest(
        root, "src/schemaguard/models/adapters/preprocessing.py"
    )
    entries: list[PilotModelEntry] = []
    decisions: list[str] = []
    for model_id in MODEL_ORDER:
        spec = specs_by_id[model_id]
        registry_row = experiment_rows[model_id]
        params = dict(spec.parameters)
        row_params = dict(registry_row.get("parameters", {}))
        decisions.extend(
            reconcile_registry_parameters(
                model_id,
                row_params,
                params,
                spec.checkpoint,
            )
        )
        checkpoint_sha = config.models.checkpoint_hashes.get(model_id)
        checkpoint_record = checkpoint_config.get(model_id)
        if (spec.checkpoint is None) != (checkpoint_record is None):
            raise PlanningError(f"checkpoint registry presence differs for {model_id}")
        if spec.checkpoint is not None:
            if (
                checkpoint_record.get("identifier") != spec.checkpoint
                or checkpoint_record.get("sha256") != checkpoint_sha
            ):
                raise PlanningError(f"checkpoint identity mismatch for {model_id}")
            recorded = [
                item
                for item in adapter_inventory.records
                if item.model_id == model_id and item.checkpoint_sha256 is not None
            ]
            if not recorded or any(item.checkpoint_sha256 != checkpoint_sha for item in recorded):
                raise PlanningError(f"accepted adapter checkpoint hashes mismatch for {model_id}")
        elif checkpoint_sha is not None:
            raise PlanningError(f"classical model unexpectedly has a checkpoint hash: {model_id}")
        adapter_source = _canonical_source_digest(root, _ADAPTER_MODULES[model_id])
        preprocessing_sha = sha256_canonical_json(
            {
                "preprocessing": registry_row["preprocessing"],
                "preprocessing_source_sha256": adapter_source_hash,
                "adapter_source_sha256": adapter_source,
            }
        )
        model_payload = {
            "id": model_id,
            "class_path": spec.class_path,
            "package": spec.package,
            "version": spec.expected_version,
            "checkpoint": spec.checkpoint,
            "parameters": params,
            "preprocessing": registry_row["preprocessing"],
            "documented_parameter_decisions": sorted(
                decision
                for decision in decisions
                if (model_id == "CAT-1.2" and decision == "catboost_yaml_no_scalar")
                or (
                    model_id == "TICL2-2.2"
                    and decision == "tabicl_checkpoint_version_from_checkpoint_field"
                )
            ),
        }
        entries.append(
            PilotModelEntry(
                model_id=model_id,
                model_class=spec.class_path,
                package=spec.package,
                version=spec.expected_version,
                preprocessing=registry_row["preprocessing"],
                parameters=params,
                model_config_sha256=sha256_canonical_json(model_payload),
                preprocessing_sha256=preprocessing_sha,
                checkpoint_identifier=spec.checkpoint,
                checkpoint_sha256=checkpoint_sha,
                device=config.models.devices[model_id],
                precision_policy=config.models.precision_policy[model_id],
                batch_policy=config.models.batch_policy[model_id],
                compatibility_evidence_sha256=adapter_hashes[model_id],
            )
        )
    if set(decisions) != {
        "catboost_yaml_no_scalar",
        "tabicl_checkpoint_version_from_checkpoint_field",
    }:
        raise PlanningError(
            "frozen registry reconciliation did not produce both documented decisions"
        )
    return tuple(entries), tuple(sorted(set(decisions)))


def _build_view_registry(transformation_config: Any) -> tuple[PilotViewRegistryEntry, ...]:
    by_id = {str(item["id"]): item for item in transformation_config.views}
    output = tuple(
        PilotViewRegistryEntry.model_validate(
            {
                "view_id": view.view_id,
                "view_name": view.name,
                "transformation_family": _VIEW_FAMILIES[view.view_id],
                "certificate_type": view.certificate_type,
                "scientific_role": view.scientific_role,
            }
        )
        for view in registered_views()
    )
    if tuple(item.view_id for item in output) != VIEW_ORDER:
        raise PlanningError("view registry order differs from the frozen eleven-view matrix")
    if any(
        by_id[item.view_id].get("name") != item.view_name
        or by_id[item.view_id].get("certificate_type") != item.certificate_type
        or by_id[item.view_id].get("scientific_role") != item.scientific_role
        for item in output
    ):
        raise PlanningError(
            "runtime view configuration differs from the accepted transformation registry"
        )
    return output


def _collect_view_applicability(
    root: Path,
    config: PilotConfig,
    dataset_ids: tuple[int, ...],
    seeds: tuple[int, int, int],
    datasets_by_id: dict[int, dict[str, Any]],
    transformation_config: Any,
) -> tuple[tuple[ViewApplicabilityRecord, ...], str, str]:
    inventory_path = _relative_path(root, config.protocol.transformation_inventory)
    inventory = TransformationInventory.model_validate(_read_json(inventory_path))
    if (
        inventory.status != "PASS_PENDING_REVIEW"
        or inventory.expected_record_count != 770
        or inventory.fail_count != 0
        or inventory.pass_count != 545
        or inventory.not_applicable_count != 225
        or inventory.property_example_count != 11000
    ):
        raise PlanningError("accepted transformation inventory does not match frozen evidence")
    expected_views = {
        (dataset_id, seed, view_id)
        for dataset_id in dataset_ids
        for seed in seeds
        for view_id in VIEW_ORDER
    }
    records = {
        (record.dataset_id, record.seed, record.view_id): record for record in inventory.records
    }
    if len(records) != inventory.expected_record_count:
        raise PlanningError("transformation inventory contains duplicate tuple identities")
    if not expected_views.issubset(records):
        raise PlanningError("selected dataset-seed-view tuples are missing from accepted evidence")
    config_hash = _sha_file(root, config.protocol.transformation_config)
    implementation_hash = transformation_engine_implementation_hash(root / "src/schemaguard")
    view_types = {view.view_id: view.certificate_type for view in registered_views()}
    rtol = float(transformation_config.numerical_rtol)
    atol = float(transformation_config.numerical_atol)
    output: list[ViewApplicabilityRecord] = []
    for dataset_id in dataset_ids:
        for seed in seeds:
            for view_id in VIEW_ORDER:
                item = records[(dataset_id, seed, view_id)]
                if item.status not in {"PASS", "N/A"}:
                    raise PlanningError(
                        "accepted transformation record is not PASS/N/A for "
                        f"{(dataset_id, seed, view_id)}"
                    )
                applicable = item.status == "PASS"
                evidence_hash = sha256_canonical_json(item.model_dump(mode="json"))
                output.append(
                    ViewApplicabilityRecord.model_validate(
                        {
                            "dataset_id": dataset_id,
                            "seed": seed,
                            "view_id": view_id,
                            "view_name": item.view_name,
                            "applicability": ("APPLICABLE" if applicable else "NOT_APPLICABLE"),
                            "reason_code": None if applicable else item.reason_code,
                            "reason": None if applicable else item.reason,
                            "transformation_inventory_record_sha256": evidence_hash,
                            "transformation_config_sha256": config_hash,
                            "transformation_implementation_sha256": implementation_hash,
                            "certificate_type": view_types[view_id],
                            "certificate_identities": tuple(item.certificate_ids),
                            "source_feature_sha256": item.source_hash,
                            "target_artifact_sha256": datasets_by_id[dataset_id][
                                "target_artifact_sha256"
                            ],
                            "reconstruction_rtol": rtol,
                            "reconstruction_atol": atol,
                            "reconstruction_max_abs_error": item.reconstruction_max_abs_error,
                        }
                    )
                )
    return tuple(output), config_hash, implementation_hash


def _all_source_hashes(root: Path, config: PilotConfig) -> dict[str, str]:
    relative_inputs = (
        "configs/runtime/pilot_protocol.yaml",
        config.protocol.registry_config,
        config.protocol.experiment_registry,
        config.protocol.split_config,
        config.protocol.transformation_config,
        config.protocol.adapter_config,
        config.protocol.model_runtime_config,
        config.protocol.dependency_lock,
        config.protocol.dataset_report,
        config.protocol.split_inventory,
        config.protocol.transformation_inventory,
        config.protocol.adapter_inventory,
        "artifacts/handoff/model_adapter_leakage_evidence.json",
        "artifacts/handoff/cache_scheduler_inventory.json",
        "artifacts/handoff/cache_scheduler_fault_evidence.json",
        "artifacts/handoff/cache_scheduler_probe_evidence.json",
        "artifacts/handoff/cache_scheduler_protected_hash_comparison.json",
        config.protocol.smoke_review,
        "artifacts/handoff/smoke_experiment_inventory.json",
        "results/validation/gpu_capacity_report.json",
        "results/validation/model_compatibility_report.json",
        "results/validation/checkpoint_inventory.json",
        "results/validation/license_inventory.json",
        config.protocol.smoke_runtime_report,
    )
    hashes = {relative: _sha_file(root, relative) for relative in relative_inputs}
    hashes["artifacts/handoff/smoke_independent_review.schema.json"] = _sha_file(
        root, "artifacts/handoff/smoke_independent_review.schema.json"
    )
    for relative in _PROTOCOL_SOURCE_PATHS:
        hashes[relative] = _canonical_source_digest(root, relative)
    for model_id, relative in _ADAPTER_MODULES.items():
        hashes[relative] = _canonical_source_digest(root, relative)
    hashes["src/schemaguard/transformations/implementation_tree"] = (
        transformation_engine_implementation_hash(root / "src/schemaguard")
    )
    return dict(sorted(hashes.items()))


def _runtime_estimate(
    root: Path,
    config: PilotConfig,
    selected: tuple[PilotDatasetSelection, ...],
    applicability: tuple[ViewApplicabilityRecord, ...],
    models: tuple[PilotModelEntry, ...],
) -> RuntimeEstimate:
    smoke_path = _relative_path(root, config.protocol.smoke_runtime_report)
    smoke = _read_json(smoke_path)
    adapter = ModelAdapterInventory.model_validate(
        _read_json(_relative_path(root, config.protocol.adapter_inventory))
    )
    gpu = _read_json(_relative_path(root, "results/validation/gpu_capacity_report.json"))
    cpu_observed: dict[str, list[float]] = {model_id: [] for model_id in MODEL_ORDER}
    for condition in smoke.get("conditions", []):
        resource = condition.get("resource") or {}
        duration = resource.get("wall_seconds")
        if condition.get("status") == "PASS" and duration is not None:
            cpu_observed[condition["model_id"]].append(float(duration))
    adapter_times: dict[str, list[float]] = {model_id: [] for model_id in MODEL_ORDER}
    peak_ram = 0.0
    for record in adapter.records:
        if record.runtime_seconds is not None:
            adapter_times[record.model_id].append(float(record.runtime_seconds))
        if record.peak_process_tree_ram_mib is not None:
            peak_ram = max(peak_ram, float(record.peak_process_tree_ram_mib))
    for condition in smoke.get("conditions", []):
        resource = condition.get("resource") or {}
        ram = resource.get("peak_process_tree_rss_mib")
        if ram is not None:
            peak_ram = max(peak_ram, float(ram))
    gpu_times: dict[str, list[float]] = {model_id: [] for model_id in MODEL_ORDER}
    peak_vram = 0.0
    for record in gpu.get("rows", []):
        if record.get("status") != "PASS":
            continue
        model_id = record.get("model_id")
        duration = record.get("runtime_seconds")
        if model_id in gpu_times and duration is not None:
            gpu_times[model_id].append(float(duration))
        for key in ("peak_vram_reserved_mib", "peak_vram_allocated_mib"):
            if record.get(key) is not None:
                peak_vram = max(peak_vram, float(record[key]))

    base_seconds: dict[str, float] = {}
    for model_id in MODEL_ORDER:
        observed = cpu_observed[model_id]
        adapter_median = statistics.median(adapter_times[model_id])
        gpu_max = max(gpu_times[model_id], default=0.0)
        base_seconds[model_id] = max(max(observed, default=0.0), adapter_median, gpu_max)
    dataset_map = {item.dataset_id: item for item in selected}
    seed_split = {
        (item.dataset_id, split.seed): split
        for item in selected
        for split in item.five_split_identities
    }
    applicable_views = [item for item in applicability if item.applicability == "APPLICABLE"]
    condition_counts = Counter({model_id: 0 for model_id in MODEL_ORDER})
    estimated_work = {"cpu": 0.0, "cuda": 0.0}
    work_by_dataset: dict[int, dict[Literal["cpu", "cuda"], float]] = {
        dataset_id: {"cpu": 0.0, "cuda": 0.0} for dataset_id in dataset_map
    }
    prediction_bytes = 0
    prediction_storage_by_dataset = {dataset_id: 0 for dataset_id in dataset_map}
    for view in applicable_views:
        dataset = dataset_map[view.dataset_id]
        split = seed_split[(view.dataset_id, view.seed)]
        train_rows = split.partition_counts["train"]
        scale = (
            max(1.0, train_rows / 449.0)
            * math.sqrt(max(1.0, dataset.feature_count / 4.0))
            * max(1.0, dataset.class_count / 2.0)
        )
        inference_rows = split.partition_counts["calibration"] + split.partition_counts["test"]
        per_row = (
            config.resources.storage_bytes_per_row_id
            + dataset.class_count * config.resources.storage_bytes_per_row_class
            + config.resources.per_row_metadata_overhead_bytes
        )
        view_prediction_bytes = (
            inference_rows * per_row + 2 * config.resources.prediction_artifact_overhead_bytes
        ) * len(models)
        prediction_bytes += view_prediction_bytes
        prediction_storage_by_dataset[dataset.dataset_id] += view_prediction_bytes
        for model in models:
            condition_counts[model.model_id] += 1
            estimated_work[model.device] += base_seconds[model.model_id] * scale
            work_by_dataset[dataset.dataset_id][model.device] += (
                base_seconds[model.model_id] * scale
            )
    safety = config.resources.estimate_safety_factor
    cpu_critical = estimated_work["cpu"] / config.resources.cpu_workers
    cuda_critical = estimated_work["cuda"]
    wall = max(cpu_critical, cuda_critical) * safety + config.resources.coordination_seconds
    cache_bytes = int(prediction_bytes * 2.0)
    measurement_paths = (
        config.protocol.smoke_runtime_report,
        config.protocol.adapter_inventory,
        "results/validation/gpu_capacity_report.json",
    )
    measurement_hash = sha256_canonical_json(
        {path: _sha_file(root, path) for path in measurement_paths}
    )
    return RuntimeEstimate(
        source_measurement_sha256=measurement_hash,
        model_condition_counts=dict(condition_counts),
        estimated_work_seconds_by_dataset=work_by_dataset,
        estimated_prediction_storage_by_dataset_bytes=prediction_storage_by_dataset,
        estimated_cpu_seconds=estimated_work["cpu"] * safety,
        estimated_cuda_seconds=estimated_work["cuda"] * safety,
        conservative_wall_seconds=wall,
        estimated_prediction_storage_bytes=prediction_bytes,
        estimated_cache_storage_bytes=cache_bytes,
        peak_ram_mib=peak_ram,
        peak_vram_mib=peak_vram,
        estimate_method=(
            "For each scheduled applicable tuple, use the maximum of measured smoke wall time, "
            "median accepted adapter-probe time, and maximum accepted GPU-profile time for its "
            "model; scale by train-row ratio, square-root feature ratio, and class ratio; apply "
            "the configured 1.5 safety factor; CPU work is scheduled over two workers and CUDA "
            "work is strictly serial. Storage includes calibration/test rows, class probabilities, "
            "row IDs, per-row metadata allowance, two partition-artifact headers, and a 2x cache "
            "allowance. The full condition matrix is retained; a staged schedule is required if "
            "the preferred local target is exceeded."
        ),
        preferred_wall_limit_passed=wall <= config.resources.preferred_wall_seconds,
        hard_wall_limit_passed=wall <= config.resources.hard_wall_seconds,
        preferred_storage_limit_passed=cache_bytes <= config.resources.preferred_storage_bytes,
        staged_schedule_required=(
            wall > config.resources.preferred_wall_seconds
            or cache_bytes > config.resources.preferred_storage_bytes
        ),
    )


def build_staged_schedule(
    protocol: PilotProtocolArtifact,
    inventory: PilotConditionInventory,
) -> PilotStagedSchedule:
    """Build a deterministic whole-dataset schedule without executing conditions."""
    if inventory.protocol_sha256 != protocol.protocol_sha256:
        raise PlanningError("staged schedule protocol hash differs from condition inventory")
    dataset_ids = tuple(
        sorted(protocol_dataset.dataset_id for protocol_dataset in protocol.selected_datasets)
    )
    if inventory.selected_dataset_ids != dataset_ids:
        raise PlanningError("staged schedule dataset IDs differ from the frozen protocol")
    estimate = protocol.runtime_estimate
    if set(estimate.estimated_work_seconds_by_dataset) != set(dataset_ids):
        raise PlanningError("runtime estimate lacks a selected dataset work breakdown")

    policy = protocol.resource_policy
    stage_groups = tuple(
        tuple(dataset_ids[index : index + policy.staging.datasets_per_stage])
        for index in range(0, len(dataset_ids), policy.staging.datasets_per_stage)
    )
    records_by_dataset: dict[int, list[PlannedCondition]] = {
        dataset_id: [] for dataset_id in dataset_ids
    }
    for condition in inventory.records:
        records_by_dataset[condition.dataset_id].append(condition)

    stages: list[PilotScheduleStage] = []
    for dataset_group in stage_groups:
        if len(dataset_group) != policy.staging.datasets_per_stage:
            raise PlanningError("staged schedule dataset group is incomplete")
        stage_dataset_ids = (dataset_group[0], dataset_group[1])
        condition_records = [
            condition
            for dataset_id in dataset_group
            for condition in records_by_dataset[dataset_id]
        ]
        model_counts = {
            model_id: sum(condition.model_id == model_id for condition in condition_records)
            for model_id in MODEL_ORDER
        }
        cpu_work = sum(
            estimate.estimated_work_seconds_by_dataset[dataset_id]["cpu"]
            for dataset_id in dataset_group
        ) * policy.estimate_safety_factor
        cuda_work = sum(
            estimate.estimated_work_seconds_by_dataset[dataset_id]["cuda"]
            for dataset_id in dataset_group
        ) * policy.estimate_safety_factor
        prediction_storage = sum(
            estimate.estimated_prediction_storage_by_dataset_bytes[dataset_id]
            for dataset_id in dataset_group
        )
        stage_id = "dataset_group_" + "_".join(
            str(dataset_id) for dataset_id in stage_dataset_ids
        )
        stages.append(
            PilotScheduleStage(
                stage_id=stage_id,
                dataset_ids=stage_dataset_ids,
                condition_ids=tuple(
                    sorted(condition.condition_id for condition in condition_records)
                ),
                model_condition_counts=model_counts,
                cpu_condition_count=sum(
                    condition.device == "cpu" for condition in condition_records
                ),
                cuda_condition_count=sum(
                    condition.device == "cuda" for condition in condition_records
                ),
                estimated_cpu_seconds=cpu_work,
                estimated_cuda_seconds=cuda_work,
                conservative_wall_seconds=max(
                    cpu_work / policy.cpu_workers, cuda_work
                )
                + policy.coordination_seconds,
                estimated_prediction_storage_bytes=prediction_storage,
                estimated_cache_storage_bytes=prediction_storage * 2,
            )
        )

    staged_condition_ids = [
        condition_id for stage in stages for condition_id in stage.condition_ids
    ]
    inventory_condition_ids = [condition.condition_id for condition in inventory.records]
    if len(staged_condition_ids) != len(set(staged_condition_ids)) or set(
        staged_condition_ids
    ) != set(inventory_condition_ids):
        raise PlanningError("staged schedule must preserve every frozen condition exactly once")
    if any(
        sum(stage.model_condition_counts[model_id] for stage in stages)
        != inventory.counts_by_model[model_id]
        for model_id in MODEL_ORDER
    ):
        raise PlanningError("staged schedule model counts differ from the frozen inventory")

    schedule_payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol.protocol_sha256,
        "inventory_sha256": inventory.inventory_sha256,
        "schedule_policy": policy.staging.stage_grouping,
        "dataset_ids": dataset_ids,
        "stages": [stage.model_dump(mode="json") for stage in stages],
        "total_condition_count": inventory.actual_condition_count,
        "not_applicable_view_count": inventory.not_applicable_view_count,
        "total_estimated_wall_seconds": sum(
            stage.conservative_wall_seconds for stage in stages
        ),
        "hard_wall_limit_seconds": policy.hard_wall_seconds,
        "all_stages_within_hard_wall_limit": all(
            stage.conservative_wall_seconds <= policy.hard_wall_seconds for stage in stages
        ),
        "estimated_prediction_storage_bytes": sum(
            stage.estimated_prediction_storage_bytes for stage in stages
        ),
        "estimated_cache_storage_bytes": sum(
            stage.estimated_cache_storage_bytes for stage in stages
        ),
        "full_condition_matrix_preserved": policy.staging.preserve_complete_condition_matrix,
    }
    schedule_hash = sha256_canonical_json(schedule_payload)
    return PilotStagedSchedule.model_validate(
        {"schedule_sha256": schedule_hash, **schedule_payload}
    )


def build_pilot_plan(
    root: str | Path,
    config: PilotConfig,
) -> tuple[PilotProtocolArtifact, PilotConditionInventory]:
    """Freeze the full condition matrix using metadata and accepted evidence only.

    This function never reads Parquet/ARFF rows and never imports or fits an estimator.
    It hashes complete data artifacts only to verify their already-accepted identities.
    """
    project = Path(root).resolve()
    registry, experiment_rows, model_specs, split_config, transformation_config = _load_registries(
        project, config
    )
    smoke_evidence, _smoke_hash = _validate_accepted_smoke(project, config)
    candidates, datasets_by_id, _dataset_report_hash, _split_inventory_hash = (
        _load_dataset_candidates(project, config, registry, split_config)
    )
    if len(candidates) < config.protocol.selected_dataset_count:
        raise PlanningError(
            f"BLOCKED: only {len(candidates)} of eight required datasets satisfy source, "
            "quality, and split checks"
        )
    # Derive metadata-only eligibility tags before diversity scoring.
    eligible_by_id = {
        int(item["dataset_id"]): item
        for item in candidates
        if item["numeric_feature_count"] + item["categorical_feature_count"]
        == item["feature_count"]
        and all(split.cross_partition_group_count == 0 for split in item["five_split_identities"])
    }
    if 1464 not in eligible_by_id:
        raise PlanningError("BLOCKED: protected OpenML 1464 anchor did not validate")
    adapter_inventory = ModelAdapterInventory.model_validate(
        _read_json(_relative_path(project, config.protocol.adapter_inventory))
    )
    model_adapter_raw = _read_yaml(_relative_path(project, config.protocol.adapter_config))
    model_compatibility, _ = _validate_v00_compatibility(
        list(eligible_by_id.values()),
        model_specs,
        adapter_inventory,
        _read_json(_relative_path(project, "results/validation/gpu_capacity_report.json")),
        model_adapter_raw,
        config,
    )
    if any(
        any(
            status != "PASS_REGISTRY_CAPABILITY"
            for status in model_compatibility[dataset_id].values()
        )
        for dataset_id in eligible_by_id
    ):
        raise PlanningError("V00 compatibility is not proven for all eligible datasets and models")

    candidate_for_selection = [
        {
            key: value
            for key, value in item.items()
            if key
            in {
                "dataset_id",
                "row_count",
                "feature_count",
                "numeric_feature_count",
                "categorical_feature_count",
                "class_count",
                "class_distribution",
                "missing_cells",
                "coverage_tags",
                "duplicate_group_count",
            }
        }
        | {
            "source_artifacts_valid": True,
            "validated_split_seeds": tuple(split.seed for split in item["five_split_identities"]),
            "cross_partition_group_count": max(
                split.cross_partition_group_count for split in item["five_split_identities"]
            ),
            "v00_compatible_model_ids": tuple(
                model_id
                for model_id in MODEL_ORDER
                if model_compatibility[int(item["dataset_id"])][model_id]
                == "PASS_REGISTRY_CAPABILITY"
            ),
        }
        for item in eligible_by_id.values()
    ]
    selected_ids = select_candidate_ids(candidate_for_selection, config.selection)
    required_coverage = set(config.selection.required_coverage)
    selected_tag_union = set().union(
        *(set(eligible_by_id[dataset_id]["coverage_tags"]) for dataset_id in selected_ids)
    )
    missing_coverage = required_coverage - selected_tag_union
    if missing_coverage:
        raise PlanningError(
            f"selected cohort fails required metadata coverage: {sorted(missing_coverage)}"
        )

    selected_data: list[PilotDatasetSelection] = []
    for dataset_id in selected_ids:
        item = eligible_by_id[dataset_id]
        if dataset_id == 1464:
            rationale = (
                "Protected SchemaGuard anchor; preserves the existing small numerical binary "
                "and grouped-split lineage without using smoke outcomes for selection."
            )
        else:
            rationale = (
                "Selected by the metadata-only maximum-coverage objective; contributes "
                + ", ".join(item["coverage_tags"])
                + " while minimizing duplicate task/schema/size/balance profiles."
            )
        selected_data.append(
            PilotDatasetSelection(
                **{
                    key: item[key]
                    for key in (
                        "dataset_id",
                        "provider",
                        "provider_dataset_version",
                        "provider_file_id",
                        "dataset_name",
                        "target_column",
                        "task_type",
                        "row_count",
                        "feature_count",
                        "numeric_feature_count",
                        "categorical_feature_count",
                        "class_count",
                        "class_distribution",
                        "predictor_group_count",
                        "duplicate_group_count",
                        "license",
                        "raw_source_sha256",
                        "processed_feature_sha256",
                        "target_artifact_sha256",
                        "quality_report_sha256",
                        "metadata_evidence_sha256",
                        "five_split_identities",
                        "coverage_tags",
                    )
                },
                selection_rationale=rationale,
                v00_model_compatibility="PASS_REGISTRY_CAPABILITY",
            )
        )
    selected_ids = tuple(item.dataset_id for item in selected_data)
    excluded_data: list[ExcludedDataset] = []
    for dataset_id in sorted(set(DATASET_IDS) - set(selected_ids)):
        excluded_candidate = eligible_by_id.get(dataset_id)
        if excluded_candidate is None:
            # A dataset failing source/split checks is not silently called an eligible exclusion.
            raise PlanningError(
                f"BLOCKED: excluded dataset {dataset_id} lacks complete metadata/split evidence"
            )
        best_competitor = min(
            selected_data,
            key=lambda selected: (
                sum(
                    left != right
                    for left, right in zip(
                        (
                            "binary" if excluded_candidate["class_count"] == 2 else "multiclass",
                            _schema_family(excluded_candidate),
                            "small"
                            if excluded_candidate["row_count"]
                            <= config.selection.small_dataset_max_rows
                            else "medium",
                            "balanced"
                            if max(excluded_candidate["class_distribution"].values())
                            / excluded_candidate["row_count"]
                            <= config.selection.balanced_max_class_fraction
                            else "imbalanced",
                        ),
                        (
                            selected.task_type,
                            "numeric_only"
                            if selected.categorical_feature_count == 0
                            else "categorical_only"
                            if selected.numeric_feature_count == 0
                            else "mixed",
                            "small"
                            if selected.row_count <= config.selection.small_dataset_max_rows
                            else "medium",
                            "balanced"
                            if max(selected.class_distribution.values()) / selected.row_count
                            <= config.selection.balanced_max_class_fraction
                            else "imbalanced",
                        ),
                    )
                ),
                selected.dataset_id,
            ),
        )
        excluded_data.append(
            ExcludedDataset(
                dataset_id=dataset_id,
                dataset_name=item["dataset_name"],
                eligible=True,
                exclusion_reason=(
                    "Valid registered candidate, but deterministic maximum-coverage selection "
                    "favors complementary strata; nearest selected metadata profile is "
                    f"dataset {best_competitor.dataset_id}. No outcome or transformation "
                    "result was used."
                ),
                competing_selected_dataset_id=best_competitor.dataset_id,
                metadata_evidence_sha256=item["metadata_evidence_sha256"],
            )
        )

    seeds = tuple(int(seed) for seed in split_config.seeds[:3])
    if seeds != (1729, 2718, 31415):
        raise PlanningError("REPAIR_REQUIRED: canonical first three seeds changed")
    selected_splits = [
        {
            "dataset_id": dataset.dataset_id,
            "seed": split.seed,
            "assignment_sha256": split.assignment_sha256,
            "logical_assignment_sha256": split.logical_assignment_sha256,
            "manifest_sha256": split.manifest_sha256,
        }
        for dataset in selected_data
        for split in dataset.five_split_identities
        if split.seed in seeds
    ]
    if len(selected_splits) != 24:
        raise PlanningError(
            "selected dataset-seed plan does not include 24 validated split identities"
        )
    seed_selection = PilotSeedSelection(
        seeds=seeds,
        source_config_sha256=_sha_file(project, config.seeds.source_config),
        canonical_order_preserved=True,
        dataset_seed_count=24,
        split_identity_sha256=sha256_canonical_json(selected_splits),
    )

    # Only sanitized applicability evidence is read; no view is fitted or materialized.
    applicability, transformation_config_hash, transformation_impl_hash = (
        _collect_view_applicability(
            project,
            config,
            selected_ids,
            seeds,
            eligible_by_id,
            transformation_config,
        )
    )
    view_registry = _build_view_registry(transformation_config)
    if len(applicability) != 264:
        raise PlanningError("the complete 8 x 3 x 11 applicability matrix is not present")

    model_matrix, registry_decisions = _build_model_matrix(
        project,
        config,
        experiment_rows,
        model_specs,
        adapter_inventory,
        {
            item.id: sha256_canonical_json(
                [
                    record.model_dump(mode="json")
                    for record in adapter_inventory.records
                    if record.model_id == item.id
                ]
            )
            for item in model_specs
        },
        model_adapter_raw,
    )
    if tuple(item.model_id for item in model_matrix) != MODEL_ORDER:
        raise PlanningError("frozen model order changed")
    policies_by_model = {item.model_id: item for item in model_matrix}
    model_specs_by_id = {item.id: item for item in model_specs}
    metric_policy_data = config.metrics.model_dump(mode="json")
    metric_hash = sha256_canonical_json(metric_policy_data)
    tolerance_hash = sha256_canonical_json(
        {
            "non_rank_equivalence": metric_policy_data["non_rank_equivalence"],
            "probability": metric_policy_data["probability"],
            "auroc_tie_tolerance": metric_policy_data["auroc"]["numerical_tie_tolerance"],
        }
    )
    implementation_hashes = {
        relative: _canonical_source_digest(project, relative)
        for relative in _PROTOCOL_SOURCE_PATHS
        if relative
        not in {
            "src/schemaguard/experiments/pilot_contracts.py",
            "src/schemaguard/experiments/pilot_planning.py",
            "src/schemaguard/experiments/pilot_validation.py",
            "scripts/freeze_pilot_protocol.py",
            "scripts/validate_pilot_protocol.py",
        }
    }
    for relative in _ADAPTER_MODULES.values():
        implementation_hashes[relative] = _canonical_source_digest(project, relative)
    implementation_hashes.update(
        {
            relative: _canonical_source_digest(project, relative)
            for relative in _PROTOCOL_SOURCE_PATHS
            if relative
            in {
                "src/schemaguard/experiments/pilot_contracts.py",
                "src/schemaguard/experiments/pilot_planning.py",
                "src/schemaguard/experiments/pilot_validation.py",
                "scripts/freeze_pilot_protocol.py",
                "scripts/validate_pilot_protocol.py",
            }
        }
    )
    source_implementation_hash = sha256_canonical_json(dict(sorted(implementation_hashes.items())))
    dependency_lock_hash = _sha_file(project, config.protocol.dependency_lock)

    initial_source_hashes = _all_source_hashes(project, config)
    for dataset in selected_data:
        dataset_id = dataset.dataset_id
        raw_manifest_relative = f"data/raw/openml/{dataset_id}/source_manifest.json"
        raw_manifest = _read_json(_relative_path(project, raw_manifest_relative))
        relative_raw_data = str(raw_manifest["raw_relative_path"]).replace("\\", "/")
        selected_paths = (
            raw_manifest_relative,
            f"data/{relative_raw_data}",
            f"data/processed/openml/{dataset_id}/data_manifest.json",
            f"data/processed/openml/{dataset_id}/schema.json",
            f"data/processed/openml/{dataset_id}/quality_report.json",
            f"data/processed/openml/{dataset_id}/features.parquet",
            f"data/processed/openml/{dataset_id}/targets.parquet",
        )
        for relative in selected_paths:
            initial_source_hashes[relative] = _sha_file(project, relative)
    # Keep source_hashes as a path-keyed evidence map. Derived identities are separately
    # bound into condition IDs and the protocol fields, not disguised as file paths.
    source_hashes = dict(sorted(initial_source_hashes.items()))
    runtime_estimate = _runtime_estimate(
        project, config, tuple(selected_data), applicability, model_matrix
    )
    view_na = [item for item in applicability if item.applicability == "NOT_APPLICABLE"]
    applicable_views = [item for item in applicability if item.applicability == "APPLICABLE"]
    protocol_payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config.protocol.protocol_id,
        "protocol_name": config.protocol.protocol_name,
        "protocol_version": config.protocol.protocol_version,
        "execution_authorized": False,
        "offline": True,
        "source_commit": config.protocol.source_commit,
        "source_hashes": source_hashes,
        "dataset_selection_policy": config.selection.model_dump(mode="json"),
        "selected_datasets": [item.model_dump(mode="json") for item in selected_data],
        "excluded_datasets": [item.model_dump(mode="json") for item in excluded_data],
        "seed_selection": seed_selection.model_dump(mode="json"),
        "view_registry": [item.model_dump(mode="json") for item in view_registry],
        "model_matrix": [item.model_dump(mode="json") for item in model_matrix],
        "registry_parameter_decisions": list(registry_decisions),
        "metric_policy": config.metrics.model_dump(mode="json"),
        "leakage_policy": config.leakage.model_dump(mode="json"),
        "decision_policy": config.decisions.model_dump(mode="json"),
        "resource_policy": config.resources.model_dump(mode="json"),
        "retry_policy": config.retry.model_dump(mode="json"),
        "runtime_estimate": runtime_estimate.model_dump(mode="json"),
        "summary": {
            "selected_dataset_count": len(selected_data),
            "excluded_dataset_count": len(excluded_data),
            "seed_count": len(seeds),
            "dataset_seed_count": len(selected_splits),
            "view_tuple_count": len(applicability),
            "applicable_view_count": len(applicable_views),
            "not_applicable_view_count": len(view_na),
            "maximum_model_condition_count": 1320,
            "actual_model_condition_count": len(applicable_views) * len(model_matrix),
            "cpu_condition_count": len(applicable_views) * 3,
            "cuda_condition_count": len(applicable_views) * 2,
        },
    }
    protocol_sha = sha256_canonical_json(protocol_payload)

    conditions: list[PlannedCondition] = []
    na_tuples: list[NotApplicableConditionTuple] = []
    view_by_tuple = {(item.dataset_id, item.seed, item.view_id): item for item in applicability}
    data_by_id = {item.dataset_id: item for item in selected_data}
    for dataset_id in selected_ids:
        dataset = data_by_id[dataset_id]
        for seed in seeds:
            split = next(item for item in dataset.five_split_identities if item.seed == seed)
            for view_id in VIEW_ORDER:
                view = view_by_tuple[(dataset_id, seed, view_id)]
                view_evidence_hash = sha256_canonical_json(view.model_dump(mode="json"))
                if view.applicability == "NOT_APPLICABLE":
                    na_tuples.append(
                        NotApplicableConditionTuple(
                            dataset_id=dataset_id,
                            seed=seed,
                            view_id=view_id,
                            reason_code=str(view.reason_code),
                            evidence_sha256=view_evidence_hash,
                            transformation_inventory_record_sha256=view.transformation_inventory_record_sha256,
                        )
                    )
                    continue
                for model_id in MODEL_ORDER:
                    model = policies_by_model[model_id]
                    spec = model_specs_by_id[model_id]
                    effective_parameters = _effective_parameters(spec, dataset.class_count, seed)
                    certificate_hash = sha256_canonical_json(list(view.certificate_identities))
                    components = ConditionIdentityComponents(
                        protocol_version=config.protocol.protocol_version,
                        pilot_protocol_sha256=protocol_sha,
                        dataset_id=dataset_id,
                        raw_source_sha256=dataset.raw_source_sha256,
                        processed_feature_sha256=dataset.processed_feature_sha256,
                        target_artifact_sha256=dataset.target_artifact_sha256,
                        split_strategy=SPLIT_STRATEGY,
                        seed=seed,
                        split_assignment_sha256=split.assignment_sha256,
                        split_logical_sha256=split.logical_assignment_sha256,
                        view_id=view_id,
                        transformation_config_sha256=transformation_config_hash,
                        transformation_implementation_sha256=transformation_impl_hash,
                        certificate_identity_sha256=certificate_hash,
                        model_id=model_id,
                        package_name=model.package,
                        package_version=model.version,
                        model_config_sha256=model.model_config_sha256,
                        complete_parameters_sha256=sha256_canonical_json(effective_parameters),
                        preprocessing_sha256=model.preprocessing_sha256,
                        checkpoint_identifier=model.checkpoint_identifier,
                        checkpoint_sha256=model.checkpoint_sha256,
                        device_policy=model.device,
                        precision_policy_sha256=sha256_canonical_json(model.precision_policy),
                        batch_policy_sha256=sha256_canonical_json(model.batch_policy),
                        dependency_lock_sha256=dependency_lock_hash,
                        source_implementation_sha256=source_implementation_hash,
                        metric_policy_sha256=metric_hash,
                        numerical_tolerance_policy_sha256=tolerance_hash,
                        offline_policy=True,
                    )
                    conditions.append(
                        PlannedCondition(
                            condition_id=canonical_identity_hash(components),
                            identity=components,
                            dataset_id=dataset_id,
                            seed=seed,
                            view_id=view_id,
                            model_id=model_id,
                            device=model.device,
                        )
                    )
    conditions.sort(
        key=lambda item: (
            item.dataset_id,
            item.seed,
            item.view_id,
            MODEL_ORDER.index(item.model_id),
        )
    )
    count_by_dataset = Counter(str(item.dataset_id) for item in conditions)
    count_by_seed = Counter(str(item.seed) for item in conditions)
    count_by_view = Counter(item.view_id for item in conditions)
    count_by_model = Counter(item.model_id for item in conditions)
    inventory_payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha,
        "records": [item.model_dump(mode="json") for item in conditions],
        "not_applicable_views": [item.model_dump(mode="json") for item in na_tuples],
        "maximum_possible_view_tuples": 264,
        "maximum_possible_model_conditions": 1320,
        "selected_dataset_ids": list(selected_ids),
        "selected_seeds": list(seeds),
        "actual_view_tuple_count": len(applicability),
        "applicable_view_tuple_count": len(applicable_views),
        "not_applicable_view_count": len(view_na),
        "actual_condition_count": len(conditions),
        "cpu_condition_count": sum(item.device == "cpu" for item in conditions),
        "cuda_condition_count": sum(item.device == "cuda" for item in conditions),
        "counts_by_dataset": dict(sorted(count_by_dataset.items(), key=lambda item: int(item[0]))),
        "counts_by_seed": dict(sorted(count_by_seed.items(), key=lambda item: int(item[0]))),
        "counts_by_view": {view_id: count_by_view.get(view_id, 0) for view_id in VIEW_ORDER},
        "counts_by_model": {model_id: count_by_model.get(model_id, 0) for model_id in MODEL_ORDER},
        "source_hashes": source_hashes,
        "inventory_sha256": "0" * 64,
    }
    inventory_hash = sha256_canonical_json(
        {key: value for key, value in inventory_payload.items() if key != "inventory_sha256"}
    )
    inventory_payload["inventory_sha256"] = inventory_hash
    inventory = PilotConditionInventory.model_validate(inventory_payload)
    if protocol_sha != sha256_canonical_json(protocol_payload):
        raise PlanningError(
            "protocol definition hash changed while condition identities were built"
        )
    protocol_payload["condition_inventory_sha256"] = inventory.inventory_sha256
    protocol_payload["protocol_sha256"] = protocol_sha
    protocol_artifact = PilotProtocolArtifact.model_validate(protocol_payload)
    # Independent smoke evidence is identity/provenance only; it never influences dataset selection.
    if smoke_evidence.get("verdict") != "VERIFIED_PASS":
        raise PlanningError("accepted smoke provenance changed during plan construction")
    return protocol_artifact, inventory


def attach_inventory_hash(
    protocol: PilotProtocolArtifact, inventory: PilotConditionInventory
) -> PilotProtocolArtifact:
    """Return the frozen protocol object bound to its separate condition inventory."""
    if inventory.protocol_sha256 != protocol.protocol_sha256:
        raise PlanningError("condition inventory references a different pilot protocol")
    return protocol.model_copy(update={"condition_inventory_sha256": inventory.inventory_sha256})


__all__ = [
    "DATASET_IDS",
    "PlanningError",
    "attach_inventory_hash",
    "build_staged_schedule",
    "build_pilot_plan",
    "load_pilot_config",
    "select_candidate_ids",
]
