"""Strict configuration, label-sealed preflight, and deterministic smoke planning."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..cache.keys import dependency_lock_hash
from ..models.adapters.contracts import FOUNDATION_CHECKPOINTS
from ..models.registry import build_parameters, load_model_registry
from ..transformations.certificates import certificate_identity_hash
from ..transformations.serialization import (
    read_certificate,
    read_manifest,
    validate_manifest_directory,
)
from ..utils.hashing import hash_dataframe_logically, sha256_canonical_json, sha256_file
from ..utils.io import atomic_write_json
from .contracts import (
    FOUNDATION_MODELS,
    MODEL_ORDER,
    VIEW_ORDER,
    AllPartition,
    PlanEnvelope,
    RegistryParameterDecision,
    SmokeCondition,
    SmokePlan,
    SmokeView,
    condition_identity,
)

ALL_PARTITIONS: tuple[AllPartition, ...] = ("train", "calibration", "test")

SOURCE_PATHS = (
    "src/schemaguard/experiments/__init__.py",
    "src/schemaguard/experiments/contracts.py",
    "src/schemaguard/experiments/planning.py",
    "src/schemaguard/experiments/execution.py",
    "src/schemaguard/experiments/evaluation.py",
    "src/schemaguard/experiments/evidence.py",
    "scripts/run_smoke_experiment.py",
    "scripts/validate_smoke_experiment.py",
    "src/schemaguard/models/registry.py",
    "src/schemaguard/models/adapters/logistic_regression_adapter.py",
    "src/schemaguard/models/adapters/catboost_adapter.py",
    "src/schemaguard/models/adapters/xgboost_adapter.py",
    "src/schemaguard/models/adapters/tabpfn_adapter.py",
    "src/schemaguard/models/adapters/tabicl_adapter.py",
    "src/schemaguard/models/adapters/base.py",
    "src/schemaguard/models/adapters/contracts.py",
    "src/schemaguard/models/adapters/factory.py",
    "src/schemaguard/models/adapters/preprocessing.py",
    "src/schemaguard/models/adapters/probability.py",
    "src/schemaguard/models/adapters/resources.py",
    "src/schemaguard/transformations/certificates.py",
    "src/schemaguard/transformations/serialization.py",
    "src/schemaguard/transformations/registry.py",
    "src/schemaguard/transformations/implementation.py",
    "src/schemaguard/cache/contracts.py",
    "src/schemaguard/cache/keys.py",
    "src/schemaguard/cache/locks.py",
    "src/schemaguard/cache/store.py",
    "src/schemaguard/cache/index.py",
    "src/schemaguard/runner/contracts.py",
    "src/schemaguard/runner/plan.py",
    "src/schemaguard/runner/resources.py",
    "src/schemaguard/runner/scheduler.py",
    "src/schemaguard/runner/state.py",
    "src/schemaguard/runner/worker.py",
    "src/schemaguard/utils/hashing.py",
    "src/schemaguard/utils/io.py",
    "src/schemaguard/artifact_contracts.py",
    "configs/runtime/smoke_experiment.yaml",
    "configs/runtime/cache_scheduler.yaml",
    "configs/runtime/model_adapters.yaml",
    "configs/runtime/transformation_engine.yaml",
    "configs/experiment_registry.yaml",
    "configs/baselines/data_foundation.json",
    "artifacts/handoff/smoke_experiment_decisions.md",
)


def reconcile_registry_parameters(
    model_id: str,
    experiment_parameters: dict[str, Any],
    runtime_parameters: dict[str, Any],
    checkpoint: str | None,
) -> list[RegistryParameterDecision]:
    """Translate only the two pre-documented, exact YAML representation differences."""
    observed = dict(experiment_parameters)
    decisions: list[RegistryParameterDecision] = []
    if (
        model_id == "CAT-1.2"
        and observed.get("bootstrap_type") is False
        and runtime_parameters.get("bootstrap_type") == "No"
    ):
        observed["bootstrap_type"] = "No"
        decisions.append("catboost_yaml_no_scalar")
    if (
        model_id == "TICL2-2.2"
        and "checkpoint_version" not in observed
        and checkpoint is not None
        and runtime_parameters.get("checkpoint_version") == checkpoint
    ):
        observed["checkpoint_version"] = checkpoint
        decisions.append("tabicl_checkpoint_version_from_checkpoint_field")
    if observed != runtime_parameters:
        raise ValueError(f"frozen registry parameter mismatch for {model_id}")
    return decisions


def index_experiment_registry_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate the exact model set, then expose it in frozen canonical order."""
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_id = row.get("id")
        if not isinstance(model_id, str) or model_id in indexed:
            raise ValueError("experiment registry has a missing or duplicate model identifier")
        indexed[model_id] = row
    if set(indexed) != set(MODEL_ORDER):
        raise ValueError("experiment registry does not contain the exact frozen model set")
    return {model_id: indexed[model_id] for model_id in MODEL_ORDER}


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ViewConfig(StrictConfig):
    view_id: Literal["V00", "V01"]
    view_name: Literal["identity", "numeric_affine_units"]


class ExecutionConfig(StrictConfig):
    network_enabled: Literal[False]
    cpu_workers: Literal[2]
    gpu_workers: Literal[1]
    resume: Literal[True]
    minimum_gpu_headroom_mib: int = Field(ge=512)
    maximum_vram_mib: int = Field(gt=0, le=3600)
    maximum_ram_mib: int = Field(gt=0, le=28672)
    cpu_timeout_seconds: int = Field(gt=0)
    cuda_timeout_seconds: int = Field(gt=0)


class EvaluationConfig(StrictConfig):
    log_loss_epsilon: float = Field(gt=0, lt=0.01, allow_inf_nan=False)
    ece_bins: int = Field(ge=2, le=100)
    ece_binning_rule: Literal["equal_width_max_probability_confidence"]
    probability_sum_tolerance: float = Field(gt=0, le=1.0e-4, allow_inf_nan=False)
    reconstruction_rtol: float = Field(ge=0, le=1.0e-6, allow_inf_nan=False)
    reconstruction_atol: float = Field(ge=0, le=1.0e-6, allow_inf_nan=False)
    primary_metric: Literal["brier_score"]
    label_access_boundary: Literal["after_all_predictions_complete"]


class OutputPaths(StrictConfig):
    runs: Literal["results/smoke/runs"]
    predictions: Literal["results/smoke/predictions"]
    metrics: Literal["results/smoke/metrics"]
    runtime: Literal["results/smoke/runtime"]


class SmokeConfig(StrictConfig):
    schema_version: Literal[1]
    protocol_version: Literal["schema_guard_smoke_v1"]
    dataset_id: Literal[1464]
    dataset_version: Literal["openml_file_1586225"]
    seed: Literal[1729]
    split_strategy: Literal["stratified_group_5fold_v1"]
    baseline_config: str
    dependency_lock: Literal["uv.lock"]
    model_registry: str
    experiment_registry: str
    adapter_config: str
    transformation_config: str
    scheduler_config: str
    views: list[ViewConfig] = Field(min_length=2, max_length=2)
    cpu_models: tuple[Literal["LR-1.9", "CAT-1.2", "XGB-3.4"], ...]
    cuda_models: tuple[Literal["TPFN3-8.5", "TICL2-2.2"], ...]
    execution: ExecutionConfig
    evaluation: EvaluationConfig
    output_paths: OutputPaths

    @model_validator(mode="after")
    def validate_frozen_scope(self) -> SmokeConfig:
        if tuple(item.view_id for item in self.views) != VIEW_ORDER:
            raise ValueError("smoke view matrix must remain V00, V01 in that order")
        if tuple(self.cpu_models) != MODEL_ORDER[:3] or tuple(self.cuda_models) != MODEL_ORDER[3:]:
            raise ValueError("smoke model/device matrix differs from the frozen ten conditions")
        for value in (
            self.model_registry,
            self.experiment_registry,
            self.adapter_config,
            self.transformation_config,
            self.scheduler_config,
            self.baseline_config,
        ):
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("configuration references must be portable relative paths")
        return self


def load_smoke_config(path: str | Path) -> SmokeConfig:
    """Load the semantic smoke config and reject unknown or missing keys."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload: Any = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("smoke experiment config must be a YAML mapping")
    try:
        return SmokeConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid smoke experiment config: {exc}") from exc


def _current_commit(root: Path) -> str:
    output = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if len(output) != 40 or any(item not in "0123456789abcdef" for item in output):
        raise ValueError("cannot establish the exact implementation commit")
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path.as_posix()}")
    return value


def _assignment(root: Path, dataset_id: int, seed: int):
    import pandas as pd

    path = (
        root
        / "data/splits/openml"
        / str(dataset_id)
        / "stratified_group_5fold_v1"
        / f"seed_{seed}"
        / "assignments.parquet"
    )
    return pd.read_parquet(path), path


def _view_inventory_record(root: Path, dataset_id: int, seed: int, view_id: str) -> dict[str, Any]:
    inventory = _read_json(root / "artifacts/handoff/transformation_inventory.json")
    records = inventory.get("records")
    if not isinstance(records, list):
        raise ValueError("transformation inventory has no records")
    found = [
        row
        for row in records
        if row.get("dataset_id") == dataset_id
        and row.get("seed") == seed
        and row.get("view_id") == view_id
    ]
    if len(found) != 1 or found[0].get("status") != "PASS":
        raise ValueError(f"frozen transformation evidence missing or unsuccessful for {view_id}")
    return found[0]


def _view_files(root: Path, seed: int, view: ViewConfig, assignment):
    import pandas as pd

    partition_column = "partition" if "partition" in assignment.columns else "split"
    if view.view_id == "V00":
        feature_path = root / "data/processed/openml/1464/features.parquet"
        frame = pd.read_parquet(feature_path)
        indexed = frame.set_index("__sg_row_id", drop=False)
        partitions = {}
        for name in ALL_PARTITIONS:
            row_ids = assignment.loc[assignment[partition_column] == name, "__sg_row_id"].tolist()
            partitions[name] = indexed.loc[row_ids].reset_index(drop=True)
        return (
            partitions,
            [(feature_path.relative_to(root).as_posix(), sha256_file(feature_path))],
            {name: sha256_file(feature_path) for name in ALL_PARTITIONS},
            {name: None for name in ALL_PARTITIONS},
            None,
        )

    directory = (
        root
        / "data/transformed/openml/1464/stratified_group_5fold_v1"
        / f"seed_{seed}"
        / view.view_name
    )
    manifest = read_manifest(directory / "manifest.json")
    if (
        manifest.dataset_id != 1464
        or manifest.seed != seed
        or manifest.view_id != view.view_id
        or manifest.view_name != view.view_name
        or manifest.split_strategy != "stratified_group_5fold_v1"
        or manifest.manifest_status != "PASS"
    ):
        raise ValueError("V01 transformation manifest identity or status is invalid")
    partitions = {
        name: pd.read_parquet(directory / manifest.partition_feature_paths[name])
        for name in ALL_PARTITIONS
    }
    file_hashes: list[tuple[str, str]] = []
    feature_file_hashes: dict[AllPartition, str] = {}
    for name in ALL_PARTITIONS:
        feature_path = directory / manifest.partition_feature_paths[name]
        cert_path = directory / manifest.certificate_paths[name]
        if sha256_file(feature_path) != manifest.transformed_feature_hashes[name]:
            raise ValueError(f"V01 {name} feature artifact hash differs from its manifest")
        if sha256_file(cert_path) != manifest.certificate_hashes[name]:
            raise ValueError(f"V01 {name} certificate hash differs from its manifest")
        file_hashes.extend(
            (
                (feature_path.relative_to(root).as_posix(), sha256_file(feature_path)),
                (cert_path.relative_to(root).as_posix(), sha256_file(cert_path)),
            )
        )
        feature_file_hashes[name] = sha256_file(feature_path)
    file_hashes.append(
        (
            (directory / "manifest.json").relative_to(root).as_posix(),
            sha256_file(directory / "manifest.json"),
        )
    )
    certificate_hashes = {
        name: sha256_file(directory / manifest.certificate_paths[name]) for name in ALL_PARTITIONS
    }
    manifest_sha = sha256_file(directory / "manifest.json")
    return partitions, file_hashes, feature_file_hashes, certificate_hashes, manifest_sha


def _source_identity(root: Path) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"smoke implementation dependency is missing: {relative}")
        from ..utils.hashing import canonical_source_hash

        hashes[relative] = canonical_source_hash(path)
    return sha256_canonical_json(hashes), hashes


def build_smoke_plan(
    root: str | Path,
    config_path: str | Path,
    *,
    source_commit_override: str | None = None,
) -> SmokePlan:
    """Construct the complete deterministic plan without opening target values."""
    import pandas as pd
    import yaml

    project = Path(root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project / config_file
    config = load_smoke_config(config_file)
    baseline = _read_json(project / "configs/baselines/data_foundation.json")
    expected_artifacts = baseline["dataset"]["processed_artifact_sha256"]
    if (
        expected_artifacts.get("features.parquet") is None
        or expected_artifacts.get("targets.parquet") is None
    ):
        raise ValueError("data-foundation baseline lacks required source hashes")

    features_path = project / "data/processed/openml/1464/features.parquet"
    features = pd.read_parquet(features_path)
    if sha256_file(features_path) != expected_artifacts["features.parquet"]:
        raise ValueError("processed feature artifact differs from protected baseline")
    assignment, assignment_path = _assignment(project, config.dataset_id, config.seed)
    assignment_file_sha = sha256_file(assignment_path)
    assignment_logical_sha = hash_dataframe_logically(assignment)
    partition_column = "partition" if "partition" in assignment.columns else "split"
    expected_rows = {"train": 449, "calibration": 150, "test": 149}
    observed_rows = {
        name: int((assignment[partition_column] == name).sum())
        for name in ("train", "calibration", "test")
    }
    if observed_rows != expected_rows or sum(observed_rows.values()) != len(assignment):
        raise ValueError("existing grouped split sizes or partition coverage changed")

    transformation_inventory_path = project / "artifacts/handoff/transformation_inventory.json"
    transformation_inventory_sha = sha256_file(transformation_inventory_path)
    view_records: dict[str, dict[str, Any]] = {}
    view_models: list[SmokeView] = []
    for view in config.views:
        evidence = _view_inventory_record(project, config.dataset_id, config.seed, view.view_id)
        (
            partitions,
            _,
            feature_file_hashes,
            certificate_file_hashes,
            manifest_sha,
        ) = _view_files(project, config.seed, view, assignment)
        if view.view_id == "V00" and evidence.get("source_hash") != hash_dataframe_logically(
            features
        ):
            raise ValueError("V00 source hash differs from the protected feature artifact")
        cert_ids = list(evidence.get("certificate_ids", []))
        if len(cert_ids) != 3:
            raise ValueError(
                f"transformation evidence has incomplete partition certificates: {view.view_id}"
            )
        if view.view_id == "V01":
            manifest = read_manifest(
                project
                / "data/transformed/openml/1464/stratified_group_5fold_v1"
                / f"seed_{config.seed}"
                / view.view_name
                / "manifest.json"
            )
            if (
                manifest.implementation_hash
                != "00548880e1de331d8b6540778b374a17e61838b3bd0f7e81590ff51ab595b034"
            ):
                raise ValueError("V01 implementation identity differs from the accepted repair")
            if (
                manifest.cache_identity
                != "9191149f3e8097874cb55ef2e19b5409373bed4c1d67667166e9512246b70635"
            ):
                raise ValueError("V01 cache identity differs from the accepted repair")
            actual_cert_ids = [
                certificate_identity_hash(
                    read_certificate(
                        project
                        / "data/transformed/openml/1464/stratified_group_5fold_v1"
                        / f"seed_{config.seed}"
                        / view.view_name
                        / manifest.certificate_paths[name]
                    )
                )
                for name in ("train", "calibration", "test")
            ]
            if actual_cert_ids != cert_ids:
                raise ValueError("V01 certificate identities differ from the frozen inventory")
        row_id_hashes: dict[AllPartition, str] = {}
        row_counts: dict[AllPartition, int] = {}
        partition_feature_hashes: dict[AllPartition, str] = {}
        for partition_name in ALL_PARTITIONS:
            expected_ids = assignment.loc[
                assignment[partition_column] == partition_name, "__sg_row_id"
            ].tolist()
            observed_ids = partitions[partition_name]["__sg_row_id"].tolist()
            if len(observed_ids) != len(set(observed_ids)) or observed_ids != expected_ids:
                raise ValueError(
                    f"{view.view_id} {partition_name} rows do not match the frozen split"
                )
            row_id_hashes[partition_name] = sha256_canonical_json(observed_ids)
            row_counts[partition_name] = len(observed_ids)
            partition_feature_hashes[partition_name] = hash_dataframe_logically(
                partitions[partition_name]
            )
        view_features_sha = sha256_canonical_json(partition_feature_hashes)
        view_record = SmokeView(
            view_id=view.view_id,
            view_name=view.view_name,
            source_hash=evidence["source_hash"],
            output_hash=evidence["output_hash"],
            certificate_sha256=sha256_canonical_json(cert_ids),
            certificate_ids=cert_ids,
            view_features_sha256=view_features_sha,
            partition_feature_sha256=partition_feature_hashes,
            partition_feature_file_sha256=feature_file_hashes,
            partition_certificate_file_sha256=certificate_file_hashes,
            manifest_sha256=manifest_sha,
            partition_row_id_sha256=row_id_hashes,
            partition_rows=row_counts,
        )
        view_models.append(view_record)
        view_records[view.view_id] = {"model": view_record}

    registry_path = project / config.model_registry
    experiment_path = project / config.experiment_registry
    model_specs = load_model_registry(registry_path)
    experiment_payload = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    experiment_rows = index_experiment_registry_rows(experiment_payload["models"])
    adapter_config = (
        _read_json(project / config.adapter_config)
        if config.adapter_config.endswith(".json")
        else None
    )
    if adapter_config is None:
        adapter_config = yaml.safe_load(
            (project / config.adapter_config).read_text(encoding="utf-8")
        )
    checkpoint_config = adapter_config.get("checkpoints", {})
    source_identity, _ = _source_identity(project)
    lock_sha = dependency_lock_hash(project / "uv.lock")
    conditions: list[SmokeCondition] = []
    registry_parameter_decisions: list[RegistryParameterDecision] = []
    spec_by_id = {item.id: item for item in model_specs}
    for model_id in MODEL_ORDER:
        spec = spec_by_id[model_id]
        row = experiment_rows[model_id]
        for view in config.views:
            device: Literal["cpu", "cuda"] = "cuda" if model_id in FOUNDATION_MODELS else "cpu"
            for field, expected in (
                ("name", spec.class_path),
                ("package", spec.package),
                ("version", spec.expected_version),
                ("checkpoint", spec.checkpoint),
            ):
                if row.get(field) != expected:
                    raise ValueError(f"frozen experiment registry mismatch for {model_id}.{field}")
            if row.get("preprocessing") is None:
                raise ValueError(f"missing frozen preprocessing for {model_id}")
            for decision in reconcile_registry_parameters(
                model_id,
                row.get("parameters", {}),
                spec.parameters,
                row.get("checkpoint"),
            ):
                if decision not in registry_parameter_decisions:
                    registry_parameter_decisions.append(decision)
            parameters = build_parameters(spec, seed=config.seed, device=device, target_classes=2)
            model_spec_sha = sha256_canonical_json(
                {"model_spec": spec.model_dump(mode="json"), "preprocessing": row["preprocessing"]}
            )
            parameter_sha = sha256_canonical_json(parameters)
            checkpoint = checkpoint_config.get(model_id)
            checkpoint_identifier = checkpoint.get("identifier") if checkpoint else None
            checkpoint_sha = checkpoint.get("sha256") if checkpoint else None
            if model_id in FOUNDATION_MODELS:
                expected_checkpoint = FOUNDATION_CHECKPOINTS[model_id]
                if (checkpoint_identifier, checkpoint_sha) != expected_checkpoint:
                    raise ValueError(f"frozen checkpoint identity mismatch for {model_id}")
            view_record = view_records[view.view_id]["model"]
            condition_id = condition_identity(
                dataset_features_sha256=expected_artifacts["features.parquet"],
                target_artifact_sha256=expected_artifacts["targets.parquet"],
                model_id=model_id,
                package_version=spec.expected_version,
                view_id=view.view_id,
                device=device,
                precision_policy="auto" if model_id in FOUNDATION_MODELS else "native",
                seed=config.seed,
                model_spec_sha256=model_spec_sha,
                parameters_sha256=parameter_sha,
                view_certificate_sha256=view_record.certificate_sha256,
                view_features_sha256=view_record.view_features_sha256,
                split_sha256=assignment_file_sha,
                dependency_lock_sha256=lock_sha,
                source_implementation_sha256=source_identity,
                checkpoint_sha256=checkpoint_sha,
            )
            conditions.append(
                SmokeCondition(
                    condition_id=condition_id,
                    task_name=f"{model_id}__{view.view_id}",
                    model_id=model_id,
                    package_name=spec.package,
                    package_version=spec.expected_version,
                    dataset_features_sha256=expected_artifacts["features.parquet"],
                    target_artifact_sha256=expected_artifacts["targets.parquet"],
                    view_id=view.view_id,
                    device=device,
                    precision_policy="auto" if model_id in FOUNDATION_MODELS else "native",
                    seed=config.seed,
                    model_spec_sha256=model_spec_sha,
                    parameters_sha256=parameter_sha,
                    model_parameters=parameters,
                    preprocessing=row["preprocessing"],
                    checkpoint_identifier=checkpoint_identifier,
                    checkpoint_sha256=checkpoint_sha,
                    view_certificate_sha256=view_record.certificate_sha256,
                    view_features_sha256=view_record.view_features_sha256,
                    split_sha256=assignment_file_sha,
                    dependency_lock_sha256=lock_sha,
                    source_implementation_sha256=source_identity,
                )
            )

    plan = SmokePlan(
        protocol_version=config.protocol_version,
        source_commit=source_commit_override or _current_commit(project),
        code_identity_sha256=source_identity,
        dependency_lock_sha256=lock_sha,
        dataset_features_sha256=expected_artifacts["features.parquet"],
        dataset_source_sha256=baseline["dataset"]["raw_source_sha256"],
        target_artifact_sha256=expected_artifacts["targets.parquet"],
        assignment_sha256=assignment_file_sha,
        assignment_logical_sha256=assignment_logical_sha,
        transformation_inventory_sha256=transformation_inventory_sha,
        configuration_identity_sha256=sha256_canonical_json(config.model_dump(mode="json")),
        output_path_policy_sha256=sha256_canonical_json(
            config.output_paths.model_dump(mode="json")
        ),
        registry_parameter_decisions=registry_parameter_decisions,
        views=view_models,
        conditions=conditions,
        test_label_access_boundary=config.evaluation.label_access_boundary,
        primary_metric=config.evaluation.primary_metric,
        ece_binning_rule=config.evaluation.ece_binning_rule,
        log_loss_epsilon=config.evaluation.log_loss_epsilon,
        probability_sum_tolerance=config.evaluation.probability_sum_tolerance,
        reconstruction_rtol=config.evaluation.reconstruction_rtol,
        reconstruction_atol=config.evaluation.reconstruction_atol,
        ece_bins=config.evaluation.ece_bins,
        maximum_ram_mib=config.execution.maximum_ram_mib,
        maximum_vram_mib=config.execution.maximum_vram_mib,
        minimum_gpu_headroom_mib=config.execution.minimum_gpu_headroom_mib,
        cpu_timeout_seconds=config.execution.cpu_timeout_seconds,
        cuda_timeout_seconds=config.execution.cuda_timeout_seconds,
    )
    if tuple((item.model_id, item.view_id) for item in plan.conditions) != tuple(
        (model, view) for model in MODEL_ORDER for view in VIEW_ORDER
    ):
        raise ValueError("condition construction did not produce the canonical ten-condition order")
    return plan


def write_plan(root: str | Path, plan: SmokePlan) -> Path:
    """Atomically publish the portable plan under its content-addressed identity."""
    project = Path(root).resolve()
    destination = project / "results/smoke/runtime/plans" / f"{plan.plan_sha256}.json"
    envelope = PlanEnvelope(plan_sha256=plan.plan_sha256, plan=plan)
    if destination.exists():
        try:
            prior = PlanEnvelope.model_validate(_read_json(destination))
        except (OSError, json.JSONDecodeError, ValueError):
            quarantine_directory = destination.parent / "quarantine"
            quarantine_directory.mkdir(parents=True, exist_ok=True)
            quarantine = quarantine_directory / f"{destination.name}.invalid-{uuid.uuid4().hex}"
            destination.replace(quarantine)
        else:
            if prior.plan_sha256 != envelope.plan_sha256:
                raise ValueError("existing plan path contains a different valid plan")
            return destination
    atomic_write_json(destination, envelope.model_dump(mode="json"))
    return destination


def _verified_checkpoint_digest(
    path: Path,
    expected_sha256: str,
    prior: tuple[int, int, str] | None = None,
) -> tuple[str, tuple[int, int, str]]:
    """Reuse a verified digest only while the local file size and mtime are unchanged."""
    before = path.stat()
    fingerprint = (before.st_size, before.st_mtime_ns)
    if prior is not None and fingerprint == prior[:2]:
        observed = prior[2]
    else:
        observed = sha256_file(path)
        after = path.stat()
        if fingerprint != (after.st_size, after.st_mtime_ns):
            raise ValueError("checkpoint changed while its SHA-256 was being verified")
    if observed != expected_sha256:
        raise ValueError("checkpoint digest differs from its frozen identity")
    return observed, (fingerprint[0], fingerprint[1], observed)


def verify_runtime_sources_after_plan(root: str | Path, plan: SmokePlan) -> dict[str, str]:
    """After freezing the plan, verify opaque target bytes and local checkpoint identities."""
    project = Path(root).resolve()
    target_path = project / "data/processed/openml/1464/targets.parquet"
    if sha256_file(target_path) != plan.target_artifact_sha256:
        raise ValueError("opaque target source bytes differ from the frozen plan identity")
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_stats: dict[str, tuple[int, int, str]] = {}
    package_versions: dict[str, str] = {}
    for condition in plan.conditions:
        observed_version = package_versions.setdefault(
            condition.package_name, importlib.metadata.version(condition.package_name)
        )
        if observed_version != condition.package_version:
            raise ValueError(
                f"frozen dependency mismatch for {condition.model_id}: "
                f"expected {condition.package_version}, observed {observed_version}"
            )
        if condition.checkpoint_identifier is not None:
            checkpoint_path = project / condition.checkpoint_identifier
            observed, checkpoint_stats[condition.model_id] = _verified_checkpoint_digest(
                checkpoint_path,
                condition.checkpoint_sha256 or "",
                checkpoint_stats.get(condition.model_id),
            )
            if observed != condition.checkpoint_sha256:
                raise ValueError(
                    f"checkpoint digest differs from frozen plan for {condition.model_id}"
                )
            prior = checkpoint_hashes.setdefault(condition.model_id, observed)
            if prior != observed:
                raise ValueError(f"checkpoint identity is inconsistent for {condition.model_id}")
    return checkpoint_hashes


def validate_views_at_evaluation_boundary(root: str | Path) -> None:
    """Run the full certificate validator only after all predictions are complete."""
    project = Path(root).resolve()
    view_directory = (
        project
        / "data/transformed/openml/1464/stratified_group_5fold_v1"
        / "seed_1729/numeric_affine_units"
    )
    validate_manifest_directory(view_directory, project_root=project)


__all__ = [
    "SOURCE_PATHS",
    "SmokeConfig",
    "build_smoke_plan",
    "load_smoke_config",
    "reconcile_registry_parameters",
    "validate_views_at_evaluation_boundary",
    "verify_runtime_sources_after_plan",
    "write_plan",
]
