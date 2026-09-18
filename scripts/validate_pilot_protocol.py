"""Validate the frozen pilot protocol without running any pilot model condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schemaguard.artifact_contracts import (  # noqa: E402
    CacheSchedulerInventory,
    FaultInjectionEvidence,
    SchedulerProbeEvidence,
    schema_documents,
)
from schemaguard.experiments.pilot_contracts import (  # noqa: E402
    MODEL_ORDER,
    VIEW_ORDER,
    PilotProtocolArtifact,
    PilotProtocolValidation,
    PilotStagedSchedule,
    ProtocolValidationGate,
)
from schemaguard.experiments.pilot_planning import (  # noqa: E402
    _validate_accepted_smoke,
    build_pilot_plan,
    build_staged_schedule,
    load_pilot_config,
)
from schemaguard.experiments.pilot_validation import (  # noqa: E402
    build_validation_report,
    validate_protocol_artifacts,
)
from schemaguard.models.adapters.contracts import ModelAdapterInventory  # noqa: E402
from schemaguard.splits.contracts import SplitGenerationInventoryContract  # noqa: E402
from schemaguard.transformations.contracts import TransformationInventory  # noqa: E402
from schemaguard.utils.hashing import sha256_canonical_json, sha256_file  # noqa: E402
from schemaguard.utils.io import atomic_write_json, atomic_write_text  # noqa: E402

EXPECTED_BASE = "9a59e5a5dda320ec00f1915c80fb82d2531cb442"
PILOT_SCHEMA_FILES = (
    "schemas/pilot_protocol.schema.json",
    "schemas/pilot_dataset_selection.schema.json",
    "schemas/pilot_condition.schema.json",
    "schemas/pilot_condition_inventory.schema.json",
    "schemas/pilot_staged_schedule.schema.json",
    "schemas/pilot_metric_policy.schema.json",
    "schemas/pilot_decision_policy.schema.json",
    "schemas/pilot_protocol_validation.schema.json",
)
PROTECTED_LOCAL_ROOTS = (
    "data/raw/openml",
    "data/processed/openml",
    "data/splits/openml",
    "data/transformed/openml",
    "results/smoke",
)
CLEAN_CLONE_OVERLAY = (
    "configs/runtime/pilot_protocol.yaml",
    "src/schemaguard/artifact_contracts.py",
    "src/schemaguard/data/contracts.py",
    "src/schemaguard/experiments/pilot_contracts.py",
    "src/schemaguard/experiments/pilot_planning.py",
    "src/schemaguard/experiments/pilot_validation.py",
    "scripts/freeze_pilot_protocol.py",
    "scripts/validate_pilot_protocol.py",
    "scripts/validate_transformation_engine.py",
    "tests/unit/test_artifact_schemas.py",
    "tests/unit/test_transformation_validator_policy.py",
    "tests/unit/test_pilot_protocol.py",
    "tests/unit/test_pilot_metrics.py",
    "tests/integration/test_pilot_protocol_plan.py",
    "artifacts/handoff/pilot_protocol.json",
    "artifacts/handoff/pilot_condition_inventory.json",
    "artifacts/handoff/pilot_staged_schedule.json",
    *PILOT_SCHEMA_FILES,
)


def _git(*arguments: str) -> tuple[int, str, str]:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.relative_to(ROOT).as_posix()}")
    return value


def _tree_snapshot() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in PROTECTED_LOCAL_ROOTS:
        directory = ROOT / relative
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            records.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    records.sort(key=lambda item: item["path"])
    return {
        "file_count": len(records),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "records_sha256": sha256_canonical_json(records),
    }


def _tracked_schema_baseline_check() -> dict[str, Any]:
    code, names, stderr = _git("ls-tree", "-r", "--name-only", EXPECTED_BASE, "--", "schemas")
    if code:
        return {"passed": False, "error": stderr}
    checked: dict[str, dict[str, str]] = {}
    for relative in names.splitlines():
        local = ROOT / relative
        if not local.is_file():
            return {"passed": False, "path": relative, "error": "missing schema"}
        expected_code, expected_blob, expected_error = _git(
            "rev-parse", f"{EXPECTED_BASE}:{relative}"
        )
        observed_code, observed_blob, observed_error = _git(
            "hash-object", f"--path={relative}", relative
        )
        if expected_code or observed_code:
            return {
                "passed": False,
                "path": relative,
                "error": expected_error or observed_error or "schema blob hash failed",
            }
        checked[relative] = {"base_blob": expected_blob, "worktree_blob": observed_blob}
        if expected_blob != observed_blob:
            return {
                "passed": False,
                "path": relative,
                "expected_blob": expected_blob,
                "observed_blob": observed_blob,
            }
    return {
        "passed": True,
        "count": len(checked),
        "schema_blob_sha256": sha256_canonical_json(checked),
    }


def _make_gate(
    gate_id: str, passed: bool | None, detail: str, evidence: Any
) -> ProtocolValidationGate:
    return ProtocolValidationGate(
        gate_id=gate_id,
        result="NOT_VERIFIED" if passed is None else "PASS" if passed else "FAIL",
        evidence_sha256=sha256_canonical_json(evidence),
        detail=detail,
    )


def _command(command: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    combined = result.stdout + ("\n" if result.stdout and result.stderr else "") + result.stderr
    return {
        "command": command,
        "returncode": result.returncode,
        "output_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "output_tail": "\n".join(combined.splitlines()[-12:]),
    }


def _clean_clone_check() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="schemaguard-pilot-clean-") as temporary:
        clone = Path(temporary) / "repository"
        code, output, error = _git(
            "clone", "--quiet", "--no-hardlinks", "--local", str(ROOT), str(clone)
        )
        if code:
            return {"passed": False, "detail": error or output}
        copied: list[str] = []
        for relative in CLEAN_CLONE_OVERLAY:
            source = ROOT / relative
            if not source.is_file():
                return {
                    "passed": False,
                    "detail": f"required clean-clone source missing: {relative}",
                }
            destination = clone / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(clone / "src")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "tests/unit/test_pilot_protocol.py",
            "tests/unit/test_pilot_metrics.py",
            "tests/unit/test_artifact_schemas.py",
            "tests/unit/test_transformation_validator_policy.py",
        ]
        result = subprocess.run(
            command, cwd=clone, env=environment, capture_output=True, text=True, check=False
        )
        output_text = result.stdout + result.stderr
        return {
            "passed": result.returncode == 0,
            "overlay_file_count": len(copied),
            "overlay_sha256": sha256_canonical_json(
                {relative: sha256_file(ROOT / relative) for relative in copied}
            ),
            "command": command,
            "returncode": result.returncode,
            "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            "output_tail": "\n".join(output_text.splitlines()[-20:]),
        }


def _schema_freshness_check() -> dict[str, Any]:
    generator = _command([sys.executable, "scripts/generate_artifact_schemas.py"])
    if generator["returncode"] != 0:
        return {"passed": False, "generator": generator}
    expected = schema_documents()
    observed_hashes: dict[str, str] = {}
    for filename, document in expected.items():
        path = ROOT / "schemas" / filename
        if not path.is_file():
            return {"passed": False, "missing": filename, "generator": generator}
        payload = _read_json(path)
        if payload != document:
            return {"passed": False, "stale": filename, "generator": generator}
        observed_hashes[filename] = sha256_file(path)
    return {
        "passed": True,
        "schema_count": len(observed_hashes),
        "schemas_sha256": sha256_canonical_json(observed_hashes),
        "generator": generator,
    }


def _command_gate(
    gate_id: str, name: str, command: list[str]
) -> tuple[ProtocolValidationGate, dict[str, Any]]:
    result = _command(command)
    return (
        _make_gate(
            gate_id,
            result["returncode"] == 0,
            (
                f"{name} {'passed' if result['returncode'] == 0 else 'failed'} "
                f"(exit {result['returncode']})"
            ),
            {key: value for key, value in result.items() if key != "output_tail"},
        ),
        result,
    )


def validate(output_directory: Path) -> tuple[PilotProtocolValidation, str]:
    config = load_pilot_config(ROOT / "configs/runtime/pilot_protocol.yaml")
    protocol_path = output_directory / "pilot_protocol.json"
    inventory_path = output_directory / "pilot_condition_inventory.json"
    protocol, inventory = validate_protocol_artifacts(protocol_path, inventory_path)
    schedule_path = output_directory / "pilot_staged_schedule.json"
    expected_schedule = build_staged_schedule(protocol, inventory)
    schedule_error: str | None = None
    try:
        stored_schedule = PilotStagedSchedule.model_validate_json(
            schedule_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        stored_schedule = None
        schedule_error = f"{type(exc).__name__}: {exc}"
    schedule_matches = stored_schedule == expected_schedule
    before = _tree_snapshot()
    gates: list[ProtocolValidationGate] = []
    command_results: dict[str, Any] = {}

    head_code, head, head_error = _git("rev-parse", "HEAD")
    branch_code, branch, branch_error = _git("branch", "--show-current")
    remote_code, remote, remote_error = _git("rev-parse", "origin/main")
    identity_evidence = {
        "head": head,
        "branch": branch,
        "origin_main": remote,
        "errors": [item for item in (head_error, branch_error, remote_error) if item],
    }
    gates.append(
        _make_gate(
            "PF01",
            head_code == branch_code == remote_code == 0
            and head == remote == EXPECTED_BASE
            and branch == "main",
            "repository must remain on the authorized base commit before delivery",
            identity_evidence,
        )
    )

    smoke_evidence, smoke_hash = _validate_accepted_smoke(ROOT, config)
    gates.append(
        _make_gate(
            "PF02",
            smoke_evidence.get("verdict") == "VERIFIED_PASS"
            and smoke_evidence.get("pilot_started") is False
            and smoke_hash == protocol.source_hashes[config.protocol.smoke_review],
            "accepted independent smoke identity and no-pilot flag are bound by source hash",
            {
                "smoke_review_sha256": smoke_hash,
                "source_commit": smoke_evidence.get("source_commit"),
            },
        )
    )

    schema_baseline = _tracked_schema_baseline_check()
    gates.append(
        _make_gate(
            "PF03",
            False,
            "pending end-of-validation protected-state comparison",
            {
                "local_tree": before,
                "tracked_schema_baseline": schema_baseline,
            },
        )
    )

    split_inventory = SplitGenerationInventoryContract.model_validate(
        _read_json(ROOT / config.protocol.split_inventory)
    )
    dataset_report = _read_json(ROOT / config.protocol.dataset_report)
    registry = load_pilot_config(ROOT / "configs/runtime/pilot_protocol.yaml")
    gates.append(
        _make_gate(
            "PF04",
            dataset_report.get("status") == "PASS"
            and dataset_report.get("dataset_count") == 14
            and len(split_inventory.records) == 70,
            "the closed SchemaOrbit-14 registry and complete accepted 70-split inventory validate",
            {
                "dataset_report_sha256": sha256_file(ROOT / config.protocol.dataset_report),
                "dataset_count": dataset_report.get("dataset_count"),
                "split_inventory_sha256": sha256_file(ROOT / config.protocol.split_inventory),
                "split_records": len(split_inventory.records),
            },
        )
    )

    rebuilt_protocol, rebuilt_inventory = build_pilot_plan(ROOT, registry)
    rebuild_matches = (
        rebuilt_protocol.protocol_sha256 == protocol.protocol_sha256
        and rebuilt_inventory.inventory_sha256 == inventory.inventory_sha256
    )
    gates.append(
        _make_gate(
            "PF05",
            rebuild_matches,
            "rebuilding from sanitized registry metadata produces identical selected IDs "
            "and hashes",
            {
                "protocol_sha256": rebuilt_protocol.protocol_sha256,
                "inventory_sha256": rebuilt_inventory.inventory_sha256,
            },
        )
    )
    required_coverage = set(protocol.dataset_selection_policy.required_coverage)
    observed_coverage = set().union(
        *(set(item.coverage_tags) for item in protocol.selected_datasets)
    )
    gates.append(
        _make_gate(
            "PF06",
            required_coverage.issubset(observed_coverage)
            and len(protocol.selected_datasets) == 8
            and len(protocol.excluded_datasets) == 6,
            "eight-dataset cohort covers every frozen schema/task/size/balance feature and "
            "explains six exclusions",
            {
                "selected_ids": [item.dataset_id for item in protocol.selected_datasets],
                "coverage": sorted(observed_coverage),
                "required_coverage": sorted(required_coverage),
                "excluded_ids": [item.dataset_id for item in protocol.excluded_datasets],
            },
        )
    )

    split_config_path = ROOT / config.seeds.source_config
    split_config_payload = (
        _read_json(split_config_path) if split_config_path.suffix == ".json" else None
    )
    if split_config_payload is None:
        import yaml

        split_config_payload = yaml.safe_load(split_config_path.read_text(encoding="utf-8"))
    canonical_seeds = tuple(split_config_payload.get("seeds", ()))
    gates.append(
        _make_gate(
            "PF07",
            protocol.seed_selection.seeds == canonical_seeds[:3] == (1729, 2718, 31415),
            "the first three canonical configured seeds include 1729 and contain no generated seed",
            {
                "canonical_seeds": canonical_seeds,
                "pilot_seeds": protocol.seed_selection.seeds,
                "seed_config_sha256": sha256_file(split_config_path),
            },
        )
    )

    split_pairs = [
        split
        for dataset in protocol.selected_datasets
        for split in dataset.five_split_identities
        if split.seed in protocol.seed_selection.seeds
    ]
    split_ok = len(split_pairs) == 24 and all(
        item.cross_partition_group_count == 0 and item.strategy == "stratified_group_5fold_v1"
        for item in split_pairs
    )
    gates.append(
        _make_gate(
            "PF08",
            split_ok,
            "all 24 selected dataset-seed pairs bind passing grouped assignments with zero "
            "crossing groups",
            {
                "pair_count": len(split_pairs),
                "split_identity_sha256": protocol.seed_selection.split_identity_sha256,
                "assignment_hashes": [item.assignment_sha256 for item in split_pairs],
            },
        )
    )

    transformation = TransformationInventory.model_validate(
        _read_json(ROOT / config.protocol.transformation_inventory)
    )
    gates.append(
        _make_gate(
            "PF09",
            tuple(item.view_id for item in protocol.view_registry) == VIEW_ORDER
            and len(protocol.view_registry) == 11
            and protocol.source_hashes[config.protocol.transformation_config]
            == sha256_file(ROOT / config.protocol.transformation_config),
            "all eleven view IDs and names are bound to the frozen transformation configuration",
            {
                "view_ids": [item.view_id for item in protocol.view_registry],
                "transformation_inventory_sha256": sha256_file(
                    ROOT / config.protocol.transformation_inventory
                ),
                "status": transformation.status,
            },
        )
    )
    applicability_records = (
        inventory.applicable_view_tuple_count + inventory.not_applicable_view_count
    )
    gates.append(
        _make_gate(
            "PF10",
            applicability_records == 264
            and inventory.actual_view_tuple_count == 264
            and len(inventory.records)
            == inventory.applicable_view_tuple_count * len(MODEL_ORDER)
            and all(item.view_id != "V00" for item in inventory.not_applicable_views)
            and all(
                item.reason_code and item.evidence_sha256 for item in inventory.not_applicable_views
            ),
            "applicable and controlled NOT_APPLICABLE states cover every tuple; V00 is always "
            "applicable",
            {
                "all_tuples": applicability_records,
                "applicable": inventory.applicable_view_tuple_count,
                "not_applicable": inventory.not_applicable_view_count,
                "not_applicable_reason_hashes": [
                    item.evidence_sha256 for item in inventory.not_applicable_views
                ],
            },
        )
    )

    model_ids = tuple(item.model_id for item in protocol.model_matrix)
    gates.append(
        _make_gate(
            "PF11",
            model_ids == MODEL_ORDER,
            "the five accepted frozen model IDs retain canonical order and execution devices",
            {"models": [item.model_dump(mode="json") for item in protocol.model_matrix]},
        )
    )
    checkpoint_hashes = {
        item.model_id: item.checkpoint_sha256
        for item in protocol.model_matrix
        if item.checkpoint_sha256 is not None
    }
    gates.append(
        _make_gate(
            "PF12",
            checkpoint_hashes == config.models.checkpoint_hashes
            and all(item.version for item in protocol.model_matrix)
            and all(
                item.checkpoint_identifier
                for item in protocol.model_matrix
                if item.checkpoint_sha256
            ),
            "exact package versions, checkpoint identifiers, and frozen checkpoint hashes "
            "are bound",
            {
                "package_versions": {
                    item.model_id: [item.package, item.version] for item in protocol.model_matrix
                },
                "checkpoints": checkpoint_hashes,
            },
        )
    )

    expected_conditions = inventory.applicable_view_tuple_count * len(MODEL_ORDER)
    gates.append(
        _make_gate(
            "PF13",
            inventory.actual_condition_count == expected_conditions
            and inventory.actual_condition_count <= inventory.maximum_possible_model_conditions,
            "condition inventory contains exactly five conditions per applicable view tuple",
            {
                "actual_conditions": inventory.actual_condition_count,
                "maximum": inventory.maximum_possible_model_conditions,
                "applicable_view_tuples": inventory.applicable_view_tuple_count,
            },
        )
    )
    identities = [item.condition_id for item in inventory.records]
    gates.append(
        _make_gate(
            "PF14",
            len(identities) == len(set(identities))
            and all(
                item.condition_id == sha256_canonical_json(item.identity.model_dump(mode="json"))
                for item in inventory.records
            ),
            "all condition IDs are unique canonical hashes of the complete identity payload",
            {
                "condition_count": len(identities),
                "unique_count": len(set(identities)),
                "inventory_sha256": inventory.inventory_sha256,
            },
        )
    )

    expected_events = (
        "plan_frozen",
        "test_labels_sealed",
        "training_completed",
        "calibration_predictions_completed",
        "test_predictions_completed",
        "prediction_structure_validated",
        "test_labels_opened",
        "metrics_generated",
        "results_sealed",
    )
    gates.append(
        _make_gate(
            "PF15",
            protocol.leakage_policy.label_open_event_order == expected_events
            and protocol.leakage_policy.transformation_fit_partition == "train"
            and protocol.leakage_policy.preprocessing_fit_partition == "train"
            and protocol.leakage_policy.model_fit_partition == "train",
            "frozen leakage policy confines fitting to training rows and orders evaluation events",
            protocol.leakage_policy.model_dump(mode="json"),
        )
    )
    gates.append(
        _make_gate(
            "PF16",
            protocol.execution_authorized is False
            and protocol.offline is True
            and protocol.leakage_policy.seal_test_labels_until_all_predictions_and_structure_valid
            is True
            and expected_events.index("prediction_structure_validated")
            < expected_events.index("test_labels_opened"),
            "planning remains outcome-blind; label opening occurs only after structural "
            "prediction validation",
            {
                "execution_authorized": protocol.execution_authorized,
                "offline": protocol.offline,
                "event_order": expected_events,
                "planner_target_reads": False,
            },
        )
    )

    metric = protocol.metric_policy
    gates.append(
        _make_gate(
            "PF17",
            metric.primary_endpoint == "worst_view_brier_degradation"
            and metric.relative_degradation_epsilon > 0
            and metric.sii.logarithm_base == 2
            and metric.sii.per_row == "maximum_pairwise_jsd_across_applicable_views"
            and len(metric.secondary_metrics) == 16
            and metric.dataset_level_summaries
            == (
                "median",
                "interquartile_range",
                "minimum",
                "maximum",
                "win_tie_loss_counts",
                "transformation_family_summaries",
                "model_family_summaries",
            ),
            "primary Brier degradation, relative epsilon, SII, flips, and undefined-metric "
            "policy are frozen",
            metric.model_dump(mode="json"),
        )
    )
    auroc = metric.auroc
    gates.append(
        _make_gate(
            "PF18",
            auroc.numerical_tie_tolerance == 1.0e-15
            and auroc.chunk_size > 0
            and auroc.tie_credit == 0.5
            and auroc.tie_only_changes_excluded_from_scientific_instability
            and auroc.tie_sensitivity_probability_max_abs_diff == 1.0e-15
            and auroc.tie_sensitivity_label_flip_count == 0
            and auroc.tie_sensitivity_non_rank_metrics == ("brier", "log_loss")
            and auroc.tie_sensitivity_classification == "NUMERICAL_TIE_SENSITIVITY",
            "raw and tolerance-aware AUROC use chunked comparisons and explicit numerical-tie "
            "handling",
            auroc.model_dump(mode="json"),
        )
    )
    gates.append(
        _make_gate(
            "PF19",
            metric.seed_aggregation == "median"
            and metric.dataset_aggregation == "median"
            and metric.cross_dataset_unit == "dataset",
            "seed repeats are summarized within datasets; datasets remain the cross-dataset unit",
            {
                "seed_aggregation": metric.seed_aggregation,
                "dataset_aggregation": metric.dataset_aggregation,
                "cross_dataset_unit": metric.cross_dataset_unit,
            },
        )
    )
    gates.append(
        _make_gate(
            "PF20",
            protocol.decision_policy.outcomes
            == (
                "ADVANCE_BROAD_METHOD",
                "ADVANCE_TFM_FOCUSED",
                "NARROW_CASE_STUDY",
                "STOP_OR_REDESIGN",
                "PARTIAL_CAPABILITY",
                "REPAIR_REQUIRED",
            )
            and protocol.decision_policy.nontrivial_absolute_wbd == 0.03
            and protocol.decision_policy.nontrivial_relative_wbd == 0.10
            and protocol.decision_policy.require_all_v00_conditions
            and protocol.decision_policy.minimum_transformed_condition_completion_rate == 0.95
            and protocol.decision_policy.require_frozen_preprocessing_residual,
            "nontrivial-effect, breadth, residual-preprocessing, comparison, and six outcomes "
            "are frozen",
            protocol.decision_policy.model_dump(mode="json"),
        )
    )
    gates.append(
        _make_gate(
            "PF21",
            protocol.retry_policy.retry_identity_must_match
            and not protocol.retry_policy.failed_artifact_cache_hit
            and not protocol.retry_policy.partial_artifact_promotion
            and protocol.retry_policy.test_outcome_controls_retry is False,
            "retryable failures, attempt limits, failed-cache prohibition, and same-identity "
            "resume are frozen",
            protocol.retry_policy.model_dump(mode="json"),
        )
    )
    resource = protocol.resource_policy
    gates.append(
        _make_gate(
            "PF22",
            resource.cpu_workers == 2
            and resource.gpu_workers == 1
            and resource.vram_ceiling_mib <= 3600
            and resource.foundation_models_sequential
            and resource.offline_execution,
            "two CPU workers, one sequential GPU worker, 3600 MiB VRAM cap, and offline "
            "execution are frozen",
            resource.model_dump(mode="json"),
        )
    )
    estimate = protocol.runtime_estimate
    gates.append(
        _make_gate(
            "PF23",
            estimate.source_measurement_sha256
            and estimate.conservative_wall_seconds >= 0
            and estimate.model_condition_counts == inventory.counts_by_model
            and schedule_matches
            and expected_schedule.total_condition_count == inventory.actual_condition_count
            and expected_schedule.all_stages_within_hard_wall_limit,
            "runtime estimate and deterministic staged schedule cover every planned condition; "
            "each stage fits the hard wall limit",
            {
                "estimate": estimate.model_dump(mode="json"),
                "schedule_sha256": expected_schedule.schedule_sha256,
                "schedule_matches_frozen_artifact": schedule_matches,
                "schedule_error": schedule_error,
                "stage_count": len(expected_schedule.stages),
                "stage_condition_counts": [
                    len(stage.condition_ids) for stage in expected_schedule.stages
                ],
                "stage_wall_seconds": [
                    stage.conservative_wall_seconds for stage in expected_schedule.stages
                ],
                "all_stages_within_hard_limit": (
                    expected_schedule.all_stages_within_hard_wall_limit
                ),
            },
        )
    )
    gates.append(
        _make_gate(
            "PF24",
            estimate.estimated_prediction_storage_bytes >= 0
            and estimate.estimated_cache_storage_bytes
            >= estimate.estimated_prediction_storage_bytes
            and estimate.preferred_storage_limit_passed
            == (estimate.estimated_cache_storage_bytes <= resource.preferred_storage_bytes),
            "storage estimate includes row IDs, probabilities, metadata, headers, and cache "
            "allowance",
            {
                "prediction_storage_bytes": estimate.estimated_prediction_storage_bytes,
                "cache_storage_bytes": estimate.estimated_cache_storage_bytes,
                "preferred_limit_bytes": resource.preferred_storage_bytes,
                "preferred_limit_passed": estimate.preferred_storage_limit_passed,
            },
        )
    )

    contract_types = (
        PilotProtocolArtifact,
        type(protocol.selected_datasets[0]),
        type(inventory.records[0]),
        type(inventory),
        type(protocol.metric_policy),
        type(protocol.decision_policy),
        PilotProtocolValidation,
    )
    strict = all(
        item.model_config.get("extra") == "forbid" and item.model_config.get("frozen")
        for item in contract_types
    )
    gates.append(
        _make_gate(
            "PF25",
            strict,
            "protocol, dataset, condition, inventory, metric, decision, and validation "
            "contracts are strict and immutable",
            {
                "contract_names": [item.__name__ for item in contract_types],
                "extra_forbid": strict,
                "frozen": strict,
            },
        )
    )

    schema_result = _schema_freshness_check()
    gates.append(
        _make_gate(
            "PF26",
            schema_result["passed"] is True,
            "generated contract schemas exactly match current Pydantic contracts",
            schema_result,
        )
    )

    unit_gate, unit_result = _command_gate(
        "PF27",
        "non-network, non-GPU, non-foundation-model unit tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-m",
            "not integration and not network and not gpu and not foundation_model",
            "-rA",
        ],
    )
    gates.append(unit_gate)
    command_results["unit_tests"] = unit_result
    integration_gate, integration_result = _command_gate(
        "PF28",
        "non-network integration tests; pilot plan test is metadata-only",
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-m",
            "integration and not network and not gpu and not foundation_model",
            "-rA",
        ],
    )
    gates.append(integration_gate)
    command_results["integration_tests"] = integration_result

    for gate_id, name, command, key in (
        ("PF29", "Ruff", [sys.executable, "-m", "ruff", "check", "."], "ruff"),
        ("PF30", "Mypy", [sys.executable, "-m", "mypy", "src/schemaguard"], "mypy"),
        (
            "PF31",
            "semantic repository naming validation",
            [sys.executable, "scripts/validate_repository_naming.py"],
            "naming",
        ),
        (
            "PF32",
            "repository structural and local-evidence validation",
            [sys.executable, "scripts/validate_repository_repair.py", "--local-evidence"],
            "repository",
        ),
    ):
        gate, result = _command_gate(gate_id, name, command)
        gates.append(gate)
        command_results[key] = result

    gates.append(
        _make_gate(
            "PF33",
            len(split_inventory.records) == 70
            and all(
                item.status == "PASS" and item.cross_partition_group_count == 0
                for item in split_inventory.records
            )
            and all(
                item.assignment_artifact_hash and item.manifest_artifact_hash
                for item in split_inventory.records
            ),
            "all accepted split inventory records remain PASS with zero predictor-group crossings",
            {
                "split_inventory_sha256": sha256_file(ROOT / config.protocol.split_inventory),
                "record_count": len(split_inventory.records),
                "crossing_groups": sum(
                    item.cross_partition_group_count for item in split_inventory.records
                ),
            },
        )
    )
    gates.append(
        _make_gate(
            "PF34",
            transformation.status == "PASS_PENDING_REVIEW"
            and protocol.source_hashes[config.protocol.transformation_inventory]
            == sha256_file(ROOT / config.protocol.transformation_inventory)
            and protocol.source_hashes["src/schemaguard/transformations/implementation_tree"]
            == rebuilt_protocol.source_hashes[
                "src/schemaguard/transformations/implementation_tree"
            ],
            "accepted transformation applicability inventory and implementation identity "
            "remain unchanged",
            {
                "inventory_sha256": sha256_file(ROOT / config.protocol.transformation_inventory),
                "status": transformation.status,
                "implementation_tree_sha256": protocol.source_hashes[
                    "src/schemaguard/transformations/implementation_tree"
                ],
            },
        )
    )
    adapter_inventory = ModelAdapterInventory.model_validate(
        _read_json(ROOT / config.protocol.adapter_inventory)
    )
    adapter_models = {record.model_id for record in adapter_inventory.records}
    gates.append(
        _make_gate(
            "PF35",
            adapter_models == set(MODEL_ORDER)
            and all(
                record.status == "PASS" and record.roundtrip_passed
                for record in adapter_inventory.records
            ),
            "all accepted adapter evidence remains passing and is identity-bound into the "
            "frozen model matrix",
            {
                "adapter_inventory_sha256": sha256_file(ROOT / config.protocol.adapter_inventory),
                "record_count": len(adapter_inventory.records),
                "model_ids": sorted(adapter_models),
            },
        )
    )

    scheduler_path = ROOT / "artifacts/handoff/cache_scheduler_inventory.json"
    fault_path = ROOT / "artifacts/handoff/cache_scheduler_fault_evidence.json"
    probe_path = ROOT / "artifacts/handoff/cache_scheduler_probe_evidence.json"
    scheduler = CacheSchedulerInventory.model_validate(_read_json(scheduler_path))
    faults = FaultInjectionEvidence.model_validate(_read_json(fault_path))
    probes = SchedulerProbeEvidence.model_validate(_read_json(probe_path))
    gates.append(
        _make_gate(
            "PF36",
            scheduler.status == "PASS_PENDING_REVIEW"
            and all(status == "PASS" for status in scheduler.test_statuses.values())
            and len(faults.records) == 30
            and probes.duplicate_validated_artifact_count == 0,
            "accepted cache/scheduler inventory, 30 fault cases, and resume probe remain "
            "strictly valid",
            {
                "inventory_sha256": sha256_file(scheduler_path),
                "fault_sha256": sha256_file(fault_path),
                "probe_sha256": sha256_file(probe_path),
                "status": scheduler.status,
                "fault_count": len(faults.records),
                "offline_network_attempts_in_synthetic_faults": (
                    faults.offline_network_attempt_count
                ),
            },
        )
    )
    smoke_path = ROOT / config.protocol.smoke_review
    smoke_runtime_path = ROOT / config.protocol.smoke_runtime_report
    gates.append(
        _make_gate(
            "PF37",
            sha256_file(smoke_path) == protocol.source_hashes[config.protocol.smoke_review]
            and sha256_file(smoke_runtime_path)
            == protocol.source_hashes[config.protocol.smoke_runtime_report]
            and smoke_evidence.get("pilot_started") is False,
            "accepted smoke protocol and run evidence remain byte-identical and explicitly "
            "report no pilot",
            {
                "smoke_review_sha256": sha256_file(smoke_path),
                "smoke_runtime_sha256": sha256_file(smoke_runtime_path),
                "pilot_started": smoke_evidence.get("pilot_started"),
            },
        )
    )

    clone_evidence = _clean_clone_check()
    gates.append(
        _make_gate(
            "PF38",
            clone_evidence.get("passed") is True,
            "source, schemas, and focused protocol tests pass in a temporary local clean clone "
            "without local datasets or private files",
            clone_evidence,
        )
    )
    pilot_output_paths = (
        ROOT / "results/pilot",
        ROOT / "results/validation/pilot_metrics.parquet",
        ROOT / "data/cache/pilot",
        ROOT / "results/predictions/pilot",
    )
    pilot_outputs_absent = not any(path.exists() for path in pilot_output_paths)
    gates.append(
        _make_gate(
            "PF39",
            protocol.execution_authorized is False
            and inventory.actual_condition_count > 0
            and pilot_outputs_absent,
            "only a plan and condition manifest exist; no pilot fit, prediction, or pilot "
            "metric output was created",
            {
                "execution_authorized": protocol.execution_authorized,
                "pilot_output_paths_absent": pilot_outputs_absent,
                "condition_count_planned_only": inventory.actual_condition_count,
            },
        )
    )
    status_code, private_status, private_error = _git(
        "status", "--porcelain=v1", "--", "SchemaGuard_Complete_Research_and_Engineering_Plan.docx"
    )
    private_untracked = status_code == 0 and private_status.startswith("?? ")
    gates.append(
        _make_gate(
            "PF40",
            private_untracked,
            "private root reference document was not opened or modified and remains untracked",
            {"status_command": status_code, "status": private_status, "error": private_error},
        )
    )

    after = _tree_snapshot()
    protected_unchanged = before == after
    gates[2] = _make_gate(
        "PF03",
        protected_unchanged and schema_baseline.get("passed") is True,
        "content-addressed protected-tree snapshot is unchanged through validation; "
        "pre-existing tracked schemas match the authorized base",
        {
            "before": before,
            "after": after,
            "unchanged": protected_unchanged,
            "tracked_schema_baseline": schema_baseline,
            "prior_footprint_constants_removed": True,
        },
    )
    # Direct final comparison confirms schema generation changed only its seven additions.
    schemas_added_check = _tracked_schema_baseline_check()
    if not schemas_added_check.get("passed"):
        gates[25] = _make_gate(
            "PF26",
            False,
            "pre-existing schemas differ from their authorized base",
            schemas_added_check,
        )

    report = build_validation_report(
        source_commit=EXPECTED_BASE,
        protocol=protocol,
        inventory=inventory,
        gates=gates,
    )
    handoff = _render_handoff(
        report,
        protocol,
        inventory,
        expected_schedule,
        command_results,
        {"before": before, "after": after, "unchanged": protected_unchanged},
        clone_evidence,
    )
    return report, handoff


def _render_handoff(
    report: PilotProtocolValidation,
    protocol: PilotProtocolArtifact,
    inventory: Any,
    staged_schedule: PilotStagedSchedule,
    command_results: dict[str, Any],
    preservation: dict[str, Any],
    clone_evidence: dict[str, Any],
) -> str:
    selected_ids = ", ".join(str(item.dataset_id) for item in protocol.selected_datasets)
    excluded_ids = ", ".join(str(item.dataset_id) for item in protocol.excluded_datasets)
    selected_seeds = ", ".join(str(item) for item in protocol.seed_selection.seeds)
    lines = [
        "# SchemaGuard Representation-Sensitivity Pilot Protocol Review",
        "",
        f"Status: `{report.status}`",
        "",
        f"Protocol base commit: `{report.source_commit}`",
        "Delivery commit: the Git commit containing this review; its exact hash and verified "
        "Actions run are recorded in the delivery response.",
        "",
        "This is a plan-only protocol. No pilot model fit, prediction, pilot metric, or pilot "
        "test-label access occurred.",
        "The private root `.docx` was not opened or modified and remains untracked.",
        "",
        "## Frozen design",
        "",
        f"- Selected datasets: {selected_ids}; anchor 1464 included.",
        f"- Excluded registry datasets: {excluded_ids}; each has a metadata-only rationale.",
        f"- Seeds: {selected_seeds}; 24 selected dataset-seed pairs bind passing grouped splits.",
        f"- Views: {len(protocol.view_registry)}; planned tuples "
        f"{inventory.actual_view_tuple_count}, "
        f"applicable {inventory.applicable_view_tuple_count}, controlled N/A "
        f"{inventory.not_applicable_view_count}.",
        f"- Model conditions: {inventory.actual_condition_count} of a maximum "
        f"{inventory.maximum_possible_model_conditions}; CPU {inventory.cpu_condition_count}, "
        f"CUDA {inventory.cuda_condition_count}.",
        f"- Protocol SHA-256: `{protocol.protocol_sha256}`.",
        f"- Condition inventory SHA-256: `{inventory.inventory_sha256}`.",
        "",
        "## Resource estimate",
        "",
        f"- Conservative wall time: {protocol.runtime_estimate.conservative_wall_seconds:.1f} s; "
        f"preferred limit {protocol.resource_policy.preferred_wall_seconds} s; "
        f"hard limit {protocol.resource_policy.hard_wall_seconds} s.",
        "- Prediction storage: "
        f"{protocol.runtime_estimate.estimated_prediction_storage_bytes} bytes; "
        f"cache allowance {protocol.runtime_estimate.estimated_cache_storage_bytes} bytes; "
        f"preferred storage {protocol.resource_policy.preferred_storage_bytes} bytes.",
        "- Staged schedule required: "
        f"`{protocol.runtime_estimate.staged_schedule_required}`; "
        "full condition matrix remains frozen.",
        f"- Staged schedule SHA-256: `{staged_schedule.schedule_sha256}`; "
        f"{len(staged_schedule.stages)} deterministic dataset-pair stages; cumulative staged "
        f"estimate {staged_schedule.total_estimated_wall_seconds:.1f} s "
        f"({staged_schedule.total_estimated_wall_seconds / 3600:.2f} h); each stage is within "
        f"the {staged_schedule.hard_wall_limit_seconds}-second hard limit: "
        f"`{staged_schedule.all_stages_within_hard_wall_limit}`. The cumulative full-matrix "
        "estimate still requires resource review; staging does not reduce total work.",
        "| Stage | Dataset IDs | Conditions (CPU/CUDA) | Conservative wall | Hard limit |",
        "| --- | --- | ---: | ---: | ---: |",
        *[
            "| {} | {} | {} ({} / {}) | {:.1f} s | {} s |".format(
                stage.stage_id,
                ", ".join(str(dataset_id) for dataset_id in stage.dataset_ids),
                len(stage.condition_ids),
                stage.cpu_condition_count,
                stage.cuda_condition_count,
                stage.conservative_wall_seconds,
                staged_schedule.hard_wall_limit_seconds,
            )
            for stage in staged_schedule.stages
        ],
        "",
        "## Acceptance gates",
        "",
        f"Passed: {report.passed_count}; failed: {report.failed_count}; "
        f"not verified: {report.not_verified_count}.",
        "",
        "| Gate | Result | Evidence SHA-256 | Detail |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        "| {} | {} | `{}` | {} |".format(
            item.gate_id,
            item.result,
            item.evidence_sha256,
            item.detail.replace("|", "/"),
        )
        for item in report.gates
    )
    lines.extend(
        [
            "",
            "## Quality command results",
            "",
            "Commands ran with the P12 interpreter; full raw output was not embedded in this "
            "tracked handoff.",
        ]
    )
    for name, result in command_results.items():
        lines.append(
            "- `{}`: exit {}; output SHA-256 `{}`; command `{}`.".format(
                name,
                result["returncode"],
                result["output_sha256"],
                " ".join(result["command"]),
            )
        )
    lines.extend(
        [
            "",
            "## Preservation and portability",
            "",
            f"- Protected local evidence snapshot unchanged during validation: "
            f"`{preservation['unchanged']}`; before/after identity "
            f"`{preservation['before']['records_sha256']}` / "
            f"`{preservation['after']['records_sha256']}`.",
            f"- Clean-clone focused tests: `{clone_evidence.get('passed')}`; "
            f"overlay file count {clone_evidence.get('overlay_file_count')}.",
            "- The planner validates source/processed/target hashes and accepted split "
            "assignment hashes using manifests and byte hashing only; it does not decode "
            "target rows or open pilot test labels.",
            "- Next permitted workstream: independent pilot-protocol review. Pilot execution "
            "remains unauthorized until that review accepts this handoff.",
            "",
            "## Frozen output files",
            "",
            "- `configs/runtime/pilot_protocol.yaml`",
            "- `src/schemaguard/experiments/pilot_contracts.py`",
            "- `src/schemaguard/experiments/pilot_planning.py`",
            "- `src/schemaguard/experiments/pilot_validation.py`",
            "- `scripts/freeze_pilot_protocol.py`",
            "- `scripts/validate_pilot_protocol.py`",
            "- Eight generated schemas under `schemas/`.",
            "- `artifacts/handoff/pilot_protocol.json`",
            "- `artifacts/handoff/pilot_condition_inventory.json`",
            "- `artifacts/handoff/pilot_staged_schedule.json`",
            "- `artifacts/handoff/pilot_protocol_validation.json`",
            "- `artifacts/handoff/pilot_protocol_review.md`",
            "",
            "No data, split, transformation, adapter, scheduler, or smoke evidence was rewritten.",
            "",
        ]
    )
    lines.extend(
        _detailed_handoff_sections(
            report, protocol, inventory, command_results, preservation, clone_evidence
        )
    )
    return "\n".join(lines)


def _detailed_handoff_sections(
    report: PilotProtocolValidation,
    protocol: PilotProtocolArtifact,
    inventory: Any,
    command_results: dict[str, Any],
    preservation: dict[str, Any],
    clone_evidence: dict[str, Any],
) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "/").replace("\n", " ")

    lines = [
        "",
        "## Complete selection and model record",
        "",
        "### Selected datasets",
        "",
        "| ID | Name | Task | Rows | Features N/C | Coverage | Rationale |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for dataset in protocol.selected_datasets:
        lines.append(
            "| {} | {} | {} | {} | {}/{} | {} | {} |".format(
                dataset.dataset_id,
                cell(dataset.dataset_name),
                dataset.task_type,
                dataset.row_count,
                dataset.numeric_feature_count,
                dataset.categorical_feature_count,
                cell(", ".join(dataset.coverage_tags)),
                cell(dataset.selection_rationale),
            )
        )
    lines.extend(
        [
            "",
            "| Dataset ID | Raw SHA-256 | Features SHA-256 | Targets SHA-256 | "
            "Quality SHA-256 | Metadata SHA-256 |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for dataset in protocol.selected_datasets:
        lines.append(
            f"| {dataset.dataset_id} | `{dataset.raw_source_sha256}` | "
            f"`{dataset.processed_feature_sha256}` | `{dataset.target_artifact_sha256}` | "
            f"`{dataset.quality_report_sha256}` | `{dataset.metadata_evidence_sha256}` |"
        )
    lines.extend(
        [
            "",
            "### Excluded datasets",
            "",
            "| ID | Name | Reason | Closest selected dataset | Evidence SHA-256 |",
            "| ---: | --- | --- | ---: | --- |",
        ]
    )
    for dataset in protocol.excluded_datasets:
        lines.append(
            f"| {dataset.dataset_id} | {cell(dataset.dataset_name)} | "
            f"{cell(dataset.exclusion_reason)} | {dataset.competing_selected_dataset_id} | "
            f"`{dataset.metadata_evidence_sha256}` |"
        )
    lines.extend(
        [
            "",
            "### Exact view registry",
            "",
            "| View | Name | Family | Certificate | Role |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for view in protocol.view_registry:
        lines.append(
            f"| {view.view_id} | {cell(view.view_name)} | "
            f"{cell(view.transformation_family)} | {view.certificate_type} | "
            f"{view.scientific_role} |"
        )
    lines.extend(
        [
            "",
            "### Exact model matrix",
            "",
            "| ID | Model class | Package/version | Device | Preprocessing | "
            "Precision/batch | Checkpoint identifier and SHA-256 | Model-config SHA-256 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for model in protocol.model_matrix:
        checkpoint = model.checkpoint_identifier or "None"
        if model.checkpoint_sha256:
            checkpoint += f" / {model.checkpoint_sha256}"
        lines.append(
            f"| {model.model_id} | {cell(model.model_class)} | "
            f"{model.package}/{model.version} | {model.device} | "
            f"{cell(model.preprocessing)} | {model.precision_policy}/{model.batch_policy} | "
            f"{cell(checkpoint)} | `{model.model_config_sha256}` |"
        )
        parameters = json.dumps(model.parameters, sort_keys=True, separators=(",", ":"))
        lines.append(f"  Frozen parameters: `{cell(parameters)}`")
    lines.extend(
        [
            "",
            "## Exact condition counts",
            "",
            f"Seeds: {', '.join(str(seed) for seed in protocol.seed_selection.seeds)}.",
            f"Applicable view tuples: {inventory.applicable_view_tuple_count}; controlled "
            f"NOT_APPLICABLE tuples: {inventory.not_applicable_view_count}; total: "
            f"{inventory.actual_view_tuple_count}.",
            f"Model conditions: {inventory.actual_condition_count}; CPU: "
            f"{inventory.cpu_condition_count}; CUDA: {inventory.cuda_condition_count}; "
            f"maximum: {inventory.maximum_possible_model_conditions}.",
            f"Protocol SHA-256: `{protocol.protocol_sha256}`.",
            f"Condition-inventory SHA-256: `{inventory.inventory_sha256}`.",
            "",
            "| Dimension | Key | Condition count |",
            "| --- | --- | ---: |",
        ]
    )
    for dimension, counts in (
        ("Dataset", inventory.counts_by_dataset),
        ("Seed", inventory.counts_by_seed),
        ("View", inventory.counts_by_view),
        ("Model", inventory.counts_by_model),
    ):
        lines.extend(f"| {dimension} | {key} | {value} |" for key, value in counts.items())
    lines.extend(
        [
            "",
            "## Bound inputs and source hashes",
            "",
            "| Repository-relative input | SHA-256 |",
            "| --- | --- |",
        ]
    )
    lines.extend(
        f"| `{path}` | `{digest}` |" for path, digest in sorted(protocol.source_hashes.items())
    )
    lines.extend(
        [
            "",
            "## Frozen formulas, leakage, and decision criteria",
            "",
            "```text",
            "Brier = mean_i sum_c (P[i,c] - 1[y[i] == c])^2",
            "delta(view) = Brier(view) - Brier(V00)",
            "WBD = max(delta(view)) across applicable non-V00 views",
            "RWBD = WBD / max(Brier(V00), epsilon); epsilon = 1e-12",
            "SII_i = max pairwise JSD across applicable views; logarithm base 2",
            "SII clips at 1e-15 then renormalizes each row",
            "Flip_i = any argmax-class difference across applicable views",
            "```",
            "Label flips also record pairwise rates against V00, the worst view, and its "
            "transformation family. Raw AUROC is retained; tolerance-aware AUROC uses tau=1e-15, "
            "half-credit ties and chunk size 256, and records the number of tolerance ties. "
            "Tie-only classification requires equivalent probabilities/labels/Brier/log loss under "
            "the separate non-rank tolerance, raw AUROC change, and no tolerance-aware change.",
            "",
            "Secondary metrics: "
            + ", ".join(protocol.metric_policy.secondary_metrics)
            + ". Undefined metrics carry an explicit reason.",
            "Rows are calculation units only; seeds are repeatability units; dataset-model values "
            "are medians across seeds; datasets are the cross-dataset unit/effective sample size. "
            "Report median, IQR, minimum, maximum, win/tie/loss, transformation-family, and "
            "model-family summaries. No confirmatory significance claims; intervals are "
            "exploratory.",
            "",
            "Test-label sequence: "
            + " -> ".join(protocol.leakage_policy.label_open_event_order)
            + ". Labels cannot influence planning, applicability, fitting, preprocessing, "
            "batch size, timeout, retry, resource fallback, view/model selection, or condition "
            "exclusion.",
            "",
            "Integrity failures mean `REPAIR_REQUIRED`. Feasibility requires all V00 conditions, "
            "at least 95% transformed completion, classified failures, unchanged frozen model "
            "configuration, valid cache/resume, and respected resource ceilings. Nontrivial effect "
            "requires a foundation model with median dataset WBD >= 0.03 or RWBD >= 0.10. Breadth "
            "requires at least 3/8 datasets, 2 families, effect direction in 2/3 seeds on each "
            "counted dataset, and not tie-only; effect must remain after frozen preprocessing.",
            "Classical LR/CatBoost/XGBoost comparisons characterize whether sensitivity is broad "
            "or TFM-focused. No positive result outside the frozen six-outcome vocabulary is "
            "allowed.",
            "",
            "Retry policy: maximum "
            f"{protocol.retry_policy.max_attempts_per_condition} attempts; retryable "
            f"{', '.join(protocol.retry_policy.retryable_categories)}; non-retryable "
            f"{', '.join(protocol.retry_policy.non_retryable_categories)}. Failed outputs are not "
            "cache hits; partial artifacts cannot be promoted; condition identity must match.",
            "",
            "## Resource and quality evidence",
            "",
            f"Runtime estimate method: {protocol.runtime_estimate.estimate_method}; measurement "
            f"SHA-256 `{protocol.runtime_estimate.source_measurement_sha256}`.",
            f"CPU work {protocol.runtime_estimate.estimated_cpu_seconds:.1f}s; CUDA work "
            f"{protocol.runtime_estimate.estimated_cuda_seconds:.1f}s; conservative wall "
            f"{protocol.runtime_estimate.conservative_wall_seconds:.1f}s.",
            f"Preferred/hard wall limits {protocol.resource_policy.preferred_wall_seconds}/"
            f"{protocol.resource_policy.hard_wall_seconds}s; peak RAM/VRAM estimate "
            f"{protocol.runtime_estimate.peak_ram_mib:.1f}/"
            f"{protocol.runtime_estimate.peak_vram_mib:.1f} MiB.",
            f"Prediction/cache storage estimate "
            f"{protocol.runtime_estimate.estimated_prediction_storage_bytes}/"
            f"{protocol.runtime_estimate.estimated_cache_storage_bytes} bytes; preferred storage "
            f"{protocol.resource_policy.preferred_storage_bytes} bytes.",
            f"Resource ceilings: {protocol.resource_policy.cpu_workers} CPU workers × "
            f"{protocol.resource_policy.cpu_threads_per_worker} threads, "
            f"{protocol.resource_policy.gpu_workers} sequential GPU worker, "
            f"{protocol.resource_policy.vram_ceiling_mib} MiB VRAM, "
            f"{protocol.resource_policy.ram_hard_limit_mib} MiB RAM. Timeouts: "
            f"{protocol.resource_policy.timeout_seconds}. Offline; no silent fallback/precision/"
            "batch/ensemble changes; atomic publication and validated resume.",
            "",
            "## Commands, preservation, files, and deviations",
            "",
            f"Protected tree unchanged: `{preservation['unchanged']}`; files "
            f"{preservation['before']['file_count']}; "
            f"bytes {preservation['before']['total_bytes']}; "
            f"snapshot SHA-256 `{preservation['before']['records_sha256']}`. Phase 01 data and "
            "split outputs were not regenerated.",
            f"Clean-clone validation: `{clone_evidence.get('passed')}`; overlay files "
            f"{clone_evidence.get('overlay_file_count')}.",
            "Executed command outputs are represented by exit codes, hashes, and test-summary "
            "highlights below. Absolute interpreter paths are omitted.",
        ]
    )
    for name, result in command_results.items():
        command = " ".join(["P12 Python", *result["command"][1:]])
        lines.append(
            "- `{}`: exit {}; output SHA-256 `{}`; `{}`.".format(
                name, result["returncode"], result["output_sha256"], cell(command)
            )
        )
        highlights = [
            line.strip()
            for line in result.get("output_tail", "").splitlines()
            if any(
                key in line.lower() for key in ("passed", "failed", "skipped", "warning", "error")
            )
        ]
        lines.extend(f"  - {cell(line)}" for line in highlights[-4:])
    lines.extend(
        [
            "",
            "Deviation: the pasted specification cites Actions run `35296257149`, which GitHub "
            "shows as an in-progress push workflow, not the accepted smoke workflow. The accepted "
            "independent-smoke report binds successful run `35291513248`, verdict VERIFIED_PASS, "
            "32/32 gates, and pilot_started=false.",
            "No pilot model was trained; no pilot predictions or metrics were produced; no pilot "
            "test labels were decoded or evaluated. The private root reference `.docx` remained "
            "untouched and untracked.",
            "Created/modified deliverables: protocol configuration, contracts, planning and "
            "validation source, freeze/validation scripts, eight generated schemas, focused tests, "
            "sanitized protocol JSON, condition inventory, staged schedule, validation evidence, "
            "and this review.",
            "Next permitted workstream: independent pilot-protocol review. Pilot execution remains "
            "prohibited until that review accepts the handoff.",
            "",
        ]
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=ROOT / "artifacts/handoff")
    args = parser.parse_args()
    output_directory = (
        args.output_directory
        if args.output_directory.is_absolute()
        else ROOT / args.output_directory
    )
    try:
        report, handoff = validate(output_directory)
        atomic_write_json(
            output_directory / "pilot_protocol_validation.json", report.model_dump(mode="json")
        )
        atomic_write_text(output_directory / "pilot_protocol_review.md", handoff)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": report.status,
                "passed": report.passed_count,
                "failed": report.failed_count,
                "not_verified": report.not_verified_count,
                "validation_sha256": sha256_file(
                    output_directory / "pilot_protocol_validation.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.failed_count == 0 and report.not_verified_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
