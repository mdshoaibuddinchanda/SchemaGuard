from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from schemaguard.experiments.pilot_contracts import (
    MODEL_ORDER,
    PilotConditionInventory,
    PilotConfig,
    PilotProtocolArtifact,
    PilotStagedSchedule,
)
from schemaguard.experiments.pilot_planning import (
    DATASET_IDS,
    EXPECTED_SPLIT_SEEDS,
    PlanningError,
    build_staged_schedule,
    load_pilot_config,
    select_candidate_ids,
)
from schemaguard.experiments.pilot_validation import (
    evaluate_pilot_outcome,
    pilot_feasibility_passes,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/runtime/pilot_protocol.yaml"


@pytest.fixture(scope="module")
def config() -> PilotConfig:
    return load_pilot_config(CONFIG_PATH)


def _candidate(dataset_id: int) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "row_count": 1000,
        "feature_count": 4,
        "numeric_feature_count": 4,
        "categorical_feature_count": 0,
        "class_count": 2,
        "class_distribution": {"0": 500, "1": 500},
        "missing_cells": 0,
        "duplicate_group_count": 0,
        "coverage_tags": ("numeric_only",),
        "source_artifacts_valid": True,
        "validated_split_seeds": EXPECTED_SPLIT_SEEDS,
        "cross_partition_group_count": 0,
        "v00_compatible_model_ids": MODEL_ORDER,
    }


def _effect(**overrides: Any) -> dict[str, Any]:
    return {
        "median_wbd": 0.03,
        "median_rwbd": 0.10,
        "datasets_meeting_threshold": 3,
        "transformation_family_count": 2,
        "minimum_seed_directions_per_counted_dataset": 2,
        "preprocessing_residual_pass": True,
        "tie_only": False,
        **overrides,
    }


def test_strict_config_accepts_frozen_policy_and_rejects_unknown_and_missing_keys(
    config: PilotConfig,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert PilotConfig.model_validate(raw) == config
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PilotConfig.model_validate({**raw, "unfrozen_option": True})
    missing = {**raw, "resources": dict(raw["resources"])}
    del missing["resources"]["cpu_workers"]
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(missing)


def test_eight_dataset_selection_is_deterministic_and_keeps_anchor(
    config: PilotConfig,
) -> None:
    ids = (3, 23, 29, 31, 36, 37, 38, 44, 1464)
    candidates = [_candidate(dataset_id) for dataset_id in ids]
    policy = config.selection.model_copy(update={"required_coverage": ("numeric_only",)})
    first = select_candidate_ids(candidates, policy)
    second = select_candidate_ids(list(reversed(candidates)), policy)
    assert first == second == (3, 23, 29, 31, 36, 37, 38, 1464)
    assert len(first) == 8 and 1464 in first


def test_selection_rejects_unregistered_ids_and_outcome_fields(config: PilotConfig) -> None:
    policy = config.selection.model_copy(update={"required_coverage": ("numeric_only",)})
    records = [_candidate(dataset_id) for dataset_id in DATASET_IDS[:8]]
    records[-1]["dataset_id"] = 1
    with pytest.raises(PlanningError, match="unregistered"):
        select_candidate_ids(records, policy)

    records = [_candidate(dataset_id) for dataset_id in DATASET_IDS[:8]]
    records[0]["test_accuracy"] = 0.99
    with pytest.raises(PlanningError, match="metadata and eligibility"):
        select_candidate_ids(records, policy)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("validated_split_seeds", (1729,), "five grouped split"),
        ("cross_partition_group_count", 1, "five grouped split"),
        ("v00_compatible_model_ids", MODEL_ORDER[:-1], "V00 model evidence"),
        ("source_artifacts_valid", False, "source, five grouped split"),
    ),
)
def test_selection_rejects_incomplete_source_split_or_v00_evidence(
    config: PilotConfig, field: str, value: Any, message: str
) -> None:
    policy = config.selection.model_copy(update={"required_coverage": ("numeric_only",)})
    records = [_candidate(dataset_id) for dataset_id in DATASET_IDS[:8]]
    records[0][field] = value
    with pytest.raises(PlanningError, match=message):
        select_candidate_ids(records, policy)


def test_first_three_frozen_seeds_and_all_model_ids_are_canonical(config: PilotConfig) -> None:
    assert config.seeds.selection_rule == "first_three_in_canonical_configured_order"
    assert config.seeds.required_anchor_seed == 1729
    assert config.models.required_ids == MODEL_ORDER
    assert EXPECTED_SPLIT_SEEDS[:3] == (1729, 2718, 31415)
    assert config.metrics.seed_aggregation == "median"
    assert config.metrics.dataset_aggregation == "median"
    assert config.metrics.cross_dataset_unit == "dataset"
    assert config.metrics.effective_sample_size == "number_of_datasets"
    assert config.metrics.confirmatory_significance_claims is False
    assert config.decisions.minimum_transformed_condition_completion_rate == 0.95


def test_staged_schedule_preserves_every_condition_within_each_hard_limit(
    config: PilotConfig,
) -> None:
    handoff = ROOT / "artifacts/handoff"
    protocol = PilotProtocolArtifact.model_validate_json(
        (handoff / "pilot_protocol.json").read_text(encoding="utf-8")
    )
    inventory = PilotConditionInventory.model_validate_json(
        (handoff / "pilot_condition_inventory.json").read_text(encoding="utf-8")
    )
    schedule = build_staged_schedule(protocol, inventory)
    stored_schedule = PilotStagedSchedule.model_validate_json(
        (handoff / "pilot_staged_schedule.json").read_text(encoding="utf-8")
    )
    reordered_inventory = inventory.model_copy(
        update={"records": tuple(reversed(inventory.records))}
    )
    reordered_schedule = build_staged_schedule(protocol, reordered_inventory)

    assert schedule.schedule_sha256 == reordered_schedule.schedule_sha256
    assert schedule == stored_schedule
    assert tuple(stage.dataset_ids for stage in schedule.stages) == (
        (3, 23),
        (29, 36),
        (37, 38),
        (46, 1464),
    )
    assert tuple(len(stage.condition_ids) for stage in schedule.stages) == (
        255,
        255,
        255,
        195,
    )
    assert schedule.total_condition_count == inventory.actual_condition_count == 960
    assert schedule.not_applicable_view_count == inventory.not_applicable_view_count == 72
    assert schedule.all_stages_within_hard_wall_limit is True
    assert all(
        stage.conservative_wall_seconds <= config.resources.hard_wall_seconds
        for stage in schedule.stages
    )
    scheduled_ids = [
        condition_id for stage in schedule.stages for condition_id in stage.condition_ids
    ]
    assert sorted(scheduled_ids) == sorted(item.condition_id for item in inventory.records)


def test_all_six_frozen_decision_outcomes_are_reachable(config: PilotConfig) -> None:
    policy = config.decisions
    assert (
        evaluate_pilot_outcome(
            integrity_pass=False,
            feasible=False,
            partial_capability=False,
            model_evidence={},
            policy=policy,
        )
        == "REPAIR_REQUIRED"
    )
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=False,
            partial_capability=True,
            model_evidence={},
            policy=policy,
        )
        == "PARTIAL_CAPABILITY"
    )
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=True,
            partial_capability=False,
            model_evidence={},
            policy=policy,
        )
        == "STOP_OR_REDESIGN"
    )
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=True,
            partial_capability=False,
            model_evidence={"TPFN3-8.5": _effect(datasets_meeting_threshold=1)},
            policy=policy,
        )
        == "NARROW_CASE_STUDY"
    )
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=True,
            partial_capability=False,
            model_evidence={"TPFN3-8.5": _effect()},
            policy=policy,
        )
        == "ADVANCE_TFM_FOCUSED"
    )
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=True,
            partial_capability=False,
            model_evidence={"TPFN3-8.5": _effect(), "LR-1.9": _effect()},
            policy=policy,
        )
        == "ADVANCE_BROAD_METHOD"
    )


def test_tie_only_and_classical_only_changes_cannot_pass_foundation_effect_gate(
    config: PilotConfig,
) -> None:
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=True,
            partial_capability=False,
            model_evidence={"TPFN3-8.5": _effect(tie_only=True)},
            policy=config.decisions,
        )
        == "STOP_OR_REDESIGN"
    )
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=True,
            partial_capability=False,
            model_evidence={"LR-1.9": _effect()},
            policy=config.decisions,
        )
        == "STOP_OR_REDESIGN"
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"median_wbd": 0.03, "median_rwbd": 0.0}, "ADVANCE_TFM_FOCUSED"),
        ({"median_wbd": 0.0, "median_rwbd": 0.10}, "ADVANCE_TFM_FOCUSED"),
        ({"median_wbd": 0.029999, "median_rwbd": 0.099999}, "STOP_OR_REDESIGN"),
        ({"datasets_meeting_threshold": 2}, "NARROW_CASE_STUDY"),
        ({"transformation_family_count": 1}, "NARROW_CASE_STUDY"),
        ({"minimum_seed_directions_per_counted_dataset": 1}, "STOP_OR_REDESIGN"),
        ({"preprocessing_residual_pass": False}, "STOP_OR_REDESIGN"),
    ),
)
def test_decision_threshold_boundaries_and_breadth(
    config: PilotConfig, overrides: dict[str, Any], expected: str
) -> None:
    outcome = evaluate_pilot_outcome(
        integrity_pass=True,
        feasible=True,
        partial_capability=False,
        model_evidence={"TPFN3-8.5": _effect(**overrides)},
        policy=config.decisions,
    )
    assert outcome == expected


def test_missing_conditions_prevent_positive_decision(config: PilotConfig) -> None:
    feasibility = pilot_feasibility_passes(
        v00_planned=120,
        v00_completed=119,
        transformed_applicable=1000,
        transformed_completed=1000,
        every_failure_classified=True,
        frozen_model_configuration_unchanged=True,
        resume_cache_integrity_pass=True,
        resource_ceilings_respected=True,
        policy=config.decisions,
    )
    assert feasibility is False
    assert (
        evaluate_pilot_outcome(
            integrity_pass=True,
            feasible=feasibility,
            partial_capability=False,
            model_evidence={"TPFN3-8.5": _effect()},
            policy=config.decisions,
        )
        == "REPAIR_REQUIRED"
    )


def test_feasibility_uses_exact_95_percent_boundary_and_all_required_gates(
    config: PilotConfig,
) -> None:
    common = {
        "v00_planned": 120,
        "v00_completed": 120,
        "transformed_applicable": 1000,
        "every_failure_classified": True,
        "frozen_model_configuration_unchanged": True,
        "resume_cache_integrity_pass": True,
        "resource_ceilings_respected": True,
        "policy": config.decisions,
    }
    assert pilot_feasibility_passes(transformed_completed=950, **common)
    assert not pilot_feasibility_passes(transformed_completed=949, **common)
    assert not pilot_feasibility_passes(
        transformed_completed=1000,
        every_failure_classified=False,
        **{key: value for key, value in common.items() if key != "every_failure_classified"},
    )
