from __future__ import annotations

from pathlib import Path

from schemaguard.experiments.pilot_planning import build_pilot_plan, load_pilot_config

ROOT = Path(__file__).resolve().parents[2]


def test_metadata_only_plan_freezes_the_full_matrix_deterministically() -> None:
    config = load_pilot_config(ROOT / "configs/runtime/pilot_protocol.yaml")
    first_protocol, first_inventory = build_pilot_plan(ROOT, config)
    second_protocol, second_inventory = build_pilot_plan(ROOT, config)

    assert first_protocol.protocol_sha256 == second_protocol.protocol_sha256
    assert first_inventory.inventory_sha256 == second_inventory.inventory_sha256
    assert len(first_protocol.selected_datasets) == 8
    assert 1464 in {item.dataset_id for item in first_protocol.selected_datasets}
    assert first_protocol.seed_selection.seeds == (1729, 2718, 31415)
    assert first_inventory.actual_view_tuple_count == 264
    assert first_inventory.actual_condition_count == (
        first_inventory.applicable_view_tuple_count * 5
    )
    assert first_inventory.cpu_condition_count * 5 == first_inventory.actual_condition_count * 3
    assert first_inventory.cuda_condition_count * 5 == first_inventory.actual_condition_count * 2
    assert all(
        item.identity.offline_policy is True
        and item.identity.checkpoint_identifier
        == next(
            model.checkpoint_identifier
            for model in first_protocol.model_matrix
            if model.model_id == item.model_id
        )
        for item in first_inventory.records
    )
    assert first_protocol.execution_authorized is False
