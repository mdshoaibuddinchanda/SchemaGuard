"""Deterministic scheduled-condition manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .config import ExperimentConfig, atomic_write_json


@dataclass(frozen=True)
class ConditionRecord:
    dataset_id: int
    seed: int
    model_id: str
    view_id: str
    status: str = "scheduled"
    reason: str | None = None


def build_manifest(
    config: ExperimentConfig, scope: Literal["pilot", "main"]
) -> list[ConditionRecord]:
    """Build the Cartesian product in frozen registry order."""
    return [
        ConditionRecord(dataset.id, seed, model.id, view.id)
        for dataset in config.datasets_for(scope)
        for seed in config.seeds_for(scope)
        for model in config.models
        for view in config.views
    ]


def manifest_payload(config: ExperimentConfig, scope: Literal["pilot", "main"]) -> dict:
    records = build_manifest(config, scope)
    return {
        "scope": scope,
        "benchmark": config.benchmark,
        "config_checksum": config.checksum(),
        "scheduled_count": len(records),
        "records": [asdict(record) for record in records],
    }


def write_manifest(config: ExperimentConfig, scope: Literal["pilot", "main"], path: str) -> bool:
    return atomic_write_json(path, manifest_payload(config, scope))
