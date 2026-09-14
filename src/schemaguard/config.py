"""Strict configuration and manifest primitives for the frozen experiment registry."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import yaml


class ConfigError(ValueError):
    """Raised when a frozen experiment configuration is invalid."""


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {context}: {', '.join(unknown)}")


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key {key!r} in {context}")
    return mapping[key]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    package: str
    version: str
    role: str
    checkpoint: str | None
    preprocessing: str
    parameters: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], index: int) -> ModelSpec:
        context = f"models[{index}]"
        _reject_unknown(
            raw,
            {
                "id",
                "name",
                "package",
                "version",
                "role",
                "checkpoint",
                "preprocessing",
                "parameters",
            },
            context,
        )
        return cls(
            id=str(_require(raw, "id", context)),
            name=str(_require(raw, "name", context)),
            package=str(_require(raw, "package", context)),
            version=str(_require(raw, "version", context)),
            role=str(_require(raw, "role", context)),
            checkpoint=None if raw.get("checkpoint") is None else str(raw["checkpoint"]),
            preprocessing=str(_require(raw, "preprocessing", context)),
            parameters=dict(raw.get("parameters") or {}),
        )


@dataclass(frozen=True)
class DatasetSpec:
    id: int
    name: str
    rows: int
    features: int
    task: str
    schema_property: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], index: int) -> DatasetSpec:
        context = f"datasets[{index}]"
        _reject_unknown(raw, {"id", "name", "rows", "features", "task", "schema_property"}, context)
        return cls(
            id=int(_require(raw, "id", context)),
            name=str(_require(raw, "name", context)),
            rows=int(_require(raw, "rows", context)),
            features=int(_require(raw, "features", context)),
            task=str(_require(raw, "task", context)),
            schema_property=str(_require(raw, "schema_property", context)),
        )


@dataclass(frozen=True)
class ViewSpec:
    id: str
    name: str
    operation: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], index: int) -> ViewSpec:
        context = f"views[{index}]"
        _reject_unknown(raw, {"id", "name", "operation"}, context)
        return cls(
            id=str(_require(raw, "id", context)),
            name=str(_require(raw, "name", context)),
            operation=str(_require(raw, "operation", context)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    project: str
    benchmark: str
    models: tuple[ModelSpec, ...]
    datasets: tuple[DatasetSpec, ...]
    views: tuple[ViewSpec, ...]
    pilot_datasets: tuple[int, ...]
    main_datasets: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    main_seeds: tuple[int, ...]
    split: dict[str, Any]
    smoke_dataset: int
    constraints: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ExperimentConfig:
        if not isinstance(raw, dict):
            raise ConfigError("The configuration root must be a mapping")
        _reject_unknown(
            raw,
            {
                "project",
                "benchmark",
                "models",
                "datasets",
                "views",
                "pilot_datasets",
                "main_datasets",
                "pilot_seeds",
                "main_seeds",
                "split",
                "smoke_dataset",
                "constraints",
            },
            "root",
        )
        models = tuple(
            ModelSpec.from_mapping(item, i)
            for i, item in enumerate(_require(raw, "models", "root"))
        )
        datasets = tuple(
            DatasetSpec.from_mapping(item, i)
            for i, item in enumerate(_require(raw, "datasets", "root"))
        )
        views = tuple(
            ViewSpec.from_mapping(item, i) for i, item in enumerate(_require(raw, "views", "root"))
        )

        _ensure_unique((model.id for model in models), "model")
        _ensure_unique((dataset.id for dataset in datasets), "dataset")
        _ensure_unique((view.id for view in views), "view")

        dataset_ids = {dataset.id for dataset in datasets}
        pilot_datasets = tuple(int(item) for item in _require(raw, "pilot_datasets", "root"))
        main_datasets = tuple(int(item) for item in _require(raw, "main_datasets", "root"))
        for label, ids in (("pilot_datasets", pilot_datasets), ("main_datasets", main_datasets)):
            missing = sorted(set(ids) - dataset_ids)
            if missing:
                raise ConfigError(f"{label} references unknown dataset ID(s): {missing}")
            if len(ids) != len(set(ids)):
                raise ConfigError(f"Duplicate dataset ID in {label}")

        return cls(
            project=str(_require(raw, "project", "root")),
            benchmark=str(_require(raw, "benchmark", "root")),
            models=models,
            datasets=datasets,
            views=views,
            pilot_datasets=pilot_datasets,
            main_datasets=main_datasets,
            pilot_seeds=tuple(int(item) for item in _require(raw, "pilot_seeds", "root")),
            main_seeds=tuple(int(item) for item in _require(raw, "main_seeds", "root")),
            split=dict(_require(raw, "split", "root")),
            smoke_dataset=int(_require(raw, "smoke_dataset", "root")),
            constraints=dict(_require(raw, "constraints", "root")),
        )

    def datasets_for(self, scope: Literal["pilot", "main"]) -> tuple[DatasetSpec, ...]:
        ids = self.pilot_datasets if scope == "pilot" else self.main_datasets
        by_id = {dataset.id: dataset for dataset in self.datasets}
        return tuple(by_id[item] for item in ids)

    def seeds_for(self, scope: Literal["pilot", "main"]) -> tuple[int, ...]:
        return self.pilot_seeds if scope == "pilot" else self.main_seeds

    def scheduled_count(self, scope: Literal["pilot", "main"]) -> int:
        return (
            len(self.datasets_for(scope))
            * len(self.seeds_for(scope))
            * len(self.models)
            * len(self.views)
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "benchmark": self.benchmark,
            "models": [_asdict_model(model) for model in self.models],
            "datasets": [_asdict_dataset(dataset) for dataset in self.datasets],
            "views": [_asdict_view(view) for view in self.views],
            "pilot_datasets": list(self.pilot_datasets),
            "main_datasets": list(self.main_datasets),
            "pilot_seeds": list(self.pilot_seeds),
            "main_seeds": list(self.main_seeds),
            "split": self.split,
            "smoke_dataset": self.smoke_dataset,
            "constraints": self.constraints,
        }

    def checksum(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _ensure_unique(values: Any, label: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ConfigError(f"Duplicate {label} ID")


def _asdict_model(item: ModelSpec) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "package": item.package,
        "version": item.version,
        "role": item.role,
        "checkpoint": item.checkpoint,
        "preprocessing": item.preprocessing,
        "parameters": item.parameters,
    }


def _asdict_dataset(item: DatasetSpec) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "rows": item.rows,
        "features": item.features,
        "task": item.task,
        "schema_property": item.schema_property,
    }


def _asdict_view(item: ViewSpec) -> dict[str, str]:
    return {"id": item.id, "name": item.name, "operation": item.operation}


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig.from_mapping(payload)


def atomic_write_json(path: str | Path, payload: Any) -> bool:
    """Write canonical JSON atomically; return False when identical valid content exists."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        try:
            if destination.read_text(encoding="utf-8") == rendered:
                return False
        except OSError:
            pass
    with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return True
