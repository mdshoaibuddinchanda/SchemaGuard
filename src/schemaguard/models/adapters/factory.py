"""Deterministic lazy factory validated against the existing frozen registries."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .contracts import AdapterFailure, FailureCategory, ModelAdapterConfig
from .resources import configure_thread_limits


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_config_path() -> Path:
    return _repository_root() / "configs" / "runtime" / "model_adapters.yaml"


@lru_cache(maxsize=4)
def _load_adapter_config_cached(path_text: str) -> tuple[ModelAdapterConfig, Path]:
    import yaml
    from pydantic import ValidationError

    path = Path(path_text)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ModelAdapterConfig.model_validate(raw)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        raise AdapterFailure(
            FailureCategory.FAIL_CONTRACT, f"invalid adapter config: {exc}"
        ) from exc
    return config, path


def load_adapter_config(path: str | Path | None = None) -> ModelAdapterConfig:
    candidate = Path(path) if path is not None else _default_config_path()
    config, _ = _load_adapter_config_cached(str(candidate.resolve()))
    return config


@lru_cache(maxsize=4)
def _load_registries(config_path: str) -> tuple[tuple[Any, ...], dict[str, str], Path]:
    import yaml

    from ..registry import load_model_registry

    adapter_config, path = _load_adapter_config_cached(config_path)
    registry_path = path.parent / adapter_config.registry_config
    experiment_path = (path.parent / adapter_config.experiment_registry).resolve()
    specs = load_model_registry(registry_path)
    raw = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    rows = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "experiment registry has no model list")
    strategies: dict[str, str] = {}
    by_id = {spec.id: spec for spec in specs}
    if len(rows) != len(by_id):
        raise AdapterFailure(
            FailureCategory.FAIL_CONTRACT, "experiment/model registry sizes differ"
        )
    for row in rows:
        model_id = row.get("id")
        spec = by_id.get(model_id)
        if spec is None or model_id in strategies:
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, "duplicate or unknown experiment model"
            )
        frozen_fields = {
            "name": spec.class_path,
            "package": spec.package,
            "version": spec.expected_version,
            "checkpoint": spec.checkpoint,
        }
        for field, expected in frozen_fields.items():
            if row.get(field) != expected:
                raise AdapterFailure(
                    FailureCategory.FAIL_CONTRACT,
                    f"experiment registry {model_id}.{field} differs from "
                    "the frozen model registry",
                )
        strategy = row.get("preprocessing")
        if not isinstance(strategy, str):
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, f"missing preprocessing for {model_id}"
            )
        strategies[model_id] = strategy
    if set(strategies) != set(by_id):
        raise AdapterFailure(FailureCategory.FAIL_CONTRACT, "preprocessing matrix is incomplete")
    return specs, strategies, path.parents[2]


def create_adapter(
    model_id: str,
    *,
    config_path: str | Path | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> Any:
    """Create one adapter without importing its heavyweight model package."""

    configure_thread_limits()
    candidate = Path(config_path) if config_path is not None else _default_config_path()
    normalized = str(candidate.resolve())
    try:
        config, _ = _load_adapter_config_cached(normalized)
        specs, strategies, root = _load_registries(normalized)
        spec = next((item for item in specs if item.id == model_id), None)
        if spec is None:
            raise AdapterFailure(
                FailureCategory.FAIL_CONTRACT, f"unknown frozen model ID: {model_id}"
            )
        strategy = strategies[model_id]
        classes = {
            "LR-1.9": (".logistic_regression_adapter", "LogisticRegressionAdapter"),
            "CAT-1.2": (".catboost_adapter", "CatBoostAdapter"),
            "XGB-3.4": (".xgboost_adapter", "XGBoostAdapter"),
            "TPFN3-8.5": (".tabpfn_adapter", "TabPFNAdapter"),
            "TICL2-2.2": (".tabicl_adapter", "TabICLAdapter"),
        }
        module_name, class_name = classes[model_id]
        from importlib import import_module

        adapter_class = getattr(import_module(module_name, package=__package__), class_name)
        return adapter_class(
            spec,
            config,
            root=root,
            preprocessing_strategy=strategy,
            seed=config.seed if seed is None else seed,
            device=device,
        )
    except AdapterFailure:
        raise
    except Exception as exc:
        raise AdapterFailure(
            FailureCategory.FAIL_CONTRACT, f"adapter factory failed: {type(exc).__name__}: {exc}"
        ) from exc
