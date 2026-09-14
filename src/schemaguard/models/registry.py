"""The single source of truth for the Phase 02A frozen model matrix."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

EXPECTED_MODELS = (
    ("LR-1.9", "scikit-learn", "1.9.1", "sklearn.linear_model.LogisticRegression", None),
    ("CAT-1.2", "catboost", "1.2.10", "catboost.CatBoostClassifier", None),
    ("XGB-3.4", "xgboost", "3.4.1", "xgboost.XGBClassifier", None),
    (
        "TPFN3-8.5",
        "tabpfn",
        "8.5.0",
        "tabpfn.TabPFNClassifier",
        "tabpfn-v3-classifier-v3_default.ckpt",
    ),
    (
        "TICL2-2.2",
        "tabicl",
        "2.2.0",
        "tabicl.TabICLClassifier",
        "tabicl-classifier-v2-20260212.ckpt",
    ),
)

KNOWN_PARAMETERS = {
    "LR-1.9": {"penalty", "C", "solver", "max_iter", "tol", "class_weight", "random_state"},
    "CAT-1.2": {
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "bootstrap_type",
        "loss_function",
        "task_type",
        "thread_count",
        "verbose",
        "allow_writing_files",
        "random_seed",
    },
    "XGB-3.4": {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "tree_method",
        "n_jobs",
        "verbosity",
        "random_state",
    },
    "TPFN3-8.5": {
        "n_estimators",
        "auto_scale_n_estimators",
        "fit_mode",
        "memory_saving_mode",
        "n_preprocessing_jobs",
        "inference_precision",
        "ignore_pretraining_limits",
        "show_progress_bar",
        "random_state",
        "device",
        "model_path",
    },
    "TICL2-2.2": {
        "checkpoint_version",
        "n_estimators",
        "batch_size",
        "kv_cache",
        "offload_mode",
        "use_amp",
        "feat_shuffle_method",
        "class_shuffle_method",
        "average_logits",
        "n_jobs",
        "random_state",
        "device",
        "model_path",
        "allow_auto_download",
    },
}


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    class_path: str
    package: str
    expected_version: str
    checkpoint: str | None
    parameters: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    runtime: dict[str, Any]
    execution: dict[str, Any]
    timeouts: dict[str, int]
    fixtures: dict[str, int]
    tolerances: dict[str, float]
    models: list[ModelSpec]

    @model_validator(mode="after")
    def validate_closed_sections(self) -> RuntimeConfig:
        expected = {
            "runtime": {
                "python_major_minor",
                "conda_environment",
                "cpu_workers",
                "gpu_workers",
                "gpu_headroom_mib",
            },
            "execution": {
                "lazy_imports",
                "isolate_foundation_models",
                "release_between_probes",
                "network_default",
                "checkpoint_cache_lock",
            },
            "timeouts": {
                "import_seconds",
                "constructor_seconds",
                "checkpoint_seconds",
                "cpu_probe_seconds",
                "gpu_probe_seconds",
            },
            "fixtures": {
                "binary_train_rows",
                "binary_test_rows",
                "multiclass_train_rows",
                "multiclass_test_rows",
                "random_seed",
            },
            "tolerances": {
                "probability_sum_atol",
                "cpu_repeat_max_abs_diff",
                "gpu_repeat_max_abs_diff",
            },
        }
        for section, keys in expected.items():
            observed = set(getattr(self, section))
            if observed != keys:
                raise ValueError(
                    f"{section} keys must be exactly {sorted(keys)}; observed {sorted(observed)}"
                )
        return self


class RegistryError(ValueError):
    """Raised when the frozen registry cannot be proven compatible."""


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load YAML and reject unknown top-level and model keys through Pydantic."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    try:
        config = RuntimeConfig.model_validate(raw)
    except ValidationError as exc:
        raise RegistryError(str(exc)) from exc
    if config.schema_version != 1:
        raise RegistryError(f"Unsupported runtime configuration schema: {config.schema_version}")
    _validate_runtime_limits(config)
    return config


def _validate_runtime_limits(config: RuntimeConfig) -> None:
    for name, value in config.timeouts.items():
        if value <= 0:
            raise RegistryError(f"Timeout must be positive: {name}={value}")
    workers = ("cpu_workers", "gpu_workers", "gpu_headroom_mib")
    for name in workers:
        limit_value = config.runtime.get(name)
        if not isinstance(limit_value, int) or limit_value <= 0:
            raise RegistryError(f"Invalid resource limit: {name}={limit_value!r}")
    if config.runtime.get("gpu_workers") != 1:
        raise RegistryError("Phase 02A requires exactly one GPU worker")


def validate_registry(config: RuntimeConfig) -> tuple[ModelSpec, ...]:
    """Verify exact order, identity, package and checkpoint values."""

    actual = tuple(
        (model.id, model.package, model.expected_version, model.class_path, model.checkpoint)
        for model in config.models
    )
    expected_ids = [row[0] for row in EXPECTED_MODELS]
    ids = [model.id for model in config.models]
    if len(ids) != len(set(ids)):
        raise RegistryError("Duplicate model ID")
    if actual != EXPECTED_MODELS:
        raise RegistryError(
            f"Frozen registry mismatch. Expected {EXPECTED_MODELS!r}, got {actual!r}"
        )
    if ids != expected_ids:
        raise RegistryError("Model order is not deterministic")
    return tuple(config.models)


def load_model_registry(path: str | Path) -> tuple[ModelSpec, ...]:
    return validate_registry(load_runtime_config(path))


def resolve_catboost_loss(target_classes: int) -> str:
    if target_classes == 2:
        return "Logloss"
    if target_classes >= 3:
        return "MultiClass"
    raise RegistryError("CatBoost requires at least two target classes")


def import_class(class_path: str) -> type[Any]:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def inspect_constructor(model: ModelSpec) -> inspect.Signature:
    """Import lazily and return the installed constructor signature."""

    return inspect.signature(import_class(model.class_path))


def build_parameters(
    model: ModelSpec, *, seed: int, device: str = "cpu", target_classes: int = 2
) -> dict[str, Any]:
    """Return frozen parameters plus explicit runtime seed/device values."""

    params = dict(model.parameters)
    if model.id == "CAT-1.2" and params.get("loss_function") == "auto":
        params["loss_function"] = resolve_catboost_loss(target_classes)
    if model.package == "scikit-learn":
        params["random_state"] = seed
    elif model.package == "catboost":
        params["random_seed"] = seed
    else:
        params["random_state"] = seed
    if model.id in {"TPFN3-8.5", "TICL2-2.2"}:
        params["device"] = device
    return params


def validate_constructor_parameters(model: ModelSpec, parameters: dict[str, Any]) -> None:
    signature = inspect_constructor(model)
    unknown = sorted(set(parameters) - KNOWN_PARAMETERS[model.id])
    if unknown:
        raise RegistryError(f"{model.id} constructor does not accept frozen parameters: {unknown}")
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        unknown = sorted(set(parameters) - set(signature.parameters))
        if unknown:
            raise RegistryError(f"{model.id} constructor does not expose parameters: {unknown}")
